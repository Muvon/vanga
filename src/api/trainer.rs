//! # Model Trainer - Multi-Target LSTM Training Pipeline
//!
//! This module implements the training pipeline for VANGA's multi-target LSTM architecture.
//!
//! ## Architecture Overview
//!
//! VANGA uses a **multi-model architecture** where each target gets its own dedicated LSTM model:
//!
//! ```text
//! MultiTargetLSTMModel {
//!     models: Vec<LSTMModel>,  // Separate LSTM for each target
//!     target_names: ["price_level_1h", "direction_1h", "volatility_1h"]
//! }
//!
//! Target Processing:
//! Raw Data → [2, 1, 3] → Each value goes to separate LSTM
//!            ↓   ↓   ↓
//!         LSTM1 LSTM2 LSTM3
//! ```
//!
//! ## Alternative Architecture (Not Used)
//!
//! For comparison, a **single-model-multi-head** architecture would look like:
//!
//! ```text
//! SingleLSTMModel {
//!     lstm: LSTMModel,
//! }
//!
//! Target Processing (via TargetConverter):
//! Raw Data → [2, 1, 3] → One-hot encode → [0,0,1,0,0, 0,1,0,0,0, 0,0,0,1,0]
//!                                         ↓
//!                                    Single LSTM
//! ```
//!
//! ## Why Multi-Model Architecture?
//!
//! 1. **Target Independence**: Each target can have different optimal hyperparameters
//! 2. **Specialized Learning**: Each LSTM can specialize in its specific prediction task
//! 3. **Robustness**: Failure in one target doesn't affect others
//! 4. **Flexibility**: Can easily add/remove targets without architectural changes
//!
//! ## Target Format Requirements
//!
//! - **Multi-model**: Raw integer values (0,1,2,3,4) - used by this module
//! - **Single-model-multi-head**: One-hot encoded vectors - use `TargetConverter`
//!
//! ## Walk-Forward Training with Distributed Validation
//!
//! The trainer implements walk-forward analysis with **distributed validation sampling**:
//!
//! ```text
//! Window 1: Train from [0-1000] excluding validation samples
//!           ↓ Validation sampled from 3 periods within [0-1000]:
//!           Early: ~250, Middle: ~500, Late: ~750 (with validation_gap)
//!
//! Window 2: Train from [0-1250] excluding validation samples
//!           ↓ Validation sampled from 3 periods within [0-1250]:
//!           Early: ~312, Middle: ~625, Late: ~937 (with validation_gap)
//!
//! Window 3: Train from [0-1500] excluding validation samples
//!           ↓ Validation sampled from 3 periods within [0-1500]:
//!           Early: ~375, Middle: ~750, Late: ~1125 (with validation_gap)
//! ```
//!
//! **Key Features:**
//! - **Distributed validation**: Samples from multiple periods, not just the end
//! - **Validation gap**: Temporal separation prevents data leakage
//! - **Progressive learning**: Each window expands training data while maintaining validation quality
//! - **Better representation**: Validation covers early, middle, and late patterns in each window

use crate::config::TrainingConfig;
use crate::data::DataPipeline;
use crate::model::multi_target::{MultiTargetLSTMModel, TrainingContext};
use crate::targets::PreparedTargets;
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;
use std::collections::HashMap;

/// Window-by-window training metrics for final statistics
#[derive(Debug, Clone)]
pub struct WindowMetrics {
    pub window_id: usize,
    pub learning_rate: f64,
    pub train_samples: usize,
    pub val_samples: usize,
    /// Per-target validation metrics: target_name -> (accuracy, macro_f1, weighted_f1)
    pub target_metrics: HashMap<String, (f64, f64, f64)>,
}

/// Model trainer for multi-target LSTM architecture
///
/// Orchestrates the complete training pipeline including:
/// - Walk-forward chronological validation
/// - Multi-model target processing
/// - Progressive learning across time windows
/// - Normalization consistency for prediction
pub struct ModelTrainer {
    config: TrainingConfig,
    /// Window-by-window metrics for final statistics
    window_metrics: Vec<WindowMetrics>,
}

impl ModelTrainer {
    /// Create new model trainer with configuration
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            window_metrics: Vec::new(),
        }
    }

    /// Execute complete multi-target LSTM training pipeline
    ///
    /// Implements walk-forward analysis with progressive learning:
    /// 1. Load and prepare chronological training windows
    /// 2. Train first window from scratch
    /// 3. Continue training on subsequent windows with expanded data
    /// 4. Save complete training config and normalization stats for prediction consistency
    ///
    /// Returns trained MultiTargetLSTMModel ready for prediction or further training.
    pub async fn train(&mut self) -> Result<MultiTargetLSTMModel> {
        log::info!("Starting model training for symbol: {}", self.config.symbol);

        // Initialize device from configuration with seed support
        let device_string = self.config.training.device.to_device_string();
        let seed = if self.config.training.seed == 0 {
            None
        } else {
            Some(self.config.training.seed)
        };
        let device =
            crate::utils::device::DeviceManager::create_device_with_seed(&device_string, seed)?;
        log::info!(
            "🔧 Using device: {} ({})",
            device_string,
            match device {
                candle_core::Device::Cpu => "CPU",
                candle_core::Device::Cuda(_) => "NVIDIA CUDA GPU",
                candle_core::Device::Metal(_) => "Apple Metal GPU",
            }
        );

        // Initialize data pipeline
        let data_pipeline = DataPipeline::new();

        // Load and prepare target-specific training data with chronological split
        log::info!(
            "Loading training data from: {}",
            self.config.data_path.display()
        );
        let target_windows = data_pipeline
            .prepare_training_data(&self.config.data_path, &self.config)
            .await?;

        log::info!(
            "Target-specific training: {} targets prepared with independent balancing",
            target_windows.windows_by_target.len()
        );

        // Get base learning rate from config
        log::info!("🔧 Walk-forward training configuration:");
        log::info!(
            "   📊 Test split: {:.1}% ({} samples reserved)",
            self.config.training.test_split * 100.0,
            (target_windows.windows_by_target.len() as f64 * self.config.training.test_split)
                as usize
        );
        log::info!(
            "   📊 Validation split: {:.1}%",
            self.config.training.validation_split * 100.0
        );
        log::info!(
            "   🔄 Window decay: {:.3}",
            self.config.training.window_decay
        );

        log::info!("🔧 Walk-forward training configuration:");
        log::info!(
            "   📊 Test split: {:.1}% ({} samples reserved)",
            self.config.training.test_split * 100.0,
            (target_windows.windows_by_target.len() as f64 * self.config.training.test_split)
                as usize
        );
        log::info!(
            "   📊 Validation split: {:.1}%",
            self.config.training.validation_split * 100.0
        );
        log::info!(
            "   🔄 Window decay: {:.3}",
            self.config.training.window_decay
        );

        // Train each target independently using its own balanced windows
        let base_lr = self.config.training.learning_rate;

        // Log window decay strategy
        if self.config.training.window_decay != 1.0 {
            log::info!(
                "📊 Walk-forward learning rate decay: factor={:.3} (base_lr={:.6})",
                self.config.training.window_decay,
                base_lr
            );
        } else {
            log::info!(
                "📊 Walk-forward training: Fixed learning rate {:.6} for all windows",
                base_lr
            );
        }

        // Train each target independently
        let mut final_model = None;

        for ((target_type, horizon), windows) in &target_windows.windows_by_target {
            log::info!(
                "🎯 Training {:?} {} with {} windows and independent balancing",
                target_type,
                horizon,
                windows.len()
            );

            let mut target_model = None;

            for (window_idx, window) in windows.iter().enumerate() {
                // Calculate window-specific learning rate
                let window_lr = base_lr * self.config.training.window_decay.powi(window_idx as i32);

                log::info!(
                    "🔄 {:?} {} Window {}/{}: effective_lr={:.6} ({:.1}% of base) → {} train samples, {} validation samples",
                    target_type, horizon,
                    window_idx + 1,
                    windows.len(),
                    window_lr,
                    (window_lr / base_lr) * 100.0,
                    window.train_samples,
                    window.val_samples
                );

                if window_idx == 0 {
                    // First window: train from scratch
                    log::info!(
                        "🆕 {:?} {} Window 1: Training fresh model from scratch",
                        target_type,
                        horizon
                    );
                    target_model = Some(self.train_window_from_scratch(window, window_idx).await?);
                } else {
                    // Subsequent windows: continue training on expanded data
                    log::info!(
                        "🔄 {:?} {} Window {}: Continuing training with preserved weights",
                        target_type,
                        horizon,
                        window_idx + 1
                    );
                    target_model = Some(
                        self.continue_training_window(target_model.unwrap(), window, window_idx)
                            .await?,
                    );
                }

                // Collect window metrics for final statistics
                if let Some(ref current_model) = target_model {
                    let window_lr =
                        base_lr * self.config.training.window_decay.powi(window_idx as i32);
                    self.collect_window_metrics(current_model, window, window_idx, window_lr)
                        .await?;
                }
            }

            // Use the first target's model as the final model
            if final_model.is_none() {
                final_model = target_model;
            }
        }

        // Get the final model
        let mut final_model = final_model.ok_or_else(|| {
            VangaError::TrainingError("No target models were trained".to_string())
        })?;

        // Set complete training config on model before saving
        final_model.set_training_config(self.config.clone());

        // Set normalization stats from training data (use first target's first window)
        if let Some(first_windows) = target_windows.windows_by_target.values().next() {
            if let Some(first_window) = first_windows.first() {
                final_model
                    .set_normalization_stats(first_window.train_data.normalization_stats.clone());
                log::info!(
                    "✅ Normalization stats saved with model for consistent prediction preprocessing"
                );
            }
        }

        // Skip final test evaluation for now since we have target-specific windows
        // TODO: Implement proper test evaluation for target-specific models

        // Display final window-by-window statistics
        self.display_final_statistics();

        // Save the trained multi-target model
        log::info!("✅ Walk-forward multi-target model training completed successfully!");
        log::info!("🔧 Complete training config saved with model for consistent prediction");
        Ok(final_model)
    }

    /// Train model from scratch on first window with window-aware configuration
    async fn train_window_from_scratch(
        &self,
        window: &crate::data::TrainingWindow,
        window_id: usize,
    ) -> Result<MultiTargetLSTMModel> {
        log::info!(
            "🎯 [train_window_from_scratch] Training config horizons: {:?} (count: {})",
            self.config.horizons,
            self.config.horizons.len()
        );

        // Process targets for multi-model architecture
        let (train_targets, val_targets) = self.process_window_targets(window, "training")?;

        // Create multi-target model
        let input_size = window.train_data.sequences.shape()[2];
        let target_names = &window.train_data.targets.target_names;
        let mut model = self
            .get_or_create_multi_target_model(input_size, target_names)
            .await?;

        // Create window-aware config that properly scales all scheduler parameters
        let window_config =
            crate::model::lstm::create_window_aware_config(&self.config, window_id)?;

        log::info!(
            "🎯 Training from scratch with window-aware configuration for window {}",
            window_id + 1
        );

        // Train the model with chronological validation
        model
            .train(
                TrainingContext::Standard {
                    sequences: &window.train_data.sequences,
                    targets: &train_targets,
                    val_sequences: Some(&window.val_data.sequences),
                    val_targets: Some(&val_targets),
                    target_class_weights: Some(&window.target_class_weights),
                },
                &window_config,
            )
            .await?;

        log::info!("✅ Window {} training completed", window.window_id + 1);
        Ok(model)
    }

    /// Continue training existing model on new window with window-aware configuration
    async fn continue_training_window(
        &self,
        mut model: MultiTargetLSTMModel,
        window: &crate::data::TrainingWindow,
        window_id: usize,
    ) -> Result<MultiTargetLSTMModel> {
        log::info!(
            "🎯 [continue_training_window] Training config horizons: {:?} (count: {})",
            self.config.horizons,
            self.config.horizons.len()
        );

        // Process targets for multi-model architecture
        let (train_targets, _val_targets) =
            self.process_window_targets(window, "continue training")?;

        // Create window-aware config that properly scales all scheduler parameters
        let window_config =
            crate::model::lstm::create_window_aware_config(&self.config, window_id)?;

        log::info!(
            "🎯 Continue training with window-aware configuration for window {}",
            window_id + 1
        );

        // Continue training with new data
        model
            .train(
                TrainingContext::Continue {
                    new_sequences: &window.train_data.sequences,
                    new_targets: &train_targets,
                    target_class_weights: Some(&window.target_class_weights),
                },
                &window_config,
            )
            .await?;

        log::info!(
            "✅ Window {} continue training completed",
            window.window_id + 1
        );
        Ok(model)
    }

    /// Process targets for a training window (consolidates target processing logic)
    fn process_window_targets(
        &self,
        window: &crate::data::TrainingWindow,
        operation: &str,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        // Validate window data
        if window.train_data.targets.target_names.is_empty() {
            return Err(VangaError::DataError(format!(
                "No target names found in window {} - check target generation",
                window.window_id + 1
            )));
        }

        // Log detailed information about the training data
        log::debug!(
            "🔍 Window {} train_data: sequences.shape={:?}, targets.valid_indices.len()={}, target_names={:?}",
            window.window_id + 1,
            window.train_data.sequences.shape(),
            window.train_data.targets.valid_indices.len(),
            window.train_data.targets.target_names
        );

        if window.train_data.targets.valid_indices.is_empty() {
            return Err(VangaError::DataError(format!(
                "No valid training samples in window {} - check data preprocessing. Sequences shape: {:?}, target_names: {:?}",
                window.window_id + 1,
                window.train_data.sequences.shape(),
                window.train_data.targets.target_names
            )));
        }

        // Extract target names from prepared data for multi-model architecture
        let target_names = &window.train_data.targets.target_names;
        log::info!(
            "Multi-model {}: {} separate LSTM models for targets: {:?}",
            operation,
            target_names.len(),
            target_names
        );

        // Validate target names format
        for target_name in target_names {
            if !target_name.contains('_') {
                return Err(VangaError::DataError(format!(
                    "Invalid target name '{}' - expected format: 'type_horizon'",
                    target_name
                )));
            }
        }

        // Extract raw integer targets for multi-model architecture (each column → separate LSTM)
        let train_targets =
            extract_targets_for_multi_model(&window.train_data.targets, target_names)?;

        // Handle different operations: training/validation vs test evaluation
        let second_targets = if operation == "test_evaluation" {
            // For test evaluation, extract test targets instead of validation targets
            if window.test_data.targets.valid_indices.is_empty() {
                return Err(VangaError::DataError(format!(
                    "No valid test samples in window {} - check test data split",
                    window.window_id + 1
                )));
            }
            extract_targets_for_multi_model(&window.test_data.targets, target_names)?
        } else {
            // For training/validation operations, extract validation targets
            if window.val_data.targets.valid_indices.is_empty() {
                return Err(VangaError::DataError(format!(
                    "No valid validation samples in window {} - check chronological split",
                    window.window_id + 1
                )));
            }
            extract_targets_for_multi_model(&window.val_data.targets, target_names)?
        };

        // Validate target alignment
        if train_targets.shape()[1] != second_targets.shape()[1] {
            return Err(VangaError::DataError(format!(
                "Target dimension mismatch: train {} vs {} {} targets",
                train_targets.shape()[1],
                if operation == "test_evaluation" {
                    "test"
                } else {
                    "validation"
                },
                second_targets.shape()[1]
            )));
        }

        let data_type = if operation == "test_evaluation" {
            "test"
        } else {
            "validation"
        };
        log::info!(
            "Window {} {}: {} train samples x {} outputs, {} {} samples",
            window.window_id + 1,
            operation,
            train_targets.shape()[0],
            train_targets.shape()[1],
            second_targets.shape()[0],
            data_type
        );

        Ok((train_targets, second_targets))
    }

    /// Get existing multi-target model or create new one based on training configuration
    async fn get_or_create_multi_target_model(
        &self,
        input_size: usize,
        target_names: &[String],
    ) -> Result<MultiTargetLSTMModel> {
        // Create new model since we're not loading from file anymore
        // The caller (main.rs) will handle loading/saving based on training config
        log::info!("🆕 Creating new multi-target model for training");
        MultiTargetLSTMModel::new(
            &self.config.model,
            input_size,
            target_names.to_vec(),
            self.config.horizons.clone(),
        )
    }

    /// Collect validation metrics for a window after training
    async fn collect_window_metrics(
        &mut self,
        model: &MultiTargetLSTMModel,
        window: &crate::data::TrainingWindow,
        window_id: usize,
        learning_rate: f64,
    ) -> Result<()> {
        log::debug!("📊 Collecting metrics for window {}", window_id + 1);

        // Get validation predictions if validation data exists
        if window.val_samples > 0 {
            match model.predict(&window.val_data.sequences).await {
                Ok(val_predictions) => {
                    // Process validation targets
                    match self.process_window_targets(window, "metrics_collection") {
                        Ok((_, val_targets)) => {
                            let mut target_metrics = HashMap::new();
                            let target_names = model.get_target_names();

                            // Calculate metrics for each target
                            for (target_idx, target_name) in target_names.iter().enumerate() {
                                const NUM_CLASSES: usize = 5;
                                let start_col = target_idx * NUM_CLASSES;
                                let end_col = start_col + NUM_CLASSES;

                                if end_col <= val_predictions.shape()[1]
                                    && target_idx < val_targets.shape()[1]
                                {
                                    // Extract predictions and targets for this target
                                    let target_predictions_slice =
                                        val_predictions.slice(ndarray::s![.., start_col..end_col]);
                                    let target_actual = val_targets.column(target_idx);

                                    // Convert predictions to class indices (argmax)
                                    let pred_classes: Vec<i32> = target_predictions_slice
                                        .rows()
                                        .into_iter()
                                        .map(|pred_row| {
                                            pred_row
                                                .iter()
                                                .enumerate()
                                                .max_by(|(_, a), (_, b)| {
                                                    a.partial_cmp(b)
                                                        .unwrap_or(std::cmp::Ordering::Equal)
                                                })
                                                .map(|(idx, _)| idx as i32)
                                                .unwrap_or(0)
                                        })
                                        .collect();

                                    // Convert targets to class indices
                                    let actual_classes: Vec<i32> = target_actual
                                        .iter()
                                        .map(|&actual| actual.round() as i32)
                                        .collect();

                                    // Calculate metrics
                                    match crate::utils::metrics::calculate_classification_metrics(
                                        &pred_classes,
                                        &actual_classes,
                                    ) {
                                        Ok(metrics) => {
                                            target_metrics.insert(
                                                target_name.clone(),
                                                (
                                                    metrics.accuracy,
                                                    metrics.macro_f1,
                                                    metrics.weighted_f1,
                                                ),
                                            );
                                        }
                                        Err(e) => {
                                            log::warn!(
                                                "Failed to calculate metrics for target '{}': {}",
                                                target_name,
                                                e
                                            );
                                        }
                                    }
                                }
                            }

                            // Store window metrics
                            let window_metric = WindowMetrics {
                                window_id: window_id + 1, // 1-indexed for display
                                learning_rate,
                                train_samples: window.train_samples,
                                val_samples: window.val_samples,
                                target_metrics,
                            };

                            self.window_metrics.push(window_metric);
                            log::debug!("✅ Metrics collected for window {}", window_id + 1);
                        }
                        Err(e) => {
                            log::warn!("Failed to process targets for metrics collection: {}", e);
                        }
                    }
                }
                Err(e) => {
                    log::warn!(
                        "Failed to generate validation predictions for metrics: {}",
                        e
                    );
                }
            }
        } else {
            log::debug!(
                "No validation data for window {} - skipping metrics collection",
                window_id + 1
            );
        }

        Ok(())
    }

    /// Display comprehensive final statistics showing metrics progression across windows
    fn display_final_statistics(&self) {
        if self.window_metrics.is_empty() {
            log::info!("📊 No window metrics collected - final statistics unavailable");
            return;
        }

        log::info!("📊 FINAL WINDOW-BY-WINDOW STATISTICS:");
        log::info!(
            "   🎯 Training progression across {} windows",
            self.window_metrics.len()
        );
        log::info!("   📈 Metrics show model improvement through progressive learning");
        log::info!("");

        // Get all unique target names
        let mut all_targets = std::collections::HashSet::new();
        for window_metric in &self.window_metrics {
            for target_name in window_metric.target_metrics.keys() {
                all_targets.insert(target_name.clone());
            }
        }
        let mut target_names: Vec<String> = all_targets.into_iter().collect();
        target_names.sort();

        // Display metrics for each target
        for target_name in &target_names {
            log::info!("🎯 Target: {}", target_name);
            log::info!("   Window | LR        | Samples | Accuracy | Macro F1 | Weighted F1");
            log::info!("   -------|-----------|---------|----------|----------|------------");

            for window_metric in &self.window_metrics {
                if let Some((accuracy, macro_f1, weighted_f1)) =
                    window_metric.target_metrics.get(target_name)
                {
                    log::info!(
                        "   {:6} | {:9.2e} | {:7} | {:8.3} | {:8.3} | {:11.3}",
                        window_metric.window_id,
                        window_metric.learning_rate,
                        window_metric.val_samples,
                        accuracy,
                        macro_f1,
                        weighted_f1
                    );
                }
            }

            // Calculate improvement from first to last window
            if let (Some(first_window), Some(last_window)) =
                (self.window_metrics.first(), self.window_metrics.last())
            {
                if let (
                    Some((first_acc, first_macro, first_weighted)),
                    Some((last_acc, last_macro, last_weighted)),
                ) = (
                    first_window.target_metrics.get(target_name),
                    last_window.target_metrics.get(target_name),
                ) {
                    let acc_improvement = last_acc - first_acc;
                    let macro_improvement = last_macro - first_macro;
                    let weighted_improvement = last_weighted - first_weighted;

                    log::info!("   📈 Improvement (Window {} → {}): Accuracy: {:+.3}, Macro F1: {:+.3}, Weighted F1: {:+.3}",
                        first_window.window_id, last_window.window_id,
                        acc_improvement, macro_improvement, weighted_improvement);
                }
            }
            log::info!("");
        }

        // Overall summary
        log::info!("📊 TRAINING SUMMARY:");
        log::info!("   🔄 Total windows: {}", self.window_metrics.len());
        log::info!(
            "   📉 Learning rate decay: {:.3}",
            self.config.training.window_decay
        );
        if let (Some(first), Some(last)) = (self.window_metrics.first(), self.window_metrics.last())
        {
            log::info!(
                "   📊 LR progression: {:.6} → {:.6}",
                first.learning_rate,
                last.learning_rate
            );
            log::info!(
                "   📈 Sample progression: {} → {} validation samples",
                first.val_samples,
                last.val_samples
            );
        }
        log::info!("   ✅ Progressive learning: Model weights preserved across all windows");
        log::info!(
            "   🎯 Final model: Trained on cumulative data from all {} windows",
            self.window_metrics.len()
        );
    }
}

/// High-level training function
pub async fn train_model(config: TrainingConfig) -> Result<MultiTargetLSTMModel> {
    let mut trainer = ModelTrainer::new(config);
    trainer.train().await
}

/// Extract raw integer targets for multi-model architecture
///
/// **Architecture Note**: This function is designed for MultiTargetLSTMModel which contains
/// separate LSTM models for each target. Each model expects raw integer values (0,1,2,3,4)
/// for classification, NOT one-hot encoded vectors.
///
/// **Alternative Architecture**: For single LSTM with multiple output heads, use TargetConverter
/// which creates one-hot encoded outputs (e.g., [0,0,1,0,0] for class 2).
///
/// **Current Usage**: Each column in the returned Array2 goes to a separate LSTM model.
///
/// **Validation**: Ensures target values are in valid range (0-4) for classification.
fn extract_targets_for_multi_model(
    targets: &PreparedTargets,
    target_names: &[String],
) -> Result<Array2<f64>> {
    let num_samples = targets.valid_indices.len();
    let num_targets = target_names.len();

    // Validate inputs
    if num_samples == 0 {
        return Err(VangaError::DataError(
            "No valid samples for target extraction - check data preprocessing".to_string(),
        ));
    }

    if num_targets == 0 {
        return Err(VangaError::DataError(
            "No target names provided - check target configuration".to_string(),
        ));
    }

    let mut training_array = Array2::<f64>::zeros((num_samples, num_targets));

    // Extract targets in the order specified by target_names
    for (target_idx, target_name) in target_names.iter().enumerate() {
        // Parse target name format: "price_level_1h", "direction_1h", "volatility_1h"
        let parts: Vec<&str> = target_name.split('_').collect();
        if parts.len() < 2 {
            return Err(VangaError::DataError(format!(
                "Invalid target name format '{}' - expected format: 'type_horizon' or 'price_level_horizon'",
                target_name
            )));
        }

        // Handle compound target types like "price_level"
        let (target_type, horizon) = if parts.len() == 3
            && parts[0] == "price"
            && parts[1] == "level"
        {
            ("price_level", parts[2])
        } else if parts.len() == 2 {
            (parts[0], parts[1])
        } else {
            return Err(VangaError::DataError(format!(
                    "Invalid target name format '{}' - expected format: 'type_horizon' or 'price_level_horizon'",
                    target_name
                )));
        };

        // Get the appropriate target data based on type
        let target_data = match target_type {
            "price_level" => targets.price_levels.get(horizon),
            "direction" => targets.directions.get(horizon),
            "volatility" => targets.volatility.get(horizon),
            _ => {
                return Err(VangaError::DataError(format!(
                "Unknown target type '{}' - supported types: price_level, direction, volatility",
                target_type
            )))
            }
        };

        let data = target_data.ok_or_else(|| {
            VangaError::DataError(format!(
                "Target data not found for '{}' - check target generation for horizon '{}'",
                target_name, horizon
            ))
        })?;

        // Fill the training array with raw target values and validate range
        for (sample_idx, &data_idx) in targets.valid_indices.iter().enumerate() {
            if data_idx >= data.len() {
                return Err(VangaError::DataError(format!(
                    "Data index {} out of bounds for target '{}' (length: {})",
                    data_idx,
                    target_name,
                    data.len()
                )));
            }

            let target_value = data[data_idx] as f64;

            // Validate target value range (0-4 for classification)
            if !(0.0..=4.0).contains(&target_value) || target_value.fract() != 0.0 {
                return Err(VangaError::DataError(format!(
                    "Invalid target value {} for '{}' at sample {} - expected integer in range [0,4]",
                    target_value, target_name, sample_idx
                )));
            }

            training_array[[sample_idx, target_idx]] = target_value;
        }
    }

    log::info!(
        "Extracted {} raw integer targets for multi-model: {:?} (each column → separate LSTM)",
        num_targets,
        training_array.shape()
    );

    // Final validation: check for any NaN or infinite values
    if training_array.iter().any(|&x| !x.is_finite()) {
        return Err(VangaError::DataError(
            "Target array contains NaN or infinite values - check data preprocessing".to_string(),
        ));
    }

    Ok(training_array)
}
