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
//!     output_heads: 3  // Multiple output heads from one LSTM
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

/// Model trainer for multi-target LSTM architecture
///
/// Orchestrates the complete training pipeline including:
/// - Walk-forward chronological validation
/// - Multi-model target processing
/// - Progressive learning across time windows
/// - Normalization consistency for prediction
pub struct ModelTrainer {
    config: TrainingConfig,
}

impl ModelTrainer {
    /// Create new model trainer with configuration
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
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
    pub async fn train(&self) -> Result<MultiTargetLSTMModel> {
        log::info!("Starting model training for symbol: {}", self.config.symbol);

        // Initialize device from configuration
        let device_string = self.config.training.device.to_device_string();
        let device = crate::utils::device::DeviceManager::create_device(&device_string)?;
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

        // Load and prepare training data with chronological split
        log::info!(
            "Loading training data from: {}",
            self.config.data_path.display()
        );
        let windows = data_pipeline
            .prepare_training_data(&self.config.data_path, &self.config)
            .await?;

        log::info!(
            "Walk-forward training: {} windows prepared for progressive learning",
            windows.len()
        );

        // Train model using walk-forward analysis with window-based learning rate decay
        let mut model = None;

        // Get base learning rate from config
        let base_lr = match &self.config.training.learning_rate {
            crate::config::training::LearningRateConfig::Fixed(lr) => *lr,
            crate::config::training::LearningRateConfig::Adaptive { initial_lr, .. } => *initial_lr,
            crate::config::training::LearningRateConfig::Auto { max_lr, .. } => *max_lr,
        };

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

        for (i, window) in windows.iter().enumerate() {
            // Calculate window-specific learning rate: base_lr * decay^window_id
            let window_lr = base_lr * self.config.training.window_decay.powi(i as i32);

            log::info!(
                "🔄 Walk-forward window {}/{}: lr={:.6} ({:.1}% of base) → {} train samples, {} validation samples",
                i + 1,
                windows.len(),
                window_lr,
                (window_lr / base_lr) * 100.0,
                window.train_samples,
                window.val_samples
            );

            if i == 0 {
                // First window: train from scratch
                model = Some(self.train_window_from_scratch(window, window_lr).await?);
            } else {
                // Subsequent windows: continue training on expanded data
                model = Some(
                    self.continue_training_window(model.unwrap(), window, window_lr)
                        .await?,
                );
            }
        }

        let mut final_model = model.unwrap();

        // CRITICAL FIX: Set complete training config on model before saving
        // This ensures prediction can regenerate the same features and settings as training
        final_model.set_training_config(self.config.clone());

        // CRITICAL FIX: Set normalization stats from training data
        // This ensures prediction uses the same normalization as training
        if let Some(first_window) = windows.first() {
            final_model
                .set_normalization_stats(first_window.train_data.normalization_stats.clone());
            log::info!(
                "✅ Normalization stats saved with model for consistent prediction preprocessing"
            );
        } else {
            log::warn!("⚠️  No training windows available - normalization stats not saved");
        }

        // Save the trained multi-target model
        log::info!("✅ Walk-forward multi-target model training completed successfully!");
        log::info!("🔧 Complete training config saved with model for consistent prediction");
        Ok(final_model)
    }

    /// Train model from scratch on first window with window-specific learning rate
    async fn train_window_from_scratch(
        &self,
        window: &crate::data::TrainingWindow,
        window_lr: f64,
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

        // Create config with window-specific learning rate
        let mut window_config = self.config.clone();
        window_config.training.learning_rate =
            crate::config::training::LearningRateConfig::Fixed(window_lr);

        log::info!(
            "🎯 Training from scratch with window learning rate: {:.6}",
            window_lr
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

    /// Continue training existing model on new window with window-specific learning rate
    async fn continue_training_window(
        &self,
        mut model: MultiTargetLSTMModel,
        window: &crate::data::TrainingWindow,
        window_lr: f64,
    ) -> Result<MultiTargetLSTMModel> {
        log::info!(
            "🎯 [continue_training_window] Training config horizons: {:?} (count: {})",
            self.config.horizons,
            self.config.horizons.len()
        );

        // Process targets for multi-model architecture
        let (train_targets, _val_targets) =
            self.process_window_targets(window, "continue training")?;

        // Create config with window-specific learning rate
        let mut window_config = self.config.clone();
        window_config.training.learning_rate =
            crate::config::training::LearningRateConfig::Fixed(window_lr);

        log::info!(
            "🎯 Continue training with window learning rate: {:.6}",
            window_lr
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

        if window.train_data.targets.valid_indices.is_empty() {
            return Err(VangaError::DataError(format!(
                "No valid training samples in window {} - check data preprocessing",
                window.window_id + 1
            )));
        }

        if window.val_data.targets.valid_indices.is_empty() {
            return Err(VangaError::DataError(format!(
                "No valid validation samples in window {} - check chronological split",
                window.window_id + 1
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
        let val_targets = extract_targets_for_multi_model(&window.val_data.targets, target_names)?;

        // Validate target alignment
        if train_targets.shape()[1] != val_targets.shape()[1] {
            return Err(VangaError::DataError(format!(
                "Target dimension mismatch: train {} vs validation {} targets",
                train_targets.shape()[1],
                val_targets.shape()[1]
            )));
        }

        log::info!(
            "Window {} {}: {} train samples x {} outputs, {} validation samples",
            window.window_id + 1,
            operation,
            train_targets.shape()[0],
            train_targets.shape()[1],
            val_targets.shape()[0]
        );

        Ok((train_targets, val_targets))
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
}

/// High-level training function
pub async fn train_model(config: TrainingConfig) -> Result<MultiTargetLSTMModel> {
    let trainer = ModelTrainer::new(config);
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
