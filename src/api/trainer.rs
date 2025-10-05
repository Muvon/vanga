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
use crate::targets::{PreparedTargets, TargetType};
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;
use std::collections::HashMap;

/// Perfect balance validation for PreparedTargets
/// Ensures all 5 classes have exactly the same number of samples for each target type and horizon
fn validate_prepared_targets_balance(targets: &PreparedTargets, data_name: &str) -> Result<()> {
    if targets.valid_indices.is_empty() {
        return Err(VangaError::DataError(format!(
            "🚨 BALANCE VALIDATION FAILED: {} data has no valid indices",
            data_name
        )));
    }

    let horizons = targets.get_horizons();
    if horizons.is_empty() {
        return Err(VangaError::DataError(format!(
            "🚨 BALANCE VALIDATION FAILED: {} data has no horizons",
            data_name
        )));
    }

    let target_types = [
        (TargetType::PriceLevel, "Price Level"),
        (TargetType::Direction, "Direction"),
        (TargetType::Volatility, "Volatility"),
        (TargetType::Sentiment, "Sentiment"),
        (TargetType::Volume, "Volume"),
    ];

    let mut all_balanced = true;
    let mut error_details = Vec::new();

    for horizon in &horizons {
        for (target_type, target_name) in &target_types {
            if let Some(target_values) = targets.get_targets(horizon, *target_type) {
                // Count class occurrences for valid indices only
                let mut class_counts = [0usize; 5];

                for &idx in &targets.valid_indices {
                    if idx < target_values.len() {
                        let class_value = target_values[idx];
                        if (0..5).contains(&class_value) {
                            class_counts[class_value as usize] += 1;
                        } else {
                            return Err(VangaError::DataError(format!(
                                "🚨 BALANCE VALIDATION FAILED: Invalid class {} for {} {} in {} data (expected 0-4)",
                                class_value, target_name, horizon, data_name
                            )));
                        }
                    }
                }

                // Check for perfect balance: all classes must have exactly the same count
                let min_count = *class_counts.iter().min().unwrap();
                let max_count = *class_counts.iter().max().unwrap();

                if min_count != max_count {
                    all_balanced = false;
                    let total_samples: usize = class_counts.iter().sum();
                    error_details.push(format!(
                        "  {} {} Target: {} total samples\n    Class 0: {} samples ({:.1}%)\n    Class 1: {} samples ({:.1}%)\n    Class 2: {} samples ({:.1}%)\n    Class 3: {} samples ({:.1}%)\n    Class 4: {} samples ({:.1}%)\n    ❌ IMBALANCE: min={}, max={}, ratio={:.2}x",
                        target_name,
                        horizon,
                        total_samples,
                        class_counts[0], (class_counts[0] as f64 / total_samples as f64) * 100.0,
                        class_counts[1], (class_counts[1] as f64 / total_samples as f64) * 100.0,
                        class_counts[2], (class_counts[2] as f64 / total_samples as f64) * 100.0,
                        class_counts[3], (class_counts[3] as f64 / total_samples as f64) * 100.0,
                        class_counts[4], (class_counts[4] as f64 / total_samples as f64) * 100.0,
                        min_count,
                        max_count,
                        max_count as f64 / min_count as f64
                    ));
                } else {
                    log::info!(
                        "✅ {} {} Target PERFECTLY BALANCED: {} samples per class (total: {})",
                        target_name,
                        horizon,
                        min_count,
                        min_count * 5
                    );
                }
            }
        }
    }

    if !all_balanced {
        let error_msg = format!(
            "🚨 PERFECT BALANCE VALIDATION FAILED for {} data:\n\n{}\n\n💡 SOLUTION: Use balanced data preparation pipeline to ensure exactly equal class counts.\n   Each of the 5 classes must have identical sample counts for all target types and horizons.",
            data_name,
            error_details.join("\n\n")
        );
        return Err(VangaError::DataError(error_msg));
    }

    log::info!(
        "🎯 {} DATA PERFECTLY BALANCED: All targets and horizons have equal class distribution",
        data_name.to_uppercase()
    );
    Ok(())
}

/// Validate that training and validation sequences are completely unique (no exact duplicates)
/// This prevents data leakage where the model memorizes training patterns and gets false validation scores
fn validate_sequence_uniqueness(
    train_sequences: &ndarray::Array3<f64>,
    val_sequences: &ndarray::Array3<f64>,
    window_id: usize,
) -> Result<()> {
    if train_sequences.is_empty() || val_sequences.is_empty() {
        log::warn!(
            "⚠️ UNIQUENESS VALIDATION SKIPPED: Empty sequence data for window {}",
            window_id + 1
        );
        return Ok(());
    }

    let train_samples = train_sequences.shape()[0];
    let val_samples = val_sequences.shape()[0];
    let sequence_length = train_sequences.shape()[1];
    let num_features = train_sequences.shape()[2];

    if train_sequences.shape()[1..] != val_sequences.shape()[1..] {
        return Err(VangaError::DataError(format!(
            "🚨 UNIQUENESS VALIDATION FAILED: Training sequences shape {:?} doesn't match validation shape {:?}",
            train_sequences.shape(),
            val_sequences.shape()
        )));
    }

    log::debug!(
        "🔍 Checking sequence uniqueness: {} training vs {} validation sequences ({}×{} each)",
        train_samples,
        val_samples,
        sequence_length,
        num_features
    );

    let mut duplicate_count = 0;
    let mut duplicate_examples = Vec::new();
    const MAX_EXAMPLES: usize = 5; // Limit examples to avoid log spam

    // Check each validation sequence against all training sequences
    for val_idx in 0..val_samples {
        let val_sequence = val_sequences.slice(ndarray::s![val_idx, .., ..]);

        for train_idx in 0..train_samples {
            let train_sequence = train_sequences.slice(ndarray::s![train_idx, .., ..]);

            // Check if sequences are exactly identical (within floating point precision)
            let mut is_duplicate = true;
            'sequence_check: for time_step in 0..sequence_length {
                for feature_idx in 0..num_features {
                    let diff = (val_sequence[[time_step, feature_idx]]
                        - train_sequence[[time_step, feature_idx]])
                    .abs();
                    if diff > 1e-10 {
                        // Very small tolerance for floating point comparison
                        is_duplicate = false;
                        break 'sequence_check;
                    }
                }
            }

            if is_duplicate {
                duplicate_count += 1;

                if duplicate_examples.len() < MAX_EXAMPLES {
                    duplicate_examples.push((train_idx, val_idx));
                }

                // Break after finding first duplicate for this validation sequence
                break;
            }
        }
    }

    if duplicate_count > 0 {
        let error_msg = format!(
            "🚨 SEQUENCE UNIQUENESS VALIDATION FAILED: Found {} exact duplicates between training and validation sets in window {}",
            duplicate_count, window_id + 1
        );

        log::error!("{}", error_msg);
        log::error!("📊 Duplicate Statistics:");
        log::error!("   - Training sequences: {}", train_samples);
        log::error!("   - Validation sequences: {}", val_samples);
        log::error!("   - Exact duplicates found: {}", duplicate_count);
        log::error!(
            "   - Duplicate rate: {:.2}%",
            (duplicate_count as f64 / val_samples as f64) * 100.0
        );

        if !duplicate_examples.is_empty() {
            log::error!("🔍 Example duplicates (train_idx, val_idx):");
            for (train_idx, val_idx) in &duplicate_examples {
                log::error!("   - Training[{}] == Validation[{}]", train_idx, val_idx);
            }
            if duplicate_examples.len() < duplicate_count {
                log::error!(
                    "   - ... and {} more duplicates",
                    duplicate_count - duplicate_examples.len()
                );
            }
        }

        log::error!("💡 This causes data leakage - the model memorizes training patterns and gets false validation scores!");
        log::error!("💡 Check your data splitting logic to ensure proper temporal separation.");

        return Err(VangaError::DataError(error_msg));
    }

    log::info!(
        "✅ SEQUENCE UNIQUENESS VALIDATED: No exact duplicates found between {} training and {} validation sequences in window {}",
        train_samples, val_samples, window_id + 1
    );

    Ok(())
}

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
    /// Train multi-target LSTM models using target-specific balanced sequences
    ///
    /// ARCHITECTURE OVERVIEW:
    /// 1. Each (target_type, horizon) gets its own balanced sequences with different sample counts
    /// 2. For each target, we create a MultiTargetLSTMModel wrapper containing ONLY that target
    /// 3. This ensures Direction sequences only train Direction models, not PriceLevel/Volatility
    /// 4. MultiTargetLSTMModel is a wrapper with Vec<LSTMModel> - one LSTMModel per target
    /// 5. In target-specific training, the wrapper contains only ONE LSTMModel
    ///
    /// EXPECTED LOGS:
    /// - "Direction 12h: 985 sequences" → Creates MultiTargetLSTMModel with 1 Direction LSTMModel
    /// - "PriceLevel 12h: 1250 sequences" → Creates MultiTargetLSTMModel with 1 PriceLevel LSTMModel
    /// - "🎯 Using 985 train samples" → Individual LSTMModel training log (correct per-target count)
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

        // CRITICAL: Calibrate target parameters BEFORE data pipeline
        log::info!("🎯 Calibrating target parameters for balanced class distributions...");
        let data_loader = crate::data::DataLoader::new();
        let raw_data = data_loader.load_csv(&self.config.data_path).await?;
        let ohlcv_data = crate::utils::market_data::extract_ohlcv_data(&raw_data)?;

        // Get ACTUAL sequence length from config
        let sequence_length = match &self.config.model.sequence_length {
            crate::config::model::SequenceLengthConfig::Fixed(len) => *len as usize,
            crate::config::model::SequenceLengthConfig::Auto {
                min_length,
                max_length,
            } => {
                // For auto, use the middle value as best guess for calibration
                ((min_length + max_length) / 2) as usize
            }
            crate::config::model::SequenceLengthConfig::Adaptive => {
                // For adaptive, we need to analyze the data to determine best length
                // For now use a reasonable default
                60
            }
        };

        log::info!(
            "🎯 Calibrating with sequence_length={} for {} horizons: {:?}",
            sequence_length,
            self.config.horizons.len(),
            self.config.horizons
        );

        // Use clean calibration interface with PER-HORIZON calibration
        let calibrator = crate::targets::calibration::ParameterCalibrator::from_config(&self.config.targets);

        let calibrated_params = calibrator
            .calibrate(
                &ohlcv_data,
                sequence_length,
                &self.config.horizons, // Pass ALL horizons
                None,                  // Use all available samples
                self.config.data.sequence_overlap,
            )
            .await?;

        log::info!(
            "✅ Per-horizon parameters calibrated successfully for {} horizons",
            self.config.horizons.len()
        );

        // Load and prepare target-specific training data with calibrated parameters
        log::info!(
            "Loading training data from: {}",
            self.config.data_path.display()
        );
        let target_windows = data_pipeline
            .prepare_training_data_with_calibrated_params(
                &self.config.data_path,
                &self.config,
                Some(&calibrated_params),
            )
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

        // CORRECTED MULTI-TARGET TRAINING FLOW:
        // 1. Train ALL targets within each window (proper temporal progression)
        // 2. Each target gets single-target MultiTargetLSTMModel wrapper (1 LSTM inside)
        // 3. Collect all single-target models and combine into final multi-target model
        let mut all_target_models = Vec::new(); // Collect trained LSTM models
        let mut all_target_names = Vec::new(); // Track target names for combination
        let mut all_horizons = Vec::new(); // Track horizons for combination

        // Track model state across windows for each target
        use std::collections::HashMap;
        let mut target_models: HashMap<(crate::targets::TargetType, String), MultiTargetLSTMModel> =
            HashMap::new();

        // Determine maximum number of windows across all targets
        let max_windows = target_windows
            .windows_by_target
            .values()
            .map(|windows| windows.len())
            .max()
            .unwrap_or(0);

        if max_windows == 0 {
            return Err(VangaError::DataError(
                "No training windows found".to_string(),
            ));
        }

        log::info!(
            "🔄 CORRECTED TRAINING FLOW: {} windows × {} targets (window-first approach for proper temporal progression)",
            max_windows,
            target_windows.windows_by_target.len()
        );
        log::info!(
            "📊 Training progression: Window 1 (all targets) → Window 2 (all targets) → ... → Window {} (all targets)",
            max_windows
        );

        // CORRECTED LOOP: WINDOW FIRST, THEN TARGETS (proper temporal progression)
        for window_idx in 0..max_windows {
            log::info!(
                "🪟 === WINDOW {}/{} === Training ALL targets within same temporal window",
                window_idx + 1,
                max_windows
            );

            // Calculate window-specific learning rate
            let window_lr = base_lr * self.config.training.window_decay.powi(window_idx as i32);
            log::info!(
                "📊 Window {} learning rate: {:.6} ({:.1}% of base)",
                window_idx + 1,
                window_lr,
                (window_lr / base_lr) * 100.0
            );

            // Train ALL targets within this window
            // CRITICAL: Sort targets for deterministic processing order (avoid random HashMap iteration)
            let mut sorted_targets: Vec<_> = target_windows.windows_by_target.keys().collect();
            sorted_targets.sort_by(|a, b| {
                // Sort by target type first, then by horizon
                match a.0.cmp(&b.0) {
                    std::cmp::Ordering::Equal => a.1.cmp(&b.1),
                    other => other,
                }
            });

            for (target_type, horizon) in sorted_targets {
                let windows = target_windows
                    .windows_by_target
                    .get(&(*target_type, horizon.clone()))
                    .unwrap();

                // Skip if this target doesn't have enough windows
                if window_idx >= windows.len() {
                    log::debug!(
                        "⏭️  Skipping {:?} {} - only has {} windows (need window {})",
                        target_type,
                        horizon,
                        windows.len(),
                        window_idx + 1
                    );
                    continue;
                }

                let window = &windows[window_idx];
                let target_key = (*target_type, horizon.clone());
                log::info!(
                    "🎯 {:?} {} Window {}/{}: {} train samples, {} validation samples",
                    target_type,
                    horizon,
                    window_idx + 1,
                    windows.len(),
                    window.train_samples,
                    window.val_samples
                );

                if window_idx == 0 {
                    // WINDOW 1: Train fresh model from scratch with initial data
                    log::info!(
                        "🆕 {:?} {} Window 1: Training fresh model from scratch",
                        target_type,
                        horizon
                    );
                    let new_model = self
                        .train_window_from_scratch(window, window_idx, *target_type, horizon)
                        .await?;
                    target_models.insert(target_key, new_model);
                } else {
                    // WINDOW 2+: Continue training with EXTENDED data (preserves learned weights)
                    // CRITICAL: Uses SAME target's extended sequences, NOT new sequences only
                    log::info!(
                        "🔄 {:?} {} Window {}: Continuing training with EXTENDED data (preserves weights)",
                        target_type,
                        horizon,
                        window_idx + 1
                    );

                    // EXTRACT MODEL: Get existing trained model for continuation
                    let existing_model = target_models.remove(&target_key).ok_or_else(|| {
                        VangaError::ModelError(format!(
                            "No existing model found for {:?} {} at window {}. This should not happen as window 0 creates the initial model.",
                            target_type, horizon, window_idx + 1
                        ))
                    })?;

                    let continued_model = self
                        .continue_training_window(
                            existing_model,
                            window,
                            window_idx,
                            *target_type,
                            horizon,
                            window_lr,
                        )
                        .await?;

                    // Store back the continued model
                    let target_key_for_insert = (*target_type, horizon.clone());
                    target_models.insert(target_key_for_insert, continued_model);
                }

                // Collect validation metrics for this window and target
                let target_key_for_metrics = (*target_type, horizon.clone());
                if let Some(current_model) = target_models.get(&target_key_for_metrics) {
                    self.collect_window_metrics(current_model, window, window_idx, window_lr)
                        .await?;
                }

                log::info!(
                    "✅ {:?} {} Window {} completed successfully",
                    target_type,
                    horizon,
                    window_idx + 1
                );
            }

            log::info!(
                "🏁 Window {}/{} completed - all {} targets trained within same temporal window ✅",
                window_idx + 1,
                max_windows,
                target_windows.windows_by_target.len()
            );
        }

        // COLLECT ALL TRAINED MODELS: Extract final models from HashMap
        log::info!("🔗 Collecting all trained models from window-based training");
        for ((target_type, horizon), model) in target_models {
            // Extract the single LSTMModel from the MultiTargetLSTMModel wrapper
            let single_models = model.extract_models()?;
            if single_models.len() != 1 {
                return Err(VangaError::ModelError(format!(
                    "Expected single model for {:?} {}, found {}",
                    target_type,
                    horizon,
                    single_models.len()
                )));
            }

            // COLLECT: Add to final model collection (ALL targets preserved)
            all_target_models.push(single_models.into_iter().next().unwrap());
            all_target_names.push(generate_target_name(target_type, &horizon));
            // Only add horizon if not already present to avoid duplicates
            if !all_horizons.contains(&horizon) {
                all_horizons.push(horizon.clone());
            }

            log::info!(
                "✅ Collected trained model for {:?} {} ({}/{} models total)",
                target_type,
                &horizon,
                all_target_models.len(),
                target_windows.windows_by_target.len()
            );
        }

        // COMBINE ALL MODELS: Create final MultiTargetLSTMModel containing ALL trained targets
        // ARCHITECTURE: MultiTargetLSTMModel = wrapper containing Vec<LSTMModel> (one per target)
        if all_target_models.is_empty() {
            return Err(VangaError::TrainingError(
                "No target models were trained".to_string(),
            ));
        }

        log::info!(
            "🔗 Combining {} single-target models into final MultiTargetLSTMModel: {:?}",
            all_target_models.len(),
            all_target_names
        );

        // CREATE FINAL MODEL: All targets combined, each trained on its own sequences
        let mut final_model = MultiTargetLSTMModel::from_trained_models(
            all_target_models, // Vec<LSTMModel> - one per target
            all_target_names,  // ["price_level_12h", "direction_12h", "volatility_12h"]
            all_horizons,      // ["12h", "12h", "12h"]
            target_windows.windows_by_target.values().next().unwrap()[0]
                .train_data
                .sequences
                .shape()[2], // input_size
            Some(calibrated_params.clone()), // Calibrated parameters for reconstruction
        )?;

        // Set complete training config on model before saving
        final_model.set_training_config(self.config.clone());

        // CRITICAL: Set calibrated target parameters for consistent prediction reconstruction
        final_model.set_calibrated_parameters(calibrated_params);
        log::info!("✅ Calibrated target parameters stored with model for consistent prediction reconstruction");

        // Enable final test evaluation with target-specific balanced test data
        // Each target now has its own balanced test split for proper evaluation
        if self.config.training.test_split > 0.0 {
            log::info!("🧪 Final test evaluation enabled with target-specific balanced test data");
            // TODO: Implement comprehensive test evaluation across all target models
        } else {
            log::info!("🧪 Test evaluation disabled (test_split = 0.0)");
        }

        // Display final window-by-window statistics
        self.display_final_statistics();

        // Save the trained multi-target model
        log::info!("✅ Walk-forward multi-target model training completed successfully!");
        log::info!("🔧 Complete training config saved with model for consistent prediction");
        Ok(final_model)
    }

    /// Train single-target model from scratch using target-specific balanced sequences
    ///
    /// ARCHITECTURE CLARIFICATION:
    /// - MultiTargetLSTMModel is a WRAPPER containing multiple individual LSTMModel instances
    /// - In target-specific windows, we create MultiTargetLSTMModel with ONLY ONE target
    /// - This ensures target-specific sequences are only used for the matching target
    /// - Avoids training PriceLevel models on Direction sequences, etc.
    async fn train_window_from_scratch(
        &self,
        window: &crate::data::TrainingWindow,
        window_id: usize,
        target_type: crate::targets::TargetType,
        horizon: &str,
    ) -> Result<MultiTargetLSTMModel> {
        log::info!(
            "🎯 Training {:?} {} with target-specific balanced sequences ({} samples)",
            target_type,
            horizon,
            window.train_samples
        );

        // Process targets for multi-model architecture
        let (train_targets, val_targets) = self.process_window_targets(window, "training")?;

        // CRITICAL FIX: Create MultiTargetLSTMModel with ONLY the current target being trained
        // This window contains target-specific balanced sequences, so we should only train the matching target
        let input_size = window.train_data.sequences.shape()[2];

        // Generate target name using helper function
        let current_target_name = generate_target_name(target_type, horizon);

        // Validate target exists in window
        if !window
            .train_data
            .targets
            .target_names
            .contains(&current_target_name)
        {
            return Err(VangaError::DataError(format!(
                "Target '{}' not found in window. Available targets: {:?}",
                current_target_name, window.train_data.targets.target_names
            )));
        }

        // Create MultiTargetLSTMModel wrapper with SINGLE target (avoids training wrong sequences on wrong targets)
        let target_names = vec![current_target_name.clone()];
        log::info!(
            "🎯 Creating single-target MultiTargetLSTMModel for '{}' using target-specific balanced sequences",
            current_target_name
        );

        let mut model = self
            .get_or_create_multi_target_model(input_size, &target_names)
            .await?;

        // Extract single target data using helper function
        let (single_train_target, single_val_target) = extract_single_target_data(
            &train_targets,
            Some(&val_targets),
            window,
            &current_target_name,
            "initial training",
        )?;

        // Create window-aware config that properly scales all scheduler parameters
        let window_config =
            crate::model::lstm::create_window_aware_config(&self.config, window_id)?;

        log::info!(
            "🎯 Training from scratch with window-aware configuration for window {}",
            window_id + 1
        );

        // Train the model with target-specific sequences and target-specific targets
        model
            .train(
                TrainingContext::Standard {
                    sequences: &window.train_data.sequences,
                    targets: &single_train_target,
                    val_sequences: Some(&window.val_data.sequences),
                    val_targets: single_val_target.as_ref(),
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
        target_type: crate::targets::TargetType,
        horizon: &str,
        window_lr: f64,
    ) -> Result<MultiTargetLSTMModel> {
        // Generate target name using helper function
        let current_target_name = generate_target_name(target_type, horizon);

        log::info!(
            "🎯 [continue_training_window] Single-target continuation for '{}' (window {})",
            current_target_name,
            window_id + 1
        );

        // Process targets for multi-model architecture (gets ALL targets)
        let (train_targets, _val_targets) =
            self.process_window_targets(window, "continue training")?;

        // Extract single target data using helper function (INCLUDE VALIDATION DATA)
        let (single_train_target, single_val_target) = extract_single_target_data(
            &train_targets,
            Some(&_val_targets), // ✅ CRITICAL FIX: Include validation data for continuation
            window,
            &current_target_name,
            "continuation training",
        )?;

        // Create window-aware config with the provided learning rate
        let mut window_config = self.config.clone();
        window_config.training.learning_rate = window_lr;

        log::info!(
            "🎯 Continue training with window-aware learning rate {:.6} for window {}",
            window_lr,
            window_id + 1
        );

        // Continue training with EXTENDED/CUMULATIVE data (preserves weights, includes validation to prevent overfitting)
        // CRITICAL: window.train_data.sequences contains CUMULATIVE data (Window1: 2880, Window2: 3744, etc.)
        model
            .train(
                TrainingContext::Continue {
                    new_sequences: &window.train_data.sequences, // CUMULATIVE sequences
                    new_targets: &single_train_target,
                    val_sequences: Some(&window.val_data.sequences), // CRITICAL: Prevents overfitting
                    val_targets: single_val_target.as_ref(), // CRITICAL: Prevents overfitting
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

        // 🎯 PERFECT BALANCE VALIDATION: Check training and validation data before extraction
        log::info!(
            "🔍 VALIDATING PERFECT BALANCE: Checking window {} data...",
            window.window_id + 1
        );

        // Validate training data balance
        validate_prepared_targets_balance(&window.train_data.targets, "TRAINING")?;

        // Validate validation data balance
        validate_prepared_targets_balance(&window.val_data.targets, "VALIDATION")?;

        // Validate sequence uniqueness (no exact duplicates between train/val)
        validate_sequence_uniqueness(
            &window.train_data.sequences,
            &window.val_data.sequences,
            window.window_id,
        )?;

        log::info!(
            "✅ PERFECT BALANCE CONFIRMED: Window {} data is perfectly balanced",
            window.window_id + 1
        );

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

/// XGBoost-only training function using existing LSTM model
///
/// This function loads an existing LSTM model and trains only the XGBoost component
/// on the LSTM features. It provides detailed progress logging and metrics comparison.
pub async fn train_xgboost_only_model(config: TrainingConfig) -> Result<MultiTargetLSTMModel> {
    log::info!(
        "🌲 Starting XGBoost-only training for symbol: {}",
        config.symbol
    );

    // Load existing LSTM model - let the load method handle all validation
    let model_path = crate::utils::model_path::get_model_path(&config.symbol);
    log::info!(
        "📂 Loading existing LSTM model from: {}",
        model_path.display()
    );

    let mut model = crate::model::multi_target::MultiTargetLSTMModel::load(&model_path)
        .map_err(|e| {
            crate::utils::error::VangaError::model(format!(
                "Failed to load LSTM model for symbol '{}' from {}: {}\n\
                Please train the full model first using: cargo run -- train --symbol {} --data <data_file>",
                config.symbol,
                model_path.display(),
                e,
                config.symbol
            ))
        })?;
    log::info!("✅ Successfully loaded existing LSTM model");

    // Update model with new training config
    model.set_training_config(config.clone());

    // Prepare data using existing pipeline
    log::info!("📊 Preparing training data for XGBoost training...");

    let data_pipeline = DataPipeline::new();

    // Load and prepare data with calibrated parameters from the model
    let calibrated_params = model
        .get_calibrated_parameters()
        .ok_or_else(|| VangaError::model("Model missing calibrated parameters"))?
        .clone();

    let target_windows = data_pipeline
        .prepare_training_data_with_calibrated_params(
            &config.data_path,
            &config,
            Some(&calibrated_params),
        )
        .await?;

    log::info!(
        "📊 Data prepared: {} target windows loaded",
        target_windows.windows_by_target.len()
    );

    // For XGBoost-only training, we need to train EACH model's XGBoost individually
    // with its OWN balanced data, not mix data from different targets

    let model_target_names = model.get_target_names();
    log::info!(
        "📊 Model has {} targets to train XGBoost for: {:?}",
        model_target_names.len(),
        model_target_names
    );

    // We need to train XGBoost for each target individually
    // The MultiTargetLSTMModel contains multiple individual LSTM models
    // Each needs its own XGBoost trained on its own balanced data

    // FIXED: Each target should use its OWN balanced sequences and targets
    // This matches how the regular training pipeline works

    log::info!("🔍 Collecting balanced data for each target:");
    let mut target_data_vec = Vec::new();

    for model_target_name in model_target_names.iter() {
        // Parse target name
        let parts: Vec<&str> = model_target_name.split('_').collect();
        let (target_type, horizon) =
            if parts.len() == 3 && parts[0] == "price" && parts[1] == "level" {
                (TargetType::PriceLevel, parts[2].to_string())
            } else if parts.len() == 2 {
                let target_type = match parts[0] {
                    "direction" => TargetType::Direction,
                    "volatility" => TargetType::Volatility,
                    "sentiment" => TargetType::Sentiment,
                    "volume" => TargetType::Volume,
                    _ => {
                        return Err(VangaError::DataError(format!(
                            "Unknown target type in '{}'",
                            model_target_name
                        )));
                    }
                };
                (target_type, parts[1].to_string())
            } else {
                return Err(VangaError::DataError(format!(
                    "Invalid target name format: '{}'",
                    model_target_name
                )));
            };

        // Get the window for this specific target
        let windows = target_windows
            .windows_by_target
            .get(&(target_type, horizon.clone()))
            .ok_or_else(|| {
                VangaError::DataError(format!(
                    "No balanced window found for {:?} {}",
                    target_type, horizon
                ))
            })?;

        let window = windows.first().ok_or_else(|| {
            VangaError::DataError(format!(
                "Empty window list for {:?} {}",
                target_type, horizon
            ))
        })?;

        // Extract this target's sequences (already balanced!)
        let sequences = window.train_data.sequences.clone();

        // Extract validation sequences if available
        let val_sequences = window.val_data.sequences.clone();

        // Extract the target data for this specific target
        let target_names = &window.train_data.targets.target_names;
        let all_targets =
            extract_targets_for_multi_model(&window.train_data.targets, target_names)?;

        // Find the specific target column
        let idx = target_names
            .iter()
            .position(|n| n == model_target_name)
            .ok_or_else(|| {
                VangaError::DataError(format!(
                    "Target '{}' not found in window. Available: {:?}",
                    model_target_name, target_names
                ))
            })?;

        // Extract single target as 2D array (required by train_xgboost_phase)
        let single_target = all_targets.slice(ndarray::s![.., idx..idx + 1]).to_owned();

        // Extract validation targets for this specific target
        let val_target_names = &window.val_data.targets.target_names;
        let all_val_targets =
            extract_targets_for_multi_model(&window.val_data.targets, val_target_names)?;
        let val_idx = val_target_names
            .iter()
            .position(|n| n == model_target_name)
            .ok_or_else(|| {
                VangaError::DataError(format!(
                    "Validation target '{}' not found",
                    model_target_name
                ))
            })?;
        let single_val_target = all_val_targets
            .slice(ndarray::s![.., val_idx..val_idx + 1])
            .to_owned();

        log::info!(
            "✅ Collected balanced data for {}: {} train sequences, {} val sequences",
            model_target_name,
            sequences.shape()[0],
            val_sequences.shape()[0]
        );

        // Verify balance
        let mut class_counts = std::collections::HashMap::new();
        for val in single_target.iter() {
            let class = *val as i32;
            *class_counts.entry(class).or_insert(0) += 1;
        }
        log::info!("   Class distribution for {}:", model_target_name);
        for class in 0..5 {
            let count = class_counts.get(&class).unwrap_or(&0);
            let percentage = (*count as f64 / single_target.len() as f64) * 100.0;
            log::info!(
                "      Class {}: {} samples ({:.1}%)",
                class,
                count,
                percentage
            );
        }

        target_data_vec.push((sequences, single_target, val_sequences, single_val_target));
    }

    // Train XGBoost with each target's own balanced data
    log::info!("🌲 Training XGBoost models using LSTM features with properly balanced data");
    model.train_xgboost_only(target_data_vec, &config).await?;

    log::info!("✅ XGBoost training completed successfully");

    // Save the updated model with XGBoost components
    let model_path = crate::utils::model_path::get_model_path(&config.symbol);
    log::info!(
        "💾 Saving updated model with XGBoost to: {}",
        model_path.display()
    );
    model.save(&model_path)?;
    log::info!("✅ Model saved successfully");

    log::info!("✅ XGBoost-only training completed for {}", config.symbol);
    Ok(model)
}
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
            "direction" => targets.direction.get(horizon),
            "volatility" => targets.volatility.get(horizon),
            "sentiment" => targets.sentiment.get(horizon),
            "volume" => targets.volume.get(horizon),
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

// HELPER FUNCTIONS: Code quality improvements for target handling

/// TARGET NAME GENERATION: Convert target type + horizon to standard format
/// PURPOSE: Eliminates duplication, ensures consistent naming across codebase
/// EXAMPLES: (PriceLevel, "12h") → "price_level_12h", (Direction, "4h") → "direction_4h"
fn generate_target_name(target_type: crate::targets::TargetType, horizon: &str) -> String {
    let target_type_str = match target_type {
        crate::targets::TargetType::PriceLevel => "price_level",
        crate::targets::TargetType::Direction => "direction",
        crate::targets::TargetType::Volatility => "volatility",
        crate::targets::TargetType::Sentiment => "sentiment",
        crate::targets::TargetType::Volume => "volume",
    };
    format!("{}_{}", target_type_str, horizon)
}

/// SINGLE TARGET EXTRACTION: Extract one target's data from multi-target arrays
/// PURPOSE: Prevents cross-contamination (PriceLevel models training on Direction data)
/// CRITICAL: Each target trains ONLY on its own sequences and validation data
/// INPUT: Multi-target arrays [samples, 3_targets] → OUTPUT: Single-target [samples, 1_target]
fn extract_single_target_data(
    train_targets: &ndarray::Array2<f64>, // All targets training data
    val_targets: Option<&ndarray::Array2<f64>>, // All targets validation data (optional)
    window: &crate::data::TrainingWindow, // Window containing target names
    target_name: &str,                    // Specific target to extract ("price_level_12h")
    operation: &str,                      // Operation context for error messages
) -> Result<(ndarray::Array2<f64>, Option<ndarray::Array2<f64>>)> {
    // Find target index with comprehensive error handling
    let target_idx = window
        .train_data
        .targets
        .target_names
        .iter()
        .position(|name| name == target_name)
        .ok_or_else(|| {
            VangaError::DataError(format!(
                "Target '{}' not found during {}. Available targets: {:?}",
                target_name, operation, window.train_data.targets.target_names
            ))
        })?;

    // Extract single-column target arrays
    let single_train_target = train_targets
        .column(target_idx)
        .to_owned()
        .insert_axis(ndarray::Axis(1));

    let single_val_target = val_targets.map(|val_targets| {
        val_targets
            .column(target_idx)
            .to_owned()
            .insert_axis(ndarray::Axis(1))
    });

    log::info!(
        "🎯 Extracted target '{}' (column {}) from {} total targets for {}",
        target_name,
        target_idx,
        train_targets.shape()[1],
        operation
    );

    Ok((single_train_target, single_val_target))
}
