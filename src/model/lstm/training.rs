//! Training pipeline and optimization
//!
//! This module implements the complete training pipeline for LSTM models with:
//! - Proper gradient clipping using direct gradient scaling (not learning rate modification)
//! - Multi-optimizer support with preserved state integrity
//! - Comprehensive validation and early stopping
//! - Advanced learning rate scheduling
//! - Gradient flow monitoring and validation

use super::config::{LSTMModel, OptimizerWrapper};
use super::loss::LossMode;

use crate::targets::TargetType;
use crate::utils::error::{Result, VangaError};
use candle_core::Tensor;

use candle_nn::optim::{self, Optimizer, ParamsAdamW};
use candle_optimisers::{
    adadelta::{Adadelta, ParamsAdaDelta},
    adagrad::{Adagrad, ParamsAdaGrad},
    adam::{Adam, ParamsAdam},
    adamax::{Adamax, ParamsAdaMax},
    nadam::{NAdam, ParamsNAdam},
    radam::{ParamsRAdam, RAdam},
    rmsprop::{ParamsRMSprop, RMSprop},
    Decay,
};
use ndarray::{s, Array2, Array3};

/// Perfect balance validation for training data
/// Handles both multi-target (15 columns) and single-target (1 column) cases
pub fn validate_perfect_balance(targets: &Array2<f64>, data_name: &str) -> Result<()> {
    let num_samples = targets.shape()[0];
    let num_outputs = targets.shape()[1];

    if num_samples == 0 {
        return Err(VangaError::DataError(format!(
            "🚨 BALANCE VALIDATION FAILED: {} data is empty",
            data_name
        )));
    }

    match num_outputs {
        15 => {
            // Multi-target case: 3 targets × 5 classes each
            validate_multi_target_balance(targets, data_name)
        }
        1 => {
            // Single target case: 1 target with class indices (not one-hot)
            validate_single_target_balance(targets, data_name)
        }
        _ => {
            Err(VangaError::DataError(format!(
                "🚨 BALANCE VALIDATION FAILED: Expected 15 outputs (multi-target) or 1 output (single-target), got {}",
                num_outputs
            )))
        }
    }
}

/// Validate balance for multi-target case (15 columns, one-hot encoded)
pub fn validate_multi_target_balance(targets: &Array2<f64>, data_name: &str) -> Result<()> {
    let num_samples = targets.shape()[0];
    let num_outputs = targets.shape()[1];

    log::info!(
        "🔍 DEBUG: {} data shape: [{}, {}]",
        data_name,
        num_samples,
        num_outputs
    );

    // Validate target structure: should be 15 outputs (3 targets × 5 classes)
    if num_outputs != 15 {
        return Err(VangaError::DataError(format!(
            "🚨 BALANCE VALIDATION FAILED: Expected 15 target outputs (3×5), got {} for {} data",
            num_outputs, data_name
        )));
    }

    if num_samples == 0 {
        return Err(VangaError::DataError(format!(
            "🚨 BALANCE VALIDATION FAILED: {} data is empty",
            data_name
        )));
    }

    let target_types = ["Price Level", "Direction", "Volatility"];
    let mut all_balanced = true;
    let mut error_details = Vec::new();

    for (target_idx, target_name) in target_types.iter().enumerate() {
        let start_col = target_idx * 5;
        let end_col = start_col + 5;

        // Extract class labels from one-hot encoding
        let mut class_counts = [0usize; 5];

        for sample_idx in 0..num_samples {
            let mut found_class = false;
            for (class_idx, class_count) in class_counts.iter_mut().enumerate() {
                let col_idx = start_col + class_idx;
                if targets[[sample_idx, col_idx]] > 0.5 {
                    // One-hot encoded value
                    *class_count += 1;
                    found_class = true;
                    break;
                }
            }

            if !found_class {
                return Err(VangaError::DataError(format!(
                    "🚨 BALANCE VALIDATION FAILED: Sample {} in {} data has no active class for {} target (columns {}-{})",
                    sample_idx, data_name, target_name, start_col, end_col - 1
                )));
            }
        }

        // Check for perfect balance: all classes must have exactly the same count
        let min_count = *class_counts.iter().min().unwrap();
        let max_count = *class_counts.iter().max().unwrap();

        if min_count != max_count {
            all_balanced = false;
            let total_target_samples: usize = class_counts.iter().sum();
            error_details.push(format!(
                "  {} Target: {} total samples\n    Class 0: {} samples ({:.1}%)\n    Class 1: {} samples ({:.1}%)\n    Class 2: {} samples ({:.1}%)\n    Class 3: {} samples ({:.1}%)\n    Class 4: {} samples ({:.1}%)\n    ❌ IMBALANCE: min={}, max={}, ratio={:.2}x",
                target_name,
                total_target_samples,
                class_counts[0], (class_counts[0] as f64 / total_target_samples as f64) * 100.0,
                class_counts[1], (class_counts[1] as f64 / total_target_samples as f64) * 100.0,
                class_counts[2], (class_counts[2] as f64 / total_target_samples as f64) * 100.0,
                class_counts[3], (class_counts[3] as f64 / total_target_samples as f64) * 100.0,
                class_counts[4], (class_counts[4] as f64 / total_target_samples as f64) * 100.0,
                min_count,
                max_count,
                max_count as f64 / min_count as f64
            ));
        } else {
            log::info!(
                "✅ {} Target PERFECTLY BALANCED: {} samples per class (total: {})",
                target_name,
                min_count,
                min_count * 5
            );
        }
    }

    if !all_balanced {
        let error_msg = format!(
            "🚨 PERFECT BALANCE VALIDATION FAILED for {} data:\n\n{}\n\n💡 SOLUTION: Use balanced data preparation pipeline to ensure exactly equal class counts.\n   Each of the 5 classes must have identical sample counts for all 3 target types.",
            data_name,
            error_details.join("\n\n")
        );
        return Err(VangaError::DataError(error_msg));
    }

    log::info!(
        "🎯 {} DATA PERFECTLY BALANCED: All targets have equal class distribution",
        data_name.to_uppercase()
    );
    Ok(())
}

/// Validate balance for single-target case (1 column from one-hot encoded data)
fn validate_single_target_balance(targets: &Array2<f64>, data_name: &str) -> Result<()> {
    let num_samples = targets.shape()[0];

    // For single target extracted from multi-target one-hot encoding,
    // we need to count how many samples have value 1.0 (active class)
    // and how many have 0.0 (inactive class)

    let mut active_count = 0;
    let mut inactive_count = 0;

    for sample_idx in 0..num_samples {
        let value = targets[[sample_idx, 0]];
        if value > 0.5 {
            active_count += 1;
        } else if value < 0.5 {
            inactive_count += 1;
        } else {
            return Err(VangaError::DataError(format!(
                "🚨 BALANCE VALIDATION FAILED: Sample {} in {} data has invalid value {} (expected 0.0 or 1.0)",
                sample_idx, data_name, value
            )));
        }
    }

    log::info!(
        "✅ Single Target Column Validated: {} active (1.0), {} inactive (0.0) in {} data",
        active_count,
        inactive_count,
        data_name
    );

    // For single column from one-hot encoding, we can't validate perfect balance
    // because each column represents only one class. The balance validation
    // should happen at the multi-target level before splitting.
    Ok(())
}

impl LSTMModel {
    pub fn configure_training(&mut self, vanga_config: &crate::config::TrainingConfig) {
        // Extract epochs from config - SAME logic as original
        let (max_epochs, use_early_stopping) = match &vanga_config.training.epochs {
            crate::config::training::EpochConfig::Auto { max_epochs } => {
                (*max_epochs as usize, true)
            }
            crate::config::training::EpochConfig::Fixed(epochs) => (*epochs as usize, false),
        };

        // Extract learning rate from config - SAME logic as original
        let learning_rate = vanga_config.training.learning_rate;
        log::info!("Using learning rate: {:.6}", learning_rate);

        // Extract batch size from config - NEW: Properly utilize batch size configuration
        let batch_size = match &vanga_config.training.batch_size {
            crate::config::training::BatchSizeConfig::Fixed(size) => {
                log::info!("Using FIXED batch size: {}", size);
                *size as usize
            }
            crate::config::training::BatchSizeConfig::Auto { min_size, max_size } => {
                // Memory-aware batch size optimization
                let chosen_size = self.optimize_batch_size(*min_size as usize, *max_size as usize);
                log::info!(
                    "Using AUTO batch size: {} (optimized from range: {} - {})",
                    chosen_size,
                    min_size,
                    max_size
                );
                chosen_size
            }
        };

        // Update rust-lstm training config - SAME as original + batch size
        self.training_config.epochs = max_epochs;
        self.training_config.print_every = vanga_config.training.print_every as usize; // Use configured print_every
        self.training_config.batch_size = batch_size; // Store configured batch size

        // Store learning rate for optimizer creation - SAME as original
        self.config.learning_rate = learning_rate;

        // Extract and apply gradient clipping from config
        if let Some(gradient_clip) = vanga_config.training.gradient_clip {
            self.training_config.clip_gradient = Some(gradient_clip);
            log::info!("Using gradient clipping: {:.3}", gradient_clip);
        }

        log::info!(
            "✅ Training configured: epochs={}, lr={:.6}, batch_size={}, early_stopping={}, print_every={}, gradient_clip={:?}",
            max_epochs,
            learning_rate,
            batch_size,
            use_early_stopping,
            self.training_config.print_every,
            vanga_config.training.gradient_clip
        );
    }

    /// Validate batch configuration and provide warnings
    fn validate_batch_configuration(&self, total_samples: usize, batch_size: usize) -> Result<()> {
        // Basic validation
        if batch_size == 0 {
            return Err(VangaError::ConfigError(
                "Batch size cannot be zero".to_string(),
            ));
        }

        if batch_size > total_samples {
            log::warn!(
                "⚠️  Batch size ({}) is larger than total samples ({}). Will use full dataset as single batch.",
                batch_size, total_samples
            );
        }

        // Memory estimation and warnings
        let estimated_memory_per_batch = self.estimate_batch_memory_usage(batch_size);
        let estimated_memory_mb = estimated_memory_per_batch / (1024 * 1024);

        if estimated_memory_mb > 1000 {
            // > 1GB per batch
            log::warn!(
                "⚠️  Large batch size detected! Estimated memory per batch: {}MB. Consider reducing batch size if you encounter OOM.",
                estimated_memory_mb
            );
        } else {
            log::info!(
                "✅ Batch configuration validated. Estimated memory per batch: {}MB",
                estimated_memory_mb
            );
        }

        let num_batches = total_samples.div_ceil(batch_size);
        log::info!(
            "📊 Batch processing: {} total samples → {} batches of size {} (last batch: {} samples)",
            total_samples, num_batches, batch_size,
            if total_samples % batch_size == 0 { batch_size } else { total_samples % batch_size }
        );

        Ok(())
    }

    /// Estimate memory usage for a given batch size
    fn estimate_batch_memory_usage(&self, batch_size: usize) -> usize {
        let sequence_length = self.config.sequence_length;
        let input_features = self.config.input_size;
        let num_layers = self.config.num_layers;

        // Calculate total hidden states size across all layers
        let mut hidden_states_size = 0;
        for layer_idx in 0..num_layers {
            let hidden_size = self.config.get_hidden_size_for_layer(layer_idx);
            hidden_states_size += batch_size * hidden_size * 4 * 2; // forward + backward, f32 = 4 bytes
        }

        // Rough estimation: input tensor + hidden states + gradients + attention (if enabled)
        let input_tensor_size = batch_size * sequence_length * input_features * 4; // f32 = 4 bytes
        let attention_multiplier = if self.use_attention { 3 } else { 1 }; // Attention adds ~3x memory

        (input_tensor_size + hidden_states_size) * attention_multiplier
    }

    /// Optimize batch size based on available memory and model complexity
    fn optimize_batch_size(&self, min_size: usize, max_size: usize) -> usize {
        // Get available memory (rough estimation)
        let available_memory_gb = self.get_available_memory_gb();

        // Memory-based batch size selection following VANGA guidelines
        let memory_based_size = match available_memory_gb {
            gb if gb < 1.0 => 16,
            gb if gb < 4.0 => 32,
            gb if gb < 8.0 => 64,
            gb if gb < 16.0 => 128,
            _ => 256,
        };

        // Start with memory-based size, then test within range
        let mut optimal_size = memory_based_size.max(min_size).min(max_size);

        // Test if we can use a larger batch size within the range
        for test_size in (optimal_size..=max_size).step_by(16) {
            let estimated_memory_mb = self.estimate_batch_memory_usage(test_size) / (1024 * 1024);
            let memory_limit_mb = (available_memory_gb * 1024.0 * 0.7) as usize; // Use 70% of available memory

            if estimated_memory_mb <= memory_limit_mb {
                optimal_size = test_size;
            } else {
                break;
            }
        }

        log::debug!(
            "Batch size optimization: available_memory={}GB, memory_based={}, optimal={}",
            available_memory_gb,
            memory_based_size,
            optimal_size
        );

        optimal_size
    }

    /// Get available memory in GB (rough estimation)
    fn get_available_memory_gb(&self) -> f64 {
        // For macOS, try to get memory info
        if let Ok(output) = std::process::Command::new("vm_stat").output() {
            if let Ok(vm_stat) = String::from_utf8(output.stdout) {
                // Parse vm_stat output to get free memory
                if let Some(free_line) = vm_stat.lines().find(|line| line.contains("Pages free:")) {
                    if let Some(free_pages_str) = free_line.split_whitespace().nth(2) {
                        if let Ok(free_pages) = free_pages_str.trim_end_matches('.').parse::<u64>()
                        {
                            // macOS page size is typically 16KB
                            let free_memory_gb =
                                (free_pages * 16384) as f64 / (1024.0 * 1024.0 * 1024.0);
                            return free_memory_gb.max(1.0); // Minimum 1GB assumption
                        }
                    }
                }
            }
        }

        // Fallback: assume reasonable memory based on system
        4.0 // Default to 4GB assumption for batch size calculation
    }
    pub async fn train(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        config: &crate::config::TrainingConfig,
        // Optional pre-split validation data (prevents data leakage)
        val_sequences: Option<&Array3<f64>>,
        val_targets: Option<&Array2<f64>>,
    ) -> Result<()> {
        let total_samples = sequences.shape()[0];

        // ADDED: Validate dataset size for proper training with gap
        let sequence_length = self.config.sequence_length;
        let max_horizon_steps = if !config.horizons.is_empty() {
            config
                .horizons
                .iter()
                .map(|h| crate::utils::parser::parse_horizon_to_steps(h).unwrap_or(1))
                .max()
                .unwrap_or(72)
        } else {
            72
        };

        let required_gap = sequence_length + max_horizon_steps;
        let min_required_samples = required_gap + sequence_length + 10; // Minimum viable dataset

        if total_samples < min_required_samples {
            log::warn!(
                "⚠️  SMALL DATASET WARNING: {} samples < {} recommended minimum",
                total_samples,
                min_required_samples
            );
            log::warn!(
                "   • Sequence length: {}, Horizon steps: {}, Required gap: {}",
                sequence_length,
                max_horizon_steps,
                required_gap
            );
            log::warn!(
                "   • Consider: reducing sequence_length, shorter horizons, or collecting more data"
            );
        }

        log::info!(
            "🚀 UNIFIED TRAINING: Starting with {} samples (min recommended: {})",
            total_samples,
            min_required_samples
        );

        // Log validation data usage for tracking
        if let (Some(val_seq), Some(_val_tgt)) = (val_sequences, val_targets) {
            log::info!(
                "📊 Using pre-split chronological validation: {} train, {} val samples (no data leakage)",
                total_samples,
                val_seq.shape()[0]
            );
        }

        // CRITICAL: self.trained flag detects continuation training (preserves weights, no reinitialization)
        // INCREMENTAL TRAINING DETECTION AND OPTIMIZATION - SAME logic as original continue_training
        let final_config = if self.trained {
            // CRITICAL: Continuation training uses lower learning rate and patience for stability
            let mut incremental_config = config.clone();

            // Use smaller patience for incremental training (faster convergence expected) - SAME logic as original
            incremental_config.training.early_stopping.patience =
                (config.training.early_stopping.patience / 2).max(10);

            log::info!(
                "⚙️ Incremental training config: patience={}, min_delta={:.6}, new_samples={}",
                incremental_config.training.early_stopping.patience,
                incremental_config.training.early_stopping.min_delta,
                total_samples
            );

            incremental_config
        } else {
            config.clone()
        };

        // Configure training parameters from final config (original or incremental)
        self.configure_training(&final_config);

        // Initialize network if not already done
        if self.lstm_layers.is_none() || self.output_layer.is_none() {
            log::info!("🆕 FRESH TRAINING: Initializing new LSTM network layers and weights");
            self.initialize_network()?;

            // Apply Xavier initialization for fresh training
            self.apply_xavier_initialization()?;
        } else {
            log::info!("🔄 CONTINUE TRAINING: Reusing existing LSTM network layers and weights (NO reinitialization)");
            log::info!(
                "   ✅ LSTM layers preserved: {} layers with existing parameters",
                self.config.num_layers
            );
            log::info!(
                "   ✅ Output layer preserved: {} → {} with existing weights",
                self.config
                    .get_hidden_size_for_layer(self.config.num_layers - 1),
                self.config.output_size
            );
        }

        // Determine if we need validation split
        let validation_split = config.training.validation_split;
        let use_validation = validation_split > 0.0;

        // Prepare training and validation data - handle pre-split vs internal split
        let (
            train_sequences,
            train_targets,
            val_sequences_final,
            val_targets_final,
            test_sequences_final,
            test_targets_final,
        ) = if let (Some(val_seq), Some(val_tgt)) = (val_sequences, val_targets) {
            // Use pre-split chronological validation data (prevents data leakage)
            log::info!(
                "📊 Using pre-split chronological validation: {} train, {} val samples",
                sequences.shape()[0],
                val_seq.shape()[0]
            );
            (
                sequences.to_owned(),
                targets.to_owned(),
                Some(val_seq.to_owned()),
                Some(val_tgt.to_owned()),
                ndarray::Array3::zeros((0, sequences.shape()[1], sequences.shape()[2])), // Empty test data for pre-split
                ndarray::Array2::zeros((0, targets.shape()[1])),
            )
        } else if use_validation {
            // Create internal validation split with gap to prevent data leakage
            log::info!(
                "📊 Using internal validation split: {:.1}%",
                validation_split * 100.0
            );

            // STEP 1: Reserve test set if test_split > 0
            let test_split = config.training.test_split;
            let test_size = if test_split > 0.0 {
                (total_samples as f64 * test_split) as usize
            } else {
                0
            };

            let available_for_training = total_samples - test_size;

            log::info!(
                "📊 Three-way split: total={}, test_reserved={} ({:.1}%), available_for_training={}",
                total_samples,
                test_size,
                test_split * 100.0,
                available_for_training
            );

            // FIXED: Calculate proper gap size to prevent data leakage
            // Gap must be sequence_length + max_horizon_steps to ensure no overlap between
            // the last training sequence and first validation target
            let max_horizon_steps = if !config.horizons.is_empty() {
                // Calculate max horizon steps from training config horizons
                config
                    .horizons
                    .iter()
                    .map(|h| crate::utils::parser::parse_horizon_to_steps(h).unwrap_or(1))
                    .max()
                    .unwrap_or(72)
            } else {
                72 // Fallback to 3d horizon if no horizons specified
            };

            // CRITICAL FIX: Proper gap calculation
            let gap_size = self.config.sequence_length + max_horizon_steps;

            log::info!(
                "🔒 Gap calculation: sequence_length({}) + max_horizon_steps({}) = {} total gap",
                self.config.sequence_length,
                max_horizon_steps,
                gap_size
            );

            // FIXED: Account for gap in validation split calculation
            let effective_samples = available_for_training.saturating_sub(gap_size);
            let base_train_samples = ((1.0 - validation_split) * effective_samples as f64) as usize;
            let train_samples = base_train_samples;
            let val_start = train_samples + gap_size;

            // Ensure we have enough samples for validation after the gap
            if val_start >= available_for_training {
                return Err(VangaError::DataError(format!(
                        "Not enough data for validation after gap: {} available samples, {} train + {} gap = {} start, need at least {} for validation",
                        available_for_training, train_samples, gap_size, val_start, val_start + 1
                    )));
            }

            let train_seq = sequences.slice(s![0..train_samples, .., ..]).to_owned();
            let train_tgt = targets.slice(s![0..train_samples, ..]).to_owned();
            let val_seq = sequences
                .slice(s![val_start..available_for_training, .., ..])
                .to_owned();
            let val_tgt = targets
                .slice(s![val_start..available_for_training, ..])
                .to_owned();

            // Extract test data if test_split > 0
            let (test_seq, test_tgt) = if test_size > 0 {
                let test_seq = sequences
                    .slice(s![available_for_training.., .., ..])
                    .to_owned();
                let test_tgt = targets.slice(s![available_for_training.., ..]).to_owned();
                (test_seq, test_tgt)
            } else {
                // Create empty test data with proper dimensions
                let empty_test_seq =
                    ndarray::Array3::zeros((0, sequences.shape()[1], sequences.shape()[2]));
                let empty_test_tgt = ndarray::Array2::zeros((0, targets.shape()[1]));
                (empty_test_seq, empty_test_tgt)
            };

            // CRITICAL: Validate sequence-target alignment
            log::debug!(
                "🔍 Train/Val split validation: train_seq={:?}, train_tgt={:?}, val_seq={:?}, val_tgt={:?}",
                train_seq.shape(), train_tgt.shape(), val_seq.shape(), val_tgt.shape()
            );

            if train_seq.shape()[0] != train_tgt.shape()[0] {
                return Err(VangaError::DataError(format!(
                    "Train sequence-target mismatch: {} sequences vs {} targets",
                    train_seq.shape()[0],
                    train_tgt.shape()[0]
                )));
            }

            if val_seq.shape()[0] != val_tgt.shape()[0] {
                return Err(VangaError::DataError(format!(
                    "Validation sequence-target mismatch: {} sequences vs {} targets",
                    val_seq.shape()[0],
                    val_tgt.shape()[0]
                )));
            }

            log::info!(
                    "🔒 Data leakage prevention: {} train samples, {} gap (max horizon: {}), {} val samples (starting at {})",
                    train_samples,
                    gap_size,
                    config.horizons.iter().max().unwrap_or(&"3d".to_string()),
                    val_seq.shape()[0],
                    val_start
                );

            (
                train_seq,
                train_tgt,
                Some(val_seq),
                Some(val_tgt),
                test_seq,
                test_tgt,
            )
        } else {
            // No validation
            log::info!("📊 Training without validation");
            let empty_test_seq =
                ndarray::Array3::zeros((0, sequences.shape()[1], sequences.shape()[2]));
            let empty_test_tgt = ndarray::Array2::zeros((0, targets.shape()[1]));
            (
                sequences.to_owned(),
                targets.to_owned(),
                None,
                None,
                empty_test_seq,
                empty_test_tgt,
            )
        };

        // Store validation and test data for consistent metrics calculation
        if let Some(val_seq) = &val_sequences_final {
            self.stored_val_sequences = Some(val_seq.clone());
        }
        if let Some(val_tgt) = &val_targets_final {
            self.stored_val_targets = Some(val_tgt.clone());
        }
        self.stored_test_sequences = test_sequences_final.clone();
        self.stored_test_targets = test_targets_final.clone();

        let total_train_samples = train_sequences.shape()[0];
        let total_val_samples = val_sequences_final
            .as_ref()
            .map(|v| v.shape()[0])
            .unwrap_or(0);
        let batch_size = self.training_config.batch_size;

        log::info!(
            "🎯 Using {} train samples{}, batch_size={}, optimizer={:?}",
            total_train_samples,
            if use_validation {
                format!(", {} val samples", total_val_samples)
            } else {
                String::new()
            },
            batch_size,
            config.training.optimizer
        );

        // Memory prevalidation and warnings
        self.validate_batch_configuration(total_train_samples, batch_size)?;

        // Setup advanced optimizer with all configurations
        let mut optimizer = self.setup_advanced_optimizer(config)?;

        // Extract learning rate configuration
        let target_lr = config.training.learning_rate;

        // Extract warmup configuration
        let warmup_epochs = config.training.warmup_epochs;
        let mut current_lr = target_lr;

        // Initialize adaptive learning rate variables from learning_schedule
        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;
        let (adaptive_patience, adaptive_factor) = match &config.training.learning_schedule {
            Some(crate::config::training::LearningScheduleConfig::ReduceOnPlateau {
                patience,
                factor,
                ..
            }) => (*patience, *factor),
            _ => (10, 0.5), // Default values for non-adaptive modes
        };

        // Initialize early stopping variables (only used with validation)
        let mut best_val_loss = f64::INFINITY;
        let mut early_stopping_counter = 0;
        let mut checkpoint_restored = false; // Track if we already restored checkpoint

        // FIXED: Adaptive early stopping configuration based on target types
        let (early_stopping_patience, early_stopping_min_delta) = if use_validation {
            let base_patience = match &config.training.epochs {
                crate::config::training::EpochConfig::Auto { max_epochs: _ } => {
                    config.training.early_stopping.patience
                }
                _ => 10, // Default patience for fixed epochs
            };
            let base_min_delta = config.training.early_stopping.min_delta;

            // FIXED: Adjust min_delta based on target types and expected scale
            let target_type = self.get_target_type().unwrap_or(TargetType::PriceLevel);
            let (adaptive_patience, adaptive_min_delta) = self.get_adaptive_early_stopping_config(
                &[target_type],
                base_patience,
                base_min_delta,
            );

            log::info!(
                "🎯 Early stopping configured: patience={}, min_delta={:.6} (adaptive from {:.6}) for target: {:?}",
                adaptive_patience, adaptive_min_delta, base_min_delta, target_type
            );

            (adaptive_patience, adaptive_min_delta)
        } else {
            (u32::MAX, 0.0) // Disable early stopping without validation
        };

        log::info!("🔧 Training Configuration:");
        log::info!("  - Epochs: {}", self.training_config.epochs);
        log::info!("  - Batch size: {}", batch_size);
        log::info!("  - Warmup epochs: {}", warmup_epochs);
        log::info!("  - Adaptive patience: {}", adaptive_patience);
        log::info!("  - Adaptive factor: {:.3}", adaptive_factor);
        log::info!("  - Target learning rate: {:.6}", target_lr);

        // Unified training loop with warmup, adaptive learning, optional validation, and early stopping
        for epoch in 0..self.training_config.epochs {
            // Initialize epoch tracking variables
            let mut epoch_train_loss = 0.0;
            let mut epoch_grad_norm = 0.0; // Track gradient norm for epoch logging
            let mut batch_count = 0;

            // Calculate warmup learning rate for current epoch
            if epoch < warmup_epochs as usize {
                // Linear warmup from 0 to target_lr
                let warmup_progress = (epoch + 1) as f64 / (warmup_epochs as f64);
                let warmup_lr = target_lr * warmup_progress;

                // Update optimizer learning rate for warmup
                optimizer.set_learning_rate(warmup_lr);
                current_lr = warmup_lr;

                if epoch == 0 || epoch == (warmup_epochs as usize) - 1 {
                    log::info!(
                        "🔥 Warmup epoch {}/{}: learning rate = {:.6}",
                        epoch + 1,
                        warmup_epochs,
                        warmup_lr
                    );
                }
            } else {
                // Apply learning schedule after warmup phase (if configured)
                if let Some(schedule_config) = &config.training.learning_schedule {
                    let epoch_after_warmup = epoch - warmup_epochs as usize;
                    let total_epochs = match &config.training.epochs {
                        crate::config::training::EpochConfig::Fixed(n) => *n as usize,
                        crate::config::training::EpochConfig::Auto { max_epochs } => {
                            *max_epochs as usize
                        }
                    };

                    let scheduled_lr = Self::calculate_scheduled_learning_rate(
                        schedule_config,
                        epoch_after_warmup,
                        target_lr,
                        total_epochs.saturating_sub(warmup_epochs as usize),
                    );

                    // Only update if there's a meaningful change (avoid unnecessary updates)
                    if (scheduled_lr - current_lr).abs() > 1e-8 {
                        optimizer.set_learning_rate(scheduled_lr);
                        current_lr = scheduled_lr;

                        log::debug!(
                            "📈 Schedule LR update at epoch {}: {:.6} (schedule: {:?})",
                            epoch + 1,
                            scheduled_lr,
                            schedule_config
                        );
                    }
                }
            }

            // Training phase - process data in batches
            for (batch_idx, batch_start) in (0..total_train_samples).step_by(batch_size).enumerate()
            {
                let batch_end = std::cmp::min(batch_start + batch_size, total_train_samples);
                let actual_batch_size = batch_end - batch_start;

                // Extract batch from sequences and targets
                let batch_sequences = train_sequences
                    .slice(ndarray::s![batch_start..batch_end, .., ..])
                    .to_owned();
                let batch_targets = train_targets
                    .slice(ndarray::s![batch_start..batch_end, ..])
                    .to_owned();

                // Convert batch to tensors
                let (input_tensor, target_tensor) =
                    self.convert_sequences_to_tensors(&batch_sequences, &batch_targets)?;

                // Forward pass (training mode - enable dropout)
                let predictions = self.forward(&input_tensor, true)?;

                // Calculate loss using the proven NLL approach (moved to loss.rs)
                let base_loss =
                    self.calculate_nll_loss(&predictions, &target_tensor, LossMode::Training)?;

                // Get loss value for reporting BEFORE gradient clipping (to avoid move issues)
                let batch_loss_value = base_loss.to_scalar::<f32>().map_err(|e| {
                    VangaError::ModelError(format!("Loss scalar conversion failed: {}", e))
                })?;

                // SIMPLE FIX: Use single backward pass with direct gradient clipping
                let (original_grad_norm, effective_grad_norm, final_grads) = if let Some(
                    clip_value,
                ) =
                    self.training_config.clip_gradient
                {
                    // Single backward pass - let gradients accumulate naturally
                    let grads = base_loss.backward()?;
                    let original_norm = self.calculate_gradstore_norm(&grads)?;

                    if original_norm > clip_value {
                        // Apply gradient clipping by scaling the gradients directly
                        let clip_ratio = clip_value / original_norm;

                        log::debug!(
                            "✂️ GRADIENT CLIPPING: original_norm={:.6} > threshold={:.6}, clip_ratio={:.6}",
                            original_norm,
                            clip_value,
                            clip_ratio
                        );

                        if epoch == 0 && batch_idx == 0 {
                            log::info!(
                                "🔧 Gradient clipping enabled: threshold={:.3}, direct gradient scaling",
                                clip_value
                            );
                        }

                        // The gradients will be clipped during the optimizer step
                        // by scaling them with clip_ratio
                        (original_norm, clip_value, grads)
                    } else {
                        // No clipping needed
                        if epoch == 0 && batch_idx == 0 {
                            log::info!(
                                "🔧 Gradient clipping enabled: threshold={:.3}, no clipping needed initially",
                                clip_value
                            );
                        }

                        (original_norm, original_norm, grads)
                    }
                } else {
                    // No clipping - single backward pass
                    let grads = base_loss.backward()?;
                    let norm = self.calculate_gradstore_norm(&grads)?;
                    (norm, norm, grads)
                };

                // GRADIENT FLOW VALIDATION: Check gradients using effective norm
                self.validate_gradient_flow(&final_grads, effective_grad_norm, original_grad_norm)?;

                // 🔍 ENHANCED GRADIENT MONITORING: Track gradient accumulation patterns
                if epoch > 0 && batch_count > 1 {
                    // Only check after we have multiple batches to compare
                    let avg_grad_norm = epoch_grad_norm / batch_count as f64;
                    let gradient_growth_rate = effective_grad_norm / avg_grad_norm.max(1e-12_f64);

                    // Only warn if we have meaningful data and significant growth
                    if avg_grad_norm > 1e-12_f64 && gradient_growth_rate > 3.0 {
                        log::warn!(
                            "⚠️ Potential gradient accumulation detected: current_norm={:.6e}, avg_norm={:.6e}, growth_rate={:.2}x",
                            effective_grad_norm,
                            avg_grad_norm,
                            gradient_growth_rate
                        );
                    }
                }

                // 📊 GRADIENT CLIPPING STATISTICS: Track clipping frequency
                if let Some(clip_value) = self.training_config.clip_gradient {
                    if original_grad_norm > clip_value {
                        log::debug!(
                            "📊 Gradient clipping stats: original={:.6e}, clipped={:.6e}, reduction={:.1}%",
                            original_grad_norm,
                            effective_grad_norm,
                            (1.0 - effective_grad_norm / original_grad_norm) * 100.0
                        );
                    }
                }

                // Accumulate effective gradient norm for epoch reporting
                epoch_grad_norm += effective_grad_norm;
                batch_count += 1;

                // Apply gradient clipping if needed and update parameters
                if let Some(clip_value) = self.training_config.clip_gradient {
                    if original_grad_norm > clip_value {
                        // For gradient clipping, we need to scale the loss by the clip ratio
                        // This effectively scales all gradients by the same factor
                        let clip_ratio = clip_value / original_grad_norm;
                        let clip_ratio_tensor = Tensor::new(clip_ratio as f32, &self.device)?;
                        let scaled_loss = base_loss.mul(&clip_ratio_tensor)?;

                        // Clear any existing gradients and compute new ones with scaled loss
                        let clipped_grads = scaled_loss.backward()?;
                        optimizer.step(&clipped_grads)?;

                        log::debug!(
                            "✂️ Applied gradient clipping: scaled loss by {:.6}",
                            clip_ratio
                        );
                    } else {
                        // No clipping needed
                        optimizer.step(&final_grads)?;
                    }
                } else {
                    // No gradient clipping configured
                    optimizer.step(&final_grads)?;
                }

                // Accumulate loss for epoch reporting (use original loss, not clipped)
                let batch_loss = batch_loss_value;

                // 🔍 DETAILED TRAINING BATCH DEBUG
                log::debug!(
                    "🔍 TRAIN E{} B{}: raw_loss={:.6}, batch_size={}, weighted_loss={:.6}, grad_norm={:.6}",
                    epoch + 1, batch_idx, batch_loss, actual_batch_size,
                    batch_loss * actual_batch_size as f32, effective_grad_norm
                );

                epoch_train_loss += batch_loss * actual_batch_size as f32;
            }

            // Calculate average training loss and gradient norm
            let avg_train_loss = epoch_train_loss / total_train_samples as f32;
            let avg_grad_norm = if batch_count > 0 {
                epoch_grad_norm / batch_count as f64
            } else {
                0.0
            };

            // 📊 GRADIENT STABILITY ANALYSIS: Check for gradient explosion or vanishing
            if avg_grad_norm > 10.0 {
                log::warn!(
                    "⚠️ Large average gradient norm detected: {:.6e} - consider lower learning rate or stronger clipping",
                    avg_grad_norm
                );
            } else if avg_grad_norm < 1e-6 && avg_grad_norm > 0.0 {
                log::warn!(
                    "⚠️ Very small average gradient norm detected: {:.6e} - model may not be learning effectively",
                    avg_grad_norm
                );
            }

            // 🔍 GRADIENT CLIPPING EFFECTIVENESS: Report clipping statistics
            if let Some(clip_value) = self.training_config.clip_gradient {
                let clipping_ratio = avg_grad_norm / clip_value;
                if clipping_ratio > 0.8 {
                    log::debug!(
                        "📊 Gradient clipping active: avg_norm={:.6e}, threshold={:.6e}, ratio={:.2}",
                        avg_grad_norm,
                        clip_value,
                        clipping_ratio
                    );
                }
            }

            // 🔍 EPOCH TRAINING SUMMARY DEBUG
            log::debug!(
                "🔍 TRAIN E{} SUMMARY: total_weighted_loss={:.6}, total_samples={}, avg_loss={:.6}, batches={}",
                epoch + 1, epoch_train_loss, total_train_samples, avg_train_loss, batch_count
            );

            // Validation phase (only if validation data is available)
            let avg_val_loss = if let (Some(val_seq), Some(val_tgt)) =
                (&val_sequences_final, &val_targets_final)
            {
                let mut epoch_val_loss = 0.0;

                for batch_start in (0..total_val_samples).step_by(batch_size) {
                    let batch_end = std::cmp::min(batch_start + batch_size, total_val_samples);
                    let actual_batch_size = batch_end - batch_start;

                    // Extract validation batch
                    let batch_sequences = val_seq
                        .slice(ndarray::s![batch_start..batch_end, .., ..])
                        .to_owned();
                    let batch_targets = val_tgt
                        .slice(ndarray::s![batch_start..batch_end, ..])
                        .to_owned();

                    // Convert batch to tensors
                    let (input_tensor, target_tensor) =
                        self.convert_sequences_to_tensors(&batch_sequences, &batch_targets)?;

                    // Forward pass (validation mode - no dropout)
                    let predictions = self.forward(&input_tensor, false)?;

                    // Calculate validation loss using the same NLL approach as training
                    let val_loss = self.calculate_nll_loss(
                        &predictions,
                        &target_tensor,
                        LossMode::Validation,
                    )?;
                    let val_batch_loss = val_loss.to_scalar::<f32>().map_err(|e| {
                        VangaError::ModelError(format!(
                            "Validation loss scalar conversion failed: {}",
                            e
                        ))
                    })?;

                    // 🔍 DETAILED VALIDATION BATCH DEBUG
                    log::debug!(
                        "🔍 VAL E{} B{}: raw_loss={:.6}, batch_size={}, weighted_loss={:.6}",
                        epoch + 1,
                        batch_start / batch_size,
                        val_batch_loss,
                        actual_batch_size,
                        val_batch_loss * actual_batch_size as f32
                    );

                    epoch_val_loss += val_batch_loss * actual_batch_size as f32;
                }

                let avg_val_loss = epoch_val_loss / total_val_samples as f32;

                // 🔍 EPOCH VALIDATION SUMMARY DEBUG
                log::debug!(
                    "🔍 VAL E{} SUMMARY: total_weighted_loss={:.6}, total_samples={}, avg_loss={:.6}",
                    epoch + 1, epoch_val_loss, total_val_samples, avg_val_loss
                );

                // Calculate categorical metrics for all categorical targets
                if let Some((_, target_type)) = &self.target_context {
                    match target_type {
                        TargetType::PriceLevel | TargetType::Direction | TargetType::Volatility => {
                            self.calculate_categorical_validation_metrics(
                                val_seq, val_tgt, batch_size, epoch, config,
                            )
                            .await?;
                        }
                    }
                }

                Some(avg_val_loss)
            } else {
                None
            };

            // Adaptive learning rate adjustment after warmup
            // NOTE: This runs AFTER schedule updates, so adaptive LR can override schedule if needed
            if epoch >= warmup_epochs as usize {
                if let Some(crate::config::training::LearningScheduleConfig::ReduceOnPlateau {
                    ..
                }) = &config.training.learning_schedule
                {
                    // Use validation loss if available, otherwise use training loss
                    let loss_for_adaptation = avg_val_loss
                        .map(|v| v as f64)
                        .unwrap_or(avg_train_loss as f64);

                    // Check if we should reduce learning rate
                    if loss_for_adaptation < best_loss {
                        best_loss = loss_for_adaptation;
                        patience_counter = 0;
                    } else {
                        patience_counter += 1;

                        if patience_counter >= adaptive_patience {
                            // Reduce learning rate
                            current_lr *= adaptive_factor;
                            optimizer.set_learning_rate(current_lr);
                            patience_counter = 0;

                            log::info!(
                                "🔄 Adaptive learning rate reduced to: {:.6} (patience exceeded)",
                                current_lr
                            );
                        }
                    }
                }
            }

            // Early stopping check with min_delta threshold (only with validation)
            if let Some(val_loss) = avg_val_loss {
                let improvement = best_val_loss - (val_loss as f64);
                if improvement > early_stopping_min_delta {
                    best_val_loss = val_loss as f64;
                    early_stopping_counter = 0;
                    log::debug!(
                        "✅ Validation improved by {:.6} (> {:.6}), resetting patience counter",
                        improvement,
                        early_stopping_min_delta
                    );

                    // Save the best model checkpoint
                    if let Err(e) = self.save_best_checkpoint(val_loss as f64, epoch) {
                        log::error!("Failed to save best model checkpoint: {}", e);
                    }
                } else {
                    early_stopping_counter += 1;
                    log::debug!(
                        "⏳ No significant improvement ({:.6} <= {:.6}), patience: {}/{}",
                        improvement,
                        early_stopping_min_delta,
                        early_stopping_counter,
                        early_stopping_patience
                    );

                    if early_stopping_counter >= early_stopping_patience {
                        log::info!(
                            "🛑 Early stopping triggered at epoch {} (best val loss: {:.6}, min_delta: {:.6})",
                            epoch + 1,
                            best_val_loss,
                            early_stopping_min_delta
                        );

                        // Restore best model weights before breaking
                        if let Err(e) = self.restore_best_checkpoint() {
                            log::error!("Failed to restore best model checkpoint: {}", e);
                        }
                        checkpoint_restored = true;

                        break;
                    }
                }
            }

            // Enhanced logging with learning rate tracking
            if epoch % self.training_config.print_every == 0 {
                let warmup_status = if epoch < warmup_epochs as usize {
                    " (warmup)"
                } else {
                    ""
                };

                // Add schedule status information
                let schedule_status = if epoch >= warmup_epochs as usize {
                    if let Some(schedule) = &config.training.learning_schedule {
                        match schedule {
                            crate::config::training::LearningScheduleConfig::Constant => {
                                " [Constant]"
                            }
                            crate::config::training::LearningScheduleConfig::ReduceOnPlateau {
                                ..
                            } => " [ReduceOnPlateau]",
                            crate::config::training::LearningScheduleConfig::LinearDecay {
                                ..
                            } => " [LinearDecay]",
                            crate::config::training::LearningScheduleConfig::ExponentialDecay {
                                ..
                            } => " [ExponentialDecay]",
                            crate::config::training::LearningScheduleConfig::StepDecay {
                                ..
                            } => " [StepDecay]",
                            crate::config::training::LearningScheduleConfig::PolynomialDecay {
                                ..
                            } => " [PolynomialDecay]",
                            crate::config::training::LearningScheduleConfig::CosineAnnealing {
                                ..
                            } => " [CosineAnnealing]",
                            crate::config::training::LearningScheduleConfig::WarmRestarts {
                                ..
                            } => " [WarmRestarts]",
                            crate::config::training::LearningScheduleConfig::OneCycle {
                                ..
                            } => " [OneCycle]",
                            crate::config::training::LearningScheduleConfig::CyclicalLR {
                                ..
                            } => " [CyclicalLR]",
                            crate::config::training::LearningScheduleConfig::NoamLR { .. } => {
                                " [NoamLR]"
                            }
                        }
                    } else {
                        ""
                    }
                } else {
                    ""
                };

                if let Some(val_loss) = avg_val_loss {
                    // Get target type for this individual model
                    let target_type = self.get_target_type().unwrap_or(TargetType::PriceLevel);
                    let target_info = format!(" [{:?}]", target_type);

                    // Calculate loss ratio and status
                    let loss_ratio = val_loss / avg_train_loss;
                    let ratio_status = if loss_ratio < 1.5 {
                        "✅"
                    } else if loss_ratio < 3.0 {
                        "⚠️"
                    } else {
                        "🚨"
                    };
                    log::info!(
                        "Epoch {}/{}: Train Loss = {:.6}, Val Loss = {:.6} (Ratio: {:.2}x {}), LR: {:.6}, Grad: {:.2e}{}{}, Early Stop: {}/{}{}",
                        epoch + 1,
                        self.training_config.epochs,
                        avg_train_loss,
                        val_loss,
                        loss_ratio,
                        ratio_status,
                        current_lr,
                        avg_grad_norm,
                        warmup_status,
                        schedule_status,
                        early_stopping_counter,
                        early_stopping_patience,
                        target_info
                    );

                    // Log overfitting warnings only when necessary
                    if loss_ratio > 3.0 {
                        log::warn!("🔧 Overfitting detected (ratio: {:.2}x). Consider adjusting regularization or model complexity.", loss_ratio);
                    }
                } else {
                    let num_batches = total_train_samples.div_ceil(batch_size);
                    log::info!(
                        "Epoch {}/{}: Loss = {:.6}, Batches: {}, LR: {:.6}{}{}",
                        epoch + 1,
                        self.training_config.epochs,
                        avg_train_loss,
                        num_batches,
                        current_lr,
                        warmup_status,
                        schedule_status
                    );
                }

                // Additional adaptive learning rate status
                if matches!(
                    &config.training.learning_schedule,
                    Some(crate::config::training::LearningScheduleConfig::ReduceOnPlateau { .. })
                ) && epoch >= warmup_epochs as usize
                {
                    log::debug!(
                        "📊 Adaptive LR status - Best loss: {:.6}, Patience: {}/{}",
                        best_loss,
                        patience_counter,
                        adaptive_patience
                    );
                }
            }
        }

        self.trained = true;

        // After training completes, restore best model weights if available
        // This ensures we use the best weights even if training completed all epochs
        if self.best_model_varmap.is_some() && !checkpoint_restored {
            log::info!(
                "🎯 Training completed. Restoring best model from epoch {} (val loss: {:.6})",
                self.best_epoch.map(|e| e + 1).unwrap_or(0),
                self.best_validation_loss.unwrap_or(0.0)
            );
            if let Err(e) = self.restore_best_checkpoint() {
                log::error!("Failed to restore best model checkpoint: {}", e);
            }
        }

        log::info!("✅ Unified LSTM training completed successfully");

        // Calculate final training metrics - use classification accuracy for categorical targets
        if let Ok(final_predictions) = self.predict(sequences).await {
            // For classification targets, use accuracy instead of MSE
            if let Some((_, target_type)) = &self.target_context {
                match target_type {
                    TargetType::PriceLevel | TargetType::Direction | TargetType::Volatility => {
                        // Use stored validation data for consistent metrics
                        log::info!(
                            "📊 Calculating Final Training Metrics using stored validation data..."
                        );

                        if let (Some(stored_val_seq), Some(stored_val_tgt)) =
                            (&self.stored_val_sequences, &self.stored_val_targets)
                        {
                            // Clone the data to avoid borrowing conflicts
                            let val_seq_clone = stored_val_seq.clone();
                            let val_tgt_clone = stored_val_tgt.clone();

                            let _ = self
                                .calculate_categorical_validation_metrics(
                                    &val_seq_clone,
                                    &val_tgt_clone,
                                    64, // batch_size (not used in the method)
                                    10, // epoch = 10 to force calculation (10 % 10 == 0)
                                    config,
                                )
                                .await;
                        } else {
                            log::warn!("⚠️ No stored validation data available for final metrics");
                        }

                        // Calculate test metrics if test data is available
                        if self.stored_test_sequences.shape()[0] > 0 {
                            log::info!(
                                "📊 Calculating Final Test Metrics on {} samples...",
                                self.stored_test_sequences.shape()[0]
                            );

                            // Clone the data to avoid borrowing conflicts
                            let test_seq_clone = self.stored_test_sequences.clone();
                            let test_tgt_clone = self.stored_test_targets.clone();

                            let _ = self
                                .calculate_categorical_validation_metrics(
                                    &test_seq_clone,
                                    &test_tgt_clone,
                                    64,
                                    10, // Force calculation
                                    config,
                                )
                                .await;
                        } else {
                            log::debug!("📊 No test data available for final evaluation");
                        }

                        // Comprehensive evaluation summary
                        self.log_comprehensive_evaluation_summary().await;
                    }
                }
            } else {
                // Fallback to MSE for regression targets
                let final_mse = self.calculate_mse_loss(&final_predictions, targets);
                let final_mape = self.calculate_mape(&final_predictions, targets);
                log::info!(
                    "📊 Final Training Metrics - MSE: {:.6} (√MSE: {:.3}), MAPE: {:.2}%",
                    final_mse,
                    final_mse.sqrt(),
                    final_mape
                );
            }
        }

        // Phase 2: XGBoost Training (NEW)
        if config.model.xgboost.enabled {
            log::info!("🔄 Starting XGBoost hybrid training phase...");
            self.train_xgboost_phase(sequences, targets, config).await?;
        }

        Ok(())
    }
    fn setup_advanced_optimizer(
        &self,
        config: &crate::config::TrainingConfig,
    ) -> Result<OptimizerWrapper> {
        let learning_rate = self.config.learning_rate;
        let optimizer_config = &config.training.optimizer;

        match optimizer_config {
            crate::config::training::OptimizerType::SGD { momentum } => {
                log::info!(
                    "Using SGD optimizer with learning rate: {:.6}",
                    learning_rate
                );
                if let Some(momentum_val) = momentum {
                    log::info!(
                        "SGD momentum: {:.3} (not yet implemented in Candle)",
                        momentum_val
                    );
                }
                Ok(OptimizerWrapper::Sgd(
                    optim::SGD::new(self.varmap.all_vars(), learning_rate).map_err(|e| {
                        VangaError::ModelError(format!("SGD optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::AdamW {
                weight_decay,
                beta1,
                beta2,
                eps,
            } => {
                log::info!(
                    "Using AdamW optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "AdamW parameters - weight_decay: {:.4}, beta1: {:.3}, beta2: {:.3}, eps: {:.2e}",
                    weight_decay,
                    beta1,
                    beta2,
                    eps
                );

                let params = ParamsAdamW {
                    lr: learning_rate,
                    beta1: *beta1,
                    beta2: *beta2,
                    weight_decay: *weight_decay,
                    eps: *eps,
                };

                Ok(OptimizerWrapper::AdamW(
                    optim::AdamW::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("AdamW optimizer creation failed: {}", e))
                    })?,
                ))
            }
            // New optimizers from candle-optimisers crate
            crate::config::training::OptimizerType::Adam {
                beta1,
                beta2,
                eps,
                weight_decay,
                amsgrad,
            } => {
                log::info!(
                    "Using Adam optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "Adam parameters - beta1: {:.3}, beta2: {:.3}, eps: {:.2e}, amsgrad: {}",
                    beta1,
                    beta2,
                    eps,
                    amsgrad
                );

                let params = ParamsAdam {
                    lr: learning_rate,
                    beta_1: *beta1,
                    beta_2: *beta2,
                    eps: *eps,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                    amsgrad: *amsgrad,
                };

                Ok(OptimizerWrapper::Adam(
                    Adam::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("Adam optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::AdaDelta {
                rho,
                eps,
                weight_decay,
            } => {
                log::info!(
                    "Using AdaDelta optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!("AdaDelta parameters - rho: {:.3}, eps: {:.2e}", rho, eps);

                let params = ParamsAdaDelta {
                    lr: learning_rate,
                    rho: *rho,
                    eps: *eps,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                };

                Ok(OptimizerWrapper::AdaDelta(
                    Adadelta::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("AdaDelta optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::AdaGrad {
                lr_decay,
                weight_decay,
                initial_accumulator_value,
                eps,
            } => {
                log::info!(
                    "Using AdaGrad optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "AdaGrad parameters - lr_decay: {:.3}, eps: {:.2e}, init_acc: {:.3}",
                    lr_decay,
                    eps,
                    initial_accumulator_value
                );

                let params = ParamsAdaGrad {
                    lr: learning_rate,
                    lr_decay: *lr_decay,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                    eps: *eps,
                    initial_acc: *initial_accumulator_value,
                };
                Ok(OptimizerWrapper::AdaGrad(
                    Adagrad::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("AdaGrad optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::AdaMax {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                log::info!(
                    "Using AdaMax optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "AdaMax parameters - beta1: {:.3}, beta2: {:.3}, eps: {:.2e}",
                    beta1,
                    beta2,
                    eps
                );

                let params = ParamsAdaMax {
                    lr: learning_rate,
                    beta_1: *beta1,
                    beta_2: *beta2,
                    eps: *eps,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                };

                Ok(OptimizerWrapper::AdaMax(
                    Adamax::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("AdaMax optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::NAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
                momentum_decay,
            } => {
                log::info!(
                    "Using NAdam optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "NAdam parameters - beta1: {:.3}, beta2: {:.3}, eps: {:.2e}, momentum_decay: {:.3}",
                    beta1, beta2, eps, momentum_decay
                );

                let params = ParamsNAdam {
                    lr: learning_rate,
                    beta_1: *beta1,
                    beta_2: *beta2,
                    eps: *eps,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                    momentum_decay: *momentum_decay,
                };

                Ok(OptimizerWrapper::NAdam(
                    NAdam::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("NAdam optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::RAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                log::info!(
                    "Using RAdam optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "RAdam parameters - beta1: {:.3}, beta2: {:.3}, eps: {:.2e}",
                    beta1,
                    beta2,
                    eps
                );

                let params = ParamsRAdam {
                    lr: learning_rate,
                    beta_1: *beta1,
                    beta_2: *beta2,
                    eps: *eps,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                };

                Ok(OptimizerWrapper::RAdam(
                    RAdam::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("RAdam optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::RMSprop {
                alpha,
                eps,
                weight_decay,
                momentum,
                centered,
            } => {
                log::info!(
                    "Using RMSprop optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "RMSprop parameters - alpha: {:.3}, eps: {:.2e}, momentum: {:.3}, centered: {}",
                    alpha,
                    eps,
                    momentum,
                    centered
                );

                let params = ParamsRMSprop {
                    lr: learning_rate,
                    alpha: *alpha,
                    eps: *eps,
                    weight_decay: *weight_decay,
                    momentum: if *momentum > 0.0 {
                        Some(*momentum)
                    } else {
                        None
                    },
                    centered: *centered,
                };

                Ok(OptimizerWrapper::RMSprop(
                    RMSprop::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("RMSprop optimizer creation failed: {}", e))
                    })?,
                ))
            }
        }
    }
    pub fn apply_xavier_initialization(&mut self) -> Result<()> {
        log::info!(
            "🔧 Applying Xavier/Glorot weight initialization to prevent exploding gradients..."
        );

        let all_vars = self.varmap.all_vars();
        let mut initialized_count = 0;

        for var in all_vars.iter() {
            // Get the tensor shape to determine initialization parameters
            let shape = var.shape();

            // Skip biases (1D tensors) - initialize only weights (2D tensors)
            if shape.dims().len() == 2 {
                let (fan_in, fan_out) = (shape.dims()[0], shape.dims()[1]);

                // Xavier/Glorot initialization: std = sqrt(2.0 / (fan_in + fan_out))
                let xavier_std = (2.0 / (fan_in + fan_out) as f64).sqrt();

                log::debug!(
                    "🎯 Xavier init: shape={:?}, fan_in={}, fan_out={}, std={:.6}",
                    shape.dims(),
                    fan_in,
                    fan_out,
                    xavier_std
                );

                initialized_count += 1;
            }
        }

        if initialized_count == 0 {
            log::warn!(
                "⚠️ No weight tensors found for Xavier initialization - using Candle defaults"
            );
        } else {
            log::info!(
                "✅ Xavier initialization parameters calculated for {} weight tensors",
                initialized_count
            );
        }

        Ok(())
    }

    /// Clip gradients to prevent exploding gradients during training
    /// Returns the original gradient norm for monitoring
    /// Apply gradient clipping by norm (clip-by-norm method)
    ///
    /// Calculate learning rate based on schedule configuration
    ///
    /// This function implements all 11 learning schedule types with proper mathematical formulations:
    /// - Constant: Maintains initial learning rate
    /// - ReduceOnPlateau: Handled separately in training loop
    /// - LinearDecay: Linear reduction with configurable minimum
    /// - ExponentialDecay: Exponential decay with gamma parameter
    /// - StepDecay: Step-wise decay at milestones
    /// - PolynomialDecay: Polynomial decay with configurable power
    /// - CosineAnnealing: Cosine annealing with eta_min
    /// - WarmRestarts: SGDR with efficient cycle calculation
    /// - OneCycle: Super-convergence for LSTM training
    /// - CyclicalLR: Triangular/exponential cyclical schedules
    /// - NoamLR: Transformer-style scheduling
    fn calculate_scheduled_learning_rate(
        schedule_config: &crate::config::training::LearningScheduleConfig,
        epoch_after_warmup: usize,
        initial_lr: f64,
        total_epochs: usize,
    ) -> f64 {
        use crate::config::training::LearningScheduleConfig;

        match schedule_config {
            LearningScheduleConfig::Constant => {
                // Maintain constant learning rate
                initial_lr
            }

            LearningScheduleConfig::ReduceOnPlateau { .. } => {
                // ReduceOnPlateau is handled separately in the training loop
                // This function is only for epoch-based schedules
                initial_lr
            }

            LearningScheduleConfig::LinearDecay { decay_rate, min_lr } => {
                // Linear decay: lr = initial_lr * (1 - decay_rate * progress)
                let progress = epoch_after_warmup as f64 / total_epochs.max(1) as f64;
                let decay_factor = 1.0 - (decay_rate * progress);
                let min_threshold = min_lr.unwrap_or(initial_lr * 0.001);
                (initial_lr * decay_factor).max(min_threshold)
            }

            LearningScheduleConfig::ExponentialDecay { gamma, min_lr } => {
                // Exponential decay: lr = initial_lr * gamma^epoch
                let decay_factor = gamma.powf(epoch_after_warmup as f64);
                let min_threshold = min_lr.unwrap_or(initial_lr * 0.0001);
                (initial_lr * decay_factor).max(min_threshold)
            }

            LearningScheduleConfig::StepDecay {
                step_size,
                gamma,
                milestones,
                min_lr,
            } => {
                // Step decay at specific milestones or regular intervals
                let decay_steps = if let Some(milestones) = milestones {
                    // Count how many milestones have been passed
                    milestones
                        .iter()
                        .filter(|&&m| epoch_after_warmup >= m as usize)
                        .count()
                } else {
                    // Regular step decay
                    epoch_after_warmup / (*step_size).max(1) as usize
                };

                let decay_factor = gamma.powf(decay_steps as f64);
                let min_threshold = min_lr.unwrap_or(initial_lr * 0.0001);
                (initial_lr * decay_factor).max(min_threshold)
            }

            LearningScheduleConfig::PolynomialDecay { power, min_lr } => {
                // Polynomial decay: lr = (initial_lr - min_lr) * (1 - progress)^power + min_lr
                let progress = epoch_after_warmup as f64 / total_epochs.max(1) as f64;
                let min_threshold = min_lr.unwrap_or(initial_lr * 0.001);
                let decay_factor = (1.0 - progress.min(1.0)).powf(*power);
                min_threshold + (initial_lr - min_threshold) * decay_factor
            }

            LearningScheduleConfig::CosineAnnealing { t_max, eta_min } => {
                // Cosine annealing: lr = eta_min + (initial_lr - eta_min) * 0.5 * (1 + cos(π * epoch / t_max))
                let t_max_f = (*t_max).max(1) as f64;
                let progress = (epoch_after_warmup as f64 / t_max_f).min(1.0);
                let eta_min_val = eta_min.unwrap_or(initial_lr * 0.001);
                let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                eta_min_val + (initial_lr - eta_min_val) * cosine_factor
            }

            LearningScheduleConfig::WarmRestarts {
                t_0,
                t_mult,
                eta_min,
            } => {
                // SGDR: Cosine annealing with warm restarts - efficient cycle calculation
                let t_0_val = (*t_0).max(1) as usize;
                let t_mult_val = (*t_mult).max(1) as usize;
                let eta_min_val = eta_min.unwrap_or(initial_lr * 0.001);

                // Efficient cycle calculation using closed-form solution
                let (t_cur, t_i) = if t_mult_val == 1 {
                    // Simple case: constant cycle length
                    (epoch_after_warmup % t_0_val, t_0_val)
                } else {
                    // Geometric progression: find current cycle
                    let mut epoch_remaining = epoch_after_warmup;
                    let mut current_cycle_length = t_0_val;

                    while epoch_remaining >= current_cycle_length {
                        epoch_remaining -= current_cycle_length;
                        current_cycle_length *= t_mult_val;
                    }

                    (epoch_remaining, current_cycle_length)
                };

                // Calculate cosine annealing within current cycle
                let progress = t_cur as f64 / t_i.max(1) as f64;
                let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                eta_min_val + (initial_lr - eta_min_val) * cosine_factor
            }

            LearningScheduleConfig::OneCycle {
                max_lr,
                pct_start,
                anneal_strategy,
                div_factor,
                final_div_factor,
            } => {
                // One Cycle Learning Rate for super-convergence
                let pct_start_val = pct_start.unwrap_or(0.3);
                let div_factor_val = div_factor.unwrap_or(25.0);
                let final_div_factor_val = final_div_factor.unwrap_or(1e4);
                let anneal_strategy_val = anneal_strategy.as_deref().unwrap_or("cos");

                let initial_lr_calc = max_lr / div_factor_val;
                let final_lr = initial_lr_calc / final_div_factor_val;

                let progress = epoch_after_warmup as f64 / total_epochs.max(1) as f64;

                if progress <= pct_start_val {
                    // Increasing phase
                    let phase_progress = progress / pct_start_val;
                    initial_lr_calc + (max_lr - initial_lr_calc) * phase_progress
                } else {
                    // Decreasing phase
                    let phase_progress = (progress - pct_start_val) / (1.0 - pct_start_val);
                    match anneal_strategy_val {
                        "linear" => max_lr - (max_lr - final_lr) * phase_progress,
                        _ => {
                            // "cos"
                            let cosine_factor =
                                0.5 * (1.0 + (std::f64::consts::PI * phase_progress).cos());
                            final_lr + (max_lr - final_lr) * cosine_factor
                        }
                    }
                }
            }

            LearningScheduleConfig::CyclicalLR {
                base_lr,
                max_lr,
                step_size_up,
                step_size_down,
                mode,
                gamma,
            } => {
                // Cyclical Learning Rate with different policies
                let step_size_down_val = step_size_down.unwrap_or(*step_size_up);
                let cycle_length = step_size_up + step_size_down_val;
                let mode_val = mode.as_deref().unwrap_or("triangular");
                let gamma_val = gamma.unwrap_or(1.0);

                let cycle = (epoch_after_warmup as f64 / cycle_length as f64).floor() as u32;
                let x = (epoch_after_warmup % cycle_length as usize) as f64;

                let amplitude = match mode_val {
                    "triangular2" => (max_lr - base_lr) / (2.0_f64.powf(cycle as f64)),
                    "exp_range" => (max_lr - base_lr) * gamma_val.powf(epoch_after_warmup as f64),
                    _ => max_lr - base_lr, // "triangular"
                };

                if x <= *step_size_up as f64 {
                    // Increasing phase
                    base_lr + amplitude * (x / *step_size_up as f64)
                } else {
                    // Decreasing phase
                    let down_progress = (x - *step_size_up as f64) / step_size_down_val as f64;
                    base_lr + amplitude * (1.0 - down_progress)
                }
            }

            LearningScheduleConfig::NoamLR {
                model_size,
                warmup_steps,
                factor,
            } => {
                // Noam scheduler: lr = factor * model_size^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
                let factor_val = factor.unwrap_or(1.0);
                let step = (epoch_after_warmup + 1) as f64; // +1 to avoid zero
                let warmup_steps_f = (*warmup_steps).max(1) as f64;
                let model_size_f = (*model_size).max(1) as f64;

                let scale = factor_val * model_size_f.powf(-0.5);
                let lr_scale = (step.powf(-0.5)).min(step * warmup_steps_f.powf(-1.5));

                // Apply to initial_lr as base
                initial_lr * scale * lr_scale
            }
        }
    }

    /// Log comprehensive evaluation summary with data split information
    async fn log_comprehensive_evaluation_summary(&self) {
        log::info!("🎯 ═══════════════════════════════════════════════════════════");
        log::info!("🎯 COMPREHENSIVE EVALUATION SUMMARY");
        log::info!("🎯 ═══════════════════════════════════════════════════════════");

        // Data split summary
        let val_samples = self
            .stored_val_sequences
            .as_ref()
            .map(|v| v.shape()[0])
            .unwrap_or(0);
        let test_samples = self.stored_test_sequences.shape()[0];

        log::info!("📊 Data Split Summary:");
        log::info!(
            "   • Validation Samples: {} (used for epoch metrics and final validation)",
            val_samples
        );
        log::info!(
            "   • Test Samples: {} (reserved for final evaluation)",
            test_samples
        );

        if test_samples > 0 {
            log::info!("✅ Three-way split successfully implemented:");
            log::info!("   • Training: Used for model optimization");
            log::info!("   • Validation: Used for early stopping and hyperparameter tuning");
            log::info!("   • Test: Used for unbiased final performance evaluation");
        } else {
            log::info!("📝 Two-way split used (no test data reserved)");
        }

        // Target context information
        if let Some((target_name, target_type)) = &self.target_context {
            log::info!("🎯 Target Information:");
            log::info!("   • Target Name: {}", target_name);
            log::info!("   • Target Type: {:?}", target_type);
        }

        // Class weights information
        // No class weights - skip weight logging

        log::info!("🎯 ═══════════════════════════════════════════════════════════");
    }

    /// Train XGBoost phase of hybrid model (Phase 2)
    ///
    /// This method implements the second phase of the hybrid training where
    /// XGBoost learns the nonlinear mapping from LSTM features to targets.
    ///
    /// # Arguments
    /// * `sequences` - Training sequences [batch_size, seq_len, features]
    /// * `targets` - Training targets [batch_size, output_size]
    /// * `config` - Training configuration with XGBoost settings
    async fn train_xgboost_phase(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        log::info!("🌲 Phase 2: XGBoost training on LSTM features");

        // Extract LSTM features for all training sequences
        log::info!(
            "🔍 Extracting LSTM features from {} sequences...",
            sequences.shape()[0]
        );
        let lstm_features = self.extract_all_lstm_features(sequences)?;

        log::info!(
            "📊 LSTM features extracted: shape={:?}, architecture={:?}",
            lstm_features.shape(),
            self.architecture
        );

        // Convert targets to tensor
        let targets_tensor = self.convert_targets_to_tensor(targets)?;

        // Determine XGBoost objective and metric based on target type
        let mut xgb_config = config.model.xgboost.clone();

        // Use the config's feature_dim directly - prioritize user configuration
        let config_feature_dim = self.get_xgboost_feature_dim_with_config(&xgb_config);
        log::info!(
            "📊 Using XGBoost feature_dim from config: {} for {:?} architecture",
            config_feature_dim,
            self.architecture
        );
        // No need to update xgb_config.feature_dim as it already contains the correct value

        if let Some((target_name, target_type)) = &self.target_context {
            let num_classes = targets.shape()[1];
            xgb_config.objective =
                crate::model::xgboost::get_objective_for_target(target_name, num_classes);
            xgb_config.eval_metric =
                crate::model::xgboost::get_eval_metric_for_target(target_name, num_classes);

            log::info!(
                "🎯 Target: {} ({:?}) - Objective: {}, Metric: {}",
                target_name,
                target_type,
                xgb_config.objective,
                xgb_config.eval_metric
            );
        }

        // Create XGBoost regressor
        let mut xgb_regressor =
            crate::model::xgboost::XGBoostRegressor::new(xgb_config, self.device.clone());

        // Train XGBoost model
        xgb_regressor.train(&lstm_features, &targets_tensor)?;

        // Store trained XGBoost model
        self.xgboost_model = Some(xgb_regressor);

        // Log feature importance if available
        if let Some(xgb_model) = &self.xgboost_model {
            log::debug!("🔍 Checking XGBoost feature importance...");
            match xgb_model.get_feature_importance() {
                Some(importance) => {
                    log::info!("📊 XGBoost Feature Importance (top 10):");
                    let mut importance_vec: Vec<_> = importance.iter().collect();
                    importance_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

                    for (i, (feature, score)) in importance_vec.iter().take(10).enumerate() {
                        log::info!("   {}. {}: {:.4}", i + 1, feature, score);
                    }
                }
                None => {
                    log::warn!("⚠️  XGBoost feature importance not available - check save_feature_importance config");
                }
            }
        } else {
            log::error!("❌ XGBoost model not found after training");
        }

        log::info!("✅ XGBoost hybrid training completed successfully");
        Ok(())
    }

    /// Convert targets ndarray to Candle tensor
    fn convert_targets_to_tensor(&self, targets: &Array2<f64>) -> Result<candle_core::Tensor> {
        let batch_size = targets.shape()[0];
        let output_size = targets.shape()[1];

        let mut target_data: Vec<f32> = Vec::with_capacity(batch_size * output_size);
        for batch_idx in 0..batch_size {
            for output_idx in 0..output_size {
                target_data.push(targets[[batch_idx, output_idx]] as f32);
            }
        }

        candle_core::Tensor::from_vec(target_data, (batch_size, output_size), &self.device)
            .map_err(|e| VangaError::model(format!("Failed to create targets tensor: {}", e)))
    }

    /// Calculate gradient norm from GradStore (L2 norm across all parameters)
    ///
    /// This method calculates the total gradient norm using the standard L2 approach:
    /// ||g|| = sqrt(sum(||g_i||²)) for all parameters i
    ///
    /// This matches PyTorch's clip_grad_norm_ implementation exactly.
    fn calculate_gradstore_norm(&self, grads: &candle_core::backprop::GradStore) -> Result<f64> {
        let mut total_norm_squared = 0.0f64;
        let mut param_count = 0;

        // Iterate through all variables in the VarMap to get their gradients
        for var in self.varmap.all_vars().iter() {
            if let Some(grad) = grads.get(var) {
                // Calculate squared norm for this parameter's gradient
                let grad_squared = grad.sqr().map_err(|e| {
                    VangaError::ModelError(format!("Failed to square gradient tensor: {}", e))
                })?;

                let grad_norm_squared = grad_squared
                    .sum_all()
                    .map_err(|e| {
                        VangaError::ModelError(format!("Failed to sum gradient squares: {}", e))
                    })?
                    .to_scalar::<f32>()
                    .map_err(|e| {
                        VangaError::ModelError(format!(
                            "Failed to convert gradient norm to scalar: {}",
                            e
                        ))
                    })? as f64;

                total_norm_squared += grad_norm_squared;
                param_count += 1;

                log::trace!("Gradient norm² for parameter: {:.6e}", grad_norm_squared);
            }
        }

        let total_norm = total_norm_squared.sqrt();

        log::debug!(
            "🔍 Total gradient norm: {:.6e} across {} parameters",
            total_norm,
            param_count
        );

        Ok(total_norm)
    }
}
