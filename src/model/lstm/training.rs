//! Training pipeline and optimization
//!
//! This module implements the complete training pipeline for LSTM models with:
//! - Proper gradient clipping using direct gradient scaling (not learning rate modification)
//! - Multi-optimizer support with preserved state integrity
//! - Comprehensive validation and early stopping
//! - Advanced learning rate scheduling
//! - Gradient flow monitoring and validation

use super::config::{LSTMModel, OptimizerWrapper};

use crate::targets::TargetType;
use crate::utils::diagnostics::TrainingDiagnostics;
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
use std::collections::hash_map::DefaultHasher;

/// Unified ReduceOnPlateau scheduler with proper state management.
/// Unlike epoch-based schedulers that only need config, ReduceOnPlateau needs to track
/// loss history across epochs to make reduction decisions.
#[derive(Debug, Clone)]
pub struct ReduceOnPlateauScheduler {
    /// Current learning rate
    pub current_lr: f64,
    /// Best loss observed so far
    best_loss: f64,
    /// Number of epochs since last improvement
    patience_counter: u32,
    /// Number of epochs to wait before reducing LR
    patience: u32,
    /// Factor to multiply LR by when reducing
    factor: f64,
    /// Whether this is the first step (no previous loss to compare)
    is_first_step: bool,
}

impl ReduceOnPlateauScheduler {
    /// Create a new scheduler with the given initial LR and config
    pub fn new(initial_lr: f64, patience: u32, factor: f64) -> Self {
        Self {
            current_lr: initial_lr,
            best_loss: f64::INFINITY,
            patience_counter: 0,
            patience,
            factor,
            is_first_step: true,
        }
    }

    /// Step the scheduler with the current loss.
    /// Returns the new learning rate (may be reduced if patience exceeded).
    /// Also returns whether a reduction occurred (for logging).
    pub fn step(&mut self, loss: f64) -> (f64, bool) {
        let was_reduced = if self.is_first_step {
            // First step: just record the loss, no reduction yet
            self.is_first_step = false;
            self.best_loss = loss;
            false
        } else if loss < self.best_loss {
            // Improvement: reset patience, keep current LR
            self.best_loss = loss;
            self.patience_counter = 0;
            false
        } else {
            // No improvement: increment patience
            self.patience_counter += 1;

            // CRITICAL FIX: Use > instead of >= to reduce AFTER patience exceeded, not AT patience
            // patience=3 means: wait 3 epochs (counter: 1, 2, 3), then reduce on 4th epoch
            if self.patience_counter > self.patience {
                // Patience exceeded: reduce LR
                self.current_lr *= self.factor;
                self.patience_counter = 0;
                true
            } else {
                false
            }
        };

        (self.current_lr, was_reduced)
    }

    /// Get the current learning rate without stepping
    pub fn current_lr(&self) -> f64 {
        self.current_lr
    }

    /// Get the current patience counter (epochs without improvement)
    pub fn patience_counter(&self) -> u32 {
        self.patience_counter
    }

    /// Get the configured patience threshold
    pub fn patience(&self) -> u32 {
        self.patience
    }
}

/// Deterministic shuffle using Fisher-Yates algorithm with linear congruential generator
/// This is the same robust shuffling algorithm used in validation to ensure consistency
pub fn shuffle_indices_deterministic(indices: &mut [usize], seed_components: &[u64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();

    // Hash all seed components to create unique seed
    for &component in seed_components {
        component.hash(&mut hasher);
    }

    let seed = hasher.finish();

    // Fisher-Yates shuffle with linear congruential generator (same as validation)
    let mut rng_state = seed;
    for i in (1..indices.len()).rev() {
        // Linear congruential generator with proven parameters
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let j = (rng_state as usize) % (i + 1);
        indices.swap(i, j);
    }

    seed
}

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
            if total_samples.is_multiple_of(batch_size) { batch_size } else { total_samples % batch_size }
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
            // Assume 1h timeframe for validation calculations (actual targets use detected timeframe)
            let assumed_timeframe = 60;
            config
                .horizons
                .iter()
                .map(|h| {
                    crate::utils::parser::parse_horizon_to_steps(h, assumed_timeframe).unwrap_or(1)
                })
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

        // Initialize network if not already done (with weight initialization for training)
        if self.lstm_layers.is_none() || self.output_layer.is_none() {
            log::info!("🆕 FRESH TRAINING: Initializing new LSTM network layers and weights");
            self.initialize_network(Some(false))?; // Ensure weight initialization for training

            // Apply Xavier initialization for fresh training
            self.apply_xavier_initialization()?;

            // 🔍 DIAGNOSTIC: Test initial model predictions (before any training)
            // This verifies that weight initialization produces uniform output distribution
            // Expected: [0.200, 0.200, 0.200, 0.200, 0.200] (baseline for 5-class)
            if sequences.shape()[0] > 0 {
                log::info!(
                    "🔍 DIAGNOSTIC: Testing initial model predictions (before any training)..."
                );

                // Use first batch for diagnostic (or full sequences if smaller than batch size)
                let diagnostic_batch_size = std::cmp::min(
                    sequences.shape()[0],
                    std::cmp::min(64, self.training_config.batch_size),
                );

                let diag_sequences = sequences.slice(ndarray::s![0..diagnostic_batch_size, .., ..]);
                let diag_tensor =
                    self.convert_sequences_to_prediction_tensor(&diag_sequences.to_owned())?;

                // Forward pass in inference mode (no dropout) to get initial predictions
                let initial_logits = self.forward(&diag_tensor, false)?;

                // Apply softmax to get probabilities
                let initial_probs = candle_nn::ops::softmax(&initial_logits, 1)?;

                // Calculate mean probabilities across all samples in batch
                let probs_data: Vec<f32> = initial_probs.flatten_all()?.to_vec1::<f32>()?;

                let num_classes = probs_data.len() / diagnostic_batch_size;
                let mut mean_probs = vec![0.0f64; num_classes];

                for class_idx in 0..num_classes {
                    let mut class_sum = 0.0f64;
                    for sample_idx in 0..diagnostic_batch_size {
                        class_sum += probs_data[sample_idx * num_classes + class_idx] as f64;
                    }
                    mean_probs[class_idx] = class_sum / diagnostic_batch_size as f64;
                }

                // Format probabilities for logging
                let probs_formatted: Vec<String> =
                    mean_probs.iter().map(|p| format!("{:.3}", p)).collect();
                log::info!(
                    "   Initial output probabilities: [{}]",
                    probs_formatted.join(", ")
                );

                // Calculate deviation from uniform (0.2 for 5-class)
                let uniform = 1.0 / num_classes as f64;
                let max_deviation = mean_probs
                    .iter()
                    .map(|p| (p - uniform).abs())
                    .fold(0.0f64, f64::max);

                // Check if predictions are approximately uniform (within 0.1 tolerance)
                let is_uniform = max_deviation < 0.1;

                // Log expected value
                let expected_formatted: Vec<String> = (0..num_classes)
                    .map(|_| format!("{:.3}", uniform))
                    .collect();
                log::info!(
                    "   Expected: [{}] (uniform distribution)",
                    expected_formatted.join(", ")
                );

                if is_uniform {
                    log::info!(
                        "   ✅ Initial predictions are UNIFORM (zero bias working correctly)"
                    );
                } else {
                    log::warn!(
                        "   ⚠️ Initial predictions deviate from uniform by {:.3}",
                        max_deviation
                    );
                    log::warn!(
                        "   This may indicate initialization issues or unexpected model state"
                    );
                }

                // Clean up diagnostic tensors
                drop(diag_tensor);
                drop(initial_logits);
                drop(initial_probs);
            }
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

            // Parse validation_gap from config (e.g., "1h", "30m", "0" for no gap)
            let validation_gap_steps = if !config.training.validation_gap.is_empty() {
                crate::utils::parser::parse_horizon_to_steps(&config.training.validation_gap, 60)
                    .unwrap_or(0)
            } else {
                0
            };

            // Calculate minimum gap: max of calculated gap and user-specified validation_gap
            let max_horizon_steps = if !config.horizons.is_empty() {
                // Assume 1h timeframe for validation calculations (actual targets use detected timeframe)
                let assumed_timeframe = 60;
                config
                    .horizons
                    .iter()
                    .map(|h| {
                        crate::utils::parser::parse_horizon_to_steps(h, assumed_timeframe)
                            .unwrap_or(1)
                    })
                    .max()
                    .unwrap_or(72)
            } else {
                72
            };

            // CRITICAL: Use max of calculated gap and user-specified validation_gap
            let calculated_gap = self.config.sequence_length + max_horizon_steps;
            let gap_size = calculated_gap.max(validation_gap_steps);

            log::info!(
                "🔒 Gap calculation: sequence_length({}) + max_horizon_steps({}) = {} calculated, validation_gap={} steps, final_gap={}",
                self.config.sequence_length,
                max_horizon_steps,
                calculated_gap,
                validation_gap_steps,
                gap_size
            );

            // FIXED: Account for gap in validation split calculation
            // Validation is now 20% of training data (after test split), not 20% of total
            let effective_samples = available_for_training.saturating_sub(gap_size);
            let train_samples = ((1.0 - validation_split) * effective_samples as f64) as usize;
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
            "🎯 Dataset: {} train samples{}, batch_size={}",
            total_train_samples,
            if use_validation {
                format!(", {} val samples", total_val_samples)
            } else {
                String::new()
            },
            batch_size
        );

        // Calculate and log samples that will actually be used
        // FIXED: When batch_size > total_samples, use all samples as single batch
        let (num_complete_train_batches, train_samples_used, train_samples_dropped) =
            if batch_size >= total_train_samples {
                // Use all samples as single batch
                (1, total_train_samples, 0)
            } else {
                // Use complete batches only
                let num_complete = total_train_samples / batch_size;
                let samples_used = num_complete * batch_size;
                let samples_dropped = total_train_samples - samples_used;
                (num_complete, samples_used, samples_dropped)
            };

        if train_samples_dropped > 0 {
            log::info!(
                "📊 Training: Using {} samples ({} complete batches), dropping {} incomplete samples ({:.1}%)",
                train_samples_used,
                num_complete_train_batches,
                train_samples_dropped,
                (train_samples_dropped as f64 / total_train_samples as f64) * 100.0
            );
        } else {
            log::info!(
                "📊 Training: Using all {} samples ({} complete batches)",
                train_samples_used,
                num_complete_train_batches
            );
        }

        if use_validation {
            let num_complete_val_batches = total_val_samples / batch_size;
            let val_samples_used = num_complete_val_batches * batch_size;
            let val_samples_dropped = total_val_samples - val_samples_used;

            if val_samples_dropped > 0 {
                log::info!(
                    "📊 Validation: Using {} samples ({} complete batches), dropping {} incomplete samples ({:.1}%)",
                    val_samples_used,
                    num_complete_val_batches,
                    val_samples_dropped,
                    (val_samples_dropped as f64 / total_val_samples as f64) * 100.0
                );
            } else {
                log::info!(
                    "📊 Validation: Using all {} samples ({} complete batches)",
                    val_samples_used,
                    num_complete_val_batches
                );
            }
        }

        log::info!("🔧 Optimizer: {:?}", config.training.optimizer);

        // Memory prevalidation and warnings
        self.validate_batch_configuration(total_train_samples, batch_size)?;

        // Setup or reuse optimizer with proper learning rate for this window
        let mut optimizer = if let Some(mut existing_optimizer) = self.optimizer.take() {
            // CRITICAL: Reuse existing optimizer to preserve momentum/velocity
            // BUT update learning rate for window decay
            let new_lr = config.training.learning_rate;
            log::info!(
                "♻️ REUSING optimizer with preserved momentum/velocity, updating LR to {:.6} (window decay applied)",
                new_lr
            );

            // Update learning rate while preserving optimizer state
            existing_optimizer.set_learning_rate(new_lr);
            existing_optimizer
        } else {
            // Fresh training: create new optimizer
            log::info!("🆕 Creating fresh optimizer (no previous state to preserve)");
            self.setup_advanced_optimizer(config)?
        };

        // Extract learning rate configuration
        let target_lr = config.training.learning_rate;

        // Extract warmup configuration
        let warmup_epochs = config.training.warmup_epochs;

        // Initialize ReduceOnPlateau scheduler if configured (unified with other schedulers)
        let mut reduce_on_plateau_scheduler: Option<ReduceOnPlateauScheduler> =
            match &config.training.learning_schedule {
                Some(crate::config::training::LearningScheduleConfig::ReduceOnPlateau {
                    patience,
                    factor,
                    ..
                }) => Some(ReduceOnPlateauScheduler::new(target_lr, *patience, *factor)),
                _ => None,
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
        // Log ReduceOnPlateau config if configured
        if let Some(ref scheduler) = reduce_on_plateau_scheduler {
            log::info!(
                "  - ReduceOnPlateau: patience={}, factor={:.3}",
                scheduler.patience,
                scheduler.factor
            );
        }
        log::info!("  - Target learning rate: {:.6}", target_lr);

        // 🔍 COMPREHENSIVE MODEL DIAGNOSTICS - Log model capacity and configuration
        log::info!("🏗️ MODEL ARCHITECTURE DIAGNOSTICS:");
        log::info!("   📐 Hidden sizes: {:?}", self.config.hidden_sizes);
        log::info!("   🔢 Total layers: {}", self.config.num_layers);
        log::info!("   📊 Input size: {}", self.config.input_size);
        log::info!("   🎯 Output size: {}", self.config.output_size);
        log::info!("   📏 Sequence length: {}", self.config.sequence_length);

        // Calculate total parameters for capacity analysis
        let total_params = self.config.total_parameters();
        log::info!(
            "   🧮 Total parameters: {} ({:.2}M)",
            total_params,
            total_params as f64 / 1_000_000.0
        );

        // DEBUG: Log detailed parameter calculation
        log::debug!("🔍 PARAMETER CALCULATION DEBUG:");
        log::debug!("   📊 Input size: {}", self.config.input_size);
        log::debug!("   🔢 Number of layers: {}", self.config.num_layers);
        log::debug!("   📐 Hidden sizes: {:?}", self.config.hidden_sizes);
        for layer_idx in 0..self.config.num_layers {
            let input_size = if layer_idx == 0 {
                self.config.input_size
            } else {
                self.config.get_hidden_size_for_layer(layer_idx - 1)
            };
            let hidden_size = self.config.get_hidden_size_for_layer(layer_idx);
            let layer_params = (input_size + hidden_size + 1) * hidden_size * 4;
            log::debug!(
                "   🔍 Layer {}: input={}, hidden={}, params={}",
                layer_idx,
                input_size,
                hidden_size,
                layer_params
            );
        }

        // Log regularization settings using diagnostics module
        let dropout_enabled = self
            .dropout_config
            .as_ref()
            .is_some_and(|config| config.enabled);
        let dropout_rate = self.dropout_config.as_ref().and_then(|config| {
            if config.enabled {
                match &config.rate {
                    crate::config::model::DropoutRate::Fixed(rate) => Some(*rate),
                    crate::config::model::DropoutRate::Auto { min_rate, max_rate } => {
                        // For diagnostics, show the average of the range
                        Some((min_rate + max_rate) / 2.0)
                    }
                    crate::config::model::DropoutRate::Adaptive => {
                        // For adaptive, we can't show a specific rate in diagnostics
                        None
                    }
                }
            } else {
                None
            }
        });
        TrainingDiagnostics::log_regularization_config(dropout_enabled, dropout_rate);

        // Log optimizer configuration using diagnostics module
        TrainingDiagnostics::log_optimizer_config(
            &config.training.optimizer,
            config.training.learning_rate,
        );

        // Log data configuration using diagnostics module
        TrainingDiagnostics::log_data_config(
            total_train_samples,
            total_val_samples,
            batch_size,
            use_validation,
        );

        // Log LSTM capacity assessment using diagnostics module
        let sequence_length = self.config.sequence_length;
        let num_features = self.config.input_size;
        TrainingDiagnostics::log_capacity_assessment(
            total_train_samples,
            sequence_length,
            num_features,
            total_params,
        );
        // Unified training loop with warmup, adaptive learning, optional validation, and early stopping
        // Track previous epoch's loss for ReduceOnPlateau scheduler (None for first epoch)
        let mut previous_epoch_loss: Option<f64> = None;

        for epoch in 0..self.training_config.epochs {
            // Clear variational dropout masks at the start of each epoch for fresh randomization
            self.clear_dropout_masks();

            // Initialize epoch tracking variables
            let mut epoch_train_loss = 0.0;
            let mut epoch_grad_norm = 0.0; // Track gradient norm for epoch logging
            let mut batch_count = 0;

            // CRITICAL FIX: Shuffle training data indices each epoch to prevent overfitting to batch order
            let mut sample_indices: Vec<usize> = (0..total_train_samples).collect();

            // Create deterministic but epoch-varying shuffle using the same robust algorithm as validation
            let seed_components = [
                config.training.seed,
                epoch as u64,
                total_train_samples as u64,
            ];
            let shuffle_seed = shuffle_indices_deterministic(&mut sample_indices, &seed_components);

            // Validation: Ensure all samples are present exactly once
            debug_assert_eq!(sample_indices.len(), total_train_samples);
            debug_assert!(
                {
                    let mut sorted_indices = sample_indices.clone();
                    sorted_indices.sort();
                    sorted_indices == (0..total_train_samples).collect::<Vec<_>>()
                },
                "Shuffled indices must contain all samples exactly once"
            );

            if epoch == 0 {
                log::info!("🔀 Training data will be shuffled each epoch to prevent batch order overfitting");
                log::debug!(
                    "🔍 Epoch 0 shuffle seed: {}, first batch indices: {:?}",
                    shuffle_seed,
                    &sample_indices[0..std::cmp::min(10, sample_indices.len())]
                );
            } else if epoch == 1 {
                log::debug!(
                    "🔍 Epoch 1 shuffle seed: {}, first batch indices: {:?}",
                    shuffle_seed,
                    &sample_indices[0..std::cmp::min(10, sample_indices.len())]
                );
            }

            // Check if optimizer is learning-rate-free (Prodigy/FracProdigy)
            let is_lr_free_optimizer = matches!(
                &config.training.optimizer,
                crate::config::training::OptimizerType::Prodigy { .. }
                    | crate::config::training::OptimizerType::FracProdigy { .. }
            );

            // Calculate warmup learning rate for current epoch
            if epoch < warmup_epochs as usize {
                // Linear warmup from 0 to target_lr
                let warmup_progress = (epoch + 1) as f64 / (warmup_epochs as f64);
                let warmup_lr = target_lr * warmup_progress;

                // CRITICAL: Skip LR updates for learning-rate-free optimizers
                // Prodigy/FracProdigy manage their own learning rates automatically
                if !is_lr_free_optimizer {
                    optimizer.set_learning_rate(warmup_lr);
                } else {
                    // For Prodigy/FracProdigy, just track the effective LR
                    let _ = optimizer.learning_rate();
                }

                if epoch == 0 || epoch == (warmup_epochs as usize) - 1 {
                    if is_lr_free_optimizer {
                        log::info!(
                            "🔥 Warmup epoch {}/{}: effective learning rate = {:.6} (auto-managed by {})",
                            epoch + 1,
                            warmup_epochs,
                            optimizer.learning_rate(),
                            if matches!(&config.training.optimizer, crate::config::training::OptimizerType::Prodigy { .. }) {
                                "Prodigy"
                            } else {
                                "FracProdigy"
                            }
                        );
                    } else {
                        log::info!(
                            "🔥 Warmup epoch {}/{}: learning rate = {:.6}",
                            epoch + 1,
                            warmup_epochs,
                            optimizer.learning_rate()
                        );
                    }
                }
            } else {
                // Apply learning schedule after warmup phase (if configured)
                if let Some(schedule_config) = &config.training.learning_schedule {
                    // CRITICAL: Skip schedule for learning-rate-free optimizers
                    if is_lr_free_optimizer {
                        // Prodigy/FracProdigy ignore external schedules - they manage LR automatically
                        let _ = optimizer.learning_rate();

                        if epoch == warmup_epochs as usize {
                            log::warn!(
                                "⚠️ Learning rate schedule {:?} is IGNORED for {} optimizer",
                                schedule_config,
                                if matches!(
                                    &config.training.optimizer,
                                    crate::config::training::OptimizerType::Prodigy { .. }
                                ) {
                                    "Prodigy"
                                } else {
                                    "FracProdigy"
                                }
                            );
                            log::info!(
                                "💡 {} automatically adapts learning rate using D-estimate: lr_eff = base_lr / (D × √t)",
                                if matches!(&config.training.optimizer, crate::config::training::OptimizerType::Prodigy { .. }) {
                                    "Prodigy"
                                } else {
                                    "FracProdigy"
                                }
                            );
                        }
                    } else {
                        // Handle ReduceOnPlateau using unified scheduler (called at epoch start)
                        if let Some(ref mut scheduler) = reduce_on_plateau_scheduler {
                            // Use the previous epoch's loss (passed from epoch end)
                            // If no previous loss yet (first epoch), just use initial LR
                            if let Some(previous_loss) = previous_epoch_loss {
                                let (new_lr, was_reduced) = scheduler.step(previous_loss);
                                if was_reduced {
                                    log::info!(
                                        "🔄 Adaptive learning rate reduced to: {:.6} (patience exceeded)",
                                        new_lr
                                    );
                                }
                                optimizer.set_learning_rate(new_lr);
                            } else {
                                // First epoch: just set initial LR
                                let initial_lr = scheduler.current_lr();
                                optimizer.set_learning_rate(initial_lr);
                            }

                            // Log scheduler status AFTER step to show actual best_loss
                            log::debug!(
                                "📊 ReduceOnPlateau status - Best loss: {:.6}, Patience: {}/{}",
                                scheduler.best_loss,
                                scheduler.patience_counter,
                                scheduler.patience
                            );
                        } else {
                            // Non-ReduceOnPlateau schedules: use epoch-based calculation
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

                            optimizer.set_learning_rate(scheduled_lr);
                        }
                    }
                }
            }

            // Training phase - process data in shuffled batches (drop incomplete batches for stability)
            let num_complete_batches = total_train_samples / batch_size;
            let samples_used_for_training = num_complete_batches * batch_size;

            for (batch_idx, batch_start) in (0..samples_used_for_training)
                .step_by(batch_size)
                .enumerate()
            {
                let batch_end = batch_start + batch_size; // Always complete batch
                let actual_batch_size = batch_size; // Always full batch size

                // Extract shuffled batch indices
                let batch_indices = &sample_indices[batch_start..batch_end];

                // Create batch arrays using shuffled indices
                let mut batch_sequences = ndarray::Array3::<f64>::zeros((
                    actual_batch_size,
                    train_sequences.shape()[1], // sequence_length
                    train_sequences.shape()[2], // num_features
                ));
                let mut batch_targets = ndarray::Array2::<f64>::zeros((
                    actual_batch_size,
                    train_targets.shape()[1], // num_targets
                ));

                // Fill batch with shuffled samples
                for (batch_pos, &sample_idx) in batch_indices.iter().enumerate() {
                    batch_sequences
                        .slice_mut(ndarray::s![batch_pos, .., ..])
                        .assign(&train_sequences.slice(ndarray::s![sample_idx, .., ..]));
                    batch_targets
                        .slice_mut(ndarray::s![batch_pos, ..])
                        .assign(&train_targets.slice(ndarray::s![sample_idx, ..]));
                }

                // Convert batch to tensors
                let (input_tensor, target_tensor) =
                    self.convert_sequences_to_tensors(&batch_sequences, &batch_targets)?;

                // Forward pass (training mode - enable dropout)
                // CRITICAL: Each batch gets fresh hidden states via seq_init() in forward()
                // This prevents hidden state contamination between batches
                let predictions = self.forward(&input_tensor, true)?;

                // CRITICAL FIX: Apply ONLY bias correction during training
                // Temperature scaling is POST-HOC and applied after training completes
                // Research: Guo et al. 2017, ICLR 2025 - temperature scaling is post-hoc calibration
                let predictions_for_loss = if let Some(ref corrector) = self.bias_corrector {
                    let ramp_up_epochs = corrector.config.ramp_up_epochs;

                    if epoch >= ramp_up_epochs
                        && corrector.is_calibrated
                        && corrector.config.enabled
                    {
                        // Apply bias correction to RAW LOGITS (before softmax)
                        let corrected =
                            corrector.apply_correction_to_logits(&predictions, epoch)?;

                        // Log correction impact periodically
                        if corrector.config.print_info
                            && corrector.config.recalibration_frequency > 0
                            && epoch % corrector.config.recalibration_frequency == 0
                            && batch_idx == 0
                        {
                            let original_probs = candle_nn::ops::softmax(&predictions, 1)?;
                            let corrected_probs = candle_nn::ops::softmax(&corrected, 1)?;

                            if let Ok(kl_div) = corrector
                                .calculate_correction_impact(&original_probs, &corrected_probs)
                            {
                                log::info!(
                                    "📊 Bias correction impact at epoch {} (KL divergence): {:.6}",
                                    epoch + 1,
                                    kl_div
                                );
                            }
                        }

                        corrected
                    } else {
                        predictions.clone()
                    }
                } else {
                    predictions.clone()
                };

                // Calculate loss using potentially corrected predictions
                let base_loss =
                    self.calculate_loss(&predictions_for_loss, &target_tensor, config, false)?;

                // Get loss value for reporting BEFORE gradient clipping (to avoid move issues)
                let batch_loss_value = base_loss.to_scalar::<f32>().map_err(|e| {
                    VangaError::ModelError(format!("Loss scalar conversion failed: {}", e))
                })?;

                // CRITICAL FIX: Use proper Candle backward_step API to prevent gradient accumulation
                // This replaces manual gradient handling with framework's built-in gradient management
                let effective_grad_norm = self.apply_gradient_clipping_and_step(
                    &mut optimizer,
                    &base_loss,
                    self.training_config.clip_gradient,
                    epoch,
                    batch_idx,
                )?;

                // GRADIENT FLOW VALIDATION: Basic validation without requiring gradients
                // This ensures gradients are not NaN, infinite, or problematically large/small
                self.validate_gradient_norm(effective_grad_norm)?;

                // 🔍 ENHANCED GRADIENT MONITORING: Track gradient patterns across batches
                // This helps detect gradient accumulation or instability issues
                if epoch > 0 && batch_count > 1 {
                    let avg_grad_norm = epoch_grad_norm / batch_count as f64;
                    let gradient_growth_rate = effective_grad_norm / avg_grad_norm.max(1e-12_f64);

                    // Only warn if we have meaningful data and significant growth
                    // Growth rate > 3.0x suggests potential gradient accumulation
                    if avg_grad_norm > 1e-12_f64 && gradient_growth_rate > 3.0 {
                        log::warn!(
                            "⚠️ Potential gradient accumulation detected: current_norm={:.6e}, avg_norm={:.6e}, growth_rate={:.2}x",
                            effective_grad_norm,
                            avg_grad_norm,
                            gradient_growth_rate
                        );
                    }
                }

                // Accumulate gradient norm for epoch-level reporting and analysis
                epoch_grad_norm += effective_grad_norm;
                batch_count += 1;

                // Accumulate loss for epoch reporting
                let batch_loss = batch_loss_value;

                // 🔍 DETAILED TRAINING BATCH DEBUG
                log::debug!(
                    "🔍 TRAIN E{} B{}: raw_loss={:.6}, batch_size={}, weighted_loss={:.6}, grad_norm={:.6}",
                    epoch + 1, batch_idx, batch_loss, actual_batch_size,
                    batch_loss * actual_batch_size as f32, effective_grad_norm
                );

                epoch_train_loss += batch_loss * actual_batch_size as f32;
            }

            // Calculate average training loss and gradient norm (using only complete batches)
            let avg_train_loss = epoch_train_loss / samples_used_for_training as f32;
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
            // Clear variational dropout masks before validation to ensure fresh masks
            self.clear_dropout_masks();

            let avg_val_loss = if let (Some(val_seq), Some(val_tgt)) =
                (&val_sequences_final, &val_targets_final)
            {
                let mut epoch_val_loss = 0.0;
                let total_val_samples = val_seq.shape()[0];

                // Drop incomplete batches for validation consistency
                let num_complete_val_batches = total_val_samples / batch_size;
                let val_samples_used = num_complete_val_batches * batch_size;

                // Collect all validation predictions and targets for bias correction
                let mut all_val_predictions = Vec::new();
                let mut all_val_targets = Vec::new();

                // CRITICAL ISSUE FOUND: Training uses SHUFFLED data, validation uses SEQUENTIAL data
                // This creates a fundamental distribution mismatch that causes divergence!
                //
                // TRAINING: Shuffled batches break temporal patterns, model learns on random sequences
                // VALIDATION: Sequential batches preserve temporal patterns, different data distribution
                //
                // SOLUTION: Either both should be shuffled OR both should be sequential
                // For time series, SEQUENTIAL is more appropriate for both

                // TEMPORARY FIX: Process validation in shuffled order to match training
                // Create shuffled indices for validation (same approach as training)
                let mut val_sample_indices: Vec<usize> = (0..val_samples_used).collect();
                let val_shuffle_seed_components = [
                    epoch as u64,
                    42u64, // Different base seed for validation
                    val_samples_used as u64,
                ];
                let _val_shuffle_seed = shuffle_indices_deterministic(
                    &mut val_sample_indices,
                    &val_shuffle_seed_components,
                );

                log::debug!(
                    "🔀 VALIDATION SHUFFLE FIX: Epoch {} - shuffling {} validation samples to match training distribution",
                    epoch + 1, val_samples_used
                );

                for (batch_idx, batch_start) in
                    (0..val_samples_used).step_by(batch_size).enumerate()
                {
                    let batch_end = batch_start + batch_size; // Always complete batch
                    let actual_batch_size = batch_size; // Always full batch size

                    // Extract shuffled validation batch indices (MATCHING TRAINING APPROACH)
                    let batch_indices = &val_sample_indices[batch_start..batch_end];

                    // Create batch arrays using shuffled indices (SAME AS TRAINING)
                    let mut batch_sequences = ndarray::Array3::<f64>::zeros((
                        actual_batch_size,
                        val_seq.shape()[1], // sequence_length
                        val_seq.shape()[2], // num_features
                    ));
                    let mut batch_targets = ndarray::Array2::<f64>::zeros((
                        actual_batch_size,
                        val_tgt.shape()[1], // num_targets
                    ));

                    // Fill batch with shuffled samples (SAME AS TRAINING)
                    for (batch_pos, &sample_idx) in batch_indices.iter().enumerate() {
                        batch_sequences
                            .slice_mut(ndarray::s![batch_pos, .., ..])
                            .assign(&val_seq.slice(ndarray::s![sample_idx, .., ..]));
                        batch_targets
                            .slice_mut(ndarray::s![batch_pos, ..])
                            .assign(&val_tgt.slice(ndarray::s![sample_idx, ..]));
                    }

                    // Convert batch to tensors (SAME AS TRAINING)
                    let (input_tensor, target_tensor) =
                        self.convert_sequences_to_tensors(&batch_sequences, &batch_targets)?;

                    // CRITICAL FIX: Forward pass with validation mode (no dropout)
                    // This was the main bug - dropout was being applied during validation
                    let predictions = self.forward(&input_tensor, false)?;

                    // CRITICAL FIX: Apply softmax to convert logits to probabilities for bias correction
                    // The output layer produces raw logits, but bias correction expects probabilities
                    let predictions_probs = candle_nn::ops::softmax(&predictions, 1)?;

                    // Store predictions and targets for bias correction calibration
                    let predictions_data: Vec<f32> = predictions_probs
                        .flatten_all()
                        .map_err(|e| {
                            VangaError::ModelError(format!("Failed to flatten predictions: {}", e))
                        })?
                        .to_vec1()
                        .map_err(|e| {
                            VangaError::ModelError(format!(
                                "Failed to convert predictions to vec: {}",
                                e
                            ))
                        })?;

                    let pred_shape = predictions_probs.shape();
                    let predictions_f64: Vec<f64> =
                        predictions_data.iter().map(|&x| x as f64).collect();
                    let predictions_array = ndarray::Array2::from_shape_vec(
                        (pred_shape.dims()[0], pred_shape.dims()[1]),
                        predictions_f64,
                    )
                    .map_err(|e| {
                        VangaError::ModelError(format!("Failed to create predictions array: {}", e))
                    })?;

                    all_val_predictions.push(predictions_array);
                    all_val_targets.push(batch_targets.clone());

                    // Calculate validation loss using ordinal regression (same as training)
                    // CRITICAL: Both training and validation use same ordinal loss for comparability
                    // This ensures consistent loss calculation between training and validation
                    let val_loss = self.calculate_loss(
                        &predictions,
                        &target_tensor,
                        config,
                        true, // is_validation = true
                    )?;
                    let val_batch_loss = val_loss.to_scalar::<f32>().map_err(|e| {
                        VangaError::ModelError(format!(
                            "Validation loss scalar conversion failed: {}",
                            e
                        ))
                    })?;

                    // 🔍 DETAILED VALIDATION BATCH DEBUG
                    log::debug!(
                        "🔍 VAL E{} B{}: raw_loss={:.6}, batch_size={}, weighted_loss={:.6} [SHUFFLED LIKE TRAINING]",
                        epoch + 1,
                        batch_idx,
                        val_batch_loss,
                        actual_batch_size,
                        val_batch_loss * actual_batch_size as f32
                    );

                    epoch_val_loss += val_batch_loss * actual_batch_size as f32;
                }

                let avg_val_loss = epoch_val_loss / val_samples_used as f32;

                // 🔍 EPOCH VALIDATION SUMMARY DEBUG
                log::debug!(
                    "🔍 VAL E{} SUMMARY: total_weighted_loss={:.6}, samples_used={}, avg_loss={:.6}",
                    epoch + 1, epoch_val_loss, val_samples_used, avg_val_loss
                );

                // Calibrate bias correction from all validation predictions
                if !all_val_predictions.is_empty() {
                    // Concatenate all validation predictions and targets
                    let total_val_predictions = ndarray::concatenate(
                        ndarray::Axis(0),
                        &all_val_predictions
                            .iter()
                            .map(|arr| arr.view())
                            .collect::<Vec<_>>(),
                    )
                    .map_err(|e| {
                        VangaError::ModelError(format!(
                            "Failed to concatenate validation predictions: {}",
                            e
                        ))
                    })?;

                    let total_val_targets = ndarray::concatenate(
                        ndarray::Axis(0),
                        &all_val_targets
                            .iter()
                            .map(|arr| arr.view())
                            .collect::<Vec<_>>(),
                    )
                    .map_err(|e| {
                        VangaError::ModelError(format!(
                            "Failed to concatenate validation targets: {}",
                            e
                        ))
                    })?;

                    // Bias correction will be calibrated periodically according to recalibration_frequency
                    // No initial calibration at epoch 0 - let model stabilize first

                    // ENSEMBLE CALIBRATION: Separate from bias correction, runs independently
                    if self.bias_correction_config.use_ensemble_calibration
                        && self.ensemble_calibrator.is_some()
                    {
                        let ensemble_cal = self.ensemble_calibrator.as_mut().unwrap();
                        // Ensemble calibration - initial setup (monitoring only during training)
                        if !ensemble_cal.is_calibrated {
                            log::info!("🎯 Ensemble calibration enabled (will be optimized POST-HOC after training)");

                            // Ensure predictions are 5-class format
                            if total_val_predictions.shape()[1] == 5 {
                                // Convert targets to one-hot if needed
                                let val_targets_for_ensemble = if total_val_targets.shape()[1] == 1
                                {
                                    let num_samples = total_val_targets.shape()[0];
                                    let mut one_hot =
                                        ndarray::Array2::<f64>::zeros((num_samples, 5));
                                    for (i, class_idx) in total_val_targets.iter().enumerate() {
                                        let class_index = (*class_idx as usize).min(4);
                                        one_hot[[i, class_index]] = 1.0;
                                    }
                                    one_hot
                                } else if total_val_targets.shape()[1] == 5 {
                                    total_val_targets.clone()
                                } else {
                                    total_val_targets.slice(s![.., 0..5]).to_owned()
                                };

                                // Initial calibration for monitoring only
                                ensemble_cal.calibrate_from_validation(
                                    &total_val_predictions,
                                    &val_targets_for_ensemble,
                                )?;
                            }
                        }
                    }
                }

                // Progressive bias correction recalibration during training
                if epoch > 0 && self.bias_corrector.is_some() {
                    let recalib_freq = self
                        .bias_corrector
                        .as_ref()
                        .unwrap()
                        .config
                        .recalibration_frequency;
                    if recalib_freq > 0
                        && epoch % recalib_freq == 0
                        && !all_val_predictions.is_empty()
                    {
                        // Recalibrate bias correction with recent validation data
                        if let Some(ref mut corrector) = self.bias_corrector {
                            let total_val_predictions = ndarray::concatenate(
                                ndarray::Axis(0),
                                &all_val_predictions
                                    .iter()
                                    .map(|arr| arr.view())
                                    .collect::<Vec<_>>(),
                            )
                            .map_err(|e| {
                                VangaError::ModelError(format!(
                                    "Failed to concatenate validation predictions for recalibration: {}",
                                    e
                                ))
                            })?;

                            let total_val_targets = ndarray::concatenate(
                                ndarray::Axis(0),
                                &all_val_targets
                                    .iter()
                                    .map(|arr| arr.view())
                                    .collect::<Vec<_>>(),
                            )
                            .map_err(|e| {
                                VangaError::ModelError(format!(
                                    "Failed to concatenate validation targets for recalibration: {}",
                                    e
                                ))
                            })?;

                            // Ensure proper shape for bias correction
                            if total_val_predictions.shape()[1] == 5 {
                                let val_targets_for_bias = if total_val_targets.shape()[1] == 1 {
                                    // Convert class indices to one-hot
                                    let num_samples = total_val_targets.shape()[0];
                                    let mut one_hot =
                                        ndarray::Array2::<f64>::zeros((num_samples, 5));
                                    for (i, class_idx) in total_val_targets.iter().enumerate() {
                                        let class_index = (*class_idx as usize).min(4);
                                        one_hot[[i, class_index]] = 1.0;
                                    }
                                    one_hot
                                } else {
                                    total_val_targets.clone()
                                };

                                // Recalibrate
                                corrector.calibrate_from_validation(
                                    &total_val_predictions,
                                    &val_targets_for_bias,
                                )?;

                                if corrector.is_calibrated {
                                    log::info!(
                                        "🔄 Bias correction recalibrated at epoch {} with factors: {:?}",
                                        epoch + 1, corrector.class_bias_factors
                                    );

                                    // Calculate and log class distribution improvement
                                    if let Some(stats) = &corrector.validation_stats {
                                        let pred_variance: f64 = stats
                                            .class_frequencies_predicted
                                            .iter()
                                            .map(|&f| (f - 0.2).powi(2))
                                            .sum::<f64>()
                                            / 5.0;
                                        let actual_variance: f64 = stats
                                            .class_frequencies_actual
                                            .iter()
                                            .map(|&f| (f - 0.2).powi(2))
                                            .sum::<f64>()
                                            / 5.0;

                                        log::info!(
                                            "📊 Class distribution variance - Predicted: {:.6}, Actual: {:.6}",
                                            pred_variance, actual_variance
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Ensemble calibrator recalibration during training (if enabled in config)
                // Remove ensemble recalibration during training - it's POST-HOC only
                // Temperature will be optimized once after training completes

                // Remove the periodic ensemble calibration logging during training
                // It's confusing and not useful - temperature is POST-HOC only

                // Calculate categorical metrics for all categorical targets
                if let Some((_, target_type)) = &self.target_context {
                    match target_type {
                        TargetType::PriceLevel
                        | TargetType::Direction
                        | TargetType::Volatility
                        | TargetType::Sentiment
                        | TargetType::Volume => {
                            self.calculate_categorical_validation_metrics(
                                val_seq, val_tgt, batch_size, epoch, config, None,
                            )
                            .await?;
                        }
                    }
                }

                Some(avg_val_loss)
            } else {
                None
            };

            // 📊 CRITICAL DEBUGGING: Track training vs validation consistency
            if let Some(val_loss) = avg_val_loss {
                let loss_ratio = val_loss / avg_train_loss;

                // ENHANCED DIAGNOSTIC: Detect problematic patterns with detailed analysis
                // Use recalibration_frequency for diagnostic print frequency (if available)
                let diagnostic_frequency = if let Some(ref corrector) = self.bias_corrector {
                    corrector.config.recalibration_frequency.max(1)
                } else {
                    5 // Default fallback
                };

                if loss_ratio > 1.15 && epoch % diagnostic_frequency == 0 {
                    log::warn!(
                        "\n🚨 VALIDATION LOSS DIVERGENCE DETECTED at epoch {}:",
                        epoch + 1
                    );
                    log::warn!(
                        "   📊 Val/Train ratio: {:.3}x (threshold: 1.15x)",
                        loss_ratio
                    );
                    log::warn!("   📈 Validation loss: {:.6}", val_loss);
                    log::warn!("   📉 Training loss: {:.6}", avg_train_loss);
                    log::warn!("   📊 Loss difference: {:.6}", val_loss - avg_train_loss);

                    // Analyze potential causes
                    log::warn!("🔍 POTENTIAL CAUSES ANALYSIS:");

                    // Focus on regularization and training parameters (not capacity)
                    // LSTM models can handle complex patterns with proper regularization

                    // Check dropout configuration
                    if let Some(dropout_config) = &self.dropout_config {
                        if dropout_config.enabled {
                            let dropout_rate = match &dropout_config.rate {
                                crate::config::model::DropoutRate::Fixed(rate) => *rate,
                                crate::config::model::DropoutRate::Auto { min_rate, max_rate } => {
                                    (min_rate + max_rate) / 2.0
                                }
                                _ => 0.2,
                            };
                            log::warn!(
                                "   💧 Dropout: {:.1}% (may need increase)",
                                dropout_rate * 100.0
                            );
                        } else {
                            log::warn!("   💧 Dropout: DISABLED (consider enabling)");
                        }
                    } else {
                        log::warn!("   💧 Dropout: NOT CONFIGURED (consider adding)");
                    }

                    // Check weight decay
                    match &config.training.optimizer {
                        crate::config::training::OptimizerType::AdamW { weight_decay, .. } => {
                            if *weight_decay < 0.01 {
                                log::warn!(
                                    "   🏋️ Weight decay: {:.4} (consider increasing to 0.01-0.1)",
                                    weight_decay
                                );
                            } else {
                                log::warn!(
                                    "   🏋️ Weight decay: {:.4} (seems reasonable)",
                                    weight_decay
                                );
                            }
                        }
                        _ => {
                            log::warn!("   🏋️ Weight decay: N/A (consider using AdamW)");
                        }
                    }

                    // Check learning rate (use actual effective LR from optimizer)
                    let effective_lr = optimizer.learning_rate();
                    if effective_lr > 0.001 {
                        log::warn!(
                            "   📈 Learning rate: {:.6} (consider reducing)",
                            effective_lr
                        );
                    }

                    // Check gradient norm
                    let avg_grad_norm = epoch_grad_norm / batch_count as f64;
                    if avg_grad_norm > 1.0 {
                        log::warn!(
                            "   📊 Gradient norm: {:.3} (high, may indicate instability)",
                            avg_grad_norm
                        );
                    }

                    log::warn!("🔧 SUGGESTED FIXES FOR LSTM TIME SERIES:");
                    log::warn!("   1. Increase dropout rate (current → +0.1-0.2)");
                    log::warn!("   2. Increase weight decay (0.01 → 0.05-0.1)");
                    log::warn!(
                        "   3. Reduce learning rate ({:.6} → {:.6})",
                        effective_lr,
                        effective_lr * 0.5
                    );
                    log::warn!("   4. Add early stopping with smaller patience");
                    log::warn!("   5. Use gradient clipping (< 1.0)");
                    log::warn!("   6. Consider sequence length reduction\n");
                }
            }

            // Record this epoch's loss for ReduceOnPlateau scheduler (used at next epoch start)
            // Use validation loss if available, otherwise use training loss
            let loss_for_scheduler = avg_val_loss
                .map(|v| v as f64)
                .unwrap_or(avg_train_loss as f64);
            // Store for next epoch's ReduceOnPlateau scheduler
            let _ = previous_epoch_loss.insert(loss_for_scheduler);

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
                // Get actual effective learning rate from optimizer
                // For Prodigy/FracProdigy, this returns the dynamically calculated LR
                // For other optimizers, this returns the configured LR
                let effective_lr = optimizer.learning_rate();

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
                                " [Constant]".to_string()
                            }
                            crate::config::training::LearningScheduleConfig::ReduceOnPlateau {
                                ..
                            } => {
                                // Get patience counter from scheduler if available
                                if let Some(ref scheduler) = reduce_on_plateau_scheduler {
                                    format!(
                                        " [ReduceOnPlateau: {}/{}]",
                                        scheduler.patience_counter(),
                                        scheduler.patience()
                                    )
                                } else {
                                    " [ReduceOnPlateau]".to_string()
                                }
                            }
                            crate::config::training::LearningScheduleConfig::LinearDecay {
                                ..
                            } => " [LinearDecay]".to_string(),
                            crate::config::training::LearningScheduleConfig::ExponentialDecay {
                                ..
                            } => " [ExponentialDecay]".to_string(),
                            crate::config::training::LearningScheduleConfig::StepDecay {
                                ..
                            } => " [StepDecay]".to_string(),
                            crate::config::training::LearningScheduleConfig::PolynomialDecay {
                                ..
                            } => " [PolynomialDecay]".to_string(),
                            crate::config::training::LearningScheduleConfig::CosineAnnealing {
                                ..
                            } => " [CosineAnnealing]".to_string(),
                            crate::config::training::LearningScheduleConfig::WarmRestarts {
                                ..
                            } => " [WarmRestarts]".to_string(),
                            crate::config::training::LearningScheduleConfig::OneCycle {
                                ..
                            } => " [OneCycle]".to_string(),
                            crate::config::training::LearningScheduleConfig::CyclicalLR {
                                ..
                            } => " [CyclicalLR]".to_string(),
                            crate::config::training::LearningScheduleConfig::NoamLR { .. } => {
                                " [NoamLR]".to_string()
                            }
                        }
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
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

                    // Get bias correction info if available
                    let bias_info = if let Some(ref corrector) = self.bias_corrector {
                        if corrector.is_calibrated {
                            let avg_bias = corrector.class_bias_factors.iter().sum::<f64>() / 5.0;
                            format!(", Bias: {:.2}", avg_bias)
                        } else {
                            String::new()
                        }
                    } else {
                        String::new()
                    };

                    log::info!(
                        "Epoch {}/{}: Train Loss = {:.6}, Val Loss = {:.6} (Ratio: {:.2}x {}), LR: {:.6}, Grad: {:.2e}{}{}, Early Stop: {}/{}{}{}",
                        epoch + 1,
                        self.training_config.epochs,
                        avg_train_loss,
                        val_loss,
                        loss_ratio,
                        ratio_status,
                        effective_lr,
                        avg_grad_norm,
                        warmup_status,
                        schedule_status,
                        early_stopping_counter,
                        early_stopping_patience,
                        bias_info,
                        target_info,
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
                        effective_lr,
                        warmup_status,
                        schedule_status
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
                    TargetType::PriceLevel
                    | TargetType::Direction
                    | TargetType::Volatility
                    | TargetType::Sentiment
                    | TargetType::Volume => {
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
                                    Some("validation"), // NEW: Explicitly mark as validation
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
                                    Some("test"), // NEW: Explicitly mark as test
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
            self.train_xgboost_phase(sequences, targets, config, val_sequences, val_targets)
                .await?;
        }

        // POST-HOC ENSEMBLE CALIBRATION (Temperature Scaling)
        // CRITICAL: This must happen AFTER training completes, never during training
        // Research: Guo et al. 2017, ICLR 2025 - temperature scaling is post-processing
        if self.bias_correction_config.use_ensemble_calibration {
            if let (Some(val_seq), Some(val_tgt)) = (val_sequences, val_targets) {
                log::info!("🌡️ Applying POST-HOC ensemble calibration (temperature scaling)...");
                log::info!("   📚 Research: Temperature scaling optimized on validation set AFTER training");

                self.calibrate_ensemble_post_training(val_seq, val_tgt)?;

                log::info!("✅ POST-HOC calibration complete - temperature will be applied during inference only");
            } else {
                log::warn!("⚠️ Ensemble calibration enabled but no validation data available");
                log::warn!(
                    "   Temperature scaling requires validation set for post-hoc optimization"
                );
            }
        }

        // CRITICAL: Store optimizer state for next window/continuation
        // This preserves momentum/velocity for incremental training
        self.optimizer = Some(optimizer);
        log::info!("💾 Optimizer state preserved for potential continuation training");

        // CRITICAL FIX: Clean up ALL checkpoint files at end of training to prevent memory leak
        // This ensures no checkpoint files are left behind after training completes
        let checkpoint_dir = std::env::temp_dir().join("vanga_checkpoints");
        let pid = std::process::id();
        #[allow(clippy::collapsible_if)]
        if let Ok(entries) = std::fs::read_dir(&checkpoint_dir) {
            let mut cleaned_count = 0;
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                    if filename.starts_with(&format!("best_model_{}_", pid)) {
                        if std::fs::remove_file(&path).is_ok() {
                            cleaned_count += 1;
                        }
                    }
                }
            }
            if cleaned_count > 0 {
                log::debug!(
                    "🧹 Cleaned up {} checkpoint files at end of training",
                    cleaned_count
                );
            }
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
                log::info!(
                    "AdaDelta parameters - weight_decay: {:.4}, rho: {:.3}, eps: {:.2e}",
                    weight_decay.unwrap_or(0.0),
                    rho,
                    eps
                );

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
                    "AdaGrad parameters - weight_decay: {:.4}, lr_decay: {:.3}, eps: {:.2e}, init_acc: {:.3}",
                    weight_decay.unwrap_or(0.0),
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
                    "AdaMax parameters - weight_decay: {:.4}, beta1: {:.3}, beta2: {:.3}, eps: {:.2e}",
                    weight_decay.unwrap_or(0.0),
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
                    "NAdam parameters - weight_decay: {:.4}, beta1: {:.3}, beta2: {:.3}, eps: {:.2e}, momentum_decay: {:.3}",
                    weight_decay.unwrap_or(0.0),
                    beta1,
                    beta2,
                    eps,
                    momentum_decay
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
                    "RAdam parameters - weight_decay: {:.4}, beta1: {:.3}, beta2: {:.3}, eps: {:.2e}",
                    weight_decay.unwrap_or(0.0),
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
            crate::config::training::OptimizerType::FracAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
                alpha,
                memory_window,
                step_size,
            } => {
                log::info!(
                    "Using FracAdam optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "FracAdam parameters - weight_decay: {:.4}, beta1: {:.3}, beta2: {:.3}, eps: {:.2e}",
                    weight_decay.map_or(0.0, |wd| wd),
                    beta1,
                    beta2,
                    eps
                );
                log::info!(
                    "FracAdam fractional parameters - alpha: {:.3}, memory_window: {}, step_size: {:.3}",
                    alpha,
                    memory_window,
                    step_size
                );

                let params = crate::optimization::ParamsFracAdam {
                    lr: learning_rate,
                    beta_1: *beta1,
                    beta_2: *beta2,
                    eps: *eps,
                    weight_decay: *weight_decay,
                    fractional: crate::optimization::FractionalConfig {
                        alpha: *alpha,
                        memory_window: *memory_window,
                        step_size: *step_size,
                    },
                };

                Ok(OptimizerWrapper::FracAdam(
                    crate::optimization::FracAdam::new(self.varmap.all_vars(), params).map_err(
                        |e| {
                            VangaError::ModelError(format!(
                                "FracAdam optimizer creation failed: {}",
                                e
                            ))
                        },
                    )?,
                ))
            }
            crate::config::training::OptimizerType::FracNAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
                momentum_decay,
                alpha,
                memory_window,
                step_size,
            } => {
                log::info!(
                    "Using FracNAdam optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "FracNAdam parameters - weight_decay: {:.4}, beta1: {:.3}, beta2: {:.3}, eps: {:.2e}",
                    weight_decay.map_or(0.0, |wd| wd),
                    beta1,
                    beta2,
                    eps
                );
                log::info!("FracNAdam momentum_decay: {:.4}", momentum_decay);
                log::info!(
                    "FracNAdam fractional parameters - alpha: {:.3}, memory_window: {}, step_size: {:.3}",
                    alpha,
                    memory_window,
                    step_size
                );

                let params = crate::optimization::ParamsFracNAdam {
                    lr: learning_rate,
                    beta_1: *beta1,
                    beta_2: *beta2,
                    eps: *eps,
                    weight_decay: *weight_decay,
                    momentum_decay: *momentum_decay,
                    fractional: crate::optimization::FractionalConfig {
                        alpha: *alpha,
                        memory_window: *memory_window,
                        step_size: *step_size,
                    },
                };

                Ok(OptimizerWrapper::FracNAdam(
                    crate::optimization::FracNAdam::new(self.varmap.all_vars(), params).map_err(
                        |e| {
                            VangaError::ModelError(format!(
                                "FracNAdam optimizer creation failed: {}",
                                e
                            ))
                        },
                    )?,
                ))
            }
            crate::config::training::OptimizerType::Prodigy {
                d_coef,
                growth_rate,
                beta1,
                beta2,
                eps,
                weight_decay,
                safeguard_warmup,
            } => {
                log::info!("🚀 Using Prodigy optimizer (ICLR 2024) - Learning-Rate-Free!");
                log::info!(
                    "   • Automatic LR adaptation: lr={:.1} (will auto-adjust)",
                    learning_rate
                );
                log::info!(
                    "   • D estimate coefficient: {:.3}, growth_rate: {}",
                    d_coef,
                    if growth_rate.is_infinite() {
                        "unlimited".to_string()
                    } else {
                        format!("{:.2}", growth_rate)
                    }
                );
                log::info!(
                    "   • Adam-like parameters: beta1={:.3}, beta2={:.3}, eps={:.2e}",
                    beta1,
                    beta2,
                    eps
                );
                log::info!(
                    "   • Weight decay: {:.4}, safeguard_warmup: {}",
                    weight_decay,
                    safeguard_warmup
                );
                log::info!("   • Paper: https://arxiv.org/abs/2306.06101");

                let params = crate::optimization::ParamsProdigy {
                    lr: learning_rate,
                    d_coef: *d_coef,
                    growth_rate: *growth_rate,
                    beta1: *beta1,
                    beta2: *beta2,
                    eps: *eps,
                    weight_decay: *weight_decay,
                    safeguard_warmup: *safeguard_warmup,
                };

                Ok(OptimizerWrapper::Prodigy(
                    crate::optimization::Prodigy::new(self.varmap.all_vars(), params).map_err(
                        |e| {
                            VangaError::ModelError(format!(
                                "Prodigy optimizer creation failed: {}",
                                e
                            ))
                        },
                    )?,
                ))
            }
            crate::config::training::OptimizerType::FracProdigy {
                beta1,
                beta2,
                eps,
                weight_decay,
                momentum_decay,
                d_coef,
                growth_rate,
                alpha,
                memory_window,
                step_size,
            } => {
                log::info!("🚀 Using FracProdigy optimizer - Fractional Memory + Automatic LR!");
                log::info!(
                    "   • Automatic LR adaptation: lr={:.1} (will auto-adjust)",
                    learning_rate
                );
                log::info!(
                    "   • Fractional memory: α={:.2}, window={}, step={:.1}",
                    alpha,
                    memory_window,
                    step_size
                );
                log::info!(
                    "   • Prodigy D-estimate: d_coef={:.3}, growth_rate={}",
                    d_coef,
                    if growth_rate.is_infinite() {
                        "unlimited".to_string()
                    } else {
                        format!("{:.2}", growth_rate)
                    }
                );
                log::info!(
                    "   • NAdam parameters: beta1={:.3}, beta2={:.3}, momentum_decay={:.4}, eps={:.2e}",
                    beta1,
                    beta2,
                    momentum_decay,
                    eps
                );
                log::info!(
                    "   • Weight decay: {}",
                    weight_decay.map_or("None".to_string(), |wd| format!("{:.4}", wd))
                );
                log::info!("   • Combines: FracNAdam memory + Prodigy automatic LR");

                let params = crate::optimization::ParamsFracProdigy {
                    lr: learning_rate,
                    beta1: *beta1,
                    beta2: *beta2,
                    eps: *eps,
                    weight_decay: *weight_decay,
                    momentum_decay: *momentum_decay,
                    d_coef: *d_coef,
                    growth_rate: *growth_rate,
                    fractional: crate::optimization::FractionalConfig {
                        alpha: *alpha,
                        memory_window: *memory_window,
                        step_size: *step_size,
                    },
                };

                Ok(OptimizerWrapper::FracProdigy(
                    crate::optimization::FracProdigy::new(self.varmap.all_vars(), params).map_err(
                        |e| {
                            VangaError::ModelError(format!(
                                "FracProdigy optimizer creation failed: {}",
                                e
                            ))
                        },
                    )?,
                ))
            }
        }
    }
    pub fn apply_xavier_initialization(&mut self) -> Result<()> {
        log::info!("🔧 Applying proper LSTM weight initialization (Xavier + Orthogonal)...");

        // Use the new comprehensive LSTM weight initialization
        crate::model::lstm::seeded_weights::SeededTensorUtils::apply_lstm_weight_initialization(
            &self.varmap,
            &self.device,
            self.seed,
        )?;

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
    ///   Train XGBoost phase with enhanced logging for XGBoost-only mode.
    ///
    ///   This method can be called independently for XGBoost-only training.
    pub async fn train_xgboost_phase(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        config: &crate::config::TrainingConfig,
        val_sequences: Option<&Array3<f64>>,
        val_targets: Option<&Array2<f64>>,
    ) -> Result<()> {
        // Store validation data for evaluation if provided
        if let Some(val_seq) = val_sequences {
            self.stored_val_sequences = Some(val_seq.clone());
        }
        if let Some(val_tgt) = val_targets {
            self.stored_val_targets = Some(val_tgt.clone());
        }

        // Log target context if available
        let target_info = if let Some((ref target_name, ref target_type)) = self.target_context {
            format!(" for [{}] ({:?})", target_name, target_type)
        } else {
            String::new()
        };

        log::info!(
            "🌲 Phase 2: XGBoost training on LSTM features{}",
            target_info
        );
        log::info!("📄 Following paper: XGBoost learns f(z) where z = LSTM hidden state");

        // Extract LSTM features for all training sequences
        // As per equation (8): z = h_n ∈ ℝ^k where k is hidden size
        let target_label = if let Some((ref name, _)) = self.target_context {
            format!(" [{}]", name)
        } else {
            String::new()
        };

        log::info!(
            "🔍{} Extracting latent vectors z = h_n from {} sequences (per equation 8)",
            target_label,
            sequences.shape()[0]
        );
        log::debug!("   • Input sequences shape: {:?}", sequences.shape());
        log::debug!(
            "   • Expected output: z ∈ ℝ^(N×k) where N={}, k=hidden_size",
            sequences.shape()[0]
        );
        let lstm_features = self.extract_all_lstm_features(sequences)?;

        log::info!(
            "📊 [LSTM] Latent vectors extracted: z ∈ ℝ^{:?}",
            lstm_features.shape()
        );

        // Add diagnostic for LSTM features
        log::info!("🔍 LSTM Feature Statistics:");
        if let Ok(mean) = lstm_features.mean(candle_core::D::Minus1) {
            if let Ok(mean_val) = mean.mean_all() {
                // Handle both F32 and F64 dtypes
                let mean_value = match lstm_features.dtype() {
                    candle_core::DType::F32 => mean_val.to_scalar::<f32>()? as f64,
                    candle_core::DType::F64 => mean_val.to_scalar::<f64>()?,
                    _ => {
                        log::warn!("Unexpected dtype for LSTM features");
                        0.0
                    }
                };
                log::info!("  Mean: {:.6}", mean_value);
            }
        }

        // Check feature variance using tensor operations (avoids materializing full CPU copy)
        let zero_variance_count = if lstm_features.dim(1)? > 0 {
            let feature_dim = lstm_features.dim(1)?;
            let mut zero_count = 0;
            for col_idx in 0..feature_dim {
                // Extract single column without full CPU copy
                let col = lstm_features
                    .narrow(1, col_idx, 1)?
                    .squeeze(1)?
                    .to_vec1::<f32>()?;
                // Calculate variance directly
                let mean: f32 = col.iter().sum::<f32>() / col.len() as f32;
                let variance: f32 =
                    col.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / col.len() as f32;
                if variance < 1e-10 {
                    zero_count += 1;
                }
            }
            zero_count
        } else {
            0
        };

        if zero_variance_count > 0 {
            log::warn!(
                "⚠️ {} out of {} LSTM features have near-zero variance!",
                zero_variance_count,
                lstm_features.dim(1)?
            );
        }
        log::debug!("   • z = h_n (final LSTM hidden state)");
        log::debug!("   • Architecture: {:?}", self.architecture);
        log::debug!("   • These are features, NOT predictions");

        // Convert targets to tensor
        let targets_tensor = self.convert_targets_to_tensor(targets)?;

        // Determine XGBoost objective and metric based on target type
        let mut xgb_config = config.model.xgboost.clone();

        // Use the actual LSTM feature dimension (hidden state size)
        let actual_feature_dim = lstm_features.dim(1)?;
        xgb_config.feature_dim = actual_feature_dim;
        log::info!(
            "📊 XGBoost feature_dim = {} (LSTM hidden state dimension)",
            actual_feature_dim
        );

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

        // Check if XGBoost model already exists and warn
        if self.xgboost_model.is_some() {
            log::warn!("⚠️ Existing XGBoost model will be replaced with new training");
            self.xgboost_model = None; // Clear old model
        }

        // Create XGBoost regressor
        let mut xgb_regressor =
            crate::model::xgboost::XGBoostRegressor::new(xgb_config, self.device.clone());

        // Generate validation LSTM features if validation data is available
        let val_lstm_features = if let Some(val_seqs) = val_sequences {
            log::info!("🎯 Generating validation LSTM features for XGBoost accuracy calculation");
            Some(self.extract_all_lstm_features(val_seqs)?)
        } else {
            None
        };

        // Convert validation targets to tensor if available
        let val_targets_tensor = if let Some(val_tgts) = val_targets {
            Some(self.convert_targets_to_tensor(val_tgts)?)
        } else {
            None
        };

        // Train XGBoost model on LSTM features as per paper
        // Equation (9): ŷ = f(z) = Σ f_m(z) where z is LSTM hidden state
        log::info!(
            "🎯 [XGBoost{}] Training regression model: ŷ = f(z)",
            target_label
        );
        log::info!("   • Input: LSTM latent vectors z ∈ ℝ^k (features)");
        log::info!("   • Target: True labels y (5-class categorical)");
        log::info!("   • Output: Predictions ŷ ∈ ℝ^(N×5)");
        log::info!("   • Ordinal-aware: Using trading penalty matrix for 5-class problems");

        // Pass validation data to XGBoost for proper accuracy calculation
        xgb_regressor.train(
            &lstm_features,
            &targets_tensor,
            val_lstm_features.as_ref(),
            val_targets_tensor.as_ref(),
        )?;

        // CRITICAL FIX: Calculate ordinal loss on XGBoost PREDICTIONS, not LSTM features
        // Following paper equations: z = h_n (eq 8), ŷ = f(z) (eq 9), loss on ŷ
        if targets.shape()[1] == 5 {
            log::info!("🎯 Calculating ordinal loss on XGBoost predictions ŷ = f(z) (per paper)");

            // Step 1: Get XGBoost predictions ŷ = f(z) from trained model
            let xgb_predictions = xgb_regressor.predict(&lstm_features)?;
            log::debug!(
                "📊 XGBoost predictions ŷ shape: {:?}",
                xgb_predictions.shape()
            );
            log::debug!("📊 LSTM features z shape: {:?}", lstm_features.shape());
            log::debug!("📊 Targets y shape: {:?}", targets_tensor.shape());

            // Step 2: Calculate ordinal loss on predictions ŷ (CORRECT approach)
            if let Some(backend) = xgb_regressor.get_backend() {
                let ordinal_loss =
                    backend.calculate_ordinal_loss(&xgb_predictions, &targets_tensor)?;
                log::info!("📊 XGBoost Ordinal Loss on ŷ = f(z): {:.4}", ordinal_loss);

                // Step 3: Log mathematical consistency verification
                log::info!("✅ Mathematical Framework Verified:");
                log::info!("   • LSTM extracts z = h_n ∈ ℝ^k (equation 8)");
                log::info!("   • XGBoost learns ŷ = f(z) = Σ f_m(z) (equation 9)");
                log::info!("   • Ordinal loss calculated on ŷ, not z (CORRECT)");

                // Step 4: Compare with LSTM-only ordinal loss for analysis
                let seq_tensor = self.convert_sequences_to_prediction_tensor(sequences)?;
                let lstm_only_predictions = self.forward(&seq_tensor, false)?;
                if let Some(backend) = xgb_regressor.get_backend() {
                    let lstm_ordinal_loss =
                        backend.calculate_ordinal_loss(&lstm_only_predictions, &targets_tensor)?;
                    log::info!(
                        "📊 LSTM-only Ordinal Loss (comparison): {:.4}",
                        lstm_ordinal_loss
                    );

                    let improvement = lstm_ordinal_loss - ordinal_loss;
                    if improvement > 0.0 {
                        log::info!(
                            "🎯 XGBoost improves ordinal loss by {:.4} ({:.1}%)",
                            improvement,
                            (improvement / lstm_ordinal_loss) * 100.0
                        );
                    } else {
                        log::warn!(
                            "⚠️ XGBoost ordinal loss is {:.4} higher than LSTM-only",
                            -improvement
                        );
                    }
                }
            }
        }

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

        // NEW: Evaluate hybrid model (LSTM+XGBoost) performance vs LSTM-only
        self.evaluate_hybrid_model_performance().await?;

        log::info!(
            "✅ XGBoost hybrid training completed successfully{}",
            target_label
        );
        Ok(())
    }

    /// Evaluate hybrid model (LSTM+XGBoost) performance and compare with LSTM-only
    async fn evaluate_hybrid_model_performance(&mut self) -> Result<()> {
        log::info!("🔍 Evaluating hybrid model (LSTM+XGBoost) performance...");

        if self.xgboost_model.is_none() {
            log::warn!("⚠️ XGBoost model not available for hybrid evaluation");
            return Ok(());
        }

        // Evaluate on validation data if available
        if let (Some(stored_val_seq), Some(stored_val_tgt)) =
            (&self.stored_val_sequences, &self.stored_val_targets)
        {
            log::info!(
                "📊 Evaluating hybrid model on validation data ({} samples)...",
                stored_val_seq.shape()[0]
            );

            // Clone data to avoid borrowing issues
            let val_seq_clone = stored_val_seq.clone();
            let val_tgt_clone = stored_val_tgt.clone();

            // Get LSTM-only predictions (current predict() method)
            let lstm_only_predictions = self.predict_lstm_only(&val_seq_clone).await?;

            // Get hybrid predictions (LSTM+XGBoost via predict() method which should use XGBoost when available)
            let hybrid_predictions = self.predict(&val_seq_clone).await?;

            // Calculate and log both sets of metrics
            self.log_comparison_metrics(
                "Validation",
                &lstm_only_predictions,
                &hybrid_predictions,
                &val_tgt_clone,
            )
            .await?;
        }

        // Evaluate on test data if available
        if self.stored_test_sequences.shape()[0] > 0 {
            log::info!(
                "📊 Evaluating hybrid model on test data ({} samples)...",
                self.stored_test_sequences.shape()[0]
            );

            // Clone data to avoid borrowing issues
            let test_seq_clone = self.stored_test_sequences.clone();
            let test_tgt_clone = self.stored_test_targets.clone();

            // Get LSTM-only predictions
            let lstm_only_predictions = self.predict_lstm_only(&test_seq_clone).await?;

            // Get hybrid predictions
            let hybrid_predictions = self.predict(&test_seq_clone).await?;

            // Calculate and log both sets of metrics
            self.log_comparison_metrics(
                "Test",
                &lstm_only_predictions,
                &hybrid_predictions,
                &test_tgt_clone,
            )
            .await?;
        }

        Ok(())
    }

    /// Predict using LSTM-only (bypass XGBoost for comparison)
    async fn predict_lstm_only(&mut self, sequences: &Array3<f64>) -> Result<Array2<f64>> {
        // Temporarily disable XGBoost for pure LSTM prediction
        let xgb_backup = self.xgboost_model.take();
        let result = self.predict(sequences).await;
        self.xgboost_model = xgb_backup; // Restore XGBoost model
        result
    }

    /// Log comparison metrics by calling the new extracted method
    async fn log_comparison_metrics(
        &mut self,
        data_type: &str,
        lstm_only_predictions: &Array2<f64>,
        hybrid_predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<()> {
        // Use the new method that takes predictions directly
        self.calculate_categorical_validation_metrics_from_predictions(
            lstm_only_predictions,
            targets,
            Some(&format!("LSTM-only {}", data_type)),
        )
        .await?;

        self.calculate_categorical_validation_metrics_from_predictions(
            hybrid_predictions,
            targets,
            Some(&format!("Hybrid {}", data_type)),
        )
        .await?;

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

    /// Apply gradient clipping and optimizer step in a unified, optimized method
    ///
    /// This method implements an OPTIMIZED gradient clipping approach that minimizes
    /// redundant backward passes while maintaining correctness.
    ///
    /// # Optimization Strategy:
    /// - When clipping is NOT needed: Single backward_step (optimal)
    /// - When clipping IS needed: Unfortunately requires 2 backward passes due to Candle's design
    /// - Monitoring overhead reduced by sampling (every 100 batches)
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer wrapper to apply updates with
    /// * `base_loss` - The loss tensor to compute gradients from
    /// * `clip_value` - Optional gradient clipping threshold
    /// * `epoch` - Current epoch (for logging)
    /// * `batch_idx` - Current batch index (for logging)
    ///
    /// # Returns
    /// The effective gradient norm after clipping (if applied)
    fn apply_gradient_clipping_and_step(
        &self,
        optimizer: &mut OptimizerWrapper,
        base_loss: &Tensor,
        clip_value: Option<f64>,
        epoch: usize,
        batch_idx: usize,
    ) -> Result<f64> {
        match clip_value {
            Some(threshold) => {
                // GRADIENT CLIPPING PATH
                // CRITICAL FIX: We must compute gradients ONCE, check norm, then either:
                // 1. Scale gradients and apply (if clipping needed)
                // 2. Apply gradients directly (if no clipping needed)
                // We CANNOT call backward() and then backward_step() - this causes accumulation!

                // Step 1: Compute gradients ONCE
                let grads = base_loss.backward()?;
                let grad_norm = self.calculate_gradstore_norm(&grads)?;

                if grad_norm > threshold {
                    // CLIPPING REQUIRED: Scale the loss and recompute
                    // CRITICAL: We must drop the first gradients before recomputing
                    drop(grads); // Explicitly drop to ensure no accumulation

                    let clip_ratio = threshold / grad_norm;
                    let clip_ratio_tensor = Tensor::new(clip_ratio as f32, &self.device)?;
                    let scaled_loss = base_loss.mul(&clip_ratio_tensor)?;

                    // Recompute with scaled loss (this is the ONLY backward pass that applies)
                    optimizer.backward_step(&scaled_loss)?;

                    // Log clipping activity (reduced frequency for performance)
                    if batch_idx.is_multiple_of(10) {
                        log::debug!(
                            "✂️ Gradient clipped: {:.4} → {:.4} (ratio: {:.4})",
                            grad_norm,
                            threshold,
                            clip_ratio
                        );
                    }

                    // First batch logging
                    if epoch == 0 && batch_idx == 0 {
                        log::info!(
                            "🔧 Gradient clipping active: threshold={:.3}, initial norm={:.3}",
                            threshold,
                            grad_norm
                        );
                    }

                    Ok(threshold)
                } else {
                    // NO CLIPPING NEEDED: Use the already-computed gradients
                    // CRITICAL FIX: Use step() with existing grads, NOT backward_step()
                    // backward_step() would compute gradients AGAIN, causing accumulation
                    optimizer.step(&grads)?;

                    // First batch logging
                    if epoch == 0 && batch_idx == 0 {
                        log::info!(
                            "🔧 Gradient clipping enabled: threshold={:.3}, initial norm={:.3} (no clipping needed)",
                            threshold, grad_norm
                        );
                    }

                    Ok(grad_norm)
                }
            }
            None => {
                // NO GRADIENT CLIPPING: Direct backward_step (optimal path)
                let grad_norm = if batch_idx.is_multiple_of(100) {
                    // Calculate gradient norm for monitoring
                    let grads = base_loss.backward()?;
                    let norm = self.calculate_gradstore_norm(&grads)?;

                    // Use the already-computed gradients
                    optimizer.step(&grads)?;

                    if batch_idx == 0 {
                        log::debug!("📊 Gradient monitoring enabled (every 100 batches)");
                    }
                    log::debug!("📊 Gradient norm: {:.6}", norm);

                    norm
                } else {
                    // Direct backward_step for most batches (optimal path)
                    optimizer.backward_step(base_loss)?;
                    1.0 // Return dummy value when not monitoring
                };

                Ok(grad_norm)
            }
        }
    }

    /// Simple bias correction calibration - calculate correction factors from validation data
    pub fn calibrate_simple_bias_correction(
        &mut self,
        val_predictions: &Array2<f64>,
        val_targets: &Array2<f64>,
    ) -> Result<()> {
        if val_predictions.nrows() < self.bias_correction_config.min_samples {
            log::debug!(
                "🔧 Skipping bias correction: insufficient validation samples ({} < {})",
                val_predictions.nrows(),
                self.bias_correction_config.min_samples
            );
            return Ok(());
        }

        if !self.bias_correction_config.enabled {
            log::debug!("🔧 Bias correction disabled in configuration");
            return Ok(());
        }

        log::info!(
            "🎯 Calibrating simple bias correction from {} validation samples",
            val_predictions.nrows()
        );

        // Validate input dimensions
        let num_classes = val_predictions.ncols();
        let target_classes = val_targets.ncols();

        if num_classes != 5 {
            log::error!("❌ Expected 5 classes in predictions, got {}", num_classes);
            return Err(VangaError::ModelError(format!(
                "Simple bias correction requires 5-class predictions, got {}",
                num_classes
            )));
        }

        if target_classes != 5 {
            log::error!("❌ Expected 5 classes in targets, got {}", target_classes);
            return Err(VangaError::ModelError(format!(
                "Simple bias correction requires 5-class targets, got {}",
                target_classes
            )));
        }

        if val_predictions.nrows() != val_targets.nrows() {
            log::error!(
                "❌ Predictions and targets have different number of samples: {} vs {}",
                val_predictions.nrows(),
                val_targets.nrows()
            );
            return Err(VangaError::ModelError(
                "Predictions and targets must have same number of samples".to_string(),
            ));
        }

        let mut predicted_frequencies = [0.0; 5];
        let mut actual_frequencies = [0.0; 5];

        // Calculate class frequencies
        for class_idx in 0..5 {
            predicted_frequencies[class_idx] =
                val_predictions.column(class_idx).mean().unwrap_or(0.0);
            actual_frequencies[class_idx] = val_targets.column(class_idx).mean().unwrap_or(0.0);
        }

        // Calculate correction factors with bounds from config
        let mut correction_factors = [1.0; 5];
        for class_idx in 0..5 {
            if predicted_frequencies[class_idx] > 0.001 {
                let raw_factor = actual_frequencies[class_idx] / predicted_frequencies[class_idx];
                // Apply smoothing if we already have factors
                let smoothed_factor = if self.bias_correction_factors.is_some() {
                    let existing_factor = self.bias_correction_factors.unwrap()[class_idx];
                    existing_factor * (1.0 - self.bias_correction_config.smoothing_factor)
                        + raw_factor * self.bias_correction_config.smoothing_factor
                } else {
                    raw_factor
                };
                // Apply bounds from config
                correction_factors[class_idx] = smoothed_factor
                    .max(self.bias_correction_config.correction_bounds[0])
                    .min(self.bias_correction_config.correction_bounds[1]);
            }
        }

        self.bias_correction_factors = Some(correction_factors);

        log::info!("✅ Bias correction factors: {:?}", correction_factors);
        log::debug!("   Predicted frequencies: {:?}", predicted_frequencies);
        log::debug!("   Actual frequencies: {:?}", actual_frequencies);

        Ok(())
    }

    /// POST-HOC ensemble calibration (temperature scaling) after training completes
    ///
    /// This implements proper post-hoc calibration as per research:
    /// - Guo et al. 2017: "On Calibration of Modern Neural Networks"
    /// - ICLR 2025: Temperature scaling is post-processing method
    ///
    /// Temperature scaling should ONLY be applied after training, never during training.
    /// This method:
    /// 1. Runs validation data through trained model to get logits
    /// 2. Converts logits to probabilities via softmax
    /// 3. Optimizes temperature on validation set to minimize NLL
    /// 4. Saves calibrated temperature for inference use
    pub fn calibrate_ensemble_post_training(
        &mut self,
        val_sequences: &Array3<f64>,
        val_targets: &Array2<f64>,
    ) -> Result<()> {
        if !self.bias_correction_config.use_ensemble_calibration {
            log::debug!("🔧 Ensemble calibration disabled in configuration");
            return Ok(());
        }

        if self.ensemble_calibrator.is_none() {
            log::warn!("⚠️ Ensemble calibrator not initialized, skipping post-hoc calibration");
            return Ok(());
        }

        let num_samples = val_sequences.shape()[0];
        if num_samples < self.bias_correction_config.min_samples {
            log::warn!(
                "⚠️ Insufficient validation samples for ensemble calibration: {} < {}",
                num_samples,
                self.bias_correction_config.min_samples
            );
            return Ok(());
        }

        log::info!(
            "🎯 POST-HOC ensemble calibration starting ({} validation samples)...",
            num_samples
        );
        log::info!(
            "   📚 Research: Temperature scaling is post-training calibration (Guo et al. 2017)"
        );

        // Run validation data through trained model to get predictions
        let batch_size = self.training_config.batch_size;
        let num_batches = (num_samples + batch_size - 1) / batch_size;
        let mut all_predictions = Vec::new();

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, num_samples);
            let actual_batch_size = end_idx - start_idx;

            let batch_sequences = val_sequences
                .slice(s![start_idx..end_idx, .., ..])
                .to_owned();
            let batch_targets = val_targets.slice(s![start_idx..end_idx, ..]).to_owned();

            // Convert to tensors
            let (input_tensor, _) =
                self.convert_sequences_to_tensors(&batch_sequences, &batch_targets)?;

            // Forward pass (inference mode - no dropout)
            let logits = self.forward(&input_tensor, false)?;

            // Apply softmax to get probabilities
            let probabilities = candle_nn::ops::softmax(&logits, 1)?;

            // Convert to ndarray
            let probs_data: Vec<f64> = probabilities
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|&v| v as f64)
                .collect();

            let probs_array =
                Array2::from_shape_vec((actual_batch_size, 5), probs_data).map_err(|e| {
                    VangaError::ModelError(format!("Failed to create probabilities array: {}", e))
                })?;

            all_predictions.push(probs_array);
        }

        // Concatenate all predictions
        let total_predictions = ndarray::concatenate(
            ndarray::Axis(0),
            &all_predictions
                .iter()
                .map(|arr| arr.view())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| {
            VangaError::ModelError(format!(
                "Failed to concatenate validation predictions: {}",
                e
            ))
        })?;

        // Convert targets to one-hot if needed
        let val_targets_one_hot = if val_targets.shape()[1] == 1 {
            let mut one_hot = ndarray::Array2::<f64>::zeros((num_samples, 5));
            for (i, class_idx) in val_targets.iter().enumerate() {
                let class_index = (*class_idx as usize).min(4);
                one_hot[[i, class_index]] = 1.0;
            }
            one_hot
        } else {
            val_targets.clone()
        };

        // Calibrate ensemble (optimize temperature on validation set)
        if let Some(ref mut ensemble_cal) = self.ensemble_calibrator {
            ensemble_cal.calibrate_from_validation(&total_predictions, &val_targets_one_hot)?;

            if ensemble_cal.is_calibrated {
                let metrics = ensemble_cal.get_calibration_metrics();
                log::info!("✅ POST-HOC calibration complete: {}", metrics.summary());
                log::info!(
                    "   🌡️  Temperature: {:.4} (optimized on validation set)",
                    ensemble_cal.temperature_scaling.temperature
                );
                log::info!(
                    "   📊 ECE: {:.6} (Expected Calibration Error)",
                    metrics.overall_ece
                );
            }
        }

        Ok(())
    }
}
