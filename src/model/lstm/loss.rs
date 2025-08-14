//! Loss calculation and validation metrics
//!
//! This module contains all loss functions, validation metrics,
//! and gradient-related calculations.

use super::config::{LSTMModel, TargetFormat};
use crate::targets::TargetType;
use crate::utils::error::{Result, VangaError};

use candle_core::Tensor;
use candle_nn;
use ndarray::{Array2, Array3};

/// Loss calculation mode for distinguishing between training and validation
/// Calculate Mean Squared Error between predictions and targets
///
/// **Core MSE Implementation**: Standard mathematical MSE calculation used throughout the system.
/// Pure function with no dependencies - can be called from anywhere.
///
/// # Arguments
/// * `predictions` - Model predictions as 2D array
/// * `targets` - Ground truth targets as 2D array
///
/// # Returns
/// * `f64` - MSE value, or `f64::INFINITY` on shape mismatch
pub fn calculate_mse(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    // CRITICAL FIX: Validate shapes before operations
    if predictions.shape() != targets.shape() {
        log::error!(
            "Shape mismatch in MSE calculation: predictions={:?}, targets={:?}",
            predictions.shape(),
            targets.shape()
        );
        return f64::INFINITY;
    }

    // Log input statistics for debugging
    log::debug!(
        "MSE calculation - Pred shape: {:?}, Target shape: {:?}, Pred mean: {:.6}, Target mean: {:.6}",
        predictions.shape(),
        targets.shape(),
        predictions.mean().unwrap_or(0.0),
        targets.mean().unwrap_or(0.0)
    );

    let diff = predictions - targets;
    let squared_diff = &diff * &diff;
    let mse_result = squared_diff.mean().unwrap_or(f64::INFINITY);

    log::debug!("📊 MSE Result: {:.6}", mse_result);
    mse_result
}

impl LSTMModel {
    /// Calculate MSE (Mean Squared Error) - delegates to core implementation
    ///
    /// **Delegates to**: `calculate_mse()` function for consistency.
    pub fn calculate_mse_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        calculate_mse(predictions, targets)
    }

    /// Calculate MAPE (Mean Absolute Percentage Error) for regression targets
    ///
    /// **STANDARD METHOD**: Used for continuous/regression targets where percentage error is meaningful.
    /// For categorical targets, use `calculate_categorical_mape()` instead.
    pub fn calculate_mape(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // CRITICAL FIX: Validate shapes before operations - SAME as original
        if predictions.shape() != targets.shape() {
            log::error!(
                "Shape mismatch in MAPE calculation: predictions={:?}, targets={:?}",
                predictions.shape(),
                targets.shape()
            );
            return f64::INFINITY;
        }

        let mut total_percentage_error = 0.0;
        let mut valid_samples = 0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let actual = targets[[i, j]];
                let predicted = predictions[[i, j]];

                // Avoid division by zero and very small values - SAME logic as original
                if actual.abs() > 1e-8 {
                    let percentage_error = ((actual - predicted).abs() / actual.abs()) * 100.0;
                    total_percentage_error += percentage_error;
                    valid_samples += 1;
                }
            }
        }

        if valid_samples > 0 {
            total_percentage_error / valid_samples as f64
        } else {
            f64::INFINITY
        }
    }

    /// Calculate MAPE for categorical/ordinal targets (Direction, PriceLevel, Volatility)
    ///
    /// **CATEGORICAL METHOD**: Used by all target types in validation metrics.
    /// Calculates percentage error relative to maximum possible class distance.
    ///
    /// **Formula**: `MAPE = (|predicted - actual| / max_class_value) * 100`
    ///
    /// **Examples**:
    /// - If max_class=4, predicting class 4 when actual is class 0 = 100% error
    /// - If max_class=4, predicting class 2 when actual is class 0 = 50% error
    /// - If max_class=4, predicting class 1 when actual is class 0 = 25% error
    ///
    /// Formula: MAPE = (|predicted - actual| / max_possible_error) * 100
    /// where max_possible_error = max_class_value (since min is always 0)
    fn calculate_categorical_mape(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // CRITICAL FIX: Validate shapes before operations
        if predictions.shape() != targets.shape() {
            log::error!(
                "Shape mismatch in categorical MAPE calculation: predictions={:?}, targets={:?}",
                predictions.shape(),
                targets.shape()
            );
            return f64::INFINITY;
        }

        // Find the maximum class value to determine the scale
        let max_target = targets.iter().fold(0.0f64, |acc, &x| acc.max(x));
        let max_prediction = predictions.iter().fold(0.0f64, |acc, &x| acc.max(x));
        let max_class_value = max_target.max(max_prediction);

        // If all values are 0 or max_class_value is 0, return 0% error
        if max_class_value <= 0.0 {
            return 0.0;
        }

        let mut total_percentage_error = 0.0;
        let mut total_samples = 0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let actual = targets[[i, j]];
                let predicted = predictions[[i, j]];

                // Calculate percentage error relative to maximum possible error
                let absolute_error = (actual - predicted).abs();
                let percentage_error = (absolute_error / max_class_value) * 100.0;

                total_percentage_error += percentage_error;
                total_samples += 1;
            }
        }

        if total_samples > 0 {
            total_percentage_error / total_samples as f64
        } else {
            0.0
        }
    }
    pub fn calculate_gradient_norm(&self, grads: &candle_core::backprop::GradStore) -> Result<f64> {
        let mut total_norm_squared = 0.0f64;
        let mut param_count = 0;

        // Get all variables from the VarMap (same approach as clip_gradients)
        let all_vars = self.varmap.all_vars();

        for var in all_vars.iter() {
            if let Some(grad) = grads.get(var) {
                total_norm_squared += self.calculate_tensor_norm_squared(grad)?;
                param_count += 1;

                log::trace!(
                    "Gradient norm for param {}: {:.6e}",
                    var.as_tensor()
                        .shape()
                        .dims()
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("x"),
                    self.calculate_tensor_norm_squared(grad)?.sqrt()
                );
            }
        }

        if param_count == 0 {
            return Err(VangaError::ModelError(
                "No gradients found in backward pass".to_string(),
            ));
        }

        let total_norm = total_norm_squared.sqrt();
        log::debug!(
            "🔍 Total gradient norm: {:.6e} across {} parameters",
            total_norm,
            param_count
        );

        Ok(total_norm)
    }

    /// Validate gradient norm for basic gradient flow issues (simplified version for backward_step usage)
    pub fn validate_gradient_norm(&self, effective_grad_norm: f64) -> Result<()> {
        // Check for NaN gradients
        if effective_grad_norm.is_nan() {
            return Err(VangaError::ModelError("🚨 NaN gradients detected! This indicates numerical instability in loss calculation or model architecture.".to_string()));
        }

        // Check for infinite gradients
        if effective_grad_norm.is_infinite() {
            return Err(VangaError::ModelError("🚨 Infinite gradients detected! This indicates exploding gradients - consider gradient clipping or lower learning rate.".to_string()));
        }

        // Check for zero gradients (no learning)
        if effective_grad_norm < 1e-12 {
            log::warn!(
                "⚠️ Very small gradient norm ({:.2e}) - model may not be learning effectively",
                effective_grad_norm
            );
        }

        // Check for exploding gradients
        if effective_grad_norm > 100.0 {
            log::warn!(
                "⚠️ Large gradient norm ({:.2e}) - consider gradient clipping or lower learning rate",
                effective_grad_norm
            );
        }

        log::debug!(
            "✅ Gradient norm validation passed - norm: {:.6e}",
            effective_grad_norm
        );
        Ok(())
    }

    /// Validate gradient flow to catch gradient issues early
    pub fn validate_gradient_flow(
        &self,
        grads: &candle_core::backprop::GradStore,
        effective_grad_norm: f64,
        original_grad_norm: f64,
    ) -> Result<bool> {
        // Check for NaN gradients (use effective norm)
        if effective_grad_norm.is_nan() {
            return Err(VangaError::ModelError("🚨 NaN gradients detected! This indicates numerical instability in loss calculation or model architecture.".to_string()));
        }

        // Check for infinite gradients (use effective norm)
        if effective_grad_norm.is_infinite() {
            return Err(VangaError::ModelError("🚨 Infinite gradients detected! This indicates exploding gradients - consider gradient clipping or lower learning rate.".to_string()));
        }

        // Check for zero gradients (no learning) - use effective norm
        if effective_grad_norm < 1e-12 {
            log::warn!(
                "⚠️ Very small effective gradient norm ({:.2e}) - model may not be learning effectively",
                effective_grad_norm
            );
            return Ok(false);
        }

        // Check for exploding gradients - NOW USES EFFECTIVE NORM (the actual training impact)
        if effective_grad_norm > 100.0 {
            if original_grad_norm != effective_grad_norm {
                // Clipping was applied but still too large
                log::warn!(
                    "⚠️ Large effective gradient norm ({:.2e}) after clipping from original ({:.2e}) - consider lower clipping threshold or learning rate",
                    effective_grad_norm,
                    original_grad_norm
                );
            } else {
                // No clipping was applied
                log::warn!(
                    "⚠️ Large gradient norm ({:.2e}) - consider gradient clipping",
                    effective_grad_norm
                );
            }
        } else if original_grad_norm > effective_grad_norm {
            // Clipping was successfully applied
            log::debug!(
                "✂️ Gradient clipping working: original={:.2e} -> effective={:.2e}",
                original_grad_norm,
                effective_grad_norm
            );
        }

        // Validate individual gradient tensors for NaN using sum approach
        let all_vars = self.varmap.all_vars();
        for var in all_vars.iter() {
            if let Some(grad) = grads.get(var) {
                // Check for NaN in individual gradients using sum approach with proper dtype handling
                let grad_sum = grad.sum_all()?;

                // Handle both F32 and F64 tensors (same pattern as calculate_tensor_norm_squared)
                let grad_sum_value: f64 = match grad_sum.dtype() {
                    candle_core::DType::F32 => {
                        let val: f32 = grad_sum.to_scalar().map_err(|e| {
                            VangaError::ModelError(format!("F32 gradient sum check failed: {}", e))
                        })?;
                        val as f64
                    }
                    candle_core::DType::F64 => grad_sum.to_scalar().map_err(|e| {
                        VangaError::ModelError(format!("F64 gradient sum check failed: {}", e))
                    })?,
                    other => {
                        return Err(VangaError::ModelError(format!(
                            "Unsupported gradient tensor dtype: {:?}. Expected F32 or F64.",
                            other
                        )));
                    }
                };

                if grad_sum_value.is_nan() {
                    return Err(VangaError::ModelError(format!(
                        "🚨 NaN detected in gradient for parameter with shape {:?}",
                        var.as_tensor().shape()
                    )));
                }
            }
        }

        log::debug!(
            "✅ Gradient flow validation passed - effective_norm: {:.6e}, original_norm: {:.6e}",
            effective_grad_norm,
            original_grad_norm
        );
        Ok(true)
    }

    /// Validate tensor shapes for loss calculation to catch configuration mismatches
    fn validate_tensor_shapes(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        _config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        let pred_shape = predictions.shape();
        let target_shape = targets.shape();

        // Basic shape validation
        if pred_shape.dims().len() != 2 {
            return Err(VangaError::ModelError(format!(
                "🚨 TENSOR SHAPE ERROR: Predictions must be 2D tensor, got shape {:?}",
                pred_shape
            )));
        }

        if target_shape.dims().len() != 2 {
            return Err(VangaError::ModelError(format!(
                "🚨 TENSOR SHAPE ERROR: Targets must be 2D tensor, got shape {:?}",
                target_shape
            )));
        }

        // Batch size consistency
        let pred_batch_size = pred_shape.dims()[0];
        let target_batch_size = target_shape.dims()[0];

        if pred_batch_size != target_batch_size {
            return Err(VangaError::ModelError(format!(
                "🚨 BATCH SIZE MISMATCH: Predictions batch size {} != targets batch size {}",
                pred_batch_size, target_batch_size
            )));
        }

        // Output size validation for classification targets
        let target_type = self.get_target_type()?;
        let pred_output_size = pred_shape.dims()[1];
        let expected_output_size = crate::config::model::NUM_CLASSES;

        // CRITICAL CHECK: This catches the main bug we're fixing
        if target_type == TargetType::PriceLevel
            && pred_output_size == 1
            && expected_output_size > 1
        {
            return Err(VangaError::ModelError(format!(
                "🚨 CRITICAL CONFIGURATION MISMATCH: PriceLevel classification enabled with {} bins but model output_size=1. This causes MSE fallback instead of CrossEntropy loss, breaking gradient flow. Fix: Set model output_size={}",
                expected_output_size, expected_output_size
            )));
        }

        if pred_output_size != expected_output_size {
            log::warn!(
                "⚠️ OUTPUT SIZE MISMATCH: Model output_size={} but expected {} for {:?} target. This may cause suboptimal loss calculation.",
                pred_output_size, expected_output_size, target_type
            );
        }

        log::debug!(
            "✅ Tensor shape validation passed: pred={:?}, target={:?}, expected_output={}",
            pred_shape,
            target_shape,
            expected_output_size
        );

        Ok(())
    }

    /// Set target context for this individual model
    /// This allows proper target type detection without assumptions based on output_size
    pub fn set_target_context(
        &mut self,
        target_name: String,
        target_type: crate::targets::TargetType,
    ) {
        self.target_context = Some((target_name.clone(), target_type));
        log::debug!(
            "🎯 Target context set: {} -> {:?}",
            target_name,
            target_type
        );
    }
}

// Implement From trait for Candle error conversion
impl From<candle_core::Error> for VangaError {
    fn from(err: candle_core::Error) -> Self {
        VangaError::ModelError(format!("Candle error: {}", err))
    }
}

impl LSTMModel {
    /// Get THIS model's target type - MUST be set during model creation
    /// No fallbacks, no assumptions - if not set, it's a programming error
    pub fn get_target_type(&self) -> Result<TargetType> {
        match &self.target_context {
            Some((_, target_type)) => Ok(*target_type),
            None => Err(VangaError::ModelError(
                "Target context not set for individual LSTM model. This is a programming error - models must be created with explicit target context.".to_string()
            ))
        }
    }

    /// Validate that model output_size matches expected target size
    /// This helps debug configuration issues
    fn validate_target_size_consistency(
        &self,
        config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        let target_type = self.get_target_type()?;
        let expected_size = self.get_target_size(target_type, config);
        let actual_size = self.config.output_size;

        if actual_size != expected_size {
            log::error!(
                "🚨 TARGET SIZE MISMATCH: Target {:?} expects {} outputs but model has {} outputs",
                target_type,
                expected_size,
                actual_size
            );
            return Err(VangaError::ModelError(format!(
                "Model output_size ({}) doesn't match expected size ({}) for target type {:?}",
                actual_size, expected_size, target_type
            )));
        }

        log::debug!(
            "✅ Target size validation passed: {:?} -> {} outputs",
            target_type,
            actual_size
        );
        Ok(())
    }

    /// Get target size for a specific target type based on configuration
    fn get_target_size(
        &self,
        _target_type: TargetType,
        _config: &crate::config::TrainingConfig,
    ) -> usize {
        crate::config::model::NUM_CLASSES // Use unified 5-class system
    }
}

impl LSTMModel {
    /// Detect target format from tensor shape and values
    pub fn detect_target_format(&self, targets: &Tensor) -> Result<TargetFormat> {
        let target_shape = targets.shape();
        let num_samples = target_shape.dims()[0];
        let num_outputs = target_shape.dims()[1];

        // If only 1 output dimension, it's likely raw class indices
        if num_outputs == 1 {
            return Ok(TargetFormat::RawClassIndices);
        }

        // Enhanced sampling: use more samples for better detection (reuse existing logic)
        let sample_size = std::cmp::min(50, num_samples); // Sample up to 50 rows instead of 10
        let target_tensor = targets.contiguous()?;
        let sample_data = target_tensor.to_vec2::<f32>()?;
        let mut one_hot_count = 0;
        let mut total_checked = 0;

        for row in sample_data.iter().take(sample_size) {
            total_checked += 1;

            // Count non-zero values
            let non_zero_count = row.iter().filter(|&&x| x > 0.0).count();
            let max_value = row.iter().fold(0.0f32, |a, &b| a.max(b));

            // One-hot pattern: exactly one 1.0, rest are 0.0
            if non_zero_count == 1 && max_value == 1.0 {
                one_hot_count += 1;
            }
        }

        // Validate detected format against expected target type if available
        let detected_format =
            if total_checked > 0 && one_hot_count as f32 / total_checked as f32 > 0.8 {
                TargetFormat::OneHot
            } else {
                TargetFormat::RawValues
            };

        // Cross-validate with target context if available (reuse existing target size logic)
        if let Some((_, target_type)) = &self.target_context {
            let expected_classes =
                self.get_target_size(*target_type, &crate::config::TrainingConfig::default());

            match detected_format {
                TargetFormat::OneHot => {
                    if num_outputs != expected_classes {
                        log::warn!(
                            "🚨 Target format mismatch: Detected OneHot with {} classes, but {:?} expects {} classes",
                            num_outputs, target_type, expected_classes
                        );
                    } else {
                        log::debug!(
                            "✅ Target format validation: OneHot format matches {:?} expected classes ({})",
                            target_type, expected_classes
                        );
                    }
                }
                TargetFormat::RawClassIndices => {
                    log::debug!(
                        "✅ Target format validation: RawClassIndices detected for {:?} (expected {} classes)",
                        target_type, expected_classes
                    );
                }
                _ => {
                    log::debug!(
                        "⚠️ Target format validation: {} format detected for {:?}",
                        match detected_format {
                            TargetFormat::RawValues => "RawValues",
                            _ => "Unknown",
                        },
                        target_type
                    );
                }
            }
        }

        Ok(detected_format)
    }

    /// Calculate categorical validation metrics for all categorical targets (PriceLevel, Direction, Volatility)
    pub async fn calculate_categorical_validation_metrics(
        &mut self,
        val_sequences: &Array3<f64>,
        val_targets: &Array2<f64>,
        batch_size: usize,
        epoch: usize,
        _config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        // Only calculate detailed metrics every 5 epochs to avoid overhead
        if epoch < 1 || epoch % 5 != 0 {
            return Ok(());
        }

        // Check if this is test data evaluation (when called with epoch=10 for final metrics)
        let is_test_evaluation = val_sequences.shape()[0] == self.stored_test_sequences.shape()[0];

        let total_val_samples = val_sequences.shape()[0];
        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();

        // Detect target format once for the entire validation set
        let (_input_tensor_sample, target_tensor_sample) = self.convert_sequences_to_tensors(
            &val_sequences.slice(ndarray::s![0..1, .., ..]).to_owned(),
            &val_targets.slice(ndarray::s![0..1, ..]).to_owned(),
        )?;
        let target_format = self.detect_target_format(&target_tensor_sample)?;

        log::debug!("🎯 Detected target format: {:?}", target_format);

        // SOLUTION: Process validation samples in random order to break hidden state patterns
        // This ensures that validation metrics reflect actual model performance, not hidden state artifacts
        let mut sample_indices: Vec<usize> = (0..total_val_samples).collect();

        // Shuffle indices to break any hidden state patterns (but use deterministic seed for reproducibility)
        // CRITICAL FIX: Use the same robust shuffling algorithm as training
        let seed_components = [
            epoch as u64,
            total_val_samples as u64,
            // Add current model state hash to make shuffle truly unique per validation run
            if let Some(first_var) = self.varmap.all_vars().first() {
                if let Ok(param_sum) = first_var.as_tensor().sum_all() {
                    if let Ok(param_val) = param_sum.to_scalar::<f32>() {
                        param_val as u64
                    } else {
                        0
                    }
                } else {
                    0
                }
            } else {
                0
            },
        ];
        let seed = crate::model::lstm::training::shuffle_indices_deterministic(
            &mut sample_indices,
            &seed_components,
        );

        log::debug!(
            "🔄 Processing {} validation samples in shuffled order for epoch {} (seed: {})",
            total_val_samples,
            epoch,
            seed
        );

        // Use smaller batch size for validation to ensure independence
        let validation_batch_size = std::cmp::min(batch_size, 8); // Even smaller batches

        // Process samples in shuffled order with small batches
        for chunk_start in (0..total_val_samples).step_by(validation_batch_size) {
            let chunk_end = std::cmp::min(chunk_start + validation_batch_size, total_val_samples);

            // Get shuffled indices for this chunk
            let chunk_indices = &sample_indices[chunk_start..chunk_end];

            // Create batch from shuffled indices
            let mut batch_sequences = Vec::new();
            let mut batch_targets = Vec::new();

            for &idx in chunk_indices {
                batch_sequences.push(val_sequences.slice(ndarray::s![idx, .., ..]).to_owned());
                batch_targets.push(val_targets.slice(ndarray::s![idx, ..]).to_owned());
            }

            // Convert to proper batch format
            let batch_size_actual = batch_sequences.len();
            let seq_len = batch_sequences[0].shape()[0];
            let features = batch_sequences[0].shape()[1];

            let mut batch_seq_array =
                ndarray::Array3::<f64>::zeros((batch_size_actual, seq_len, features));
            let mut batch_tgt_array =
                ndarray::Array2::<f64>::zeros((batch_size_actual, batch_targets[0].len()));

            for (i, (seq, tgt)) in batch_sequences.iter().zip(batch_targets.iter()).enumerate() {
                batch_seq_array
                    .slice_mut(ndarray::s![i, .., ..])
                    .assign(seq);
                batch_tgt_array.slice_mut(ndarray::s![i, ..]).assign(tgt);
            }

            let (input_tensor, target_tensor) =
                self.convert_sequences_to_tensors(&batch_seq_array, &batch_tgt_array)?;

            // Forward pass (inference mode for loss calculation)
            // Each small shuffled batch should produce independent predictions
            let predictions = self.forward(&input_tensor, false)?;

            // Convert predictions to class indices
            let pred_data = predictions.to_vec2::<f32>()?;
            let target_data = target_tensor.to_vec2::<f32>()?;

            for (pred_row, target_row) in pred_data.iter().zip(target_data.iter()) {
                // Get predicted class (argmax) with validation
                let predicted_class = pred_row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| {
                        // Validate predicted class against expected target type (reuse existing logic)
                        if let Some((_, target_type)) = &self.target_context {
                            let max_valid_class = self.get_target_size(*target_type, &crate::config::TrainingConfig::default()) - 1;
                            if idx > max_valid_class {
                                log::debug!(
                                    "⚠️ Model predicted class {} for {:?}, but max valid is {}. Using max valid class.",
                                    idx, target_type, max_valid_class
                                );
                                max_valid_class as i32
                            } else {
                                idx as i32
                            }
                        } else {
                            idx as i32
                        }
                    })
                    .unwrap_or(0);

                // Get true class - use detected format for proper extraction
                let true_class = match target_format {
                    TargetFormat::OneHot => {
                        // One-hot encoded - find max index
                        target_row
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(idx, _)| idx as i32)
                            .unwrap_or(0)
                    }
                    TargetFormat::RawClassIndices => {
                        // Raw class index format: [2.0] means class 2
                        let raw_value = target_row[0];
                        if raw_value >= 0.0 && raw_value.fract() == 0.0 {
                            raw_value as i32
                        } else {
                            log::warn!(
                                "Invalid class index in target: {}, defaulting to class 0",
                                raw_value
                            );
                            0
                        }
                    }
                    TargetFormat::RawValues | TargetFormat::Unknown => {
                        // For other formats, try to infer the best approach
                        if target_row.len() == 1 {
                            let raw_value = target_row[0];
                            if raw_value >= 0.0 && raw_value.fract() == 0.0 {
                                raw_value as i32
                            } else {
                                log::warn!(
                                    "Non-integer target value: {}, defaulting to class 0",
                                    raw_value
                                );
                                0
                            }
                        } else {
                            // Multi-value, assume one-hot
                            target_row
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(idx, _)| idx as i32)
                                .unwrap_or(0)
                        }
                    }
                };

                all_predictions.push(predicted_class);
                all_targets.push(true_class);
            }
        }

        // Calculate categorical metrics
        let accuracy = self.calculate_accuracy(&all_predictions, &all_targets);
        let (precision, recall, f1) =
            self.calculate_precision_recall_f1(&all_predictions, &all_targets);
        let class_distribution =
            self.analyze_prediction_distribution(&all_predictions, &all_targets);

        // DEBUG: Log first few predictions to verify they're changing between epochs
        if log::log_enabled!(log::Level::Debug) && !all_predictions.is_empty() {
            let sample_size = std::cmp::min(10, all_predictions.len());
            log::debug!(
                "🔍 Sample predictions (first {}): {:?}",
                sample_size,
                &all_predictions[..sample_size]
            );
            log::debug!(
                "🔍 Sample targets (first {}): {:?}",
                sample_size,
                &all_targets[..sample_size]
            );
        }

        // Calculate additional distance-based metrics for categorical data
        // Convert predictions and targets to Array2<f64> for MSE/MAPE calculation
        let pred_array = Array2::from_shape_vec(
            (all_predictions.len(), 1),
            all_predictions.iter().map(|&x| x as f64).collect(),
        )
        .unwrap_or_else(|_| Array2::zeros((0, 1)));

        let target_array = Array2::from_shape_vec(
            (all_targets.len(), 1),
            all_targets.iter().map(|&x| x as f64).collect(),
        )
        .unwrap_or_else(|_| Array2::zeros((0, 1)));

        let mse = if !pred_array.is_empty() && !target_array.is_empty() {
            self.calculate_mse_loss(&pred_array, &target_array)
        } else {
            f64::INFINITY
        };

        let categorical_mape = if !pred_array.is_empty() && !target_array.is_empty() {
            self.calculate_categorical_mape(&pred_array, &target_array)
        } else {
            f64::INFINITY
        };

        // Calculate quality metric for crypto winning percentage
        let quality = if !all_predictions.is_empty() && !all_targets.is_empty() {
            self.calculate_quality_metric(&all_predictions, &all_targets)
        } else {
            0.0
        };

        // Calculate error metric (percentage of predictions with distance > 2)
        let error_percentage = if !all_predictions.is_empty() && !all_targets.is_empty() {
            self.calculate_error_metric(&all_predictions, &all_targets)
        } else {
            0.0
        };

        // Log comprehensive categorical metrics with target-type aware interpretation
        let target_type_name = if let Some((_, target_type)) = &self.target_context {
            match target_type {
                TargetType::PriceLevel => "PriceLevel",
                TargetType::Direction => "Direction",
                TargetType::Volatility => "Volatility",
                TargetType::Sentiment => "Sentiment",
                TargetType::Volume => "Volume",
            }
        } else {
            "Unknown"
        };

        // Target-type aware metric interpretation (reuse existing target context)
        let metric_label = if is_test_evaluation {
            "Test"
        } else {
            "Validation"
        };

        log::info!(
            "📊 Metrics [{}] [{}]: Accuracy: {:.3}, Precision: {:.3}, Recall: {:.3}, F1: {:.3}, Quality: {:.1}%, Error: {:.1}%, MSE: {:.3}, MAPE: {:.2}%",
            target_type_name, metric_label, accuracy, precision, recall, f1, quality, error_percentage, mse, categorical_mape
        );

        log::debug!(
            "📈 Class Distribution [{}]: Pred: {:?}, True: {:?}",
            target_type_name,
            class_distribution.0,
            class_distribution.1
        );

        Ok(())
    }

    /// Calculate accuracy for categorical predictions with target-type aware validation
    fn calculate_accuracy(&self, predictions: &[i32], targets: &[i32]) -> f32 {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return 0.0;
        }

        // Get expected class range from target context (reuse existing logic)
        let max_valid_class = if let Some((_, target_type)) = &self.target_context {
            // Reuse existing get_target_size logic - 1 (since classes are 0-indexed)
            self.get_target_size(*target_type, &crate::config::TrainingConfig::default()) - 1
        } else {
            // Fallback: find max class in data if no context available
            let max_pred = predictions.iter().max().copied().unwrap_or(0);
            let max_target = targets.iter().max().copied().unwrap_or(0);
            max_pred.max(max_target) as usize
        };

        // Filter out invalid class indices and count valid pairs
        let valid_pairs: Vec<_> = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(pred, target)| {
                **pred >= 0
                    && **pred <= max_valid_class as i32
                    && **target >= 0
                    && **target <= max_valid_class as i32
            })
            .collect();

        if valid_pairs.is_empty() {
            log::warn!(
                "🚨 No valid class pairs found for accuracy calculation. Max valid class: {}, Pred range: [{}, {}], Target range: [{}, {}]",
                max_valid_class,
                predictions.iter().min().unwrap_or(&0),
                predictions.iter().max().unwrap_or(&0),
                targets.iter().min().unwrap_or(&0),
                targets.iter().max().unwrap_or(&0)
            );
            return 0.0;
        }

        // Log validation info if we filtered out invalid pairs
        if valid_pairs.len() != predictions.len() {
            log::debug!(
                "📊 Accuracy validation: {}/{} pairs valid for target type {:?} (max class: {})",
                valid_pairs.len(),
                predictions.len(),
                self.target_context.as_ref().map(|(_, t)| t),
                max_valid_class
            );
        }

        let correct = valid_pairs
            .iter()
            .filter(|(pred, target)| pred == target)
            .count();

        correct as f32 / valid_pairs.len() as f32
    }

    /// Calculate precision, recall, and F1 score (macro-averaged)
    fn calculate_precision_recall_f1(
        &self,
        predictions: &[i32],
        targets: &[i32],
    ) -> (f32, f32, f32) {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        // Find unique classes
        let mut classes = std::collections::HashSet::new();
        for &pred in predictions {
            classes.insert(pred);
        }
        for &target in targets {
            classes.insert(target);
        }

        let mut total_precision = 0.0;
        let mut total_recall = 0.0;
        let mut valid_classes = 0;

        for &class in &classes {
            let tp = predictions
                .iter()
                .zip(targets.iter())
                .filter(|(pred, target)| **pred == class && **target == class)
                .count() as f32;

            let fp = predictions
                .iter()
                .zip(targets.iter())
                .filter(|(pred, target)| **pred == class && **target != class)
                .count() as f32;

            let fn_count = predictions
                .iter()
                .zip(targets.iter())
                .filter(|(pred, target)| **pred != class && **target == class)
                .count() as f32;

            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_count > 0.0 {
                tp / (tp + fn_count)
            } else {
                0.0
            };

            if precision > 0.0 || recall > 0.0 {
                total_precision += precision;
                total_recall += recall;
                valid_classes += 1;
            }
        }

        let avg_precision = if valid_classes > 0 {
            total_precision / valid_classes as f32
        } else {
            0.0
        };
        let avg_recall = if valid_classes > 0 {
            total_recall / valid_classes as f32
        } else {
            0.0
        };
        let f1 = if avg_precision + avg_recall > 0.0 {
            2.0 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        } else {
            0.0
        };

        (avg_precision, avg_recall, f1)
    }

    /// Analyze prediction and target class distributions
    fn analyze_prediction_distribution(
        &self,
        predictions: &[i32],
        targets: &[i32],
    ) -> (Vec<usize>, Vec<usize>) {
        let max_class = predictions.iter().chain(targets.iter()).max().unwrap_or(&0);
        let num_classes = (*max_class + 1) as usize;

        let mut pred_counts = vec![0; num_classes];
        let mut target_counts = vec![0; num_classes];

        for &pred in predictions {
            if pred >= 0 && (pred as usize) < num_classes {
                pred_counts[pred as usize] += 1;
            }
        }

        for &target in targets {
            if target >= 0 && (target as usize) < num_classes {
                target_counts[target as usize] += 1;
            }
        }

        (pred_counts, target_counts)
    }

    /// Trading Quality Score (TQS) - Advanced profit-focused metric
    ///
    /// This metric evaluates predictions based on actual trading profitability potential.
    /// It rewards correct directional predictions and penalizes dangerous errors.
    ///
    /// Class System: 0=Strong Down, 1=Moderate Down, 2=Neutral, 3=Moderate Up, 4=Strong Up
    ///
    /// Scoring Philosophy:
    /// - Perfect predictions that capture full profit potential get maximum points
    /// - Conservative predictions that still profit get high scores
    /// - Missed opportunities (predicting neutral) get neutral scores
    /// - Wrong directional calls get zero or negative scores (losses)
    pub fn calculate_quality_metric(&self, predictions: &[i32], targets: &[i32]) -> f32 {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;
        let mut total_predictions = 0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            // Skip invalid predictions/targets
            if !(0..=4).contains(&pred) || !(0..=4).contains(&target) {
                continue;
            }

            total_predictions += 1;

            // Calculate the trading quality score for this prediction
            let score = self.calculate_trading_quality_score(pred, target);
            total_score += score;
        }

        if total_predictions == 0 {
            0.0
        } else {
            // Normalize to 0-100% scale
            // Maximum possible score is 1.0 per prediction, minimum is -0.5
            // We shift and scale to get a 0-100% range
            let avg_score = total_score / total_predictions as f32;
            // Convert from [-0.5, 1.0] range to [0, 100] range
            ((avg_score + 0.5) / 1.5) * 100.0
        }
    }

    /// Calculate Trading Quality Score for a single prediction
    ///
    /// Returns a score between -0.5 and 1.0 based on trading profitability
    fn calculate_trading_quality_score(&self, pred: i32, target: i32) -> f32 {
        // Class meanings: 0=Strong Down, 1=Moderate Down, 2=Neutral, 3=Moderate Up, 4=Strong Up

        match (pred, target) {
            // PERFECT PREDICTIONS (1.0 points) - Maximum profit potential
            (p, t) if p == t => 1.0, // Exact match

            // EXCELLENT PREDICTIONS (1.0 points) - Conservative but exceeded
            (1, 0) | (3, 4) => 1.0, // Predicted moderate, got strong (same direction)
            (0, 1) | (4, 3) => 0.9, // Predicted strong, got moderate (still profitable)

            // GOOD PREDICTIONS (0.7-0.8 points) - Correct direction, off by 1
            (0, 2) | (4, 2) => 0.7, // Predicted strong move, got neutral (false signal but no loss)
            (1, 2) | (3, 2) => 0.8, // Predicted moderate move, got neutral (minor false signal)

            // ACCEPTABLE PREDICTIONS (0.5-0.6 points) - Missed opportunities
            (2, 0) | (2, 4) => 0.5, // Predicted neutral, got strong move (big missed opportunity)
            (2, 1) | (2, 3) => 0.6, // Predicted neutral, got moderate move (missed opportunity)

            // POOR PREDICTIONS (0.0 points) - Wrong direction, moderate errors
            (0, 3) | (1, 3) => 0.0, // Predicted down, got moderate up
            (3, 0) | (3, 1) => 0.0, // Predicted moderate up, got down
            (1, 4) => 0.0,          // Predicted moderate down, got strong up
            (4, 1) => 0.0,          // Predicted strong up, got moderate down

            // CRITICAL ERRORS (-0.5 points) - Dangerous predictions causing losses
            (0, 4) => -0.5, // Predicted strong down, got strong up (maximum loss)
            (4, 0) => -0.5, // Predicted strong up, got strong down (maximum loss)

            // Default case for any remaining combinations
            _ => {
                // Calculate distance-based penalty for unmapped cases
                let distance = (pred - target).abs();
                match distance {
                    1 => 0.7,
                    2 => 0.3,
                    3 => -0.2,
                    4 => -0.5,
                    _ => 0.5,
                }
            }
        }
    }

    /// Critical Error Rate (CER) - Advanced loss-focused metric
    ///
    /// This metric specifically tracks predictions that would cause actual trading losses,
    /// distinguishing between critical errors (losses) and missed opportunities (no loss).
    ///
    /// Critical Errors (cause trading losses):
    /// - Wrong direction: Predicting up when market goes down or vice versa
    /// - Extreme errors: Distance >= 3 (e.g., Strong Up when Strong Down)
    /// - False breakouts: Predicting strong movement (0,4) when market is neutral
    ///
    /// Non-Critical (missed opportunities, not losses):
    /// - Predicting neutral when market moves (missed profit, not loss)
    /// - Being conservative (underestimating movement strength in correct direction)
    pub fn calculate_error_metric(&self, predictions: &[i32], targets: &[i32]) -> f32 {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return 0.0;
        }

        #[derive(Debug, PartialEq)]
        enum Direction {
            Down,
            Neutral,
            Up,
        }

        #[derive(Debug, Default)]
        struct ErrorDetails {
            directional_errors: usize,
            extreme_errors: usize,
            false_breakouts: usize,
        }

        let mut total_predictions = 0;
        let mut critical_errors = 0;
        let mut error_details = ErrorDetails::default();

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            // Skip invalid predictions/targets
            if !(0..=4).contains(&pred) || !(0..=4).contains(&target) {
                continue;
            }

            total_predictions += 1;

            // Check if this is a critical error
            // Class meanings: 0=Strong Down, 1=Moderate Down, 2=Neutral, 3=Moderate Up, 4=Strong Up

            // 1. Check for directional errors (most common trading losses)
            let pred_direction = if pred < 2 {
                Direction::Down
            } else if pred > 2 {
                Direction::Up
            } else {
                Direction::Neutral
            };

            let target_direction = if target < 2 {
                Direction::Down
            } else if target > 2 {
                Direction::Up
            } else {
                Direction::Neutral
            };

            let mut is_critical = false;

            // Critical: Predicting up when market goes down or vice versa
            if (pred_direction == Direction::Up && target_direction == Direction::Down)
                || (pred_direction == Direction::Down && target_direction == Direction::Up)
            {
                error_details.directional_errors += 1;
                is_critical = true;
            }

            // 2. Check for extreme errors (distance >= 3)
            let distance = (pred - target).abs();
            if distance >= 3 {
                error_details.extreme_errors += 1;
                is_critical = true;
            }

            // 3. Check for false breakout signals (predicting strong movement when neutral)
            if target == 2 && (pred == 0 || pred == 4) {
                // Predicted strong movement but market was neutral - dangerous false signal
                error_details.false_breakouts += 1;
                is_critical = true;
            }

            // 4. Check for panic signals (predicting neutral when strong movement)
            // This is NOT critical - it's a missed opportunity, not a loss
            // if pred == 2 && (target == 0 || target == 4) { is_critical = false; }

            if is_critical {
                critical_errors += 1;
            }
        }

        // Log detailed error breakdown for analysis
        if total_predictions > 0 && log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "🔍 Error Analysis - Directional: {}, Extreme: {}, False Breakouts: {}, Total Critical: {}/{}",
                error_details.directional_errors,
                error_details.extreme_errors,
                error_details.false_breakouts,
                critical_errors,
                total_predictions
            );
        }

        if total_predictions == 0 {
            0.0
        } else {
            (critical_errors as f32 / total_predictions as f32) * 100.0
        }
    }

    /// Calculate loss for single target type
    fn calculate_single_target_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        target_type: TargetType,
        _config: &crate::config::TrainingConfig,
        _is_validation: bool,
    ) -> Result<Tensor> {
        log::debug!(
            "🎯 Single target loss - Type: {:?}, Pred shape: {:?}, Target shape: {:?}",
            target_type,
            predictions.shape(),
            targets.shape()
        );

        // 🎯 TRADING-AWARE ORDINAL LOSS for 5-class market prediction
        // Optimized for profitability, not just mathematical accuracy
        // Classes: 0=VeryDown, 1=Down, 2=Sideways(safe), 3=Up, 4=VeryUp
        // Key: Wrong direction worse than wrong magnitude
        log::debug!(
            "🎯 Using Trading-Aware Ordinal Loss for {:?} (asymmetric penalties for market profitability)",
            target_type
        );

        let pred_contiguous = predictions.contiguous()?;
        let target_contiguous = targets.contiguous()?;
        let batch_size = pred_contiguous.dim(0)?;
        let num_classes = pred_contiguous.dim(1)?;

        log::debug!(
            "📐 Ordinal shapes: pred {:?}, target {:?}, batch_size: {}",
            predictions.shape(),
            targets.shape(),
            batch_size
        );

        // Validate 5-class system
        if num_classes != crate::config::model::NUM_CLASSES {
            return Err(VangaError::ModelError(format!(
                "{:?} target requires model output_size={}, got {}. Please update model configuration.",
                target_type, crate::config::model::NUM_CLASSES, num_classes
            )));
        }

        // Convert targets to class indices [batch_size]
        let target_indices = if target_contiguous.dim(1)? == 1 {
            target_contiguous.squeeze(1)?.contiguous()?
        } else {
            target_contiguous.contiguous()?
        };

        // Apply softmax to get probabilities
        let class_probs = candle_nn::ops::softmax(&pred_contiguous, 1)?; // [batch_size, num_classes]

        log::debug!(
            "📊 Ordinal Loss - Applied softmax to get class probabilities: {:?}",
            class_probs.shape()
        );

        // TRADING-AWARE ORDINAL LOSS: Optimized for Market Profitability
        //
        // Classes: 0=VeryDown, 1=Down, 2=Sideways, 3=Up, 4=VeryUp
        // Key principle: Wrong DIRECTION is worse than wrong MAGNITUDE
        //
        // Penalty Design:
        // 1. Being on wrong side of middle (2) is heavily penalized
        // 2. Missing opportunities (predict 2 when 0 or 4) is moderately penalized
        // 3. Wrong magnitude on same side is lightly penalized
        // 4. Asymmetric: Buying in crash (predict 4 when 0) worse than shorting rally
        //
        // This optimizes for:
        // - Capital preservation (avoid wrong direction trades)
        // - Opportunity capture (avoid sitting out big moves)
        // - Risk management (middle point = safe zone)

        // Get target indices as integers for indexing
        let target_indices_vec = target_indices.to_vec1::<f32>()?;
        let target_indices_int: Vec<usize> =
            target_indices_vec.iter().map(|&x| x as usize).collect();

        // Create one-hot encoded targets [batch_size, num_classes]
        let mut one_hot_targets = vec![0.0f32; batch_size * num_classes];
        for (batch_idx, &target_class) in target_indices_int.iter().enumerate() {
            if target_class < num_classes {
                one_hot_targets[batch_idx * num_classes + target_class] = 1.0;
            }
        }
        let one_hot_tensor = Tensor::from_vec(
            one_hot_targets,
            (batch_size, num_classes),
            pred_contiguous.device(),
        )?;

        // Standard cross-entropy: -sum(target * log(pred))
        let eps = 1e-7f32;
        let eps_tensor = Tensor::new(eps, pred_contiguous.device())?;
        let safe_probs = class_probs.broadcast_maximum(&eps_tensor)?;
        let log_probs = safe_probs.log()?;
        let ce_loss = one_hot_tensor
            .mul(&log_probs)?
            .sum(1)? // Sum across classes
            .neg()?; // Negate for loss

        // TRADING-AWARE PENALTY MATRIX
        // Rows = true class, Columns = predicted class
        // Higher values = worse mistakes for trading
        let penalty_matrix: [[f32; 5]; 5] = [
            // True = 0 (VeryDown)
            [0.0, 0.3, 0.5, 1.5, 2.0], // Predicting UP in crash is worst
            // True = 1 (Down)
            [0.3, 0.0, 0.3, 1.2, 1.8], // Still bad to predict UP
            // True = 2 (Sideways)
            [0.8, 0.4, 0.0, 0.4, 0.8], // Symmetric penalties for unnecessary trades
            // True = 3 (Up)
            [1.8, 1.2, 0.3, 0.0, 0.3], // Bad to predict DOWN
            // True = 4 (VeryUp)
            [2.0, 1.5, 0.5, 0.3, 0.0], // Shorting in rally is worst
        ];

        // Create ordinal weight tensor based on trading-aware penalties
        let mut ordinal_weights = vec![0.0f32; batch_size * num_classes];
        for (batch_idx, &target_class) in target_indices_int.iter().enumerate() {
            if target_class < num_classes {
                for pred_class in 0..num_classes {
                    let penalty = penalty_matrix[target_class][pred_class];
                    ordinal_weights[batch_idx * num_classes + pred_class] = penalty;
                }
            }
        }
        let ordinal_weight_tensor = Tensor::from_vec(
            ordinal_weights,
            (batch_size, num_classes),
            pred_contiguous.device(),
        )?;

        // Calculate ordinal penalty: sum of weighted probabilities
        // This penalizes based on trading impact, not just mathematical distance
        let ordinal_penalty = class_probs.mul(&ordinal_weight_tensor)?.sum(1)?; // Sum across classes for each sample

        // Combine losses: CE loss + λ * ordinal penalty
        // λ = 0.3 for trading (lower than pure ordinal since penalties are already strong)
        let lambda = 0.3f32;
        let lambda_tensor = Tensor::new(lambda, pred_contiguous.device())?;
        let scaled_penalty = ordinal_penalty.broadcast_mul(&lambda_tensor)?;
        let combined_loss = ce_loss.add(&scaled_penalty)?;

        // Take mean across batch
        let final_loss = combined_loss.mean_all()?;

        // Log the loss value for debugging
        let loss_value = final_loss.to_scalar::<f32>().unwrap_or(0.0);
        log::debug!(
            "✅ Trading-Aware Ordinal Loss: {:.6} for {:?} (asymmetric penalties for market profitability)",
            loss_value,
            target_type
        );

        // Additional debug: Check if loss is reasonable
        if loss_value > 10.0 {
            log::warn!(
                "⚠️ High trading loss detected: {:.6} - model may need lower learning rate",
                loss_value
            );
        } else if loss_value < 0.01 {
            log::warn!(
                "⚠️ Very low trading loss detected: {:.6} - model may be overfitting",
                loss_value
            );
        }

        Ok(final_loss)
    }

    /// Calculate multi-target loss with proper combination
    /// Calculate loss using configured loss function with target-aware logic
    pub fn calculate_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        config: &crate::config::TrainingConfig,
        is_validation: bool, // New parameter to distinguish training vs validation
    ) -> Result<Tensor> {
        // TENSOR SHAPE VALIDATION: Critical for catching configuration mismatches early
        self.validate_tensor_shapes(predictions, targets, config)?;

        // Log loss calculation context
        log::debug!(
            "🔍 LOSS CALCULATION - Pred shape: {:?}, Target shape: {:?}",
            predictions.shape(),
            targets.shape()
        );

        // Detect active target types from configuration
        let target_type = self.get_target_type()?;
        log::debug!("🎯 Target type: {:?}", target_type);
        log::debug!("🔧 Model output size: {}", self.config.output_size);

        // CRITICAL: Validate target size consistency
        if let Err(e) = self.validate_target_size_consistency(config) {
            log::error!("Target size validation failed: {}", e);
            // Continue with warning instead of failing - for debugging
        }

        // FIXED: Use single-target loss for individual models (they should always have correct size)
        // The validation above will catch and log any size mismatches
        log::debug!("📊 Using single target loss calculation");
        let loss_result = self.calculate_single_target_loss(
            predictions,
            targets,
            target_type,
            config,
            is_validation,
        )?;

        // Validate loss is not NaN or infinite
        let loss_value = loss_result.to_scalar::<f32>().unwrap_or(0.0);
        log::debug!(
            "🎯 FINAL LOSS - Value: {:.6}, Target type: {:?}",
            loss_value,
            target_type
        );

        Ok(loss_result)
    }
    pub fn calculate_tensor_norm_squared(&self, tensor: &Tensor) -> Result<f64> {
        let squared = tensor.sqr().map_err(|e| {
            VangaError::ModelError(format!(
                "Failed to square tensor for norm calculation: {}",
                e
            ))
        })?;

        let sum = squared.sum_all().map_err(|e| {
            VangaError::ModelError(format!("Failed to sum tensor for norm calculation: {}", e))
        })?;

        // Handle both F32 and F64 tensors
        let norm_squared: f64 = match sum.dtype() {
            candle_core::DType::F32 => {
                let val: f32 = sum.to_scalar().map_err(|e| {
                    VangaError::ModelError(format!("Failed to convert F32 norm to scalar: {}", e))
                })?;
                val as f64
            }
            candle_core::DType::F64 => sum.to_scalar().map_err(|e| {
                VangaError::ModelError(format!("Failed to convert F64 norm to scalar: {}", e))
            })?,
            _ => {
                return Err(VangaError::ModelError(format!(
                    "Unsupported tensor dtype for norm calculation: {:?}",
                    sum.dtype()
                )));
            }
        };

        Ok(norm_squared)
    }
    /// Get adaptive early stopping configuration based on target types
    pub fn get_adaptive_early_stopping_config(
        &self,
        target_types: &[TargetType],
        base_patience: u32,
        base_min_delta: f64,
    ) -> (u32, f64) {
        // Adjust thresholds based on target types
        let min_delta = if target_types.iter().all(|t| {
            matches!(
                t,
                TargetType::PriceLevel | TargetType::Direction | TargetType::Volatility
            )
        }) {
            // Categorical targets need smaller deltas
            base_min_delta * 0.1
        } else {
            // Mixed targets use intermediate threshold
            base_min_delta * 0.5
        };

        (base_patience, min_delta)
    }
}
