//! Loss calculation and validation metrics
//!
//! This module contains all loss functions, validation metrics,
//! and gradient-related calculations.

use super::config::{LSTMModel, TargetFormat};
use crate::targets::TargetType;
use crate::utils::error::{Result, VangaError};

use candle_core::Tensor;
use ndarray::{Array2, Array3};

impl LSTMModel {
    pub fn calculate_mse_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // CRITICAL FIX: Validate shapes before operations - SAME as original
        if predictions.shape() != targets.shape() {
            log::error!(
                "Shape mismatch in MSE calculation: predictions={:?}, targets={:?}",
                predictions.shape(),
                targets.shape()
            );
            return f64::INFINITY;
        }

        let diff = predictions - targets;
        let squared_diff = &diff * &diff;
        squared_diff.mean().unwrap_or(f64::INFINITY)
    }

    /// Calculate MAPE (Mean Absolute Percentage Error) for better understanding - EXACT same as original
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

    /// Calculate MAPE for categorical data (price levels)
    ///
    /// For ordinal categorical data (price levels 0,1,2,3,4), we calculate MAPE as
    /// the percentage of the maximum possible error. This gives meaningful results:
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
        config: &crate::config::TrainingConfig,
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
        let expected_output_size = match target_type {
            TargetType::PriceLevel => {
                if config.model.output_heads.price_levels.enabled {
                    config.model.output_heads.price_levels.bins as usize
                } else {
                    1 // Regression mode
                }
            }
            TargetType::Direction => 3,  // Up/Down/Sideways
            TargetType::Volatility => 3, // Low/Medium/High
        };

        // CRITICAL CHECK: This catches the main bug we're fixing
        if config.model.output_heads.price_levels.enabled
            && target_type == TargetType::PriceLevel
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
        target_type: TargetType,
        config: &crate::config::TrainingConfig,
    ) -> usize {
        match target_type {
            TargetType::PriceLevel => {
                if config.model.output_heads.price_levels.enabled {
                    config.model.output_heads.price_levels.bins as usize
                } else {
                    // Use output_size from LSTM config as fallback
                    self.config.output_size
                }
            }
            TargetType::Direction => 3,  // Up/Down/Sideways
            TargetType::Volatility => 3, // Low/Medium/High
        }
    }

    /// Calculate CrossEntropy loss for categorical targets with optional class weighting
    fn calculate_crossentropy_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        num_classes: usize,
    ) -> Result<Tensor> {
        log::debug!(
            "🔍 CrossEntropy Loss - Pred shape: {:?}, Target shape: {:?}, Classes: {}",
            predictions.shape(),
            targets.shape(),
            num_classes
        );

        // Handle different prediction shapes
        let logits = if predictions.dims().last() == Some(&num_classes) {
            // Already correct shape for multi-class
            predictions.clone()
        } else if predictions.dims().len() == 2 && predictions.dims()[1] == 1 {
            // CRITICAL BUG FIX: Single output with classification targets is a configuration error
            return Err(VangaError::ModelError(format!(
                "🚨 CONFIGURATION MISMATCH: Model has output_size=1 but classification target requires {} classes. This causes MSE fallback instead of CrossEntropy loss, breaking gradient flow. Fix: Set model output_size={} for classification targets.",
                num_classes, num_classes
            )));
        } else {
            return Err(VangaError::ModelError(format!(
                "Invalid prediction shape for CrossEntropy: {:?}, expected last dim = {}",
                predictions.shape(),
                num_classes
            )));
        };

        // Ensure targets are in correct format for CrossEntropy
        let target_shape = targets.shape();
        if target_shape.dims().len() != 2 {
            return Err(VangaError::ModelError(format!(
                "Invalid target shape for CrossEntropy: {:?}, expected 2D tensor",
                target_shape
            )));
        }

        // Use global class weights if available, otherwise calculate per-batch (fallback)
        let class_weights = if let Some((_target_name, target_type)) = &self.target_context {
            match target_type {
                TargetType::PriceLevel | TargetType::Direction | TargetType::Volatility => {
                    if let Some(ref global_weights) = self.global_class_weights {
                        log::debug!(
                            "🌍 Using global class weights for {:?}: {:?}",
                            target_type,
                            global_weights
                        );
                        Some(global_weights.clone())
                    } else {
                        log::debug!(
                            "⚠️ Global weights not available for {:?}, calculating per-batch",
                            target_type
                        );
                        self.calculate_class_weights_from_tensor(targets, num_classes)?
                    }
                }
            }
        } else {
            None
        };

        // Apply label smoothing for categorical targets
        let smoothed_targets = if let Some((_, target_type)) = &self.target_context {
            match target_type {
                TargetType::PriceLevel => {
                    // 10% smoothing for price levels (existing behavior)
                    self.apply_label_smoothing(targets, num_classes, 0.1)?
                }
                TargetType::Direction => {
                    // 5% smoothing for direction targets (less aggressive for 3-class)
                    self.apply_label_smoothing(targets, num_classes, 0.05)?
                }
                TargetType::Volatility => {
                    // 5% smoothing for volatility targets (less aggressive for 3-class)
                    self.apply_label_smoothing(targets, num_classes, 0.05)?
                }
            }
        } else {
            targets.clone()
        };

        // Check the smoothed targets shape to determine loss calculation path
        let smoothed_target_shape = smoothed_targets.shape();

        log::debug!(
            "🎯 Loss calculation: Original targets {:?} → Smoothed targets {:?}, Classes: {}",
            target_shape,
            smoothed_target_shape,
            num_classes
        );

        // For CrossEntropy, targets should be class indices (integers) or one-hot encoded
        let loss = if smoothed_target_shape.dims()[1] == 1 {
            log::debug!("📊 Using class indices path (no label smoothing applied)");
            // Targets are class indices - use proper CrossEntropy loss
            let target_indices = smoothed_targets.to_dtype(candle_core::DType::I64)?;

            if let Some(weights) = class_weights {
                log::debug!("⚖️ Applying class weights to indices");
                // Use weighted CrossEntropy for imbalanced classes
                self.calculate_weighted_crossentropy_loss(
                    &logits,
                    &target_indices.squeeze(1)?,
                    &weights,
                )?
            } else {
                log::debug!("📈 Using standard CrossEntropy for indices");
                // Use standard CrossEntropy loss
                candle_nn::loss::cross_entropy(&logits, &target_indices.squeeze(1)?)?
            }
        } else if smoothed_target_shape.dims()[1] == num_classes {
            log::debug!("🎯 Using one-hot path (label smoothing applied)");
            // Targets are one-hot encoded (from label smoothing) - use soft CrossEntropy
            let log_softmax =
                candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?.contiguous()?;

            // For one-hot targets with class weights, we need to apply weights differently
            if let Some(weights) = class_weights {
                log::debug!("⚖️ Applying class weights to one-hot targets");
                // Apply class weights to one-hot encoded targets
                self.calculate_weighted_soft_crossentropy_loss(
                    &logits,
                    &smoothed_targets,
                    &weights,
                )?
            } else {
                log::debug!("📈 Using standard soft CrossEntropy for one-hot");
                // Standard soft CrossEntropy for one-hot targets - ensure all tensors are contiguous
                let smoothed_contiguous = smoothed_targets.contiguous()?;
                let loss = smoothed_contiguous
                    .mul(&log_softmax)?
                    .contiguous()?
                    .sum(candle_core::D::Minus1)?
                    .contiguous()?;
                loss.neg()?.mean_all()?
            }
        } else {
            return Err(VangaError::ModelError(format!(
                "Target dimension mismatch: got {}, expected 1 (indices) or {} (one-hot)",
                smoothed_target_shape.dims()[1],
                num_classes
            )));
        };

        let loss_value = loss.to_scalar::<f32>().unwrap_or(0.0);

        log::debug!("🎯 CrossEntropy Loss: {:.6}", loss_value);
        Ok(loss)
    }

    /// Calculate global class weights from entire training dataset
    /// This ensures consistent loss calculation across all batches
    pub fn calculate_global_class_weights(
        &mut self,
        train_targets: &Array2<f64>,
        num_classes: usize,
        provided_weights: Option<Vec<f32>>,
    ) -> Result<()> {
        // Calculate for all categorical targets: PriceLevel, Direction, and Volatility
        if let Some((_, target_type)) = &self.target_context {
            match target_type {
                TargetType::PriceLevel => {
                    log::debug!(
                        "🎯 Calculating global class weights for PriceLevel target with {} classes",
                        num_classes
                    );
                }
                TargetType::Direction => {
                    log::debug!(
                        "🎯 Calculating global class weights for Direction target (3 classes: Down=0, Sideways=1, Up=2)"
                    );
                }
                TargetType::Volatility => {
                    log::debug!(
                        "🎯 Calculating global class weights for Volatility target (3 classes: Low=0, Medium=1, High=2)"
                    );
                }
            }
        } else {
            log::debug!("🎯 No target context set, skipping global class weights");
            self.global_class_weights = None;
            return Ok(());
        }

        // Check if pre-calculated weights are provided (for per-window class weights)
        if let Some(weights) = provided_weights {
            log::info!(
                "🎯 Using provided per-window class weights for {:?}: {:?}",
                self.target_context.as_ref().map(|(_, t)| t),
                weights
            );
            self.global_class_weights = Some(weights);
            return Ok(());
        }

        // Convert to tensor for consistent processing - ensure F32 dtype
        let targets_f32: Vec<f32> = train_targets
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x| x as f32)
            .collect();
        let targets_tensor = Tensor::from_slice(&targets_f32, train_targets.dim(), &self.device)?;

        // Calculate global class weights from entire training dataset
        let weights = self.calculate_class_weights_from_tensor(&targets_tensor, num_classes)?;

        if let Some(weights) = weights {
            log::info!(
                "🌍 Global class weights calculated from {} training samples for {:?}: {:?}",
                train_targets.shape()[0],
                self.target_context.as_ref().map(|(_, t)| t),
                weights
            );
            self.global_class_weights = Some(weights);
        } else {
            log::warn!("⚠️ Failed to calculate global class weights, using per-batch calculation");
            self.global_class_weights = None;
        }

        Ok(())
    }

    /// Calculate class weights for imbalanced datasets (helper method)
    fn calculate_class_weights_from_tensor(
        &self,
        targets: &Tensor,
        num_classes: usize,
    ) -> Result<Option<Vec<f32>>> {
        // Extract target values to calculate class distribution
        let target_data = targets.to_vec2::<f32>()?;
        let mut class_counts = vec![0usize; num_classes];
        let mut total_samples = 0;

        // Count class occurrences
        for row in &target_data {
            if let Some(&target_val) = row.first() {
                let class_idx = target_val as usize;
                if class_idx < num_classes {
                    class_counts[class_idx] += 1;
                    total_samples += 1;
                }
            }
        }

        if total_samples == 0 {
            return Ok(None);
        }

        // Calculate inverse frequency weights
        let mut weights = Vec::new();
        let mut max_weight = 0.0f32;

        for &count in &class_counts {
            if count > 0 {
                let weight = total_samples as f32 / (num_classes as f32 * count as f32);
                weights.push(weight);
                max_weight = max_weight.max(weight);
            } else {
                // Handle empty classes with high weight
                weights.push(max_weight * 2.0);
            }
        }

        // Normalize weights to prevent extreme values
        let weight_sum: f32 = weights.iter().sum();
        if weight_sum > 0.0 {
            for weight in &mut weights {
                *weight = (*weight / weight_sum) * num_classes as f32;
                *weight = weight.clamp(0.1, 10.0); // Clamp to reasonable range
            }
        }

        log::debug!(
            "📊 Class weights calculated: {:?} (from counts: {:?})",
            weights,
            class_counts
        );

        Ok(Some(weights))
    }

    /// Calculate weighted CrossEntropy loss for imbalanced classes
    fn calculate_weighted_crossentropy_loss(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        class_weights: &[f32],
    ) -> Result<Tensor> {
        // Calculate standard CrossEntropy loss per sample
        let log_softmax =
            candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)?.contiguous()?;

        // Validate tensor dimensions
        let batch_size = targets.dim(0)?;
        let logits_batch_size = logits.dim(0)?;
        let num_classes = class_weights.len();

        if batch_size != logits_batch_size {
            return Err(VangaError::ModelError(format!(
                "Batch size mismatch: targets {} vs logits {}",
                batch_size, logits_batch_size
            )));
        }

        let mut weighted_losses = Vec::with_capacity(batch_size);
        let target_data = targets.contiguous()?.to_vec1::<i64>()?;
        let log_softmax_data = log_softmax.to_vec2::<f32>()?;

        // Validate data consistency
        if target_data.len() != batch_size {
            return Err(VangaError::ModelError(format!(
                "Target data length {} doesn't match batch size {}",
                target_data.len(),
                batch_size
            )));
        }

        if log_softmax_data.len() != batch_size {
            return Err(VangaError::ModelError(format!(
                "Log softmax data length {} doesn't match batch size {}",
                log_softmax_data.len(),
                batch_size
            )));
        }

        for (i, &target_class) in target_data.iter().enumerate() {
            let class_idx = target_class as usize;
            if class_idx < num_classes {
                let log_prob = log_softmax_data[i][class_idx];
                let weight = class_weights[class_idx];
                let weighted_loss = -log_prob * weight;
                weighted_losses.push(weighted_loss);
            } else {
                log::warn!(
                    "Invalid class index {} >= {}, skipping sample {}",
                    class_idx,
                    num_classes,
                    i
                );
            }
        }

        if weighted_losses.is_empty() {
            return Err(VangaError::ModelError(
                "No valid samples for weighted loss calculation".to_string(),
            ));
        }

        // Convert back to tensor and calculate mean
        let loss_values = weighted_losses.clone(); // Clone before move
        let loss_tensor = Tensor::from_vec(weighted_losses, (loss_values.len(),), logits.device())?
            .contiguous()?;
        let mean_loss = loss_tensor.mean_all()?;

        log::debug!(
            "⚖️ Weighted CrossEntropy: {:.6} (vs unweighted: {:.6}) for {} samples",
            mean_loss.to_scalar::<f32>().unwrap_or(0.0),
            candle_nn::loss::cross_entropy(logits, targets)?
                .to_scalar::<f32>()
                .unwrap_or(0.0),
            batch_size
        );

        Ok(mean_loss)
    }

    /// Calculate weighted soft CrossEntropy loss for one-hot encoded targets
    fn calculate_weighted_soft_crossentropy_loss(
        &self,
        logits: &Tensor,
        one_hot_targets: &Tensor,
        class_weights: &[f32],
    ) -> Result<Tensor> {
        // Ensure ALL input tensors are contiguous from the start
        let logits_contiguous = logits.contiguous()?;
        let targets_contiguous = one_hot_targets.contiguous()?;

        let log_softmax = candle_nn::ops::log_softmax(&logits_contiguous, candle_core::D::Minus1)?
            .contiguous()?;

        // Validate tensor dimensions
        let batch_size = targets_contiguous.dim(0)?;
        let num_classes = class_weights.len();

        if targets_contiguous.dim(1)? != num_classes {
            return Err(VangaError::ModelError(format!(
                "One-hot targets dimension {} doesn't match class weights {}",
                targets_contiguous.dim(1)?,
                num_classes
            )));
        }

        log::debug!(
            "🔍 Weighted soft CrossEntropy shapes: targets {:?}, logits {:?}, weights len {}",
            targets_contiguous.shape(),
            logits_contiguous.shape(),
            num_classes
        );

        // Create weight tensor with shape [1, num_classes] and ensure contiguous
        let weight_tensor = Tensor::from_vec(
            class_weights.to_vec(),
            (1, num_classes),
            logits_contiguous.device(),
        )?
        .contiguous()?;

        log::debug!(
            "🔍 Broadcasting shapes: targets {:?} × weights {:?}",
            targets_contiguous.shape(),
            weight_tensor.shape()
        );

        // Use broadcast_as to explicitly match tensor shapes before multiplication
        // Broadcasting: [1, num_classes] -> [batch_size, num_classes]
        let weight_tensor_broadcast = weight_tensor.broadcast_as(targets_contiguous.shape())?;

        log::debug!(
            "🔍 After broadcast_as: targets {:?} × weights {:?}",
            targets_contiguous.shape(),
            weight_tensor_broadcast.shape()
        );

        // Now multiply tensors with matching shapes and ensure result is contiguous
        let weighted_targets = targets_contiguous
            .mul(&weight_tensor_broadcast)?
            .contiguous()?;

        // Calculate weighted soft CrossEntropy loss - ensure all intermediate results are contiguous
        let weighted_log_loss = weighted_targets.mul(&log_softmax)?.contiguous()?;
        let loss_per_sample = weighted_log_loss
            .sum(candle_core::D::Minus1)?
            .contiguous()?;
        let mean_loss = loss_per_sample.neg()?.mean_all()?.contiguous()?;

        log::debug!(
            "⚖️ Weighted Soft CrossEntropy: {:.6} for {} samples with {} classes",
            mean_loss.to_scalar::<f32>().unwrap_or(0.0),
            batch_size,
            num_classes
        );

        Ok(mean_loss)
    }

    /// Apply label smoothing to reduce overconfidence in categorical predictions
    fn apply_label_smoothing(
        &self,
        targets: &Tensor,
        num_classes: usize,
        smoothing: f32,
    ) -> Result<Tensor> {
        let target_shape = targets.shape();

        if target_shape.dims()[1] == 1 {
            // Convert class indices to smoothed one-hot encoding
            let batch_size = target_shape.dims()[0];
            let target_data = targets.to_vec2::<f32>()?;

            let mut smoothed_data = Vec::new();

            for row in &target_data {
                if let Some(&target_class) = row.first() {
                    let class_idx = target_class as usize;

                    // Create smoothed one-hot vector
                    let mut one_hot = vec![smoothing / (num_classes - 1) as f32; num_classes];
                    if class_idx < num_classes {
                        one_hot[class_idx] = 1.0 - smoothing;
                    }

                    smoothed_data.extend(one_hot);
                }
            }

            let smoothed_tensor =
                Tensor::from_vec(smoothed_data, (batch_size, num_classes), targets.device())?
                    .contiguous()?; // Ensure contiguity

            log::debug!(
                "🎯 Label smoothing applied: {:.1}% smoothing for {} classes",
                smoothing * 100.0,
                num_classes
            );

            Ok(smoothed_tensor)
        } else if target_shape.dims()[1] == num_classes {
            // Already one-hot encoded - apply smoothing
            let uniform_dist = smoothing / num_classes as f32;

            // Ensure ALL intermediate tensors are contiguous
            let targets_contiguous = targets.contiguous()?;
            let scale_tensor =
                Tensor::from_slice(&[1.0 - smoothing], (1,), targets.device())?.contiguous()?;
            let uniform_tensor =
                Tensor::from_slice(&[uniform_dist], (1,), targets.device())?.contiguous()?;

            let scaled = targets_contiguous.mul(&scale_tensor)?.contiguous()?;
            let smoothed = scaled.add(&uniform_tensor)?.contiguous()?;

            log::debug!(
                "🎯 Label smoothing applied to one-hot targets: {:.1}% smoothing",
                smoothing * 100.0
            );

            Ok(smoothed)
        } else {
            // Invalid target format - return original
            log::warn!(
                "⚠️ Cannot apply label smoothing to targets with shape: {:?}",
                target_shape
            );
            Ok(targets.clone())
        }
    }

    /// Detect target format based on tensor shape and values
    fn detect_target_format(&self, target_tensor: &Tensor) -> Result<TargetFormat> {
        let shape = target_tensor.shape();
        let dims = shape.dims();
        if dims.len() != 2 {
            return Ok(TargetFormat::Unknown);
        }

        let num_outputs = dims[1];

        // If only 1 output dimension, it's likely raw class indices
        if num_outputs == 1 {
            return Ok(TargetFormat::RawClassIndices);
        }

        // If multiple outputs, check if it looks like one-hot encoding
        // Sample a few rows to check the pattern
        let sample_data = target_tensor.to_vec2::<f32>()?;
        let mut one_hot_count = 0;
        let mut total_checked = 0;

        for row in sample_data.iter().take(10) {
            // Check first 10 rows
            total_checked += 1;

            // Count non-zero values
            let non_zero_count = row.iter().filter(|&&x| x > 0.0).count();
            let max_value = row.iter().fold(0.0f32, |a, &b| a.max(b));

            // One-hot pattern: exactly one 1.0, rest are 0.0
            if non_zero_count == 1 && max_value == 1.0 {
                one_hot_count += 1;
            }
        }

        // If most samples follow one-hot pattern, classify as one-hot
        if total_checked > 0 && one_hot_count as f32 / total_checked as f32 > 0.8 {
            Ok(TargetFormat::OneHot)
        } else {
            Ok(TargetFormat::RawValues)
        }
    }

    /// Calculate categorical validation metrics for price level targets
    pub async fn calculate_categorical_validation_metrics(
        &self,
        val_sequences: &Array3<f64>,
        val_targets: &Array2<f64>,
        _batch_size: usize,
        epoch: usize,
        _config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        // Only calculate detailed metrics every 10 epochs to avoid overhead
        if epoch % 10 != 0 {
            return Ok(());
        }

        let total_val_samples = val_sequences.shape()[0];
        let validation_batch_size = 64; // Fixed batch size for validation metrics
        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();

        // Detect target format once for the entire validation set
        let (_input_tensor_sample, target_tensor_sample) = self.convert_sequences_to_tensors(
            &val_sequences.slice(ndarray::s![0..1, .., ..]).to_owned(),
            &val_targets.slice(ndarray::s![0..1, ..]).to_owned(),
        )?;
        let target_format = self.detect_target_format(&target_tensor_sample)?;

        log::debug!("🎯 Detected target format: {:?}", target_format);

        // Collect all predictions and targets
        for batch_start in (0..total_val_samples).step_by(validation_batch_size) {
            let batch_end = std::cmp::min(batch_start + validation_batch_size, total_val_samples);

            let batch_sequences = val_sequences
                .slice(ndarray::s![batch_start..batch_end, .., ..])
                .to_owned();
            let batch_targets = val_targets
                .slice(ndarray::s![batch_start..batch_end, ..])
                .to_owned();

            let (input_tensor, target_tensor) =
                self.convert_sequences_to_tensors(&batch_sequences, &batch_targets)?;

            let predictions = self.forward(&input_tensor)?;

            // Convert predictions to class indices
            let pred_data = predictions.to_vec2::<f32>()?;
            let target_data = target_tensor.to_vec2::<f32>()?;

            for (pred_row, target_row) in pred_data.iter().zip(target_data.iter()) {
                // Get predicted class (argmax)
                let predicted_class = pred_row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as i32)
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

        // Debug logging for first few samples to verify target extraction
        if epoch == 10 {
            log::debug!("🔍 Target extraction verification (first 5 samples):");
            for i in 0..std::cmp::min(5, all_predictions.len()) {
                log::debug!(
                    "  Sample {}: Predicted={}, True={}",
                    i,
                    all_predictions[i],
                    all_targets[i]
                );
            }
        }

        // Log comprehensive categorical metrics
        log::info!(
            "📊 Categorical Metrics [Epoch {}]: Accuracy: {:.3}, Precision: {:.3}, Recall: {:.3}, F1: {:.3}, MSE: {:.3}, MAPE: {:.2}%",
            epoch, accuracy, precision, recall, f1, mse, categorical_mape
        );

        log::debug!(
            "📈 Class Distribution: Pred: {:?}, True: {:?}",
            class_distribution.0,
            class_distribution.1
        );

        Ok(())
    }

    /// Calculate accuracy for categorical predictions
    fn calculate_accuracy(&self, predictions: &[i32], targets: &[i32]) -> f32 {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return 0.0;
        }

        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(pred, target)| pred == target)
            .count();

        correct as f32 / predictions.len() as f32
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

    /// Calculate loss for single target type
    fn calculate_single_target_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        target_type: TargetType,
        config: &crate::config::TrainingConfig,
    ) -> Result<Tensor> {
        log::debug!(
            "🎯 Single target loss - Type: {:?}, Pred shape: {:?}, Target shape: {:?}",
            target_type,
            predictions.shape(),
            targets.shape()
        );

        match target_type {
            TargetType::PriceLevel => {
                if config.model.output_heads.price_levels.enabled {
                    // CrossEntropy for categorical price levels
                    let num_classes = config.model.output_heads.price_levels.bins as usize;
                    self.calculate_crossentropy_loss(predictions, targets, num_classes)
                } else {
                    // MSE for continuous price prediction
                    Ok(predictions.sub(targets)?.sqr()?.mean_all()?)
                }
            }
            TargetType::Direction => {
                // Direction targets are ALWAYS 3-class classification (Down=0, Sideways=1, Up=2)
                // Use CrossEntropy loss with proper error handling - NO FALLBACKS
                log::debug!(
                    "🎯 Direction target: Using CrossEntropy loss for 3-class classification"
                );

                // Validate model output matches Direction classes (3)
                if predictions.dims().last() != Some(&3) {
                    return Err(VangaError::ModelError(format!(
                        "Direction target requires model output_size=3, got {}. Please update model configuration.",
                        predictions.dims().last().unwrap_or(&0)
                    )));
                }

                // Use proper 3-class CrossEntropy loss (same pattern as PriceLevel)
                self.calculate_crossentropy_loss(predictions, targets, 3)
            }
            TargetType::Volatility => {
                // Volatility targets are ALWAYS 3-class classification (Low=0, Medium=1, High=2)
                // Use CrossEntropy loss with proper error handling - NO FALLBACKS
                log::debug!(
                    "🎯 Volatility target: Using CrossEntropy loss for 3-class classification"
                );

                // Validate model output matches Volatility classes (3)
                if predictions.dims().last() != Some(&3) {
                    return Err(VangaError::ModelError(format!(
                        "Volatility target requires model output_size=3, got {}. Please update model configuration.",
                        predictions.dims().last().unwrap_or(&0)
                    )));
                }

                // Use proper 3-class CrossEntropy loss (same pattern as PriceLevel)
                self.calculate_crossentropy_loss(predictions, targets, 3)
            }
        }
    }

    /// Calculate multi-target loss with proper combination
    /// Calculate loss using configured loss function with target-aware logic
    pub fn calculate_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        config: &crate::config::TrainingConfig,
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
        let loss_result =
            self.calculate_single_target_loss(predictions, targets, target_type, config)?;

        // Fallback to existing loss function system if configured
        let final_loss = if matches!(
            self.loss_function,
            crate::model::loss::CryptoLossFunction::MSE
        ) {
            // Use new target-aware loss for MSE (most common case)
            log::debug!("✅ Using target-aware loss calculation");
            loss_result
        } else {
            // Use existing advanced loss functions for specialized cases
            log::debug!("🔄 Using advanced loss function: {:?}", self.loss_function);
            use crate::model::loss::TensorCryptoLossFunction;
            let mut tensor_loss_fn = TensorCryptoLossFunction::new_with_class_weights(
                self.loss_function.clone(),
                self.global_class_weights.clone(),
            );

            let market_regime = match &self.loss_function {
                crate::model::loss::CryptoLossFunction::RegimeAware { .. }
                | crate::model::loss::CryptoLossFunction::Composite { .. } => {
                    let regime = self.detect_market_regime(predictions, targets)?;
                    log::debug!("🔍 REGIME DETECTION - Calculated regime: {:?}", regime);
                    regime
                }
                _ => crate::optimization::objective::MarketRegime::MediumVolatility,
            };

            tensor_loss_fn.calculate_tensor_loss(predictions, targets, market_regime)?
        };

        let loss_value = final_loss.to_scalar::<f32>().unwrap_or(0.0);
        log::debug!(
            "🎯 FINAL LOSS - Value: {:.6}, Target type: {:?}, Loss function: {:?}",
            loss_value,
            target_type,
            self.loss_function
        );

        // Validate loss is not NaN or infinite
        if !loss_value.is_finite() {
            log::error!("🚨 Invalid loss value: {}", loss_value);
            return Err(VangaError::ModelError(format!(
                "Loss calculation produced invalid value: {}",
                loss_value
            )));
        }

        Ok(final_loss)
    }

    /// Detect market regime using mathematically sound approach
    /// Uses target statistics to determine market conditions for regime-aware loss functions
    fn detect_market_regime(
        &self,
        _predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<crate::optimization::objective::MarketRegime> {
        use crate::optimization::objective::MarketRegime;

        // ADDED: Validate tensor shapes and log for debugging
        log::debug!(
            "🔍 Market regime detection - Input targets shape: {:?}",
            targets.shape()
        );

        // Validate minimum tensor dimensions
        if targets.dims().len() < 2 {
            return Err(VangaError::ModelError(format!(
                "Invalid targets tensor for regime detection: expected 2D tensor, got shape {:?}",
                targets.shape()
            )));
        }

        // Use targets for regime detection - they represent actual market conditions
        // targets shape: [batch_size, num_targets] where num_targets = 9

        // Calculate adaptive statistics from the actual target data
        let targets_contiguous = targets.contiguous()?;
        let target_mean = targets_contiguous.mean_all()?;

        // FIXED: Proper scalar broadcasting for tensor subtraction
        // Create a tensor with the same shape as targets filled with the mean value
        let target_mean_scalar = target_mean
            .to_scalar::<f32>()
            .map_err(|e| VangaError::ModelError(format!("Failed to extract mean scalar: {}", e)))?;

        let target_mean_broadcast = Tensor::full(
            target_mean_scalar,
            targets_contiguous.shape(),
            targets_contiguous.device(),
        )?
        .contiguous()?;

        log::debug!(
            "🔍 Market regime detection shapes: targets {:?}, mean_broadcast {:?}",
            targets_contiguous.shape(),
            target_mean_broadcast.shape()
        );

        let target_variance = targets_contiguous
            .sub(&target_mean_broadcast)?
            .contiguous()?
            .sqr()?
            .mean_all()?;
        let volatility =
            target_variance.sqrt()?.to_scalar::<f32>().map_err(|e| {
                VangaError::ModelError(format!("Volatility calculation failed: {}", e))
            })? as f64;

        let target_mean_value = target_mean.to_scalar::<f32>().unwrap_or(0.0) as f64;

        // Calculate adaptive thresholds based on actual data distribution
        let target_std = volatility; // Standard deviation
        let target_abs_mean = target_mean_value.abs();

        // Dynamic thresholds based on data characteristics
        let high_vol_threshold = target_std * 2.0; // 2 standard deviations
        let low_vol_threshold = target_std * 0.5; // 0.5 standard deviations
        let trend_threshold = target_abs_mean * 0.1 + target_std * 0.5; // Adaptive trend detection
        let range_threshold = target_std * 1.0; // 1 standard deviation for range-bound

        // Classify market regime using adaptive thresholds
        let regime = match (volatility, target_mean_value) {
            (v, _) if v > high_vol_threshold => MarketRegime::HighVolatility,
            (v, t) if v < low_vol_threshold && t.abs() < trend_threshold * 0.5 => {
                MarketRegime::LowVolatility
            }
            (_, t) if t > trend_threshold => MarketRegime::BullMarket,
            (_, t) if t < -trend_threshold => MarketRegime::BearMarket,
            (v, _) if v < range_threshold => MarketRegime::RangeBound,
            _ => MarketRegime::MediumVolatility,
        };

        Ok(regime)
    }

    /// Validate loss function configuration and mathematical correctness
    pub fn validate_loss_function(&self) -> Result<()> {
        match &self.loss_function {
            crate::model::loss::CryptoLossFunction::MSE => {
                log::info!("✅ Using MSE loss function");
            }
            crate::model::loss::CryptoLossFunction::Composite {
                accuracy_weight,
                direction_weight,
                volatility_weight,
                risk_weight,
            } => {
                // Validate weights are non-negative
                if *accuracy_weight < 0.0
                    || *direction_weight < 0.0
                    || *volatility_weight < 0.0
                    || *risk_weight < 0.0
                {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "Composite loss weights must be non-negative".to_string(),
                    ));
                }

                // Validate at least one weight is positive
                let total_weight =
                    accuracy_weight + direction_weight + volatility_weight + risk_weight;
                if total_weight <= 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "Composite loss must have at least one positive weight".to_string(),
                    ));
                }

                // Log configuration for debugging
                log::info!(
                    "✅ Composite loss validated: acc={:.2}, dir={:.2}, vol={:.2}, risk={:.2} (total={:.2})",
                    accuracy_weight, direction_weight, volatility_weight, risk_weight, total_weight
                );
            }
            crate::model::loss::CryptoLossFunction::DirectionalFocused { direction_penalty } => {
                if *direction_penalty <= 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "DirectionalFocused direction_penalty must be positive".to_string(),
                    ));
                }
                log::info!(
                    "✅ DirectionalFocused loss validated: penalty={:.2}",
                    direction_penalty
                );
            }
            crate::model::loss::CryptoLossFunction::RiskAdjusted {
                sharpe_weight,
                drawdown_weight,
            } => {
                if *sharpe_weight < 0.0 || *drawdown_weight < 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "RiskAdjusted loss weights must be non-negative".to_string(),
                    ));
                }
                if *sharpe_weight + *drawdown_weight <= 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "RiskAdjusted loss must have at least one positive weight".to_string(),
                    ));
                }
                log::info!(
                    "✅ RiskAdjusted loss validated: sharpe={:.2}, drawdown={:.2}",
                    sharpe_weight,
                    drawdown_weight
                );
            }
            crate::model::loss::CryptoLossFunction::VolatilityAware {
                volatility_threshold,
                penalty_factor,
            } => {
                if *volatility_threshold < 0.0 || *penalty_factor < 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "VolatilityAware loss parameters must be non-negative".to_string(),
                    ));
                }
                log::info!(
                    "✅ VolatilityAware loss validated: threshold={:.4}, penalty={:.2}",
                    volatility_threshold,
                    penalty_factor
                );
            }
            crate::model::loss::CryptoLossFunction::RegimeAware { volatility_penalty } => {
                if *volatility_penalty < 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "RegimeAware volatility_penalty must be non-negative".to_string(),
                    ));
                }
                log::info!(
                    "✅ RegimeAware loss validated: penalty={:.2}",
                    volatility_penalty
                );
            }
            crate::model::loss::CryptoLossFunction::MultiObjective { horizon_weights } => {
                if horizon_weights.is_empty() {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "MultiObjective loss must have at least one horizon weight".to_string(),
                    ));
                }
                if horizon_weights.iter().any(|&w| w < 0.0) {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "MultiObjective horizon weights must be non-negative".to_string(),
                    ));
                }
                let total_weight: f64 = horizon_weights.iter().sum();
                if total_weight <= 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "MultiObjective loss must have at least one positive weight".to_string(),
                    ));
                }
                log::info!(
                    "✅ MultiObjective loss validated: {} horizons, total_weight={:.2}",
                    horizon_weights.len(),
                    total_weight
                );
            }
        }

        Ok(())
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
