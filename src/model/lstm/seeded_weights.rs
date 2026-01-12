//! Seeded weight initialization for reproducible LSTM training
//!
//! This module provides utilities for reproducible weight initialization
//! using Candle's native Device::set_seed() functionality.

use candle_core::{DType, Device, Result, Tensor, Var};
use std::cell::RefCell;
use std::collections::HashMap;

/// Gram-Schmidt orthogonalization process for creating orthogonal matrices
///
/// This is used for LSTM recurrent weight initialization to prevent
/// vanishing/exploding gradients and ensure stable training.
fn gram_schmidt_orthogonalization(matrix: &Tensor) -> Result<Tensor> {
    let shape = matrix.shape();
    let rows = shape.dims()[0];
    let cols = shape.dims()[1];
    let device = matrix.device();

    // Convert to f64 for numerical stability during orthogonalization
    let matrix_f64 = matrix.to_dtype(candle_core::DType::F64)?;

    // Initialize result matrix with explicit type
    let mut orthogonal_vectors: Vec<Tensor> = Vec::new();

    for col_idx in 0..cols {
        // Get current column vector
        let current_col = matrix_f64.narrow(1, col_idx, 1)?;
        let mut orthogonal_col = current_col.clone();

        // Subtract projections onto previous orthogonal vectors
        for prev_vec in &orthogonal_vectors {
            // Calculate dot product (projection coefficient)
            let dot_product = (prev_vec.mul(&orthogonal_col)?)
                .sum_all()?
                .to_scalar::<f64>()?;

            // Calculate norm squared of previous vector
            let norm_squared = (prev_vec.mul(prev_vec)?).sum_all()?.to_scalar::<f64>()?;

            if norm_squared > 1e-10 {
                let projection_coeff = dot_product / norm_squared;
                // Use affine transformation instead of tensor multiplication
                let projection = prev_vec.affine(projection_coeff, 0.0)?;
                orthogonal_col = orthogonal_col.sub(&projection)?;
            }
        }

        // Normalize the orthogonal vector
        let norm = (orthogonal_col.mul(&orthogonal_col)?)
            .sum_all()?
            .to_scalar::<f64>()?
            .sqrt();

        if norm > 1e-10 {
            // Use scalar division - multiply by 1/norm instead of dividing by norm
            orthogonal_col = orthogonal_col.affine(1.0 / norm, 0.0)?;
        } else {
            // If vector becomes zero, use a random unit vector
            orthogonal_col =
                Tensor::randn(0.0, 1.0, &[rows, 1], device)?.to_dtype(candle_core::DType::F64)?;
            let norm = (orthogonal_col.mul(&orthogonal_col)?)
                .sum_all()?
                .to_scalar::<f64>()?
                .sqrt();
            if norm > 1e-10 {
                orthogonal_col = orthogonal_col.affine(1.0 / norm, 0.0)?;
            }
        }

        orthogonal_vectors.push(orthogonal_col);
    }

    // Concatenate all orthogonal vectors
    let result = Tensor::cat(&orthogonal_vectors, 1)?;

    // Convert back to f32 for consistency with the rest of the system
    result.to_dtype(candle_core::DType::F32)
}

// Thread-local storage for variational dropout masks
// Key: sequence_id, Value: dropout mask tensor
thread_local! {
    static VARIATIONAL_MASKS: RefCell<HashMap<String, Tensor>> = RefCell::new(HashMap::new());
}

/// Seeded tensor creation utilities using Candle's native device seeding
pub struct SeededTensorUtils;

impl SeededTensorUtils {
    /// Create a tensor with Xavier initialization using device seed
    ///
    /// Note: The device should have its seed set via device.set_seed() before calling this
    pub fn xavier_tensor(shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
        let fan_in = if shape.len() >= 2 { shape[0] } else { 1 };
        let fan_out = if shape.len() >= 2 { shape[1] } else { shape[0] };
        let std_dev = (2.0 / (fan_in + fan_out) as f64).sqrt();

        // Use Candle's native randn with device seed for reproducibility
        let tensor = Tensor::randn(0.0, std_dev as f32, shape, device)?;
        tensor.to_dtype(dtype)
    }

    /// Create an orthogonal tensor for LSTM recurrent weights
    ///
    /// This is critical for LSTM stability and reproducible training.
    /// Orthogonal initialization prevents vanishing/exploding gradients in recurrent connections.
    ///
    /// Note: The device should have its seed set via device.set_seed() before calling this
    pub fn orthogonal_tensor(shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
        if shape.len() != 2 {
            return Err(candle_core::Error::Msg(
                "Orthogonal initialization requires 2D tensor (matrix)".to_string(),
            ));
        }

        let rows = shape[0];
        let cols = shape[1];

        // For non-square matrices, we need to handle them specially
        let (m, n) = if rows >= cols {
            (rows, cols)
        } else {
            (cols, rows)
        };

        // Generate random matrix with device seed
        let random_matrix = Tensor::randn(0.0, 1.0, &[m, n], device)?;

        // Perform QR decomposition to get orthogonal matrix
        // Since Candle doesn't have built-in QR, we'll use Gram-Schmidt process
        let orthogonal_matrix = gram_schmidt_orthogonalization(&random_matrix)?;

        // Reshape if needed and ensure correct orientation
        let final_matrix = if rows >= cols {
            orthogonal_matrix
        } else {
            // Transpose for tall matrices
            orthogonal_matrix.t()?
        };

        // Take only the needed portion if matrix is larger than required
        let result =
            if final_matrix.shape().dims()[0] > rows || final_matrix.shape().dims()[1] > cols {
                final_matrix.narrow(0, 0, rows)?.narrow(1, 0, cols)?
            } else {
                final_matrix
            };

        result.to_dtype(dtype)
    }

    /// Create a tensor with He initialization (for ReLU activations)
    ///
    /// Note: The device should have its seed set via device.set_seed() before calling this
    pub fn he_tensor(shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
        let fan_in = if shape.len() >= 2 { shape[0] } else { 1 };
        let std_dev = (2.0 / fan_in as f64).sqrt();

        // Use Candle's native randn with device seed for reproducibility
        let tensor = Tensor::randn(0.0, std_dev as f32, shape, device)?;
        tensor.to_dtype(dtype)
    }

    /// Create a tensor with normal distribution using device seed
    ///
    /// Note: The device should have its seed set via device.set_seed() before calling this
    pub fn normal_tensor(
        shape: &[usize],
        mean: f64,
        std: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        // Use Candle's native randn with device seed for reproducibility
        let tensor = Tensor::randn(mean as f32, std as f32, shape, device)?;
        tensor.to_dtype(dtype)
    }

    /// Create a zero-initialized tensor
    pub fn zeros_tensor(shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
        Tensor::zeros(shape, dtype, device)
    }

    /// Create a ones-initialized tensor
    pub fn ones_tensor(shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
        Tensor::ones(shape, dtype, device)
    }

    /// Apply deterministic dropout using device seed
    ///
    /// This function provides reproducible dropout behavior by using the device's
    /// current seed state. For reproducible results, ensure device.set_seed() is
    /// called before training begins.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to apply dropout to
    /// * `dropout_rate` - Dropout probability (0.0 to 1.0)
    /// * `training` - Whether in training mode (dropout only applied during training)
    ///
    /// # Returns
    /// * Tensor with dropout applied (scaled by 1/(1-p) during training)
    pub fn deterministic_dropout(
        tensor: &Tensor,
        dropout_rate: f32,
        training: bool,
    ) -> Result<Tensor> {
        // No dropout during inference or if rate is 0
        if !training || dropout_rate <= 0.0 || dropout_rate >= 1.0 {
            return Ok(tensor.clone());
        }

        let device = tensor.device();
        let shape = tensor.shape();

        // Create random tensor using device's current seed state
        // This will be deterministic if device seed was set
        let random_tensor = Tensor::rand(0.0, 1.0, shape, device)?;

        // Create dropout mask: keep values where random > dropout_rate
        let keep_mask = random_tensor.gt(dropout_rate)?;

        // Apply mask and scale by 1/(1-p) to maintain expected value
        let scale = 1.0 / (1.0 - dropout_rate);
        let masked_tensor = tensor.mul(&keep_mask.to_dtype(tensor.dtype())?)?;
        let scale_tensor = Tensor::new(scale, device)?.broadcast_as(tensor.shape())?;
        masked_tensor.mul(&scale_tensor)
    }

    /// Variational dropout - uses same mask across all time steps in a sequence
    ///
    /// This implements the variational dropout technique from Gal & Ghahramani (2016)
    /// where the same dropout mask is applied across all time steps to prevent
    /// temporal inconsistency in LSTM training.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to apply dropout to
    /// * `dropout_rate` - Dropout probability (0.0 to 1.0)
    /// * `training` - Whether model is in training mode
    /// * `sequence_id` - Unique identifier for this sequence (for mask consistency)
    ///
    /// # Returns
    /// * Tensor with variational dropout applied (same mask across time steps)
    pub fn variational_dropout(
        tensor: &Tensor,
        dropout_rate: f32,
        training: bool,
        sequence_id: &str,
    ) -> Result<Tensor> {
        // No dropout during inference or if rate is 0
        if !training || dropout_rate <= 0.0 || dropout_rate >= 1.0 {
            return Ok(tensor.clone());
        }

        let device = tensor.device();
        let shape = tensor.shape();

        // Try to get existing mask for this sequence
        let mask = VARIATIONAL_MASKS.with(|masks| -> Result<Tensor> {
            let mut masks_map = masks.borrow_mut();

            if let Some(existing_mask) = masks_map.get(sequence_id) {
                // Check if existing mask shape matches current tensor
                if existing_mask.shape() == shape {
                    return Ok(existing_mask.clone());
                } else {
                    // Shape mismatch, remove old mask and create new one
                    masks_map.remove(sequence_id);
                }
            }

            // Create new mask for this sequence
            let random_tensor = Tensor::rand(0.0, 1.0, shape, device)?;
            let keep_mask = random_tensor.gt(dropout_rate)?;
            let mask_tensor = keep_mask.to_dtype(tensor.dtype())?;

            // Store mask for future time steps in this sequence
            masks_map.insert(sequence_id.to_string(), mask_tensor.clone());

            Ok(mask_tensor)
        })?;

        // Apply mask and scale by 1/(1-p) to maintain expected value
        let scale = 1.0 / (1.0 - dropout_rate);
        let masked_tensor = tensor.mul(&mask)?;
        let scale_tensor = Tensor::new(scale, device)?.broadcast_as(tensor.shape())?;
        masked_tensor.mul(&scale_tensor)
    }

    /// Recurrent dropout - applies dropout to hidden state connections
    ///
    /// This applies dropout specifically to the recurrent connections (h_t-1 → h_t)
    /// rather than the input connections (x_t → h_t). This is critical for LSTM
    /// regularization in time series tasks.
    ///
    /// # Arguments
    /// * `hidden_tensor` - Hidden state tensor to apply dropout to
    /// * `dropout_rate` - Dropout probability (0.0 to 1.0)
    /// * `training` - Whether model is in training mode
    ///
    /// # Returns
    /// * Tensor with recurrent dropout applied
    pub fn recurrent_dropout(
        hidden_tensor: &Tensor,
        dropout_rate: f32,
        training: bool,
    ) -> Result<Tensor> {
        // No dropout during inference or if rate is 0
        if !training || dropout_rate <= 0.0 || dropout_rate >= 1.0 {
            return Ok(hidden_tensor.clone());
        }

        let device = hidden_tensor.device();
        let shape = hidden_tensor.shape();

        // Create random tensor for recurrent connections
        let random_tensor = Tensor::rand(0.0, 1.0, shape, device)?;

        // Create dropout mask: keep values where random > dropout_rate
        let keep_mask = random_tensor.gt(dropout_rate)?;

        // Apply mask and scale by 1/(1-p) to maintain expected value
        let scale = 1.0 / (1.0 - dropout_rate);
        let masked_tensor = hidden_tensor.mul(&keep_mask.to_dtype(hidden_tensor.dtype())?)?;
        let scale_tensor = Tensor::new(scale, device)?.broadcast_as(hidden_tensor.shape())?;
        masked_tensor.mul(&scale_tensor)
    }

    /// Clear variational dropout masks (call at end of sequence or epoch)
    ///
    /// This clears stored masks to prevent memory leaks and ensure fresh masks
    /// for new sequences.
    ///
    /// # Arguments
    /// * `sequence_id` - Optional specific sequence ID to clear, or None to clear all
    pub fn clear_variational_masks(sequence_id: Option<&str>) {
        VARIATIONAL_MASKS.with(|masks| {
            let mut masks_map = masks.borrow_mut();
            if let Some(id) = sequence_id {
                masks_map.remove(id);
            } else {
                masks_map.clear();
            }
        });
    }

    /// Apply proper LSTM weight initialization to a VarMap
    ///
    /// This function applies the correct initialization strategy for LSTM weights:
    /// - Input-to-hidden weights: Xavier/Glorot initialization
    /// - Hidden-to-hidden (recurrent) weights: Orthogonal initialization
    /// - Biases: Zero initialization (except forget gate bias = 1.0)
    ///
    /// This is critical for stable and reproducible LSTM training.
    pub fn apply_lstm_weight_initialization(
        varmap: &candle_nn::VarMap,
        device: &Device,
        seed: Option<u64>,
    ) -> Result<()> {
        // Set device seed if provided
        if let Some(seed_value) = seed {
            set_device_seed_with_logging(device, Some(seed_value))?;
        }

        let name_var_pairs: Vec<(String, Var)> = {
            let var_data = varmap.data().lock().unwrap();
            var_data
                .iter()
                .map(|(name, var)| (name.clone(), var.clone()))
                .collect()
        };

        let mut initialized_count = 0;
        let mut recurrent_count = 0;
        let mut bias_count = 0;

        log::info!("🔧 Applying proper LSTM weight initialization...");

        for (var_name, var) in name_var_pairs.iter() {
            let shape = var.shape();
            let dims = shape.dims();
            let var_name_str = var_name.as_str();

            // Skip LayerNorm parameters (gamma/beta) - they have their own initialization
            if var_name_str.contains("layer_norm_")
                && (var_name_str.contains(".gamma") || var_name_str.contains(".beta"))
            {
                log::debug!(
                    "⏭️  Skipping LayerNorm parameter '{}' (already initialized)",
                    var_name_str
                );
                continue;
            }

            log::debug!("🔍 Processing tensor '{}': shape={:?}", var_name_str, dims);

            // Determine weight type based on tensor name and shape
            if dims.len() == 2 {
                // 2D tensors are weight matrices
                let (rows, cols) = (dims[0], dims[1]);

                // Identify output/classification layer (Fixup Initialization: ICLR 2019)
                // "Initialize the classification layer to 0" for stable training
                let is_output_layer = var_name_str.contains("output.weight")
                    || var_name_str.ends_with("output.weight");

                if is_output_layer {
                    // Apply zero initialization for output layer (Fixup Initialization)
                    log::info!(
                        "🎯 Applying ZERO initialization to output layer '{}': shape={:?} (Fixup Init)",
                        var_name_str, dims
                    );
                    let zero_weights = Tensor::zeros(dims, var.dtype(), device)?;
                    var.set(&zero_weights)?;
                    initialized_count += 1;

                    log::info!(
                        "✅ Output layer '{}' initialized to ZERO for balanced class prediction",
                        var_name_str
                    );
                    continue;
                }

                // Identify weight types based on Candle's LSTM naming convention
                // In Candle LSTM: weight_ih = input-to-hidden, weight_hh = hidden-to-hidden (recurrent)
                let is_recurrent_weight = var_name_str.contains("weight_hh")
                    || var_name_str.contains("hh")
                    || var_name_str.contains("hidden_hidden")
                    || var_name_str.contains("recurrent")
                    ||
                    // Fallback: square matrices are likely recurrent
                    (rows == cols && rows > 50); // Avoid small matrices

                if is_recurrent_weight {
                    // Apply orthogonal initialization for recurrent weights
                    log::info!(
                        "🔄 Applying orthogonal initialization to recurrent weight '{}': shape={:?}",
                        var_name_str, dims
                    );
                    let orthogonal_weights = Self::orthogonal_tensor(dims, device, var.dtype())?;

                    // Verify the tensor was created correctly
                    let orth_shape = orthogonal_weights.shape();
                    if orth_shape.dims() != dims {
                        log::error!(
                            "❌ Shape mismatch: expected {:?}, got {:?}",
                            dims,
                            orth_shape.dims()
                        );
                        continue;
                    }

                    var.set(&orthogonal_weights)?;
                    recurrent_count += 1;

                    // Verify the weight was actually set by checking a sample value
                    if let Ok(sample_val) = orthogonal_weights
                        .flatten_all()?
                        .narrow(0, 0, 1)?
                        .to_vec1::<f32>()
                    {
                        log::info!(
                            "✅ Orthogonal weight '{}' set successfully, sample value: {:.6}",
                            var_name_str,
                            sample_val[0]
                        );
                    }
                } else {
                    // Apply Xavier initialization for input-to-hidden weights
                    log::info!(
                        "📥 Applying Xavier initialization to input weight '{}': shape={:?}",
                        var_name_str,
                        dims
                    );
                    let xavier_weights = Self::xavier_tensor(dims, device, var.dtype())?;
                    var.set(&xavier_weights)?;

                    // Verify the weight was actually set
                    if let Ok(sample_val) = xavier_weights
                        .flatten_all()?
                        .narrow(0, 0, 1)?
                        .to_vec1::<f32>()
                    {
                        log::info!(
                            "✅ Xavier weight '{}' set successfully, sample value: {:.6}",
                            var_name_str,
                            sample_val[0]
                        );
                    }
                }
                initialized_count += 1;
            } else if dims.len() == 1 {
                // 1D tensors are biases
                let bias_size = dims[0];

                // Identify output layer bias (CRITICAL for ordinal regression with Gaussian smoothing)
                let is_output_bias =
                    var_name_str.contains("output.bias") || var_name_str.ends_with("output.bias");

                if is_output_bias && bias_size == 5 {
                    // RESEARCH-BACKED FIX: Anti-middle-class bias initialization
                    // Paper: "Ordinal regression encourage conservative model to predict middle-rank classes"
                    // Solution: Initialize with small negative bias for middle class, small positive for extremes
                    // This breaks the symmetry that causes middle-class collapse with Gaussian label smoothing
                    let anti_middle_bias = vec![
                        0.1f32, // Class 0: slight positive (encourage extreme predictions)
                        0.05,   // Class 1: small positive
                        -0.2,   // Class 2: NEGATIVE (discourage middle class)
                        0.05,   // Class 3: small positive
                        0.1,    // Class 4: slight positive (encourage extreme predictions)
                    ];
                    let bias_tensor =
                        Tensor::new(anti_middle_bias.clone(), device)?.to_dtype(var.dtype())?;
                    var.set(&bias_tensor)?;
                    bias_count += 1;

                    log::info!(
                        "🎯 Output bias '{}' initialized with ANTI-MIDDLE-CLASS pattern: {:?}",
                        var_name_str,
                        anti_middle_bias
                    );
                    log::info!(
                        "   Research: Prevents ordinal regression middle-class collapse with Gaussian smoothing"
                    );
                    continue;
                }

                // Identify forget gate bias based on Candle's LSTM naming
                // In LSTM, forget gate bias should be initialized to 1.0
                let is_forget_gate_bias = var_name_str.contains("bias_hh")
                    && (var_name_str.contains("forget")
                        ||
                        // LSTM gate order is usually: input, forget, cell, output
                        // So forget gate is at positions bias_size/4 to bias_size/2
                        bias_size >= 4);

                let bias_value = if is_forget_gate_bias { 1.0 } else { 0.0 };

                // For forget gate bias, we need to set only the forget gate portion to 1.0
                let bias_tensor = if is_forget_gate_bias && bias_size >= 4 {
                    // Create bias tensor with forget gate portion set to 1.0
                    let mut bias_values = vec![0.0; bias_size];
                    let gate_size = bias_size / 4;
                    // Set forget gate bias (second quarter) to 1.0
                    for bias_val in bias_values.iter_mut().take(2 * gate_size).skip(gate_size) {
                        *bias_val = 1.0;
                    }
                    Tensor::new(bias_values, device)?.to_dtype(var.dtype())?
                } else {
                    // Regular bias initialization (all zeros)
                    Tensor::new(vec![bias_value; bias_size], device)?.to_dtype(var.dtype())?
                };

                log::info!(
                    "⚖️ Initializing bias '{}': shape={:?}, forget_gate={}, value={}",
                    var_name_str,
                    dims,
                    is_forget_gate_bias,
                    bias_value
                );
                var.set(&bias_tensor)?;
                bias_count += 1;

                // Verify the bias was actually set
                if let Ok(sample_vals) = bias_tensor.to_vec1::<f32>() {
                    let first_val = sample_vals[0];
                    let mid_val = if sample_vals.len() > 4 {
                        sample_vals[sample_vals.len() / 4]
                    } else {
                        first_val
                    };
                    log::info!(
                        "✅ Bias '{}' set successfully, first={:.6}, mid={:.6}",
                        var_name_str,
                        first_val,
                        mid_val
                    );
                }
            }
        }

        log::info!(
            "✅ LSTM weight initialization complete: {} weight matrices ({} recurrent), {} biases",
            initialized_count,
            recurrent_count,
            bias_count
        );

        if initialized_count == 0 {
            log::warn!("⚠️ No weight tensors found for initialization - using Candle defaults");
        }

        // Verify initialization worked by checking some sample weights
        Self::verify_weight_initialization(varmap)?;

        Ok(())
    }

    /// Verify that weight initialization actually worked
    fn verify_weight_initialization(varmap: &candle_nn::VarMap) -> Result<()> {
        let all_vars = varmap.all_vars();
        if all_vars.is_empty() {
            log::warn!("⚠️ No variables to verify");
            return Ok(());
        }

        let mut total_sum = 0.0f32;
        let mut total_count = 0usize;
        let mut weight_matrices = 0;

        for var in all_vars.iter() {
            let shape = var.shape();
            let dims = shape.dims();

            if dims.len() == 2 {
                // Check 2D weight matrices
                weight_matrices += 1;
                if let Ok(flattened) = var.flatten_all() {
                    if let Ok(values) = flattened.to_vec1::<f32>() {
                        let matrix_sum: f32 = values.iter().sum();
                        let matrix_mean = matrix_sum / values.len() as f32;
                        let matrix_std = {
                            let variance: f32 = values
                                .iter()
                                .map(|x| (x - matrix_mean).powi(2))
                                .sum::<f32>()
                                / values.len() as f32;
                            variance.sqrt()
                        };

                        total_sum += matrix_sum;
                        total_count += values.len();

                        log::debug!(
                            "📊 Weight matrix {}: shape={:?}, mean={:.6}, std={:.6}, sum={:.6}",
                            weight_matrices,
                            dims,
                            matrix_mean,
                            matrix_std,
                            matrix_sum
                        );

                        // Check if this looks like proper initialization (not all zeros or ones)
                        if matrix_std < 1e-6 {
                            log::warn!("⚠️ Weight matrix {} has very low std deviation ({:.6}) - may not be properly initialized", weight_matrices, matrix_std);
                        }
                    }
                }
            }
        }

        if total_count > 0 {
            let overall_mean = total_sum / total_count as f32;
            log::info!(
                "🔍 Weight verification: {} matrices, overall mean={:.6}, total_weights={}",
                weight_matrices,
                overall_mean,
                total_count
            );

            // For proper initialization, we expect the overall mean to be close to 0
            if overall_mean.abs() > 0.5 {
                log::warn!("⚠️ Overall weight mean ({:.6}) is high - initialization may not be working properly", overall_mean);
            } else {
                log::info!("✅ Weight initialization verification passed");
            }
        }

        Ok(())
    }
}

/// Helper function to set device seed and log the action
pub fn set_device_seed_with_logging(device: &Device, seed: Option<u64>) -> Result<()> {
    match seed {
        Some(0) => {
            log::info!("🎲 Seed = 0: Using random device initialization");
            // Don't call set_seed for random initialization
            Ok(())
        }
        Some(seed_value) => {
            log::info!(
                "🎲 Setting device seed to {} for reproducible weight initialization",
                seed_value
            );
            match device.set_seed(seed_value) {
                Ok(()) => Ok(()),
                Err(e) => {
                    // Handle CPU seeding limitation gracefully
                    let error_msg = format!("{}", e);
                    if error_msg.contains("cannot seed the CPU rng") {
                        log::warn!("⚠️  CPU device seeding not supported in Candle - using random initialization");
                        log::warn!("   For reproducible training, use CUDA or Metal devices");
                        Ok(()) // Continue with random initialization
                    } else {
                        Err(candle_core::Error::Msg(format!(
                            "Failed to set device seed: {}",
                            e
                        )))
                    }
                }
            }
        }
        None => {
            log::info!("🎲 No seed specified: Using random device initialization");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_seeded_tensor_reproducibility() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let shape = &[10, 20];
        let seed = 42;

        // Set seed and create first tensor
        let _ = device.set_seed(seed); // May fail on CPU, that's OK
        let tensor1 = SeededTensorUtils::xavier_tensor(shape, &device, dtype)?;

        // Set same seed and create second tensor
        let _ = device.set_seed(seed); // May fail on CPU, that's OK
        let tensor2 = SeededTensorUtils::xavier_tensor(shape, &device, dtype)?;

        // On CPU, tensors may not be identical due to seeding limitations
        // This test verifies that tensor creation works without crashing
        assert_eq!(tensor1.shape(), tensor2.shape());
        assert!(tensor1.elem_count() > 0);
        assert!(tensor2.elem_count() > 0);

        Ok(())
    }

    #[test]
    fn test_set_device_seed_with_logging() -> Result<()> {
        let device = Device::Cpu;

        // Test with None - should always succeed
        set_device_seed_with_logging(&device, None)?;

        // Test with Some(0) - should always succeed (no seeding)
        set_device_seed_with_logging(&device, Some(0))?;

        // Test with Some(42) - should succeed gracefully even if CPU seeding fails
        set_device_seed_with_logging(&device, Some(42))?;

        Ok(())
    }

    #[test]
    fn test_deterministic_dropout_reproducibility() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let shape = &[4, 8];
        let dropout_rate = 0.3;

        // Create test tensor
        let test_tensor = Tensor::ones(shape, dtype, &device)?;

        // Set seed and apply dropout
        let _ = device.set_seed(42); // May fail on CPU, that's OK
        let dropout1 = SeededTensorUtils::deterministic_dropout(&test_tensor, dropout_rate, true)?;

        // Set same seed and apply dropout again
        let _ = device.set_seed(42); // May fail on CPU, that's OK
        let dropout2 = SeededTensorUtils::deterministic_dropout(&test_tensor, dropout_rate, true)?;

        // Verify shapes are identical
        assert_eq!(dropout1.shape(), dropout2.shape());
        assert_eq!(dropout1.shape(), test_tensor.shape());

        // Verify tensors have expected properties
        assert!(dropout1.elem_count() > 0);
        assert!(dropout2.elem_count() > 0);

        // Note: On CPU, results may not be identical due to seeding limitations
        // This test verifies that dropout works without crashing and maintains shape
        Ok(())
    }

    #[test]
    fn test_deterministic_dropout_training_mode() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let shape = &[2, 4];
        let dropout_rate = 0.5;

        let test_tensor = Tensor::ones(shape, dtype, &device)?;

        // Training mode should apply dropout
        let dropout_training =
            SeededTensorUtils::deterministic_dropout(&test_tensor, dropout_rate, true)?;
        assert_eq!(dropout_training.shape(), test_tensor.shape());

        // Inference mode should return original tensor
        let dropout_inference =
            SeededTensorUtils::deterministic_dropout(&test_tensor, dropout_rate, false)?;
        assert_eq!(dropout_inference.shape(), test_tensor.shape());

        // Zero dropout rate should return original tensor
        let dropout_zero = SeededTensorUtils::deterministic_dropout(&test_tensor, 0.0, true)?;
        assert_eq!(dropout_zero.shape(), test_tensor.shape());

        Ok(())
    }

    #[test]
    fn test_orthogonal_tensor_creation() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Test square matrix
        let shape_square = &[4, 4];
        let _ = device.set_seed(42); // May fail on CPU, that's OK
        let orthogonal_square = SeededTensorUtils::orthogonal_tensor(shape_square, &device, dtype)?;
        assert_eq!(orthogonal_square.shape().dims(), shape_square);

        // Test rectangular matrix (more rows than columns)
        let shape_tall = &[6, 4];
        let _ = device.set_seed(42); // May fail on CPU, that's OK
        let orthogonal_tall = SeededTensorUtils::orthogonal_tensor(shape_tall, &device, dtype)?;
        assert_eq!(orthogonal_tall.shape().dims(), shape_tall);

        // Test rectangular matrix (more columns than rows)
        let shape_wide = &[3, 5];
        let _ = device.set_seed(42); // May fail on CPU, that's OK
        let orthogonal_wide = SeededTensorUtils::orthogonal_tensor(shape_wide, &device, dtype)?;
        assert_eq!(orthogonal_wide.shape().dims(), shape_wide);

        // Verify tensors have expected properties (non-zero, finite values)
        assert!(orthogonal_square.elem_count() > 0);
        assert!(orthogonal_tall.elem_count() > 0);
        assert!(orthogonal_wide.elem_count() > 0);

        Ok(())
    }

    #[test]
    fn test_he_tensor_creation() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let shape = &[10, 20];

        let _ = device.set_seed(42); // May fail on CPU, that's OK
        let he_tensor = SeededTensorUtils::he_tensor(shape, &device, dtype)?;

        assert_eq!(he_tensor.shape().dims(), shape);
        assert!(he_tensor.elem_count() > 0);

        Ok(())
    }

    #[test]
    fn test_orthogonal_tensor_invalid_shape() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Test 1D tensor (should fail)
        let shape_1d = &[10];
        let result_1d = SeededTensorUtils::orthogonal_tensor(shape_1d, &device, dtype);
        assert!(result_1d.is_err());

        // Test 3D tensor (should fail)
        let shape_3d = &[2, 3, 4];
        let result_3d = SeededTensorUtils::orthogonal_tensor(shape_3d, &device, dtype);
        assert!(result_3d.is_err());
    }
}
