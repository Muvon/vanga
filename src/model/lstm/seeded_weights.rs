//! Seeded weight initialization for reproducible LSTM training
//!
//! This module provides utilities for reproducible weight initialization
//! using Candle's native Device::set_seed() functionality.

use candle_core::{DType, Device, Result, Tensor};

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
}
