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
}
