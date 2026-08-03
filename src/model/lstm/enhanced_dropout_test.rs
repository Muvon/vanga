//! Tests for enhanced dropout functionality (variational and recurrent)

use crate::model::lstm::seeded_weights::SeededTensorUtils;
use candle_core::{DType, Device, Tensor};

#[test]
fn test_variational_dropout_constant_across_timesteps() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let shape = &[2, 5, 4]; // [batch, sequence, features]

    // Create test tensor of ones so surviving values are exactly the scale factor
    let tensor = Tensor::ones(shape, DType::F32, &device)?;
    let dropout_rate = 0.5;

    let result = SeededTensorUtils::variational_dropout(&tensor, dropout_rate, true)?;
    let values: Vec<f32> = result.flatten_all()?.to_vec1()?;

    // Gal & Ghahramani: the mask must be identical at every timestep — for each
    // (batch, feature) pair, all timesteps are either all dropped or all kept
    let (batch, seq_len, features) = (shape[0], shape[1], shape[2]);
    for b in 0..batch {
        for f in 0..features {
            let first = values[b * seq_len * features + f];
            for t in 1..seq_len {
                let v = values[b * seq_len * features + t * features + f];
                assert!(
                    (v - first).abs() < 1e-6,
                    "Variational dropout mask must be constant across timesteps (b={}, f={}, t={})",
                    b,
                    f,
                    t
                );
            }
        }
    }

    // Kept values must be scaled by 1/(1-p)
    let scale = 1.0 / (1.0 - dropout_rate);
    for &v in &values {
        assert!(
            v.abs() < 1e-6 || (v - scale).abs() < 1e-5,
            "Values must be either dropped (0) or scaled ({}), got {}",
            scale,
            v
        );
    }

    Ok(())
}

#[test]
fn test_variational_dropout_resampled_per_call() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let shape = &[4, 3, 32];
    let tensor = Tensor::ones(shape, DType::F32, &device)?;
    let dropout_rate = 0.5;

    // A fresh mask must be sampled on every call — with 4*32=128 Bernoulli(0.5)
    // mask entries, two identical consecutive masks are astronomically unlikely
    let result1 = SeededTensorUtils::variational_dropout(&tensor, dropout_rate, true)?;
    let result2 = SeededTensorUtils::variational_dropout(&tensor, dropout_rate, true)?;

    let diff: f32 = result1.sub(&result2)?.abs()?.sum_all()?.to_scalar()?;
    assert!(
        diff > 1e-6,
        "Variational dropout must resample its mask on every call"
    );

    Ok(())
}

#[test]
fn test_recurrent_dropout_functionality() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let shape = &[2, 3, 4]; // [batch, sequence, hidden]

    // Create test hidden state tensor
    let hidden_tensor = Tensor::ones(shape, DType::F32, &device)?;
    let dropout_rate = 0.3;

    // Apply recurrent dropout
    let result = SeededTensorUtils::recurrent_dropout(&hidden_tensor, dropout_rate, true)?;

    // Check that some values are zeroed out and others are scaled
    let original_sum: f32 = hidden_tensor.sum_all()?.to_scalar()?;
    let result_sum: f32 = result.sum_all()?.to_scalar()?;

    // With dropout, the sum should be different but scaled appropriately
    // The scaling factor is 1/(1-p), so we expect the mean to be preserved
    let expected_mean = original_sum / (shape[0] * shape[1] * shape[2]) as f32;
    let actual_mean = result_sum / (shape[0] * shape[1] * shape[2]) as f32;

    // Allow some tolerance due to random dropout
    let mean_diff = (expected_mean - actual_mean).abs();
    assert!(
        mean_diff < 0.5,
        "Recurrent dropout should preserve expected value"
    );

    Ok(())
}

#[test]
fn test_dropout_training_vs_inference() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let shape = &[2, 3, 4];
    let tensor = Tensor::ones(shape, DType::F32, &device)?;
    let dropout_rate = 0.5;

    // During inference, dropout should be disabled
    let inference_result = SeededTensorUtils::variational_dropout(&tensor, dropout_rate, false)?;
    let diff = tensor.sub(&inference_result)?.abs()?.sum_all()?;
    let diff_value: f32 = diff.to_scalar()?;

    assert!(
        diff_value < 1e-6,
        "Dropout should be disabled during inference"
    );

    // During training, dropout should be applied
    let training_result = SeededTensorUtils::variational_dropout(&tensor, dropout_rate, true)?;
    let training_diff = tensor.sub(&training_result)?.abs()?.sum_all()?;
    let training_diff_value: f32 = training_diff.to_scalar()?;

    assert!(
        training_diff_value > 1e-6,
        "Dropout should be applied during training"
    );

    Ok(())
}

#[test]
fn test_zero_dropout_rate() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let shape = &[2, 3, 4];
    let tensor = Tensor::ones(shape, DType::F32, &device)?;

    // Test with zero dropout rate
    let result = SeededTensorUtils::variational_dropout(&tensor, 0.0, true)?;
    let diff = tensor.sub(&result)?.abs()?.sum_all()?;
    let diff_value: f32 = diff.to_scalar()?;

    assert!(
        diff_value < 1e-6,
        "Zero dropout rate should not modify tensor"
    );

    Ok(())
}

#[test]
fn test_full_dropout_rate() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let shape = &[2, 3, 4];
    let tensor = Tensor::ones(shape, DType::F32, &device)?;

    // Test with full dropout rate (should not apply dropout)
    let result = SeededTensorUtils::variational_dropout(&tensor, 1.0, true)?;
    let diff = tensor.sub(&result)?.abs()?.sum_all()?;
    let diff_value: f32 = diff.to_scalar()?;

    assert!(
        diff_value < 1e-6,
        "Full dropout rate should not modify tensor"
    );

    Ok(())
}
