//! Tests for enhanced dropout functionality (variational and recurrent)

use crate::model::lstm::seeded_weights::SeededTensorUtils;
use candle_core::{DType, Device, Tensor};

#[test]
fn test_variational_dropout_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let shape = &[2, 3, 4]; // [batch, sequence, features]

    // Create test tensor
    let tensor = Tensor::ones(shape, DType::F32, &device)?;
    let dropout_rate = 0.5;
    let sequence_id = "test_sequence";

    // Apply variational dropout multiple times with same sequence ID
    let result1 = SeededTensorUtils::variational_dropout(&tensor, dropout_rate, true, sequence_id)?;
    let result2 = SeededTensorUtils::variational_dropout(&tensor, dropout_rate, true, sequence_id)?;

    // Results should be identical (same mask used)
    let diff = result1.sub(&result2)?.abs()?.sum_all()?;
    let diff_value: f32 = diff.to_scalar()?;

    assert!(
        diff_value < 1e-6,
        "Variational dropout should use same mask across calls"
    );

    // Clear masks and test with different sequence ID
    SeededTensorUtils::clear_variational_masks(Some(sequence_id));
    let result3 =
        SeededTensorUtils::variational_dropout(&tensor, dropout_rate, true, "different_sequence")?;

    // This should be different from previous results
    let diff2 = result1.sub(&result3)?.abs()?.sum_all()?;
    let diff2_value: f32 = diff2.to_scalar()?;

    assert!(
        diff2_value > 1e-6,
        "Different sequence IDs should produce different masks"
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
    let inference_result =
        SeededTensorUtils::variational_dropout(&tensor, dropout_rate, false, "test")?;
    let diff = tensor.sub(&inference_result)?.abs()?.sum_all()?;
    let diff_value: f32 = diff.to_scalar()?;

    assert!(
        diff_value < 1e-6,
        "Dropout should be disabled during inference"
    );

    // During training, dropout should be applied
    let training_result =
        SeededTensorUtils::variational_dropout(&tensor, dropout_rate, true, "test")?;
    let training_diff = tensor.sub(&training_result)?.abs()?.sum_all()?;
    let training_diff_value: f32 = training_diff.to_scalar()?;

    assert!(
        training_diff_value > 1e-6,
        "Dropout should be applied during training"
    );

    Ok(())
}

#[test]
fn test_mask_clearing() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let shape = &[2, 3, 4];
    let tensor = Tensor::ones(shape, DType::F32, &device)?;
    let dropout_rate = 0.5;
    let sequence_id = "test_clear";

    // Apply variational dropout to create a mask
    let result1 = SeededTensorUtils::variational_dropout(&tensor, dropout_rate, true, sequence_id)?;

    // Clear the specific mask
    SeededTensorUtils::clear_variational_masks(Some(sequence_id));

    // Apply again - should get a different result due to new mask
    let result2 = SeededTensorUtils::variational_dropout(&tensor, dropout_rate, true, sequence_id)?;

    let diff = result1.sub(&result2)?.abs()?.sum_all()?;
    let diff_value: f32 = diff.to_scalar()?;

    // Results should be different after clearing masks
    assert!(
        diff_value > 1e-6,
        "Clearing masks should result in different dropout patterns"
    );

    Ok(())
}

#[test]
fn test_zero_dropout_rate() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let shape = &[2, 3, 4];
    let tensor = Tensor::ones(shape, DType::F32, &device)?;

    // Test with zero dropout rate
    let result = SeededTensorUtils::variational_dropout(&tensor, 0.0, true, "test")?;
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
    let result = SeededTensorUtils::variational_dropout(&tensor, 1.0, true, "test")?;
    let diff = tensor.sub(&result)?.abs()?.sum_all()?;
    let diff_value: f32 = diff.to_scalar()?;

    assert!(
        diff_value < 1e-6,
        "Full dropout rate should not modify tensor"
    );

    Ok(())
}
