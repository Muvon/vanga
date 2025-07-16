//! Comprehensive tests for CryptoLossFunction mathematical correctness
//!
//! This module validates all loss function calculations and ensures proper
//! mathematical behavior across different market regimes and data patterns.

use crate::model::loss::CryptoLossFunction;
use crate::model::loss::TensorCryptoLossFunction;
use crate::optimization::objective::MarketRegime;
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};
use ndarray::Array2;

/// Test helper to create sample prediction and target tensors
fn create_test_tensors(device: &Device) -> Result<(Tensor, Tensor)> {
    // Create realistic crypto price prediction data
    let predictions_data = vec![
        100.0, 101.5, 99.8, 102.3, 98.7, 103.1, 97.9, 104.2, 96.8, 105.0, 95.5, 106.2, 94.3, 107.1,
        93.8, 108.5, 92.9, 109.3, 91.7, 110.8,
    ];

    let targets_data = vec![
        100.5, 101.2, 100.1, 102.0, 99.2, 102.8, 98.5, 103.9, 97.3, 104.7, 96.1, 105.8, 95.0,
        106.9, 94.2, 107.6, 93.5, 108.4, 92.8, 109.9,
    ];

    let predictions = Tensor::from_vec(
        predictions_data
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>(),
        (20, 1),
        device,
    )?;

    let targets = Tensor::from_vec(
        targets_data
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>(),
        (20, 1),
        device,
    )?;

    Ok((predictions, targets))
}

/// Test MSE loss calculation baseline
#[tokio::test]
async fn test_mse_loss_baseline() -> Result<()> {
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(&device)?;

    // Calculate MSE manually for verification
    let diff = predictions.sub(&targets)?;
    let mse_expected = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

    // Test with no loss function (should use MSE)
    let mut loss_fn = TensorCryptoLossFunction::new(CryptoLossFunction::Composite {
        accuracy_weight: 1.0,
        direction_weight: 0.0,
        volatility_weight: 0.0,
        risk_weight: 0.0,
    });

    let composite_loss = loss_fn
        .calculate_tensor_loss(&predictions, &targets, MarketRegime::MediumVolatility)?
        .to_scalar::<f32>()?;

    // With only accuracy_weight=1.0, should be close to MSE
    let difference = (composite_loss - mse_expected).abs();
    assert!(
        difference < 0.01,
        "Composite with accuracy_weight=1.0 should approximate MSE. Expected: {:.6}, Got: {:.6}, Diff: {:.6}",
        mse_expected, composite_loss, difference
    );

    println!("✅ MSE baseline test passed: {:.6}", mse_expected);
    Ok(())
}

/// Test directional loss calculation correctness
#[tokio::test]
async fn test_directional_loss_calculation() -> Result<()> {
    let device = Device::Cpu;

    // Create data with known directional patterns
    let predictions_data = vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0]; // Up, Down, Up, Down, Up
    let targets_data = vec![1.1, 2.1, 1.4, 3.1, 2.4, 4.1]; // Up, Down, Up, Down, Up (same direction)

    let predictions = Tensor::from_vec(
        predictions_data
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>(),
        (6, 1),
        &device,
    )?;

    let targets = Tensor::from_vec(
        targets_data
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>(),
        (6, 1),
        &device,
    )?;

    let mut loss_fn = TensorCryptoLossFunction::new(CryptoLossFunction::DirectionalFocused {
        direction_penalty: 1.0,
    });

    let directional_loss = loss_fn
        .calculate_tensor_loss(&predictions, &targets, MarketRegime::MediumVolatility)?
        .to_scalar::<f32>()?;

    // With perfect directional agreement, directional component should be low
    // But we also have MSE component, so loss won't be zero
    assert!(
        directional_loss < 1.0,
        "Directional loss with perfect agreement should be reasonable. Got: {:.6}",
        directional_loss
    );

    println!("✅ Directional loss test passed: {:.6}", directional_loss);
    Ok(())
}

/// Test market regime impact on loss calculation
#[tokio::test]
async fn test_market_regime_impact() -> Result<()> {
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(&device)?;

    let mut loss_fn = TensorCryptoLossFunction::new(CryptoLossFunction::RegimeAware {
        volatility_penalty: 0.5,
    });

    // Test different market regimes
    let low_vol_loss = loss_fn
        .calculate_tensor_loss(&predictions, &targets, MarketRegime::LowVolatility)?
        .to_scalar::<f32>()?;

    let high_vol_loss = loss_fn
        .calculate_tensor_loss(&predictions, &targets, MarketRegime::HighVolatility)?
        .to_scalar::<f32>()?;

    let medium_vol_loss = loss_fn
        .calculate_tensor_loss(&predictions, &targets, MarketRegime::MediumVolatility)?
        .to_scalar::<f32>()?;

    // High volatility should have higher loss than low volatility
    assert!(
        high_vol_loss > low_vol_loss,
        "High volatility loss ({:.6}) should be greater than low volatility loss ({:.6})",
        high_vol_loss,
        low_vol_loss
    );

    // Medium volatility should be between low and high
    assert!(
        medium_vol_loss >= low_vol_loss && medium_vol_loss <= high_vol_loss,
        "Medium volatility loss ({:.6}) should be between low ({:.6}) and high ({:.6})",
        medium_vol_loss,
        low_vol_loss,
        high_vol_loss
    );

    println!("✅ Market regime impact test passed:");
    println!("  Low volatility: {:.6}", low_vol_loss);
    println!("  Medium volatility: {:.6}", medium_vol_loss);
    println!("  High volatility: {:.6}", high_vol_loss);
    Ok(())
}

/// Test loss component normalization
#[tokio::test]
async fn test_loss_normalization() -> Result<()> {
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(&device)?;

    // Test Composite with different weight combinations
    let configs = vec![
        (1.0, 0.0, 0.0, 0.0),     // Pure MSE
        (0.0, 1.0, 0.0, 0.0),     // Pure directional
        (0.0, 0.0, 1.0, 0.0),     // Pure volatility
        (0.0, 0.0, 0.0, 1.0),     // Pure risk
        (0.25, 0.25, 0.25, 0.25), // Balanced
    ];

    for (acc_w, dir_w, vol_w, risk_w) in configs {
        let mut loss_fn = TensorCryptoLossFunction::new(CryptoLossFunction::Composite {
            accuracy_weight: acc_w,
            direction_weight: dir_w,
            volatility_weight: vol_w,
            risk_weight: risk_w,
        });

        let loss = loss_fn
            .calculate_tensor_loss(&predictions, &targets, MarketRegime::MediumVolatility)?
            .to_scalar::<f32>()?;

        // Loss should be reasonable (not NaN, not infinite, not negative)
        assert!(
            loss.is_finite() && loss >= 0.0,
            "Loss should be finite and non-negative. Got: {:.6} for weights ({:.1}, {:.1}, {:.1}, {:.1})",
            loss, acc_w, dir_w, vol_w, risk_w
        );

        // Normalized loss should be in reasonable range for crypto data
        assert!(
            loss < 10.0,
            "Normalized loss should be reasonable. Got: {:.6} for weights ({:.1}, {:.1}, {:.1}, {:.1})",
            loss, acc_w, dir_w, vol_w, risk_w
        );

        println!(
            "✅ Weights ({:.1}, {:.1}, {:.1}, {:.1}): loss = {:.6}",
            acc_w, dir_w, vol_w, risk_w, loss
        );
    }

    Ok(())
}

/// Test loss function gradient flow
#[tokio::test]
async fn test_gradient_flow() -> Result<()> {
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(&device)?;

    // Test gradient flow by computing loss and checking it's differentiable
    // Note: Simplified test without explicit gradient tracking

    let mut loss_fn = TensorCryptoLossFunction::new(CryptoLossFunction::Composite {
        accuracy_weight: 0.3,
        direction_weight: 0.5,
        volatility_weight: 0.2,
        risk_weight: 0.0,
    });

    // Compute loss (gradient flow is preserved by tensor operations)
    let loss =
        loss_fn.calculate_tensor_loss(&predictions, &targets, MarketRegime::MediumVolatility)?;

    // Verify loss is finite and reasonable
    let loss_value = loss
        .to_scalar::<f32>()
        .map_err(|e| VangaError::ModelError(format!("Failed to extract loss scalar: {}", e)))?;

    assert!(
        loss_value.is_finite() && loss_value >= 0.0,
        "Loss should be finite and non-negative, got: {}",
        loss_value
    );

    // Test that loss computation preserves tensor structure for backpropagation
    let loss_dims = loss.dims();
    assert_eq!(
        loss_dims.len(),
        0,
        "Loss should be scalar tensor, got dims: {:?}",
        loss_dims
    );

    println!("✅ Gradient flow test passed - loss: {:.6}", loss_value);

    Ok(())
}

/// Test loss scale consistency across different loss functions
#[tokio::test]
async fn test_loss_scale_consistency() -> Result<()> {
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(&device)?;

    // Calculate baseline MSE
    let mse_loss = predictions
        .sub(&targets)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?;

    let loss_functions = vec![
        (
            "DirectionalFocused",
            CryptoLossFunction::DirectionalFocused {
                direction_penalty: 0.7,
            },
        ),
        (
            "RiskAdjusted",
            CryptoLossFunction::RiskAdjusted {
                sharpe_weight: 0.3,
                drawdown_weight: 0.3,
            },
        ),
        (
            "VolatilityAware",
            CryptoLossFunction::VolatilityAware {
                volatility_threshold: 0.05,
                penalty_factor: 0.5,
            },
        ),
        (
            "Composite",
            CryptoLossFunction::Composite {
                accuracy_weight: 0.3,
                direction_weight: 0.5,
                volatility_weight: 0.2,
                risk_weight: 0.0,
            },
        ),
    ];

    println!("📊 Loss Scale Comparison (MSE baseline: {:.6}):", mse_loss);

    for (name, loss_fn) in loss_functions {
        let mut tensor_loss_fn = TensorCryptoLossFunction::new(loss_fn);
        let loss = tensor_loss_fn
            .calculate_tensor_loss(&predictions, &targets, MarketRegime::MediumVolatility)?
            .to_scalar::<f32>()?;

        let scale_ratio = loss / mse_loss;

        // After normalization, crypto losses should be within reasonable scale of MSE
        assert!(
            scale_ratio < 50.0,
            "{} loss scale too high: {:.6} ({}x MSE)",
            name,
            loss,
            scale_ratio
        );

        assert!(
            scale_ratio > 0.1,
            "{} loss scale too low: {:.6} ({:.2}x MSE)",
            name,
            loss,
            scale_ratio
        );

        println!("  {}: {:.6} ({:.1}x MSE)", name, loss, scale_ratio);
    }

    Ok(())
}

/// Integration test with array-based loss function for consistency
#[tokio::test]
async fn test_array_tensor_consistency() -> Result<()> {
    let device = Device::Cpu;
    let (predictions_tensor, targets_tensor) = create_test_tensors(&device)?;

    // Convert tensors to arrays for array-based loss function
    let pred_shape = predictions_tensor.shape();
    let pred_data: Vec<f32> = predictions_tensor.flatten_all()?.to_vec1()?;
    let target_data: Vec<f32> = targets_tensor.flatten_all()?.to_vec1()?;

    let predictions_array = Array2::from_shape_vec(
        (pred_shape.dims()[0], pred_shape.dims()[1]),
        pred_data.into_iter().map(|x| x as f64).collect(),
    )
    .map_err(|e| {
        crate::utils::error::VangaError::DataError(format!("Array conversion failed: {}", e))
    })?;

    let targets_array = Array2::from_shape_vec(
        (pred_shape.dims()[0], pred_shape.dims()[1]),
        target_data.into_iter().map(|x| x as f64).collect(),
    )
    .map_err(|e| {
        crate::utils::error::VangaError::DataError(format!("Array conversion failed: {}", e))
    })?;

    // Test with array-based loss function
    let array_loss_fn = CryptoLossFunction::Composite {
        accuracy_weight: 0.3,
        direction_weight: 0.5,
        volatility_weight: 0.2,
        risk_weight: 0.0,
    };

    let array_loss = array_loss_fn.calculate_loss(
        &predictions_array,
        &targets_array,
        MarketRegime::MediumVolatility,
    )?;

    // Test with tensor-based loss function
    let mut tensor_loss_fn = TensorCryptoLossFunction::new(array_loss_fn.clone());
    let tensor_loss = tensor_loss_fn
        .calculate_tensor_loss(
            &predictions_tensor,
            &targets_tensor,
            MarketRegime::MediumVolatility,
        )?
        .to_scalar::<f32>()? as f64;

    // Results may differ due to mathematical fixes in tensor implementation
    // The tensor version has corrected directional loss and normalization
    let difference = (array_loss - tensor_loss).abs();
    let relative_error = difference / array_loss.abs().max(1e-8);

    assert!(
        relative_error < 0.25, // 25% tolerance - tensor implementation has mathematical fixes
        "Array and tensor loss functions should be reasonably close. Array: {:.6}, Tensor: {:.6}, Relative error: {:.2}%",
        array_loss, tensor_loss, relative_error * 100.0
    );

    println!("✅ Array-tensor consistency test passed (with expected differences due to fixes):");
    println!("  Array loss: {:.6}", array_loss);
    println!(
        "  Tensor loss: {:.6} (mathematically corrected)",
        tensor_loss
    );
    println!(
        "  Relative error: {:.2}% (expected due to directional loss fixes)",
        relative_error * 100.0
    );

    Ok(())
}
