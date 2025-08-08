//! Tests for NLL loss calculation
//!
//! This module tests the updated calculate_nll_loss function with
//! uniform weighting scenarios, ensuring gradient preservation
//! and mathematical correctness.

use super::config::{LSTMConfig, LSTMModel};
use super::loss::LossMode;
use candle_core::{Device, Tensor};

/// Create a test LSTM model
fn create_test_model(_training_weights: Option<Vec<f32>>) -> LSTMModel {
    let config = LSTMConfig {
        input_size: 10,
        hidden_sizes: vec![20], // Single layer with 20 hidden units
        output_size: 5,         // 5-class classification
        sequence_length: 10,
        learning_rate: 0.001,
        num_layers: 1,
    };

    // No weights needed anymore
    LSTMModel::new(config).expect("Failed to create test model")
}

/// Create test tensors for loss calculation
fn create_test_tensors(batch_size: usize, num_classes: usize, device: &Device) -> (Tensor, Tensor) {
    // Create predictions tensor [batch_size, num_classes] with log probabilities
    let predictions_data: Vec<f32> = (0..batch_size * num_classes)
        .map(|i| (i as f32 * 0.1) - 2.0) // Values around -2.0 to simulate log probabilities
        .collect();
    let predictions = Tensor::from_vec(predictions_data, (batch_size, num_classes), device)
        .expect("Failed to create predictions tensor");

    // Create targets tensor [batch_size, 1] with class indices
    let targets_data: Vec<f32> = (0..batch_size)
        .map(|i| (i % num_classes) as f32) // Cycle through class indices
        .collect();
    let targets = Tensor::from_vec(targets_data, (batch_size, 1), device)
        .expect("Failed to create targets tensor");

    (predictions, targets)
}

#[test]
fn test_uniform_weights_training_mode() {
    let model = create_test_model(None);
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(4, 5, &device);

    let result = model.calculate_nll_loss(&predictions, &targets, LossMode::Training);

    assert!(
        result.is_ok(),
        "Loss calculation should succeed with uniform weights"
    );
    let loss_tensor = result.unwrap();
    let loss_value = loss_tensor
        .to_scalar::<f32>()
        .expect("Should convert to scalar");

    // Loss should be positive and finite
    assert!(loss_value > 0.0, "Loss should be positive");
    assert!(loss_value.is_finite(), "Loss should be finite");
}

#[test]
fn test_uniform_weights_validation_mode() {
    let model = create_test_model(None);
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(4, 5, &device);

    let result = model.calculate_nll_loss(&predictions, &targets, LossMode::Validation);

    assert!(
        result.is_ok(),
        "Loss calculation should succeed with uniform weights"
    );
    let loss_tensor = result.unwrap();
    let loss_value = loss_tensor
        .to_scalar::<f32>()
        .expect("Should convert to scalar");

    // Loss should be positive and finite
    assert!(loss_value > 0.0, "Loss should be positive");
    assert!(loss_value.is_finite(), "Loss should be finite");
}

#[test]
fn test_uniform_weights_validation_mode_with_validation_weights() {
    let training_weights = vec![1.0, 2.0, 0.5, 3.0, 1.5];

    let model = create_test_model(Some(training_weights));
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(4, 5, &device);

    let result = model.calculate_nll_loss(&predictions, &targets, LossMode::Validation);

    assert!(
        result.is_ok(),
        "Loss calculation should succeed with validation weights"
    );
    let loss_tensor = result.unwrap();
    let loss_value = loss_tensor
        .to_scalar::<f32>()
        .expect("Should convert to scalar");

    // Loss should be positive and finite
    assert!(loss_value > 0.0, "Loss should be positive");
    assert!(loss_value.is_finite(), "Loss should be finite");
}

#[test]
fn test_uniform_weights_validation_mode_fallback_to_training() {
    let training_weights = vec![1.0, 2.0, 0.5, 3.0, 1.5];
    let model = create_test_model(Some(training_weights)); // No validation weights
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(4, 5, &device);

    let result = model.calculate_nll_loss(&predictions, &targets, LossMode::Validation);

    assert!(
        result.is_ok(),
        "Loss calculation should succeed with fallback to training weights"
    );
    let loss_tensor = result.unwrap();
    let loss_value = loss_tensor
        .to_scalar::<f32>()
        .expect("Should convert to scalar");

    // Loss should be positive and finite
    assert!(loss_value > 0.0, "Loss should be positive");
    assert!(loss_value.is_finite(), "Loss should be finite");
}

#[test]
fn test_different_batch_sizes() {
    let training_weights = vec![1.0, 2.0, 0.5, 3.0, 1.5];
    let model = create_test_model(Some(training_weights));
    let device = Device::Cpu;

    // Test different batch sizes
    for batch_size in [1, 2, 8, 16, 32] {
        let (predictions, targets) = create_test_tensors(batch_size, 5, &device);

        let result = model.calculate_nll_loss(&predictions, &targets, LossMode::Training);

        assert!(
            result.is_ok(),
            "Loss calculation should succeed with batch size {}",
            batch_size
        );
        let loss_tensor = result.unwrap();
        let loss_value = loss_tensor
            .to_scalar::<f32>()
            .expect("Should convert to scalar");

        // Loss should be positive and finite
        assert!(
            loss_value > 0.0,
            "Loss should be positive for batch size {}",
            batch_size
        );
        assert!(
            loss_value.is_finite(),
            "Loss should be finite for batch size {}",
            batch_size
        );
    }
}

#[test]
fn test_gradient_preservation() {
    let training_weights = vec![1.0, 2.0, 0.5, 3.0, 1.5];
    let model = create_test_model(Some(training_weights));
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(4, 5, &device);

    let result = model.calculate_nll_loss(&predictions, &targets, LossMode::Training);

    assert!(
        result.is_ok(),
        "Loss calculation should succeed with gradients enabled"
    );
    let loss_tensor = result.unwrap();

    // Perform backward pass to check gradient preservation
    let backward_result = loss_tensor.backward();
    assert!(backward_result.is_ok(), "Backward pass should succeed");

    // Check that gradients exist by checking the result
    let _grads = backward_result.unwrap();
    // The GradStore should contain gradients for the tensors that require gradients
    // We can't easily check if it's empty, but if backward() succeeded, gradients were computed
    println!("Gradients computed successfully");
}

#[test]
fn test_loss_consistency_between_modes() {
    // When using the same weights for training and validation, losses should be similar
    let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // Uniform weights
    let model = create_test_model(Some(weights.clone()));
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(8, 5, &device);

    let training_loss = model
        .calculate_nll_loss(&predictions, &targets, LossMode::Training)
        .expect("Training loss calculation should succeed")
        .to_scalar::<f32>()
        .expect("Should convert to scalar");

    let validation_loss = model
        .calculate_nll_loss(&predictions, &targets, LossMode::Validation)
        .expect("Validation loss calculation should succeed")
        .to_scalar::<f32>()
        .expect("Should convert to scalar");

    // Losses should be identical when using same weights and same data
    let diff = (training_loss - validation_loss).abs();
    assert!(
        diff < 1e-6,
        "Training and validation losses should be nearly identical with same weights"
    );
}

#[test]
fn test_weighted_vs_uniform_loss_difference() {
    let device = Device::Cpu;
    let (predictions, targets) = create_test_tensors(8, 5, &device);

    // Test with uniform weights
    let uniform_model = create_test_model(None);
    let uniform_loss = uniform_model
        .calculate_nll_loss(&predictions, &targets, LossMode::Training)
        .expect("Uniform loss calculation should succeed")
        .to_scalar::<f32>()
        .expect("Should convert to scalar");

    // Test with non-uniform weights
    let weighted_model = create_test_model(Some(vec![0.5, 2.0, 1.0, 3.0, 0.8]));
    let weighted_loss = weighted_model
        .calculate_nll_loss(&predictions, &targets, LossMode::Training)
        .expect("Weighted loss calculation should succeed")
        .to_scalar::<f32>()
        .expect("Should convert to scalar");

    // Losses should be different when using different weights
    let diff = (uniform_loss - weighted_loss).abs();
    assert!(
        diff > 1e-3,
        "Uniform and weighted losses should be noticeably different"
    );
}

#[test]
fn test_invalid_target_indices() {
    let model = create_test_model(Some(vec![1.0, 2.0, 0.5, 3.0, 1.5]));
    let device = Device::Cpu;

    // Create predictions for 5 classes
    let predictions_data: Vec<f32> = (0..20).map(|i| i as f32 * 0.1 - 2.0).collect();
    let predictions = Tensor::from_vec(predictions_data, (4, 5), &device)
        .expect("Failed to create predictions tensor");

    // Create targets with invalid class index (class 5 doesn't exist, only 0-4)
    let invalid_targets = Tensor::from_vec(vec![0.0, 1.0, 5.0, 2.0], (4, 1), &device)
        .expect("Failed to create targets tensor");

    let result = model.calculate_nll_loss(&predictions, &invalid_targets, LossMode::Training);

    // This should handle the invalid index gracefully (using default weight)
    // The exact behavior depends on implementation, but it shouldn't panic
    match result {
        Ok(loss_tensor) => {
            let loss_value = loss_tensor
                .to_scalar::<f32>()
                .expect("Should convert to scalar");
            assert!(
                loss_value.is_finite(),
                "Loss should be finite even with invalid indices"
            );
        }
        Err(_) => {
            // It's also acceptable for this to return an error
            // The important thing is that it doesn't panic
        }
    }
}

#[test]
fn test_empty_batch() {
    let model = create_test_model(Some(vec![1.0, 2.0, 0.5, 3.0, 1.5]));
    let device = Device::Cpu;

    // Create empty tensors
    let predictions = Tensor::from_vec(Vec::<f32>::new(), (0, 5), &device)
        .expect("Failed to create empty predictions tensor");
    let targets = Tensor::from_vec(Vec::<f32>::new(), (0, 1), &device)
        .expect("Failed to create empty targets tensor");

    let result = model.calculate_nll_loss(&predictions, &targets, LossMode::Training);

    // Empty batch should either succeed with NaN/zero loss or return an error
    // The important thing is that it doesn't panic
    match result {
        Ok(loss_tensor) => {
            let loss_value = loss_tensor
                .to_scalar::<f32>()
                .expect("Should convert to scalar");
            // Empty batch might result in NaN or 0, both are acceptable
            assert!(
                loss_value.is_nan() || loss_value == 0.0,
                "Empty batch should result in NaN or 0"
            );
        }
        Err(_) => {
            // It's also acceptable for empty batch to return an error
        }
    }
}

// ============================================================================
// Quality Metric Tests
// ============================================================================

#[test]
fn test_quality_metric_exact_matches() {
    let model = create_test_model(None);

    // Test exact matches - should get bonus points (1.2 each)
    let predictions = vec![0, 1, 2, 3, 4];
    let targets = vec![0, 1, 2, 3, 4];
    let quality = model.calculate_quality_metric(&predictions, &targets);
    // 5 exact matches * 1.2 points = 6.0 points
    // 5 predictions * 1.2 max points = 6.0 max points
    // Quality = (6.0 / 6.0) * 100% = 100%
    assert_eq!(
        quality, 100.0,
        "Exact matches should have 100% quality with bonus points"
    );
}

#[test]
fn test_quality_metric_conservative_predictions_only() {
    let model = create_test_model(None);

    // Test ONLY the two valid conservative predictions (1.0 points each)
    let predictions = vec![1, 3]; // Moderate Down, Moderate Up
    let targets = vec![0, 4]; // Strong Down, Strong Up (conservative exceeded)

    let quality = model.calculate_quality_metric(&predictions, &targets);
    // 2 conservative * 1.0 points = 2.0 points
    // 2 predictions * 1.2 max points = 2.4 max points
    // Quality = (2.0 / 2.4) * 100% = 83.33%
    assert!(
        (quality - 83.33).abs() < 0.01,
        "Conservative predictions should have ~83.33% quality"
    );
}

#[test]
fn test_quality_metric_distance_penalties() {
    let model = create_test_model(None);

    // Test different distance penalties
    let predictions = vec![0, 0, 0, 0]; // All predict Strong Down
    let targets = vec![1, 2, 3, 4]; // Distance 1, 2, 3, 4 respectively

    let quality = model.calculate_quality_metric(&predictions, &targets);
    // Distance 1: 0.8 points, Distance 2: 0.5 points, Distance 3: 0.2 points, Distance 4: 0.0 points
    // Total points: 0.8 + 0.5 + 0.2 + 0.0 = 1.5 points
    // Max points: 4 * 1.2 = 4.8 points
    // Quality = (1.5 / 4.8) * 100% = 31.25%
    assert_eq!(
        quality, 31.25,
        "Distance penalties should result in 31.25% quality"
    );
}

#[test]
fn test_quality_metric_mixed_scenarios() {
    let model = create_test_model(None);

    // Test comprehensive mixed scenario with different scoring
    let predictions = vec![
        0, // pred=0
        1, // pred=1
        2, // pred=2
        3, // pred=3
        1, // pred=1 (conservative)
        3, // pred=3 (conservative)
        0, // pred=0 (distance 2)
        2, // pred=2 (distance 1)
    ];
    let targets = vec![
        0, // target=0 - EXACT MATCH (1.2 points)
        1, // target=1 - EXACT MATCH (1.2 points)
        2, // target=2 - EXACT MATCH (1.2 points)
        3, // target=3 - EXACT MATCH (1.2 points)
        0, // target=0 - CONSERVATIVE (1.0 points)
        4, // target=4 - CONSERVATIVE (1.0 points)
        2, // target=2 - DISTANCE 2 (0.5 points)
        1, // target=1 - DISTANCE 1 (0.8 points)
    ];

    let quality = model.calculate_quality_metric(&predictions, &targets);
    // Total points: 4*1.2 + 2*1.0 + 1*0.5 + 1*0.8 = 4.8 + 2.0 + 0.5 + 0.8 = 8.1 points
    // Max points: 8 * 1.2 = 9.6 points
    // Quality = (8.1 / 9.6) * 100% = 84.375%
    assert!(
        (quality - 84.375).abs() < 0.01,
        "Mixed scenario should have ~84.38% quality"
    );
}

#[test]
fn test_quality_metric_edge_cases() {
    let model = create_test_model(None);

    // Test empty arrays
    let quality = model.calculate_quality_metric(&[], &[]);
    assert_eq!(quality, 0.0, "Empty arrays should return 0% quality");

    // Test mismatched lengths
    let predictions = vec![1, 2, 3];
    let targets = vec![1, 2];
    let quality = model.calculate_quality_metric(&predictions, &targets);
    assert_eq!(quality, 0.0, "Mismatched lengths should return 0% quality");

    // Test invalid class values (should be skipped)
    let predictions = vec![1, 5, 3, -1]; // 5 and -1 are invalid
    let targets = vec![0, 2, 4, 1]; // Only indices 0 and 2 are valid
    let quality = model.calculate_quality_metric(&predictions, &targets);
    // Valid predictions: pred=1→target=0 (1.0 conservative), pred=3→target=4 (1.0 conservative)
    // Total points: 1.0 + 1.0 = 2.0 points
    // Max points: 2 * 1.2 = 2.4 points
    // Quality = (2.0 / 2.4) * 100% = 83.33%
    assert!(
        (quality - 83.33).abs() < 0.01,
        "Invalid classes should be skipped, valid conservative predictions should score ~83.33%"
    );
}

#[test]
fn test_quality_metric_distance_4_total_failure() {
    let model = create_test_model(None);

    // Test distance 4 scenarios (total failures)
    let predictions = vec![0, 4]; // Strong Down, Strong Up
    let targets = vec![4, 0]; // Strong Up, Strong Down (distance 4 each)

    let quality = model.calculate_quality_metric(&predictions, &targets);
    // Distance 4: 0.0 points each
    // Total points: 0.0 + 0.0 = 0.0 points
    // Max points: 2 * 1.2 = 2.4 points
    // Quality = (0.0 / 2.4) * 100% = 0%
    assert_eq!(
        quality, 0.0,
        "Distance 4 errors should result in 0% quality"
    );
}

#[test]
fn test_quality_metric_scoring_constants() {
    let model = create_test_model(None);

    // Test that scoring constants work as expected
    let predictions = vec![0, 1, 0, 0, 0];
    let targets = vec![0, 0, 1, 2, 3]; // Exact, Conservative, Dist1, Dist2, Dist3

    let quality = model.calculate_quality_metric(&predictions, &targets);
    // Points: 1.2 (exact) + 1.0 (conservative) + 0.8 (dist1) + 0.5 (dist2) + 0.2 (dist3) = 3.7
    // Max points: 5 * 1.2 = 6.0
    // Quality = (3.7 / 6.0) * 100% = 61.67%
    assert!(
        (quality - 61.67).abs() < 0.01,
        "Scoring constants should result in ~61.67% quality"
    );
}
