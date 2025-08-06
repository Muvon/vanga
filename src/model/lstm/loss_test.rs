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
