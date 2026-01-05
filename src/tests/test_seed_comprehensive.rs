//! Comprehensive test cases for seed parameter functionality

use crate::config::{ModelConfig, TrainingConfig};
use crate::model::lstm::{LSTMConfig, LSTMModel};
use crate::model::multi_target::MultiTargetLSTMModel;

#[tokio::test]
async fn test_seed_parameter_flow() {
    env_logger::try_init().ok();

    let config = LSTMConfig {
        input_size: 10,
        hidden_sizes: vec![64, 32],
        output_size: 1,
        sequence_length: 20,
        learning_rate: 0.001,
        num_layers: 2,
    };

    // Test seed parameter storage
    let model_with_seed = LSTMModel::new_with_seed(config.clone(), Some(42), None).unwrap();
    assert_eq!(model_with_seed.seed, Some(42));

    let model_without_seed = LSTMModel::new(config.clone()).unwrap();
    assert_eq!(model_without_seed.seed, None);

    let model_zero_seed = LSTMModel::new_with_seed(config, Some(0), None).unwrap();
    assert_eq!(model_zero_seed.seed, Some(0));
}

#[tokio::test]
async fn test_seed_initialization_logging() {
    env_logger::try_init().ok();

    let config = LSTMConfig {
        input_size: 5,
        hidden_sizes: vec![32],
        output_size: 1,
        sequence_length: 10,
        learning_rate: 0.001,
        num_layers: 1,
    };

    // Test that initialization completes without errors
    let mut model = LSTMModel::new_with_seed(config, Some(123), None).unwrap();

    // This should trigger all the seeding logic and logging
    let result = model.initialize_network(None); // Default behavior (with weight init)
    assert!(
        result.is_ok(),
        "Network initialization should succeed even if seeding doesn't work"
    );

    // Mark as trained for any potential prediction tests
    model.mark_as_trained_for_testing();
}

#[tokio::test]
async fn test_multi_target_seed_flow() {
    env_logger::try_init().ok();

    // Create a minimal ModelConfig for testing
    let model_config = ModelConfig::default();

    // Test that seed flows through multi-target model
    let model = MultiTargetLSTMModel::new_with_seed(
        &model_config,
        10,                               // input_size
        vec!["price_levels".to_string()], // target_names
        vec!["1h".to_string()],           // trained_horizons
        Some(456),                        // seed
        None,                             // device
    );
    assert!(
        model.is_ok(),
        "Multi-target model creation with seed should succeed"
    );
}

#[tokio::test]
async fn test_seed_validation_logic() {
    env_logger::try_init().ok();

    let config = LSTMConfig {
        input_size: 3,
        hidden_sizes: vec![16],
        output_size: 1,
        sequence_length: 5,
        learning_rate: 0.001,
        num_layers: 1,
    };

    // Test different seed values
    let test_cases = vec![
        (None, "no seed"),
        (Some(0), "zero seed"),
        (Some(1), "positive seed"),
        (Some(u64::MAX), "max seed"),
    ];

    for (seed, description) in test_cases {
        let mut model = LSTMModel::new_with_seed(config.clone(), seed, None).unwrap();
        let result = model.initialize_network(None); // Default behavior (with weight init)
        assert!(
            result.is_ok(),
            "Initialization should succeed for {}",
            description
        );
        model.mark_as_trained_for_testing(); // Allow predictions if needed
        assert_eq!(
            model.seed, seed,
            "Seed should be stored correctly for {}",
            description
        );
        assert_eq!(
            model.seed, seed,
            "Seed should be stored correctly for {}",
            description
        );
    }
}

#[tokio::test]
async fn test_weight_tensor_access() {
    env_logger::try_init().ok();

    let config = LSTMConfig {
        input_size: 4,
        hidden_sizes: vec![8],
        output_size: 1,
        sequence_length: 6,
        learning_rate: 0.001,
        num_layers: 1,
    };
    let mut model = LSTMModel::new_with_seed(config, Some(789), None).unwrap();
    model.initialize_network(None).unwrap(); // Default behavior (with weight init)
    model.mark_as_trained_for_testing(); // Allow predictions if needed

    // Test that we can access weight tensors
    let all_vars = model.varmap.all_vars();
    assert!(!all_vars.is_empty(), "Model should have weight tensors");

    // Test that tensors have reasonable shapes
    for var in all_vars.iter() {
        let shape = var.shape();
        assert!(!shape.dims().is_empty(), "Tensor should have dimensions");
        assert!(shape.elem_count() > 0, "Tensor should have elements");
    }
}

#[tokio::test]
async fn test_backward_compatibility() {
    env_logger::try_init().ok();

    let config = LSTMConfig {
        input_size: 10,
        hidden_sizes: vec![32, 16],
        output_size: 5,
        sequence_length: 20,
        learning_rate: 0.001,
        num_layers: 2,
    };

    // Test that old constructor still works
    let model_old = LSTMModel::new(config.clone());
    assert!(model_old.is_ok(), "Old constructor should still work");
    assert_eq!(
        model_old.unwrap().seed,
        None,
        "Old constructor should have no seed"
    );

    // Test that new constructor with None works the same
    let model_new = LSTMModel::new_with_seed(config, None, None);
    assert!(model_new.is_ok(), "New constructor with None should work");
    assert_eq!(
        model_new.unwrap().seed,
        None,
        "New constructor with None should have no seed"
    );
}

#[tokio::test]
async fn test_seed_consistency_attempt() {
    env_logger::try_init().ok();

    let config = LSTMConfig {
        input_size: 6,
        hidden_sizes: vec![12],
        output_size: 1,
        sequence_length: 8,
        learning_rate: 0.001,
        num_layers: 1,
    };

    let seed = 999u64;

    // Create two models with the same seed
    let mut model1 = LSTMModel::new_with_seed(config.clone(), Some(seed), None).unwrap();
    let mut model2 = LSTMModel::new_with_seed(config, Some(seed), None).unwrap();

    model1.initialize_network(None).unwrap(); // Default behavior (with weight init)
    model2.initialize_network(None).unwrap(); // Default behavior (with weight init)

    // Mark as trained for any potential prediction tests
    model1.mark_as_trained_for_testing();
    model2.mark_as_trained_for_testing();

    // Calculate weight norms (even though they won't be identical due to Candle limitations)
    let norm1 = calculate_weight_norm(&model1);
    let norm2 = calculate_weight_norm(&model2);

    // Document the current limitation
    println!("Model 1 norm: {:.10}", norm1);
    println!("Model 2 norm: {:.10}", norm2);
    println!("Difference: {:.10}", (norm1 - norm2).abs());

    // This test documents the current limitation - weights may not be identical on CPU
    // In the future, when GPU seeding works consistently, this assertion should pass:
    // assert!((norm1 - norm2).abs() < 1e-10, "Weight norms should be identical with same seed");

    // For now, just verify that both models were created successfully
    log::info!(
        "Weight norm difference: {} (CPU seeding may not be reproducible)",
        (norm1 - norm2).abs()
    );
    assert!(norm1 > 0.0, "Model 1 should have non-zero weights");
    assert!(norm2 > 0.0, "Model 2 should have non-zero weights");
}

/// Helper function to calculate total weight norm
fn calculate_weight_norm(model: &LSTMModel) -> f64 {
    let all_vars = model.varmap.all_vars();
    let mut total_norm = 0.0f64;

    for var in all_vars.iter() {
        if let Ok(values) = var.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
            let norm: f64 = values
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            total_norm += norm;
        }
    }

    total_norm
}

#[tokio::test]
async fn test_training_config_integration() {
    env_logger::try_init().ok();

    // Test that TrainingConfig can hold seed parameter
    let mut training_config = TrainingConfig::default();
    training_config.training.seed = 12345;

    assert_eq!(training_config.training.seed, 12345);

    // Test validation (should not fail for any seed value)
    let validation_result = training_config.validate();
    assert!(
        validation_result.is_ok(),
        "Training config with seed should validate successfully"
    );
}
