use crate::config::model::ModelConfig;
use crate::model::multi_target::*;
use ndarray::Array3;

#[tokio::test]
async fn test_multi_target_creation() {
    let model_config = ModelConfig::default();
    let target_names = vec![
        "price_1h".to_string(),
        "direction".to_string(),
        "volatility".to_string(),
    ];

    let result = MultiTargetLSTMModel::new(
        &model_config,
        10,
        target_names.clone(),
        vec!["1h".to_string()],
    );
    assert!(result.is_ok());

    let model = result.unwrap();
    assert_eq!(model.get_num_targets(), 3);
    assert_eq!(model.get_input_size(), 10);
    assert_eq!(model.get_target_names(), &target_names);
}

#[tokio::test]
async fn test_multi_target_training_validation() {
    let model_config = ModelConfig::default();
    let target_names = vec!["target1".to_string(), "target2".to_string()];
    let mut model =
        MultiTargetLSTMModel::new(&model_config, 5, target_names, vec!["1h".to_string()]).unwrap();

    // Create a test config
    let config = crate::config::TrainingConfig {
        symbol: "BTCUSDT".to_string(),
        data_path: std::path::PathBuf::from("test.csv"),
        fresh_training: true,
        continue_training: false,
        horizons: vec!["1h".to_string()],
        features: crate::config::FeatureConfig::default(),
        model: ModelConfig::default(),
        training: crate::config::training::TrainingParams {
            epochs: crate::config::training::EpochConfig::Fixed(1),
            batch_size: crate::config::training::BatchSizeConfig::Fixed(32),
            learning_rate: 0.01,
            optimizer: crate::config::training::OptimizerType::AdamW {
                weight_decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
            warmup_epochs: 0,
            learning_schedule: None,
            validation_split: 0.0,
            validation_gap: "0".to_string(), // No gap needed when no validation
            test_split: 0.0,
            window_decay: 1.0, // No decay for tests
            early_stopping: crate::config::training::EarlyStoppingConfig {
                patience: 10,
                min_delta: 0.0001,
            },
            device: crate::config::training::DeviceConfig::Auto,
            gradient_clip: Some(1.0),
            print_every: 1, // Add missing print_every field
            class_weight_strategy: crate::config::training::ClassWeightStrategy::Global, // Add missing class_weight_strategy field
        },
        data: crate::config::training::DataConfig::default(),
        optimization: crate::config::training::OptimizationConfig::default(),
    };

    // Create test data with wrong target dimensions
    let sequences = Array3::zeros((10, 30, 5)); // [batch, seq_len, features]
    let wrong_targets = Array2::zeros((10, 3)); // Wrong: 3 targets instead of 2

    let result = model
        .train(
            TrainingContext::Standard {
                sequences: &sequences,
                targets: &wrong_targets,
                val_sequences: None,
                val_targets: None,
                target_class_weights: None,
            },
            &config,
        )
        .await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Target dimension mismatch"));
}
