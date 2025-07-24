// Integration tests for Candle migration
// Validates that new implementation maintains exact same behavior

#[cfg(test)]
mod candle_integration_tests {
    use crate::config::{ModelConfig, TrainingConfig};
    use crate::data::TensorConverter;
    use crate::model::{LSTMConfig, LSTMModel, MultiTargetLSTMModel};
    use ndarray::{Array2, Array3};
    use tempfile::TempDir;

    /// Test that Candle LSTM produces same interface as rust-lstm
    #[tokio::test]
    async fn test_candle_lstm_interface_compatibility() {
        // Create test configuration
        let config = LSTMConfig {
            input_size: 10,
            hidden_size: 32,
            output_size: 1,
            sequence_length: 20,
            learning_rate: 0.001,
        };

        // Create model - should work exactly like rust-lstm version
        let mut model = LSTMModel::new(config).expect("Failed to create model");

        // Create test data
        let sequences = Array3::zeros((5, 20, 10)); // 5 samples, 20 timesteps, 10 features
        let targets = Array2::zeros((5, 1)); // 5 samples, 1 target

        // Test training - same interface
        let result = model.train(&sequences, &targets).await;
        assert!(result.is_ok(), "Training should succeed");

        // Test prediction - same interface
        let predictions = model.predict(&sequences).await;
        assert!(predictions.is_ok(), "Prediction should succeed");

        let pred_array = predictions.unwrap();
        assert_eq!(pred_array.shape(), &[5, 1], "Prediction shape should match");

        // Test serialization - same interface
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.bin");

        let save_result = model.save(&model_path);
        assert!(save_result.is_ok(), "Model save should succeed");

        // Test loading - same interface
        let loaded_model = LSTMModel::load(&model_path);
        assert!(loaded_model.is_ok(), "Model load should succeed");

        let loaded = loaded_model.unwrap();
        assert_eq!(
            loaded.get_input_size(),
            10,
            "Input size should be preserved"
        );
    }

    /// Test multi-target model compatibility
    #[tokio::test]
    async fn test_multi_target_interface_compatibility() {
        // Create test model config
        let model_config = create_test_model_config();
        let target_names = vec![
            "price".to_string(),
            "direction".to_string(),
            "volatility".to_string(),
        ];

        // Create multi-target model - same interface as original
        let mut model = MultiTargetLSTMModel::new(
            &model_config,
            15,
            target_names.clone(),
            vec!["1h".to_string()],
        )
        .expect("Failed to create multi-target model");

        // Create test data
        let sequences = Array3::zeros((8, 30, 15)); // 8 samples, 30 timesteps, 15 features
        let targets = Array2::zeros((8, 3)); // 8 samples, 3 targets

        // Create training config
        let training_config = create_test_training_config();

        // Test training - same interface
        // Test training - same interface
        let result = model
            .train(
                crate::model::TrainingContext::Standard {
                    sequences: &sequences,
                    targets: &targets,
                    val_sequences: None,
                    val_targets: None,
                },
                &training_config,
            )
            .await;

        // Test prediction - same interface
        let predictions = model.predict(&sequences).await;
        assert!(
            predictions.is_ok(),
            "Multi-target prediction should succeed"
        );

        let pred_array = predictions.unwrap();
        assert_eq!(
            pred_array.shape(),
            &[8, 3],
            "Multi-target prediction shape should match"
        );

        // Test metadata - same interface
        assert_eq!(model.get_target_names(), &target_names);
        assert_eq!(model.get_input_size(), 15);
        assert_eq!(model.get_num_targets(), 3);

        // Test serialization - same interface
        let temp_dir = TempDir::new().unwrap();
        let result = model.save(temp_dir.path()).await;
        assert!(result.is_ok(), "Multi-target model save should succeed");

        // Test loading - same interface
        let loaded_model = MultiTargetLSTMModel::load(temp_dir.path()).await;
        assert!(
            loaded_model.is_ok(),
            "Multi-target model load should succeed"
        );
    }

    /// Test advanced training methods compatibility
    #[tokio::test]
    async fn test_advanced_training_compatibility() {
        let config = LSTMConfig {
            input_size: 8,
            hidden_size: 16,
            output_size: 1,
            sequence_length: 15,
            learning_rate: 0.001,
        };

        let mut model = LSTMModel::new(config).unwrap();
        let training_config = create_test_training_config();

        // Test data
        let sequences = Array3::zeros((6, 15, 8));
        let targets = Array2::zeros((6, 1));

        // Test parallel batch training - same interface
        let result = model.train_parallel_batches(&sequences, &targets, 2).await;
        assert!(result.is_ok(), "Parallel batch training should work");

        // Test early stopping training - same interface
        // Test early stopping training - same interface
        let result = model
            .train(
                crate::model::TrainingContext::Standard {
                    sequences: &sequences,
                    targets: &targets,
                    val_sequences: None,
                    val_targets: None,
                },
                &training_config,
            )
            .await;

        // Test incremental training - same interface
        let new_sequences = Array3::zeros((3, 15, 8));
        let new_targets = Array2::zeros((3, 1));

        let result = model
            .train(
                crate::model::Training_Context::Continue {
                    new_sequences: &new_sequences,
                    new_targets: &new_targets,
                },
                &training_config,
            )
            .await;

        // Test retrain with appended data - same interface
        let result = model
            .retrain_with_appended_data(
                &sequences,
                &targets,
                &new_sequences,
                &new_targets,
                &training_config,
            )
            .await;
        assert!(result.is_ok(), "Retrain with appended data should work");
    }

    /// Test tensor conversion utilities
    #[test]
    fn test_tensor_conversion_utilities() {
        let converter = TensorConverter::new();

        // Test 3D array conversion
        let sequences = Array3::zeros((4, 10, 5));
        let tensor_result = converter.array3_to_tensor(&sequences);
        assert!(tensor_result.is_ok(), "3D tensor conversion should work");

        // Test 2D array conversion
        let targets = Array2::zeros((4, 2));
        let tensor_result = converter.array2_to_tensor(&targets);
        assert!(tensor_result.is_ok(), "2D tensor conversion should work");

        // Test round-trip conversion
        let original = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor = converter.array2_to_tensor(&original).unwrap();
        let recovered = converter.tensor_to_array2(&tensor).unwrap();

        // Check values are preserved (within floating point precision)
        for i in 0..2 {
            for j in 0..3 {
                let diff = (original[[i, j]] - recovered[[i, j]]).abs();
                assert!(diff < 1e-6, "Round-trip conversion should preserve values");
            }
        }
    }

    /// Performance comparison test (optional - for validation)
    #[tokio::test]
    #[ignore] // Run manually for performance validation
    async fn test_performance_comparison() {
        use std::time::Instant;

        let config = LSTMConfig {
            input_size: 50,
            hidden_size: 128,
            output_size: 1,
            sequence_length: 60,
            learning_rate: 0.001,
        };

        // Large dataset for performance testing
        let sequences = Array3::zeros((1000, 60, 50));
        let targets = Array2::zeros((1000, 1));

        let mut model = LSTMModel::new(config).unwrap();

        // Time training
        let start = Instant::now();
        let result = model.train(&sequences, &targets).await;
        let training_time = start.elapsed();

        assert!(result.is_ok(), "Large dataset training should succeed");
        println!("Candle training time: {:?}", training_time);

        // Time prediction
        let start = Instant::now();
        let predictions = model.predict(&sequences).await;
        let prediction_time = start.elapsed();

        assert!(
            predictions.is_ok(),
            "Large dataset prediction should succeed"
        );
        println!("Candle prediction time: {:?}", prediction_time);

        // Basic performance expectations
        assert!(
            training_time.as_secs() < 60,
            "Training should complete within reasonable time"
        );
        assert!(prediction_time.as_secs() < 5, "Prediction should be fast");
    }

    /// Test error handling compatibility
    #[tokio::test]
    async fn test_error_handling_compatibility() {
        let config = LSTMConfig {
            input_size: 5,
            hidden_size: 10,
            output_size: 1,
            sequence_length: 8,
            learning_rate: 0.001,
        };

        let mut model = LSTMModel::new(config).unwrap();

        // Test prediction before training - should fail gracefully
        let sequences = Array3::zeros((2, 8, 5));
        let result = model.predict(&sequences).await;
        assert!(result.is_err(), "Prediction before training should fail");

        // Test mismatched dimensions - should fail gracefully
        let bad_sequences = Array3::zeros((2, 8, 10)); // Wrong feature count
        let targets = Array2::zeros((2, 1));
        let result = model.train(&bad_sequences, &targets).await;
        assert!(
            result.is_err(),
            "Training with wrong dimensions should fail"
        );

        // Test empty data - should fail gracefully
        let empty_sequences = Array3::zeros((0, 8, 5));
        let empty_targets = Array2::zeros((0, 1));
        let result = model.train(&empty_sequences, &empty_targets).await;
        assert!(result.is_err(), "Training with empty data should fail");
    }

    // Helper functions for test setup
    fn create_test_model_config() -> ModelConfig {
        ModelConfig {
            hidden_units: crate::config::model::HiddenUnitsConfig::Fixed(vec![64]),
            sequence_length: crate::config::model::SequenceLengthConfig::Fixed(30),
            output_heads: crate::config::model::OutputHeadsConfig {
                price_levels: crate::config::model::PriceLevelHead {
                    enabled: true,
                    range_percent: 0.05,
                },
                direction: crate::config::model::DirectionHead {
                    enabled: true,
                    threshold: 0.001,
                },
                volatility: crate::config::model::VolatilityHead {
                    enabled: true,
                    horizons: vec![1, 4, 24],
                },
            },
        }
    }

    fn create_test_training_config() -> TrainingConfig {
        TrainingConfig {
            symbol: "BTCUSDT".to_string(),
            model: create_test_model_config(),
            training_params: crate::config::training::TrainingParams {
                epochs: crate::config::training::EpochConfig::Fixed(10),
                learning_rate: crate::config::training::LearningRateConfig::Fixed(0.001),
                batch_size: 16,
                validation_split: 0.2,
                early_stopping_patience: 5,
            },
            data: crate::config::data::DataConfig {
                file_path: "test.csv".to_string(),
                target_column: "close".to_string(),
                feature_columns: vec![],
                date_column: Some("timestamp".to_string()),
                validation: crate::config::data::DataValidationConfig {
                    min_rows: 100,
                    max_missing_percent: 0.1,
                    required_columns: vec!["timestamp".to_string(), "close".to_string()],
                },
            },
            feature_config: crate::config::features::FeatureConfig {
                technical_indicators: crate::config::features::TechnicalIndicatorsConfig {
                    enabled: true,
                    indicators: vec![],
                },
                custom_features: crate::config::features::CustomFeaturesConfig {
                    enabled: false,
                    features: vec![],
                },
            },
        }
    }
}

/// Benchmark tests for performance validation
#[cfg(test)]
mod benchmark_tests {
    use super::*;
    use criterion::{black_box, Criterion};

    pub fn benchmark_tensor_conversion(c: &mut Criterion) {
        let converter = crate::data::TensorConverter::new();
        let sequences = Array3::zeros((100, 60, 50));

        c.bench_function("tensor_conversion_3d", |b| {
            b.iter(|| {
                let _tensor = converter.array3_to_tensor(black_box(&sequences)).unwrap();
            })
        });
    }

    pub fn benchmark_model_prediction(c: &mut Criterion) {
        let rt = tokio::runtime::Runtime::new().unwrap();

        c.bench_function("candle_lstm_prediction", |b| {
            b.to_async(&rt).iter(|| async {
                let config = crate::model::LSTMConfig {
                    input_size: 50,
                    hidden_size: 64,
                    output_size: 1,
                    sequence_length: 60,
                    learning_rate: 0.001,
                };

                let mut model = crate::model::LSTMModel::new(config).unwrap();
                let sequences = Array3::zeros((10, 60, 50));
                let targets = Array2::zeros((10, 1));

                // Quick training
                let _ = model
                    .train(black_box(&sequences), black_box(&targets))
                    .await;

                // Benchmark prediction
                let _predictions = model.predict(black_box(&sequences)).await.unwrap();
            })
        });
    }
}

// Add to Cargo.toml dev-dependencies if not already present:
// criterion = { version = "0.5", features = ["html_reports"] }
