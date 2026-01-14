//! LSTM model implementation with modular architecture
//!
//! This module provides a complete LSTM implementation for cryptocurrency
//! price prediction with support for multi-layer, bidirectional, and
//! attention-enhanced architectures.
//!
//! ## Architecture
//!
//! The LSTM implementation is organized into focused modules:
//! - `config`: Configuration structs, enums, and validation
//! - `core`: Model creation, initialization, and persistence
//! - `training`: Training pipeline, optimization, and batch management
//! - `inference`: Prediction pipeline and forward pass
//! - `loss`: Loss calculation, validation metrics, and gradient utilities
//!
//! ## Usage
//!
//! ```rust
//! use crate::model::lstm::{LSTMConfig, LSTMModel};
//!
//! // Create model configuration
//! let config = LSTMConfig {
//!     input_size: 10,
//!     hidden_sizes: vec![64, 32],
//!     output_size: 6,
//!     sequence_length: 60,
//!     learning_rate: 0.001,
//!     num_layers: 2,
//! };
//!
//! // Create and train model
//! let mut model = LSTMModel::new(config)?;
//! model.train(&sequences, &targets, &training_config).await?;
//!
//! // Make predictions
//! let predictions = model.predict(&test_sequences).await?;
//! ```

pub mod config;
pub mod core;
pub mod inference;
pub mod loss;
pub mod schedule_benchmark;
pub mod schedule_validation;
pub mod seeded_weights;
pub mod training;
pub mod window_aware_lr;

#[cfg(test)]
mod hidden_state_test;

#[cfg(test)]
mod layer_norm_position_test;

#[cfg(test)]
mod layer_norm_integration_test;

#[cfg(test)]
mod inference_test;

#[cfg(test)]
mod loss_test;

#[cfg(test)]
#[cfg(test)]
mod reduce_on_plateau_test;

// Re-export main types for backward compatibility

pub use config::{
    LSTMConfig, LSTMModel, ModelState, OptimizerWrapper, TargetFormat, TrainingConfig,
};

// Re-export core functionality including test helpers
#[cfg(test)]
#[allow(unused_imports)]
pub use core::*;

// Re-export window-aware learning rate functionality
pub use window_aware_lr::{create_window_aware_config, WindowAwareLearningRate};

// Re-export ReduceOnPlateauScheduler for testing
pub use training::ReduceOnPlateauScheduler;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{
        AttentionConfig, AttentionMechanism, DropoutConfig, DropoutRate, HiddenUnitsConfig,
        LSTMArchitecture, ModelConfig, SequenceLengthConfig,
    };

    use crate::config::training::OptimizerType;
    use crate::config::training::{EpochConfig, TrainingParams};
    use candle_core::Tensor;
    use ndarray::{Array2, Array3};

    #[tokio::test]
    async fn test_early_stopping_functionality() {
        // Create a simple LSTM model
        let config = LSTMConfig {
            input_size: 3,
            hidden_sizes: vec![8, 8], // Two layers with 8 hidden units each
            output_size: 5,           // Use 5 classes for Direction target
            sequence_length: 5,
            learning_rate: 0.01,
            num_layers: 2, // Default multi-layer
        };

        let mut model = LSTMModel::new(config).expect("Failed to create model");

        // Set target context for regression (single output)
        model.set_target_context(
            "test_target".to_string(),
            crate::targets::TargetType::Direction,
        );

        // Create simple training data (small dataset to trigger early stopping quickly)
        let sequences =
            Array3::from_shape_vec((10, 5, 3), (0..150).map(|i| (i as f64) * 0.1).collect())
                .expect("Failed to create sequences");

        let targets = Array2::from_shape_vec(
            (10, 5),
            (0..50)
                .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
                .collect(),
        )
        .expect("Failed to create targets");

        // Create training config with early stopping enabled
        let training_config = crate::config::TrainingConfig {
            symbol: "TEST".to_string(),
            data_path: std::path::PathBuf::from("test.csv"),
            fresh_training: true,
            continue_training: false,
            horizons: vec!["1h".to_string()],
            features: crate::config::FeatureConfig::default(),
            model: crate::config::ModelConfig::default(),
            targets: crate::config::training::TargetsConfig::default(),
            training: TrainingParams {
                epochs: EpochConfig::Auto { max_epochs: 100 },
                batch_size: crate::config::training::BatchSizeConfig::Fixed(32),
                learning_rate: 0.01,
                optimizer: crate::config::training::OptimizerType::AdamW {
                    weight_decay: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                },
                warmup_epochs: 0, // No warmup for tests
                learning_schedule: None,
                test_split: 0.1,
                early_stopping: crate::config::training::EarlyStoppingConfig {
                    patience: 10,
                    min_delta: 0.0001,
                },
                gradient_clip: Some(1.0),
                validation_split: 0.2,            // 20% validation
                validation_gap: "1h".to_string(), // Default gap for tests
                device: crate::config::training::DeviceConfig::Auto,
                print_every: 1, // Add missing print_every field

                window_decay: 1.0,        // No decay for tests
                min_train_ratio: 0.4,     // Add missing min_train_ratio field
                min_increment_ratio: 0.3, // Add missing min_increment_ratio field
                seed: 42,                 // Fixed seed for reproducible tests
            },
            data: crate::config::training::DataConfig::default(),
        };

        // Test that early stopping training completes without errors
        let result = model
            .train(&sequences, &targets, &training_config, None, None)
            .await;

        if let Err(ref e) = result {
            println!("Training error: {:?}", e);
        }

        assert!(
            result.is_ok(),
            "Early stopping training should complete successfully: {:?}",
            result.err()
        );
        assert!(
            model.trained,
            "Model should be marked as trained after early stopping"
        );
    }

    #[tokio::test]
    async fn test_fixed_epochs_fallback() {
        // Test that fixed epoch configuration bypasses early stopping
        let config = LSTMConfig {
            input_size: 3,
            hidden_sizes: vec![8, 8], // Two layers with 8 hidden units each
            output_size: 5,           // Use 5 classes for Direction target
            sequence_length: 5,
            learning_rate: 0.01,
            num_layers: 2, // Default multi-layer
        };

        let mut model = LSTMModel::new(config).expect("Failed to create model");

        // Set target context for regression (single output)
        model.set_target_context(
            "test_target".to_string(),
            crate::targets::TargetType::Direction,
        );

        // Create simple training data
        let sequences =
            Array3::from_shape_vec((8, 5, 3), (0..120).map(|i| (i as f64) * 0.1).collect())
                .expect("Failed to create sequences");

        let targets = Array2::from_shape_vec(
            (8, 5),
            (0..40)
                .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
                .collect(),
        )
        .expect("Failed to create targets");

        // Create training config with fixed epochs (should bypass early stopping)
        let training_config = crate::config::TrainingConfig {
            symbol: "TEST".to_string(),
            data_path: std::path::PathBuf::from("test.csv"),
            fresh_training: true,
            continue_training: false,
            horizons: vec!["1h".to_string()],
            features: crate::config::FeatureConfig::default(),
            model: crate::config::ModelConfig::default(),
            targets: crate::config::training::TargetsConfig::default(),
            data: crate::config::training::DataConfig::default(),
            training: TrainingParams {
                epochs: EpochConfig::Fixed(5), // Fixed epochs - should bypass early stopping
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
                validation_split: 0.2,
                validation_gap: "1h".to_string(), // Default gap for tests
                device: crate::config::training::DeviceConfig::Auto,
                window_decay: 1.0,        // No decay for tests
                min_train_ratio: 0.4,     // Add missing min_train_ratio field
                min_increment_ratio: 0.3, // Add missing min_increment_ratio field
                test_split: 0.0,
                early_stopping: crate::config::training::EarlyStoppingConfig {
                    patience: 10,
                    min_delta: 0.0001,
                },
                gradient_clip: Some(1.0),
                print_every: 1, // Add missing print_every field

                seed: 42, // Fixed seed for reproducible tests
            },
        };

        // Test that fixed epochs training completes without errors
        let result = model
            .train(&sequences, &targets, &training_config, None, None)
            .await;

        assert!(
            result.is_ok(),
            "Fixed epochs training should complete successfully"
        );
        assert!(
            model.trained,
            "Model should be marked as trained after fixed epochs training"
        );
    }

    #[tokio::test]
    async fn test_model_save_load_predict_workflow() {
        use std::path::PathBuf;
        use tempfile::tempdir;

        // Create a temporary directory for the test
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let model_path = temp_dir.path().join("test_model");

        // Step 1: Create and train a model
        let config = LSTMConfig {
            input_size: 3,
            hidden_sizes: vec![8, 8], // Two layers with 8 hidden units each
            output_size: 5,           // Use 5 classes for Direction target
            sequence_length: 5,
            learning_rate: 0.01,
            num_layers: 2, // Default multi-layer
        };

        let mut model = LSTMModel::new(config).expect("Failed to create model");

        // Set target context for regression (single output)
        model.set_target_context(
            "test_target".to_string(),
            crate::targets::TargetType::Direction,
        );

        // Create training data
        let sequences =
            Array3::from_shape_vec((10, 5, 3), (0..150).map(|i| (i as f64) * 0.1).collect())
                .expect("Failed to create sequences");

        let targets = Array2::from_shape_vec(
            (10, 5),
            (0..50)
                .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
                .collect(),
        )
        .expect("Failed to create targets");

        // Train the model with fixed epochs for quick testing
        let training_config = crate::config::TrainingConfig {
            symbol: "TEST".to_string(),
            data_path: PathBuf::from("test.csv"),
            fresh_training: true,
            continue_training: false,
            horizons: vec!["1h".to_string()],
            features: crate::config::FeatureConfig::default(),
            model: crate::config::ModelConfig::default(),
            targets: crate::config::training::TargetsConfig::default(),
            training: TrainingParams {
                epochs: EpochConfig::Fixed(3), // Quick training for test
                batch_size: crate::config::training::BatchSizeConfig::Fixed(32),
                learning_rate: 0.01,
                optimizer: OptimizerType::SGD { momentum: None },
                warmup_epochs: 0,
                learning_schedule: None,
                validation_split: 0.0,           // No validation for this test
                validation_gap: "0".to_string(), // No gap needed when no validation
                test_split: 0.0,
                early_stopping: crate::config::training::EarlyStoppingConfig {
                    patience: 10,
                    min_delta: 0.0001,
                },
                gradient_clip: Some(1.0),
                device: crate::config::training::DeviceConfig::Auto,
                print_every: 1, // Add missing print_every field

                window_decay: 1.0,        // No decay for tests
                min_train_ratio: 0.4,     // Add missing min_train_ratio field
                min_increment_ratio: 0.3, // Add missing min_increment_ratio field
                seed: 42,                 // Fixed seed for reproducible tests
            },

            data: crate::config::training::DataConfig::default(),
        };

        model
            .train(&sequences, &targets, &training_config, None, None)
            .await
            .expect("Training should complete successfully");

        // Step 2: Save the model
        model.save(&model_path).expect("Model save should succeed");

        // Step 3: Load the model
        let loaded_model = LSTMModel::load(&model_path).expect("Model load should succeed");

        // Step 4: Test prediction with loaded model
        let prediction_result = loaded_model.predict(&sequences).await;

        assert!(
            prediction_result.is_ok(),
            "Prediction with loaded model should succeed"
        );

        let predictions = prediction_result.unwrap();
        assert_eq!(
            predictions.nrows(),
            sequences.shape()[0],
            "Should predict for all sequences"
        );
        assert_eq!(
            predictions.ncols(),
            5,
            "Should have 5 output columns for Direction target"
        );

        // Verify that the loaded model is properly initialized
        assert!(
            loaded_model.trained,
            "Loaded model should be marked as trained"
        );
        assert!(
            loaded_model.lstm_layers.is_some(),
            "Loaded model should have initialized LSTM stack"
        );
        assert!(
            loaded_model.output_layer.is_some(),
            "Loaded model should have initialized output layer"
        );
    }

    #[tokio::test]
    async fn test_multi_layer_lstm_functionality() {
        // Test multi-layer LSTM creation and training
        let config = LSTMConfig {
            input_size: 4,
            hidden_sizes: vec![16, 16, 16], // Three layers with 16 hidden units each
            output_size: 5,                 // Use 5 classes for Direction target
            sequence_length: 10,
            learning_rate: 0.01,
            num_layers: 3, // Test 3-layer LSTM
        };

        let mut model = LSTMModel::new(config).expect("Failed to create multi-layer model");

        // Set target context for regression (single output)
        model.set_target_context(
            "test_target".to_string(),
            crate::targets::TargetType::Direction,
        );

        // Create training data with more complexity for multi-layer testing
        let sequences =
            Array3::from_shape_vec((20, 10, 4), (0..800).map(|i| (i as f64) * 0.01).collect())
                .expect("Failed to create sequences");

        let targets = Array2::from_shape_vec(
            (20, 5),
            (0..100)
                .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
                .collect(),
        )
        .expect("Failed to create targets");

        // Create training config for multi-layer testing
        let training_config = crate::config::TrainingConfig {
            symbol: "TEST_MULTI".to_string(),
            data_path: std::path::PathBuf::from("test_multi.csv"),
            fresh_training: true,
            continue_training: false,
            horizons: vec!["1h".to_string()],
            features: crate::config::FeatureConfig::default(),
            model: crate::config::ModelConfig {
                architecture: crate::config::model::LSTMArchitecture::StackedLSTM { layers: 3 },
                ..crate::config::ModelConfig::default()
            },
            targets: crate::config::training::TargetsConfig::default(),
            training: TrainingParams {
                epochs: EpochConfig::Fixed(5), // Quick training for test
                batch_size: crate::config::training::BatchSizeConfig::Fixed(16),
                learning_rate: 0.01,
                optimizer: crate::config::training::OptimizerType::AdamW {
                    weight_decay: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                },
                warmup_epochs: 0, // No warmup for tests
                learning_schedule: None,
                validation_split: 0.2,
                validation_gap: "1h".to_string(), // Default gap for tests
                device: crate::config::training::DeviceConfig::Auto,
                test_split: 0.0,
                early_stopping: crate::config::training::EarlyStoppingConfig {
                    patience: 10,
                    min_delta: 0.0001,
                },
                gradient_clip: Some(1.0),
                print_every: 1, // Add missing print_every field

                window_decay: 1.0,        // No decay for tests
                min_train_ratio: 0.4,     // Add missing min_train_ratio field
                min_increment_ratio: 0.3, // Add missing min_increment_ratio field
                seed: 42,                 // Fixed seed for reproducible tests
            },
            data: crate::config::training::DataConfig::default(),
        };

        // Test multi-layer training
        let result = model
            .train(&sequences, &targets, &training_config, None, None)
            .await;

        assert!(
            result.is_ok(),
            "Multi-layer LSTM training should complete successfully"
        );
        assert!(
            model.trained,
            "Multi-layer model should be marked as trained"
        );

        // Test prediction with multi-layer model
        let prediction_result = model.predict(&sequences).await;
        assert!(
            prediction_result.is_ok(),
            "Multi-layer prediction should succeed"
        );

        let predictions = prediction_result.unwrap();
        assert_eq!(
            predictions.nrows(),
            sequences.shape()[0],
            "Should predict for all sequences"
        );
        assert_eq!(
            predictions.ncols(),
            5,
            "Should have 5 output columns for Direction target"
        );

        // Verify multi-layer architecture is properly initialized
        assert!(
            model.lstm_layers.is_some(),
            "Multi-layer LSTM stack should be initialized"
        );
        assert_eq!(
            model.config.num_layers, 3,
            "Model should have 3 layers as configured"
        );

        // Verify multi-layer architecture is properly initialized
        assert!(
            model.lstm_layers.is_some(),
            "Multi-layer LSTM layers should be initialized"
        );
    }

    #[tokio::test]
    async fn test_bidirectional_lstm_initialization() {
        // Create a bidirectional LSTM configuration
        let model_config = ModelConfig {
            architecture: LSTMArchitecture::BidirectionalLSTM { layers: 2 },
            sequence_length: SequenceLengthConfig::Fixed(10),
            hidden_units: HiddenUnitsConfig::Fixed(vec![32, 16]),
            layer_norm: crate::config::model::LayerNormConfig::default(),
            dropout: DropoutConfig {
                enabled: false,
                rate: DropoutRate::Fixed(0.0),
                variational: false,
                recurrent: false,
            },
            attention: AttentionConfig {
                enabled: true,
                mechanism: crate::config::model::AttentionMechanism::MultiHeadAttention,
                heads: 8,
                head_dim: Some(64),
                dropout_rate: 0.1,
                dropout_weights: true,
                dropout_output: true,
                dropout_projections: true,
                dropout_scores: true,
                temperature_scaling: 1.0,
                use_relative_position: true,
                visualization: crate::config::model::VisualizationConfig::default(),
                moh: None,
            },
            xgboost: crate::config::model::XGBoostConfig::default(),
            bias_correction: crate::model::bias_correction::BiasCorrection::default(),
            quantile_outputs: None,
            dain: None,
        };

        // Create model with bidirectional architecture
        let input_size = 10;
        let output_size = 5;

        let mut model =
            LSTMModel::from_model_config(&model_config, input_size, output_size).unwrap();

        // Verify architecture is stored
        assert!(matches!(
            model.architecture,
            Some(LSTMArchitecture::BidirectionalLSTM { .. })
        ));

        // Initialize the network - this should create both forward and backward layers
        model.initialize_network(None).unwrap(); // Default behavior (with weight init)

        // Verify both forward and backward layers are created
        assert!(model.lstm_layers.is_some());
        assert!(model.backward_lstm_layers.is_some());

        let forward_layers = model.lstm_layers.as_ref().unwrap();
        let backward_layers = model.backward_lstm_layers.as_ref().unwrap();

        // Should have 2 layers each
        assert_eq!(forward_layers.len(), 2);
        assert_eq!(backward_layers.len(), 2);

        println!("✅ Bidirectional LSTM initialization test passed!");
    }

    #[tokio::test]
    async fn test_bidirectional_lstm_forward_pass() {
        // Create a bidirectional LSTM configuration
        let model_config = ModelConfig {
            architecture: LSTMArchitecture::BidirectionalLSTM { layers: 1 },
            sequence_length: SequenceLengthConfig::Fixed(5),
            hidden_units: HiddenUnitsConfig::Fixed(vec![8]),
            layer_norm: crate::config::model::LayerNormConfig::default(),
            dropout: DropoutConfig {
                enabled: false,
                rate: DropoutRate::Fixed(0.0),
                variational: false,
                recurrent: false,
            },
            attention: AttentionConfig {
                enabled: true,
                mechanism: crate::config::model::AttentionMechanism::MultiHeadAttention,
                heads: 8,
                head_dim: Some(64),
                dropout_rate: 0.1,
                dropout_weights: true,
                dropout_output: true,
                dropout_projections: true,
                dropout_scores: true,
                temperature_scaling: 1.0,
                use_relative_position: true,
                visualization: crate::config::model::VisualizationConfig::default(),
                moh: None,
            },
            quantile_outputs: None,
            dain: None,
            xgboost: crate::config::model::XGBoostConfig::default(),
            bias_correction: crate::model::bias_correction::BiasCorrection::default(),
        };

        let input_size = 4;
        let expected_output_size = 15;

        let mut model =
            LSTMModel::from_model_config(&model_config, input_size, expected_output_size).unwrap();

        model.initialize_network(None).unwrap(); // Default behavior (with weight init)

        // Mark model as trained for prediction (bypass training requirement for test)
        model.trained = true;

        // Clear target context to get raw output shape instead of converted class indices
        model.target_context = None;

        // Create test input data: [batch_size=2, seq_len=5, features=4]
        let batch_size = 2;
        let seq_len = 5;
        let features = 4;

        let mut input_data = Array3::<f64>::zeros((batch_size, seq_len, features));

        // Fill with some test data
        for i in 0..batch_size {
            for j in 0..seq_len {
                for k in 0..features {
                    input_data[[i, j, k]] = (i as f64 + j as f64 * 0.1 + k as f64 * 0.01) * 0.1;
                }
            }
        }

        // Test forward pass directly to get raw output shape (not converted to class indices)
        // Convert input data to tensor manually using the same logic as predict method
        let batch_size = input_data.shape()[0];
        let seq_len = input_data.shape()[1];
        let features = input_data.shape()[2];

        let mut seq_data: Vec<f32> = Vec::with_capacity(batch_size * seq_len * features);

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                for feature_idx in 0..features {
                    seq_data.push(input_data[[batch_idx, seq_idx, feature_idx]] as f32);
                }
            }
        }

        let input_tensor = Tensor::from_vec(
            seq_data,
            (batch_size, seq_len, features),
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let predictions = model.forward(&input_tensor, false).unwrap();

        // Verify output shape
        let shape_dims = predictions.shape().dims();
        assert_eq!(shape_dims, &[batch_size, expected_output_size]);

        println!("✅ Bidirectional LSTM forward pass test passed!");
        println!(
            "   Input shape: [{}, {}, {}]",
            batch_size, seq_len, features
        );
        println!("   Output shape: {:?}", predictions.shape().dims());
        println!("   Expected output size: {}", expected_output_size);
    }

    #[tokio::test]
    async fn test_unidirectional_vs_bidirectional_output_size() {
        let input_size = 6;
        let output_size = 4;
        let hidden_size = 12;

        // Test unidirectional LSTM
        let unidirectional_config = ModelConfig {
            architecture: LSTMArchitecture::MultiLSTM { layers: 1 },
            sequence_length: SequenceLengthConfig::default(),
            hidden_units: HiddenUnitsConfig::Fixed(vec![hidden_size]),
            layer_norm: crate::config::model::LayerNormConfig::default(),
            dropout: DropoutConfig::default(),
            attention: AttentionConfig {
                enabled: true,
                mechanism: crate::config::model::AttentionMechanism::MultiHeadAttention,
                heads: 8,
                head_dim: Some(64),
                dropout_rate: 0.1,
                dropout_weights: true,
                dropout_output: true,
                dropout_projections: true,
                dropout_scores: true,
                temperature_scaling: 1.0,
                use_relative_position: true,
                visualization: crate::config::model::VisualizationConfig::default(),
                moh: None,
            },
            quantile_outputs: None,
            dain: None,
            xgboost: crate::config::model::XGBoostConfig::default(),
            bias_correction: crate::model::bias_correction::BiasCorrection::default(),
        };

        // Test bidirectional LSTM
        let bidirectional_config = ModelConfig {
            architecture: LSTMArchitecture::BidirectionalLSTM { layers: 1 },
            ..unidirectional_config.clone()
        };

        let mut uni_model =
            LSTMModel::from_model_config(&unidirectional_config, input_size, output_size).unwrap();
        let mut bi_model =
            LSTMModel::from_model_config(&bidirectional_config, input_size, output_size).unwrap();

        uni_model.initialize_network(None).unwrap(); // Default behavior (with weight init)
        bi_model.initialize_network(None).unwrap(); // Default behavior (with weight init)

        // Verify unidirectional has no backward layers
        assert!(uni_model.backward_lstm_layers.is_none());

        // Verify bidirectional has backward layers
        assert!(bi_model.backward_lstm_layers.is_some());

        println!("✅ Unidirectional vs Bidirectional architecture test passed!");
    }

    #[tokio::test]
    async fn test_bidirectional_lstm_with_attention() {
        // Test bidirectional LSTM with attention enabled
        let model_config = ModelConfig {
            architecture: LSTMArchitecture::BidirectionalLSTM { layers: 1 },
            sequence_length: SequenceLengthConfig::Fixed(5),
            hidden_units: HiddenUnitsConfig::Fixed(vec![8]),
            layer_norm: crate::config::model::LayerNormConfig::default(),
            dropout: DropoutConfig {
                enabled: false,
                rate: DropoutRate::Fixed(0.0),
                variational: false,
                recurrent: false,
            },
            attention: AttentionConfig {
                enabled: true, // Enable attention
                mechanism: AttentionMechanism::SelfAttention,
                heads: 2,
                head_dim: Some(8),
                dropout_rate: 0.0,
                dropout_weights: false,
                dropout_output: false,
                dropout_projections: false,
                dropout_scores: false,
                temperature_scaling: 1.0,
                use_relative_position: false,
                visualization: crate::config::model::VisualizationConfig::default(),
                moh: None,
            },
            quantile_outputs: None,
            dain: None,
            xgboost: crate::config::model::XGBoostConfig::default(),
            bias_correction: crate::model::bias_correction::BiasCorrection::default(),
        };

        // Create a simple test to verify the config is valid
        let input_size = 4;
        let expected_output_size = 15;

        let mut model =
            LSTMModel::from_model_config(&model_config, input_size, expected_output_size).unwrap();
        model.initialize_network(None).unwrap(); // Default behavior (with weight init)

        println!("✅ Attention integration test configuration created!");
    }
}
