use crate::model::lstm::config::{LSTMConfig, LSTMModel};
use ndarray::{Array2, Array3};

#[tokio::test]
async fn test_validation_metrics_method_signature() {
    // Test that the validation metrics method has correct signature
    let config = LSTMConfig {
        input_size: 5,
        hidden_sizes: vec![16],
        output_size: 5,
        sequence_length: 10,
        learning_rate: 0.001,
        num_layers: 1,
    };

    let mut model = LSTMModel::new(config).unwrap();
    model.initialize_network().unwrap();

    // Create dummy validation data
    let val_sequences = Array3::<f64>::zeros((2, 10, 5));
    let val_targets = Array2::<f64>::zeros((2, 5));
    let training_config = crate::config::TrainingConfig::default();

    // Test that the method can be called (it should return early due to epoch % 5 != 0)
    let result = model
        .calculate_categorical_validation_metrics(
            &val_sequences,
            &val_targets,
            32,
            1, // epoch 1, should return early
            &training_config,
            None, // data_type
        )
        .await;

    assert!(
        result.is_ok(),
        "Validation metrics calculation should succeed"
    );
}
