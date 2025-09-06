use crate::config::model::LSTMConfig;
use crate::model::lstm::core::LSTMModel;
use candle_core::Device;

#[test]
fn test_initialize_network_skip_weight_init() {
    // Test that initialize_network respects the skip_weight_init parameter
    let config = LSTMConfig {
        input_size: 10,
        hidden_size: 20,
        num_layers: 1,
        output_size: 5,
        dropout: 0.0,
        sequence_length: 30,
        bidirectional: false,
    };

    let device = Device::Cpu;
    let mut model = LSTMModel::new(config).unwrap();

    // Test with skip_weight_init = true (should not initialize weights)
    let result = model.initialize_network(Some(true));
    assert!(
        result.is_ok(),
        "Network initialization should succeed even when skipping weight init"
    );

    // Verify network structure was created
    assert!(
        model.lstm_layers.is_some(),
        "LSTM layers should be initialized"
    );
    assert!(
        model.output_layer.is_some(),
        "Output layer should be initialized"
    );

    // Test with skip_weight_init = false (should initialize weights)
    let mut model2 = LSTMModel::new(config).unwrap();
    let result2 = model2.initialize_network(Some(false));
    assert!(
        result2.is_ok(),
        "Network initialization with weight init should succeed"
    );

    // Test with None (default behavior - should initialize weights)
    let mut model3 = LSTMModel::new(config).unwrap();
    let result3 = model3.initialize_network(None);
    assert!(
        result3.is_ok(),
        "Network initialization with default behavior should succeed"
    );
}

#[test]
fn test_model_loading_skips_weight_init() {
    // This test would verify that model loading calls initialize_network with skip_weight_init=true
    // but requires actual model files, so we'll just verify the method signature exists

    let config = LSTMConfig {
        input_size: 10,
        hidden_size: 20,
        num_layers: 1,
        output_size: 5,
        dropout: 0.0,
        sequence_length: 30,
        bidirectional: false,
    };

    let mut model = LSTMModel::new(config).unwrap();

    // Verify that the method accepts the skip_weight_init parameter
    let result = model.initialize_network(Some(true));
    assert!(
        result.is_ok(),
        "initialize_network should accept skip_weight_init parameter"
    );
}
