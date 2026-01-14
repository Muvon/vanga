// Test to verify output layer initialization is correct
// This test catches the bug where output layer was initialized to zeros
use crate::config::model::{
    AttentionConfig, DropoutConfig, HiddenUnitsConfig, LSTMArchitecture, LayerNormConfig,
};
use crate::config::ModelConfig;
use crate::model::lstm::LSTMModel;
use candle_core::Device;

#[test]
fn test_output_layer_initialization_not_zero() {
    let model_config = ModelConfig {
        architecture: LSTMArchitecture::MultiLSTM { layers: 2 },
        hidden_units: HiddenUnitsConfig::Fixed(vec![128, 128]),
        sequence_length: crate::config::model::SequenceLengthConfig::Fixed(60),
        layer_norm: LayerNormConfig::default(),
        dropout: DropoutConfig::default(),
        attention: AttentionConfig::default(),
        bias_correction: crate::model::bias_correction::BiasCorrection::default(),
        xgboost: crate::config::model::XGBoostConfig::default(),
        dain: None,
        quantile_outputs: None,
    };

    // Create a new model using correct API
    let mut model = LSTMModel::from_model_config_with_seed(
        &model_config,
        50, // input_size
        5,  // output_size (num_classes)
        Some(42),
        Some(Device::Cpu),
    )
    .expect("Failed to create model");

    // CRITICAL: Initialize the network to create the VarMap tensors
    model
        .initialize_network(None)
        .expect("Failed to initialize network");

    // Get the varmap and check output layer weights
    let var_data = model.varmap.data().lock().unwrap();

    // Find output.weight tensor
    let mut found_output_weight = false;

    for (name, var) in var_data.iter() {
        if name.contains("output") && name.contains("weight") {
            found_output_weight = true;
            let tensor = var.as_tensor();
            if let Ok(flattened) = tensor.flatten_all() {
                if let Ok(values) = flattened.to_vec1::<f32>() {
                    let mean = values.iter().sum::<f32>() / values.len() as f32;
                    let std = (values.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                        / values.len() as f32)
                        .sqrt();

                    // Verify NOT zero initialization
                    assert!(mean.abs() > 1e-6,
                        "Output layer weight mean is {} - likely initialized to ZERO! This is the bug.",
                        mean);

                    // Verify small but non-zero standard deviation (0.01 for small output weights)
                    // This prevents large hidden state activations from creating class bias
                    assert!(std >= 0.005,
                        "Output layer weight std is {} - too small, indicates non-random initialization",
                        std);

                    println!(
                        "✅ Output layer '{}': shape={:?}, mean={:.6}, std={:.6}",
                        name,
                        tensor.shape(),
                        mean,
                        std
                    );
                }
            }
            break;
        }
    }

    assert!(
        found_output_weight,
        "Could not find output.weight tensor in VarMap"
    );
}

#[test]
fn test_output_layer_bias_initialization_not_anti_middle() {
    let model_config = ModelConfig {
        architecture: LSTMArchitecture::MultiLSTM { layers: 2 },
        hidden_units: HiddenUnitsConfig::Fixed(vec![128, 128]),
        sequence_length: crate::config::model::SequenceLengthConfig::Fixed(60),
        layer_norm: LayerNormConfig::default(),
        dropout: DropoutConfig::default(),
        attention: AttentionConfig::default(),
        bias_correction: crate::model::bias_correction::BiasCorrection::default(),
        xgboost: crate::config::model::XGBoostConfig::default(),
        dain: None,
        quantile_outputs: None,
    };

    let mut model =
        LSTMModel::from_model_config_with_seed(&model_config, 50, 5, Some(42), Some(Device::Cpu))
            .expect("Failed to create model");

    // CRITICAL: Initialize the network to create the VarMap tensors
    model
        .initialize_network(None)
        .expect("Failed to initialize network");

    let var_data = model.varmap.data().lock().unwrap();

    // Find output.bias tensor
    let mut found_output_bias = false;

    for (name, var) in var_data.iter() {
        if name.contains("output") && name.contains("bias") {
            found_output_bias = true;
            let tensor = var.as_tensor();
            if let Ok(values) = tensor.to_vec1::<f32>() {
                // Should be uniform [0.05, 0.05, 0.05, 0.05, 0.05], NOT [0.1, 0.05, -0.2, 0.05, 0.1]
                let all_equal = values.iter().all(|&v| (v - values[0]).abs() < 0.001);

                assert!(all_equal,
                    "Output bias is not uniform: {:?}. This may indicate anti-middle-class initialization (bug).", 
                    values);

                println!("✅ Output bias '{}': {:?}", name, values);
            }
            break;
        }
    }

    assert!(
        found_output_bias,
        "Could not find output.bias tensor in VarMap"
    );
}
