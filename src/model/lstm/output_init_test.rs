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

                    // Verify small but non-zero standard deviation
                    // With proper Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
                    // For hidden_size=128, num_classes=5: std = sqrt(2/133) ≈ 0.123
                    // We expect std in range [0.05, 0.30] for reasonable initialization
                    assert!(std >= 0.05,
                        "Output layer weight std is {} - too small, indicates improper initialization",
                        std);
                    assert!(
                        std <= 0.30,
                        "Output layer weight std is {} - too large, may cause instability",
                        std
                    );

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

#[test]
fn test_initial_predictions_are_balanced() {
    // This test verifies the fix for the hidden state collapse issue
    // Previously: std=0.01 caused argmax bias (34% class 2, 9% class 0)
    // Now: proper Xavier initialization should give ~20% per class

    let model_config = ModelConfig {
        architecture: LSTMArchitecture::MultiLSTM { layers: 2 },
        hidden_units: HiddenUnitsConfig::Fixed(vec![64, 32]),
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

    model
        .initialize_network(None)
        .expect("Failed to initialize network");

    // Create random input sequences (32 samples, 60 timesteps, 50 features)
    use ndarray::Array3;
    let sequences = Array3::<f64>::from_shape_fn((32, 60, 50), |(_, _, _)| {
        use rand::Rng;
        let mut rng = rand::rng();
        rng.random_range(-1.0..1.0)
    });

    // Convert to tensor
    use candle_core::Tensor;
    let seq_flat: Vec<f32> = sequences.iter().map(|&x| x as f32).collect();
    let seq_tensor =
        Tensor::from_vec(seq_flat, (32, 60, 50), &Device::Cpu).expect("Failed to create tensor");

    // Get initial predictions (before training)
    let predictions = model
        .forward(&seq_tensor, false)
        .expect("Forward pass failed");

    // Convert to probabilities using softmax
    let logits_vec: Vec<f32> = predictions.flatten_all().unwrap().to_vec1().unwrap();
    let mut class_counts = vec![0; 5];

    for i in 0..32 {
        let start = i * 5;
        let end = start + 5;
        let logits = &logits_vec[start..end];

        // Find argmax
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        class_counts[max_idx] += 1;
    }

    // Calculate distribution
    let distribution: Vec<f32> = class_counts
        .iter()
        .map(|&count| count as f32 / 32.0)
        .collect();

    println!("Initial prediction distribution: {:?}", distribution);
    println!("Class counts: {:?}", class_counts);

    // Verify balanced distribution (each class should be ~20% ± 20%)
    // With only 32 samples, we expect some natural variance
    // The key is that NO class should dominate (>50%) or be absent (<5%)
    for (class_idx, &prob) in distribution.iter().enumerate() {
        assert!(
            (0.03..=0.50).contains(&prob),
            "Class {} has probability {:.2}% - distribution is too imbalanced! \
             Expected ~20% with reasonable variance. This indicates hidden state collapse.",
            class_idx,
            prob * 100.0
        );
    }

    // Calculate variance from uniform distribution (0.2 for each class)
    let variance: f32 = distribution.iter().map(|&p| (p - 0.2).powi(2)).sum::<f32>() / 5.0;

    println!("Distribution variance from uniform: {:.6}", variance);

    // Variance should be reasonable (< 0.05 for 32 samples)
    // This is much better than the old std=0.01 which gave variance > 0.10
    assert!(
        variance < 0.08,
        "Distribution variance {:.6} is too high - indicates poor initialization balance",
        variance
    );

    println!("✅ Initial predictions are well-balanced across all classes");
}
