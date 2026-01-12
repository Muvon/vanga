//! Layer Normalization Tests for LSTM
//!
//! Tests for Layer Normalization implementation based on Ba et al. (2016)
//! Layer Normalization stabilizes training in deep LSTMs by normalizing
//! activations across features for each sample independently.

use crate::config::model::{LayerNormConfig, ModelConfig};
use crate::model::lstm::config::{LSTMConfig, LSTMModel};
use candle_core::{Device, Tensor};

/// Helper to create test model
fn create_test_model(
    input_size: usize,
    hidden_sizes: Vec<usize>,
    output_size: usize,
) -> Result<LSTMModel, crate::utils::error::VangaError> {
    let num_layers = hidden_sizes.len();
    let config = LSTMConfig {
        input_size,
        hidden_sizes,
        output_size,
        sequence_length: 10,
        learning_rate: 0.001,
        num_layers,
    };
    LSTMModel::new(config)
}

#[test]
fn test_layer_norm_config_defaults() {
    let config = LayerNormConfig::default();
    assert!(!config.enabled, "LayerNorm should be disabled by default");
    assert_eq!(config.epsilon, 1e-5, "Default epsilon should be 1e-5");
    assert!(
        config.lstm_cell,
        "LayerNorm should apply to LSTM cell by default"
    );
    assert_eq!(config.position, "post", "Default position should be post");
}

#[test]
fn test_layer_norm_config_enabled() {
    let config = LayerNormConfig {
        enabled: true,
        epsilon: 1e-4,
        lstm_cell: true,
        position: "pre".to_string(),
    };
    assert!(config.enabled);
    assert_eq!(config.epsilon, 1e-4);
    assert_eq!(config.position, "pre");
}

#[test]
fn test_model_config_includes_layer_norm() {
    let config = ModelConfig::default();
    // ModelConfig should have layer_norm field
    assert!(
        !config.layer_norm.enabled,
        "LayerNorm should be disabled in default ModelConfig"
    );
}

#[test]
fn test_layer_norm_epsilon_boundaries() {
    // Test various epsilon values
    let test_epsilons = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];

    for eps in test_epsilons {
        let config = LayerNormConfig {
            enabled: true,
            epsilon: eps,
            lstm_cell: true,
            position: "post".to_string(),
        };
        assert_eq!(config.epsilon, eps);
    }
}

#[tokio::test]
async fn test_lstm_model_layer_norm_disabled_by_default(
) -> Result<(), crate::utils::error::VangaError> {
    let model = create_test_model(10, vec![64, 32], 5)?;

    // LayerNorm should be disabled by default
    assert!(
        !model.is_layer_norm_enabled(),
        "LayerNorm should be disabled by default"
    );
    assert_eq!(
        model.layer_norm_epsilon(),
        1e-5,
        "Default epsilon should be 1e-5"
    );

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_application() -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32], 5)?;

    // Enable LayerNorm
    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    // Create test input: [batch=2, seq_len=3, features=8]
    let input = Tensor::randn(0.0, 1.0, (2, 3, 8), &Device::Cpu)?;

    // Apply LayerNorm
    let result = model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), 0)?;

    // Output should have same shape as input
    assert_eq!(
        result.dims(),
        input.dims(),
        "Output shape should match input shape"
    );

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_preserves_shape() -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32, 16], 5)?;

    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    // Test various shapes
    let test_cases = vec![
        (1, 5, 32),    // single sample
        (16, 10, 64),  // batch of 16
        (32, 20, 128), // larger batch
    ];

    for (batch, seq, features) in test_cases {
        let input = Tensor::randn(0.0, 1.0, (batch, seq, features), &Device::Cpu)?;
        let result =
            model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), 0)?;

        assert_eq!(
            result.dims(),
            input.dims(),
            "Shape should be preserved for [{}, {}, {}]",
            batch,
            seq,
            features
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_when_disabled() -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32], 5)?;

    // Disable LayerNorm
    model.layer_norm_config = Some(LayerNormConfig {
        enabled: false,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    let input = Tensor::randn(0.0, 1.0, (2, 3, 8), &Device::Cpu)?;
    let result = model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), 0)?;

    // When disabled, should return original input
    // (Currently returns clone, which is equivalent for shape comparison)
    assert_eq!(result.dims(), input.dims());

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_normalizes_correctly() -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32], 5)?;

    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    // Create input with known mean and variance
    // Shape: [1, 1, 4] for easy verification
    let input_data = vec![vec![vec![1.0_f32, 2.0, 3.0, 4.0]]]; // mean = 2.5, std ≈ 1.29
    let input = Tensor::new(input_data, &Device::Cpu)?;

    let result = model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), 0)?;

    // After LayerNorm, mean should be approximately 0 and std approximately 1
    // Extract values using flatten_all and to_vec1
    let flattened = result.flatten_all()?.to_vec1::<f32>()?;
    let mean: f32 = flattened.iter().sum::<f32>() / flattened.len() as f32;
    let variance: f32 =
        flattened.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flattened.len() as f32;

    // Mean should be close to 0
    assert!(
        mean.abs() < 1e-6,
        "Normalized mean should be close to 0, got {}",
        mean
    );

    // Variance should be close to 1 (within epsilon factor)
    assert!(
        (variance - 1.0).abs() < 1e-3,
        "Normalized variance should be close to 1, got {}",
        variance
    );

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_with_different_feature_sizes(
) -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32], 5)?;

    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    // Test with different feature sizes matching typical LSTM hidden sizes
    let feature_sizes = vec![32, 64, 128, 256];

    for features in feature_sizes {
        let input = Tensor::randn(0.0, 1.0, (4, 5, features), &Device::Cpu)?;
        let result =
            model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), 0)?;

        assert_eq!(
            result.dims()[2],
            features,
            "Feature dimension should be preserved for size {}",
            features
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_with_layer_index_logging() -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32, 16], 5)?;

    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    let input = Tensor::randn(0.0, 1.0, (2, 3, 32), &Device::Cpu)?;

    // Apply LayerNorm with different layer indices
    for layer_idx in 0..3 {
        let result =
            model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), layer_idx)?;
        assert_eq!(result.dims(), input.dims());
    }

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_2d_tensor() -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32], 5)?;

    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    // LayerNorm on 2D tensor [batch, features]
    let input = Tensor::randn(0.0, 1.0, (4, 8), &Device::Cpu)?;
    let result = model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), 0)?;

    assert_eq!(result.dims(), input.dims());

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_3d_tensor() -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32], 5)?;

    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    // Standard LSTM output shape: [batch, seq_len, features]
    let input = Tensor::randn(0.0, 1.0, (8, 10, 64), &Device::Cpu)?;
    let result = model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), 0)?;

    assert_eq!(result.dims(), input.dims());

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_4d_tensor() -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32], 5)?;

    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    // 4D tensor [batch, heads, seq_len, head_dim] for multi-head attention
    let input = Tensor::randn(0.0, 1.0, (4, 8, 10, 16), &Device::Cpu)?;
    let result = model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), 0)?;

    assert_eq!(result.dims(), input.dims());

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_with_large_epsilon() -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32], 5)?;

    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-2, // Large epsilon for numerical stability
        lstm_cell: true,
        position: "post".to_string(),
    });

    let input = Tensor::randn(0.0, 1.0, (2, 3, 8), &Device::Cpu)?;
    let result = model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), 0)?;

    assert_eq!(result.dims(), input.dims());

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_numerical_stability() -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32], 5)?;

    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    // Test with very small values - use f32 explicitly
    let small_input = Tensor::randn(0.0_f32, 0.001_f32, (2, 3, 8), &Device::Cpu)?;
    let result =
        model.apply_layer_norm(&small_input, model.layer_norm_config.as_ref().unwrap(), 0)?;

    // Should not NaN or Inf - extract and check values
    let flattened = result.flatten_all()?.to_vec1::<f32>()?;
    assert!(
        flattened.iter().all(|x: &f32| x.is_finite()),
        "Result should be finite"
    );

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_multi_layer_consistency() -> Result<(), crate::utils::error::VangaError> {
    let mut model = create_test_model(10, vec![64, 32, 16], 5)?;

    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    let input = Tensor::randn(0.0_f32, 1.0_f32, (4, 5, 64), &Device::Cpu)?;

    // Apply LayerNorm multiple times (simulating multi-layer LSTM)
    let mut current = input.clone();
    for layer_idx in 0..3 {
        current = model.apply_layer_norm(
            &current,
            model.layer_norm_config.as_ref().unwrap(),
            layer_idx,
        )?;
    }

    // Final shape should still match original
    assert_eq!(current.dims(), input.dims());

    // Output should be finite - extract and check values
    let flattened = current.flatten_all()?.to_vec1::<f32>()?;
    assert!(
        flattened.iter().all(|x: &f32| x.is_finite()),
        "Result should be finite after multiple applications"
    );

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_learnable_parameters_initialization(
) -> Result<(), crate::utils::error::VangaError> {
    use crate::config::model::{
        DropoutConfig, HiddenUnitsConfig, LSTMArchitecture, SequenceLengthConfig,
    };

    // Create ModelConfig with LayerNorm enabled
    let model_config = ModelConfig {
        architecture: LSTMArchitecture::MultiLSTM { layers: 2 },
        sequence_length: SequenceLengthConfig::Fixed(10),
        hidden_units: HiddenUnitsConfig::Fixed(vec![64, 32]),
        layer_norm: LayerNormConfig {
            enabled: true,
            epsilon: 1e-5,
            lstm_cell: true,
            position: "post".to_string(),
        },
        dropout: DropoutConfig::default(),
        attention: crate::config::model::AttentionConfig::default(),
        xgboost: crate::config::model::XGBoostConfig::default(),
        quantile_outputs: None,
        bias_correction: crate::model::bias_correction::BiasCorrection::default(),
    };

    let mut model = LSTMModel::from_model_config(&model_config, 10, 5)?;
    model.initialize_network(None)?;

    // Check that gamma and beta parameters exist in VarMap for each layer
    let var_data = model.varmap.data().lock().unwrap();

    for layer_idx in 0..2 {
        let gamma_key = format!("layer_norm_{}_gamma.gamma", layer_idx);
        let beta_key = format!("layer_norm_{}_beta.beta", layer_idx);

        assert!(
            var_data.contains_key(&gamma_key),
            "Gamma parameter should exist for layer {}",
            layer_idx
        );
        assert!(
            var_data.contains_key(&beta_key),
            "Beta parameter should exist for layer {}",
            layer_idx
        );

        // Check gamma is initialized to 1.0
        let gamma = var_data.get(&gamma_key).unwrap();
        let gamma_vec = gamma.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            gamma_vec.iter().all(|&x| (x - 1.0).abs() < 1e-3),
            "Gamma should be initialized to 1.0 for layer {}",
            layer_idx
        );

        // Check beta is initialized to 0.0
        let beta = var_data.get(&beta_key).unwrap();
        let beta_vec = beta.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            beta_vec.iter().all(|&x| x.abs() < 1e-3),
            "Beta should be initialized to 0.0 for layer {}",
            layer_idx
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_affine_transformation_applied(
) -> Result<(), crate::utils::error::VangaError> {
    use crate::config::model::{
        DropoutConfig, HiddenUnitsConfig, LSTMArchitecture, SequenceLengthConfig,
    };

    let model_config = ModelConfig {
        architecture: LSTMArchitecture::MultiLSTM { layers: 1 },
        sequence_length: SequenceLengthConfig::Fixed(10),
        hidden_units: HiddenUnitsConfig::Fixed(vec![8]),
        layer_norm: LayerNormConfig {
            enabled: true,
            epsilon: 1e-5,
            lstm_cell: true,
            position: "post".to_string(),
        },
        dropout: DropoutConfig::default(),
        attention: crate::config::model::AttentionConfig::default(),
        xgboost: crate::config::model::XGBoostConfig::default(),
        quantile_outputs: None,
        bias_correction: crate::model::bias_correction::BiasCorrection::default(),
    };

    let mut model = LSTMModel::from_model_config(&model_config, 10, 5)?;
    model.initialize_network(None)?;

    // Create test input with known values
    let input_data = vec![vec![vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]];
    let input = Tensor::new(input_data, &Device::Cpu)?;

    // Apply LayerNorm (should use gamma=1.0, beta=0.0 initially)
    let result = model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), 0)?;

    // Result should be normalized (mean≈0, std≈1) since gamma=1, beta=0
    let flattened = result.flatten_all()?.to_vec1::<f32>()?;
    let mean: f32 = flattened.iter().sum::<f32>() / flattened.len() as f32;
    let variance: f32 =
        flattened.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flattened.len() as f32;

    assert!(
        mean.abs() < 1e-5,
        "Mean should be close to 0 with gamma=1, beta=0, got {}",
        mean
    );
    assert!(
        (variance - 1.0).abs() < 1e-3,
        "Variance should be close to 1 with gamma=1, beta=0, got {}",
        variance
    );

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_parameters_per_layer() -> Result<(), crate::utils::error::VangaError> {
    use crate::config::model::{
        DropoutConfig, HiddenUnitsConfig, LSTMArchitecture, SequenceLengthConfig,
    };

    // Create 3-layer model with different hidden sizes
    let model_config = ModelConfig {
        architecture: LSTMArchitecture::MultiLSTM { layers: 3 },
        sequence_length: SequenceLengthConfig::Fixed(10),
        hidden_units: HiddenUnitsConfig::Fixed(vec![64, 32, 16]),
        layer_norm: LayerNormConfig {
            enabled: true,
            epsilon: 1e-5,
            lstm_cell: true,
            position: "post".to_string(),
        },
        dropout: DropoutConfig::default(),
        attention: crate::config::model::AttentionConfig::default(),
        xgboost: crate::config::model::XGBoostConfig::default(),
        quantile_outputs: None,
        bias_correction: crate::model::bias_correction::BiasCorrection::default(),
    };

    let mut model = LSTMModel::from_model_config(&model_config, 10, 5)?;
    model.initialize_network(None)?;

    // Check that each layer has parameters with correct feature size
    let var_data = model.varmap.data().lock().unwrap();
    let expected_sizes = vec![64, 32, 16];

    for (layer_idx, expected_size) in expected_sizes.iter().enumerate() {
        let gamma_key = format!("layer_norm_{}_gamma.gamma", layer_idx);
        let beta_key = format!("layer_norm_{}_beta.beta", layer_idx);

        let gamma = var_data.get(&gamma_key).unwrap();
        let beta = var_data.get(&beta_key).unwrap();

        assert_eq!(
            gamma.dims()[0],
            *expected_size,
            "Gamma size should match hidden size for layer {}",
            layer_idx
        );
        assert_eq!(
            beta.dims()[0],
            *expected_size,
            "Beta size should match hidden size for layer {}",
            layer_idx
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_bidirectional_parameter_size(
) -> Result<(), crate::utils::error::VangaError> {
    use crate::config::model::{
        DropoutConfig, HiddenUnitsConfig, LSTMArchitecture, SequenceLengthConfig,
    };

    // Create bidirectional model
    let model_config = ModelConfig {
        architecture: LSTMArchitecture::BidirectionalLSTM { layers: 2 },
        sequence_length: SequenceLengthConfig::Fixed(10),
        hidden_units: HiddenUnitsConfig::Fixed(vec![64, 32]),
        layer_norm: LayerNormConfig {
            enabled: true,
            epsilon: 1e-5,
            lstm_cell: true,
            position: "post".to_string(),
        },
        dropout: DropoutConfig::default(),
        attention: crate::config::model::AttentionConfig::default(),
        xgboost: crate::config::model::XGBoostConfig::default(),
        quantile_outputs: None,
        bias_correction: crate::model::bias_correction::BiasCorrection::default(),
    };

    let mut model = LSTMModel::from_model_config(&model_config, 10, 5)?;
    model.initialize_network(None)?;

    // For bidirectional, feature size should be 2x hidden size (concatenated)
    let var_data = model.varmap.data().lock().unwrap();
    let expected_sizes = vec![128, 64]; // 2x [64, 32]

    for (layer_idx, expected_size) in expected_sizes.iter().enumerate() {
        let gamma_key = format!("layer_norm_{}_gamma.gamma", layer_idx);
        let gamma = var_data.get(&gamma_key).unwrap();

        assert_eq!(
            gamma.dims()[0],
            *expected_size,
            "Bidirectional gamma size should be 2x hidden size for layer {}",
            layer_idx
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_layer_norm_without_affine_parameters_warns(
) -> Result<(), crate::utils::error::VangaError> {
    // Create model without initializing network (no parameters in VarMap)
    let mut model = create_test_model(10, vec![64], 5)?;
    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: "post".to_string(),
    });

    let input = Tensor::randn(0.0, 1.0, (2, 3, 64), &Device::Cpu)?;

    // Should still work but without affine transformation
    let result = model.apply_layer_norm(&input, model.layer_norm_config.as_ref().unwrap(), 0)?;

    assert_eq!(result.dims(), input.dims());

    Ok(())
}
