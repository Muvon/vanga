//! Integration test for Layer Normalization configuration loading

use crate::config::model::{
    AttentionConfig, DropoutConfig, DropoutRate, HiddenUnitsConfig, LSTMArchitecture,
    LayerNormConfig, LayerNormPosition, ModelConfig, SequenceLengthConfig,
};
use crate::model::lstm::config::LSTMModel;

#[tokio::test]
async fn test_layer_norm_config_propagation() {
    // Create ModelConfig with LayerNorm enabled
    let model_config = ModelConfig {
        architecture: LSTMArchitecture::MultiLSTM { layers: 3 },
        sequence_length: SequenceLengthConfig::Fixed(60),
        hidden_units: HiddenUnitsConfig::Fixed(vec![128, 64, 32]),
        layer_norm: LayerNormConfig {
            enabled: true,
            epsilon: 1e-5,
            lstm_cell: true,
            position: LayerNormPosition::Post,
        },
        dropout: DropoutConfig {
            enabled: true,
            rate: DropoutRate::Fixed(0.3),
            variational: false,
            recurrent: false,
        },
        attention: AttentionConfig {
            enabled: false,
            mechanism: crate::config::model::AttentionMechanism::SelfAttention,
            heads: 4,
            head_dim: Some(32),
            dropout_rate: 0.1,
            dropout_weights: false,
            dropout_output: false,
            dropout_projections: false,
            dropout_scores: false,
            temperature_scaling: 1.0,
            use_relative_position: false,
            visualization: crate::config::model::VisualizationConfig::default(),
            moh: None,
        },
        xgboost: crate::config::model::XGBoostConfig::default(),
        quantile_outputs: None,
        bias_correction: crate::model::bias_correction::BiasCorrection::default(),
    };

    // Create model from config
    let model = LSTMModel::from_model_config(&model_config, 10, 5).unwrap();

    // Verify layer_norm_config is propagated
    assert!(
        model.layer_norm_config.is_some(),
        "layer_norm_config should be set from ModelConfig"
    );

    let layer_norm = model.layer_norm_config.as_ref().unwrap();
    assert!(layer_norm.enabled, "LayerNorm should be enabled");
    assert_eq!(layer_norm.epsilon, 1e-5, "Epsilon should match config");
    assert!(layer_norm.lstm_cell, "lstm_cell should be true");
    assert_eq!(
        layer_norm.position,
        LayerNormPosition::Post,
        "Position should be 'post'"
    );

    // Verify helper methods work
    assert!(
        model.is_layer_norm_enabled(),
        "is_layer_norm_enabled() should return true"
    );
    assert_eq!(
        model.layer_norm_epsilon(),
        1e-5,
        "layer_norm_epsilon() should return correct value"
    );
}

#[tokio::test]
async fn test_layer_norm_disabled_by_default() {
    // Create ModelConfig with default LayerNorm (disabled)
    let model_config = ModelConfig {
        architecture: LSTMArchitecture::MultiLSTM { layers: 2 },
        sequence_length: SequenceLengthConfig::Fixed(60),
        hidden_units: HiddenUnitsConfig::Fixed(vec![64, 32]),
        layer_norm: LayerNormConfig::default(), // Default = disabled
        dropout: DropoutConfig::default(),
        attention: AttentionConfig {
            enabled: false,
            mechanism: crate::config::model::AttentionMechanism::SelfAttention,
            heads: 4,
            head_dim: Some(32),
            dropout_rate: 0.1,
            dropout_weights: false,
            dropout_output: false,
            dropout_projections: false,
            dropout_scores: false,
            temperature_scaling: 1.0,
            use_relative_position: false,
            visualization: crate::config::model::VisualizationConfig::default(),
            moh: None,
        },
        xgboost: crate::config::model::XGBoostConfig::default(),
        quantile_outputs: None,
        bias_correction: crate::model::bias_correction::BiasCorrection::default(),
    };

    let model = LSTMModel::from_model_config(&model_config, 10, 5).unwrap();

    // Verify layer_norm_config is set but disabled
    assert!(
        model.layer_norm_config.is_some(),
        "layer_norm_config should be set"
    );
    assert!(
        !model.is_layer_norm_enabled(),
        "LayerNorm should be disabled by default"
    );
}
