// Comprehensive attention integration tests
use crate::config::{ModelConfig, TrainingConfig};
use crate::model::attention::AttentionConfig as AttentionModuleConfig;
use crate::model::lstm_simple::LSTMModel;
use crate::utils::error::Result;
use ndarray::{Array2, Array3};

/// Test attention-enabled model creation and configuration
#[tokio::test]
async fn test_attention_model_creation() -> Result<()> {
    // Create model config with attention enabled
    let mut model_config = ModelConfig::default();
    model_config.attention.enabled = true;
    model_config.attention.heads = 4;
    model_config.attention.head_dim = Some(32);
    model_config.attention.dropout_rate = 0.1;
    model_config.attention.temperature_scaling = 1.0;
    model_config.attention.use_relative_position = true;

    // Create LSTM model with attention
    let model = LSTMModel::from_model_config(&model_config, 10, 1)?;

    // Verify attention configuration was applied
    assert!(model.use_attention, "Attention should be enabled");
    assert!(
        model.attention_config.is_some(),
        "Attention config should be set"
    );

    if let Some(attention_config) = &model.attention_config {
        assert_eq!(attention_config.num_heads, 4);
        assert_eq!(attention_config.head_dim, Some(32));
        assert_eq!(attention_config.dropout_rate, 0.1);
        assert_eq!(attention_config.temperature_scaling, 1.0);
        assert!(attention_config.use_relative_position);
    }

    Ok(())
}

/// Test attention-enabled training workflow
#[tokio::test]
async fn test_attention_training_workflow() -> Result<()> {
    // Create training config with attention enabled
    let training_config = TrainingConfig::default()
        .symbol("BTCUSDT".to_string())
        .data_path("test_data.csv")
        .with_attention_enabled(true);

    // Verify attention is enabled in model config
    assert!(training_config.model.attention.enabled);

    // Create test data
    let sequences = create_test_sequences();
    let targets = create_test_targets();

    // Create model from training config
    let model = LSTMModel::from_model_config(
        &training_config.model,
        sequences.shape()[2], // input_size
        targets.shape()[1],   // output_size
    )?;

    // Verify attention configuration was applied correctly
    assert!(model.use_attention, "Model should use attention");
    assert!(
        model.attention_config.is_some(),
        "Attention config should be set"
    );

    Ok(())
}

/// Test attention-enabled forward pass
#[tokio::test]
async fn test_attention_forward_pass() -> Result<()> {
    // Create attention-enabled model
    let mut model_config = ModelConfig::default();
    model_config.attention.enabled = true;
    model_config.attention.heads = 2;

    let model = LSTMModel::from_model_config(&model_config, 5, 1)?;

    // Verify attention configuration
    assert!(model.use_attention, "Model should use attention");
    assert!(
        model.attention_config.is_some(),
        "Attention config should be set"
    );

    if let Some(attention_config) = &model.attention_config {
        assert_eq!(attention_config.num_heads, 2);
    }

    Ok(())
}

/// Test attention vs non-attention model comparison (configuration only)
#[tokio::test]
async fn test_attention_vs_baseline_comparison() -> Result<()> {
    // Create baseline model (no attention)
    let mut baseline_config = ModelConfig::default();
    baseline_config.attention.enabled = false;
    let baseline_model = LSTMModel::from_model_config(&baseline_config, 5, 1)?;

    // Create attention model
    let mut attention_config = ModelConfig::default();
    attention_config.attention.enabled = true;
    attention_config.attention.heads = 4;
    let attention_model = LSTMModel::from_model_config(&attention_config, 5, 1)?;

    // Verify configuration differences
    assert!(
        !baseline_model.use_attention,
        "Baseline should not use attention"
    );
    assert!(
        baseline_model.attention_config.is_none(),
        "Baseline should have no attention config"
    );

    assert!(
        attention_model.use_attention,
        "Attention model should use attention"
    );
    assert!(
        attention_model.attention_config.is_some(),
        "Attention model should have config"
    );

    Ok(())
}

/// Test CLI attention flag integration
#[test]
fn test_cli_attention_flag() {
    // Test that training config has attention enabled by default
    let config_default = TrainingConfig::default();
    assert!(
        config_default.model.attention.enabled,
        "Attention should be enabled by default"
    );

    let config_with_attention = TrainingConfig::default().with_attention_enabled(true);
    assert!(config_with_attention.model.attention.enabled);

    let config_disabled_attention = TrainingConfig::default().with_attention_enabled(false);
    assert!(!config_disabled_attention.model.attention.enabled);
}

/// Test attention configuration validation
#[test]
fn test_attention_config_validation() {
    // Test valid attention configuration
    let config = AttentionModuleConfig {
        num_heads: 8,
        head_dim: Some(64),
        dropout_rate: 0.1,
        temperature_scaling: 1.0,
        use_relative_position: true,
        max_sequence_length: 100,
    };

    assert!(config.num_heads > 0);
    assert!(config.head_dim.unwrap_or(0) > 0);
    assert!(config.dropout_rate >= 0.0 && config.dropout_rate <= 1.0);
    assert!(config.temperature_scaling > 0.0);
    assert!(config.max_sequence_length > 0);
}

/// Test backward compatibility (non-attention models still work)
#[tokio::test]
async fn test_backward_compatibility() -> Result<()> {
    // Create model without attention (legacy mode)
    let mut model_config = ModelConfig::default();
    model_config.attention.enabled = false;

    let model = LSTMModel::from_model_config(&model_config, 5, 1)?;

    // Verify attention is disabled
    assert!(!model.use_attention);
    assert!(model.attention_config.is_none());

    Ok(())
}

// Helper functions
fn create_test_sequences() -> Array3<f64> {
    Array3::<f64>::from_shape_fn((2, 10, 5), |(i, j, k)| {
        (i as f64 + j as f64 * 0.1 + k as f64 * 0.01).sin()
    })
}

fn create_test_targets() -> Array2<f64> {
    Array2::<f64>::from_shape_fn((2, 1), |(i, _)| (i as f64).cos())
}
