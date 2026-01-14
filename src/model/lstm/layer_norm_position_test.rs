//! Layer Normalization Position Tests
//!
//! Tests for LayerNormPosition enum and Pre/Post normalization support

use crate::config::model::{LayerNormConfig, LayerNormPosition};
use crate::model::lstm::config::{LSTMConfig, LSTMModel};
use candle_core::{Device, Tensor};

fn create_test_model() -> Result<LSTMModel, crate::utils::error::VangaError> {
    let lstm_config = LSTMConfig {
        input_size: 3,
        hidden_sizes: vec![3],
        output_size: 1,
        sequence_length: 10,
        learning_rate: 0.001,
        num_layers: 1,
    };
    let mut model = LSTMModel::new(lstm_config)?;
    model.layer_norm_config = Some(LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: LayerNormPosition::Post,
    });
    Ok(model)
}

#[test]
fn test_layer_norm_position_enum_default() {
    let position: LayerNormPosition = LayerNormPosition::Post;
    assert_eq!(position, LayerNormPosition::Post);
}

#[test]
fn test_layer_norm_position_pre() {
    let position = LayerNormPosition::Pre;
    assert_eq!(position, LayerNormPosition::Pre);
}

#[test]
fn test_layer_norm_position_display() {
    assert_eq!(format!("{}", LayerNormPosition::Post), "post");
    assert_eq!(format!("{}", LayerNormPosition::Pre), "pre");
}

#[test]
fn test_layer_norm_position_from_str() {
    let pre: LayerNormPosition = "pre".parse().unwrap();
    let post: LayerNormPosition = "post".parse().unwrap();
    assert_eq!(pre, LayerNormPosition::Pre);
    assert_eq!(post, LayerNormPosition::Post);
}

#[test]
fn test_layer_norm_config_with_position() {
    let config = LayerNormConfig {
        enabled: true,
        epsilon: 1e-5,
        lstm_cell: true,
        position: LayerNormPosition::Pre,
    };
    assert!(config.enabled);
    assert_eq!(config.position, LayerNormPosition::Pre);
}

#[test]
fn test_layer_norm_config_default_position() {
    let config = LayerNormConfig::default();
    assert_eq!(config.position, LayerNormPosition::Post);
}

#[test]
fn test_apply_layer_norm_with_position_post() -> Result<(), crate::utils::error::VangaError> {
    let model = create_test_model()?;
    let input = Tensor::randn(0.0, 1.0, (2, 3, 8), &Device::Cpu)?;
    let result = model.apply_layer_norm(
        &input,
        model.layer_norm_config.as_ref().unwrap(),
        0,
        LayerNormPosition::Post,
    )?;
    assert_eq!(result.dims(), input.dims());
    Ok(())
}

#[test]
fn test_apply_layer_norm_with_position_pre() -> Result<(), crate::utils::error::VangaError> {
    let model = create_test_model()?;
    let input = Tensor::randn(0.0, 1.0, (2, 3, 8), &Device::Cpu)?;
    let result = model.apply_layer_norm(
        &input,
        model.layer_norm_config.as_ref().unwrap(),
        0,
        LayerNormPosition::Pre,
    )?;
    assert_eq!(result.dims(), input.dims());
    Ok(())
}

#[test]
fn test_apply_layer_norm_both_positions_produce_output(
) -> Result<(), crate::utils::error::VangaError> {
    let model = create_test_model()?;
    let input = Tensor::randn(0.0, 1.0, (4, 5, 16), &Device::Cpu)?;

    let result_post = model.apply_layer_norm(
        &input,
        model.layer_norm_config.as_ref().unwrap(),
        0,
        LayerNormPosition::Post,
    )?;

    let result_pre = model.apply_layer_norm(
        &input,
        model.layer_norm_config.as_ref().unwrap(),
        0,
        LayerNormPosition::Pre,
    )?;

    // Both should produce valid output
    assert_eq!(result_post.dims(), input.dims());
    assert_eq!(result_pre.dims(), input.dims());
    Ok(())
}
