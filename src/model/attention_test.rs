// Tests for attention module
use crate::config::model::AttentionConfig;
use crate::model::attention::MultiHeadAttention;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

#[test]
fn test_attention_config_defaults() {
    let config = AttentionConfig::default();
    assert_eq!(config.heads, 8);
    assert_eq!(config.head_dim, Some(64));
    assert!(config.use_relative_position);
}

#[tokio::test]
async fn test_attention_creation() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = AttentionConfig::default();
    let attention = MultiHeadAttention::new(64, config, vs, device);
    assert!(attention.is_ok());
}
