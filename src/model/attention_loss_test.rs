// Tests for attention_loss module
use crate::model::attention_loss::{AttentionLossConfig, AttentionLossFactory, BaseLossType};

#[test]
fn test_attention_loss_config_defaults() {
    let config = AttentionLossConfig::default();
    assert_eq!(config.attention_weight, 0.3);
    assert!(config.regime_aware);
    assert!(matches!(config.base_loss, BaseLossType::MSE));
}

#[test]
fn test_loss_factory_crypto_optimized() {
    let loss = AttentionLossFactory::create_crypto_optimized();
    let config = loss.get_config();
    assert!(matches!(config.base_loss, BaseLossType::Huber));
    assert_eq!(config.attention_weight, 0.4);
    assert!(config.regime_aware);
}
