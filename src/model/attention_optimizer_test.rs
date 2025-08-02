use crate::model::attention_optimizer::*;

#[test]
fn test_optimized_config_defaults() {
    let config = OptimizedAttentionConfig::default();
    assert!(config.sequence_optimization.use_sparse_attention);
    assert!(config.crypto_optimizations.recent_price_focus);
    assert!(config.memory_optimization.cache_attention_patterns);
}

#[test]
fn test_factory_short_term() {
    let config = OptimizedAttentionFactory::create_short_term_crypto();
    assert!(!config.sequence_optimization.use_sparse_attention);
    assert!(config.crypto_optimizations.recent_price_focus);
    // Note: max_sequence_length is not part of AttentionConfig in config/model.rs
    assert_eq!(config.base_config.heads, 8);
}

#[test]
fn test_factory_long_term() {
    let config = OptimizedAttentionFactory::create_long_term_crypto();
    assert!(config.sequence_optimization.use_sparse_attention);
    assert!(!config.crypto_optimizations.recent_price_focus);
    // Note: max_sequence_length is not part of AttentionConfig in config/model.rs
    assert_eq!(config.base_config.heads, 8);
}
