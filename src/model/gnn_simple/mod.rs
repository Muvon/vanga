// Simplified GNN module for compilation compatibility
// This is a minimal implementation that compiles cleanly

pub mod cross_asset_simple;
pub mod graph_attention_simple;
pub mod regime_detection_simple;

pub use cross_asset_simple::{AssetNode, CrossAssetConfig, CrossAssetGNN, MarketGraph};
pub use graph_attention_simple::{GraphAttentionConfig, GraphAttentionNetwork};
pub use regime_detection_simple::{MarketRegime, RegimeConfig, RegimeDetector};

use crate::utils::error::Result;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

/// GNN configuration for different market scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNNConfig {
    /// Enable cross-asset learning
    pub cross_asset_enabled: bool,
    /// Enable market regime detection
    pub regime_detection_enabled: bool,
    /// Graph attention configuration
    pub graph_attention: GraphAttentionConfig,
    /// Cross-asset learning configuration
    pub cross_asset: CrossAssetConfig,
    /// Regime detection configuration
    pub regime_detection: RegimeConfig,
}

impl Default for GNNConfig {
    fn default() -> Self {
        Self {
            cross_asset_enabled: true,
            regime_detection_enabled: true,
            graph_attention: GraphAttentionConfig::default(),
            cross_asset: CrossAssetConfig::default(),
            regime_detection: RegimeConfig::default(),
        }
    }
}

/// Factory for creating GNN-enhanced models
pub struct GNNFactory;

impl GNNFactory {
    /// Create crypto-optimized GNN configuration
    pub fn crypto_optimized() -> GNNConfig {
        GNNConfig {
            cross_asset_enabled: true,
            regime_detection_enabled: true,
            graph_attention: GraphAttentionConfig::crypto_optimized(),
            cross_asset: CrossAssetConfig::crypto_optimized(),
            regime_detection: RegimeConfig::crypto_optimized(),
        }
    }

    /// Create single-asset GNN configuration (regime detection only)
    pub fn single_asset() -> GNNConfig {
        GNNConfig {
            cross_asset_enabled: false,
            regime_detection_enabled: true,
            graph_attention: GraphAttentionConfig::single_asset(),
            cross_asset: CrossAssetConfig::disabled(),
            regime_detection: RegimeConfig::default(),
        }
    }

    /// Create multi-asset portfolio configuration
    pub fn multi_asset_portfolio() -> GNNConfig {
        GNNConfig {
            cross_asset_enabled: true,
            regime_detection_enabled: true,
            graph_attention: GraphAttentionConfig::multi_asset(),
            cross_asset: CrossAssetConfig::portfolio_optimized(),
            regime_detection: RegimeConfig::portfolio_optimized(),
        }
    }
}

/// Trait for models that can be enhanced with GNN
pub trait GNNCompatible {
    /// Get feature embeddings for graph processing
    fn get_embeddings(&self, input: &Tensor) -> Result<Tensor>;
    /// Apply GNN-enhanced features to model
    fn apply_gnn_features(&mut self, gnn_features: &Tensor) -> Result<()>;
    /// Get model predictions
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnn_config_defaults() {
        let config = GNNConfig::default();
        assert!(config.cross_asset_enabled);
        assert!(config.regime_detection_enabled);
    }

    #[test]
    fn test_gnn_factory_crypto_optimized() {
        let config = GNNFactory::crypto_optimized();
        assert!(config.cross_asset_enabled);
        assert!(config.regime_detection_enabled);
    }

    #[test]
    fn test_gnn_factory_single_asset() {
        let config = GNNFactory::single_asset();
        assert!(!config.cross_asset_enabled);
        assert!(config.regime_detection_enabled);
    }

    #[test]
    fn test_gnn_factory_multi_asset() {
        let config = GNNFactory::multi_asset_portfolio();
        assert!(config.cross_asset_enabled);
        assert!(config.regime_detection_enabled);
    }
}
