// Graph Neural Network module for cross-asset learning and market regime detection
// Builds on existing TFT Variable Selection Network

pub mod cross_asset;
pub mod graph_attention;
pub mod regime_detection;

pub use cross_asset::{
    AssetNode, CrossAssetConfig, CrossAssetGNN, MarketGraph, MarketGraphFactory,
};
pub use graph_attention::{GraphAttentionConfig, GraphAttentionNetwork};
pub use regime_detection::{MarketRegime, RegimeConfig, RegimeDetector};

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

/// GNN-enhanced model that wraps existing TFT components
pub struct GNNEnhancedModel {
    /// Base TFT model with variable selection
    base_model: Box<dyn GNNCompatible>,
    /// Graph attention network
    graph_attention: Option<GraphAttentionNetwork>,
    /// Cross-asset learning component
    cross_asset_gnn: Option<CrossAssetGNN>,
    /// Market regime detector
    regime_detector: Option<RegimeDetector>,
    /// Configuration
    config: GNNConfig,
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

impl GNNEnhancedModel {
    /// Create new GNN-enhanced model
    pub fn new(base_model: Box<dyn GNNCompatible>, config: GNNConfig) -> Result<Self> {
        let graph_attention = if config.cross_asset_enabled || config.regime_detection_enabled {
            Some(GraphAttentionNetwork::new(config.graph_attention.clone())?)
        } else {
            None
        };

        let cross_asset_gnn = if config.cross_asset_enabled {
            Some(CrossAssetGNN::new(config.cross_asset.clone())?)
        } else {
            None
        };

        let regime_detector = if config.regime_detection_enabled {
            Some(RegimeDetector::new(config.regime_detection.clone())?)
        } else {
            None
        };

        log::info!(
            "Created GNN-enhanced model with cross_asset={}, regime_detection={}",
            config.cross_asset_enabled,
            config.regime_detection_enabled
        );

        Ok(Self {
            base_model,
            graph_attention,
            cross_asset_gnn,
            regime_detector,
            config,
        })
    }

    /// Forward pass with GNN enhancements
    pub fn forward(&self, input: &Tensor, market_data: Option<&MarketGraph>) -> Result<Tensor> {
        // Get base embeddings
        let embeddings = self.base_model.get_embeddings(input)?;

        // Apply graph attention if enabled
        let enhanced_embeddings = if let Some(ref graph_attention) = self.graph_attention {
            graph_attention.forward(&embeddings)?
        } else {
            embeddings
        };

        // Apply cross-asset learning if enabled and market data available
        let cross_asset_features = if let (Some(ref cross_asset_gnn), Some(market_data)) =
            (&self.cross_asset_gnn, market_data)
        {
            Some(cross_asset_gnn.forward(&enhanced_embeddings, market_data)?)
        } else {
            None
        };

        // Detect market regime if enabled
        let regime_features = if let Some(ref regime_detector) = self.regime_detector {
            Some(regime_detector.detect_regime(&enhanced_embeddings)?)
        } else {
            None
        };

        // Combine all features
        let final_features = self.combine_features(
            &enhanced_embeddings,
            cross_asset_features.as_ref(),
            regime_features.as_ref(),
        )?;

        // Apply to base model and get predictions
        let mut base_model = self.base_model.as_ref();
        // Note: This would need proper mutable access in real implementation
        self.base_model.forward(&final_features)
    }

    /// Combine different feature types
    fn combine_features(
        &self,
        base_features: &Tensor,
        cross_asset_features: Option<&Tensor>,
        regime_features: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut combined = base_features.clone();

        if let Some(cross_asset) = cross_asset_features {
            // Concatenate cross-asset features
            combined = Tensor::cat(&[&combined, cross_asset], 1)?;
        }

        if let Some(regime) = regime_features {
            // Concatenate regime features
            combined = Tensor::cat(&[&combined, regime], 1)?;
        }

        Ok(combined)
    }
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
