// Simplified Cross-Asset GNN - compilation ready
use crate::utils::error::Result;
use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cross-asset learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAssetConfig {
    /// Enable cross-asset feature sharing
    pub enabled: bool,
    /// List of assets to include in cross-asset learning
    pub assets: Vec<String>,
    /// Minimum correlation threshold for asset connections
    pub correlation_threshold: f64,
    /// Maximum number of connected assets per node
    pub max_connections: usize,
}

impl Default for CrossAssetConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            assets: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            correlation_threshold: 0.3,
            max_connections: 5,
        }
    }
}

impl CrossAssetConfig {
    /// Crypto-optimized configuration
    pub fn crypto_optimized() -> Self {
        Self {
            enabled: true,
            assets: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "ADAUSDT".to_string(),
                "DOTUSDT".to_string(),
                "LINKUSDT".to_string(),
            ],
            correlation_threshold: 0.25,
            max_connections: 8,
        }
    }

    /// Portfolio-optimized configuration
    pub fn portfolio_optimized() -> Self {
        Self {
            enabled: true,
            assets: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "ADAUSDT".to_string(),
                "DOTUSDT".to_string(),
                "LINKUSDT".to_string(),
                "UNIUSDT".to_string(),
                "AAVEUSDT".to_string(),
                "COMPUSDT".to_string(),
            ],
            correlation_threshold: 0.2,
            max_connections: 12,
        }
    }

    /// Disabled configuration
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            assets: vec![],
            correlation_threshold: 0.0,
            max_connections: 0,
        }
    }
}

/// Asset node in the market graph
#[derive(Debug, Clone)]
pub struct AssetNode {
    /// Asset symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Node features (price, volume, technical indicators)
    pub features: Tensor,
    /// Market cap rank
    pub market_cap_rank: usize,
    /// Asset category (e.g., "Layer1", "DeFi", "Exchange")
    pub category: String,
}

/// Market graph representing asset relationships
#[derive(Debug, Clone)]
pub struct MarketGraph {
    /// Asset nodes
    pub nodes: HashMap<String, AssetNode>,
    /// Asset symbol to index mapping
    pub symbol_to_index: HashMap<String, usize>,
}

impl MarketGraph {
    /// Create new market graph
    pub fn new(assets: Vec<AssetNode>) -> Result<Self> {
        let mut symbol_to_index = HashMap::new();
        let mut nodes = HashMap::new();

        for (i, asset) in assets.into_iter().enumerate() {
            symbol_to_index.insert(asset.symbol.clone(), i);
            nodes.insert(asset.symbol.clone(), asset);
        }

        Ok(Self {
            nodes,
            symbol_to_index,
        })
    }

    /// Get node features as tensor (simplified)
    pub fn get_node_features(&self) -> Result<Tensor> {
        // Simplified implementation - return first node's features
        if let Some(node) = self.nodes.values().next() {
            Ok(node.features.clone())
        } else {
            Err(crate::utils::error::VangaError::DataError(
                "No nodes in market graph".to_string(),
            ))
        }
    }
}

/// Simplified Cross-Asset GNN
#[allow(dead_code)]
pub struct CrossAssetGNN {
    /// Feature processor
    feature_processor: Linear,
    /// Output projection
    output_projection: Linear,
    /// Configuration
    config: CrossAssetConfig,
}

impl CrossAssetGNN {
    /// Create new Cross-Asset GNN
    pub fn new(config: CrossAssetConfig, vs: VarBuilder) -> Result<Self> {
        if !config.enabled {
            return Err(crate::utils::error::VangaError::ConfigError(
                "CrossAssetGNN created with disabled config".to_string(),
            ));
        }

        let feature_processor = linear(256, 128, vs.pp("feature_processor"))?;
        let output_projection = linear(128, 128, vs.pp("output_projection"))?;

        log::info!(
            "Created simplified Cross-Asset GNN with {} assets",
            config.assets.len()
        );

        Ok(Self {
            feature_processor,
            output_projection,
            config,
        })
    }

    /// Forward pass with cross-asset learning (simplified)
    pub fn forward(&self, input_features: &Tensor, _market_graph: &MarketGraph) -> Result<Tensor> {
        let processed = self.feature_processor.forward(input_features)?;
        let output = self.output_projection.forward(&processed)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_asset_config_defaults() {
        let config = CrossAssetConfig::default();
        assert!(config.enabled);
        assert_eq!(config.assets.len(), 2);
        assert_eq!(config.correlation_threshold, 0.3);
    }

    #[test]
    fn test_cross_asset_config_crypto_optimized() {
        let config = CrossAssetConfig::crypto_optimized();
        assert!(config.enabled);
        assert_eq!(config.assets.len(), 5);
        assert_eq!(config.correlation_threshold, 0.25);
    }

    #[test]
    fn test_cross_asset_config_disabled() {
        let config = CrossAssetConfig::disabled();
        assert!(!config.enabled);
        assert_eq!(config.assets.len(), 0);
    }
}
