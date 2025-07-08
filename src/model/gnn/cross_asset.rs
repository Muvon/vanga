// Cross-Asset Graph Neural Network for portfolio-level learning
use crate::model::gnn::graph_attention::{GraphAttentionConfig, GraphAttentionNetwork};
use crate::utils::error::{Result, VangaError};
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
    /// Enable dynamic asset graph updates
    pub dynamic_graph_updates: bool,
    /// Update frequency for graph structure (in epochs)
    pub graph_update_frequency: usize,
    /// Cross-asset feature fusion method
    pub fusion_method: CrossAssetFusionMethod,
    /// Asset-specific feature weights
    pub asset_specific_weights: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossAssetFusionMethod {
    /// Simple concatenation of features
    Concatenation,
    /// Weighted average based on correlations
    WeightedAverage,
    /// Attention-based fusion
    AttentionFusion,
    /// Graph convolution-based fusion
    GraphConvolution,
}

impl Default for CrossAssetConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            assets: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            correlation_threshold: 0.3,
            max_connections: 5,
            dynamic_graph_updates: true,
            graph_update_frequency: 10,
            fusion_method: CrossAssetFusionMethod::AttentionFusion,
            asset_specific_weights: true,
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
            correlation_threshold: 0.25, // Lower threshold for crypto volatility
            max_connections: 8,
            dynamic_graph_updates: true,
            graph_update_frequency: 5, // More frequent updates for volatile markets
            fusion_method: CrossAssetFusionMethod::GraphConvolution,
            asset_specific_weights: true,
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
            correlation_threshold: 0.2, // Include more diverse assets
            max_connections: 12,
            dynamic_graph_updates: true,
            graph_update_frequency: 20, // Less frequent for portfolio stability
            fusion_method: CrossAssetFusionMethod::AttentionFusion,
            asset_specific_weights: true,
        }
    }

    /// Disabled configuration
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            assets: vec![],
            correlation_threshold: 0.0,
            max_connections: 0,
            dynamic_graph_updates: false,
            graph_update_frequency: 0,
            fusion_method: CrossAssetFusionMethod::Concatenation,
            asset_specific_weights: false,
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
    /// Exchange information
    pub exchange: String,
}

/// Market graph representing asset relationships
#[derive(Debug, Clone)]
pub struct MarketGraph {
    /// Asset nodes
    pub nodes: HashMap<String, AssetNode>,
    /// Adjacency matrix (correlation-based connections)
    pub adjacency_matrix: Tensor,
    /// Edge features (correlation, volatility spillover, etc.)
    pub edge_features: Tensor,
    /// Asset symbol to index mapping
    pub symbol_to_index: HashMap<String, usize>,
    /// Graph update timestamp
    pub last_updated: std::time::SystemTime,
}

impl MarketGraph {
    /// Create new market graph
    pub fn new(assets: Vec<AssetNode>) -> Result<Self> {
        let num_assets = assets.len();
        let mut symbol_to_index = HashMap::new();
        let mut nodes = HashMap::new();

        for (i, asset) in assets.into_iter().enumerate() {
            symbol_to_index.insert(asset.symbol.clone(), i);
            nodes.insert(asset.symbol.clone(), asset);
        }

        // Initialize empty adjacency matrix and edge features
        let adjacency_matrix = Tensor::zeros((num_assets, num_assets), &candle_core::Device::Cpu)?;
        let edge_features = Tensor::zeros((num_assets, num_assets, 4), &candle_core::Device::Cpu)?;

        Ok(Self {
            nodes,
            adjacency_matrix,
            edge_features,
            symbol_to_index,
            last_updated: std::time::SystemTime::now(),
        })
    }

    /// Update graph structure based on current correlations
    pub fn update_graph_structure(
        &mut self,
        correlation_matrix: &Tensor,
        threshold: f64,
    ) -> Result<()> {
        // Create adjacency matrix based on correlation threshold
        let abs_correlations = correlation_matrix.abs()?;
        let threshold_tensor = Tensor::full(
            threshold as f32,
            abs_correlations.shape(),
            abs_correlations.device(),
        )?;
        self.adjacency_matrix = abs_correlations
            .gt(&threshold_tensor)?
            .to_dtype(candle_core::DType::F32)?;

        self.last_updated = std::time::SystemTime::now();

        log::debug!(
            "Updated market graph structure with threshold {}",
            threshold
        );
        Ok(())
    }

    /// Get node features as tensor
    pub fn get_node_features(&self) -> Result<Tensor> {
        let mut feature_tensors = Vec::new();

        // Collect features in index order
        for i in 0..self.nodes.len() {
            let symbol = self
                .symbol_to_index
                .iter()
                .find(|(_, &idx)| idx == i)
                .map(|(sym, _)| sym)
                .ok_or_else(|| VangaError::DataError(format!("Missing asset at index {}", i)))?;

            let node = self.nodes.get(symbol).ok_or_else(|| {
                VangaError::DataError(format!("Missing node for symbol {}", symbol))
            })?;

            feature_tensors.push(node.features.clone());
        }

        Tensor::stack(&feature_tensors, 0)
    }

    /// Add new asset to the graph
    pub fn add_asset(&mut self, asset: AssetNode) -> Result<()> {
        let new_index = self.nodes.len();
        self.symbol_to_index.insert(asset.symbol.clone(), new_index);
        self.nodes.insert(asset.symbol.clone(), asset);

        // Expand adjacency matrix and edge features
        let new_size = self.nodes.len();
        let device = self.adjacency_matrix.device();

        // Create new larger matrices
        let new_adjacency = Tensor::zeros((new_size, new_size), device)?;
        let new_edge_features = Tensor::zeros((new_size, new_size, 4), device)?;

        // Copy old values
        let old_size = new_size - 1;
        if old_size > 0 {
            let old_adj_slice = self
                .adjacency_matrix
                .narrow(0, 0, old_size)?
                .narrow(1, 0, old_size)?;
            new_adjacency
                .narrow(0, 0, old_size)?
                .narrow(1, 0, old_size)?
                .copy_(&old_adj_slice)?;

            let old_edge_slice = self
                .edge_features
                .narrow(0, 0, old_size)?
                .narrow(1, 0, old_size)?;
            new_edge_features
                .narrow(0, 0, old_size)?
                .narrow(1, 0, old_size)?
                .copy_(&old_edge_slice)?;
        }

        self.adjacency_matrix = new_adjacency;
        self.edge_features = new_edge_features;

        Ok(())
    }
}

/// Cross-Asset Graph Neural Network
pub struct CrossAssetGNN {
    /// Graph attention network
    graph_attention: GraphAttentionNetwork,
    /// Asset-specific feature processors
    asset_processors: HashMap<String, Linear>,
    /// Cross-asset fusion layer
    fusion_layer: CrossAssetFusionLayer,
    /// Output projection
    output_projection: Linear,
    /// Configuration
    config: CrossAssetConfig,
    /// Current market graph
    market_graph: Option<MarketGraph>,
}

/// Cross-asset feature fusion layer
pub struct CrossAssetFusionLayer {
    /// Fusion method
    method: CrossAssetFusionMethod,
    /// Attention weights for fusion (if using attention)
    attention_weights: Option<Linear>,
    /// Graph convolution layers (if using graph convolution)
    graph_conv_layers: Option<Vec<Linear>>,
}

impl CrossAssetGNN {
    /// Create new Cross-Asset GNN
    pub fn new(config: CrossAssetConfig, vs: VarBuilder) -> Result<Self> {
        if !config.enabled {
            return Err(VangaError::ConfigError(
                "CrossAssetGNN created with disabled config".to_string(),
            ));
        }

        // Create graph attention network
        let graph_attention_config = GraphAttentionConfig {
            num_heads: 8,
            hidden_dim: 256,
            dropout_rate: 0.1,
            use_edge_features: true,
            multi_hop_attention: true,
            temperature: 1.0,
            learnable_position_encoding: true,
        };
        let graph_attention =
            GraphAttentionNetwork::new(graph_attention_config, vs.pp("graph_attention"))?;

        // Create asset-specific processors
        let mut asset_processors = HashMap::new();
        for asset in &config.assets {
            let processor = linear(256, 128, vs.pp(format!("asset_{}", asset)))?;
            asset_processors.insert(asset.clone(), processor);
        }

        // Create fusion layer
        let fusion_layer = CrossAssetFusionLayer::new(
            config.fusion_method.clone(),
            config.assets.len(),
            vs.pp("fusion"),
        )?;

        // Output projection
        let output_projection = linear(256, 128, vs.pp("output_projection"))?;

        log::info!(
            "Created Cross-Asset GNN with {} assets, fusion method: {:?}",
            config.assets.len(),
            config.fusion_method
        );

        Ok(Self {
            graph_attention,
            asset_processors,
            fusion_layer,
            output_projection,
            config,
            market_graph: None,
        })
    }

    /// Set market graph
    pub fn set_market_graph(&mut self, graph: MarketGraph) {
        self.market_graph = Some(graph);
    }

    /// Forward pass with cross-asset learning
    pub fn forward(&self, input_features: &Tensor, market_graph: &MarketGraph) -> Result<Tensor> {
        // Get node features from market graph
        let node_features = market_graph.get_node_features()?;

        // Apply graph attention
        let attended_features = self.graph_attention.forward(
            &node_features,
            Some(&market_graph.adjacency_matrix),
            Some(&market_graph.edge_features),
        )?;

        // Apply asset-specific processing
        let processed_features =
            self.apply_asset_specific_processing(&attended_features, market_graph)?;

        // Fuse cross-asset features
        let fused_features = self.fusion_layer.forward(&processed_features)?;

        // Apply output projection
        let output = self.output_projection.forward(&fused_features)?;

        Ok(output)
    }

    /// Apply asset-specific feature processing
    fn apply_asset_specific_processing(
        &self,
        features: &Tensor,
        market_graph: &MarketGraph,
    ) -> Result<Tensor> {
        if !self.config.asset_specific_weights {
            return Ok(features.clone());
        }

        let mut processed_features = Vec::new();

        for (i, (symbol, _)) in market_graph.symbol_to_index.iter().enumerate() {
            let asset_features = features.i(i)?;

            if let Some(processor) = self.asset_processors.get(symbol) {
                let processed = processor.forward(&asset_features)?;
                processed_features.push(processed);
            } else {
                // Use identity if no specific processor
                processed_features.push(asset_features);
            }
        }

        Tensor::stack(&processed_features, 0)
    }

    /// Update market graph structure
    pub fn update_market_graph(&mut self, correlation_matrix: &Tensor) -> Result<()> {
        if let Some(ref mut graph) = self.market_graph {
            graph.update_graph_structure(correlation_matrix, self.config.correlation_threshold)?;
        }
        Ok(())
    }

    /// Get cross-asset attention weights for interpretability
    pub fn get_cross_asset_attention(&self, market_graph: &MarketGraph) -> Result<Vec<Tensor>> {
        let node_features = market_graph.get_node_features()?;
        self.graph_attention
            .get_attention_weights(&node_features, Some(&market_graph.adjacency_matrix))
    }
}

impl CrossAssetFusionLayer {
    /// Create new fusion layer
    pub fn new(method: CrossAssetFusionMethod, num_assets: usize, vs: VarBuilder) -> Result<Self> {
        let attention_weights = match method {
            CrossAssetFusionMethod::AttentionFusion => {
                Some(linear(256, num_assets, vs.pp("attention_weights"))?)
            }
            _ => None,
        };

        let graph_conv_layers = match method {
            CrossAssetFusionMethod::GraphConvolution => {
                let mut layers = Vec::new();
                layers.push(linear(256, 128, vs.pp("graph_conv_1"))?);
                layers.push(linear(128, 256, vs.pp("graph_conv_2"))?);
                Some(layers)
            }
            _ => None,
        };

        Ok(Self {
            method,
            attention_weights,
            graph_conv_layers,
        })
    }

    /// Forward pass through fusion layer
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        match self.method {
            CrossAssetFusionMethod::Concatenation => {
                // Simple concatenation along feature dimension
                let flattened = features.flatten(0, 1)?;
                Ok(flattened)
            }
            CrossAssetFusionMethod::WeightedAverage => {
                // Weighted average across assets
                features.mean(0)
            }
            CrossAssetFusionMethod::AttentionFusion => {
                if let Some(ref attention_layer) = self.attention_weights {
                    // Compute attention weights
                    let attention_scores = attention_layer.forward(features)?;
                    let attention_weights = attention_scores.softmax(0)?;

                    // Apply attention weights
                    let weighted_features = features.mul(&attention_weights.unsqueeze(2)?)?;
                    weighted_features.sum(0)
                } else {
                    Err(VangaError::ModelError(
                        "Missing attention weights for AttentionFusion".to_string(),
                    ))
                }
            }
            CrossAssetFusionMethod::GraphConvolution => {
                if let Some(ref conv_layers) = self.graph_conv_layers {
                    let mut output = features.clone();
                    for layer in conv_layers {
                        output = layer.forward(&output)?;
                        output = output.relu()?; // Apply ReLU activation
                    }
                    output.mean(0) // Average across assets
                } else {
                    Err(VangaError::ModelError(
                        "Missing graph convolution layers".to_string(),
                    ))
                }
            }
        }
    }
}

/// Factory for creating market graphs
pub struct MarketGraphFactory;

impl MarketGraphFactory {
    /// Create crypto market graph with major assets
    pub fn create_crypto_market_graph() -> Result<MarketGraph> {
        let assets = vec![
            AssetNode {
                symbol: "BTCUSDT".to_string(),
                features: Tensor::randn(0f32, 1f32, (256,), &candle_core::Device::Cpu)?,
                market_cap_rank: 1,
                category: "Layer1".to_string(),
                exchange: "Binance".to_string(),
            },
            AssetNode {
                symbol: "ETHUSDT".to_string(),
                features: Tensor::randn(0f32, 1f32, (256,), &candle_core::Device::Cpu)?,
                market_cap_rank: 2,
                category: "Layer1".to_string(),
                exchange: "Binance".to_string(),
            },
            AssetNode {
                symbol: "ADAUSDT".to_string(),
                features: Tensor::randn(0f32, 1f32, (256,), &candle_core::Device::Cpu)?,
                market_cap_rank: 8,
                category: "Layer1".to_string(),
                exchange: "Binance".to_string(),
            },
            AssetNode {
                symbol: "DOTUSDT".to_string(),
                features: Tensor::randn(0f32, 1f32, (256,), &candle_core::Device::Cpu)?,
                market_cap_rank: 12,
                category: "Layer0".to_string(),
                exchange: "Binance".to_string(),
            },
            AssetNode {
                symbol: "LINKUSDT".to_string(),
                features: Tensor::randn(0f32, 1f32, (256,), &candle_core::Device::Cpu)?,
                market_cap_rank: 15,
                category: "Oracle".to_string(),
                exchange: "Binance".to_string(),
            },
        ];

        MarketGraph::new(assets)
    }

    /// Create DeFi-focused market graph
    pub fn create_defi_market_graph() -> Result<MarketGraph> {
        let assets = vec![
            AssetNode {
                symbol: "UNIUSDT".to_string(),
                features: Tensor::randn(0f32, 1f32, (256,), &candle_core::Device::Cpu)?,
                market_cap_rank: 20,
                category: "DeFi".to_string(),
                exchange: "Binance".to_string(),
            },
            AssetNode {
                symbol: "AAVEUSDT".to_string(),
                features: Tensor::randn(0f32, 1f32, (256,), &candle_core::Device::Cpu)?,
                market_cap_rank: 25,
                category: "DeFi".to_string(),
                exchange: "Binance".to_string(),
            },
            AssetNode {
                symbol: "COMPUSDT".to_string(),
                features: Tensor::randn(0f32, 1f32, (256,), &candle_core::Device::Cpu)?,
                market_cap_rank: 30,
                category: "DeFi".to_string(),
                exchange: "Binance".to_string(),
            },
        ];

        MarketGraph::new(assets)
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
        assert_eq!(config.graph_update_frequency, 5);
    }

    #[test]
    fn test_cross_asset_config_disabled() {
        let config = CrossAssetConfig::disabled();
        assert!(!config.enabled);
        assert_eq!(config.assets.len(), 0);
    }

    #[test]
    fn test_market_graph_creation() {
        let graph = MarketGraphFactory::create_crypto_market_graph().unwrap();
        assert_eq!(graph.nodes.len(), 5);
        assert!(graph.nodes.contains_key("BTCUSDT"));
        assert!(graph.nodes.contains_key("ETHUSDT"));
    }

    #[test]
    fn test_asset_node_creation() {
        let asset = AssetNode {
            symbol: "BTCUSDT".to_string(),
            features: Tensor::randn(0f32, 1f32, (256,), &candle_core::Device::Cpu).unwrap(),
            market_cap_rank: 1,
            category: "Layer1".to_string(),
            exchange: "Binance".to_string(),
        };

        assert_eq!(asset.symbol, "BTCUSDT");
        assert_eq!(asset.market_cap_rank, 1);
        assert_eq!(asset.category, "Layer1");
    }
}
