// Graph Attention Network for cross-asset learning and market structure modeling
use crate::utils::error::{Result, VangaError};
use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Graph Attention Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Hidden dimension for graph features
    pub hidden_dim: usize,
    /// Dropout rate for attention weights
    pub dropout_rate: f64,
    /// Enable edge features (correlation, volatility spillover)
    pub use_edge_features: bool,
    /// Enable multi-hop attention (2nd order connections)
    pub multi_hop_attention: bool,
    /// Temperature scaling for attention softmax
    pub temperature: f64,
    /// Enable learnable positional encoding
    pub learnable_position_encoding: bool,
}

impl Default for GraphAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            hidden_dim: 128,
            dropout_rate: 0.1,
            use_edge_features: true,
            multi_hop_attention: false,
            temperature: 1.0,
            learnable_position_encoding: true,
        }
    }
}

impl GraphAttentionConfig {
    /// Crypto-optimized configuration
    pub fn crypto_optimized() -> Self {
        Self {
            num_heads: 12, // More heads for complex crypto relationships
            hidden_dim: 256,
            dropout_rate: 0.15, // Higher dropout for noise resistance
            use_edge_features: true,
            multi_hop_attention: true, // Capture indirect correlations
            temperature: 0.8,          // Sharper attention for volatile markets
            learnable_position_encoding: true,
        }
    }

    /// Single-asset configuration (regime detection only)
    pub fn single_asset() -> Self {
        Self {
            num_heads: 4,
            hidden_dim: 64,
            dropout_rate: 0.1,
            use_edge_features: false,
            multi_hop_attention: false,
            temperature: 1.0,
            learnable_position_encoding: false,
        }
    }

    /// Multi-asset portfolio configuration
    pub fn multi_asset() -> Self {
        Self {
            num_heads: 16, // Maximum heads for complex portfolio relationships
            hidden_dim: 512,
            dropout_rate: 0.2,
            use_edge_features: true,
            multi_hop_attention: true,
            temperature: 0.9,
            learnable_position_encoding: true,
        }
    }
}

/// Graph Attention Network for processing asset relationships
pub struct GraphAttentionNetwork {
    /// Multi-head attention layers
    attention_heads: Vec<GraphAttentionHead>,
    /// Output projection layer
    output_projection: Linear,
    /// Edge feature processor (if enabled)
    edge_processor: Option<Linear>,
    /// Positional encoding (if enabled)
    position_encoding: Option<Linear>,
    /// Configuration
    config: GraphAttentionConfig,
}

/// Single attention head for graph processing
pub struct GraphAttentionHead {
    /// Query projection
    query_projection: Linear,
    /// Key projection
    key_projection: Linear,
    /// Value projection
    value_projection: Linear,
    /// Edge feature projection (optional)
    edge_projection: Option<Linear>,
    /// Hidden dimension
    hidden_dim: usize,
}

impl GraphAttentionNetwork {
    /// Create new Graph Attention Network
    pub fn new(config: GraphAttentionConfig, vs: VarBuilder) -> Result<Self> {
        let head_dim = config.hidden_dim / config.num_heads;
        if config.hidden_dim % config.num_heads != 0 {
            return Err(VangaError::ConfigError(format!(
                "Hidden dim {} must be divisible by num_heads {}",
                config.hidden_dim, config.num_heads
            )));
        }

        // Create attention heads
        let mut attention_heads = Vec::new();
        for i in 0..config.num_heads {
            let head = GraphAttentionHead::new(
                config.hidden_dim,
                head_dim,
                config.use_edge_features,
                vs.pp(format!("head_{}", i)),
            )?;
            attention_heads.push(head);
        }

        // Output projection
        let output_projection = linear(
            config.hidden_dim,
            config.hidden_dim,
            vs.pp("output_projection"),
        )?;

        // Edge feature processor
        let edge_processor = if config.use_edge_features {
            Some(linear(
                4, // correlation, volatility_spillover, volume_ratio, price_ratio
                config.hidden_dim / 4,
                vs.pp("edge_processor"),
            )?)
        } else {
            None
        };

        // Positional encoding
        let position_encoding = if config.learnable_position_encoding {
            Some(linear(
                1, // Position index
                config.hidden_dim,
                vs.pp("position_encoding"),
            )?)
        } else {
            None
        };

        log::info!(
            "Created Graph Attention Network with {} heads, hidden_dim={}, edge_features={}",
            config.num_heads,
            config.hidden_dim,
            config.use_edge_features
        );

        Ok(Self {
            attention_heads,
            output_projection,
            edge_processor,
            position_encoding,
            config,
        })
    }

    /// Forward pass through graph attention
    pub fn forward(
        &self,
        node_features: &Tensor,
        adjacency_matrix: Option<&Tensor>,
        edge_features: Option<&Tensor>,
    ) -> Result<Tensor> {
        let batch_size = node_features.dim(0)?;
        let num_nodes = node_features.dim(1)?;
        let feature_dim = node_features.dim(2)?;

        // Add positional encoding if enabled
        let enhanced_features = if let Some(ref pos_encoder) = self.position_encoding {
            let positions = Tensor::arange(0f32, num_nodes as f32, node_features.device())?
                .unsqueeze(0)?
                .unsqueeze(2)?
                .expand((batch_size, num_nodes, 1))?;
            let pos_encoding = pos_encoder.forward(&positions)?;
            node_features.add(&pos_encoding)?
        } else {
            node_features.clone()
        };

        // Process through each attention head
        let mut head_outputs = Vec::new();
        for head in &self.attention_heads {
            let head_output = head.forward(
                &enhanced_features,
                adjacency_matrix,
                edge_features,
                self.config.temperature,
            )?;
            head_outputs.push(head_output);
        }

        // Concatenate head outputs
        let concatenated = Tensor::cat(&head_outputs, 2)?;

        // Apply output projection
        let output = self.output_projection.forward(&concatenated)?;

        // Apply multi-hop attention if enabled
        if self.config.multi_hop_attention {
            self.apply_multi_hop_attention(&output, adjacency_matrix)
        } else {
            Ok(output)
        }
    }

    /// Apply multi-hop attention for capturing indirect relationships
    fn apply_multi_hop_attention(
        &self,
        features: &Tensor,
        adjacency_matrix: Option<&Tensor>,
    ) -> Result<Tensor> {
        if let Some(adj_matrix) = adjacency_matrix {
            // Compute 2nd order adjacency (A^2) for indirect connections
            let adj_squared = adj_matrix.matmul(adj_matrix)?;

            // Weighted combination of direct and indirect attention
            let direct_weight = 0.7;
            let indirect_weight = 0.3;

            // Apply attention with 2nd order adjacency
            let indirect_features = self.apply_attention_with_adjacency(features, &adj_squared)?;

            // Combine direct and indirect features
            let direct_features = features.mul_scalar(direct_weight)?;
            let weighted_indirect = indirect_features.mul_scalar(indirect_weight)?;

            Ok(direct_features.add(&weighted_indirect)?)
        } else {
            Ok(features.clone())
        }
    }

    /// Apply attention mechanism with custom adjacency matrix
    fn apply_attention_with_adjacency(
        &self,
        features: &Tensor,
        adjacency: &Tensor,
    ) -> Result<Tensor> {
        // Simplified attention application with adjacency masking
        // In practice, this would involve more sophisticated graph convolution
        let attention_weights = adjacency.softmax(2)?;
        features.matmul(&attention_weights)
    }

    /// Extract attention weights for interpretability
    pub fn get_attention_weights(
        &self,
        node_features: &Tensor,
        adjacency_matrix: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        let mut attention_weights = Vec::new();

        for head in &self.attention_heads {
            let weights = head.get_attention_weights(
                node_features,
                adjacency_matrix,
                self.config.temperature,
            )?;
            attention_weights.push(weights);
        }

        Ok(attention_weights)
    }
}

impl GraphAttentionHead {
    /// Create new attention head
    pub fn new(
        input_dim: usize,
        head_dim: usize,
        use_edge_features: bool,
        vs: VarBuilder,
    ) -> Result<Self> {
        let query_projection = linear(input_dim, head_dim, vs.pp("query"))?;
        let key_projection = linear(input_dim, head_dim, vs.pp("key"))?;
        let value_projection = linear(input_dim, head_dim, vs.pp("value"))?;

        let edge_projection = if use_edge_features {
            Some(linear(4, head_dim, vs.pp("edge"))?) // 4 edge feature types
        } else {
            None
        };

        Ok(Self {
            query_projection,
            key_projection,
            value_projection,
            edge_projection,
            hidden_dim: head_dim,
        })
    }

    /// Forward pass for single attention head
    pub fn forward(
        &self,
        node_features: &Tensor,
        adjacency_matrix: Option<&Tensor>,
        edge_features: Option<&Tensor>,
        temperature: f64,
    ) -> Result<Tensor> {
        // Project to Q, K, V
        let queries = self.query_projection.forward(node_features)?;
        let keys = self.key_projection.forward(node_features)?;
        let values = self.value_projection.forward(node_features)?;

        // Compute attention scores
        let attention_scores = queries.matmul(&keys.transpose(1, 2)?)?;
        let scaled_scores = attention_scores.div_scalar(temperature)?;

        // Apply adjacency mask if provided
        let masked_scores = if let Some(adj_matrix) = adjacency_matrix {
            // Mask out non-connected nodes (set to large negative value)
            let mask = adj_matrix.eq(&Tensor::zeros_like(adj_matrix)?)?;
            let large_negative = Tensor::full(-1e9f32, mask.shape(), mask.device())?;
            scaled_scores.where_cond(&mask, &large_negative)?
        } else {
            scaled_scores
        };

        // Add edge features if available
        let enhanced_scores =
            if let (Some(edge_proj), Some(edge_feat)) = (&self.edge_projection, edge_features) {
                let edge_contribution = edge_proj.forward(edge_feat)?;
                masked_scores.add(&edge_contribution)?
            } else {
                masked_scores
            };

        // Apply softmax to get attention weights
        let attention_weights = enhanced_scores.softmax(2)?;

        // Apply attention to values
        let attended_values = attention_weights.matmul(&values)?;

        Ok(attended_values)
    }

    /// Get attention weights for interpretability
    pub fn get_attention_weights(
        &self,
        node_features: &Tensor,
        adjacency_matrix: Option<&Tensor>,
        temperature: f64,
    ) -> Result<Tensor> {
        let queries = self.query_projection.forward(node_features)?;
        let keys = self.key_projection.forward(node_features)?;

        let attention_scores = queries.matmul(&keys.transpose(1, 2)?)?;
        let scaled_scores = attention_scores.div_scalar(temperature)?;

        let masked_scores = if let Some(adj_matrix) = adjacency_matrix {
            let mask = adj_matrix.eq(&Tensor::zeros_like(adj_matrix)?)?;
            let large_negative = Tensor::full(-1e9f32, mask.shape(), mask.device())?;
            scaled_scores.where_cond(&mask, &large_negative)?
        } else {
            scaled_scores
        };

        masked_scores.softmax(2)
    }
}

/// Edge feature calculator for market relationships
pub struct EdgeFeatureCalculator;

impl EdgeFeatureCalculator {
    /// Calculate edge features between two assets
    pub fn calculate_edge_features(
        asset_a_data: &Tensor,
        asset_b_data: &Tensor,
        window_size: usize,
    ) -> Result<Tensor> {
        // Extract price and volume data
        let prices_a = asset_a_data.i((.., 0))?; // Assuming close price is first feature
        let prices_b = asset_b_data.i((.., 0))?;
        let volumes_a = asset_a_data.i((.., 1))?; // Assuming volume is second feature
        let volumes_b = asset_b_data.i((.., 1))?;

        // Calculate correlation
        let correlation = Self::calculate_correlation(&prices_a, &prices_b, window_size)?;

        // Calculate volatility spillover
        let volatility_spillover =
            Self::calculate_volatility_spillover(&prices_a, &prices_b, window_size)?;

        // Calculate volume ratio
        let volume_ratio = volumes_a.div(&volumes_b.add_scalar(1e-8)?)?;

        // Calculate price ratio
        let price_ratio = prices_a.div(&prices_b.add_scalar(1e-8)?)?;

        // Stack edge features
        let edge_features = Tensor::stack(
            &[correlation, volatility_spillover, volume_ratio, price_ratio],
            1,
        )?;

        Ok(edge_features)
    }

    /// Calculate rolling correlation between two price series
    fn calculate_correlation(
        prices_a: &Tensor,
        prices_b: &Tensor,
        window_size: usize,
    ) -> Result<Tensor> {
        let seq_len = prices_a.dim(0)?;
        let mut correlations = Vec::new();

        for i in window_size..seq_len {
            let window_a = prices_a.narrow(0, i - window_size, window_size)?;
            let window_b = prices_b.narrow(0, i - window_size, window_size)?;

            // Calculate correlation coefficient
            let mean_a = window_a.mean_all()?;
            let mean_b = window_b.mean_all()?;

            let centered_a = window_a.sub(&mean_a)?;
            let centered_b = window_b.sub(&mean_b)?;

            let numerator = centered_a.mul(&centered_b)?.sum_all()?;
            let denom_a = centered_a.pow_tensor_scalar(2.0)?.sum_all()?.sqrt()?;
            let denom_b = centered_b.pow_tensor_scalar(2.0)?.sum_all()?.sqrt()?;

            let correlation = numerator.div(&denom_a.mul(&denom_b)?.add_scalar(1e-8)?)?;
            correlations.push(correlation);
        }

        // Pad with zeros for initial window
        let mut padded_correlations = vec![Tensor::zeros((), prices_a.device())?; window_size];
        padded_correlations.extend(correlations);

        Tensor::stack(&padded_correlations, 0)
    }

    /// Calculate volatility spillover effect
    fn calculate_volatility_spillover(
        prices_a: &Tensor,
        prices_b: &Tensor,
        window_size: usize,
    ) -> Result<Tensor> {
        // Calculate returns
        let returns_a = Self::calculate_returns(prices_a)?;
        let returns_b = Self::calculate_returns(prices_b)?;

        // Calculate rolling volatility
        let vol_a = Self::calculate_rolling_volatility(&returns_a, window_size)?;
        let vol_b = Self::calculate_rolling_volatility(&returns_b, window_size)?;

        // Spillover as correlation of volatilities
        Self::calculate_correlation(&vol_a, &vol_b, window_size)
    }

    /// Calculate price returns
    fn calculate_returns(prices: &Tensor) -> Result<Tensor> {
        let shifted_prices = prices.narrow(0, 0, prices.dim(0)? - 1)?;
        let current_prices = prices.narrow(0, 1, prices.dim(0)? - 1)?;
        let returns = current_prices
            .div(&shifted_prices.add_scalar(1e-8)?)?
            .log()?;

        // Pad with zero for first return
        let zero_return = Tensor::zeros((1,), prices.device())?;
        Tensor::cat(&[zero_return, returns], 0)
    }

    /// Calculate rolling volatility
    fn calculate_rolling_volatility(returns: &Tensor, window_size: usize) -> Result<Tensor> {
        let seq_len = returns.dim(0)?;
        let mut volatilities = Vec::new();

        for i in window_size..seq_len {
            let window_returns = returns.narrow(0, i - window_size, window_size)?;
            let mean_return = window_returns.mean_all()?;
            let centered_returns = window_returns.sub(&mean_return)?;
            let variance = centered_returns.pow_tensor_scalar(2.0)?.mean_all()?;
            let volatility = variance.sqrt()?;
            volatilities.push(volatility);
        }

        // Pad with zeros for initial window
        let mut padded_volatilities = vec![Tensor::zeros((), returns.device())?; window_size];
        padded_volatilities.extend(volatilities);

        Tensor::stack(&padded_volatilities, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;

    #[test]
    fn test_graph_attention_config_defaults() {
        let config = GraphAttentionConfig::default();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.hidden_dim, 128);
        assert!(config.use_edge_features);
    }

    #[test]
    fn test_graph_attention_config_crypto_optimized() {
        let config = GraphAttentionConfig::crypto_optimized();
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.hidden_dim, 256);
        assert!(config.multi_hop_attention);
        assert_eq!(config.temperature, 0.8);
    }

    #[test]
    fn test_graph_attention_config_single_asset() {
        let config = GraphAttentionConfig::single_asset();
        assert_eq!(config.num_heads, 4);
        assert!(!config.use_edge_features);
        assert!(!config.multi_hop_attention);
    }

    #[test]
    fn test_graph_attention_config_multi_asset() {
        let config = GraphAttentionConfig::multi_asset();
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.hidden_dim, 512);
        assert!(config.use_edge_features);
        assert!(config.multi_hop_attention);
    }
}
