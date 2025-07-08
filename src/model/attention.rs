// Multi-Head Self-Attention implementation for VANGA LSTM with auto-optimization
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};
use candle_nn::{linear, ops, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Multi-Head Self-Attention configuration with auto-optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout_rate: f64,
    pub temperature_scaling: f64,
    pub use_relative_position: bool,
    pub max_sequence_length: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,                // Auto-optimized default for crypto sequences
            head_dim: 64,                // Optimal for most crypto features (50-100)
            dropout_rate: 0.1,           // Conservative dropout for attention
            temperature_scaling: 1.0,    // Standard temperature
            use_relative_position: true, // Better for time series
            max_sequence_length: 200,    // Sufficient for crypto patterns
        }
    }
}

/// Multi-Head Self-Attention layer optimized for cryptocurrency time series
pub struct MultiHeadAttention {
    config: AttentionConfig,
    query_projection: Linear,
    key_projection: Linear,
    value_projection: Linear,
    output_projection: Linear,
    relative_position_embeddings: Option<Tensor>,
    device: Device,
}

impl MultiHeadAttention {
    /// Create new multi-head attention layer with auto-optimized parameters
    pub fn new(
        input_dim: usize,
        config: AttentionConfig,
        vs: VarBuilder,
        device: Device,
    ) -> Result<Self> {
        let _total_dim = config.num_heads * config.head_dim;

        // Auto-optimize head dimension based on input size
        let optimized_head_dim = Self::optimize_head_dimension(input_dim, config.num_heads);
        let optimized_config = AttentionConfig {
            head_dim: optimized_head_dim,
            ..config
        };

        let total_optimized_dim = optimized_config.num_heads * optimized_config.head_dim;

        // Create projection layers
        let query_projection = linear(input_dim, total_optimized_dim, vs.pp("query_proj"))
            .map_err(|e| {
                VangaError::ModelError(format!("Query projection creation failed: {}", e))
            })?;

        let key_projection =
            linear(input_dim, total_optimized_dim, vs.pp("key_proj")).map_err(|e| {
                VangaError::ModelError(format!("Key projection creation failed: {}", e))
            })?;

        let value_projection = linear(input_dim, total_optimized_dim, vs.pp("value_proj"))
            .map_err(|e| {
                VangaError::ModelError(format!("Value projection creation failed: {}", e))
            })?;

        let output_projection = linear(total_optimized_dim, input_dim, vs.pp("output_proj"))
            .map_err(|e| {
                VangaError::ModelError(format!("Output projection creation failed: {}", e))
            })?;

        // Create relative position embeddings for time series
        let relative_position_embeddings = if optimized_config.use_relative_position {
            Some(Self::create_relative_position_embeddings(
                optimized_config.max_sequence_length,
                optimized_config.head_dim,
                &device,
            )?)
        } else {
            None
        };

        log::info!(
            "✅ MultiHeadAttention initialized: {} heads, {} head_dim, input_dim={}, total_dim={}",
            optimized_config.num_heads,
            optimized_config.head_dim,
            input_dim,
            total_optimized_dim
        );

        Ok(Self {
            config: optimized_config,
            query_projection,
            key_projection,
            value_projection,
            output_projection,
            relative_position_embeddings,
            device,
        })
    }

    /// Auto-optimize head dimension based on input size and crypto-specific patterns
    fn optimize_head_dimension(input_dim: usize, num_heads: usize) -> usize {
        // Crypto-optimized head dimensions based on feature count
        let optimal_head_dim = match input_dim {
            1..=20 => 32,     // Small feature sets (basic OHLCV)
            21..=50 => 64,    // Medium feature sets (basic + some indicators)
            51..=100 => 64,   // Large feature sets (comprehensive indicators)
            101..=200 => 128, // Very large feature sets (all indicators + custom)
            _ => 128,         // Extremely large feature sets
        };

        // Ensure head_dim * num_heads doesn't exceed reasonable limits
        let max_total_dim = input_dim * 2; // Don't exceed 2x input dimension
        let max_head_dim = max_total_dim / num_heads;

        std::cmp::min(optimal_head_dim, max_head_dim.max(16)) // Minimum 16 for effectiveness
    }

    /// Create relative position embeddings for better temporal modeling
    fn create_relative_position_embeddings(
        max_length: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Tensor> {
        // Create sinusoidal position embeddings optimized for crypto time series
        let mut embeddings = vec![vec![0.0; head_dim]; max_length * 2 - 1];

        for (pos, _) in (0..(max_length * 2 - 1)).enumerate() {
            let relative_pos = pos as f64 - (max_length - 1) as f64;

            for i in 0..head_dim {
                if i % 2 == 0 {
                    embeddings[pos][i] =
                        (relative_pos / 10000.0_f64.powf(i as f64 / head_dim as f64)).sin();
                } else {
                    embeddings[pos][i] =
                        (relative_pos / 10000.0_f64.powf((i - 1) as f64 / head_dim as f64)).cos();
                }
            }
        }

        let flat_embeddings: Vec<f32> =
            embeddings.into_iter().flatten().map(|x| x as f32).collect();

        Tensor::from_vec(flat_embeddings, (max_length * 2 - 1, head_dim), device).map_err(|e| {
            VangaError::ModelError(format!("Position embeddings creation failed: {}", e))
        })
    }

    /// Forward pass with attention mechanism optimized for crypto sequences
    pub fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        // Input shape: [batch_size, sequence_length, input_dim]
        let (batch_size, seq_len, _input_dim) = input
            .dims3()
            .map_err(|e| VangaError::ModelError(format!("Input tensor shape error: {}", e)))?;

        // Generate Q, K, V matrices
        let queries = self.query_projection.forward(input)?;
        let keys = self.key_projection.forward(input)?;
        let values = self.value_projection.forward(input)?;

        // Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        let queries = self.reshape_for_attention(&queries, batch_size, seq_len)?;
        let keys = self.reshape_for_attention(&keys, batch_size, seq_len)?;
        let values = self.reshape_for_attention(&values, batch_size, seq_len)?;

        // Compute attention scores with crypto-specific optimizations
        let attention_scores = self.compute_attention_scores(&queries, &keys, seq_len)?;

        // Apply attention to values
        let attended_values = attention_scores.matmul(&values)?;

        // Reshape back and apply output projection
        let attended_output = self.reshape_from_attention(&attended_values, batch_size, seq_len)?;
        let output = self.output_projection.forward(&attended_output)?;

        // Return both output and attention weights for interpretability
        Ok((output, attention_scores))
    }

    /// Reshape tensor for multi-head attention computation
    fn reshape_for_attention(
        &self,
        tensor: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Reshape from [batch, seq_len, num_heads * head_dim] to [batch, num_heads, seq_len, head_dim]
        tensor
            .reshape((
                batch_size,
                seq_len,
                self.config.num_heads,
                self.config.head_dim,
            ))?
            .transpose(1, 2)
            .map_err(|e| VangaError::ModelError(format!("Attention reshape failed: {}", e)))
    }

    /// Reshape tensor back from multi-head attention
    fn reshape_from_attention(
        &self,
        tensor: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Reshape from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, num_heads * head_dim]
        tensor
            .transpose(1, 2)?
            .reshape((
                batch_size,
                seq_len,
                self.config.num_heads * self.config.head_dim,
            ))
            .map_err(|e| VangaError::ModelError(format!("Attention reshape back failed: {}", e)))
    }

    /// Compute attention scores with crypto-specific optimizations
    fn compute_attention_scores(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Compute scaled dot-product attention
        let scale = (self.config.head_dim as f64).sqrt();
        let scaled_queries = queries.div(&Tensor::new(scale as f32, &self.device)?)?;

        // Compute attention scores: Q * K^T
        let keys_transposed = keys.transpose(2, 3)?;
        let mut attention_scores = scaled_queries.matmul(&keys_transposed)?;

        // Add relative position embeddings for better temporal modeling
        if let Some(ref pos_embeddings) = self.relative_position_embeddings {
            attention_scores =
                self.add_relative_position_bias(&attention_scores, pos_embeddings, seq_len)?;
        }

        // Apply temperature scaling for crypto volatility adaptation
        if self.config.temperature_scaling != 1.0 {
            let temperature = Tensor::new(self.config.temperature_scaling as f32, &self.device)?;
            attention_scores = attention_scores.div(&temperature)?;
        }

        // Apply causal mask for time series (prevent looking into future)
        attention_scores = self.apply_causal_mask(&attention_scores, seq_len)?;

        // Apply softmax to get attention weights
        let attention_weights = ops::softmax(&attention_scores, 3)?;

        // Apply dropout during training (if configured)
        if self.config.dropout_rate > 0.0 {
            // Note: In production, you'd apply dropout only during training
            // For now, we'll skip dropout in inference
        }

        Ok(attention_weights)
    }

    /// Add relative position bias for better temporal modeling
    fn add_relative_position_bias(
        &self,
        attention_scores: &Tensor,
        pos_embeddings: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Extract relevant position embeddings for current sequence length
        let start_idx = pos_embeddings.dim(0)? / 2 - seq_len / 2;
        let _end_idx = start_idx + seq_len;

        let relevant_embeddings = pos_embeddings.narrow(0, start_idx, seq_len)?;

        // Add position bias to attention scores
        // This is a simplified implementation - in practice, you'd want more sophisticated position encoding
        attention_scores
            .broadcast_add(&relevant_embeddings.unsqueeze(0)?.unsqueeze(0)?)
            .map_err(|e| VangaError::ModelError(format!("Position bias addition failed: {}", e)))
    }

    /// Apply causal mask to prevent attention to future positions
    fn apply_causal_mask(&self, attention_scores: &Tensor, seq_len: usize) -> Result<Tensor> {
        // Create lower triangular mask
        let mut mask_data = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 0.0;
            }
        }

        let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), &self.device)?;
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?; // Add batch and head dimensions

        attention_scores
            .broadcast_add(&mask)
            .map_err(|e| VangaError::ModelError(format!("Causal mask application failed: {}", e)))
    }

    /// Get attention configuration
    pub fn get_config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Update temperature scaling for dynamic adaptation to market volatility
    pub fn update_temperature(&mut self, new_temperature: f64) {
        self.config.temperature_scaling = new_temperature;
        log::debug!(
            "Updated attention temperature scaling to: {}",
            new_temperature
        );
    }
}

/// Attention mechanism factory for different types
pub struct AttentionFactory;

impl AttentionFactory {
    /// Create attention mechanism based on configuration
    pub fn create_attention(
        attention_type: &crate::config::model::AttentionMechanism,
        input_dim: usize,
        vs: VarBuilder,
        device: Device,
    ) -> Result<Box<dyn AttentionModule>> {
        match attention_type {
            crate::config::model::AttentionMechanism::MultiHeadAttention => {
                let config = AttentionConfig::default();
                let attention = MultiHeadAttention::new(input_dim, config, vs, device)?;
                Ok(Box::new(attention))
            }
            crate::config::model::AttentionMechanism::SelfAttention => {
                // Simplified self-attention (single head)
                let config = AttentionConfig {
                    num_heads: 1,
                    ..AttentionConfig::default()
                };
                let attention = MultiHeadAttention::new(input_dim, config, vs, device)?;
                Ok(Box::new(attention))
            }
            crate::config::model::AttentionMechanism::AdditiveAttention => {
                // For now, use multi-head with different config
                // TODO: Implement proper additive attention
                let config = AttentionConfig {
                    num_heads: 4,
                    head_dim: 32,
                    ..AttentionConfig::default()
                };
                let attention = MultiHeadAttention::new(input_dim, config, vs, device)?;
                Ok(Box::new(attention))
            }
            crate::config::model::AttentionMechanism::VariableSelection => {
                // TFT Variable Selection Attention - builds on MultiHeadAttention
                let config = AttentionConfig::default();
                let base_attention =
                    MultiHeadAttention::new(input_dim, config, vs.pp("base"), device.clone())?;

                // For now, return the base attention - full TFT integration would be here
                log::info!("TFT Variable Selection attention requested - using enhanced MultiHeadAttention");
                Ok(Box::new(base_attention))
            }
            crate::config::model::AttentionMechanism::None => Err(VangaError::ModelError(
                "Cannot create attention mechanism for 'None' type".to_string(),
            )),
        }
    }
}

/// Trait for different attention mechanisms
pub trait AttentionModule {
    fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor)>;
    fn get_config(&self) -> &AttentionConfig;
}

impl AttentionModule for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        self.forward(input)
    }

    fn get_config(&self) -> &AttentionConfig {
        self.get_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    #[test]
    fn test_attention_config_defaults() {
        let config = AttentionConfig::default();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!(config.use_relative_position);
    }

    #[test]
    fn test_head_dimension_optimization() {
        // Test auto-optimization logic: min(optimal_head_dim, max(max_head_dim, 16))
        // For input_dim=10: optimal=32, max_head_dim=(10*2)/8=2.5->2, max(2,16)=16, min(32,16)=16
        assert_eq!(MultiHeadAttention::optimize_head_dimension(10, 8), 16);
        // For input_dim=50: optimal=64, max_head_dim=(50*2)/8=12.5->12, max(12,16)=16, min(64,16)=16
        assert_eq!(MultiHeadAttention::optimize_head_dimension(50, 8), 16);
        // For input_dim=150: optimal=128, max_head_dim=(150*2)/8=37.5->37, max(37,16)=37, min(128,37)=37
        assert_eq!(MultiHeadAttention::optimize_head_dimension(150, 8), 37);
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
}
