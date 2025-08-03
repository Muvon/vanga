// Multi-Head Self-Attention implementation for VANGA LSTM with auto-optimization
use crate::config::model::AttentionConfig;
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};
use candle_nn::{linear, ops, Linear, Module, VarBuilder};

// AttentionConfig now comes from src/config/model.rs - no local default needed

/// Multi-Head Self-Attention layer optimized for cryptocurrency time series
pub struct MultiHeadAttention {
    config: AttentionConfig,
    head_dim: usize,
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
        // Auto-optimize head dimension based on input size
        let head_dim = config
            .head_dim
            .map(|h| h as usize)
            .unwrap_or_else(|| Self::optimize_head_dimension(input_dim, config.heads as usize));
        let total_dim = config.heads as usize * head_dim;

        // Create projection layers
        let query_projection = linear(input_dim, total_dim, vs.pp("query_proj")).map_err(|e| {
            VangaError::ModelError(format!("Query projection creation failed: {}", e))
        })?;

        let key_projection = linear(input_dim, total_dim, vs.pp("key_proj")).map_err(|e| {
            VangaError::ModelError(format!("Key projection creation failed: {}", e))
        })?;

        let value_projection = linear(input_dim, total_dim, vs.pp("value_proj")).map_err(|e| {
            VangaError::ModelError(format!("Value projection creation failed: {}", e))
        })?;

        let output_projection =
            linear(total_dim, input_dim, vs.pp("output_proj")).map_err(|e| {
                VangaError::ModelError(format!("Output projection creation failed: {}", e))
            })?;

        // Create relative position embeddings for time series
        let relative_position_embeddings = if config.use_relative_position {
            // Use a reasonable default max sequence length for crypto time series
            let max_seq_len = 200; // Standard crypto sequence length
            Some(Self::create_relative_position_embeddings(
                max_seq_len,
                head_dim,
                &device,
            )?)
        } else {
            None
        };

        log::info!(
            "✅ MultiHeadAttention initialized: {} heads, {} head_dim, input_dim={}, total_dim={}",
            config.heads,
            head_dim,
            input_dim,
            config.heads as usize * head_dim
        );

        Ok(Self {
            config,
            head_dim,
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
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch_size, sequence_length, input_dim]
    /// * `training` - Whether model is in training mode (affects dropout)
    ///
    /// # Returns
    /// * `(output, attention_scores)` - Attended output and attention weights
    pub fn forward(&self, input: &Tensor, training: bool) -> Result<(Tensor, Tensor)> {
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
        let attention_scores = self.compute_attention_scores(&queries, &keys, seq_len, training)?;

        // Apply attention to values
        let attended_values = attention_scores.matmul(&values)?.contiguous()?;

        // Reshape back and apply output projection
        let attended_output = self.reshape_from_attention(&attended_values, batch_size, seq_len)?;
        let mut output = self.output_projection.forward(&attended_output)?;

        // Apply consistent dropout to output (controlled by config AND training mode)
        if self.config.dropout_output && self.config.dropout_rate > 0.0 && training {
            output = ops::dropout(&output, self.config.dropout_rate as f32)?;
            log::debug!(
                "🔧 Applied MultiHead attention output dropout (rate: {:.3}) to tensor shape {:?} [CONSISTENT]",
                self.config.dropout_rate,
                output.shape()
            );
        }

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
                self.config.heads as usize,
                self.head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()
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
            .contiguous()?
            .reshape((
                batch_size,
                seq_len,
                self.config.heads as usize * self.head_dim,
            ))
            .map_err(|e| VangaError::ModelError(format!("Attention reshape back failed: {}", e)))
    }

    /// Compute attention scores with crypto-specific optimizations
    fn compute_attention_scores(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        seq_len: usize,
        training: bool,
    ) -> Result<Tensor> {
        // Compute scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt() as f32;
        let scale_tensor = Tensor::new(scale, &self.device)?;
        let scaled_queries = queries.broadcast_div(&scale_tensor)?.contiguous()?;

        // Compute attention scores: Q * K^T
        let keys_transposed = keys.transpose(2, 3)?.contiguous()?;
        let mut attention_scores = scaled_queries.matmul(&keys_transposed)?.contiguous()?;

        // Add relative position embeddings for better temporal modeling
        if let Some(ref pos_embeddings) = self.relative_position_embeddings {
            attention_scores =
                self.add_relative_position_bias(&attention_scores, pos_embeddings, seq_len)?;
        }

        // Apply temperature scaling for crypto volatility adaptation
        if self.config.temperature_scaling != 1.0 {
            let temperature = Tensor::new(self.config.temperature_scaling as f32, &self.device)?;
            attention_scores = attention_scores.broadcast_div(&temperature)?.contiguous()?;
        }

        // Apply causal mask for time series (prevent looking into future)
        attention_scores = self.apply_causal_mask(&attention_scores, seq_len)?;

        // Apply softmax to get attention weights
        let mut attention_weights = ops::softmax(&attention_scores, 3)?.contiguous()?;

        // Apply consistent dropout to attention weights (controlled by config AND training mode)
        if self.config.dropout_weights && self.config.dropout_rate > 0.0 && training {
            attention_weights = ops::dropout(&attention_weights, self.config.dropout_rate as f32)?;
            log::debug!(
                "🔧 Applied MultiHead attention weights dropout (rate: {:.3}) to tensor shape {:?} [CONSISTENT]",
                self.config.dropout_rate,
                attention_weights.shape()
            );
        } else {
            log::debug!(
                "🔧 No MultiHead attention weights dropout configured (rate: {:.3})",
                self.config.dropout_rate
            );
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

        let relevant_embeddings = pos_embeddings.narrow(0, start_idx, seq_len)?.contiguous()?;

        // Add position bias to attention scores
        // This is a simplified implementation - in practice, you'd want more sophisticated position encoding
        let relative_bias = relevant_embeddings
            .matmul(&relevant_embeddings.transpose(0, 1)?.contiguous()?)?
            .contiguous()?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        attention_scores
            .broadcast_add(&relative_bias)?
            .contiguous()
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
            .broadcast_add(&mask)?
            .contiguous()
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
        config: AttentionConfig, // Accept actual config parameter
        vs: VarBuilder,
        device: Device,
    ) -> Result<Box<dyn AttentionModule>> {
        match attention_type {
            crate::config::model::AttentionMechanism::MultiHeadAttention => {
                let attention = MultiHeadAttention::new(input_dim, config, vs, device)?;
                Ok(Box::new(attention))
            }
            crate::config::model::AttentionMechanism::SelfAttention => {
                // Simplified self-attention (single head) - override heads but keep other config
                let self_attention_config = AttentionConfig { heads: 1, ..config };
                let attention =
                    MultiHeadAttention::new(input_dim, self_attention_config, vs, device)?;
                Ok(Box::new(attention))
            }
            crate::config::model::AttentionMechanism::AdditiveAttention => {
                // Create proper additive attention mechanism - override specific fields but keep other config
                let additive_config = AttentionConfig {
                    heads: 1,                               // Additive attention typically uses single head
                    head_dim: Some((input_dim / 2) as u32), // Hidden dimension for additive scoring
                    temperature_scaling: 1.0,
                    use_relative_position: true,
                    ..config
                };
                let attention = AdditiveAttention::new(input_dim, additive_config, vs, device)?;
                Ok(Box::new(attention))
            }
            crate::config::model::AttentionMechanism::VariableSelection => {
                // TFT Variable Selection Attention - builds on MultiHeadAttention
                let base_attention =
                    MultiHeadAttention::new(input_dim, config, vs.pp("base"), device.clone())?;

                // For now, return the base attention - full TFT integration would be here
                log::info!("TFT Variable Selection attention requested - using enhanced MultiHeadAttention");
                Ok(Box::new(base_attention))
            }
            crate::config::model::AttentionMechanism::MixtureOfHeads => {
                // Mixture-of-Head Attention with dynamic routing
                // Note: MoH requires direct instantiation due to mutable routing state
                // This factory method is not suitable for MoH - use EnhancedAttentionFactory instead
                log::warn!("MixtureOfHeads attention requires EnhancedAttentionFactory for proper instantiation");
                Err(VangaError::ModelError(
                    "MixtureOfHeads attention must be created through EnhancedAttentionFactory"
                        .to_string(),
                ))
            }
            crate::config::model::AttentionMechanism::None => Err(VangaError::ModelError(
                "Cannot create attention mechanism for 'None' type".to_string(),
            )),
        }
    }
}

/// Additive Attention mechanism (Bahdanau-style) optimized for cryptocurrency sequences
pub struct AdditiveAttention {
    #[allow(dead_code)]
    input_dim: usize,
    hidden_dim: usize,
    config: AttentionConfig,

    // Learnable parameters for additive attention: score = v^T * tanh(W1*h + W2*s)
    w_query: Linear,     // W1 projection
    w_key: Linear,       // W2 projection
    v_attention: Linear, // v scoring vector

    // Crypto-specific optimizations
    position_bias: Option<Tensor>,
    device: Device,
}

impl AdditiveAttention {
    /// Create new additive attention layer with crypto-specific optimizations
    pub fn new(
        input_dim: usize,
        config: AttentionConfig,
        vs: VarBuilder,
        device: Device,
    ) -> Result<Self> {
        let hidden_dim = config.head_dim.unwrap_or((input_dim / 2) as u32) as usize;

        // Initialize learnable parameters
        let w_query = linear(input_dim, hidden_dim, vs.pp("w_query"))?;
        let w_key = linear(input_dim, hidden_dim, vs.pp("w_key"))?;
        let v_attention = linear(hidden_dim, 1, vs.pp("v_attention"))?;

        // Create crypto-specific position bias for recency weighting
        let position_bias = if config.use_relative_position {
            Some(Self::create_crypto_position_bias(
                200, // Standard crypto sequence length
                &device,
            )?)
        } else {
            None
        };

        log::info!(
            "Created AdditiveAttention: input_dim={}, hidden_dim={}, position_bias={}",
            input_dim,
            hidden_dim,
            position_bias.is_some()
        );

        Ok(Self {
            input_dim,
            hidden_dim,
            config,
            w_query,
            w_key,
            v_attention,
            position_bias,
            device,
        })
    }

    /// Forward pass with additive attention mechanism
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch_size, sequence_length, input_dim]
    /// * `training` - Whether model is in training mode (affects dropout)
    ///
    /// # Returns
    /// * `(output, attention_weights)` - Attended output and attention weights
    pub fn forward(&self, input: &Tensor, training: bool) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, _) = input.dims3()?;

        // Compute additive attention scores: score = v^T * tanh(W1*h + W2*s)
        let scores = self.compute_additive_scores(input, training)?;

        // Apply crypto-specific optimizations
        let optimized_scores = self.apply_crypto_optimizations(&scores, seq_len)?;

        // Apply softmax to get attention weights
        let mut attention_weights = ops::softmax_last_dim(&optimized_scores)?;

        // Apply consistent dropout to attention weights (controlled by config AND training mode)
        if self.config.dropout_weights && self.config.dropout_rate > 0.0 && training {
            attention_weights = ops::dropout(&attention_weights, self.config.dropout_rate as f32)?;
            log::debug!(
                "🔧 Applied Additive attention weights dropout (rate: {:.3}) to tensor shape {:?} [CONSISTENT]",
                self.config.dropout_rate,
                attention_weights.shape()
            );
        }

        // Apply attention weights to input values
        let output = self.apply_attention_weights(input, &attention_weights)?;

        log::debug!(
            "AdditiveAttention forward: batch_size={}, seq_len={}, output_shape={:?}",
            batch_size,
            seq_len,
            output.shape()
        );

        Ok((output, attention_weights))
    }

    /// Compute additive attention scores using the formula: score = v^T * tanh(W1*h + W2*s)
    fn compute_additive_scores(&self, input: &Tensor, _training: bool) -> Result<Tensor> {
        let (batch_size, seq_len, _) = input.dims3()?;

        // Project input through query and key transformations
        let mut query_projection = self.w_query.forward(input)?; // [batch, seq, hidden]
        let mut key_projection = self.w_key.forward(input)?; // [batch, seq, hidden]

        // Apply consistent dropout to projections (controlled by config)
        if self.config.dropout_projections && self.config.dropout_rate > 0.0 {
            query_projection = ops::dropout(&query_projection, self.config.dropout_rate as f32)?;
            key_projection = ops::dropout(&key_projection, self.config.dropout_rate as f32)?;
            log::debug!(
                "🔧 Applied Additive attention projections dropout (rate: {:.3}) [CONSISTENT]",
                self.config.dropout_rate
            );
        }

        // For additive attention, we compute attention between each position and all others
        // Expand dimensions for pairwise computation
        let query_expanded = query_projection.unsqueeze(2)?; // [batch, seq, 1, hidden]
        let key_expanded = key_projection.unsqueeze(1)?; // [batch, 1, seq, hidden]

        // Broadcast addition: each query position with each key position
        let query_broadcast =
            query_expanded.broadcast_as(&[batch_size, seq_len, seq_len, self.hidden_dim])?;
        let key_broadcast =
            key_expanded.broadcast_as(&[batch_size, seq_len, seq_len, self.hidden_dim])?;

        // Add projections and apply tanh activation
        let combined = (query_broadcast + key_broadcast)?.tanh()?;

        // Apply final scoring transformation with v vector
        let mut scores = self.v_attention.forward(&combined)?; // [batch, seq, seq, 1]

        // Apply consistent dropout to final scores (controlled by config)
        if self.config.dropout_scores && self.config.dropout_rate > 0.0 {
            scores = ops::dropout(&scores, self.config.dropout_rate as f32)?;
            log::debug!(
                "🔧 Applied Additive attention final scores dropout (rate: {:.3}) [CONSISTENT]",
                self.config.dropout_rate
            );
        }

        let scores = scores.squeeze(candle_core::D::Minus1)?; // [batch, seq, seq]

        Ok(scores)
    }

    /// Apply cryptocurrency-specific optimizations to attention scores
    fn apply_crypto_optimizations(&self, scores: &Tensor, seq_len: usize) -> Result<Tensor> {
        let mut optimized_scores = scores.clone();

        // Apply position bias for cryptocurrency recency preference
        if let Some(ref position_bias) = self.position_bias {
            let bias = position_bias.narrow(0, 0, seq_len)?;
            let bias_expanded = bias.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq]
            optimized_scores = (optimized_scores + bias_expanded)?;
        }

        // Apply temperature scaling for crypto volatility adaptation
        if self.config.temperature_scaling != 1.0 {
            optimized_scores = (optimized_scores / self.config.temperature_scaling)?;
        }

        // Apply causal mask to prevent attention to future positions
        optimized_scores = self.apply_causal_mask(&optimized_scores)?;

        Ok(optimized_scores)
    }

    /// Apply causal mask to prevent attention to future positions
    fn apply_causal_mask(&self, scores: &Tensor) -> Result<Tensor> {
        let seq_len = scores.dim(candle_core::D::Minus1)?;

        // Create lower triangular mask
        let mut mask_data = vec![-1e9; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 0.0;
            }
        }

        let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), &self.device)?;
        let mask_expanded = mask.unsqueeze(0)?; // [1, seq, seq]

        let masked_scores = (scores + mask_expanded)?;

        Ok(masked_scores)
    }

    /// Apply attention weights to input values
    fn apply_attention_weights(&self, input: &Tensor, weights: &Tensor) -> Result<Tensor> {
        // weights: [batch, seq, seq], input: [batch, seq, input_dim]
        let output = weights.matmul(input)?; // [batch, seq, input_dim]

        Ok(output)
    }

    /// Create cryptocurrency-specific position bias for recency weighting
    fn create_crypto_position_bias(max_length: usize, device: &Device) -> Result<Tensor> {
        // Create exponential decay bias favoring recent positions
        let positions: Vec<f64> = (0..max_length)
            .map(|i| {
                let position_ratio = i as f64 / max_length as f64;
                // Exponential decay: more recent positions get higher bias
                let decay_factor = 0.1; // Adjust for stronger/weaker recency bias
                (position_ratio * decay_factor).exp() - 1.0
            })
            .collect();

        Tensor::from_slice(&positions, (max_length,), device)
            .map_err(|e| VangaError::ModelError(format!("Position bias creation failed: {}", e)))
    }

    /// Get attention configuration
    pub fn get_config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Update temperature scaling for dynamic adaptation to market volatility
    pub fn update_temperature(&mut self, new_temperature: f64) {
        self.config.temperature_scaling = new_temperature;
        log::debug!(
            "Updated AdditiveAttention temperature scaling to {:.2}",
            new_temperature
        );
    }
}

impl AttentionModule for AdditiveAttention {
    fn forward(&self, input: &Tensor, training: bool) -> Result<(Tensor, Tensor)> {
        self.forward(input, training)
    }

    fn get_config(&self) -> AttentionConfig {
        self.get_config().clone()
    }
}

/// Trait for different attention mechanisms
pub trait AttentionModule {
    fn forward(&self, input: &Tensor, training: bool) -> Result<(Tensor, Tensor)>;
    fn get_config(&self) -> AttentionConfig; // Return owned config instead of reference
}

impl AttentionModule for MultiHeadAttention {
    fn forward(&self, input: &Tensor, training: bool) -> Result<(Tensor, Tensor)> {
        self.forward(input, training)
    }

    fn get_config(&self) -> AttentionConfig {
        self.config.clone()
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
        assert_eq!(config.heads, 8);
        assert_eq!(config.head_dim, Some(64));
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
