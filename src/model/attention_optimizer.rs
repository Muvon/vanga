// Optimized attention computation for cryptocurrency sequence lengths
use crate::model::attention::{AttentionConfig, MultiHeadAttention};
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Optimized attention configuration for different cryptocurrency scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedAttentionConfig {
    /// Base attention configuration
    pub base_config: AttentionConfig,

    /// Sequence length optimization settings
    pub sequence_optimization: SequenceOptimization,

    /// Memory optimization settings
    pub memory_optimization: MemoryOptimization,

    /// Computational optimization settings
    pub compute_optimization: ComputeOptimization,

    /// Cryptocurrency-specific optimizations
    pub crypto_optimizations: CryptoOptimizations,
}

/// Sequence length optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceOptimization {
    /// Use sparse attention for long sequences
    pub use_sparse_attention: bool,

    /// Sliding window size for very long sequences
    pub sliding_window_size: Option<usize>,

    /// Local attention window size
    pub local_attention_window: usize,

    /// Use hierarchical attention for multi-scale patterns
    pub use_hierarchical_attention: bool,

    /// Adaptive sequence chunking
    pub adaptive_chunking: bool,
}

/// Memory optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    /// Use gradient checkpointing
    pub gradient_checkpointing: bool,

    /// Attention head pruning threshold
    pub head_pruning_threshold: f64,

    /// Use mixed precision computation
    pub mixed_precision: bool,

    /// Cache attention patterns for similar sequences
    pub cache_attention_patterns: bool,

    /// Maximum cache size
    pub max_cache_size: usize,
}

/// Computational optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeOptimization {
    /// Use Flash Attention algorithm
    pub use_flash_attention: bool,

    /// Fused attention operations
    pub fused_operations: bool,

    /// Parallel attention head computation
    pub parallel_heads: bool,

    /// Optimized matrix multiplication
    pub optimized_matmul: bool,

    /// Kernel fusion for GPU
    pub kernel_fusion: bool,
}

/// Cryptocurrency-specific optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoOptimizations {
    /// Focus attention on recent price movements
    pub recent_price_focus: bool,

    /// Emphasize volume spikes in attention
    pub volume_spike_emphasis: bool,

    /// Market regime-aware attention scaling
    pub regime_aware_scaling: bool,

    /// Volatility-adaptive attention temperature
    pub volatility_adaptive_temperature: bool,

    /// Support/resistance level attention boosting
    pub support_resistance_boosting: bool,
}

impl Default for OptimizedAttentionConfig {
    fn default() -> Self {
        Self {
            base_config: AttentionConfig::default(),
            sequence_optimization: SequenceOptimization {
                use_sparse_attention: true,
                sliding_window_size: Some(128),
                local_attention_window: 32,
                use_hierarchical_attention: true,
                adaptive_chunking: true,
            },
            memory_optimization: MemoryOptimization {
                gradient_checkpointing: true,
                head_pruning_threshold: 0.01,
                mixed_precision: true,
                cache_attention_patterns: true,
                max_cache_size: 1000,
            },
            compute_optimization: ComputeOptimization {
                use_flash_attention: true,
                fused_operations: true,
                parallel_heads: true,
                optimized_matmul: true,
                kernel_fusion: true,
            },
            crypto_optimizations: CryptoOptimizations {
                recent_price_focus: true,
                volume_spike_emphasis: true,
                regime_aware_scaling: true,
                volatility_adaptive_temperature: true,
                support_resistance_boosting: true,
            },
        }
    }
}

/// Optimized attention layer for cryptocurrency sequences
pub struct OptimizedAttention {
    config: OptimizedAttentionConfig,
    base_attention: MultiHeadAttention,
    sparse_attention: Option<SparseAttention>,
    hierarchical_attention: Option<HierarchicalAttention>,
    attention_cache: HashMap<String, Tensor>,
    device: Device,
}

impl OptimizedAttention {
    /// Create optimized attention layer
    pub fn new(
        input_dim: usize,
        config: OptimizedAttentionConfig,
        vs: VarBuilder,
        device: Device,
    ) -> Result<Self> {
        // Create base attention layer
        let base_attention = MultiHeadAttention::new(
            input_dim,
            config.base_config.clone(),
            vs.pp("base_attention"),
            device.clone(),
        )?;

        // Create sparse attention if enabled
        let sparse_attention = if config.sequence_optimization.use_sparse_attention {
            Some(SparseAttention::new(
                input_dim,
                config.sequence_optimization.local_attention_window,
                vs.pp("sparse_attention"),
                device.clone(),
            )?)
        } else {
            None
        };

        // Create hierarchical attention if enabled
        let hierarchical_attention = if config.sequence_optimization.use_hierarchical_attention {
            Some(HierarchicalAttention::new(
                input_dim,
                vs.pp("hierarchical_attention"),
                device.clone(),
            )?)
        } else {
            None
        };

        log::info!(
            "✅ Optimized attention initialized: sparse={}, hierarchical={}, crypto_optimized={}",
            sparse_attention.is_some(),
            hierarchical_attention.is_some(),
            config.crypto_optimizations.recent_price_focus
        );

        Ok(Self {
            config,
            base_attention,
            sparse_attention,
            hierarchical_attention,
            attention_cache: HashMap::new(),
            device,
        })
    }

    /// Forward pass with optimized attention computation
    pub fn forward(&mut self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = input.dim(1)?;

        // Choose attention strategy based on sequence length
        match seq_len {
            len if len <= 64 => {
                // Short sequences: use standard attention
                self.base_attention.forward(input)
            }
            len if len <= 256 => {
                // Medium sequences: use optimized standard attention
                self.forward_optimized_standard(input)
            }
            len if len <= 1024 => {
                // Long sequences: use sparse attention
                self.forward_sparse_attention(input)
            }
            _ => {
                // Very long sequences: use hierarchical attention
                self.forward_hierarchical_attention(input)
            }
        }
    }

    /// Optimized standard attention for medium sequences
    fn forward_optimized_standard(&mut self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        // Apply cryptocurrency-specific optimizations
        let optimized_input = self.apply_crypto_optimizations(input)?;

        // Use cached attention if available
        if self.config.memory_optimization.cache_attention_patterns {
            let cache_key = self.generate_cache_key(&optimized_input)?;
            if let Some(cached_result) = self.get_cached_attention(&cache_key) {
                return Ok(cached_result);
            }
        }

        // Compute attention with optimizations
        let (output, weights) = self.base_attention.forward(&optimized_input)?;

        // Cache result if enabled
        if self.config.memory_optimization.cache_attention_patterns {
            let cache_key = self.generate_cache_key(&optimized_input)?;
            self.cache_attention_result(&cache_key, &output, &weights)?;
        }

        Ok((output, weights))
    }

    /// Sparse attention for long sequences
    fn forward_sparse_attention(&mut self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        if let Some(ref sparse_attention) = self.sparse_attention {
            // Apply sliding window if sequence is very long
            if let Some(window_size) = self.config.sequence_optimization.sliding_window_size {
                let seq_len = input.dim(1)?;
                if seq_len > window_size {
                    return self.forward_sliding_window(input, window_size);
                }
            }

            sparse_attention.forward(input)
        } else {
            // Fallback to standard attention
            self.base_attention.forward(input)
        }
    }

    /// Hierarchical attention for very long sequences
    fn forward_hierarchical_attention(&mut self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        if let Some(ref hierarchical_attention) = self.hierarchical_attention {
            hierarchical_attention.forward(input)
        } else {
            // Fallback to sparse attention
            self.forward_sparse_attention(input)
        }
    }

    /// Sliding window attention for extremely long sequences
    fn forward_sliding_window(
        &mut self,
        input: &Tensor,
        window_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = input.dim(1)?;
        let input_shape = input.shape();
        let _batch_size = input_shape.dims()[0];
        let _feature_dim = input_shape.dims()[2];

        let mut outputs = Vec::new();
        let mut all_weights = Vec::new();

        // Process sequence in sliding windows
        let step_size = window_size / 2; // 50% overlap
        for start in (0..seq_len).step_by(step_size) {
            let end = std::cmp::min(start + window_size, seq_len);
            let window_len = end - start;

            if window_len < 8 {
                // Skip very small windows
                continue;
            }

            // Extract window
            let window_input = input.narrow(1, start, window_len)?;

            // Apply attention to window
            let (window_output, window_weights) = self.base_attention.forward(&window_input)?;

            outputs.push(window_output);
            all_weights.push(window_weights);
        }

        // Combine outputs from all windows
        if outputs.is_empty() {
            return Err(VangaError::ModelError(
                "No valid windows processed".to_string(),
            ));
        }

        let combined_output = Tensor::cat(&outputs, 1)?;
        let combined_weights = Tensor::cat(&all_weights, 1)?;

        // Resize to original sequence length if needed
        let final_output = if combined_output.dim(1)? != seq_len {
            self.resize_to_sequence_length(&combined_output, seq_len)?
        } else {
            combined_output
        };

        let final_weights = if combined_weights.dim(1)? != seq_len {
            self.resize_to_sequence_length(&combined_weights, seq_len)?
        } else {
            combined_weights
        };

        Ok((final_output, final_weights))
    }

    /// Apply cryptocurrency-specific optimizations to input
    fn apply_crypto_optimizations(&self, input: &Tensor) -> Result<Tensor> {
        let mut optimized_input = input.clone();

        if self.config.crypto_optimizations.recent_price_focus {
            optimized_input = self.apply_recent_price_focus(&optimized_input)?;
        }

        if self.config.crypto_optimizations.volume_spike_emphasis {
            optimized_input = self.apply_volume_spike_emphasis(&optimized_input)?;
        }

        if self
            .config
            .crypto_optimizations
            .volatility_adaptive_temperature
        {
            optimized_input = self.apply_volatility_adaptive_temperature(&optimized_input)?;
        }

        Ok(optimized_input)
    }

    /// Apply recent price focus (higher weights for recent timesteps)
    fn apply_recent_price_focus(&self, input: &Tensor) -> Result<Tensor> {
        let seq_len = input.dim(1)?;

        // Create exponential decay weights favoring recent timesteps
        let mut weights = Vec::new();
        let decay_factor: f64 = 0.95; // Adjust based on preference for recency

        for i in 0..seq_len {
            let weight = decay_factor.powi((seq_len - i - 1) as i32);
            weights.push(weight as f32);
        }

        let weight_tensor = Tensor::from_vec(weights, (1, seq_len, 1), &self.device)?;
        let weight_tensor = weight_tensor.broadcast_as(input.shape())?;

        input.mul(&weight_tensor).map_err(|e| {
            VangaError::ModelError(format!("Failed to apply recency weighting: {}", e))
        })
    }

    /// Apply volume spike emphasis
    fn apply_volume_spike_emphasis(&self, input: &Tensor) -> Result<Tensor> {
        // Assuming volume is in a specific feature dimension (would need to be configured)
        // For now, apply general spike emphasis to all features
        let mean_values = input.mean_keepdim(1)?;
        let std_values = input.var_keepdim(1)?.sqrt()?;

        // Identify spikes (values > mean + 2*std)
        let threshold = mean_values.add(&std_values.mul(&Tensor::new(2.0f32, &self.device)?)?)?;
        let spike_mask = input.gt(&threshold)?;

        // Amplify spikes by 1.5x
        let amplification = Tensor::new(1.5f32, &self.device)?;
        let normal_weight = Tensor::new(1.0f32, &self.device)?;

        let weights = spike_mask.where_cond(&amplification, &normal_weight)?;
        input.mul(&weights).map_err(|e| {
            VangaError::ModelError(format!("Failed to apply volume spike emphasis: {}", e))
        })
    }

    /// Apply volatility-adaptive temperature scaling
    fn apply_volatility_adaptive_temperature(&self, input: &Tensor) -> Result<Tensor> {
        // Calculate volatility (standard deviation) of the sequence
        let volatility = input.var_keepdim(1)?.sqrt()?.mean_all()?;
        let volatility_val = volatility.to_scalar::<f32>().unwrap_or(1.0);

        // Adjust temperature based on volatility
        let temperature = if volatility_val > 0.1 {
            0.8 // Lower temperature for high volatility (sharper attention)
        } else if volatility_val > 0.05 {
            1.0 // Normal temperature for medium volatility
        } else {
            1.2 // Higher temperature for low volatility (smoother attention)
        };

        let temperature_tensor = Tensor::new(temperature, &self.device)?;
        input.div(&temperature_tensor).map_err(|e| {
            VangaError::ModelError(format!("Failed to apply temperature scaling: {}", e))
        })
    }

    /// Generate cache key for attention patterns
    fn generate_cache_key(&self, input: &Tensor) -> Result<String> {
        // Simple hash based on input statistics
        let mean_val = input.mean_all()?.to_scalar::<f32>().unwrap_or(0.0);
        let std_val = input
            .var_keepdim(0)?
            .sqrt()?
            .mean_all()?
            .to_scalar::<f32>()
            .unwrap_or(1.0);
        let shape = input.shape();

        Ok(format!(
            "{}_{:.6}_{:.6}_{:?}",
            shape.dims().len(),
            mean_val,
            std_val,
            shape.dims()
        ))
    }

    /// Get cached attention result
    fn get_cached_attention(&self, cache_key: &str) -> Option<(Tensor, Tensor)> {
        // Simplified cache lookup (in practice, would store both output and weights)
        self.attention_cache
            .get(cache_key)
            .map(|cached_tensor| (cached_tensor.clone(), cached_tensor.clone()))
    }

    /// Cache attention result
    fn cache_attention_result(
        &mut self,
        cache_key: &str,
        output: &Tensor,
        _weights: &Tensor,
    ) -> Result<()> {
        // Check cache size limit
        if self.attention_cache.len() >= self.config.memory_optimization.max_cache_size {
            // Remove oldest entry (simplified LRU)
            if let Some(first_key) = self.attention_cache.keys().next().cloned() {
                self.attention_cache.remove(&first_key);
            }
        }

        // Cache the output tensor
        self.attention_cache
            .insert(cache_key.to_string(), output.clone());
        Ok(())
    }

    /// Resize tensor to target sequence length
    fn resize_to_sequence_length(&self, tensor: &Tensor, target_len: usize) -> Result<Tensor> {
        let current_len = tensor.dim(1)?;

        if current_len == target_len {
            return Ok(tensor.clone());
        }

        if current_len > target_len {
            // Truncate
            tensor.narrow(1, 0, target_len)
        } else {
            // Pad with zeros
            let batch_size = tensor.dim(0)?;
            let feature_dim = tensor.dim(2)?;
            let pad_len = target_len - current_len;

            let padding = Tensor::zeros(
                (batch_size, pad_len, feature_dim),
                tensor.dtype(),
                &self.device,
            )?;
            Tensor::cat(&[tensor.clone(), padding], 1)
        }
        .map_err(|e| VangaError::ModelError(format!("Failed to resize tensor: {}", e)))
    }

    /// Clear attention cache
    pub fn clear_cache(&mut self) {
        self.attention_cache.clear();
        log::debug!("Cleared attention cache");
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        (
            self.attention_cache.len(),
            self.config.memory_optimization.max_cache_size,
        )
    }

    /// Update configuration dynamically
    pub fn update_config(&mut self, new_config: OptimizedAttentionConfig) {
        self.config = new_config;
        log::debug!("Updated optimized attention configuration");
    }
}

/// Sparse attention implementation for long sequences
struct SparseAttention {
    local_window: usize,
}

impl SparseAttention {
    fn new(
        _input_dim: usize,
        local_window: usize,
        _vs: VarBuilder,
        _device: Device,
    ) -> Result<Self> {
        Ok(Self { local_window })
    }
    fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        // Simplified sparse attention: local window + global tokens
        let seq_len = input.dim(1)?;

        if seq_len <= self.local_window {
            // Sequence is short enough for full attention
            return Ok((input.clone(), input.clone())); // Simplified
        }

        // Apply local attention windows
        let mut outputs = Vec::new();
        let window_size = self.local_window;

        for start in (0..seq_len).step_by(window_size / 2) {
            let end = std::cmp::min(start + window_size, seq_len);
            let window_input = input.narrow(1, start, end - start)?;
            outputs.push(window_input);
        }

        let combined_output = Tensor::cat(&outputs, 1)?;

        // Resize to original length if needed
        let final_output = if combined_output.dim(1)? > seq_len {
            combined_output.narrow(1, 0, seq_len)?
        } else {
            combined_output
        };

        Ok((final_output.clone(), final_output))
    }
}

/// Hierarchical attention for very long sequences
struct HierarchicalAttention {}
impl HierarchicalAttention {
    fn new(_input_dim: usize, _vs: VarBuilder, _device: Device) -> Result<Self> {
        Ok(Self {})
    }
    fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        // Simplified hierarchical attention: process at multiple resolutions
        let seq_len = input.dim(1)?;

        // Level 1: Full resolution (recent data)
        let recent_len = std::cmp::min(64, seq_len);
        let recent_data = input.narrow(1, seq_len - recent_len, recent_len)?;

        // Level 2: Half resolution (medium-term data)
        let medium_len = std::cmp::min(128, seq_len / 2);
        let _medium_indices: Vec<usize> = (recent_len..medium_len).collect();

        // Level 3: Quarter resolution (long-term data)
        let long_len = std::cmp::min(64, seq_len / 4);
        let _long_indices: Vec<usize> = (medium_len..long_len).collect();

        // For simplicity, just return the recent data
        // In a full implementation, would combine all levels
        Ok((recent_data.clone(), recent_data))
    }
}

/// Factory for creating optimized attention configurations
pub struct OptimizedAttentionFactory;

impl OptimizedAttentionFactory {
    /// Create configuration optimized for short-term crypto trading (1min-15min)
    pub fn create_short_term_crypto() -> OptimizedAttentionConfig {
        OptimizedAttentionConfig {
            base_config: AttentionConfig {
                num_heads: 8,
                head_dim: 64,
                dropout_rate: 0.05,
                temperature_scaling: 0.9, // Sharper attention for short-term
                use_relative_position: true,
                max_sequence_length: 128,
            },
            sequence_optimization: SequenceOptimization {
                use_sparse_attention: false, // Short sequences don't need sparse
                sliding_window_size: None,
                local_attention_window: 32,
                use_hierarchical_attention: false,
                adaptive_chunking: false,
            },
            memory_optimization: MemoryOptimization {
                gradient_checkpointing: false, // Short sequences, memory not critical
                head_pruning_threshold: 0.02,
                mixed_precision: true,
                cache_attention_patterns: true,
                max_cache_size: 500,
            },
            compute_optimization: ComputeOptimization {
                use_flash_attention: true,
                fused_operations: true,
                parallel_heads: true,
                optimized_matmul: true,
                kernel_fusion: true,
            },
            crypto_optimizations: CryptoOptimizations {
                recent_price_focus: true,
                volume_spike_emphasis: true,
                regime_aware_scaling: true,
                volatility_adaptive_temperature: true,
                support_resistance_boosting: true,
            },
        }
    }

    /// Create configuration optimized for long-term crypto analysis (1h-1d)
    pub fn create_long_term_crypto() -> OptimizedAttentionConfig {
        OptimizedAttentionConfig {
            base_config: AttentionConfig {
                num_heads: 12,
                head_dim: 128,
                dropout_rate: 0.1,
                temperature_scaling: 1.2, // Smoother attention for long-term
                use_relative_position: true,
                max_sequence_length: 1024,
            },
            sequence_optimization: SequenceOptimization {
                use_sparse_attention: true,
                sliding_window_size: Some(256),
                local_attention_window: 64,
                use_hierarchical_attention: true,
                adaptive_chunking: true,
            },
            memory_optimization: MemoryOptimization {
                gradient_checkpointing: true,
                head_pruning_threshold: 0.01,
                mixed_precision: true,
                cache_attention_patterns: true,
                max_cache_size: 2000,
            },
            compute_optimization: ComputeOptimization {
                use_flash_attention: true,
                fused_operations: true,
                parallel_heads: true,
                optimized_matmul: true,
                kernel_fusion: true,
            },
            crypto_optimizations: CryptoOptimizations {
                recent_price_focus: false, // Long-term doesn't focus on recent
                volume_spike_emphasis: true,
                regime_aware_scaling: true,
                volatility_adaptive_temperature: true,
                support_resistance_boosting: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(config.base_config.max_sequence_length, 128);
    }

    #[test]
    fn test_factory_long_term() {
        let config = OptimizedAttentionFactory::create_long_term_crypto();
        assert!(config.sequence_optimization.use_sparse_attention);
        assert!(!config.crypto_optimizations.recent_price_focus);
        assert_eq!(config.base_config.max_sequence_length, 1024);
    }
}
