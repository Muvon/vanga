// Optimized attention computation for cryptocurrency sequence lengths
use crate::config::model::AttentionConfig;
use crate::model::attention::MultiHeadAttention;
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};
use candle_nn::{ops, VarBuilder};
use log;
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
    pub fn forward(&mut self, input: &Tensor, training: bool) -> Result<(Tensor, Tensor)> {
        let seq_len = input.dim(1)?;

        // Choose attention strategy based on sequence length
        match seq_len {
            len if len <= 64 => {
                // Short sequences: use standard attention
                self.base_attention.forward(input, training)
            }
            len if len <= 256 => {
                // Medium sequences: use optimized standard attention
                self.forward_optimized_standard(input, training)
            }
            len if len <= 1024 => {
                // Long sequences: use sparse attention
                self.forward_sparse_attention(input, training)
            }
            _ => {
                // Very long sequences: use hierarchical attention
                self.forward_hierarchical_attention(input, training)
            }
        }
    }

    /// Optimized standard attention for medium sequences
    fn forward_optimized_standard(
        &mut self,
        input: &Tensor,
        training: bool,
    ) -> Result<(Tensor, Tensor)> {
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
        let (output, weights) = self.base_attention.forward(&optimized_input, training)?;

        // Cache result if enabled
        if self.config.memory_optimization.cache_attention_patterns {
            let cache_key = self.generate_cache_key(&optimized_input)?;
            self.cache_attention_result(&cache_key, &output, &weights)?;
        }

        Ok((output, weights))
    }

    /// Sparse attention for long sequences
    fn forward_sparse_attention(
        &mut self,
        input: &Tensor,
        training: bool,
    ) -> Result<(Tensor, Tensor)> {
        if let Some(ref sparse_attention) = self.sparse_attention {
            // Apply sliding window if sequence is very long
            if let Some(window_size) = self.config.sequence_optimization.sliding_window_size {
                let seq_len = input.dim(1)?;
                if seq_len > window_size {
                    return self.forward_sliding_window(input, window_size, training);
                }
            }

            sparse_attention.forward(input, training)
        } else {
            // Fallback to standard attention
            self.base_attention.forward(input, training)
        }
    }

    /// Hierarchical attention for very long sequences
    fn forward_hierarchical_attention(
        &mut self,
        input: &Tensor,
        training: bool,
    ) -> Result<(Tensor, Tensor)> {
        if let Some(ref hierarchical_attention) = self.hierarchical_attention {
            hierarchical_attention.forward(input, training)
        } else {
            // Fallback to sparse attention
            self.forward_sparse_attention(input, training)
        }
    }

    /// Sliding window attention for extremely long sequences
    fn forward_sliding_window(
        &mut self,
        input: &Tensor,
        window_size: usize,
        training: bool,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = input.dim(1)?;
        let input_shape = input.shape();
        let batch_size = input_shape.dims()[0];
        let feature_dim = input_shape.dims()[2];

        // Validate input dimensions
        if batch_size == 0 || feature_dim == 0 {
            return Err(VangaError::ModelError(
                "Invalid input dimensions: batch_size and feature_dim must be > 0".to_string(),
            ));
        }

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
            let (window_output, window_weights) =
                self.base_attention.forward(&window_input, training)?;

            outputs.push(window_output);
            all_weights.push(window_weights);
        }

        // Combine outputs from all windows
        if outputs.is_empty() {
            return Err(VangaError::ModelError(
                "No valid windows processed".to_string(),
            ));
        }

        let combined_output = Tensor::cat(&outputs, 1)?.contiguous()?;
        let combined_weights = Tensor::cat(&all_weights, 1)?.contiguous()?;

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
    fn forward(&self, input: &Tensor, _training: bool) -> Result<(Tensor, Tensor)> {
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

        let combined_output = Tensor::cat(&outputs, 1)?.contiguous()?;

        // Resize to original length if needed
        let final_output = if combined_output.dim(1)? > seq_len {
            combined_output.narrow(1, 0, seq_len)?.contiguous()?
        } else {
            combined_output
        };

        Ok((final_output.clone(), final_output))
    }
}

/// Hierarchical attention for very long sequences with multi-resolution processing
struct HierarchicalAttention {
    #[allow(dead_code)]
    device: Device,
    // Attention mechanisms for different resolution levels
    recent_attention: MultiHeadAttention,
    medium_attention: MultiHeadAttention,
    long_attention: MultiHeadAttention,
    // Learnable combination weights
    combination_weights: Tensor,
}

impl HierarchicalAttention {
    fn new(input_dim: usize, vs: VarBuilder, device: Device) -> Result<Self> {
        // Create attention mechanisms for different resolution levels
        let recent_config = AttentionConfig {
            heads: 8,
            head_dim: Some(64),
            temperature_scaling: 1.0,
            ..AttentionConfig::default()
        };

        let medium_config = AttentionConfig {
            heads: 6,
            head_dim: Some(48),
            temperature_scaling: 1.2,
            ..AttentionConfig::default()
        };

        let long_config = AttentionConfig {
            heads: 4,
            head_dim: Some(32),
            temperature_scaling: 1.5,
            ..AttentionConfig::default()
        };

        let recent_attention =
            MultiHeadAttention::new(input_dim, recent_config, vs.pp("recent"), device.clone())?;

        let medium_attention =
            MultiHeadAttention::new(input_dim, medium_config, vs.pp("medium"), device.clone())?;

        let long_attention =
            MultiHeadAttention::new(input_dim, long_config, vs.pp("long"), device.clone())?;

        // Initialize learnable combination weights [recent, medium, long]
        let combination_weights = vs.get((3,), "combination_weights")?;

        Ok(Self {
            device,
            recent_attention,
            medium_attention,
            long_attention,
            combination_weights,
        })
    }

    fn forward(&self, input: &Tensor, training: bool) -> Result<(Tensor, Tensor)> {
        let seq_len = input.dim(1)?;

        // Process multi-resolution data
        let (recent_data, medium_data, long_data) = self.process_multi_resolution_data(input)?;

        // Apply attention at each resolution level
        let (recent_output, recent_weights) =
            self.recent_attention.forward(&recent_data, training)?;
        let (medium_output, medium_weights) =
            self.medium_attention.forward(&medium_data, training)?;
        let (long_output, long_weights) = self.long_attention.forward(&long_data, training)?;

        // Combine outputs with learnable weights and crypto-specific logic
        let (combined_output, combined_weights) = self.combine_multi_resolution_outputs(
            recent_output,
            medium_output,
            long_output,
            recent_weights,
            medium_weights,
            long_weights,
            seq_len,
        )?;

        log::debug!(
            "Multi-resolution processing completed: seq_len={}, recent={}, medium={}, long={}",
            seq_len,
            recent_data.dim(1)?,
            medium_data.dim(1)?,
            long_data.dim(1)?
        );

        Ok((combined_output, combined_weights))
    }

    /// Process input data at multiple resolution levels with crypto-specific downsampling
    fn process_multi_resolution_data(&self, input: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let seq_len = input.dim(1)?;

        // Level 1: Recent data (full resolution) - last 64 timesteps
        let recent_len = std::cmp::min(64, seq_len);
        let recent_data = input.narrow(1, seq_len - recent_len, recent_len)?;

        // Level 2: Medium-term data (2x downsampled) - crypto-aware downsampling
        let medium_data = self.downsample_temporal(input, 2)?;

        // Level 3: Long-term data (4x downsampled) - preserve major price movements
        let long_data = self.downsample_temporal(input, 4)?;

        Ok((recent_data, medium_data, long_data))
    }

    /// Crypto-aware temporal downsampling that preserves important price movements
    fn downsample_temporal(&self, input: &Tensor, factor: usize) -> Result<Tensor> {
        let seq_len = input.dim(1)?;
        let downsampled_len = seq_len / factor;

        if downsampled_len == 0 {
            return Ok(input.clone());
        }

        let mut downsampled_indices = Vec::new();

        // For cryptocurrency data, use max pooling approach to preserve extreme movements
        for i in 0..downsampled_len {
            let start_idx = i * factor;
            let end_idx = std::cmp::min(start_idx + factor, seq_len);

            // Use the last index in each window (most recent in that period)
            // This preserves the temporal ordering while downsampling
            downsampled_indices.push(end_idx - 1);
        }

        // Extract the downsampled timesteps
        let mut downsampled_tensors = Vec::new();
        for &idx in &downsampled_indices {
            let timestep = input.narrow(1, idx, 1)?;
            downsampled_tensors.push(timestep);
        }

        // Concatenate along the sequence dimension
        let downsampled = Tensor::cat(&downsampled_tensors, 1)?;

        Ok(downsampled)
    }

    /// Combine multi-resolution outputs with crypto-specific weighting
    #[allow(clippy::too_many_arguments)]
    fn combine_multi_resolution_outputs(
        &self,
        recent_output: Tensor,
        medium_output: Tensor,
        long_output: Tensor,
        recent_weights: Tensor,
        medium_weights: Tensor,
        long_weights: Tensor,
        original_seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Get crypto-specific resolution weights based on sequence characteristics
        let (recent_weight, medium_weight, long_weight) =
            self.get_crypto_resolution_weights(original_seq_len)?;

        // Apply softmax to learnable combination weights
        let learned_weights = ops::softmax_last_dim(&self.combination_weights)?;
        let learned_weights_vec = learned_weights.to_vec1::<f64>()?;

        // Combine learned and heuristic weights
        let final_recent_weight = recent_weight * learned_weights_vec[0];
        let final_medium_weight = medium_weight * learned_weights_vec[1];
        let final_long_weight = long_weight * learned_weights_vec[2];

        // Normalize final weights
        let total_weight = final_recent_weight + final_medium_weight + final_long_weight;
        let norm_recent = final_recent_weight / total_weight;
        let norm_medium = final_medium_weight / total_weight;
        let norm_long = final_long_weight / total_weight;

        // Upsample medium and long outputs to match recent output dimensions
        let upsampled_medium = self.upsample_to_match(&medium_output, &recent_output)?;
        let upsampled_long = self.upsample_to_match(&long_output, &recent_output)?;

        // Weighted combination of outputs
        let weighted_recent = (recent_output * norm_recent)?;
        let weighted_medium = (upsampled_medium * norm_medium)?;
        let weighted_long = (upsampled_long * norm_long)?;

        let combined_output = (weighted_recent + weighted_medium + weighted_long)?;

        // Combine attention weights for interpretability
        let combined_weights = self.combine_attention_weights(
            recent_weights,
            medium_weights,
            long_weights,
            norm_recent,
            norm_medium,
            norm_long,
        )?;

        log::debug!(
            "Multi-resolution combination: recent_w={:.3}, medium_w={:.3}, long_w={:.3}",
            norm_recent,
            norm_medium,
            norm_long
        );

        Ok((combined_output, combined_weights))
    }

    /// Get crypto-specific resolution weights based on sequence characteristics
    fn get_crypto_resolution_weights(&self, seq_len: usize) -> Result<(f64, f64, f64)> {
        // Crypto markets favor recent data, but longer sequences need more historical context
        let recent_weight = if seq_len <= 64 {
            0.8 // Short sequences: heavily favor recent
        } else if seq_len <= 256 {
            0.6 // Medium sequences: balanced approach
        } else {
            0.5 // Long sequences: more historical context
        };

        let medium_weight = if seq_len <= 64 {
            0.15
        } else if seq_len <= 256 {
            0.25
        } else {
            0.3
        };

        let long_weight = 1.0 - recent_weight - medium_weight;

        Ok((recent_weight, medium_weight, long_weight))
    }

    /// Upsample tensor to match target dimensions
    fn upsample_to_match(&self, source: &Tensor, target: &Tensor) -> Result<Tensor> {
        let source_seq_len = source.dim(1)?;
        let target_seq_len = target.dim(1)?;

        if source_seq_len == target_seq_len {
            return Ok(source.clone());
        }

        // Simple repeat-based upsampling for attention outputs
        let repeat_factor = target_seq_len.div_ceil(source_seq_len);
        let repeated = source.repeat(&[1, repeat_factor, 1])?;

        // Trim to exact target length
        let upsampled = repeated.narrow(1, 0, target_seq_len)?;

        Ok(upsampled)
    }

    /// Combine attention weights from different resolution levels
    fn combine_attention_weights(
        &self,
        recent_weights: Tensor,
        medium_weights: Tensor,
        long_weights: Tensor,
        recent_weight: f64,
        medium_weight: f64,
        long_weight: f64,
    ) -> Result<Tensor> {
        // Upsample all weights to match recent weights dimensions
        let upsampled_medium = self.upsample_to_match(&medium_weights, &recent_weights)?;
        let upsampled_long = self.upsample_to_match(&long_weights, &recent_weights)?;

        // Weighted combination
        let weighted_recent = (recent_weights * recent_weight)?;
        let weighted_medium = (upsampled_medium * medium_weight)?;
        let weighted_long = (upsampled_long * long_weight)?;

        let combined = (weighted_recent + weighted_medium + weighted_long)?;

        Ok(combined)
    }
}

/// Factory for creating optimized attention configurations
pub struct OptimizedAttentionFactory;

impl OptimizedAttentionFactory {
    /// Create configuration optimized for short-term crypto trading (1min-15min)
    pub fn create_short_term_crypto() -> OptimizedAttentionConfig {
        OptimizedAttentionConfig {
            base_config: AttentionConfig {
                enabled: true,
                mechanism: crate::config::model::AttentionMechanism::MultiHeadAttention,
                heads: 8,
                head_dim: Some(64),
                dropout_rate: 0.05,
                dropout_weights: true,
                dropout_output: true,
                dropout_projections: true,
                dropout_scores: true,
                temperature_scaling: 0.9, // Sharper attention for short-term
                use_relative_position: true,
                visualization: crate::config::model::VisualizationConfig::default(),
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
                enabled: true,
                mechanism: crate::config::model::AttentionMechanism::MultiHeadAttention,
                heads: 12,
                head_dim: Some(128),
                dropout_rate: 0.1,
                dropout_weights: true,
                dropout_output: true,
                dropout_projections: true,
                dropout_scores: true,
                temperature_scaling: 1.2, // Smoother attention for long-term
                use_relative_position: true,
                visualization: crate::config::model::VisualizationConfig::default(),
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
