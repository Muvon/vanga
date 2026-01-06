// Mixture-of-Head Attention (MoH) implementation for VANGA LSTM
// Based on "Mixture-of-Head Attention for Multi-Modal Learning" paper
// OPTIMIZED: Fully vectorized implementation for CUDA performance
use crate::config::model::{AttentionConfig, MoHConfig};
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor, D};
use candle_nn::init::DEFAULT_KAIMING_UNIFORM;
use candle_nn::{linear, ops, Linear, Module, VarBuilder};
use std::collections::HashMap;

/// Mixture-of-Head Attention mechanism with dynamic head routing
/// Vectorized implementation for high performance on GPU
pub struct MixtureOfHeadAttention {
    config: AttentionConfig,
    moh_config: MoHConfig,
    input_dim: usize,
    head_dim: usize,

    // Unified projection layers for all heads
    // Shape: [input_dim, total_heads * head_dim]
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,

    // Output projection parameters (custom batched linear)
    // Weights: [total_heads, head_dim, input_dim]
    // Bias: [total_heads, input_dim]
    o_proj_weights: Tensor,
    o_proj_bias: Tensor,

    // Two-stage routing components (Equations 5-6 from paper)
    shared_router: Linear,    // W_s ∈ ℝ^{hs×d_in}
    routed_router: Linear,    // W_r ∈ ℝ^{(h-hs)×d_in}
    head_type_router: Linear, // W_h ∈ ℝ^{2×d_in}

    // Routing history for load balance loss calculation
    routing_history: Vec<Tensor>, // Store tensors directly [Batch, Seq, Heads]
    history_index: usize,         // For circular buffer implementation
    max_history_size: usize,      // Maximum history size to prevent unbounded growth
    step_count: usize,            // Track training steps for periodic cleanup

    device: Device,
}

impl MixtureOfHeadAttention {
    /// Create new Mixture-of-Head Attention layer
    pub fn new(
        input_dim: usize,
        config: AttentionConfig,
        vs: VarBuilder,
        device: Device,
    ) -> Result<Self> {
        let moh_config = config.moh.clone().unwrap_or_default();

        // Validate MoH configuration
        moh_config
            .validate()
            .map_err(|e| VangaError::ModelError(format!("MoH config validation failed: {}", e)))?;

        // Auto-optimize head dimension
        let head_dim = config.head_dim.map(|h| h as usize).unwrap_or_else(|| {
            Self::optimize_head_dimension(input_dim, moh_config.total_heads as usize)
        });

        let total_heads = moh_config.total_heads as usize;
        let total_dim = total_heads * head_dim;

        // Unified projections for Q, K, V
        let q_proj = linear(input_dim, total_dim, vs.pp("q_proj"))
            .map_err(|e| VangaError::ModelError(format!("Q projection failed: {}", e)))?;
        let k_proj = linear(input_dim, total_dim, vs.pp("k_proj"))
            .map_err(|e| VangaError::ModelError(format!("K projection failed: {}", e)))?;
        let v_proj = linear(input_dim, total_dim, vs.pp("v_proj"))
            .map_err(|e| VangaError::ModelError(format!("V projection failed: {}", e)))?;

        // Output projection parameters
        // We use raw tensors to enable batched matrix multiplication
        let o_proj_weights = vs
            .get_with_hints(
                (total_heads, head_dim, input_dim),
                "o_proj_weights",
                DEFAULT_KAIMING_UNIFORM,
            )
            .map_err(|e| {
                VangaError::ModelError(format!("Output projection weights failed: {}", e))
            })?;
        let o_proj_bias = vs
            .get((total_heads, input_dim), "o_proj_bias")
            .map_err(|e| VangaError::ModelError(format!("Output projection bias failed: {}", e)))?;

        // Create routing components for two-stage routing
        let shared_router = linear(
            input_dim,
            moh_config.shared_heads as usize,
            vs.pp("shared_router"),
        )
        .map_err(|e| VangaError::ModelError(format!("Shared router creation failed: {}", e)))?;

        let routed_heads_count = moh_config.total_heads - moh_config.shared_heads;
        let routed_router = linear(
            input_dim,
            routed_heads_count as usize,
            vs.pp("routed_router"),
        )
        .map_err(|e| VangaError::ModelError(format!("Routed router creation failed: {}", e)))?;

        let head_type_router = linear(input_dim, 2, vs.pp("head_type_router")).map_err(|e| {
            VangaError::ModelError(format!("Head type router creation failed: {}", e))
        })?;

        log::info!(
            "✅ MixtureOfHeadAttention initialized (Vectorized): {} total heads ({} shared + {} routed, top_k={}), head_dim={}, efficiency={:.1}%",
            moh_config.total_heads,
            moh_config.shared_heads,
            routed_heads_count,
            moh_config.top_k,
            head_dim,
            moh_config.efficiency_ratio() * 100.0
        );

        Ok(Self {
            config,
            moh_config,
            input_dim,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj_weights,
            o_proj_bias,
            shared_router,
            routed_router,
            head_type_router,
            routing_history: Vec::with_capacity(1000),
            history_index: 0,
            max_history_size: 1000,
            step_count: 0,
            device,
        })
    }

    /// Auto-optimize head dimension based on input size and crypto-specific patterns
    fn optimize_head_dimension(input_dim: usize, total_heads: usize) -> usize {
        let optimal_head_dim = match input_dim {
            1..=20 => 32,     // Small feature sets
            21..=50 => 64,    // Medium feature sets
            51..=100 => 64,   // Large feature sets
            101..=200 => 128, // Very large feature sets
            _ => 128,         // Extremely large feature sets
        };

        // Ensure reasonable limits
        let max_total_dim = input_dim * 2;
        let max_head_dim = max_total_dim / total_heads;

        std::cmp::min(optimal_head_dim, max_head_dim.max(16))
    }

    /// Forward pass with Mixture-of-Head attention and routing
    /// Fully vectorized implementation
    pub fn forward(&mut self, input: &Tensor, training: bool) -> Result<(Tensor, Tensor)> {
        // Increment step counter for memory management
        if training {
            self.step_count += 1;
        }

        // Validate input dimensions
        self.validate_input_dimensions(input)?;

        let (batch_size, seq_len, _) = input.dims3()?;
        let total_heads = self.moh_config.total_heads as usize;

        // 1. Compute Routing Scores (Vectorized)
        // Returns [Batch, Seq, TotalHeads]
        let (routing_weights, load_balance_loss) =
            self.compute_routing_scores_vectorized(input, training)?;

        // 2. Compute Multi-Head Attention for ALL heads (Vectorized)
        // Returns [Batch, Seq, TotalHeads, HeadDim]
        let head_outputs = self.compute_all_heads_attention(input, training)?;

        // 3. Apply Output Projections (Batched)
        // We need to project each head's output back to input_dim
        // head_outputs: [Batch, Seq, TotalHeads, HeadDim]
        // o_proj_weights: [TotalHeads, HeadDim, InputDim]

        // Reshape for batched matmul: [TotalHeads, Batch * Seq, HeadDim]
        let head_outputs_permuted = head_outputs
            .permute((2, 0, 1, 3))? // [TotalHeads, Batch, Seq, HeadDim]
            .reshape((total_heads, batch_size * seq_len, self.head_dim))?;

        // Perform batched matmul: [TotalHeads, Batch*Seq, HeadDim] @ [TotalHeads, HeadDim, InputDim]
        // Result: [TotalHeads, Batch*Seq, InputDim]
        let projected_outputs_flat = head_outputs_permuted.matmul(&self.o_proj_weights)?;

        // Add bias: [TotalHeads, InputDim] -> broadcast to [TotalHeads, Batch*Seq, InputDim]
        let bias_broadcast = self
            .o_proj_bias
            .unsqueeze(1)?
            .broadcast_as(projected_outputs_flat.shape())?;

        let projected_outputs_flat = (projected_outputs_flat + bias_broadcast)?;

        // Reshape back: [TotalHeads, Batch, Seq, InputDim] -> [Batch, Seq, TotalHeads, InputDim]
        let projected_outputs = projected_outputs_flat
            .reshape((total_heads, batch_size, seq_len, self.input_dim))?
            .permute((1, 2, 0, 3))?
            .contiguous()?;

        // 4. Apply Routing Weights
        // routing_weights: [Batch, Seq, TotalHeads] -> expand to [Batch, Seq, TotalHeads, 1]
        let routing_weights_expanded = routing_weights
            .unsqueeze(3)?
            .broadcast_as(projected_outputs.shape())?;

        // Weighted sum: sum(O_h * w_h) over heads
        let weighted_outputs = (projected_outputs * routing_weights_expanded)?;
        let final_output = weighted_outputs.sum(2)?; // Sum over heads dimension

        // Store routing history for load balance loss (only during training)
        if training {
            // Average routing scores across batch and sequence for history
            // routing_weights: [Batch, Seq, TotalHeads] -> mean over Batch, Seq -> [TotalHeads]
            let avg_routing_scores = routing_weights.mean(0)?.mean(0)?.detach(); // Detach to avoid keeping graph

            // MEMORY FIX: Use circular buffer with proper memory management
            if self.routing_history.len() < self.max_history_size {
                self.routing_history.push(avg_routing_scores);
            } else {
                // Circular buffer: overwrite oldest entry
                self.routing_history[self.history_index] = avg_routing_scores;
                self.history_index = (self.history_index + 1) % self.max_history_size;
            }
        }

        log::debug!(
            "MoH forward (Vectorized): batch_size={}, seq_len={}, load_balance_loss={:.6}",
            batch_size,
            seq_len,
            load_balance_loss
        );

        Ok((final_output, routing_weights))
    }

    /// Compute routing scores using two-stage routing (Vectorized)
    /// CRITICAL: Routing weights MUST have gradients for learning!
    /// Only detach for load balance loss calculation (as per paper)
    fn compute_routing_scores_vectorized(
        &self,
        input: &Tensor,
        training: bool,
    ) -> Result<(Tensor, f32)> {
        let (batch_size, seq_len, _) = input.dims3()?;

        // Stage 1: Head type routing (α1, α2)
        let head_type_logits = self.head_type_router.forward(input)?; // [Batch, Seq, 2]

        // CRITICAL: Do NOT detach routing decisions - they need gradients!
        // Only detach when calculating load balance loss (separate from forward pass)
        let head_type_probs = ops::softmax(&head_type_logits, 2)?;

        // Extract α1 and α2: [Batch, Seq, 1]
        let alpha1 = head_type_probs.narrow(2, 0, 1)?;
        let alpha2 = head_type_probs.narrow(2, 1, 1)?;

        // Stage 2: Individual head routing
        let shared_logits = self.shared_router.forward(input)?; // [Batch, Seq, SharedHeads]

        // CRITICAL: Keep gradients flowing through routing
        let shared_probs = ops::softmax(&shared_logits, 2)?;

        // Calculate adaptive temperature (Vectorized)
        let adaptive_temperature = self.calculate_adaptive_temperature_vectorized(input)?;
        // adaptive_temperature: [Batch, Seq, 1]

        // Routed heads
        let num_routed_heads =
            (self.moh_config.total_heads - self.moh_config.shared_heads) as usize;

        let routed_scores = if num_routed_heads > 0 {
            let routed_logits = self.routed_router.forward(input)?; // [Batch, Seq, RoutedHeads]

            // Apply temperature scaling
            let routed_probs_scaled = routed_logits.broadcast_div(&adaptive_temperature)?;

            // CRITICAL: Keep gradients for routing learning
            ops::softmax(&routed_probs_scaled, 2)?
        } else {
            Tensor::zeros((batch_size, seq_len, 0), input.dtype(), &self.device)?
        };

        // Combine scores
        // Shared: alpha1 * shared_probs
        let shared_final = shared_probs.broadcast_mul(&alpha1)?;

        // Routed: alpha2 * routed_probs (Soft Top-K)
        // Note: We implement Soft Top-K by just using the probabilities directly.
        // Hard Top-K is difficult to vectorize efficiently and differentiable.
        // The softmax naturally suppresses non-top heads.
        let routed_final = if num_routed_heads > 0 {
            routed_scores.broadcast_mul(&alpha2)?
        } else {
            routed_scores
        };

        // Concatenate: [Batch, Seq, TotalHeads]
        let final_scores = Tensor::cat(&[&shared_final, &routed_final], 2)?;

        // Calculate load balance loss (simplified for vectorized)
        // CRITICAL: Detach ONLY for load balance loss calculation (as per paper)
        // This prevents load balance loss from affecting routing gradients
        let load_balance_loss = if training && num_routed_heads > 0 {
            // Detach routed_final for load balance calculation only
            let routed_final_detached = routed_final.detach();
            let mean_usage = routed_final_detached.mean(0)?.mean(0)?; // [RoutedHeads]
                                                                      // Use coefficient of variation as load balance metric
            let usage_std = mean_usage.var(0)?.sqrt()?;
            let usage_mean = mean_usage.mean(0)?;
            let cv = usage_std / (usage_mean + 1e-6);
            cv?.to_scalar::<f32>().unwrap_or(0.0)
        } else {
            0.0
        };

        Ok((final_scores, load_balance_loss))
    }

    /// Calculate adaptive temperature (Vectorized)
    fn calculate_adaptive_temperature_vectorized(&self, input: &Tensor) -> Result<Tensor> {
        // input: [Batch, Seq, Dim]
        // Calculate variance across feature dimension
        let mean = input.mean_keepdim(2)?; // [Batch, Seq, 1]
        let diff = input.broadcast_sub(&mean)?;
        let variance = diff.sqr()?.mean_keepdim(2)?; // [Batch, Seq, 1]
        let std_dev = variance.sqrt()?;

        // Map std_dev to temperature
        let base_temp = self.moh_config.routing_temperature;
        let temp = ((std_dev * 0.5)? + base_temp)?;

        // Clamp [0.5, 2.5]
        let temp = temp.clamp(0.5, 2.5)?;

        Ok(temp)
    }

    /// Compute attention for all heads in parallel
    fn compute_all_heads_attention(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        let (batch_size, seq_len, _) = input.dims3()?;
        let total_heads = self.moh_config.total_heads as usize;
        let head_dim = self.head_dim;

        // 1. Project Q, K, V
        let q = self.q_proj.forward(input)?; // [Batch, Seq, TotalHeads * HeadDim]
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // 2. Reshape to [Batch, Seq, Heads, HeadDim]
        let q = q.reshape((batch_size, seq_len, total_heads, head_dim))?;
        let k = k.reshape((batch_size, seq_len, total_heads, head_dim))?;
        let v = v.reshape((batch_size, seq_len, total_heads, head_dim))?;

        // 3. Transpose for attention: [Batch, Heads, Seq, HeadDim]
        let q = q.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;

        // 4. Scaled Dot-Product Attention
        let scale = (head_dim as f64).sqrt();
        let q_scaled = (q / scale)?;

        // Attn = Q @ K^T
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn_scores = q_scaled.matmul(&k_t)?; // [Batch, Heads, Seq, Seq]

        // 5. Causal Mask
        let mask = self.create_causal_mask(seq_len)?;
        let mask = mask.broadcast_as(attn_scores.shape())?;
        let attn_scores = (attn_scores + mask)?;

        // 6. Softmax
        let attn_weights = ops::softmax(&attn_scores, D::Minus1)?;

        // 7. Dropout
        let attn_weights = if training && self.config.dropout_rate > 0.0 {
            ops::dropout(&attn_weights, self.config.dropout_rate as f32)?
        } else {
            attn_weights
        };

        // 8. Apply to V
        let output = attn_weights.matmul(&v)?; // [Batch, Heads, Seq, HeadDim]

        // 9. Transpose back: [Batch, Seq, Heads, HeadDim]
        let output = output.permute((0, 2, 1, 3))?.contiguous()?;

        Ok(output)
    }

    /// Create causal mask
    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mut mask_data = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 0.0;
            }
        }
        let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), &self.device)?;
        Ok(mask.unsqueeze(0)?.unsqueeze(0)?) // [1, 1, Seq, Seq]
    }

    /// Calculate total load balance loss from routing history
    pub fn calculate_load_balance_loss(&self) -> Result<Tensor> {
        if self.routing_history.is_empty() {
            return Tensor::new(0.0f32, &self.device).map_err(|e| {
                VangaError::ModelError(format!("Load balance loss tensor creation failed: {}", e))
            });
        }

        // Stack history: [HistorySize, TotalHeads]
        let history = Tensor::stack(&self.routing_history, 0)?;

        // Only routed heads
        let routed_start = self.moh_config.shared_heads as usize;
        let routed_end = self.moh_config.total_heads as usize;

        if routed_start >= routed_end {
            return Tensor::new(0.0f32, &self.device)
                .map_err(|e| VangaError::ModelError(e.to_string()));
        }

        let routed_history = history.narrow(1, routed_start, routed_end - routed_start)?;

        // Calculate variance of usage across heads
        let mean_usage = routed_history.mean(0)?; // [RoutedHeads]
        let variance = mean_usage.var(0)?;

        // We want to minimize variance (maximize balance)
        Ok(variance)
    }

    // ... Helper methods ...
    pub fn get_routing_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert(
            "total_heads".to_string(),
            self.moh_config.total_heads as f64,
        );
        stats.insert(
            "shared_heads".to_string(),
            self.moh_config.shared_heads as f64,
        );
        stats.insert("top_k".to_string(), self.moh_config.top_k as f64);
        stats.insert(
            "efficiency_ratio".to_string(),
            self.moh_config.efficiency_ratio(),
        );
        stats.insert(
            "routing_history_length".to_string(),
            self.routing_history.len() as f64,
        );
        stats
    }

    pub fn clear_routing_history(&mut self) {
        self.routing_history.clear();
        self.history_index = 0;
        self.step_count = 0;
    }

    pub fn compact_memory(&mut self) {
        self.routing_history.shrink_to_fit();
    }

    pub fn memory_usage(&self) -> usize {
        self.routing_history.len() * self.moh_config.total_heads as usize
    }

    pub fn validate_input_dimensions(&self, input: &Tensor) -> Result<()> {
        let input_dims = input.dims();
        if input_dims.len() != 3 {
            return Err(VangaError::ModelError(format!(
                "Expected 3D input tensor [batch, seq, features], got shape: {:?}",
                input_dims
            )));
        }
        if input_dims[2] != self.input_dim {
            return Err(VangaError::ModelError(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.input_dim, input_dims[2]
            )));
        }
        Ok(())
    }

    pub fn get_architecture_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert(
            "model_type".to_string(),
            "MixtureOfHeadAttention (Vectorized)".to_string(),
        );
        info.insert("input_dim".to_string(), self.input_dim.to_string());
        info.insert("head_dim".to_string(), self.head_dim.to_string());
        info
    }

    pub fn get_config(&self) -> &AttentionConfig {
        &self.config
    }
}
