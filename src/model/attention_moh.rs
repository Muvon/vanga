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

    // Advanced features (optional, enabled via config)
    volatility_estimator: Option<Linear>, // Volatility prediction for adaptive routing
    importance_scorers: Option<Vec<Linear>>, // Token importance for sparse attention
    offset_predictors: Option<Vec<Linear>>, // Temporal offsets for deformable attention

    // Routing history for load balance loss calculation
    routing_history: Vec<Tensor>, // Store tensors directly [Batch, Seq, Heads]
    history_index: usize,         // For circular buffer implementation
    max_history_size: usize,      // Maximum history size to prevent unbounded growth
    step_count: usize,            // Track training steps for periodic cleanup

    last_sparsity_mean: Option<f32>, // Track last adaptive sparsity decision for debugging/tests

    // Performance optimization: cached causal mask and position encodings
    causal_mask_cache: std::cell::RefCell<HashMap<usize, Tensor>>,
    position_cache: std::cell::RefCell<HashMap<usize, (Tensor, Tensor)>>,

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

        // Initialize advanced features if enabled
        let volatility_estimator = if moh_config.volatility_adaptive {
            Some(
                linear(input_dim, 1, vs.pp("volatility_estimator")).map_err(|e| {
                    VangaError::ModelError(format!("Volatility estimator creation failed: {}", e))
                })?,
            )
        } else {
            None
        };

        let importance_scorers = if moh_config.sparse_attention || moh_config.learnable_sampling {
            let mut scorers = Vec::with_capacity(total_heads);
            for i in 0..total_heads {
                let scorer = linear(head_dim, 1, vs.pp(format!("importance_scorer_{}", i)))
                    .map_err(|e| {
                        VangaError::ModelError(format!(
                            "Importance scorer {} creation failed: {}",
                            i, e
                        ))
                    })?;
                scorers.push(scorer);
            }
            Some(scorers)
        } else {
            None
        };

        let offset_predictors = if moh_config.deformable_attention {
            let mut predictors = Vec::with_capacity(total_heads);
            for i in 0..total_heads {
                let predictor = linear(
                    head_dim,
                    moh_config.num_offsets,
                    vs.pp(format!("offset_predictor_{}", i)),
                )
                .map_err(|e| {
                    VangaError::ModelError(format!("Offset predictor {} creation failed: {}", i, e))
                })?;
                predictors.push(predictor);
            }
            Some(predictors)
        } else {
            None
        };

        log::info!(
            "✅ MixtureOfHeadAttention initialized (Vectorized): {} total heads ({} shared + {} routed, top_k={}), head_dim={}, efficiency={:.1}%{}",
            moh_config.total_heads,
            moh_config.shared_heads,
            routed_heads_count,
            moh_config.top_k,
            head_dim,
            moh_config.efficiency_ratio() * 100.0,
            if moh_config.has_advanced_features() {
                format!(" [Advanced: vol={}, sparse={}, deform={}, sampling={}]",
                    moh_config.volatility_adaptive,
                    moh_config.sparse_attention,
                    moh_config.deformable_attention,
                    moh_config.learnable_sampling)
            } else {
                String::new()
            }
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
            volatility_estimator,
            importance_scorers,
            offset_predictors,
            routing_history: Vec::with_capacity(1000),
            history_index: 0,
            max_history_size: 1000,
            step_count: 0,
            last_sparsity_mean: None,
            causal_mask_cache: std::cell::RefCell::new(HashMap::new()),
            position_cache: std::cell::RefCell::new(HashMap::new()),
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
        let (routing_weights, load_balance_loss, volatility_signal) =
            self.compute_routing_scores_vectorized(input, training)?;

        // 2. Compute Multi-Head Attention for ALL heads (Vectorized)
        // Returns [Batch, Seq, TotalHeads, HeadDim]
        let head_outputs = self.compute_all_heads_attention(
            input,
            training,
            Some(&volatility_signal),
            Some(&routing_weights),
        )?;

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
    ) -> Result<(Tensor, f32, Tensor)> {
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
        let (adaptive_temperature, volatility_signal) =
            self.calculate_adaptive_temperature_vectorized(input)?;
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

        Ok((final_scores, load_balance_loss, volatility_signal))
    }

    /// Calculate adaptive temperature (Vectorized)
    /// With optional volatility-based adaptation
    fn calculate_adaptive_temperature_vectorized(
        &self,
        input: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // input: [Batch, Seq, Dim]

        if self.moh_config.volatility_adaptive {
            if let Some(ref vol_estimator) = self.volatility_estimator {
                let vol_logits = vol_estimator.forward(input)?; // [Batch, Seq, 1]
                let volatility = ops::sigmoid(&vol_logits)?;
                let vol_smoothed = self.smooth_volatility(&volatility)?;

                let base_temp = self.moh_config.routing_temperature;
                let vol_mult = self.moh_config.volatility_multiplier;

                let base_temp_tensor = Tensor::new(base_temp as f32, &self.device)?
                    .broadcast_as(vol_smoothed.shape())?;
                let vol_mult_tensor = Tensor::new(vol_mult as f32, &self.device)?
                    .broadcast_as(vol_smoothed.shape())?;

                let weighted_vol = vol_smoothed.broadcast_mul(&vol_mult_tensor)?;
                let temp = (weighted_vol + base_temp_tensor)?.clamp(0.5, 2.5)?;
                let volatility_signal = vol_smoothed.clamp(0.0, 1.0)?;

                return Ok((temp, volatility_signal));
            }
        }

        // Fallback: use per-sequence variance as volatility proxy
        let mean = input.mean_keepdim(2)?; // [Batch, Seq, 1]
        let diff = input.broadcast_sub(&mean)?;
        let variance = diff.sqr()?.mean_keepdim(2)?; // [Batch, Seq, 1]
        let std_dev = variance.sqrt()?;

        let base_temp = self.moh_config.routing_temperature;
        let base_temp_tensor =
            Tensor::new(base_temp as f32, &self.device)?.broadcast_as(std_dev.shape())?;
        let half = Tensor::new(0.5f32, &self.device)?.broadcast_as(std_dev.shape())?;
        let scaled_std = std_dev.broadcast_mul(&half)?;
        let temp = (scaled_std + base_temp_tensor)?.clamp(0.5, 2.5)?;

        let mean_std = std_dev
            .mean_all()?
            .to_scalar::<f32>()
            .unwrap_or(1.0)
            .max(1e-3);
        let norm_factor = Tensor::new(1.0 / mean_std, &self.device)?;
        let norm_factor = norm_factor.broadcast_as(std_dev.shape())?;
        let normalized = std_dev.broadcast_mul(&norm_factor)?;
        let volatility_signal = normalized.tanh()?.clamp(0.0, 1.0)?;

        Ok((temp, volatility_signal))
    }

    /// Smooth volatility over window using exponential moving average
    fn smooth_volatility(&self, volatility: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = volatility.dims3()?;
        if seq_len <= 1 {
            return Ok(volatility.clone());
        }

        let window = self.moh_config.volatility_window.max(2);
        let alpha = 2.0f32 / (window as f32 + 1.0);
        let alpha_tensor = Tensor::new(alpha, &self.device)?;
        let complement = Tensor::new(1.0f32 - alpha, &self.device)?;

        let mut smoothed: Vec<Tensor> = Vec::with_capacity(seq_len);
        let mut previous = volatility.narrow(1, 0, 1)?;
        smoothed.push(previous.clone());

        for step in 1..seq_len {
            let current = volatility.narrow(1, step, 1)?;
            let alpha_weights = alpha_tensor.broadcast_as(current.shape())?;
            let complement_weights = complement.broadcast_as(previous.shape())?;
            let blended = ((current * alpha_weights)? + (previous * complement_weights)?)?;
            smoothed.push(blended.clone());
            previous = blended;
        }

        Ok(Tensor::cat(&smoothed, 1)?.reshape((batch, seq_len, 1))?)
    }

    /// Compute attention for all heads in parallel
    /// With optional sparse and deformable attention
    fn compute_all_heads_attention(
        &mut self,
        input: &Tensor,
        training: bool,
        volatility_signal: Option<&Tensor>,
        routing_weights: Option<&Tensor>,
    ) -> Result<Tensor> {
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

        // 4. Apply sparse or deformable attention if enabled
        let output = if self.moh_config.sparse_attention || self.moh_config.deformable_attention {
            self.compute_sparse_deformable_attention(
                &q,
                &k,
                &v,
                training,
                volatility_signal,
                routing_weights,
            )?
        } else {
            self.compute_full_attention(&q, &k, &v, training)?
        };

        // 5. Transpose back: [Batch, Seq, Heads, HeadDim]
        let output = output.permute((0, 2, 1, 3))?.contiguous()?;

        Ok(output)
    }

    /// Compute full (dense) attention
    fn compute_full_attention(
        &self,
        q: &Tensor, // [Batch, Heads, Seq, HeadDim]
        k: &Tensor,
        v: &Tensor,
        training: bool,
    ) -> Result<Tensor> {
        let head_dim = self.head_dim;
        let seq_len = q.dim(2)?;

        // Scaled Dot-Product Attention
        let scale = (head_dim as f64).sqrt();
        let q_scaled = (q / scale)?;

        // Attn = Q @ K^T
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn_scores = q_scaled.matmul(&k_t)?; // [Batch, Heads, Seq, Seq]

        // Causal Mask
        let mask = self.create_causal_mask(seq_len)?;
        let mask = mask.broadcast_as(attn_scores.shape())?;
        let attn_scores = (attn_scores + mask)?;

        // Softmax
        let attn_weights = ops::softmax(&attn_scores, D::Minus1)?;

        // Dropout
        let attn_weights = if training && self.config.dropout_rate > 0.0 {
            ops::dropout(&attn_weights, self.config.dropout_rate as f32)?
        } else {
            attn_weights
        };

        // Apply to V
        let output = attn_weights.matmul(v)?; // [Batch, Heads, Seq, HeadDim]

        Ok(output)
    }

    /// Compute sparse and/or deformable attention
    /// OPTIMIZED: Batched processing for all heads simultaneously
    fn compute_sparse_deformable_attention(
        &mut self,
        q: &Tensor, // [Batch, Heads, Seq, HeadDim]
        k: &Tensor,
        v: &Tensor,
        training: bool,
        volatility_signal: Option<&Tensor>,
        routing_weights: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, total_heads, seq_len, head_dim) = q.dims4()?;
        let scale = (head_dim as f64).sqrt();

        let sparsity_ratio =
            self.compute_sparsity_signal(batch_size, seq_len, volatility_signal, routing_weights)?;

        // OPTIMIZATION: Compute attention for ALL heads at once
        let q_scaled = (q / scale)?;
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let mut attn_scores = q_scaled.matmul(&k_t)?; // [Batch, Heads, Seq, Seq]

        // Compute importance for all heads (fully batched)
        let importance = if self.moh_config.learnable_sampling && self.importance_scorers.is_some()
        {
            if let Some(ref scorers) = self.importance_scorers {
                let q_for_scoring = q.permute((0, 2, 1, 3))?.contiguous()?;
                let q_flat =
                    q_for_scoring.reshape((batch_size * seq_len * total_heads, head_dim))?;

                let mut head_scores = Vec::with_capacity(total_heads);
                for (head_idx, scorer) in scorers.iter().enumerate() {
                    let start_idx = head_idx * batch_size * seq_len;
                    let q_head = q_flat.narrow(0, start_idx, batch_size * seq_len)?;
                    head_scores.push(scorer.forward(&q_head)?);
                }

                Tensor::stack(&head_scores, 1)?
                    .reshape((batch_size, seq_len, total_heads, 1))?
                    .permute((0, 2, 1, 3))? // [Batch, Heads, Seq, 1]
            } else {
                q.sqr()?.sum(D::Minus1)?.unsqueeze(3)?
            }
        } else {
            q.sqr()?.sum(D::Minus1)?.unsqueeze(3)?
        };

        // Add deformable bias if enabled (batched)
        if self.moh_config.deformable_attention {
            let deform_bias = self.compute_deformable_bias_batched(q, seq_len)?;
            attn_scores = (attn_scores + deform_bias)?;
        }

        // Apply sparse mask (batched)
        let sparse_logits =
            self.apply_sparse_mask_batched(&attn_scores, &importance, &sparsity_ratio)?;

        // Causal mask (batched)
        let mask = self.create_causal_mask(seq_len)?;
        let mask = mask.broadcast_as(sparse_logits.shape())?;
        let masked_scores = (sparse_logits + mask)?;

        // Softmax + dropout (batched)
        let attn_weights = ops::softmax(&masked_scores, D::Minus1)?;
        let attn_weights = if training && self.config.dropout_rate > 0.0 {
            ops::dropout(&attn_weights, self.config.dropout_rate as f32)?
        } else {
            attn_weights
        };

        // Apply to V (batched)
        attn_weights
            .matmul(v)
            .map_err(|e| VangaError::ModelError(format!("Batched attention failed: {}", e)))
    }

    /// Compute deformable bias for all heads (batched)
    fn compute_deformable_bias_batched(&self, q: &Tensor, seq_len: usize) -> Result<Tensor> {
        let (batch_size, total_heads, _, head_dim) = q.dims4()?;
        let num_offsets = self.moh_config.num_offsets.max(2);

        let offsets = if let Some(ref predictors) = self.offset_predictors {
            let q_for_offsets = q.permute((0, 2, 1, 3))?.contiguous()?;
            let q_flat = q_for_offsets.reshape((batch_size * seq_len * total_heads, head_dim))?;

            let mut head_offsets = Vec::with_capacity(total_heads);
            for (head_idx, predictor) in predictors.iter().enumerate() {
                let start_idx = head_idx * batch_size * seq_len;
                let q_head = q_flat.narrow(0, start_idx, batch_size * seq_len)?;
                head_offsets.push(predictor.forward(&q_head)?.tanh()?);
            }

            Tensor::stack(&head_offsets, 1)?
                .reshape((batch_size, seq_len, total_heads, num_offsets))?
                .permute((0, 2, 1, 3))? // [Batch, Heads, Seq, NumOffsets]
        } else {
            Tensor::zeros(
                (batch_size, total_heads, seq_len, num_offsets),
                q.dtype(),
                &self.device,
            )?
        };

        // Gaussian bias computation (vectorized with caching)
        let (key_pos, query_pos) = {
            let mut cache = self.position_cache.borrow_mut();
            if let Some(cached) = cache.get(&seq_len) {
                cached.clone()
            } else {
                let seq_pos: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
                let key_pos = Tensor::from_vec(seq_pos.clone(), (1, 1, 1, seq_len), &self.device)?;
                let query_pos = Tensor::from_vec(seq_pos, (1, 1, seq_len, 1), &self.device)?;
                cache.insert(seq_len, (key_pos.clone(), query_pos.clone()));
                (key_pos, query_pos)
            }
        };

        let scale = (seq_len as f32) / 2.0;
        let scale_tensor = Tensor::new(scale, &self.device)?;
        let scale_broadcast = scale_tensor.broadcast_as(offsets.shape())?;
        let scaled_offsets = offsets.broadcast_mul(&scale_broadcast)?;
        let target_pos =
            (query_pos.broadcast_as((batch_size, total_heads, seq_len, num_offsets))?
                - scaled_offsets)?;
        let target_pos = target_pos.unsqueeze(4)?;

        let key_grid =
            key_pos.broadcast_as((batch_size, total_heads, seq_len, num_offsets, seq_len))?;
        let target_grid =
            target_pos.broadcast_as((batch_size, total_heads, seq_len, num_offsets, seq_len))?;

        let diff = (key_grid - target_grid)?;
        let bandwidth = (seq_len as f32 / num_offsets as f32).max(1.0);
        let denom = Tensor::new(2.0f32 * bandwidth * bandwidth, &self.device)?
            .broadcast_as(diff.shape())?;
        let gaussian = diff.sqr()?.broadcast_div(&denom)?.neg()?.exp()?;
        let mask = gaussian.mean(3)?;

        let mask_sum = mask.sum(D::Minus1)?.unsqueeze(3)?;
        let normalized = mask.broadcast_div(&(mask_sum + 1e-6)?)?;
        Ok((normalized + 1e-6)?.log()?)
    }
    /// Apply sparse mask (batched)
    fn apply_sparse_mask_batched(
        &self,
        attn_scores: &Tensor,
        importance: &Tensor,
        sparsity_ratio: &Tensor,
    ) -> Result<Tensor> {
        let sparsity_expanded = sparsity_ratio
            .unsqueeze(1)?
            .broadcast_as(attn_scores.shape())?;
        let clipped = sparsity_expanded.clamp(0.05, 1.0)?;

        let ones = Tensor::ones_like(attn_scores)?;
        let deficit = (ones.clone() - &clipped)?;
        let sharpness = (ones + (deficit * 8.0)?)?;

        let mut logits = attn_scores.broadcast_mul(&sharpness)?;
        let importance_bias = importance.broadcast_as(attn_scores.shape())?;
        logits = (logits + (importance_bias * 0.1)?)?;

        Ok(logits)
    }

    /// Compute sparse attention for a single head with top-K selection
    fn compute_sparsity_signal(
        &mut self,
        batch_size: usize,
        seq_len: usize,
        volatility_signal: Option<&Tensor>,
        routing_weights: Option<&Tensor>,
    ) -> Result<Tensor> {
        let base_signal = if let Some(volatility) = volatility_signal {
            volatility.clamp(0.0, 1.0)?
        } else if let Some(weights) = routing_weights {
            let safe = weights.clamp(1e-6, 1.0)?;
            let log_safe = safe.log()?;
            let entropy_term = safe.broadcast_mul(&log_safe)?;
            let entropy = entropy_term.neg()?.sum(D::Minus1)?.unsqueeze(2)?;
            let max_entropy = (weights.dim(2)? as f32).ln().max(1e-6);
            let norm = Tensor::new(1.0 / max_entropy, &self.device)?;
            let norm = norm.broadcast_as(entropy.shape())?;
            let normalized_entropy = entropy.broadcast_mul(&norm)?;
            normalized_entropy.clamp(0.0, 1.0)?
        } else {
            Tensor::new(0.5f32, &self.device)?.broadcast_as((batch_size, seq_len, 1))?
        };

        let min_ratio = self.moh_config.min_sparse_ratio;
        let max_ratio = self.moh_config.max_sparse_ratio;
        let base_shape = base_signal.shape();
        let min_tensor = Tensor::new(min_ratio, &self.device)?.broadcast_as(base_shape.clone())?;
        let range_tensor =
            Tensor::new(max_ratio - min_ratio, &self.device)?.broadcast_as(base_shape.clone())?;
        let scaled = base_signal.broadcast_mul(&range_tensor)?;
        let ratio = (scaled + min_tensor)?;

        let mean_ratio = ratio.mean_all()?.to_scalar::<f32>().unwrap_or(min_ratio);
        self.last_sparsity_mean = Some(mean_ratio);

        Ok(ratio)
    }

    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mut cache = self.causal_mask_cache.borrow_mut();

        if let Some(cached_mask) = cache.get(&seq_len) {
            return Ok(cached_mask.clone());
        }

        let mut mask_data = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 0.0;
            }
        }
        let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), &self.device)?;
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, Seq, Seq]

        cache.insert(seq_len, mask.clone());
        Ok(mask)
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
        if let Some(ratio) = self.last_sparsity_mean {
            stats.insert("last_sparsity_ratio".to_string(), ratio as f64);
        }
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
    pub fn last_sparsity_ratio(&self) -> Option<f32> {
        self.last_sparsity_mean
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
