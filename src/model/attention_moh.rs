// Mixture-of-Head Attention (MoH) implementation for VANGA LSTM
// Based on "Mixture-of-Head Attention for Multi-Modal Learning" paper
use crate::config::model::{AttentionConfig, MoHConfig};
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};
use candle_nn::{linear, ops, Linear, Module, VarBuilder};
use std::collections::HashMap;

/// Individual attention head for MoH mechanism
#[derive(Debug)]
struct AttentionHead {
    head_id: usize,
    head_dim: usize,
    query_projection: Linear,
    key_projection: Linear,
    value_projection: Linear,
    output_projection: Linear,
}

impl AttentionHead {
    fn new(
        head_id: usize,
        input_dim: usize,
        head_dim: usize,
        vs: VarBuilder,
        _device: &Device,
    ) -> Result<Self> {
        let query_projection = linear(
            input_dim,
            head_dim,
            vs.pp(format!("head_{}_query", head_id)),
        )
        .map_err(|e| {
            VangaError::ModelError(format!("Head {} query projection failed: {}", head_id, e))
        })?;

        let key_projection = linear(input_dim, head_dim, vs.pp(format!("head_{}_key", head_id)))
            .map_err(|e| {
                VangaError::ModelError(format!("Head {} key projection failed: {}", head_id, e))
            })?;

        let value_projection = linear(
            input_dim,
            head_dim,
            vs.pp(format!("head_{}_value", head_id)),
        )
        .map_err(|e| {
            VangaError::ModelError(format!("Head {} value projection failed: {}", head_id, e))
        })?;

        let output_projection = linear(
            head_dim,
            input_dim,
            vs.pp(format!("head_{}_output", head_id)),
        )
        .map_err(|e| {
            VangaError::ModelError(format!("Head {} output projection failed: {}", head_id, e))
        })?;

        log::debug!(
            "✅ Created AttentionHead {}: input_dim={}, head_dim={}",
            head_id,
            input_dim,
            head_dim
        );

        Ok(Self {
            head_id,
            head_dim,
            query_projection,
            key_projection,
            value_projection,
            output_projection,
        })
    }

    /// Forward pass for individual attention head - OPTIMIZED for stock market sequences
    fn forward(&self, input: &Tensor, training: bool, dropout_rate: f64) -> Result<Tensor> {
        let (batch_size, seq_len, _) = input.dims3()?;

        // Generate Q, K, V for this head
        let queries = self.query_projection.forward(input)?;
        let keys = self.key_projection.forward(input)?;
        let values = self.value_projection.forward(input)?;

        // Compute scaled dot-product attention with numerical stability
        let scale = (self.head_dim as f64).sqrt() as f32;
        let scale_tensor = Tensor::new(scale, input.device())?;
        let scaled_queries = queries.broadcast_div(&scale_tensor)?.contiguous()?;

        // Attention scores: Q * K^T with memory-efficient computation
        let keys_transposed = keys.transpose(1, 2)?.contiguous()?;
        let mut attention_scores = scaled_queries.matmul(&keys_transposed)?.contiguous()?;

        // Apply causal mask for time series (critical for stock market temporal data)
        attention_scores = self.apply_causal_mask(&attention_scores, seq_len)?;

        // Apply softmax with numerical stability (use softmax_last_dim for efficiency)
        let mut attention_weights = ops::softmax_last_dim(&attention_scores)?.contiguous()?;

        // Apply dropout to attention weights if training
        if training && dropout_rate > 0.0 {
            attention_weights = ops::dropout(&attention_weights, dropout_rate as f32)?;
        }

        // Apply attention to values
        let attended_values = attention_weights.matmul(&values)?.contiguous()?;

        // Apply output projection
        let output = self.output_projection.forward(&attended_values)?;

        log::trace!(
            "Head {} forward: batch_size={}, seq_len={}, output_shape={:?}",
            self.head_id,
            batch_size,
            seq_len,
            output.shape()
        );

        Ok(output)
    }

    /// Apply causal mask to prevent attention to future positions
    fn apply_causal_mask(&self, attention_scores: &Tensor, seq_len: usize) -> Result<Tensor> {
        let mut mask_data = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 0.0;
            }
        }

        let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), attention_scores.device())?;
        let mask = mask.unsqueeze(0)?; // Add batch dimension

        attention_scores
            .broadcast_add(&mask)?
            .contiguous()
            .map_err(|e| VangaError::ModelError(format!("Causal mask application failed: {}", e)))
    }
}

/// Mixture-of-Head Attention mechanism with dynamic head routing
pub struct MixtureOfHeadAttention {
    config: AttentionConfig,
    moh_config: MoHConfig,
    input_dim: usize,
    head_dim: usize,

    // Individual attention heads
    heads: Vec<AttentionHead>,

    // Two-stage routing components (Equations 5-6 from paper)
    shared_router: Linear,    // W_s ∈ ℝ^{hs×d_in}
    routed_router: Linear,    // W_r ∈ ℝ^{(h-hs)×d_in}
    head_type_router: Linear, // W_h ∈ ℝ^{2×d_in}

    // Routing history for load balance loss calculation
    routing_history: Vec<Vec<f32>>,
    history_index: usize,    // For circular buffer implementation
    max_history_size: usize, // Maximum history size to prevent unbounded growth
    step_count: usize,       // Track training steps for periodic cleanup

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

        // Create individual attention heads
        let mut heads = Vec::new();
        for i in 0..moh_config.total_heads {
            let head = AttentionHead::new(
                i as usize,
                input_dim,
                head_dim,
                vs.pp("heads".to_string()),
                &device,
            )?;
            heads.push(head);
        }

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
            "✅ MixtureOfHeadAttention initialized: {} total heads ({} shared + {} routed, top_k={}), head_dim={}, efficiency={:.1}%",
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
            heads,
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
    pub fn forward(&mut self, input: &Tensor, training: bool) -> Result<(Tensor, Tensor)> {
        // Increment step counter for memory management
        if training {
            self.step_count += 1;
        }

        // Validate input dimensions
        self.validate_input_dimensions(input)?;

        let (batch_size, seq_len, _) = input.dims3()?;

        // Compute routing scores for each token in the sequence
        let mut batch_outputs = Vec::new();
        let mut batch_routing_scores = Vec::new();
        let mut sequence_load_balance_loss = 0.0;

        for t in 0..seq_len {
            // Extract token at position t: [batch_size, 1, input_dim]
            let token = input.narrow(1, t, 1)?;
            let token_squeezed = token.squeeze(1)?; // [batch_size, input_dim]

            // Compute routing scores for this token (two-stage routing)
            let (routing_scores, load_balance_loss) =
                self.compute_routing_scores(&token_squeezed, training)?;
            sequence_load_balance_loss += load_balance_loss;

            // Apply heads with routing weights
            let token_output = self.apply_routed_heads(&token, &routing_scores, training)?;

            batch_outputs.push(token_output);
            batch_routing_scores.push(routing_scores);
        }

        // Concatenate outputs along sequence dimension
        let output = Tensor::cat(&batch_outputs, 1)?.contiguous()?;

        // Create routing scores tensor for interpretability
        let routing_tensor = self.create_routing_tensor(&batch_routing_scores)?;

        // Store routing history for load balance loss (only during training)
        if training {
            // Average routing scores across batch and sequence for history
            let avg_routing_scores = self.average_routing_scores(&batch_routing_scores)?;

            // MEMORY FIX: Use circular buffer with proper memory management
            if self.routing_history.len() < self.max_history_size {
                self.routing_history.push(avg_routing_scores);
            } else {
                // Circular buffer: overwrite oldest entry and clear old memory
                // Clone the new scores to ensure we don't hold references
                let new_scores = avg_routing_scores;

                // Clear the old entry to ensure memory is freed
                self.routing_history[self.history_index].clear();
                self.routing_history[self.history_index] = new_scores;
                self.history_index = (self.history_index + 1) % self.max_history_size;
            }

            // Periodically compact memory to reduce fragmentation
            if self.step_count % 1000 == 0 {
                self.routing_history.shrink_to_fit();
            }
        }

        log::debug!(
            "MoH forward: batch_size={}, seq_len={}, active_heads={}/{}, load_balance_loss={:.6}",
            batch_size,
            seq_len,
            self.moh_config.active_heads(),
            self.moh_config.total_heads,
            sequence_load_balance_loss
        );

        Ok((output, routing_tensor))
    }

    /// Compute routing scores using two-stage routing (Equations 5-6 from paper)
    /// ENHANCED: Added gradient stopping, market-aware temperature, and performance optimizations
    fn compute_routing_scores(&self, token: &Tensor, training: bool) -> Result<(Vec<f32>, f32)> {
        let batch_size = token.dim(0)?;

        // Stage 1: Head type routing (α1, α2) - Equation 6
        let head_type_logits = self.head_type_router.forward(token)?;

        // CRITICAL FIX: Stop gradient for routing decisions during training (as per paper)
        // This prevents the routing mechanism from affecting the gradient flow
        let head_type_probs = if training {
            // Detach from computation graph to prevent gradient flow through routing
            ops::softmax(&head_type_logits.detach(), 1)?
        } else {
            ops::softmax(&head_type_logits, 1)?
        };

        // Extract α1 and α2 (average across batch for simplicity)
        let head_type_vec = head_type_probs.to_vec2::<f32>()?;
        let alpha1 = head_type_vec.iter().map(|row| row[0]).sum::<f32>() / batch_size as f32;
        let alpha2 = head_type_vec.iter().map(|row| row[1]).sum::<f32>() / batch_size as f32;

        // Stage 2: Individual head routing
        let shared_logits = self.shared_router.forward(token)?;

        // OPTIMIZATION: Detach shared routing during training
        let shared_probs = if training {
            ops::softmax(&shared_logits.detach(), 1)?
        } else {
            ops::softmax(&shared_logits, 1)?
        };

        // Calculate adaptive temperature (used for logging even if no routed heads)
        let adaptive_temperature = self.calculate_adaptive_temperature(token)?;

        // Handle edge case where there are no routed heads
        let num_routed_heads =
            (self.moh_config.total_heads - self.moh_config.shared_heads) as usize;

        let routed_scores = if num_routed_heads > 0 {
            let routed_logits = self.routed_router.forward(token)?;

            // Apply temperature scaling to routed probabilities
            let temperature_tensor = Tensor::new(adaptive_temperature, &self.device)?;
            let routed_probs_scaled = routed_logits
                .broadcast_div(&temperature_tensor)?
                .contiguous()?;

            // OPTIMIZATION: Detach routed probabilities during training
            let routed_probs_final = if training {
                ops::softmax(&routed_probs_scaled.detach(), 1)?
            } else {
                ops::softmax(&routed_probs_scaled, 1)?
            };

            let routed_vec = routed_probs_final.to_vec2::<f32>()?;
            let routed_scores: Vec<f32> = (0..num_routed_heads)
                .map(|i| routed_vec.iter().map(|row| row[i]).sum::<f32>() / batch_size as f32)
                .collect();

            routed_scores
        } else {
            // No routed heads, skip routed computation entirely
            Vec::new()
        };

        // Convert shared probabilities to scores
        let shared_vec = shared_probs.to_vec2::<f32>()?;
        let shared_scores: Vec<f32> = (0..self.moh_config.shared_heads as usize)
            .map(|i| shared_vec.iter().map(|row| row[i]).sum::<f32>() / batch_size as f32)
            .collect();

        // OPTIMIZATION: Use optimized top-K selection
        let top_k_routed =
            self.select_top_k_optimized(&routed_scores, self.moh_config.top_k as usize);

        // Combine scores according to Equation 5
        let mut final_scores = vec![0.0; self.moh_config.total_heads as usize];

        // Shared heads: α1 * softmax(W_s * x_t)_i
        for i in 0..self.moh_config.shared_heads as usize {
            final_scores[i] = alpha1 * shared_scores[i];
        }

        // Routed heads: α2 * softmax(W_r * x_t)_i if in top-K, else 0
        for (idx, &score) in top_k_routed.iter().enumerate() {
            if score > 0.0 {
                final_scores[self.moh_config.shared_heads as usize + idx] = alpha2 * score;
            }
        }

        // Calculate load balance loss for this token (Equation 7)
        let load_balance_loss = if training {
            self.calculate_token_load_balance_loss(&routed_scores, &top_k_routed)?
        } else {
            0.0
        };

        if self.moh_config.log_routing_decisions {
            log::debug!(
                "Routing: α1={:.3}, α2={:.3}, temp={:.2}, active_heads={}, load_loss={:.6}",
                alpha1,
                alpha2,
                adaptive_temperature,
                final_scores.iter().filter(|&&x| x > 0.0).count(),
                load_balance_loss
            );
        }

        Ok((final_scores, load_balance_loss))
    }

    /// Calculate adaptive temperature based on input variance (proxy for market volatility)
    fn calculate_adaptive_temperature(&self, token: &Tensor) -> Result<f32> {
        // Handle edge case where token might be empty or have single dimension
        let token_shape = token.shape();
        if token_shape.dims().is_empty() || token_shape.dims().contains(&0) {
            // Return base temperature for empty tensors
            return Ok(self.moh_config.routing_temperature as f32);
        }

        // Calculate variance as proxy for volatility
        // token shape is [batch_size, features]
        // Calculate mean across features dimension (dim=1)
        let mean = token.mean(1)?.contiguous()?;

        // Properly broadcast mean to match token shape
        // mean shape is [batch_size], need to expand to [batch_size, features]
        let mean_expanded = mean.unsqueeze(1)?.broadcast_as(token_shape)?.contiguous()?;

        // Now subtract with matching shapes
        let diff = token.sub(&mean_expanded)?.contiguous()?;
        let variance = diff.sqr()?.mean_all()?;
        let std_dev = variance.to_scalar::<f32>()?.sqrt();

        // Map std_dev to temperature (higher volatility = higher temperature)
        // Base temperature + volatility adjustment
        let base_temp = self.moh_config.routing_temperature as f32;
        let volatility_adjustment = (std_dev * 0.5).min(1.0); // Cap adjustment at 1.0
        let adaptive_temp = base_temp + volatility_adjustment;

        // Clamp to reasonable range [0.5, 2.5]
        Ok(adaptive_temp.clamp(0.5, 2.5))
    }

    /// Optimized top-K selection using partial sort (more efficient for large arrays)
    fn select_top_k_optimized(&self, scores: &[f32], k: usize) -> Vec<f32> {
        if k >= scores.len() {
            return scores.to_vec();
        }

        // Use partial sort for O(n + k log k) complexity instead of O(n log n)
        let mut indexed_scores: Vec<(usize, f32)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();

        // Only partially sort to get top-k elements
        if k > 0 && k < scores.len() {
            indexed_scores.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let mut result = vec![0.0; scores.len()];
        for &(idx, score) in &indexed_scores[..k] {
            result[idx] = score;
        }

        result
    }

    /// Apply routed heads with computed routing scores - OPTIMIZED for sparse computation
    fn apply_routed_heads(
        &self,
        token: &Tensor,
        routing_scores: &[f32],
        training: bool,
    ) -> Result<Tensor> {
        // OPTIMIZATION: Count active heads first to avoid unnecessary computation
        let active_heads: Vec<(usize, f32)> = routing_scores
            .iter()
            .enumerate()
            .filter(|(_, &w)| w > 1e-6) // Skip near-zero weights
            .map(|(i, &w)| (i, w))
            .collect();

        if active_heads.is_empty() {
            // Early return with zeros if no heads are active
            let (batch_size, seq_len, input_dim) = token.dims3()?;
            return Tensor::zeros(
                (batch_size, seq_len, input_dim),
                token.dtype(),
                &self.device,
            )
            .map_err(|e| VangaError::ModelError(format!("Zero tensor creation failed: {}", e)));
        }

        // OPTIMIZATION: Pre-allocate result tensor for efficiency
        let mut weighted_outputs = Vec::with_capacity(active_heads.len());

        for (head_idx, weight) in active_heads {
            // Apply individual head
            let head_output =
                self.heads[head_idx].forward(token, training, self.config.dropout_rate)?;

            // Weight the output efficiently
            let weight_tensor = Tensor::new(weight, &self.device)?;
            let weighted_output = head_output.broadcast_mul(&weight_tensor)?.contiguous()?;
            weighted_outputs.push(weighted_output);
        }

        // OPTIMIZATION: Use efficient summation (tree reduction for many heads)
        if weighted_outputs.len() == 1 {
            Ok(weighted_outputs.into_iter().next().unwrap())
        } else {
            // Tree reduction for better numerical stability and performance
            let mut result = weighted_outputs[0].clone();
            for output in weighted_outputs.iter().skip(1) {
                result = (result + output)?.contiguous()?;
            }
            Ok(result)
        }
    }

    /// Calculate load balance loss for a single token (part of Equation 7)
    fn calculate_token_load_balance_loss(
        &self,
        routed_probs: &[f32],
        top_k_selection: &[f32],
    ) -> Result<f32> {
        let mut loss = 0.0;

        for i in 0..routed_probs.len() {
            let f_i = if top_k_selection[i] > 0.0 { 1.0 } else { 0.0 }; // Selection indicator
            let p_i = routed_probs[i]; // Routing probability
            loss += f_i * p_i;
        }

        Ok(loss)
    }

    /// Calculate total load balance loss from routing history (Equation 7)
    pub fn calculate_load_balance_loss(&self) -> Result<Tensor> {
        if self.routing_history.is_empty() {
            return Tensor::new(0.0f32, &self.device).map_err(|e| {
                VangaError::ModelError(format!("Load balance loss tensor creation failed: {}", e))
            });
        }

        let t = self.routing_history.len() as f32;
        let mut total_loss = 0.0;

        // Only consider routed heads (not shared heads)
        let routed_start = self.moh_config.shared_heads as usize;
        let routed_end = self.moh_config.total_heads as usize;

        for i in routed_start..routed_end {
            // f_i: frequency of head i being selected
            let f_i = self
                .routing_history
                .iter()
                .map(|scores| if scores[i] > 0.0 { 1.0 } else { 0.0 })
                .sum::<f32>()
                / t;

            // P_i: average routing probability for head i
            let p_i = self
                .routing_history
                .iter()
                .map(|scores| scores[i])
                .sum::<f32>()
                / t;

            total_loss += f_i * p_i;
        }

        Tensor::new(total_loss, &self.device).map_err(|e| {
            VangaError::ModelError(format!("Load balance loss tensor creation failed: {}", e))
        })
    }

    /// Create routing tensor for interpretability
    fn create_routing_tensor(&self, batch_routing_scores: &[Vec<f32>]) -> Result<Tensor> {
        let seq_len = batch_routing_scores.len();
        let num_heads = self.moh_config.total_heads as usize;

        let mut routing_data = Vec::new();
        for scores in batch_routing_scores {
            routing_data.extend_from_slice(scores);
        }

        Tensor::from_vec(routing_data, (1, seq_len, num_heads), &self.device)
            .map_err(|e| VangaError::ModelError(format!("Routing tensor creation failed: {}", e)))
    }

    /// Average routing scores across batch and sequence for history
    fn average_routing_scores(&self, batch_routing_scores: &[Vec<f32>]) -> Result<Vec<f32>> {
        let num_heads = self.moh_config.total_heads as usize;
        let mut avg_scores = vec![0.0; num_heads];

        for scores in batch_routing_scores {
            for (i, &score) in scores.iter().enumerate() {
                avg_scores[i] += score;
            }
        }

        let count = batch_routing_scores.len() as f32;
        for score in &mut avg_scores {
            *score /= count;
        }

        Ok(avg_scores)
    }

    /// Get routing statistics for analysis
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
        stats.insert("input_dim".to_string(), self.input_dim as f64);
        stats.insert("head_dim".to_string(), self.head_dim as f64);

        if !self.routing_history.is_empty() {
            let recent_scores = &self.routing_history[self.routing_history.len() - 1];
            let active_heads = recent_scores.iter().filter(|&&x| x > 0.0).count();
            stats.insert("recent_active_heads".to_string(), active_heads as f64);

            // Calculate routing entropy for diversity analysis
            let entropy = self.calculate_routing_entropy(recent_scores);
            stats.insert("routing_entropy".to_string(), entropy);
        }

        stats
    }

    /// Clear routing history to free memory
    pub fn clear_routing_history(&mut self) {
        self.routing_history.clear();
        self.history_index = 0;
        self.step_count = 0;
    }

    /// Compact memory by reducing allocated but unused capacity
    pub fn compact_memory(&mut self) {
        self.routing_history.shrink_to_fit();
        // Also shrink individual score vectors
        for scores in &mut self.routing_history {
            scores.shrink_to_fit();
        }
    }

    /// Get memory usage estimate (number of float values stored)
    pub fn memory_usage(&self) -> usize {
        self.routing_history.len() * self.moh_config.total_heads as usize
    }

    /// Calculate routing entropy to measure head usage diversity
    fn calculate_routing_entropy(&self, scores: &[f32]) -> f64 {
        let total: f32 = scores.iter().sum();
        if total == 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &score in scores {
            if score > 0.0 {
                let prob = score as f64 / total as f64;
                entropy -= prob * prob.ln();
            }
        }

        entropy
    }

    /// Validate input dimensions match expected values
    pub fn validate_input_dimensions(&self, input: &Tensor) -> Result<()> {
        let input_dims = input.dims();
        if input_dims.len() != 3 {
            return Err(VangaError::ModelError(format!(
                "Expected 3D input tensor [batch, seq, features], got shape: {:?}",
                input_dims
            )));
        }

        let actual_input_dim = input_dims[2];
        if actual_input_dim != self.input_dim {
            return Err(VangaError::ModelError(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.input_dim, actual_input_dim
            )));
        }

        Ok(())
    }

    /// Get model architecture information
    pub fn get_architecture_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();

        info.insert(
            "model_type".to_string(),
            "MixtureOfHeadAttention".to_string(),
        );
        info.insert("input_dim".to_string(), self.input_dim.to_string());
        info.insert("head_dim".to_string(), self.head_dim.to_string());
        info.insert(
            "total_heads".to_string(),
            self.moh_config.total_heads.to_string(),
        );
        info.insert(
            "shared_heads".to_string(),
            self.moh_config.shared_heads.to_string(),
        );
        info.insert(
            "routed_heads".to_string(),
            (self.moh_config.total_heads - self.moh_config.shared_heads).to_string(),
        );
        info.insert("top_k".to_string(), self.moh_config.top_k.to_string());
        info.insert(
            "efficiency".to_string(),
            format!("{:.1}%", self.moh_config.efficiency_ratio() * 100.0),
        );

        info
    }

    /// Get configuration (for wrapper access)
    pub fn get_config(&self) -> &AttentionConfig {
        &self.config
    }
}

// Note: MixtureOfHeadAttention does NOT implement AttentionModule directly
// It should only be used through MoHAttentionWrapper which handles the mutable reference requirement
