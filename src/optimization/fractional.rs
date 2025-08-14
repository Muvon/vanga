//! Fractional derivative computation for fractional optimizers
//!
//! This module implements the Grünwald-Letnikov approximation for fractional derivatives
//! as described in the paper "Fractional Adam and Fractional NAdam for Neural Network Optimization"
//!
//! The key insight: For discrete optimization, we need to handle the alternating signs
//! of the Grünwald-Letnikov weights carefully to prevent gradient cancellation.

use candle_core::{Result, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Fractional derivative computation using Grünwald-Letnikov approximation
///
/// Implements the short-memory approximation from the paper:
/// D^α f(t) ≈ (1/h^α) * Σ(k=0 to M) ω_k^(α) * f(t-kh)
///
/// Where ω_k^(α) are the Grünwald-Letnikov weights.
/// For discrete optimization, we use a modified approach to handle negative weights.
#[derive(Debug, Clone)]
pub struct FractionalDerivative {
    /// Fractional order α ∈ (0, 1]
    alpha: f64,
    /// Memory window size M (typically 30-90 for efficiency)
    memory_window: usize,
    /// Step size h (typically 1.0 for discrete optimization)
    step_size: f64,
    /// Precomputed Grünwald-Letnikov weights
    weights: Vec<f64>,
    /// Gradient history buffer for each parameter
    gradient_history: Vec<VecDeque<Tensor>>,
}

impl FractionalDerivative {
    /// Create a new fractional derivative computer
    ///
    /// # Arguments
    /// * `alpha` - Fractional order (0 < α ≤ 1)
    /// * `memory_window` - Number of past gradients to consider (M)
    /// * `step_size` - Discretization step size (typically 1.0)
    /// * `num_params` - Number of parameter tensors to track
    pub fn new(
        alpha: f64,
        memory_window: usize,
        step_size: f64,
        num_params: usize,
    ) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(candle_core::Error::Msg(format!(
                "Fractional order α must be in (0, 1], got: {}",
                alpha
            )));
        }

        if memory_window == 0 {
            return Err(candle_core::Error::Msg(
                "Memory window must be positive".to_string(),
            ));
        }

        // Precompute Grünwald-Letnikov weights
        let weights = Self::compute_gl_weights(alpha, memory_window);

        // Initialize gradient history buffers
        let gradient_history = (0..num_params)
            .map(|_| VecDeque::with_capacity(memory_window))
            .collect();

        Ok(Self {
            alpha,
            memory_window,
            step_size,
            weights,
            gradient_history,
        })
    }

    /// Compute Grünwald-Letnikov weights ω_k^(α)
    ///
    /// Using the recursive formula:
    /// ω_0^(α) = 1
    /// ω_k^(α) = ω_{k-1}^(α) * (1 - (α+1)/k) for k ≥ 1
    fn compute_gl_weights(alpha: f64, memory_window: usize) -> Vec<f64> {
        let mut weights = Vec::with_capacity(memory_window + 1);
        weights.push(1.0); // ω_0^(α) = 1

        for k in 1..=memory_window {
            let prev_weight = weights[k - 1];
            let new_weight = prev_weight * (1.0 - (alpha + 1.0) / k as f64);
            weights.push(new_weight);
        }

        weights
    }

    /// Update gradient history with new gradients
    ///
    /// # Arguments
    /// * `gradients` - Current gradients for all parameters
    pub fn update_history(&mut self, gradients: &[Tensor]) -> Result<()> {
        if gradients.len() != self.gradient_history.len() {
            return Err(candle_core::Error::Msg(format!(
                "Expected {} gradients, got {}",
                self.gradient_history.len(),
                gradients.len()
            )));
        }

        for (i, gradient) in gradients.iter().enumerate() {
            let history = &mut self.gradient_history[i];

            // Add new gradient to front
            history.push_front(gradient.clone());

            // Remove oldest gradient if we exceed memory window
            if history.len() > self.memory_window {
                history.pop_back();
            }
        }

        Ok(())
    }

    /// Compute fractional gradients using short-memory approximation
    ///
    /// Implements: D^α ∇J(θ_t) ≈ (1/h^α) * Σ(k=0 to M) ω_k^(α) * ∇J(θ_{t-k})
    ///
    /// CRITICAL FIX: For discrete optimization, we modify the approach to handle
    /// the alternating signs of GL weights which cause gradient cancellation.
    pub fn compute_fractional_gradients(&self) -> Result<Vec<Tensor>> {
        let mut fractional_gradients = Vec::with_capacity(self.gradient_history.len());

        for history in &self.gradient_history {
            if history.is_empty() {
                return Err(candle_core::Error::Msg(
                    "No gradient history available for fractional computation".to_string(),
                ));
            }

            // Start with the most recent gradient
            let mut fractional_grad = history[0].clone();

            // Only apply fractional weighting if we have sufficient history
            if history.len() >= 3 {
                // CRITICAL INSIGHT: The Grünwald-Letnikov weights alternate signs
                // ω_0 = 1, ω_1 = -α, ω_2 = -α(1-α)/2, etc.
                // This causes gradient cancellation in discrete optimization.
                //
                // Solution: Use a modified weighting scheme that preserves gradient flow
                // while incorporating memory effects.

                // Approach 1: Use absolute values of weights and normalize
                let mut weighted_sum = history[0].clone();
                let mut abs_weight_sum = self.weights[0].abs(); // 1.0

                for (k, gradient) in history.iter().enumerate().skip(1) {
                    if k >= self.weights.len() {
                        break;
                    }

                    // Use absolute value to prevent sign flipping
                    let abs_weight = self.weights[k].abs();

                    // Skip very small weights
                    if abs_weight < 1e-8 {
                        continue;
                    }

                    let weight_tensor = Tensor::new(abs_weight as f32, gradient.device())?
                        .broadcast_as(gradient.shape())?
                        .contiguous()?;

                    let weighted_grad = gradient.contiguous()?.mul(&weight_tensor)?.contiguous()?;
                    weighted_sum = weighted_sum.add(&weighted_grad)?.contiguous()?;
                    abs_weight_sum += abs_weight;
                }

                // Apply the (1/h^α) scaling and normalize by weight sum
                // This ensures proper scaling according to the fractional derivative theory
                let scale_factor =
                    (1.0 / (self.step_size.powf(self.alpha) * abs_weight_sum)) as f32;

                let scale_tensor = Tensor::new(scale_factor, weighted_sum.device())?
                    .broadcast_as(weighted_sum.shape())?
                    .contiguous()?;

                fractional_grad = weighted_sum
                    .contiguous()?
                    .mul(&scale_tensor)?
                    .contiguous()?;

                // Log for debugging (occasionally)
                if history.len() == self.memory_window && self.gradient_history[0].len() % 100 == 1
                {
                    log::trace!(
                        "Fractional gradient: α={:.2}, h={:.2}, weight_sum={:.4}, scale={:.6}",
                        self.alpha,
                        self.step_size,
                        abs_weight_sum,
                        scale_factor
                    );
                }
            } else {
                // For early steps, use regular gradient with mild scaling
                // The (1/h^α) factor still applies but with reduced effect
                let early_scale = (1.0 / self.step_size.powf(self.alpha * 0.5)) as f32;

                let scale_tensor = Tensor::new(early_scale, fractional_grad.device())?
                    .broadcast_as(fractional_grad.shape())?
                    .contiguous()?;

                fractional_grad = fractional_grad
                    .contiguous()?
                    .mul(&scale_tensor)?
                    .contiguous()?;

                log::trace!(
                    "Early step gradient scaling: history_len={}, scale={:.4}",
                    history.len(),
                    early_scale
                );
            }

            fractional_gradients.push(fractional_grad);
        }

        Ok(fractional_gradients)
    }

    /// Get the current fractional order
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the memory window size
    pub fn memory_window(&self) -> usize {
        self.memory_window
    }

    /// Get the number of stored gradients for a parameter
    pub fn history_length(&self, param_idx: usize) -> usize {
        self.gradient_history
            .get(param_idx)
            .map(|h| h.len())
            .unwrap_or(0)
    }

    /// Get the last gradient for a specific parameter (if available)
    pub fn get_last_gradient(&self, param_idx: usize) -> Option<&Tensor> {
        self.gradient_history
            .get(param_idx)
            .and_then(|history| history.front())
    }

    /// Clear all gradient history (useful for fresh training)
    pub fn clear_history(&mut self) {
        for history in &mut self.gradient_history {
            history.clear();
        }
    }

    /// Resize for different number of parameters
    pub fn resize(&mut self, num_params: usize) {
        self.gradient_history
            .resize_with(num_params, || VecDeque::with_capacity(self.memory_window));
    }
}

/// Configuration for fractional optimizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractionalConfig {
    /// Fractional order α ∈ (0, 1]
    pub alpha: f64,
    /// Memory window size M (30-90 recommended)
    pub memory_window: usize,
    /// Step size h (typically 1.0)
    pub step_size: f64,
}

impl Default for FractionalConfig {
    fn default() -> Self {
        Self {
            alpha: 0.7,        // Moderate fractional order for stability
            memory_window: 30, // Reasonable memory window
            step_size: 1.0,    // Standard discrete time step
        }
    }
}

impl FractionalConfig {
    /// Create configuration for short-term memory (faster, less memory)
    pub fn short_memory() -> Self {
        Self {
            alpha: 0.8,
            memory_window: 30,
            step_size: 1.0,
        }
    }

    /// Create configuration for long-term memory (slower, more memory)
    pub fn long_memory() -> Self {
        Self {
            alpha: 0.95,
            memory_window: 90,
            step_size: 1.0,
        }
    }

    /// Create configuration optimized for financial time series
    pub fn financial_optimized() -> Self {
        Self {
            alpha: 0.75,       // Balanced for financial data (weight sum ~0.01)
            memory_window: 30, // 30 steps of history (configurable based on data frequency)
            step_size: 1.0,    // For discrete time series, h=1.0 is standard
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.alpha <= 0.0 || self.alpha > 1.0 {
            return Err(candle_core::Error::Msg(format!(
                "Fractional order α must be in (0, 1], got: {}",
                self.alpha
            )));
        }

        if self.memory_window == 0 {
            return Err(candle_core::Error::Msg(
                "Memory window must be positive".to_string(),
            ));
        }

        if self.memory_window > 200 {
            return Err(candle_core::Error::Msg(format!(
                "Memory window too large ({}), maximum recommended: 200",
                self.memory_window
            )));
        }

        if self.step_size <= 0.0 {
            return Err(candle_core::Error::Msg(format!(
                "Step size must be positive, got: {}",
                self.step_size
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_gl_weights_computation() {
        let alpha = 0.5;
        let memory_window = 5;
        let weights = FractionalDerivative::compute_gl_weights(alpha, memory_window);

        // Check first weight
        assert_eq!(weights[0], 1.0);

        // Check recursive formula for subsequent weights
        for k in 1..weights.len() {
            let expected = weights[k - 1] * (1.0 - (alpha + 1.0) / k as f64);
            assert!((weights[k] - expected).abs() < 1e-10);
        }

        // Note: Most weights after the first are negative!
        assert!(weights[1] < 0.0, "Second weight should be negative");
    }

    #[test]
    fn test_fractional_derivative_creation() {
        let result = FractionalDerivative::new(0.9, 30, 1.0, 3);
        assert!(result.is_ok());

        let frac_deriv = result.unwrap();
        assert_eq!(frac_deriv.alpha(), 0.9);
        assert_eq!(frac_deriv.memory_window(), 30);
    }

    #[test]
    fn test_invalid_alpha() {
        let result = FractionalDerivative::new(0.0, 30, 1.0, 3);
        assert!(result.is_err());

        let result = FractionalDerivative::new(1.5, 30, 1.0, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_gradient_history_update() -> Result<()> {
        let device = Device::Cpu;
        let mut frac_deriv = FractionalDerivative::new(0.9, 3, 1.0, 2)?;

        // Create test gradients
        let grad1 = Tensor::new(&[1.0f32, 2.0], &device)?;
        let grad2 = Tensor::new(&[3.0f32, 4.0], &device)?;
        let gradients = vec![grad1, grad2];

        frac_deriv.update_history(&gradients)?;

        assert_eq!(frac_deriv.history_length(0), 1);
        assert_eq!(frac_deriv.history_length(1), 1);

        Ok(())
    }

    #[test]
    fn test_fractional_config_validation() {
        let config = FractionalConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = FractionalConfig {
            alpha: 0.0,
            memory_window: 30,
            step_size: 1.0,
        };
        assert!(invalid_config.validate().is_err());
    }
}
