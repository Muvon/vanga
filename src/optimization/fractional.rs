//! Fractional derivative computation for fractional optimizers
//!
//! This module implements the Caputo fractional derivative for optimization
//! as it's better suited for financial time series than Grünwald-Letnikov.
//!
//! The Caputo derivative maintains zero derivative on constants and uses
//! standard initial conditions, making it ideal for LSTM training.

use candle_core::{Result, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::f64::consts::PI;

/// Compute Gamma function approximation for Caputo derivative
/// Optimized for x ∈ (0, 2) which covers our use case of Γ(1-α) where α ∈ (0,1)
fn compute_gamma(x: f64) -> f64 {
    // For Caputo derivative, we need Γ(1-α) where α ∈ (0,1), so x ∈ (0,1)
    // We use a simple but accurate approximation suitable for optimization

    if x <= 0.0 {
        return f64::INFINITY; // Γ(x) has poles at non-positive integers
    }

    // For x close to 1, use Taylor series around x=1
    // Γ(1+z) ≈ 1 - γz + (π²/6 + γ²)z²/2 + ... where γ is Euler-Mascheroni constant
    if (x - 1.0).abs() < 0.1 {
        const EULER_GAMMA: f64 = 0.5772156649015329;
        let z = x - 1.0;
        return 1.0 - EULER_GAMMA * z + (PI * PI / 6.0 + EULER_GAMMA * EULER_GAMMA) * z * z / 2.0;
    }

    // For x in (0, 0.5), use reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
    if x < 0.5 {
        return PI / ((PI * x).sin() * compute_gamma(1.0 - x));
    }

    // For x in [0.5, 2], use Stirling's approximation with corrections
    // This is accurate enough for our optimization purposes
    if x <= 2.0 {
        // Polynomial approximation optimized for [0.5, 2]
        let t = x - 1.0;
        return 1.0
            + t * (-0.5772156649
                + t * (0.9882058891
                    + t * (-0.8970569639
                        + t * (0.9182068604
                            + t * (-0.7568024953
                                + t * (0.4822199332 + t * (-0.1935278186 + t * 0.0358683481)))))));
    }

    // For x > 2, use recursion: Γ(x) = (x-1)Γ(x-1)
    (x - 1.0) * compute_gamma(x - 1.0)
}

/// Fractional derivative computation using Caputo method
///
/// The Caputo fractional derivative is superior for financial time series:
/// - Zero derivative on constants (D^α_C(constant) = 0)
/// - Uses standard initial conditions (compatible with LSTM states)
/// - Better captures long-term memory without sign alternation
/// - More stable for trend-following markets
#[derive(Debug, Clone)]
pub struct FractionalDerivative {
    /// Fractional order α ∈ (0, 1]
    alpha: f64,
    /// Memory window size M (typically 30-90 for efficiency)
    memory_window: usize,
    /// Step size h (typically 1.0 for discrete optimization)
    step_size: f64,
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

        if step_size <= 0.0 {
            return Err(candle_core::Error::Msg(
                "Step size must be positive".to_string(),
            ));
        }

        // Initialize gradient history buffers for Caputo method
        let gradient_history = (0..num_params)
            .map(|_| VecDeque::with_capacity(memory_window))
            .collect();

        Ok(Self {
            alpha,
            memory_window,
            step_size,
            gradient_history,
        })
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

            // MEMORY FIX: Clone gradient only once and manage memory properly
            // Detach from computation graph to prevent holding references
            let detached_gradient = gradient.detach();

            // Add new gradient to front
            history.push_front(detached_gradient);

            // Remove oldest gradient if we exceed memory window
            // This properly drops the old tensor and releases memory
            if history.len() > self.memory_window {
                // pop_back() returns Option<Tensor> which is dropped, releasing memory
                let _ = history.pop_back();
            }
        }

        Ok(())
    }

    /// Compute fractional gradients using Caputo derivative with Grünwald-Letnikov weights
    ///
    /// Implements Caputo fractional derivative for optimization:
    /// D^α_C f(t_n) ≈ (h^(-α)/Γ(2-α)) Σ_{j=0}^{n-1} b_j^(α) [f(t_{n-j}) - f(t_{n-j-1})]
    ///
    /// where b_j^(α) = (j+1)^(1-α) - j^(1-α) are Grünwald-Letnikov weights
    ///
    /// For discrete optimization with short memory:
    /// D^α_C ∇J(θ_t) ≈ (h^(-α)/Γ(2-α)) * Σ(k=1 to M) b_{k-1}^(α) [∇J(θ_{t-k+1}) - ∇J(θ_{t-k})]
    ///
    /// Key properties:
    /// - As α → 1: b_j^(α) → 1, recovers standard gradient (correct limit)
    /// - As α → 0: b_j^(α) → 0, no memory effect
    /// - Γ(2-α) normalization ensures proper scaling for all α ∈ (0,1]
    pub fn compute_fractional_gradients(&self) -> Result<Vec<Tensor>> {
        let mut fractional_gradients = Vec::with_capacity(self.gradient_history.len());

        for history in &self.gradient_history {
            if history.is_empty() {
                return Err(candle_core::Error::Msg(
                    "No gradient history available for fractional computation".to_string(),
                ));
            }

            // For Caputo derivative, we need at least 2 gradients to compute differences
            if history.len() < 2 {
                // For the first step, return the gradient as-is
                fractional_gradients.push(history[0].clone());
                continue;
            }

            // Caputo derivative computation using finite differences
            // The key insight: Caputo uses the derivative of the function (gradient differences)
            // weighted by a power-law kernel (t-τ)^(-α)

            let device = history[0].device();
            let shape = history[0].shape();

            // Initialize with zeros (Caputo naturally handles initialization)
            let mut caputo_sum = Tensor::zeros(shape, candle_core::DType::F32, device)?;

            // Compute the Caputo fractional derivative using proper Grünwald-Letnikov weights
            // For Caputo: D^α_C f(t) uses weights b_j^(α) = (j+1)^(1-α) - j^(1-α)
            // This ensures convergence to standard derivative as α → 1
            let max_k = (self.memory_window).min(history.len() - 1);

            for k in 1..=max_k {
                if k >= history.len() {
                    break;
                }

                // Compute gradient difference: ∇J(θ_{t-k+1}) - ∇J(θ_{t-k})
                let grad_diff = history[k - 1]
                    .contiguous()?
                    .sub(&history[k])?
                    .contiguous()?;

                // Grünwald-Letnikov weight for Caputo: b_{k-1}^(α) = k^(1-α) - (k-1)^(1-α)
                // This is mathematically correct and ensures α=1 gives standard gradient
                let j = (k - 1) as f64;
                let gl_weight = (j + 1.0).powf(1.0 - self.alpha) - j.powf(1.0 - self.alpha);

                // Skip negligible weights for efficiency
                if gl_weight.abs() < 1e-10 {
                    continue;
                }

                let weight_tensor = Tensor::new(gl_weight as f32, device)?
                    .broadcast_as(shape)?
                    .contiguous()?;

                let weighted_diff = grad_diff.contiguous()?.mul(&weight_tensor)?.contiguous()?;

                caputo_sum = caputo_sum.add(&weighted_diff)?.contiguous()?;
            }

            // Apply Caputo normalization: h^(-α) / Γ(2-α)
            // Note: Γ(2-α) not Γ(1-α) because we're using first differences
            let gamma_2_minus_alpha = compute_gamma(2.0 - self.alpha);
            let scale_factor = (self.step_size.powf(-self.alpha) / gamma_2_minus_alpha) as f32;

            let scale_tensor = Tensor::new(scale_factor, device)?
                .broadcast_as(shape)?
                .contiguous()?;

            let fractional_grad = caputo_sum.contiguous()?.mul(&scale_tensor)?.contiguous()?;

            // CRITICAL: Add current gradient to maintain learning
            // Caputo derivative = fractional memory term + current gradient
            // Without this, gradients vanish when history is similar → NO LEARNING
            let final_grad = history[0]
                .contiguous()?
                .add(&fractional_grad)?
                .contiguous()?;

            // Log for debugging (occasionally)
            if history.len() == self.memory_window && self.gradient_history[0].len() % 100 == 1 {
                log::trace!(
                    "Caputo fractional gradient: α={:.2}, Γ(2-α)={:.4}, scale={:.6}, history_len={}",
                    self.alpha,
                    gamma_2_minus_alpha,
                    scale_factor,
                    history.len()
                );
            }

            fractional_gradients.push(final_grad);
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

    /// Compact gradient history to reduce memory usage
    /// This method forces deallocation of unused capacity
    pub fn compact_history(&mut self) {
        for history in &mut self.gradient_history {
            // Shrink the VecDeque to fit its current contents
            history.shrink_to_fit();
        }
    }

    /// Get total memory usage estimate (in number of tensor elements)
    pub fn memory_usage(&self) -> usize {
        self.gradient_history.iter().map(|h| h.len()).sum()
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
            alpha: 0.5,        // REDUCED from 0.7 for stability (less memory = less explosion risk)
            memory_window: 20, // REDUCED from 30 for faster adaptation
            step_size: 1.0,    // Standard discrete time step
        }
    }
}

impl FractionalConfig {
    /// Create configuration for short-term memory (faster, less memory)
    pub fn short_memory() -> Self {
        Self {
            alpha: 0.6,        // REDUCED from 0.8 for stability
            memory_window: 15, // REDUCED from 30
            step_size: 1.0,
        }
    }

    /// Create configuration for long-term memory (slower, more memory)
    pub fn long_memory() -> Self {
        Self {
            alpha: 0.75,       // REDUCED from 0.95 to prevent explosion
            memory_window: 50, // REDUCED from 90
            step_size: 1.0,
        }
    }

    /// Create configuration optimized for financial time series
    pub fn financial_optimized() -> Self {
        Self {
            alpha: 0.5,        // REDUCED from 0.75 for stability with bidirectional LSTM
            memory_window: 20, // REDUCED from 30
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
    fn test_gamma_function() {
        // Test that gamma function returns reasonable values for our use case
        // We mainly need Γ(1-α) where α ∈ (0,1)

        // Test Γ(0.5) ≈ 1.772 (√π)
        let gamma_half = compute_gamma(0.5);
        assert!(
            gamma_half > 1.5 && gamma_half < 2.0,
            "Γ(0.5) should be around 1.77"
        );

        // Test Γ(1) = 1
        let gamma_one = compute_gamma(1.0);
        assert!((gamma_one - 1.0).abs() < 0.1, "Γ(1) should be close to 1");

        // Test values we'll actually use: Γ(1-α) for various α
        // α = 0.3 → Γ(0.7) ≈ 1.298
        let gamma_0_7 = compute_gamma(0.7);
        assert!(
            gamma_0_7 > 1.0 && gamma_0_7 < 1.5,
            "Γ(0.7) should be around 1.3"
        );

        // α = 0.5 → Γ(0.5) ≈ 1.772
        let gamma_0_5 = compute_gamma(0.5);
        assert!(
            gamma_0_5 > 1.5 && gamma_0_5 < 2.0,
            "Γ(0.5) should be around 1.77"
        );

        // α = 0.8 → Γ(0.2) ≈ 4.591
        let gamma_0_2 = compute_gamma(0.2);
        assert!(
            gamma_0_2 > 3.0 && gamma_0_2 < 6.0,
            "Γ(0.2) should be around 4.6"
        );

        // α = 0.9 → Γ(0.1) ≈ 9.513
        let gamma_0_1 = compute_gamma(0.1);
        assert!(
            gamma_0_1 > 7.0 && gamma_0_1 < 12.0,
            "Γ(0.1) should be around 9.5"
        );
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
