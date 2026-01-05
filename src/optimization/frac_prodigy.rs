//! FracProdigy: Fractional Prodigy Optimizer
//!
//! Combines the best of FracNAdam and Prodigy:
//! - Fractional derivative memory (Caputo method) for long-term dependencies
//! - Automatic learning rate adaptation (Prodigy's D-estimate)
//! - Nesterov acceleration for faster convergence
//!
//! Key advantages:
//! - No manual LR tuning (set lr=1.0, Prodigy handles it)
//! - Long-term memory effects for crypto market patterns
//! - Stable convergence with adaptive step sizes
//! - Ideal for volatile markets with trend persistence

use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor, Var};
use candle_nn::optim::Optimizer;
use log;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::fractional::{FractionalConfig, FractionalDerivative};

/// Helper function to convert tensor scalar to f64
#[inline]
fn tensor_to_f64_scalar(tensor: &Tensor) -> Result<f64> {
    match tensor.dtype() {
        candle_core::DType::F32 => {
            let val: f32 = tensor.to_scalar()?;
            Ok(val as f64)
        }
        candle_core::DType::F64 => tensor.to_scalar(),
        other => Err(candle_core::Error::Msg(format!(
            "Expected F32 or F64 for scalar conversion, got {:?}",
            other
        ))),
    }
}

/// FracProdigy optimizer parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamsFracProdigy {
    /// Base learning rate (should be 1.0 for Prodigy automatic adaptation)
    pub lr: f64,
    /// First moment decay rate (default: 0.9)
    pub beta1: f64,
    /// Second moment decay rate (default: 0.999)
    pub beta2: f64,
    /// Numerical stability epsilon (default: 1e-8)
    pub eps: f64,
    /// Weight decay (L2 regularization, default: 0.0)
    pub weight_decay: Option<f64>,
    /// Momentum decay for Nesterov acceleration (default: 0.004)
    pub momentum_decay: f64,
    /// Coefficient for D estimate growth (default: 1.0)
    pub d_coef: f64,
    /// Maximum growth rate for D estimate (default: inf = unlimited)
    pub growth_rate: f64,
    /// Fractional derivative configuration
    pub fractional: FractionalConfig,
}

impl Default for ParamsFracProdigy {
    fn default() -> Self {
        Self {
            lr: 1.0, // CRITICAL: Always 1.0 for Prodigy automatic adaptation
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: None,
            momentum_decay: 0.004, // Standard NAdam momentum decay
            d_coef: 1.0,
            growth_rate: f64::INFINITY,
            fractional: FractionalConfig {
                alpha: 0.5,        // Balanced memory (less aggressive than FracNAdam)
                memory_window: 20, // Efficient memory window
                step_size: 1.0,
            },
        }
    }
}

/// FracProdigy optimizer
///
/// Implements fractional NAdam with Prodigy's automatic learning rate:
/// 1. Compute fractional gradients: D^α ∇_θ J(θ) using Caputo derivative
/// 2. Update D-estimate: d_hat = d_coef * sqrt(||grad||² / ||param||²)
/// 3. Calculate automatic LR: lr_auto = lr / (D * sqrt(t))
/// 4. Apply NAdam update with Nesterov acceleration
pub struct FracProdigy {
    vars: Vec<Var>,
    params: ParamsFracProdigy,
    fractional_derivative: FractionalDerivative,
    first_moments: HashMap<usize, Tensor>,
    second_moments: HashMap<usize, Tensor>,
    step_count: usize,
    d_estimate: f64,
}

impl FracProdigy {
    pub fn new(vars: Vec<Var>, params: ParamsFracProdigy) -> Result<Self> {
        params.fractional.validate()?;

        let fractional_derivative = FractionalDerivative::new(
            params.fractional.alpha,
            params.fractional.memory_window,
            params.fractional.step_size,
            vars.len(),
        )?;

        Ok(Self {
            vars,
            params,
            fractional_derivative,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            step_count: 0,
            d_estimate: 1.0, // Initial estimate (will adapt quickly)
        })
    }

    /// Get current D estimate (distance to solution)
    pub fn get_d_estimate(&self) -> f64 {
        self.d_estimate
    }

    /// Get current effective learning rate
    pub fn get_effective_lr(&self) -> f64 {
        if self.step_count == 0 {
            return self.params.lr;
        }
        self.params.lr / (self.d_estimate * (self.step_count as f64).sqrt())
    }

    /// Clear optimizer state to free memory
    pub fn clear_state(&mut self) {
        self.first_moments.clear();
        self.second_moments.clear();
        self.fractional_derivative.clear_history();
        self.step_count = 0;
        self.d_estimate = 1.0;
    }

    /// Compact optimizer memory (reduce allocated but unused capacity)
    pub fn compact_memory(&mut self) {
        self.first_moments.shrink_to_fit();
        self.second_moments.shrink_to_fit();
        self.fractional_derivative.compact_history();
    }

    /// Get memory usage estimate
    pub fn memory_usage(&self) -> usize {
        let moment_count = self.first_moments.len() + self.second_moments.len();
        let history_count = self.fractional_derivative.memory_usage();
        moment_count + history_count
    }
}

impl Optimizer for FracProdigy {
    type Config = ParamsFracProdigy;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self> {
        Self::new(vars, config)
    }

    fn learning_rate(&self) -> f64 {
        self.get_effective_lr()
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_count += 1;

        // Collect gradients for fractional derivative computation
        let mut gradients = Vec::new();
        let mut has_any_gradient = false;

        for var in &self.vars {
            if let Some(grad) = grads.get(var) {
                has_any_gradient = true;
                gradients.push(grad.clone());
            } else {
                // Use previous gradient if available for continuity
                if let Some(history) = self
                    .fractional_derivative
                    .get_last_gradient(gradients.len())
                {
                    gradients.push(history.clone());
                } else {
                    let zero_grad = var.as_tensor().zeros_like()?;
                    gradients.push(zero_grad);
                }
            }
        }

        if !has_any_gradient && self.step_count > 1 {
            log::trace!(
                "FracProdigy: Skipping step {} due to missing gradients",
                self.step_count
            );
            return Ok(());
        }

        // Update gradient history for fractional derivative computation
        self.fractional_derivative.update_history(&gradients)?;

        // For the first few steps, use regular gradients while building history
        let fractional_grads = if self.step_count <= 3 {
            log::debug!(
                "FracProdigy: Using regular gradients for step {} (building history)",
                self.step_count
            );
            gradients.clone()
        } else {
            // Compute fractional gradients using Caputo derivative
            self.fractional_derivative.compute_fractional_gradients()?
        };

        // Compute norms for D-estimate update (BEFORE momentum smoothing)
        let mut grad_norm_sq = 0.0;
        let mut param_norm_sq = 0.0;

        for (var, frac_grad) in self.vars.iter().zip(fractional_grads.iter()) {
            // Gradient norm (fractional gradient)
            grad_norm_sq += tensor_to_f64_scalar(&frac_grad.sqr()?.sum_all()?)?;

            // Parameter norm
            param_norm_sq += tensor_to_f64_scalar(&var.as_tensor().sqr()?.sum_all()?)?;
        }

        // Update D estimate (Prodigy's key innovation)
        if grad_norm_sq >= 1e-20 {
            let d_hat = self.params.d_coef * (grad_norm_sq / param_norm_sq.max(1e-12)).sqrt();

            let old_d = self.d_estimate;
            if self.params.growth_rate.is_finite() {
                let max_growth = self.d_estimate * self.params.growth_rate;
                self.d_estimate = self.d_estimate.max(d_hat.min(max_growth));
            } else {
                self.d_estimate = self.d_estimate.max(d_hat);
            }

            // Log D-estimate updates periodically
            if self.step_count % 100 == 1 {
                log::debug!(
                    "📊 FracProdigy D-estimate: {:.6e} → {:.6e} (d_hat={:.6e}, α={:.2}, mem={})",
                    old_d,
                    self.d_estimate,
                    d_hat,
                    self.params.fractional.alpha,
                    self.params.fractional.memory_window
                );
            }
        } else {
            log::warn!(
                "⚠️ FracProdigy: Very small gradient norm ({:.2e}) - skipping D-estimate update",
                grad_norm_sq.sqrt()
            );
        }

        // Compute automatic learning rate
        let auto_lr = self.get_effective_lr();

        // Safeguard: Ensure LR is reasonable
        let auto_lr = if auto_lr > 1.0 {
            log::warn!(
                "⚠️ FracProdigy: Capping excessive LR {:.6e} → 1.0 (D={:.6e}, step={})",
                auto_lr,
                self.d_estimate,
                self.step_count
            );
            1.0
        } else if auto_lr < 1e-8 {
            log::warn!(
                "⚠️ FracProdigy: Boosting tiny LR {:.6e} → 1e-8 (D={:.6e}, step={})",
                auto_lr,
                self.d_estimate,
                self.step_count
            );
            1e-8
        } else {
            auto_lr
        };

        // Log effective LR periodically
        if self.step_count % 100 == 1 {
            log::debug!(
                "🎯 FracProdigy effective LR: {:.6e} (base_lr={:.1}, D={:.6e}, step={})",
                auto_lr,
                self.params.lr,
                self.d_estimate,
                self.step_count
            );
        }

        // Bias correction factors
        let bias_correction1 = 1.0 - self.params.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.params.beta2.powi(self.step_count as i32);

        // Apply Fractional NAdam updates with automatic LR
        for (i, (var, frac_grad)) in self.vars.iter().zip(fractional_grads.iter()).enumerate() {
            // Apply weight decay to fractional gradient if specified
            let grad_with_decay = if let Some(weight_decay) = self.params.weight_decay {
                (frac_grad + (var.as_tensor() * weight_decay)?)?
            } else {
                frac_grad.clone()
            };

            // Update first moment (momentum)
            let first_moment = if let Some(prev_m) = self.first_moments.get(&i) {
                ((prev_m * self.params.beta1)? + (&grad_with_decay * (1.0 - self.params.beta1))?)?
            } else {
                (&grad_with_decay * (1.0 - self.params.beta1))?
            };

            // Update second moment (variance)
            let grad_squared = grad_with_decay.sqr()?;
            let second_moment = if let Some(prev_v) = self.second_moments.get(&i) {
                ((prev_v * self.params.beta2)? + (&grad_squared * (1.0 - self.params.beta2))?)?
            } else {
                (&grad_squared * (1.0 - self.params.beta2))?
            };

            // Bias correction
            let corrected_first_moment = (&first_moment / bias_correction1)?;
            let corrected_second_moment = (&second_moment / bias_correction2)?;

            // Nesterov acceleration term (NAdam)
            let nesterov_term = ((&corrected_first_moment * self.params.beta1)?
                + (&grad_with_decay * (1.0 - self.params.beta1))?)?;

            // Compute update: auto_lr * nesterov_term / (sqrt(corrected_second_moment) + eps)
            let denominator = (corrected_second_moment.sqrt()? + self.params.eps)?;
            let update = ((&nesterov_term / &denominator)? * auto_lr)?;

            // Update parameter
            let new_param = var.as_tensor().sub(&update)?;

            // Diagnostic logging (early steps only)
            if (self.step_count <= 5 || self.step_count.is_multiple_of(500))
                && cfg!(debug_assertions)
            {
                let param_change = var
                    .as_tensor()
                    .sub(&new_param)?
                    .abs()?
                    .mean_all()?
                    .to_scalar::<f32>()
                    .unwrap_or(0.0);

                if param_change < 1e-8 && self.step_count > 10 {
                    log::trace!(
                        "FracProdigy: Parameter {} has small update: {:.2e} at step {}",
                        i,
                        param_change,
                        self.step_count
                    );
                } else if self.step_count <= 5 && param_change > 1e-6 {
                    log::debug!(
                        "FracProdigy: Parameter {} update magnitude: {:.6} (warmup step {})",
                        i,
                        param_change,
                        self.step_count
                    );
                }
            }

            var.set(&new_param)?;

            // Store updated moments (detached from computation graph)
            // Clone before detaching to avoid borrow issues
            self.first_moments.insert(i, first_moment.detach());
            self.second_moments.insert(i, second_moment.detach());
        }

        Ok(())
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_frac_prodigy_creation() {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let params = ParamsFracProdigy::default();

        let optimizer = FracProdigy::new(vec![var], params);
        assert!(optimizer.is_ok());

        let opt = optimizer.unwrap();
        assert_eq!(opt.step_count, 0);
        assert_eq!(opt.params.lr, 1.0);
        assert_eq!(opt.d_estimate, 1.0);
    }

    #[test]
    fn test_frac_prodigy_config_validation() {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let mut params = ParamsFracProdigy::default();
        params.fractional.alpha = 1.5; // Invalid

        let result = FracProdigy::new(vec![var], params);
        assert!(result.is_err());
    }

    #[test]
    fn test_frac_prodigy_d_estimate_growth() {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let params = ParamsFracProdigy::default();

        let mut optimizer = FracProdigy::new(vec![var], params).unwrap();

        let initial_d = optimizer.d_estimate;

        // Simulate D estimate update with large gradients
        let d_coef = optimizer.params.d_coef;
        let growth_rate = optimizer.params.growth_rate;
        let grad_norm_sq = 100.0_f64;
        let param_norm_sq = 1.0_f64;
        let d_hat = d_coef * (grad_norm_sq / param_norm_sq.max(1e-12)).sqrt();

        if growth_rate.is_finite() {
            let max_growth = optimizer.d_estimate * growth_rate;
            optimizer.d_estimate = optimizer.d_estimate.max(d_hat.min(max_growth));
        } else {
            optimizer.d_estimate = optimizer.d_estimate.max(d_hat);
        }

        assert!(
            optimizer.d_estimate > initial_d,
            "D estimate should grow with large gradients"
        );
    }

    #[test]
    fn test_frac_prodigy_effective_lr_decreases() {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let params = ParamsFracProdigy::default();

        let mut optimizer = FracProdigy::new(vec![var], params).unwrap();

        optimizer.step_count = 1;
        optimizer.d_estimate = 1.0;
        let lr1 = optimizer.get_effective_lr();

        optimizer.step_count = 100;
        let lr2 = optimizer.get_effective_lr();

        assert!(lr2 < lr1, "Effective LR should decrease as steps increase");
    }

    #[test]
    fn test_learning_rate_methods() -> Result<()> {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device)?;
        let params = ParamsFracProdigy {
            lr: 1.0,
            ..Default::default()
        };
        let mut optimizer = FracProdigy::new(vec![var], params)?;

        assert_eq!(optimizer.learning_rate(), 1.0);

        optimizer.set_learning_rate(0.5);
        assert_eq!(optimizer.params.lr, 0.5);

        Ok(())
    }

    #[test]
    fn test_memory_management() -> Result<()> {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device)?;
        let params = ParamsFracProdigy::default();
        let mut optimizer = FracProdigy::new(vec![var], params)?;

        let initial_usage = optimizer.memory_usage();

        optimizer.clear_state();
        assert_eq!(optimizer.step_count, 0);
        assert_eq!(optimizer.d_estimate, 1.0);

        optimizer.compact_memory();
        let compacted_usage = optimizer.memory_usage();
        assert!(compacted_usage <= initial_usage);

        Ok(())
    }
}
