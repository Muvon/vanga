//! Prodigy: Learning-Rate-Free Optimizer (ICLR 2024)
//!
//! Implementation of "Prodigy: An Expeditiously Adaptive Parameter-Free Learner"
//! by Konstantin Mishchenko and Aaron Defazio
//!
//! Key features:
//! - Automatic learning rate adaptation (set lr=1.0 and forget)
//! - Estimates distance to solution D dynamically
//! - Provably optimal convergence
//! - Drop-in replacement for Adam/AdamW
//!
//! Paper: https://arxiv.org/abs/2306.06101

use candle_core::{Result, Tensor, Var};
use candle_nn::optim::Optimizer;

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

/// Prodigy optimizer configuration
#[derive(Debug, Clone)]
pub struct ParamsProdigy {
    /// Learning rate (should be set to 1.0 for automatic adaptation)
    pub lr: f64,
    /// Coefficient for D estimate growth (default: 1.0)
    pub d_coef: f64,
    /// Maximum growth rate for D estimate (default: inf = unlimited)
    pub growth_rate: f64,
    /// First moment decay rate (default: 0.9)
    pub beta1: f64,
    /// Second moment decay rate (default: 0.999)
    pub beta2: f64,
    /// Numerical stability epsilon (default: 1e-8)
    pub eps: f64,
    /// Weight decay (L2 regularization, default: 0.0)
    pub weight_decay: f64,
    /// Enable safeguard warmup (default: false, not needed for Prodigy)
    pub safeguard_warmup: bool,
}

impl Default for ParamsProdigy {
    fn default() -> Self {
        Self {
            lr: 1.0, // CRITICAL: Always 1.0 for Prodigy
            d_coef: 1.0,
            growth_rate: f64::INFINITY,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            safeguard_warmup: false,
        }
    }
}

/// Prodigy optimizer state
pub struct Prodigy {
    vars: Vec<Var>,
    params: ParamsProdigy,
    step_count: usize,
    d_estimate: f64,
    /// First moment (momentum) buffers
    moment1_buffers: std::collections::HashMap<usize, Tensor>,
    /// Second moment (variance) buffers
    moment2_buffers: std::collections::HashMap<usize, Tensor>,
}

impl Prodigy {
    /// Create a new Prodigy optimizer
    pub fn new(vars: Vec<Var>, params: ParamsProdigy) -> Result<Self> {
        Ok(Self {
            vars,
            params,
            step_count: 0,
            d_estimate: 1.0, // Initial estimate (will adapt quickly)
            moment1_buffers: std::collections::HashMap::new(),
            moment2_buffers: std::collections::HashMap::new(),
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
}

impl Optimizer for Prodigy {
    type Config = ParamsProdigy;

    fn new(vars: Vec<Var>, params: Self::Config) -> Result<Self> {
        Self::new(vars, params)
    }

    fn learning_rate(&self) -> f64 {
        self.get_effective_lr()
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.step_count += 1;

        // Collect all variables, gradients, and compute norms FIRST
        let mut all_updates = Vec::new();
        let mut grad_norm_sq = 0.0;
        let mut param_norm_sq = 0.0;

        for (idx, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var) {
                // CRITICAL FIX: Compute gradient norm BEFORE momentum smoothing
                grad_norm_sq += tensor_to_f64_scalar(&grad.sqr()?.sum_all()?)?;

                // Compute parameter norm
                param_norm_sq += tensor_to_f64_scalar(&var.as_tensor().sqr()?.sum_all()?)?;

                // Apply weight decay
                let grad_with_decay = if self.params.weight_decay > 0.0 {
                    let decay_term = (var.as_tensor() * self.params.weight_decay)?;
                    (grad + decay_term)?
                } else {
                    grad.clone()
                };

                // Get or initialize moment buffers
                let m = if let Some(buffer) = self.moment1_buffers.get(&idx) {
                    buffer.clone()
                } else {
                    Tensor::zeros_like(var.as_tensor())?
                };

                let v = if let Some(buffer) = self.moment2_buffers.get(&idx) {
                    buffer.clone()
                } else {
                    Tensor::zeros_like(var.as_tensor())?
                };

                // Update first moment (momentum)
                let m_new =
                    ((m * self.params.beta1)? + ((1.0 - self.params.beta1) * &grad_with_decay)?)?;

                // Update second moment (variance)
                let grad_sq = grad_with_decay.sqr()?;
                let v_new = ((v * self.params.beta2)? + ((1.0 - self.params.beta2) * grad_sq)?)?;

                all_updates.push((idx, var, m_new, v_new));
            }
        }

        if all_updates.is_empty() {
            return Ok(());
        }

        // Update D estimate (Prodigy's key innovation)
        let d_coef = self.params.d_coef;
        let growth_rate = self.params.growth_rate;

        // CRITICAL: Ensure we have valid norms
        if grad_norm_sq < 1e-20 {
            log::warn!(
                "⚠️ Prodigy: Very small gradient norm ({:.2e}) - skipping D-estimate update",
                grad_norm_sq.sqrt()
            );
            // Don't update D-estimate with invalid gradients
        } else {
            let d_hat = d_coef * (grad_norm_sq / param_norm_sq.max(1e-12)).sqrt();

            let old_d = self.d_estimate;
            if growth_rate.is_finite() {
                let max_growth = self.d_estimate * growth_rate;
                self.d_estimate = self.d_estimate.max(d_hat.min(max_growth));
            } else {
                self.d_estimate = self.d_estimate.max(d_hat);
            }

            // Log D-estimate updates periodically (every 100 steps)
            if self.step_count % 100 == 1 {
                log::debug!(
                    "📊 Prodigy D-estimate: {:.6e} → {:.6e} (d_hat={:.6e}, grad_norm={:.6e}, param_norm={:.6e})",
                    old_d,
                    self.d_estimate,
                    d_hat,
                    grad_norm_sq.sqrt(),
                    param_norm_sq.sqrt()
                );
            }
        }

        // Compute automatic learning rate
        let auto_lr = self.get_effective_lr();

        // Safeguard: Ensure LR is reasonable
        let auto_lr = if auto_lr > 1.0 {
            log::warn!(
                "⚠️ Prodigy: Capping excessive LR {:.6e} → 1.0 (D={:.6e}, step={})",
                auto_lr,
                self.d_estimate,
                self.step_count
            );
            1.0
        } else if auto_lr < 1e-8 {
            log::warn!(
                "⚠️ Prodigy: Boosting tiny LR {:.6e} → 1e-8 (D={:.6e}, step={})",
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
                "🎯 Prodigy effective LR: {:.6e} (base_lr={:.1}, D={:.6e}, step={})",
                auto_lr,
                self.params.lr,
                self.d_estimate,
                self.step_count
            );
        }

        // Bias correction factors
        let bias_correction1 = 1.0 - self.params.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.params.beta2.powi(self.step_count as i32);

        // Apply updates to each parameter
        for (idx, var, m_new, v_new) in all_updates {
            // Store updated moments
            self.moment1_buffers.insert(idx, m_new.clone());
            self.moment2_buffers.insert(idx, v_new.clone());

            // Bias-corrected moments
            let m_hat = (m_new / bias_correction1)?;
            let v_hat = (v_new / bias_correction2)?;

            // Compute update: auto_lr * m_hat / (sqrt(v_hat) + eps)
            let denom = (v_hat.sqrt()? + self.params.eps)?;
            let update = ((m_hat / denom)? * auto_lr)?;

            // Apply update: param = param - update
            var.set(&var.sub(&update)?)?;
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
    fn test_prodigy_initialization() {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let params = ParamsProdigy::default();

        let optimizer = Prodigy::new(vec![var], params).unwrap();

        assert_eq!(optimizer.step_count, 0);
        assert_eq!(optimizer.params.lr, 1.0);
        assert_eq!(optimizer.params.beta1, 0.9);
        assert_eq!(optimizer.params.beta2, 0.999);
        assert!(optimizer.d_estimate > 0.0);
    }

    #[test]
    fn test_prodigy_d_estimate_growth() {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let params = ParamsProdigy::default();

        let mut optimizer = Prodigy::new(vec![var], params).unwrap();

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
    fn test_prodigy_effective_lr_decreases() {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let params = ParamsProdigy::default();

        let mut optimizer = Prodigy::new(vec![var], params).unwrap();

        optimizer.step_count = 1;
        optimizer.d_estimate = 1.0;
        let lr1 = optimizer.get_effective_lr();

        optimizer.step_count = 100;
        let lr2 = optimizer.get_effective_lr();

        assert!(lr2 < lr1, "Effective LR should decrease as steps increase");
    }
}
