//! Fractional Adam optimizer implementation
//!
//! Based on equations 21-25 from the paper:
//! "Training long short-term memory (LSTM) networks efficiently"
//!
//! Implements Fractional Adam with short-memory approximation of fractional derivatives.

use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor, Var};
use candle_nn::optim::Optimizer;
use log;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::fractional::{FractionalConfig, FractionalDerivative};

/// Fractional Adam optimizer parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamsFracAdam {
    pub lr: f64,
    pub beta_1: f64,
    pub beta_2: f64,
    pub eps: f64,
    pub weight_decay: Option<f64>,
    pub fractional: FractionalConfig,
}

impl Default for ParamsFracAdam {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta_1: 0.9,
            beta_2: 0.999,
            eps: 1e-8,
            weight_decay: None,
            fractional: FractionalConfig::default(),
        }
    }
}

/// Fractional Adam optimizer
///
/// Implements equations 21-25 from the paper:
/// m_t^(α) = β₁ m_{t-1}^(α) + (1-β₁) D^α ∇_θ J(θ)
/// v_t^(α) = β₂ v_{t-1}^(α) + (1-β₂) (D^α ∇_θ J(θ))²
///
/// Where D^α is approximated using short-memory Grünwald-Letnikov:
/// D^α ∇_θ J(θ_t) ≈ (1/h^α) Σ_{k=0}^M ω_k^(α) ∇_θ J(θ_{t-k})
pub struct FracAdam {
    vars: Vec<Var>,
    params: ParamsFracAdam,
    fractional_derivative: FractionalDerivative,
    first_moments: HashMap<usize, Tensor>,
    second_moments: HashMap<usize, Tensor>,
    step_count: usize,
}

impl FracAdam {
    pub fn new(vars: Vec<Var>, params: ParamsFracAdam) -> Result<Self> {
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
        })
    }
}

impl Optimizer for FracAdam {
    type Config = ParamsFracAdam;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self> {
        Self::new(vars, config)
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_count += 1;

        // Collect gradients for all variables
        // IMPORTANT: Only add actual gradients to history, not artificial zeros
        let mut gradients = Vec::new();
        let mut has_any_gradient = false;

        for var in &self.vars {
            if let Some(grad) = grads.get(var) {
                // Record presence of a gradient (do not gate by magnitude)
                has_any_gradient = true;
                gradients.push(grad.clone());
            } else {
                // For missing gradients, use the previous gradient if available
                // This maintains continuity in the fractional derivative computation
                if let Some(history) = self
                    .fractional_derivative
                    .get_last_gradient(gradients.len())
                {
                    gradients.push(history.clone());
                } else {
                    // Only create zero gradient if we have no history
                    let zero_grad = var.as_tensor().zeros_like()?;
                    gradients.push(zero_grad);
                }
            }
        }

        // Proceed regardless of gradient magnitude; only skip if literally no gradients exist
        if !has_any_gradient && self.step_count > 1 {
            log::trace!(
                "FracAdam: Skipping step {} due to missing gradients",
                self.step_count
            );
            return Ok(());
        }

        // Update gradient history for fractional derivative computation
        self.fractional_derivative.update_history(&gradients)?;

        // For the first few steps, use regular gradients while building history
        // This ensures stable initial training
        let fractional_grads = if self.step_count <= 3 {
            log::debug!(
                "FracAdam: Using regular gradients for step {} (building history)",
                self.step_count
            );
            gradients.clone()
        } else {
            // Compute fractional gradients using short-memory approximation
            self.fractional_derivative.compute_fractional_gradients()?
        };

        // Apply Fractional Adam updates (equations 21-25)
        for (i, (var, frac_grad)) in self.vars.iter().zip(fractional_grads.iter()).enumerate() {
            // Do not skip small gradients; Adam variants are designed to handle tiny updates
            // Apply weight decay to fractional gradient if specified
            let grad_with_decay = if let Some(weight_decay) = self.params.weight_decay {
                let weight_decay_tensor = Tensor::new(weight_decay as f32, var.device())?
                    .broadcast_as(var.as_tensor().shape())?
                    .contiguous()?;
                let var_tensor = var.as_tensor().contiguous()?;
                frac_grad
                    .contiguous()?
                    .add(&var_tensor.mul(&weight_decay_tensor)?.contiguous()?)?
                    .contiguous()?
            } else {
                frac_grad.contiguous()?
            };

            // Equation 21: m_t^(α) = β₁ m_{t-1}^(α) + (1-β₁) D^α ∇_θ J(θ)
            let beta1_tensor = Tensor::new(self.params.beta_1 as f32, var.device())?
                .broadcast_as(grad_with_decay.shape())?
                .contiguous()?;
            let one_minus_beta1 = Tensor::new((1.0 - self.params.beta_1) as f32, var.device())?
                .broadcast_as(grad_with_decay.shape())?
                .contiguous()?;

            let first_moment = if let Some(prev_m) = self.first_moments.get(&i) {
                prev_m
                    .contiguous()?
                    .mul(&beta1_tensor)?
                    .contiguous()?
                    .add(
                        &grad_with_decay
                            .contiguous()?
                            .mul(&one_minus_beta1)?
                            .contiguous()?,
                    )?
                    .contiguous()?
            } else {
                grad_with_decay
                    .contiguous()?
                    .mul(&one_minus_beta1)?
                    .contiguous()?
            };

            // Equation 22: v_t^(α) = β₂ v_{t-1}^(α) + (1-β₂) (D^α ∇_θ J(θ))²
            let beta2_tensor = Tensor::new(self.params.beta_2 as f32, var.device())?
                .broadcast_as(grad_with_decay.shape())?
                .contiguous()?;
            let one_minus_beta2 = Tensor::new((1.0 - self.params.beta_2) as f32, var.device())?
                .broadcast_as(grad_with_decay.shape())?
                .contiguous()?;
            let grad_squared = grad_with_decay.contiguous()?.sqr()?.contiguous()?;

            let second_moment = if let Some(prev_v) = self.second_moments.get(&i) {
                prev_v
                    .contiguous()?
                    .mul(&beta2_tensor)?
                    .contiguous()?
                    .add(
                        &grad_squared
                            .contiguous()?
                            .mul(&one_minus_beta2)?
                            .contiguous()?,
                    )?
                    .contiguous()?
            } else {
                grad_squared
                    .contiguous()?
                    .mul(&one_minus_beta2)?
                    .contiguous()?
            };

            // Bias correction
            let bias_correction1 = 1.0 - self.params.beta_1.powi(self.step_count as i32);
            let bias_correction2 = 1.0 - self.params.beta_2.powi(self.step_count as i32);

            let bias_corr1_tensor = Tensor::new(bias_correction1 as f32, var.device())?
                .broadcast_as(first_moment.shape())?
                .contiguous()?;
            let bias_corr2_tensor = Tensor::new(bias_correction2 as f32, var.device())?
                .broadcast_as(second_moment.shape())?
                .contiguous()?;

            let corrected_first_moment = first_moment
                .contiguous()?
                .div(&bias_corr1_tensor)?
                .contiguous()?;
            let corrected_second_moment = second_moment
                .contiguous()?
                .div(&bias_corr2_tensor)?
                .contiguous()?;

            // Equation 25: θ = θ - η * m̂_t^(α) / (√v̂_t^(α) + ε)
            let eps_tensor = Tensor::new(self.params.eps as f32, var.device())?
                .broadcast_as(corrected_second_moment.shape())?
                .contiguous()?;
            let lr_tensor = Tensor::new(self.params.lr as f32, var.device())?
                .broadcast_as(corrected_first_moment.shape())?
                .contiguous()?;

            let denominator = corrected_second_moment
                .contiguous()?
                .sqrt()?
                .contiguous()?
                .add(&eps_tensor)?
                .contiguous()?;
            let update = corrected_first_moment
                .contiguous()?
                .div(&denominator)?
                .contiguous()?
                .mul(&lr_tensor)?
                .contiguous()?;

            // Update parameter using Var::set_tensor
            let old_param = var.as_tensor().contiguous()?;
            let new_param = old_param.sub(&update)?.contiguous()?;

            // Only check parameter updates occasionally and in early steps
            if (self.step_count <= 5 || self.step_count % 500 == 0) && cfg!(debug_assertions) {
                let param_change = old_param
                    .sub(&new_param)?
                    .abs()?
                    .mean_all()?
                    .to_scalar::<f32>()
                    .unwrap_or(0.0);

                // Much more lenient threshold and only trace level
                if param_change < 1e-8 && self.step_count > 10 {
                    // This is often normal - some parameters don't update every step
                    log::trace!(
                        "FracAdam: Parameter {} has small update: {:.2e} at step {}",
                        i,
                        param_change,
                        self.step_count
                    );
                } else if self.step_count <= 5 && param_change > 1e-6 {
                    log::debug!(
                        "FracAdam: Parameter {} update magnitude: {:.6} (warmup step {})",
                        i,
                        param_change,
                        self.step_count
                    );
                }
            }

            var.set(&new_param)?;

            // Store updated moments
            self.first_moments.insert(i, first_moment);
            self.second_moments.insert(i, second_moment);
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
    fn test_frac_adam_creation() {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let params = ParamsFracAdam::default();

        let optimizer = FracAdam::new(vec![var], params);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_frac_adam_config_validation() {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let mut params = ParamsFracAdam::default();
        params.fractional.alpha = 0.0; // Invalid

        let result = FracAdam::new(vec![var], params);
        assert!(result.is_err());
    }

    #[test]
    fn test_learning_rate_methods() -> Result<()> {
        let device = Device::Cpu;
        let var = Var::new(&[1.0f32, 2.0, 3.0], &device)?;
        let params = ParamsFracAdam {
            lr: 0.01,
            ..Default::default()
        };
        let mut optimizer = FracAdam::new(vec![var], params)?;

        assert_eq!(optimizer.learning_rate(), 0.01);

        optimizer.set_learning_rate(0.005);
        assert_eq!(optimizer.learning_rate(), 0.005);

        Ok(())
    }
}
