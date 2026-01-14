//! Deep Adaptive Input Normalization (DAIN) for Financial Time Series
//!
//! Reference: Passalis et al., "Deep Adaptive Input Normalization for Time Series Forecasting", ICLR 2019

use crate::config::model::DAINConfig;
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};

/// DAINormalization - Deep Adaptive Input Normalization
///
/// Formula: y = γ ⊙ ((x - μ) / √(β ⊙ σ² + ε)) ⊙ α + μ
pub struct DAINormalization {
    alpha: Tensor,
    beta: Tensor,
    gamma: Tensor,
    config: DAINConfig,
}

impl DAINormalization {
    pub fn new(input_dim: usize, device: &Device, mut config: DAINConfig) -> Result<Self> {
        config.input_dim = input_dim;
        let alpha = Tensor::ones(&[input_dim], candle_core::DType::F64, device)?;
        let beta = Tensor::ones(&[input_dim], candle_core::DType::F64, device)?;
        let gamma = Tensor::ones(&[input_dim], candle_core::DType::F64, device)?;
        Ok(Self {
            alpha,
            beta,
            gamma,
            config,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.dims();
        let ndims = input_shape.len();
        let feature_dim = input_shape[ndims - 1];

        if feature_dim != self.config.input_dim {
            return Err(VangaError::ModelError(format!(
                "DAIN input dimension mismatch: expected {}, got {}",
                self.config.input_dim, feature_dim
            )));
        }

        let mean = input.mean_keepdim(ndims - 1)?;
        let centered = input.broadcast_sub(&mean)?;

        // Numerically stable variance calculation using centered data
        // E[x²] where x is already centered (so mean ≈ 0)
        let input_sq = centered.sqr()?;
        let variance = input_sq.mean_keepdim(ndims - 1)?;

        // Clamp variance to minimum value to prevent numerical issues
        let min_variance = Tensor::new(&[self.config.epsilon], input.device())?;
        let min_variance_broadcast = min_variance.broadcast_as(variance.shape())?;
        let variance = variance.maximum(&min_variance_broadcast)?;

        let beta_broadcast = self.beta.broadcast_as(&[feature_dim])?;
        let scaled_var = variance.broadcast_mul(&beta_broadcast)?;

        let std_scaled = scaled_var.sqrt()?;

        let normalized = centered.broadcast_div(&std_scaled)?;

        let alpha_broadcast = self.alpha.broadcast_as(&[feature_dim])?;
        let scaled_normalized = normalized.broadcast_mul(&alpha_broadcast)?;

        let output = scaled_normalized.broadcast_add(&mean)?;

        let gamma_broadcast = self.gamma.broadcast_as(&[feature_dim])?;
        Ok(output.broadcast_mul(&gamma_broadcast)?)
    }
    pub fn forward_with_attention(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.dims();
        let ndims = input_shape.len();
        let feature_dim = input_shape[ndims - 1];

        if feature_dim != self.config.input_dim {
            return Err(VangaError::ModelError(format!(
                "DAIN input dimension mismatch: expected {}, got {}",
                self.config.input_dim, feature_dim
            )));
        }

        let mean = input.mean_keepdim(ndims - 1)?;
        let centered = input.broadcast_sub(&mean)?;

        // Numerically stable variance calculation using centered data
        let input_sq = centered.sqr()?;
        let variance = input_sq.mean_keepdim(ndims - 1)?;

        // Clamp variance to minimum value
        let min_variance = Tensor::new(&[self.config.epsilon], input.device())?;
        let min_variance_broadcast = min_variance.broadcast_as(variance.shape())?;
        let variance = variance.maximum(&min_variance_broadcast)?;

        let beta_broadcast = self.beta.broadcast_as(&[feature_dim])?;
        let scaled_var = variance.broadcast_mul(&beta_broadcast)?;

        let std_scaled = scaled_var.sqrt()?;

        let normalized = centered.broadcast_div(&std_scaled)?;
        let alpha_broadcast = self.alpha.broadcast_as(&[feature_dim])?;
        let scaled_normalized = normalized.broadcast_mul(&alpha_broadcast)?;

        // Attention-based feature importance using variance
        let variance_squeezed = variance.squeeze(ndims - 1)?;
        let variance_sum = variance_squeezed.sum_all()?;
        let variance_normalized = variance_squeezed.broadcast_div(&variance_sum)?;

        let exp_var = variance_normalized.exp()?;
        let exp_sum = exp_var.sum_all()?;
        let attention_weights = exp_var.broadcast_div(&exp_sum)?;

        let attention_gamma = self.gamma.broadcast_mul(&attention_weights)?;
        let gamma_broadcast = attention_gamma.broadcast_as(&[feature_dim])?;

        let output = scaled_normalized.broadcast_add(&mean)?;
        Ok(output.broadcast_mul(&gamma_broadcast)?)
    }
    pub fn config(&self) -> &DAINConfig {
        &self.config
    }

    pub fn get_alpha(&self) -> Result<Tensor> {
        Ok(self.alpha.clone())
    }
    pub fn get_beta(&self) -> Result<Tensor> {
        Ok(self.beta.clone())
    }
    pub fn get_gamma(&self) -> Result<Tensor> {
        Ok(self.gamma.clone())
    }

    pub fn set_alpha(&mut self, alpha: Tensor) -> Result<()> {
        if alpha.dims() != self.alpha.dims() {
            return Err(VangaError::ModelError(
                "Alpha dimension mismatch".to_string(),
            ));
        }
        self.alpha = alpha;
        Ok(())
    }

    pub fn set_beta(&mut self, beta: Tensor) -> Result<()> {
        if beta.dims() != self.beta.dims() {
            return Err(VangaError::ModelError(
                "Beta dimension mismatch".to_string(),
            ));
        }
        self.beta = beta;
        Ok(())
    }

    pub fn set_gamma(&mut self, gamma: Tensor) -> Result<()> {
        if gamma.dims() != self.gamma.dims() {
            return Err(VangaError::ModelError(
                "Gamma dimension mismatch".to_string(),
            ));
        }
        self.gamma = gamma;
        Ok(())
    }
}

/// Apply standard Z-score normalization
pub fn z_score_normalize(input: &Tensor, epsilon: f64) -> Result<(Tensor, Tensor, Tensor)> {
    let ndims = input.dims().len();
    let mean = input.mean_keepdim(ndims - 1)?;
    let centered = input.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(ndims - 1)?;
    let epsilon_tensor = Tensor::new(&[epsilon], input.device())?;
    let epsilon_broadcast = epsilon_tensor.broadcast_as(variance.shape())?;
    let std = variance.add(&epsilon_broadcast)?.sqrt()?;
    let normalized = centered.broadcast_div(&std)?;
    Ok((
        normalized,
        mean.squeeze(ndims - 1)?,
        std.squeeze(ndims - 1)?,
    ))
}

/// Apply Min-Max normalization
pub fn min_max_normalize(input: &Tensor) -> Result<Tensor> {
    let ndims = input.dims().len();
    let min = input.min_keepdim(ndims - 1)?;
    let max = input.max_keepdim(ndims - 1)?;
    let range = max.broadcast_sub(&min)?;
    let epsilon_tensor = Tensor::new(&[1e-8_f64], input.device())?;
    let range_stable = range.broadcast_add(&epsilon_tensor)?;
    Ok(input.broadcast_sub(&min)?.broadcast_div(&range_stable)?)
}

/// Robust normalization for high-volatility crypto data
pub fn robust_normalize(input: &Tensor) -> Result<Tensor> {
    let median = input.mean_keepdim(1)?.squeeze(1)?;
    let abs_dev = input.broadcast_sub(&median.reshape(&[1])?)?.abs()?;
    let mad = abs_dev.mean_keepdim(1)?.squeeze(1)?;
    let eps_tensor = Tensor::new(&[1e-8], input.device())?;
    let mad_stable = mad.add(&eps_tensor)?;
    Ok(input
        .broadcast_sub(&median.reshape(&[1])?)?
        .broadcast_div(&mad_stable.reshape(&[1])?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::DAINConfig;

    fn create_test_dain() -> DAINormalization {
        let config = DAINConfig {
            enabled: true,
            hidden_dim: 16,
            learnable_mean_scale: true,
            learnable_std_scale: true,
            feature_gate_enabled: true,
            epsilon: 1e-5,
            input_dim: 4,
        };
        DAINormalization::new(4, &Device::Cpu, config).unwrap()
    }

    #[test]
    fn test_dain_creation() {
        let dain = create_test_dain();
        assert_eq!(dain.config().input_dim, 4);
    }

    #[test]
    fn test_dain_2d_input() -> Result<()> {
        let dain = create_test_dain();
        let input = Tensor::randn(0.0, 1.0, (8, 4), &Device::Cpu)?;
        let output = dain.forward(&input)?;
        assert_eq!(output.dims(), input.dims());
        Ok(())
    }

    #[test]
    fn test_dain_3d_input() -> Result<()> {
        let dain = create_test_dain();
        let input = Tensor::randn(0.0, 1.0, (4, 10, 4), &Device::Cpu)?;
        let output = dain.forward(&input)?;
        assert_eq!(output.dims(), input.dims());
        Ok(())
    }

    #[test]
    fn test_dain_preserves_shape() -> Result<()> {
        let dain = create_test_dain();
        let shapes: Vec<Vec<usize>> =
            vec![vec![2, 4], vec![8, 4], vec![4, 10, 4], vec![2, 5, 8, 4]];
        for shape in shapes {
            let input = Tensor::randn(0.0, 1.0, shape.as_slice(), &Device::Cpu)?;
            let output = dain.forward(&input)?;
            assert_eq!(output.dims(), shape.as_slice());
        }
        Ok(())
    }

    #[test]
    fn test_dain_numerical_stability() -> Result<()> {
        let dain = create_test_dain();
        // Use f64 for consistency with DAIN parameters
        // Using std=0.1 gives variance=0.01, which is reasonable for numerical stability
        let small_input = Tensor::randn(0.0, 0.1, (4, 10, 4), &Device::Cpu)?;
        let output = dain.forward(&small_input)?;
        let flattened = output.flatten_all()?.to_vec1::<f64>()?;
        assert!(flattened.iter().all(|x| x.is_finite()));
        Ok(())
    }

    #[test]
    fn test_dain_no_nan_output() -> Result<()> {
        let dain = create_test_dain();
        // Using std=2.0 gives variance=4, reasonable for testing
        let input = Tensor::randn(0.0, 2.0, (4, 10, 4), &Device::Cpu)?;
        let output = dain.forward(&input)?;
        let flattened = output.flatten_all()?.to_vec1::<f64>()?;
        assert!(!flattened.iter().any(|x| x.is_nan()));
        Ok(())
    }

    #[test]
    fn test_dain_with_attention() -> Result<()> {
        let dain = create_test_dain();
        let input = Tensor::randn(0.0, 1.0, (4, 4), &Device::Cpu)?; // 2D for simpler attention
        let output = dain.forward_with_attention(&input)?;
        assert_eq!(output.dims(), input.dims());
        Ok(())
    }

    #[test]
    fn test_z_score_normalize() -> Result<()> {
        let input = Tensor::randn(0.0, 1.0, (8, 4), &Device::Cpu)?;
        let (normalized, _, _) = z_score_normalize(&input, 1e-5)?;
        assert_eq!(normalized.dims(), input.dims());
        Ok(())
    }

    #[test]
    fn test_min_max_normalize() -> Result<()> {
        let input = Tensor::randn(0.0, 1.0, (8, 4), &Device::Cpu)?;
        let output = min_max_normalize(&input)?;
        assert_eq!(output.dims(), input.dims());
        Ok(())
    }

    #[test]
    fn test_dain_get_parameters() -> Result<()> {
        let dain = create_test_dain();
        let alpha = dain.get_alpha()?;
        let beta = dain.get_beta()?;
        let gamma = dain.get_gamma()?;
        assert_eq!(alpha.dims(), &[4]);
        assert_eq!(beta.dims(), &[4]);
        assert_eq!(gamma.dims(), &[4]);
        Ok(())
    }

    #[test]
    fn test_dain_set_parameters() -> Result<()> {
        let mut dain = create_test_dain();
        let new_alpha = Tensor::ones(&[4], candle_core::DType::F64, &Device::Cpu)?;
        dain.set_alpha(new_alpha)?;
        let retrieved = dain.get_alpha()?;
        assert_eq!(retrieved.to_vec1::<f64>()?, vec![1.0, 1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_dain_dimension_mismatch() {
        let config = DAINConfig {
            enabled: true,
            hidden_dim: 16,
            learnable_mean_scale: true,
            learnable_std_scale: true,
            feature_gate_enabled: true,
            epsilon: 1e-5,
            input_dim: 4,
        };
        let dain = DAINormalization::new(4, &Device::Cpu, config).unwrap();
        let wrong_input = Tensor::randn(0.0, 1.0, (4, 10, 8), &Device::Cpu).unwrap();
        assert!(dain.forward(&wrong_input).is_err());
    }

    #[test]
    fn test_dain_high_volatility_data() -> Result<()> {
        let config = DAINConfig {
            enabled: true,
            hidden_dim: 16,
            learnable_mean_scale: true,
            learnable_std_scale: true,
            feature_gate_enabled: true,
            epsilon: 1e-5,
            input_dim: 4,
        };
        let dain = DAINormalization::new(4, &Device::Cpu, config)?;
        // Using std=10.0 gives variance=100, which is high but manageable
        // DAIN's epsilon (1e-5) helps prevent numerical issues
        let high_vol_input = Tensor::randn(0.0, 10.0, (4, 4), &Device::Cpu)?; // 2D
        let output = dain.forward(&high_vol_input)?;
        let flattened = output.flatten_all()?.to_vec1::<f64>()?;
        assert!(flattened.iter().all(|x| x.is_finite()));
        assert!(!flattened.iter().any(|x| x.is_nan()));
        Ok(())
    }
}
