//! Tensor-based crypto loss functions that maintain gradient flow
//!
//! This module implements CryptoLossFunction variants using native Candle tensor operations
//! to preserve gradients during backpropagation, unlike the array-based implementation.

use crate::model::loss::CryptoLossFunction;
use crate::optimization::objective::MarketRegime;
use crate::utils::error::{Result, VangaError};
use candle_core::Tensor;

/// Cached weight tensors for performance optimization
#[derive(Debug, Clone)]
struct CachedWeightTensors {
    accuracy: Tensor,
    direction: Tensor,
    volatility: Tensor,
    risk: Tensor,
    regime_multiplier: Tensor,
}

impl CachedWeightTensors {
    /// Create cached weight tensors for given device
    fn new(
        accuracy_weight: f64,
        direction_weight: f64,
        volatility_weight: f64,
        risk_weight: f64,
        regime_multiplier: f64,
        device: &candle_core::Device,
    ) -> Result<Self> {
        Ok(Self {
            accuracy: Tensor::new(accuracy_weight as f32, device).map_err(|e| {
                VangaError::ModelError(format!("Failed to create accuracy tensor: {}", e))
            })?,
            direction: Tensor::new(direction_weight as f32, device).map_err(|e| {
                VangaError::ModelError(format!("Failed to create direction tensor: {}", e))
            })?,
            volatility: Tensor::new(volatility_weight as f32, device).map_err(|e| {
                VangaError::ModelError(format!("Failed to create volatility tensor: {}", e))
            })?,
            risk: Tensor::new(risk_weight as f32, device).map_err(|e| {
                VangaError::ModelError(format!("Failed to create risk tensor: {}", e))
            })?,
            regime_multiplier: Tensor::new(regime_multiplier as f32, device).map_err(|e| {
                VangaError::ModelError(format!("Failed to create regime tensor: {}", e))
            })?,
        })
    }
}

/// Configuration for crypto composite loss weights
#[derive(Debug, Clone, PartialEq)]
struct CryptoCompositeConfig {
    accuracy_weight: f64,
    direction_weight: f64,
    volatility_weight: f64,
    risk_weight: f64,
}

/// Tensor-based implementation of crypto loss functions with caching
pub struct TensorCryptoLossFunction {
    loss_type: CryptoLossFunction,
    cached_weights: Option<CachedWeightTensors>,
    last_config: Option<CryptoCompositeConfig>,
}

impl TensorCryptoLossFunction {
    /// Create new tensor-based crypto loss function
    pub fn new(loss_type: CryptoLossFunction) -> Self {
        Self {
            loss_type,
            cached_weights: None,
            last_config: None,
        }
    }

    /// Calculate loss using tensor operations that maintain gradients
    pub fn calculate_tensor_loss(
        &mut self,
        predictions: &Tensor,
        targets: &Tensor,
        market_regime: MarketRegime,
    ) -> Result<Tensor> {
        match &self.loss_type {
            CryptoLossFunction::MultiObjective { horizon_weights } => {
                self.calculate_multi_objective_tensor_loss(predictions, targets, horizon_weights)
            }
            CryptoLossFunction::RegimeAware { volatility_penalty } => self
                .calculate_regime_aware_tensor_loss(
                    predictions,
                    targets,
                    *volatility_penalty,
                    market_regime,
                ),
            CryptoLossFunction::RiskAdjusted {
                sharpe_weight,
                drawdown_weight,
            } => self.calculate_risk_adjusted_tensor_loss(
                predictions,
                targets,
                *sharpe_weight,
                *drawdown_weight,
            ),
            CryptoLossFunction::CryptoComposite {
                accuracy_weight,
                direction_weight,
                volatility_weight,
                risk_weight,
            } => {
                let composite_config = CryptoCompositeConfig {
                    accuracy_weight: *accuracy_weight,
                    direction_weight: *direction_weight,
                    volatility_weight: *volatility_weight,
                    risk_weight: *risk_weight,
                };
                self.calculate_crypto_composite_tensor_loss(
                    predictions,
                    targets,
                    &composite_config,
                    market_regime,
                )
            }
            CryptoLossFunction::DirectionalFocused { direction_penalty } => self
                .calculate_directional_focused_tensor_loss(
                    predictions,
                    targets,
                    *direction_penalty,
                ),
            CryptoLossFunction::VolatilityAware {
                volatility_threshold,
                penalty_factor,
            } => self.calculate_volatility_aware_tensor_loss(
                predictions,
                targets,
                *volatility_threshold,
                *penalty_factor,
            ),
        }
    }

    /// Base MSE loss using tensor operations
    fn calculate_mse_tensor_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        predictions
            .sub(targets)?
            .contiguous()?
            .sqr()?
            .mean_all()
            .map_err(|e| {
                VangaError::ModelError(format!("MSE tensor loss calculation failed: {}", e))
            })
    }

    /// Directional accuracy loss using tensor operations
    fn calculate_directional_tensor_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor> {
        // Calculate price changes (differences between consecutive predictions/targets)
        let pred_shape = predictions.shape();
        if pred_shape.dims()[0] < 2 {
            // Not enough data for directional comparison, return zero loss
            return Tensor::zeros_like(predictions)?.mean_all().map_err(|e| {
                VangaError::ModelError(format!("Directional loss zero tensor failed: {}", e))
            });
        }

        // FIXED: Use proper tensor operations for directional calculation
        let pred_diff = self.calculate_tensor_diff(predictions)?;
        let target_diff = self.calculate_tensor_diff(targets)?;

        // FIXED: Calculate directional agreement using sign-based comparison
        // Get signs: -1 for negative, 0 for zero, +1 for positive
        let pred_signs = pred_diff.sign()?;
        let target_signs = target_diff.sign()?;

        // Calculate agreement: same sign = agreement (positive product)
        let sign_product = pred_signs.mul(&target_signs)?;

        // Agreement when product > 0 (same signs), disagreement when product <= 0
        let zero_tensor = Tensor::zeros_like(&sign_product)?;
        let agreement_mask = sign_product.gt(&zero_tensor)?;

        // FIXED: Convert boolean mask to f32 properly for arithmetic
        let agreement_f32 = agreement_mask
            .to_dtype(candle_core::DType::F32)?
            .contiguous()?;

        // Calculate directional accuracy (proportion of agreements)
        let directional_accuracy = agreement_f32.mean_all()?;

        // FIXED: Return directional loss as (1 - accuracy) for proper loss semantics
        let one_scalar = Tensor::new(1.0f32, predictions.device())?;

        one_scalar.sub(&directional_accuracy).map_err(|e| {
            VangaError::ModelError(format!("Directional tensor loss calculation failed: {}", e))
        })
    }

    /// Volatility penalty using tensor operations
    fn calculate_volatility_tensor_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor> {
        // Calculate prediction volatility (standard deviation)
        let pred_mean = predictions.mean_all()?;
        let pred_mean_broadcast = pred_mean.broadcast_as(predictions.shape())?;
        let pred_variance = predictions
            .sub(&pred_mean_broadcast)?
            .contiguous()?
            .sqr()?
            .mean_all()?;
        let pred_volatility = pred_variance.sqrt()?;

        // Calculate target volatility
        let target_mean = targets.mean_all()?;
        let target_mean_broadcast = target_mean.broadcast_as(targets.shape())?;
        let target_variance = targets
            .sub(&target_mean_broadcast)?
            .contiguous()?
            .sqr()?
            .mean_all()?;
        let target_volatility = target_variance.sqrt()?;

        // Volatility difference as penalty
        pred_volatility.sub(&target_volatility)?.abs().map_err(|e| {
            VangaError::ModelError(format!("Volatility tensor loss calculation failed: {}", e))
        })
    }

    /// Efficient tensor difference calculation
    fn calculate_tensor_diff(&self, tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.shape();
        let len = shape.dims()[0];

        let slice1 = tensor.narrow(0, 0, len - 1)?;
        let slice2 = tensor.narrow(0, 1, len - 1)?;

        slice2
            .sub(&slice1)?
            .contiguous()
            .map_err(|e| VangaError::ModelError(format!("Tensor diff calculation failed: {}", e)))
    }

    /// Multi-objective loss with horizon weighting
    fn calculate_multi_objective_tensor_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        horizon_weights: &[f64],
    ) -> Result<Tensor> {
        let base_loss = self.calculate_mse_tensor_loss(predictions, targets)?;

        // Apply horizon weighting (simplified - use first weight for now)
        let weight = horizon_weights.first().unwrap_or(&1.0);
        let weight_tensor = Tensor::new(*weight as f32, predictions.device())?;

        base_loss.mul(&weight_tensor).map_err(|e| {
            VangaError::ModelError(format!("Multi-objective tensor loss failed: {}", e))
        })
    }

    /// Regime-aware loss with market condition adjustment
    fn calculate_regime_aware_tensor_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        volatility_penalty: f64,
        market_regime: MarketRegime,
    ) -> Result<Tensor> {
        let base_loss = self.calculate_mse_tensor_loss(predictions, targets)?;
        let volatility_loss = self.calculate_volatility_tensor_loss(predictions, targets)?;

        // Market regime multiplier
        let regime_multiplier = match market_regime {
            MarketRegime::LowVolatility => 0.8,
            MarketRegime::MediumVolatility => 1.0,
            MarketRegime::HighVolatility => 1.3,
            MarketRegime::BullMarket => 1.1,
            MarketRegime::BearMarket => 1.2,
            MarketRegime::RangeBound => 0.9,
        };

        let regime_tensor = Tensor::new(regime_multiplier as f32, predictions.device())?;
        let penalty_tensor = Tensor::new(volatility_penalty as f32, predictions.device())?;

        // Combined loss: base_loss * regime_multiplier + volatility_penalty * volatility_loss
        base_loss
            .mul(&regime_tensor)?
            .add(&volatility_loss.mul(&penalty_tensor)?)
            .map_err(|e| VangaError::ModelError(format!("Regime-aware tensor loss failed: {}", e)))
    }

    /// Risk-adjusted loss incorporating trading metrics with normalization
    fn calculate_risk_adjusted_tensor_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        sharpe_weight: f64,
        drawdown_weight: f64,
    ) -> Result<Tensor> {
        let base_mse_loss = self.calculate_mse_tensor_loss(predictions, targets)?;
        let volatility_loss = self.calculate_volatility_tensor_loss(predictions, targets)?;

        // FIXED: Normalize volatility loss to MSE scale for risk adjustment
        let epsilon = Tensor::new(1e-8f32, predictions.device())?;
        let mse_scale = base_mse_loss.add(&epsilon)?;
        let normalized_volatility = volatility_loss
            .div(&volatility_loss.add(&epsilon)?)?
            .mul(&mse_scale)?;

        // Simplified risk adjustment using normalized volatility as proxy
        let sharpe_tensor = Tensor::new(sharpe_weight as f32, predictions.device())?;
        let drawdown_tensor = Tensor::new(drawdown_weight as f32, predictions.device())?;

        // Risk-adjusted loss: base_mse + normalized_risk_penalties
        let risk_penalty = normalized_volatility
            .mul(&sharpe_tensor)?
            .add(&normalized_volatility.mul(&drawdown_tensor)?)?;

        base_mse_loss
            .add(&risk_penalty)
            .map_err(|e| VangaError::ModelError(format!("Risk-adjusted tensor loss failed: {}", e)))
    }

    /// Get or create cached weight tensors for composite loss
    fn get_cached_weights(
        &mut self,
        config: &CryptoCompositeConfig,
        regime_multiplier: f64,
        device: &candle_core::Device,
    ) -> Result<&CachedWeightTensors> {
        // Check if we need to update cache
        let needs_update =
            self.last_config.as_ref() != Some(config) || self.cached_weights.is_none();

        if needs_update {
            self.cached_weights = Some(CachedWeightTensors::new(
                config.accuracy_weight,
                config.direction_weight,
                config.volatility_weight,
                config.risk_weight,
                regime_multiplier,
                device,
            )?);
            self.last_config = Some(config.clone());
        }

        Ok(self.cached_weights.as_ref().unwrap())
    }

    /// Get regime multiplier for market conditions
    fn get_regime_multiplier(&self, market_regime: MarketRegime) -> f64 {
        match market_regime {
            MarketRegime::LowVolatility => 0.9,
            MarketRegime::MediumVolatility => 1.0,
            MarketRegime::HighVolatility => 1.2,
            MarketRegime::BullMarket => 1.1,
            MarketRegime::BearMarket => 1.3,
            MarketRegime::RangeBound => 0.8,
        }
    }

    /// Crypto composite loss combining multiple factors with caching and normalization
    fn calculate_crypto_composite_tensor_loss(
        &mut self,
        predictions: &Tensor,
        targets: &Tensor,
        config: &CryptoCompositeConfig,
        market_regime: MarketRegime,
    ) -> Result<Tensor> {
        // FIXED: Calculate base MSE loss first for normalization reference
        let base_mse_loss = self.calculate_mse_tensor_loss(predictions, targets)?;

        // Calculate all needed loss components conditionally
        let directional_loss = if config.direction_weight > 0.0 {
            Some(self.calculate_directional_tensor_loss(predictions, targets)?)
        } else {
            None
        };

        let volatility_loss = if config.volatility_weight > 0.0 || config.risk_weight > 0.0 {
            Some(self.calculate_volatility_tensor_loss(predictions, targets)?)
        } else {
            None
        };

        // FIXED: Normalize all loss components relative to MSE scale
        let epsilon = Tensor::new(1e-8f32, predictions.device())?; // Prevent division by zero
        let mse_scale = base_mse_loss.add(&epsilon)?; // Use MSE as reference scale

        // Normalize directional loss to MSE scale
        let normalized_directional_loss = if let Some(dir_loss) = directional_loss {
            // Directional loss is 0-1 range, scale it to MSE magnitude
            Some(dir_loss.mul(&mse_scale)?)
        } else {
            None
        };

        // Normalize volatility loss to MSE scale
        let normalized_volatility_loss = if let Some(vol_loss) = volatility_loss {
            // Volatility loss can be large, normalize it relative to MSE
            let vol_normalized = vol_loss.div(&vol_loss.add(&epsilon)?)?.mul(&mse_scale)?;
            Some(vol_normalized)
        } else {
            None
        };

        // Get regime multiplier and cached weights
        let regime_multiplier = self.get_regime_multiplier(market_regime);
        let weights = self.get_cached_weights(config, regime_multiplier, predictions.device())?;

        // FIXED: Combine normalized components with proper weighting
        let mut total_loss = base_mse_loss.mul(&weights.accuracy)?;

        if let Some(norm_directional) = normalized_directional_loss {
            total_loss = total_loss.add(&norm_directional.mul(&weights.direction)?)?;
        }

        if let Some(norm_volatility) = normalized_volatility_loss {
            if config.volatility_weight > 0.0 {
                total_loss = total_loss.add(&norm_volatility.mul(&weights.volatility)?)?;
            }
            if config.risk_weight > 0.0 {
                total_loss = total_loss.add(&norm_volatility.mul(&weights.risk)?)?;
            }
        }

        // FIXED: Apply regime adjustment as multiplier, not additive
        total_loss.mul(&weights.regime_multiplier).map_err(|e| {
            VangaError::ModelError(format!("Crypto composite tensor loss failed: {}", e))
        })
    }

    /// Directional focused loss with normalization
    fn calculate_directional_focused_tensor_loss(
        &mut self,
        predictions: &Tensor,
        targets: &Tensor,
        direction_penalty: f64,
    ) -> Result<Tensor> {
        let base_mse_loss = self.calculate_mse_tensor_loss(predictions, targets)?;
        let directional_loss = self.calculate_directional_tensor_loss(predictions, targets)?;

        // FIXED: Normalize directional loss to MSE scale
        let epsilon = Tensor::new(1e-8f32, predictions.device())?;
        let mse_scale = base_mse_loss.add(&epsilon)?;
        let normalized_directional = directional_loss.mul(&mse_scale)?;

        let base_weight = Tensor::new(0.3f32, predictions.device())?;
        let direction_weight = Tensor::new(direction_penalty as f32, predictions.device())?;

        base_mse_loss
            .mul(&base_weight)?
            .add(&normalized_directional.mul(&direction_weight)?)
            .map_err(|e| {
                VangaError::ModelError(format!("Directional focused tensor loss failed: {}", e))
            })
    }

    /// Volatility-aware loss with dynamic penalties
    fn calculate_volatility_aware_tensor_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        volatility_threshold: f64,
        penalty_factor: f64,
    ) -> Result<Tensor> {
        let base_loss = self.calculate_mse_tensor_loss(predictions, targets)?;
        let volatility_loss = self.calculate_volatility_tensor_loss(predictions, targets)?;

        // Apply penalty if volatility exceeds threshold
        let threshold_tensor = Tensor::new(volatility_threshold as f32, predictions.device())?;
        let penalty_tensor = Tensor::new(penalty_factor as f32, predictions.device())?;
        let one_tensor = Tensor::ones_like(&base_loss)?;

        // Penalty = 1 + penalty_factor if volatility > threshold, else 1
        let volatility_penalty = volatility_loss
            .gt(&threshold_tensor)?
            .contiguous()?
            .where_cond(&one_tensor.add(&penalty_tensor)?, &one_tensor)?
            .contiguous()?;

        base_loss.mul(&volatility_penalty).map_err(|e| {
            VangaError::ModelError(format!("Volatility-aware tensor loss failed: {}", e))
        })
    }
}
