//! Tensor-based crypto loss functions that maintain gradient flow
//!
//! This module implements CryptoLossFunction variants using native Candle tensor operations
//! to preserve gradients during backpropagation, unlike the array-based implementation.

use crate::model::loss::CryptoLossFunction;
use crate::optimization::objective::MarketRegime;
use crate::utils::error::{Result, VangaError};
use candle_core::Tensor;

/// Weight configuration for crypto composite loss function
#[derive(Debug, Clone)]
struct CryptoCompositeWeights {
    accuracy_weight: f64,
    direction_weight: f64,
    volatility_weight: f64,
    risk_weight: f64,
}

/// Tensor-based implementation of crypto loss functions
pub struct TensorCryptoLossFunction {
    loss_type: CryptoLossFunction,
}

impl TensorCryptoLossFunction {
    /// Create new tensor-based crypto loss function
    pub fn new(loss_type: CryptoLossFunction) -> Self {
        Self { loss_type }
    }

    /// Calculate loss using tensor operations that maintain gradients
    pub fn calculate_tensor_loss(
        &self,
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
                let weights = CryptoCompositeWeights {
                    accuracy_weight: *accuracy_weight,
                    direction_weight: *direction_weight,
                    volatility_weight: *volatility_weight,
                    risk_weight: *risk_weight,
                };
                self.calculate_crypto_composite_tensor_loss(
                    predictions,
                    targets,
                    &weights,
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
        predictions.sub(targets)?.sqr()?.mean_all().map_err(|e| {
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

        // Get consecutive differences for direction calculation
        let pred_slice1 = predictions.narrow(0, 0, pred_shape.dims()[0] - 1)?;
        let pred_slice2 = predictions.narrow(0, 1, pred_shape.dims()[0] - 1)?;
        let target_slice1 = targets.narrow(0, 0, pred_shape.dims()[0] - 1)?;
        let target_slice2 = targets.narrow(0, 1, pred_shape.dims()[0] - 1)?;

        let pred_direction = pred_slice2.sub(&pred_slice1)?;
        let target_direction = target_slice2.sub(&target_slice1)?;

        // Calculate directional agreement using sign comparison
        // If both positive or both negative, agreement = 1, otherwise 0
        let pred_positive = pred_direction.gt(&Tensor::zeros_like(&pred_direction)?)?;
        let target_positive = target_direction.gt(&Tensor::zeros_like(&target_direction)?)?;

        // Agreement when both positive or both negative
        let both_positive = pred_positive.mul(&target_positive)?;
        let both_negative = pred_positive
            .neg()?
            .add(&Tensor::ones_like(&pred_positive)?)?
            .mul(
                &target_positive
                    .neg()?
                    .add(&Tensor::ones_like(&target_positive)?)?,
            )?;
        let agreement = both_positive.add(&both_negative)?;

        // Directional loss = 1 - agreement_rate
        let one_tensor = Tensor::ones_like(&agreement)?;
        let directional_accuracy = agreement.mean_all()?;

        one_tensor.sub(&directional_accuracy).map_err(|e| {
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
        let pred_variance = predictions.sub(&pred_mean)?.sqr()?.mean_all()?;
        let pred_volatility = pred_variance.sqrt()?;

        // Calculate target volatility
        let target_mean = targets.mean_all()?;
        let target_variance = targets.sub(&target_mean)?.sqr()?.mean_all()?;
        let target_volatility = target_variance.sqrt()?;

        // Volatility difference as penalty
        pred_volatility.sub(&target_volatility)?.abs().map_err(|e| {
            VangaError::ModelError(format!("Volatility tensor loss calculation failed: {}", e))
        })
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

    /// Risk-adjusted loss incorporating trading metrics
    fn calculate_risk_adjusted_tensor_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        sharpe_weight: f64,
        drawdown_weight: f64,
    ) -> Result<Tensor> {
        let base_loss = self.calculate_mse_tensor_loss(predictions, targets)?;
        let volatility_loss = self.calculate_volatility_tensor_loss(predictions, targets)?;

        // Simplified risk adjustment using volatility as proxy
        let sharpe_tensor = Tensor::new(sharpe_weight as f32, predictions.device())?;
        let drawdown_tensor = Tensor::new(drawdown_weight as f32, predictions.device())?;

        // Risk-adjusted loss: base_loss + sharpe_weight * volatility + drawdown_weight * volatility
        let risk_penalty = volatility_loss
            .mul(&sharpe_tensor)?
            .add(&volatility_loss.mul(&drawdown_tensor)?)?;

        base_loss
            .add(&risk_penalty)
            .map_err(|e| VangaError::ModelError(format!("Risk-adjusted tensor loss failed: {}", e)))
    }

    /// Crypto composite loss combining multiple factors
    fn calculate_crypto_composite_tensor_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        weights: &CryptoCompositeWeights,
        market_regime: MarketRegime,
    ) -> Result<Tensor> {
        // Calculate individual loss components
        let mse_loss = self.calculate_mse_tensor_loss(predictions, targets)?;
        let directional_loss = self.calculate_directional_tensor_loss(predictions, targets)?;
        let volatility_loss = self.calculate_volatility_tensor_loss(predictions, targets)?;

        // Create weight tensors
        let accuracy_tensor = Tensor::new(weights.accuracy_weight as f32, predictions.device())?;
        let direction_tensor = Tensor::new(weights.direction_weight as f32, predictions.device())?;
        let volatility_tensor =
            Tensor::new(weights.volatility_weight as f32, predictions.device())?;
        let risk_tensor = Tensor::new(weights.risk_weight as f32, predictions.device())?;

        // Market regime adjustment
        let regime_multiplier = match market_regime {
            MarketRegime::LowVolatility => 0.9,
            MarketRegime::MediumVolatility => 1.0,
            MarketRegime::HighVolatility => 1.2,
            MarketRegime::BullMarket => 1.1,
            MarketRegime::BearMarket => 1.3,
            MarketRegime::RangeBound => 0.8,
        };
        let regime_tensor = Tensor::new(regime_multiplier as f32, predictions.device())?;

        // Weighted combination
        let weighted_loss = mse_loss
            .mul(&accuracy_tensor)?
            .add(&directional_loss.mul(&direction_tensor)?)?
            .add(&volatility_loss.mul(&volatility_tensor)?)?
            .add(&volatility_loss.mul(&risk_tensor)?)?; // Use volatility as risk proxy

        // Apply regime adjustment
        weighted_loss.mul(&regime_tensor).map_err(|e| {
            VangaError::ModelError(format!("Crypto composite tensor loss failed: {}", e))
        })
    }

    /// Directional focused loss
    fn calculate_directional_focused_tensor_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        direction_penalty: f64,
    ) -> Result<Tensor> {
        let base_loss = self.calculate_mse_tensor_loss(predictions, targets)?;
        let directional_loss = self.calculate_directional_tensor_loss(predictions, targets)?;

        let base_weight = Tensor::new(0.3f32, predictions.device())?;
        let direction_weight = Tensor::new(direction_penalty as f32, predictions.device())?;

        base_loss
            .mul(&base_weight)?
            .add(&directional_loss.mul(&direction_weight)?)
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
            .where_cond(&one_tensor.add(&penalty_tensor)?, &one_tensor)?;

        base_loss.mul(&volatility_penalty).map_err(|e| {
            VangaError::ModelError(format!("Volatility-aware tensor loss failed: {}", e))
        })
    }
}
