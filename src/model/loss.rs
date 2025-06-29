//! Crypto-specific loss functions for VANGA LSTM
//!
//! Implements specialized loss functions designed for cryptocurrency forecasting,
//! including multi-objective, regime-aware, and risk-adjusted loss calculations.

use crate::optimization::objective::MarketRegime;
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Crypto-specific loss functions for LSTM training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CryptoLossFunction {
    /// Multi-objective loss balancing accuracy across different prediction horizons
    MultiObjective { horizon_weights: Vec<f64> },
    /// Regime-aware loss that adjusts based on market volatility conditions
    RegimeAware { volatility_penalty: f64 },
    /// Risk-adjusted loss incorporating Sharpe ratio and maximum drawdown
    RiskAdjusted {
        sharpe_weight: f64,
        drawdown_weight: f64,
    },
    /// Composite crypto loss combining multiple factors
    CryptoComposite {
        accuracy_weight: f64,
        direction_weight: f64,
        volatility_weight: f64,
        risk_weight: f64,
    },
    /// Directional accuracy focused loss
    DirectionalFocused { direction_penalty: f64 },
    /// Volatility-aware loss that penalizes predictions during high volatility
    VolatilityAware {
        volatility_threshold: f64,
        penalty_factor: f64,
    },
}

impl CryptoLossFunction {
    /// Calculate loss value for given predictions and targets
    pub fn calculate_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        market_regime: MarketRegime,
    ) -> Result<f64> {
        match self {
            CryptoLossFunction::MultiObjective { horizon_weights } => {
                self.calculate_multi_objective_loss(predictions, targets, horizon_weights)
            }
            CryptoLossFunction::RegimeAware { volatility_penalty } => self
                .calculate_regime_aware_loss(
                    predictions,
                    targets,
                    market_regime,
                    *volatility_penalty,
                ),
            CryptoLossFunction::RiskAdjusted {
                sharpe_weight,
                drawdown_weight,
            } => self.calculate_risk_adjusted_loss(
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
            } => self.calculate_crypto_composite_loss(
                predictions,
                targets,
                market_regime,
                *accuracy_weight,
                *direction_weight,
                *volatility_weight,
                *risk_weight,
            ),
            CryptoLossFunction::DirectionalFocused { direction_penalty } => {
                self.calculate_directional_focused_loss(predictions, targets, *direction_penalty)
            }
            CryptoLossFunction::VolatilityAware {
                volatility_threshold,
                penalty_factor,
            } => self.calculate_volatility_aware_loss(
                predictions,
                targets,
                market_regime,
                *volatility_threshold,
                *penalty_factor,
            ),
        }
    }

    /// Multi-objective loss balancing different prediction horizons
    fn calculate_multi_objective_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        horizon_weights: &[f64],
    ) -> Result<f64> {
        if predictions.shape() != targets.shape() {
            return Err(VangaError::DataError(
                "Predictions and targets shape mismatch".to_string(),
            ));
        }

        let num_horizons = predictions.ncols();
        let weights = if horizon_weights.len() == num_horizons {
            horizon_weights
        } else {
            // Default equal weights if mismatch
            &vec![1.0 / num_horizons as f64; num_horizons]
        };

        let mut total_loss = 0.0;
        let mut total_weight = 0.0;

        for (horizon_idx, &weight) in weights.iter().enumerate() {
            if horizon_idx >= num_horizons {
                break;
            }

            // Calculate MSE for this horizon
            let mut horizon_loss = 0.0;
            for row_idx in 0..predictions.nrows() {
                let pred = predictions[[row_idx, horizon_idx]];
                let target = targets[[row_idx, horizon_idx]];
                horizon_loss += (pred - target).powi(2);
            }
            horizon_loss /= predictions.nrows() as f64;

            total_loss += horizon_loss * weight;
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return Ok(0.0);
        }

        Ok(total_loss / total_weight)
    }

    /// Regime-aware loss that adjusts based on market conditions
    fn calculate_regime_aware_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        market_regime: MarketRegime,
        volatility_penalty: f64,
    ) -> Result<f64> {
        // Base MSE loss
        let base_loss = self.calculate_mse_loss(predictions, targets)?;

        // Regime-specific adjustments
        let regime_multiplier = match market_regime {
            MarketRegime::HighVolatility => 1.0 + volatility_penalty,
            MarketRegime::LowVolatility => 0.9, // Easier to predict, lower penalty
            MarketRegime::MediumVolatility => 1.0,
            MarketRegime::BullMarket => 1.1, // Slight penalty for trending markets
            MarketRegime::BearMarket => 1.1,
            MarketRegime::RangeBound => 1.2, // Higher penalty for range-bound (harder to predict)
        };

        Ok(base_loss * regime_multiplier)
    }

    /// Risk-adjusted loss incorporating trading metrics
    fn calculate_risk_adjusted_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        sharpe_weight: f64,
        drawdown_weight: f64,
    ) -> Result<f64> {
        // Base prediction accuracy loss
        let accuracy_loss = self.calculate_mse_loss(predictions, targets)?;

        // Calculate directional accuracy loss
        let direction_loss = self.calculate_directional_loss(predictions, targets)?;

        // Simulate trading performance for risk metrics
        let (sharpe_penalty, drawdown_penalty) =
            self.calculate_risk_penalties(predictions, targets)?;

        // Combine losses
        let total_loss = accuracy_loss * 0.4
            + direction_loss * 0.2
            + sharpe_penalty * sharpe_weight
            + drawdown_penalty * drawdown_weight;

        Ok(total_loss)
    }

    /// Comprehensive crypto-specific composite loss
    fn calculate_crypto_composite_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        market_regime: MarketRegime,
        accuracy_weight: f64,
        direction_weight: f64,
        volatility_weight: f64,
        risk_weight: f64,
    ) -> Result<f64> {
        // Component losses
        let accuracy_loss = self.calculate_mse_loss(predictions, targets)?;
        let direction_loss = self.calculate_directional_loss(predictions, targets)?;
        let volatility_loss = self.calculate_volatility_prediction_loss(predictions, targets)?;
        let (sharpe_penalty, drawdown_penalty) =
            self.calculate_risk_penalties(predictions, targets)?;
        let risk_loss = (sharpe_penalty + drawdown_penalty) / 2.0;

        // Base composite loss
        let base_loss = accuracy_loss * accuracy_weight
            + direction_loss * direction_weight
            + volatility_loss * volatility_weight
            + risk_loss * risk_weight;

        // Market regime adjustments
        let regime_adjustment = match market_regime {
            MarketRegime::HighVolatility => 1.15, // Higher penalty in volatile markets
            MarketRegime::LowVolatility => 0.95,  // Lower penalty in stable markets
            MarketRegime::BullMarket => 1.05,     // Slight penalty in trending markets
            MarketRegime::BearMarket => 1.05,
            MarketRegime::MediumVolatility => 1.0,
            MarketRegime::RangeBound => 1.1, // Higher penalty for sideways markets
        };

        Ok(base_loss * regime_adjustment)
    }

    /// Directional accuracy focused loss
    fn calculate_directional_focused_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        direction_penalty: f64,
    ) -> Result<f64> {
        let base_loss = self.calculate_mse_loss(predictions, targets)?;
        let direction_loss = self.calculate_directional_loss(predictions, targets)?;

        // Heavily weight directional accuracy
        Ok(base_loss * 0.3 + direction_loss * direction_penalty)
    }

    /// Volatility-aware loss with dynamic penalties
    fn calculate_volatility_aware_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        market_regime: MarketRegime,
        volatility_threshold: f64,
        penalty_factor: f64,
    ) -> Result<f64> {
        let base_loss = self.calculate_mse_loss(predictions, targets)?;

        // Apply volatility penalty based on regime
        let volatility_penalty = match market_regime {
            MarketRegime::HighVolatility => penalty_factor,
            MarketRegime::MediumVolatility => penalty_factor * 0.5,
            _ => 0.0,
        };

        Ok(base_loss * (1.0 + volatility_penalty))
    }

    /// Helper: Calculate basic MSE loss
    fn calculate_mse_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != targets.shape() {
            return Err(VangaError::DataError(
                "Predictions and targets shape mismatch".to_string(),
            ));
        }

        let mse = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| (pred - target).powi(2))
            .sum::<f64>()
            / (predictions.len() as f64);

        Ok(mse)
    }

    /// Helper: Calculate directional accuracy loss
    fn calculate_directional_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<f64> {
        if predictions.nrows() < 2 {
            return Ok(0.0);
        }

        let mut incorrect_directions = 0;
        let mut total_comparisons = 0;

        for i in 1..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred_direction = predictions[[i, j]] > predictions[[i - 1, j]];
                let target_direction = targets[[i, j]] > targets[[i - 1, j]];

                if pred_direction != target_direction {
                    incorrect_directions += 1;
                }
                total_comparisons += 1;
            }
        }

        if total_comparisons == 0 {
            return Ok(0.0);
        }

        Ok(incorrect_directions as f64 / total_comparisons as f64)
    }

    /// Helper: Calculate volatility prediction loss
    fn calculate_volatility_prediction_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<f64> {
        // Calculate volatility of predictions vs targets
        let pred_volatility = self.calculate_array_volatility(predictions);
        let target_volatility = self.calculate_array_volatility(targets);

        // Penalize volatility mismatch
        Ok((pred_volatility - target_volatility).abs())
    }

    /// Helper: Calculate risk penalties (Sharpe and Drawdown)
    fn calculate_risk_penalties(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<(f64, f64)> {
        // Simulate returns from predictions
        let pred_returns = self.simulate_returns_from_predictions(predictions)?;
        let target_returns = self.simulate_returns_from_targets(targets)?;

        // Calculate Sharpe ratio penalty
        let pred_sharpe = self.calculate_sharpe_ratio(&pred_returns);
        let target_sharpe = self.calculate_sharpe_ratio(&target_returns);
        let sharpe_penalty = (target_sharpe - pred_sharpe).max(0.0);

        // Calculate drawdown penalty
        let pred_drawdown = self.calculate_max_drawdown(&pred_returns);
        let target_drawdown = self.calculate_max_drawdown(&target_returns);
        let drawdown_penalty = (pred_drawdown - target_drawdown).max(0.0);

        Ok((sharpe_penalty, drawdown_penalty))
    }

    /// Helper: Calculate volatility of an array
    fn calculate_array_volatility(&self, array: &Array2<f64>) -> f64 {
        if array.is_empty() {
            return 0.0;
        }

        let mean = array.sum() / array.len() as f64;
        let variance = array.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / array.len() as f64;

        variance.sqrt()
    }

    /// Helper: Simulate returns from predictions
    fn simulate_returns_from_predictions(&self, predictions: &Array2<f64>) -> Result<Vec<f64>> {
        if predictions.nrows() < 2 {
            return Ok(Vec::new());
        }

        let mut returns = Vec::new();
        for i in 1..predictions.nrows() {
            // Use first column as primary prediction
            if predictions.ncols() > 0 {
                let return_val = predictions[[i, 0]] - predictions[[i - 1, 0]];
                returns.push(return_val);
            }
        }

        Ok(returns)
    }

    /// Helper: Simulate returns from targets
    fn simulate_returns_from_targets(&self, targets: &Array2<f64>) -> Result<Vec<f64>> {
        if targets.nrows() < 2 {
            return Ok(Vec::new());
        }

        let mut returns = Vec::new();
        for i in 1..targets.nrows() {
            if targets.ncols() > 0 {
                let return_val = targets[[i, 0]] - targets[[i - 1, 0]];
                returns.push(return_val);
            }
        }

        Ok(returns)
    }

    /// Helper: Calculate Sharpe ratio
    fn calculate_sharpe_ratio(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            0.0
        } else {
            mean_return / std_dev
        }
    }

    /// Helper: Calculate maximum drawdown
    fn calculate_max_drawdown(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut cumulative_returns = vec![1.0];
        for &ret in returns {
            let new_value = cumulative_returns.last().unwrap() * (1.0 + ret);
            cumulative_returns.push(new_value);
        }

        let mut max_drawdown = 0.0;
        let mut peak = cumulative_returns[0];

        for &value in &cumulative_returns {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }
}

impl Default for CryptoLossFunction {
    fn default() -> Self {
        CryptoLossFunction::CryptoComposite {
            accuracy_weight: 0.3,
            direction_weight: 0.3,
            volatility_weight: 0.2,
            risk_weight: 0.2,
        }
    }
}

/// Loss function factory for creating crypto-specific loss functions
pub struct CryptoLossFunctionFactory;

impl CryptoLossFunctionFactory {
    /// Create loss function optimized for small datasets
    pub fn for_small_dataset() -> CryptoLossFunction {
        CryptoLossFunction::DirectionalFocused {
            direction_penalty: 0.7,
        }
    }

    /// Create loss function optimized for medium datasets
    pub fn for_medium_dataset() -> CryptoLossFunction {
        CryptoLossFunction::CryptoComposite {
            accuracy_weight: 0.4,
            direction_weight: 0.3,
            volatility_weight: 0.2,
            risk_weight: 0.1,
        }
    }

    /// Create loss function optimized for large datasets
    pub fn for_large_dataset() -> CryptoLossFunction {
        CryptoLossFunction::RiskAdjusted {
            sharpe_weight: 0.3,
            drawdown_weight: 0.3,
        }
    }

    /// Create loss function for high volatility markets
    pub fn for_high_volatility() -> CryptoLossFunction {
        CryptoLossFunction::VolatilityAware {
            volatility_threshold: 0.05,
            penalty_factor: 0.5,
        }
    }

    /// Create multi-horizon loss function
    pub fn for_multi_horizon(horizon_weights: Vec<f64>) -> CryptoLossFunction {
        CryptoLossFunction::MultiObjective { horizon_weights }
    }
}
