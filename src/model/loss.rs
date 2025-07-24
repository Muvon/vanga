//! Crypto-specific loss functions for VANGA LSTM
//!
//! Implements specialized loss functions designed for cryptocurrency forecasting,
//! including multi-objective, regime-aware, and risk-adjusted loss calculations.

use crate::optimization::objective::MarketRegime;
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Crypto-specific loss functions for LSTM training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CryptoLossFunction {
    /// Standard Mean Squared Error loss
    MSE,
    /// Multi-objective loss balancing accuracy across different prediction horizons
    MultiObjective { horizon_weights: Vec<f64> },
    /// Regime-aware loss that adjusts based on market volatility conditions
    RegimeAware { volatility_penalty: f64 },
    /// Risk-adjusted loss incorporating Sharpe ratio and maximum drawdown
    RiskAdjusted {
        sharpe_weight: f64,
        drawdown_weight: f64,
    },
    /// Composite loss combining multiple factors
    Composite {
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
            CryptoLossFunction::MSE => self.calculate_mse_loss(predictions, targets),
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
            CryptoLossFunction::Composite {
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
    #[allow(clippy::too_many_arguments)]
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

        // Calculate actual volatility from predictions (standard deviation)
        let pred_mean = predictions.mean().unwrap_or(0.0);
        let pred_variance = predictions
            .iter()
            .map(|&x| (x - pred_mean).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        let actual_volatility = pred_variance.sqrt();

        // Apply volatility penalty based on regime and threshold
        let volatility_penalty = match market_regime {
            MarketRegime::HighVolatility => penalty_factor,
            MarketRegime::MediumVolatility => penalty_factor * 0.5,
            _ => 0.0,
        };

        // Additional penalty if actual volatility exceeds threshold
        let threshold_penalty = if actual_volatility > volatility_threshold {
            penalty_factor * (actual_volatility - volatility_threshold)
        } else {
            0.0
        };

        Ok(base_loss * (1.0 + volatility_penalty + threshold_penalty))
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
        CryptoLossFunction::Composite {
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
    /// Create loss function optimized for basic training
    pub fn for_basic_training() -> CryptoLossFunction {
        CryptoLossFunction::MSE
    }

    /// Create MSE loss function
    pub fn mse() -> CryptoLossFunction {
        CryptoLossFunction::MSE
    }

    /// Create loss function optimized for small datasets
    pub fn for_small_dataset() -> CryptoLossFunction {
        CryptoLossFunction::DirectionalFocused {
            direction_penalty: 0.7,
        }
    }

    /// Create loss function optimized for medium datasets
    pub fn for_medium_dataset() -> CryptoLossFunction {
        CryptoLossFunction::Composite {
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
// Tensor-based crypto loss functions that maintain gradient flow
//
// This module implements CryptoLossFunction variants using native Candle tensor operations
// to preserve gradients during backpropagation, unlike the array-based implementation.
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
    /// Global class weights for categorical targets (matches LSTM implementation)
    training_class_weights: Option<Vec<f32>>,
}

impl TensorCryptoLossFunction {
    /// Create new tensor-based crypto loss function
    pub fn new(loss_type: CryptoLossFunction) -> Self {
        Self {
            loss_type,
            cached_weights: None,
            last_config: None,
            training_class_weights: None,
        }
    }

    /// Create new tensor-based crypto loss function with global class weights
    pub fn new_with_class_weights(
        loss_type: CryptoLossFunction,
        class_weights: Option<Vec<f32>>,
    ) -> Self {
        Self {
            loss_type,
            cached_weights: None,
            last_config: None,
            training_class_weights: class_weights,
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
            CryptoLossFunction::MSE => self.calculate_mse_tensor_loss(predictions, targets),
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
            CryptoLossFunction::Composite {
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

    /// Detect if targets are categorical (class indices) or regression (continuous values)
    fn is_categorical_target(&self, predictions: &Tensor, targets: &Tensor) -> bool {
        // If predictions have more dimensions than targets, it's likely categorical
        // Categorical: predictions=[batch, num_classes], targets=[batch, 1] or [batch]
        // Regression: predictions=[batch, num_targets], targets=[batch, num_targets]

        let pred_dims = predictions.dims();
        let target_dims = targets.dims();

        // Check if predictions have more features than targets (logits vs class indices)
        if pred_dims.len() >= 2 && target_dims.len() >= 2 {
            pred_dims[pred_dims.len() - 1] > target_dims[target_dims.len() - 1]
        } else if pred_dims.len() >= 2 && target_dims.len() == 1 {
            // predictions=[batch, classes], targets=[batch] - definitely categorical
            true
        } else {
            // Default to regression for same-shape tensors
            false
        }
    }

    /// Calculate appropriate base loss based on target type
    fn calculate_base_tensor_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        if self.is_categorical_target(predictions, targets) {
            // Use stabilized CrossEntropy for categorical targets (matching LSTM implementation)
            log::debug!(
                "🎯 Using stabilized CrossEntropy base loss for categorical targets: pred {:?}, target {:?}",
                predictions.shape(),
                targets.shape()
            );

            // Apply label smoothing for stability (10% smoothing like LSTM implementation)
            let num_classes = predictions.dims()[predictions.dims().len() - 1];
            let smoothed_targets = self.apply_simple_label_smoothing(targets, num_classes, 0.1)?;

            log::debug!(
                "🔧 Applied label smoothing: {} classes, 10% smoothing",
                num_classes
            );

            // Use log_softmax for numerical stability
            let log_softmax =
                candle_nn::ops::log_softmax(predictions, candle_core::D::Minus1)?.contiguous()?;

            // Apply class weights if available (matching LSTM behavior)
            if let Some(ref class_weights) = self.training_class_weights {
                if class_weights.len() == num_classes {
                    log::debug!("🌍 Applying global class weights: {:?}", class_weights);

                    // Use the same weighted soft crossentropy as LSTM
                    self.calculate_weighted_soft_crossentropy_loss(
                        predictions,
                        &smoothed_targets,
                        class_weights,
                    )
                } else {
                    log::error!(
                        "🚨 CRITICAL: Class weights length {} doesn't match model output classes {}",
                        class_weights.len(),
                        num_classes
                    );
                    log::error!(
                        "🚨 This causes training/validation loss inconsistency! Model output_size must match target type."
                    );

                    // FIXED: Truncate or pad class weights to match model output size
                    let aligned_weights = if class_weights.len() > num_classes {
                        // Truncate weights to match model output
                        log::warn!(
                            "⚠️ Truncating class weights from {} to {} classes",
                            class_weights.len(),
                            num_classes
                        );
                        class_weights[..num_classes].to_vec()
                    } else {
                        // Pad weights with 1.0 for missing classes
                        log::warn!(
                            "⚠️ Padding class weights from {} to {} classes with weight 1.0",
                            class_weights.len(),
                            num_classes
                        );
                        let mut padded_weights = class_weights.clone();
                        padded_weights.resize(num_classes, 1.0);
                        padded_weights
                    };

                    log::debug!("🔧 Using aligned class weights: {:?}", aligned_weights);

                    // Use aligned weights for consistent loss calculation
                    self.calculate_weighted_soft_crossentropy_loss(
                        predictions,
                        &smoothed_targets,
                        &aligned_weights,
                    )
                }
            } else {
                log::debug!("📊 No class weights available, using unweighted CrossEntropy");
                // Calculate soft CrossEntropy with smoothed targets
                let cross_entropy = smoothed_targets
                    .mul(&log_softmax)?
                    .sum(candle_core::D::Minus1)?
                    .neg()?
                    .mean_all()?;
                Ok(cross_entropy)
            }
        } else {
            // Use MSE for regression targets
            log::debug!(
                "📊 Using MSE base loss for regression targets: pred {:?}, target {:?}",
                predictions.shape(),
                targets.shape()
            );
            self.calculate_mse_tensor_loss(predictions, targets)
        }
    }

    /// Simple label smoothing implementation for stability
    fn apply_simple_label_smoothing(
        &self,
        targets: &Tensor,
        num_classes: usize,
        smoothing: f32,
    ) -> Result<Tensor> {
        // Convert target indices to one-hot with label smoothing
        let targets_indices = if targets.dims().len() == 2 && targets.dims()[1] == 1 {
            targets.squeeze(1)?
        } else {
            targets.clone()
        };

        let batch_size = targets_indices.dims()[0];
        let targets_data = targets_indices.to_vec1::<f32>().map_err(|e| {
            VangaError::ModelError(format!("Failed to extract target indices: {}", e))
        })?;

        // Create smoothed one-hot encoding
        let smooth_value = smoothing / (num_classes as f32 - 1.0);
        let target_value = 1.0 - smoothing;

        let mut one_hot_data = vec![smooth_value; batch_size * num_classes];
        for (i, &class_idx) in targets_data.iter().enumerate() {
            if class_idx >= 0.0 && (class_idx as usize) < num_classes {
                one_hot_data[i * num_classes + class_idx as usize] = target_value;
            }
        }

        Tensor::from_vec(one_hot_data, (batch_size, num_classes), targets.device()).map_err(|e| {
            VangaError::ModelError(format!("Failed to create smoothed targets: {}", e))
        })
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
        // ADDED: Validate tensor shapes and log for debugging
        log::debug!(
            "🔍 Volatility loss calculation - Predictions shape: {:?}, Targets shape: {:?}",
            predictions.shape(),
            targets.shape()
        );

        // Validate tensor shapes match
        if predictions.shape() != targets.shape() {
            return Err(VangaError::ModelError(format!(
                "Shape mismatch in volatility loss: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }
        // Calculate prediction volatility (standard deviation)
        let predictions_contiguous = predictions.contiguous()?;
        let pred_mean = predictions_contiguous.mean_all()?;

        // FIXED: Proper scalar broadcasting for tensor subtraction
        let pred_mean_scalar = pred_mean.to_scalar::<f32>().map_err(|e| {
            VangaError::ModelError(format!("Failed to extract prediction mean scalar: {}", e))
        })?;
        let pred_mean_broadcast = Tensor::full(
            pred_mean_scalar,
            predictions_contiguous.shape(),
            predictions_contiguous.device(),
        )?
        .contiguous()?;

        let pred_variance = predictions_contiguous
            .sub(&pred_mean_broadcast)?
            .contiguous()?
            .sqr()?
            .mean_all()?;
        let pred_volatility = pred_variance.sqrt()?;

        // Calculate target volatility
        let targets_contiguous = targets.contiguous()?;
        let target_mean = targets_contiguous.mean_all()?;

        // FIXED: Proper scalar broadcasting for tensor subtraction
        let target_mean_scalar = target_mean.to_scalar::<f32>().map_err(|e| {
            VangaError::ModelError(format!("Failed to extract target mean scalar: {}", e))
        })?;
        let target_mean_broadcast = Tensor::full(
            target_mean_scalar,
            targets_contiguous.shape(),
            targets_contiguous.device(),
        )?
        .contiguous()?;

        let target_variance = targets_contiguous
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
        // ADDED: Validate tensor shapes and log for debugging
        log::debug!(
            "🔍 Composite loss calculation - Predictions shape: {:?}, Targets shape: {:?}",
            predictions.shape(),
            targets.shape()
        );

        // Validate minimum tensor dimensions
        if predictions.dims().is_empty() || targets.dims().is_empty() {
            return Err(VangaError::ModelError(format!(
                "Invalid tensor dimensions for composite loss: predictions {:?}, targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        // FIXED: Use appropriate base loss based on target type (categorical vs regression)
        let base_loss = self.calculate_base_tensor_loss(predictions, targets)?;

        // Calculate all needed loss components conditionally
        // FIXED: Skip directional and volatility losses for categorical targets
        let is_categorical = self.is_categorical_target(predictions, targets);

        let directional_loss = if config.direction_weight > 0.0 && !is_categorical {
            // Directional loss only makes sense for regression targets
            Some(self.calculate_directional_tensor_loss(predictions, targets)?)
        } else {
            if is_categorical && config.direction_weight > 0.0 {
                log::debug!("⚠️ Skipping directional loss for categorical targets");
            }
            None
        };

        let volatility_loss =
            if (config.volatility_weight > 0.0 || config.risk_weight > 0.0) && !is_categorical {
                // Volatility loss only makes sense for regression targets
                Some(self.calculate_volatility_tensor_loss(predictions, targets)?)
            } else {
                if is_categorical && (config.volatility_weight > 0.0 || config.risk_weight > 0.0) {
                    log::debug!("⚠️ Skipping volatility loss for categorical targets");
                }
                None
            };

        // FIXED: Simplified and mathematically correct loss combination
        // No complex normalization - just weighted combination of loss components

        // Check if we have multiple components before consuming the values
        let has_directional = directional_loss.is_some();
        let has_volatility = volatility_loss.is_some();
        let has_multiple_components = has_directional || has_volatility;

        // Get regime multiplier and cached weights
        let regime_multiplier = self.get_regime_multiplier(market_regime);
        let weights = self.get_cached_weights(config, regime_multiplier, predictions.device())?;

        // FIXED: For categorical targets with only base loss, normalize accuracy weight to 1.0
        // This prevents severe loss downscaling that causes train/val mismatch
        //
        // Problem: With accuracy_weight=0.2, categorical loss becomes: 3.93 * 0.2 = 0.785
        // This creates 75% loss reduction compared to standard MSE path, causing:
        // - Training sees tiny gradients (0.785)
        // - Validation uses full-scale loss (~3.93)
        // - Result: Severe train/val mismatch and instability
        //
        // Solution: For single-component loss, use weight=1.0 to maintain loss magnitude
        let effective_accuracy_weight = if has_multiple_components {
            // Multiple components: use configured weight
            weights.accuracy.clone()
        } else {
            // Single component (categorical): normalize to maintain loss magnitude
            log::debug!(
                "🔧 Normalizing accuracy weight to 1.0 for single-component categorical loss"
            );
            Tensor::new(1.0f32, predictions.device())?
        };

        // Start with base loss weighted by effective accuracy weight
        let mut total_loss = base_loss.mul(&effective_accuracy_weight)?;

        log::debug!(
            "🔍 Composite loss components - Base: {:.6}, Effective accuracy weight: {:.3}",
            base_loss.to_scalar::<f32>().unwrap_or(0.0),
            effective_accuracy_weight.to_scalar::<f32>().unwrap_or(0.0)
        );

        // Add directional component if available (no additional scaling needed)
        if let Some(dir_loss) = directional_loss {
            let weighted_directional = dir_loss.mul(&weights.direction)?;
            total_loss = total_loss.add(&weighted_directional)?;
        }

        // Add volatility component if available (no additional scaling needed)
        if let Some(vol_loss) = volatility_loss {
            if config.volatility_weight > 0.0 {
                let weighted_volatility = vol_loss.mul(&weights.volatility)?;
                total_loss = total_loss.add(&weighted_volatility)?;
            }
            if config.risk_weight > 0.0 {
                let weighted_risk = vol_loss.mul(&weights.risk)?;
                total_loss = total_loss.add(&weighted_risk)?;
            }
        }

        // Apply regime adjustment only if we have multiple components
        // For categorical targets with only base loss, skip regime multiplier to prevent instability
        if has_multiple_components {
            total_loss.mul(&weights.regime_multiplier).map_err(|e| {
                VangaError::ModelError(format!("Crypto composite tensor loss failed: {}", e))
            })
        } else {
            // For single component (categorical targets), return weighted base loss without regime multiplier
            log::debug!("🎯 Using simplified loss for categorical targets (no regime multiplier)");
            Ok(total_loss)
        }
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

    /// Calculate weighted soft CrossEntropy loss for one-hot encoded targets (matches LSTM implementation)
    fn calculate_weighted_soft_crossentropy_loss(
        &self,
        logits: &Tensor,
        one_hot_targets: &Tensor,
        class_weights: &[f32],
    ) -> Result<Tensor> {
        // Ensure ALL input tensors are contiguous from the start
        let logits_contiguous = logits.contiguous()?;
        let targets_contiguous = one_hot_targets.contiguous()?;

        let log_softmax = candle_nn::ops::log_softmax(&logits_contiguous, candle_core::D::Minus1)?
            .contiguous()?;

        // Validate tensor dimensions
        let batch_size = targets_contiguous.dim(0)?;
        let num_classes = class_weights.len();

        if targets_contiguous.dim(1)? != num_classes {
            return Err(VangaError::ModelError(format!(
                "One-hot targets dimension {} doesn't match class weights {}",
                targets_contiguous.dim(1)?,
                num_classes
            )));
        }

        log::debug!(
            "🔍 Composite weighted soft CrossEntropy shapes: targets {:?}, logits {:?}, weights len {}",
            targets_contiguous.shape(),
            logits_contiguous.shape(),
            num_classes
        );

        // Create weight tensor with shape [1, num_classes] and ensure contiguous
        let weight_tensor = Tensor::from_vec(
            class_weights.to_vec(),
            (1, num_classes),
            logits_contiguous.device(),
        )?
        .contiguous()?;

        log::debug!(
            "🔍 Composite broadcasting shapes: targets {:?} × weights {:?}",
            targets_contiguous.shape(),
            weight_tensor.shape()
        );

        // Use broadcast_as to explicitly match tensor shapes before multiplication
        // Broadcasting: [1, num_classes] -> [batch_size, num_classes]
        let weight_tensor_broadcast = weight_tensor.broadcast_as(targets_contiguous.shape())?;

        log::debug!(
            "🔍 Composite after broadcast_as: targets {:?} × weights {:?}",
            targets_contiguous.shape(),
            weight_tensor_broadcast.shape()
        );

        // Now multiply tensors with matching shapes and ensure result is contiguous
        let weighted_targets = targets_contiguous
            .mul(&weight_tensor_broadcast)?
            .contiguous()?;

        // Calculate weighted soft CrossEntropy loss - ensure all intermediate results are contiguous
        let weighted_log_loss = weighted_targets.mul(&log_softmax)?.contiguous()?;
        let loss_per_sample = weighted_log_loss
            .sum(candle_core::D::Minus1)?
            .contiguous()?;
        let mean_loss = loss_per_sample.neg()?.mean_all()?.contiguous()?;

        log::debug!(
            "⚖️ Composite Weighted Soft CrossEntropy: {:.6} for {} samples with {} classes",
            mean_loss.to_scalar::<f32>().unwrap_or(0.0),
            batch_size,
            num_classes
        );

        Ok(mean_loss)
    }
}
