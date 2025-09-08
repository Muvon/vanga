//! Objective functions for VANGA LSTM optimization
//!
//! Defines optimization metrics and objective functions specifically designed
//! for cryptocurrency forecasting performance evaluation.

use crate::output::prediction_types::{
    DirectionPrediction, PriceBin, PriceLevelPrediction, VolatilityPrediction,
};
use crate::output::trading_orders::TradingOrders;
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Optimization metrics for model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationMetric {
    /// Mean Absolute Error
    MAE,
    /// Root Mean Square Error
    RMSE,
    /// Mean Absolute Percentage Error
    MAPE,
    /// Sharpe Ratio (risk-adjusted returns)
    SharpeRatio,
    /// Maximum Drawdown
    MaxDrawdown,
    /// Directional Accuracy
    DirectionalAccuracy,
    /// Multi-objective combining multiple metrics
    MultiObjective { weights: Vec<(String, f64)> },
    /// Crypto-specific composite metric
    Composite,
}

/// Market regime classification for regime-aware optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Low volatility, sideways movement
    LowVolatility,
    /// Medium volatility, trending
    MediumVolatility,
    /// High volatility, rapid changes
    HighVolatility,
    /// Bull market trending upward
    BullMarket,
    /// Bear market trending downward
    BearMarket,
    /// Range-bound market
    RangeBound,
}

/// Objective function for optimization
#[derive(Debug, Clone)]
pub struct ObjectiveFunction {
    primary_metric: OptimizationMetric,
    secondary_metrics: Vec<OptimizationMetric>,
    regime_awareness: bool,
    horizon_weights: HashMap<String, f64>,
}

impl ObjectiveFunction {
    /// Create new objective function with primary metric
    pub fn new(primary_metric: OptimizationMetric) -> Self {
        Self {
            primary_metric,
            secondary_metrics: Vec::new(),
            regime_awareness: false,
            horizon_weights: HashMap::new(),
        }
    }

    /// Create crypto-specific objective function
    pub fn crypto_specific() -> Self {
        let mut horizon_weights = HashMap::new();
        horizon_weights.insert("1h".to_string(), 0.2);
        horizon_weights.insert("4h".to_string(), 0.3);
        horizon_weights.insert("1d".to_string(), 0.3);
        horizon_weights.insert("7d".to_string(), 0.2);

        Self {
            primary_metric: OptimizationMetric::Composite,
            secondary_metrics: vec![
                OptimizationMetric::SharpeRatio,
                OptimizationMetric::MaxDrawdown,
                OptimizationMetric::DirectionalAccuracy,
            ],
            regime_awareness: true,
            horizon_weights,
        }
    }

    /// Add secondary metric for multi-objective optimization
    pub fn add_secondary_metric(mut self, metric: OptimizationMetric) -> Self {
        self.secondary_metrics.push(metric);
        self
    }

    /// Enable regime-aware optimization
    pub fn with_regime_awareness(mut self) -> Self {
        self.regime_awareness = true;
        self
    }

    /// Set horizon weights for multi-horizon optimization
    pub fn with_horizon_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.horizon_weights = weights;
        self
    }

    /// Evaluate objective function
    pub fn evaluate(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        prices: &[f64],
        market_regime: Option<MarketRegime>,
    ) -> Result<f64> {
        let primary_score = self.calculate_metric_score(
            &self.primary_metric,
            predictions,
            targets,
            prices,
            market_regime.as_ref(),
        )?;

        if self.secondary_metrics.is_empty() {
            return Ok(primary_score);
        }

        // Multi-objective optimization
        let mut total_score = primary_score * 0.7; // Primary metric gets 70% weight

        let secondary_weight = 0.3 / self.secondary_metrics.len() as f64;
        for metric in &self.secondary_metrics {
            let score = self.calculate_metric_score(
                metric,
                predictions,
                targets,
                prices,
                market_regime.as_ref(),
            )?;
            total_score += score * secondary_weight;
        }

        Ok(total_score)
    }

    /// Calculate individual metric score
    fn calculate_metric_score(
        &self,
        metric: &OptimizationMetric,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        prices: &[f64],
        market_regime: Option<&MarketRegime>,
    ) -> Result<f64> {
        match metric {
            OptimizationMetric::MAE => self.calculate_mae(predictions, targets),
            OptimizationMetric::RMSE => self.calculate_rmse(predictions, targets),
            OptimizationMetric::MAPE => self.calculate_mape(predictions, targets),
            OptimizationMetric::SharpeRatio => self.calculate_sharpe_ratio(predictions, prices),
            OptimizationMetric::MaxDrawdown => self.calculate_max_drawdown(predictions, prices),
            OptimizationMetric::DirectionalAccuracy => {
                self.calculate_directional_accuracy(predictions, targets)
            }
            OptimizationMetric::MultiObjective { weights } => {
                self.calculate_multi_objective(predictions, targets, prices, weights)
            }
            OptimizationMetric::Composite => {
                self.calculate_crypto_composite(predictions, targets, prices, market_regime)
            }
        }
    }

    /// Calculate Mean Absolute Error
    fn calculate_mae(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != targets.shape() {
            return Err(VangaError::DataError(
                "Predictions and targets shape mismatch".to_string(),
            ));
        }

        let mae = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| (pred - target).abs())
            .sum::<f64>()
            / (predictions.len() as f64);

        // Convert to score (lower MAE = higher score)
        Ok(1.0 / (1.0 + mae))
    }

    /// Calculate Root Mean Square Error
    fn calculate_rmse(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
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

        let rmse = mse.sqrt();

        // Convert to score (lower RMSE = higher score)
        Ok(1.0 / (1.0 + rmse))
    }

    /// Calculate Mean Absolute Percentage Error
    fn calculate_mape(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != targets.shape() {
            return Err(VangaError::DataError(
                "Predictions and targets shape mismatch".to_string(),
            ));
        }

        let mut total_percentage_error = 0.0;
        let mut valid_count = 0;

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if target.abs() > 1e-8 {
                // Avoid division by zero
                total_percentage_error += ((pred - target) / target).abs();
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            return Ok(0.0);
        }

        let mape = total_percentage_error / valid_count as f64;

        // Convert to score (lower MAPE = higher score)
        Ok(1.0 / (1.0 + mape))
    }

    /// Calculate Sharpe Ratio for risk-adjusted returns
    fn calculate_sharpe_ratio(&self, predictions: &Array2<f64>, prices: &[f64]) -> Result<f64> {
        if predictions.is_empty() || prices.len() < 2 {
            return Ok(0.0);
        }

        // Calculate returns from predictions
        let returns = self.calculate_returns_from_predictions(predictions, prices)?;

        if returns.len() < 2 {
            return Ok(0.0);
        }

        // Calculate mean return
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

        // Calculate standard deviation of returns
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(0.0);
        }

        // Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        let sharpe_ratio = mean_return / std_dev;

        // Normalize to 0-1 range (assuming good Sharpe ratio is around 1.0)
        Ok(((sharpe_ratio + 1.0) / 3.0).clamp(0.0, 1.0))
    }

    /// Calculate Maximum Drawdown
    fn calculate_max_drawdown(&self, predictions: &Array2<f64>, prices: &[f64]) -> Result<f64> {
        let returns = self.calculate_returns_from_predictions(predictions, prices)?;

        if returns.is_empty() {
            return Ok(0.0);
        }

        // Calculate cumulative returns
        let mut cumulative_returns = vec![1.0];
        for &ret in &returns {
            let new_value = cumulative_returns.last().unwrap() * (1.0 + ret);
            cumulative_returns.push(new_value);
        }

        // Calculate maximum drawdown
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

        // Convert to score (lower drawdown = higher score)
        Ok(1.0 - max_drawdown.min(1.0))
    }

    /// Calculate Directional Accuracy
    fn calculate_directional_accuracy(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<f64> {
        if predictions.shape() != targets.shape() || predictions.is_empty() {
            return Ok(0.0);
        }

        let mut correct_directions = 0;
        let mut total_predictions = 0;

        // Compare prediction directions with target directions
        for i in 1..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred_direction = predictions[[i, j]] > predictions[[i - 1, j]];
                let target_direction = targets[[i, j]] > targets[[i - 1, j]];

                if pred_direction == target_direction {
                    correct_directions += 1;
                }
                total_predictions += 1;
            }
        }

        if total_predictions == 0 {
            return Ok(0.0);
        }

        Ok((correct_directions as f64 / total_predictions as f64).min(1.0))
    }

    /// Calculate multi-objective score with custom weights
    fn calculate_multi_objective(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        prices: &[f64],
        weights: &Vec<(String, f64)>,
    ) -> Result<f64> {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        for (metric_name, weight) in weights {
            let metric = match metric_name.as_str() {
                "mae" => OptimizationMetric::MAE,
                "rmse" => OptimizationMetric::RMSE,
                "mape" => OptimizationMetric::MAPE,
                "sharpe" => OptimizationMetric::SharpeRatio,
                "drawdown" => OptimizationMetric::MaxDrawdown,
                "direction" => OptimizationMetric::DirectionalAccuracy,
                _ => continue,
            };

            let score = self.calculate_metric_score(&metric, predictions, targets, prices, None)?;
            total_score += score * weight;
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return Ok(0.0);
        }

        Ok(total_score / total_weight)
    }

    /// Calculate crypto-specific composite score
    fn calculate_crypto_composite(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        prices: &[f64],
        market_regime: Option<&MarketRegime>,
    ) -> Result<f64> {
        // Base metrics with crypto-specific weights
        let mae_score = self.calculate_mae(predictions, targets)? * 0.2;
        let directional_score = self.calculate_directional_accuracy(predictions, targets)? * 0.3;
        let sharpe_score = self.calculate_sharpe_ratio(predictions, prices)? * 0.25;
        let drawdown_score = self.calculate_max_drawdown(predictions, prices)? * 0.25;

        let mut composite_score = mae_score + directional_score + sharpe_score + drawdown_score;

        // Regime-specific adjustments
        if let Some(regime) = market_regime {
            composite_score *= match regime {
                MarketRegime::HighVolatility => 0.9, // Slightly penalize in high volatility
                MarketRegime::LowVolatility => 1.1,  // Bonus for low volatility performance
                MarketRegime::BullMarket => 1.05,    // Small bonus for bull market
                MarketRegime::BearMarket => 1.05,    // Small bonus for bear market (harder)
                MarketRegime::MediumVolatility => 1.0,
                MarketRegime::RangeBound => 0.95, // Slightly harder to predict
            };
        }

        Ok(composite_score.clamp(0.0, 1.0))
    }

    /// Helper: Calculate returns from predictions and prices
    fn calculate_returns_from_predictions(
        &self,
        predictions: &Array2<f64>,
        prices: &[f64],
    ) -> Result<Vec<f64>> {
        if predictions.is_empty() || prices.is_empty() {
            return Ok(Vec::new());
        }

        let mut returns = Vec::new();

        // Sophisticated trading strategy using TradingOrders infrastructure
        // Extract multi-target predictions and generate trading signals
        for i in 1..prices.len().min(predictions.nrows()) {
            let price_return = (prices[i] - prices[i - 1]) / prices[i - 1];
            let current_price = prices[i];

            // Extract predictions from multi-target array
            let direction_pred = if predictions.ncols() > 0 {
                let up_prob = predictions[[i, 0]].clamp(0.0, 1.0);
                let down_prob = 1.0 - up_prob;

                // Convert 2-class to 5-class probabilities for new structure
                let sideways_prob = 0.2; // Neutral probability
                let remaining = 1.0 - sideways_prob;
                let dump_prob = if down_prob > 0.5 {
                    (down_prob - 0.5) * remaining
                } else {
                    0.0
                };
                let pump_prob = if up_prob > 0.5 {
                    (up_prob - 0.5) * remaining
                } else {
                    0.0
                };
                let down_moderate = down_prob - dump_prob;
                let up_moderate = up_prob - pump_prob;

                DirectionPrediction::from_probabilities(
                    dump_prob,
                    down_moderate,
                    sideways_prob,
                    up_moderate,
                    pump_prob,
                )
            } else {
                // Default neutral prediction
                DirectionPrediction::from_probabilities(
                    0.1, 0.2, 0.4, 0.2,
                    0.1, // dump, down, sideways, up, pump - neutral distribution
                )
            };

            // Extract volatility prediction (column 1 if available)
            let volatility_pred = if predictions.ncols() > 1 {
                let vol_value = predictions[[i, 1]].clamp(0.0, 1.0);

                // Convert prediction value to actual volatility using proper financial scaling
                // Assume vol_value is a normalized prediction (0-1) that needs to be mapped to realistic volatility
                // Classify regime using VANGA's percentile-based thresholds
                // Based on src/targets/volatility.rs: (0.33, 0.67) percentiles
                let regime = if vol_value <= 0.33 {
                    "LOW"
                } else if vol_value <= 0.67 {
                    "MEDIUM"
                } else {
                    "HIGH"
                };

                // Create probability distribution based on regime
                let (very_low, low, medium, high, very_high) = match regime {
                    "LOW" => (0.2, 0.6, 0.2, 0.0, 0.0),
                    "MEDIUM" => (0.1, 0.2, 0.4, 0.2, 0.1),
                    "HIGH" => (0.0, 0.0, 0.2, 0.6, 0.2),
                    _ => (0.1, 0.2, 0.4, 0.2, 0.1),
                };

                VolatilityPrediction::from_probabilities(very_low, low, medium, high, very_high)
            } else {
                // Default medium volatility using 5-class probabilities
                VolatilityPrediction::from_probabilities(
                    0.1, 0.2, 0.4, 0.2, 0.1, // very_low, low, medium, high, very_high
                )
            };

            // Extract price level predictions (columns 2+ if available)
            let price_levels = if predictions.ncols() > 2 {
                let mut bins = HashMap::new();

                // Create price bins around current price
                let range_pct = 0.05; // 5% range
                let lower_bound = current_price * (1.0 - range_pct);
                let upper_bound = current_price * (1.0 + range_pct);

                bins.insert(
                    "support".to_string(),
                    PriceBin {
                        range: [lower_bound, current_price],
                        vwap_range: [0.0, 0.0], // Placeholder for optimization context
                        price: [lower_bound, current_price],
                        probability: predictions[[i, 2]].clamp(0.0, 1.0),
                    },
                );

                bins.insert(
                    "resistance".to_string(),
                    PriceBin {
                        range: [current_price, upper_bound],
                        vwap_range: [0.0, 0.0], // Placeholder for optimization context
                        price: [current_price, upper_bound],
                        probability: if predictions.ncols() > 3 {
                            predictions[[i, 3]].clamp(0.0, 1.0)
                        } else {
                            0.5
                        },
                    },
                );

                PriceLevelPrediction {
                    bins,
                    most_likely_range: [lower_bound, upper_bound],
                    confidence: 0.7,
                }
            } else {
                // Default price levels
                let range_pct = 0.03;
                let mut bins = HashMap::new();
                bins.insert(
                    "support".to_string(),
                    PriceBin {
                        range: [current_price * (1.0 - range_pct), current_price],
                        vwap_range: [0.0, 0.0], // Placeholder for optimization context
                        price: [current_price * (1.0 - range_pct), current_price],
                        probability: 0.5,
                    },
                );
                bins.insert(
                    "resistance".to_string(),
                    PriceBin {
                        range: [current_price, current_price * (1.0 + range_pct)],
                        vwap_range: [0.0, 0.0], // Placeholder for optimization context
                        price: [current_price, current_price * (1.0 + range_pct)],
                        probability: 0.5,
                    },
                );

                PriceLevelPrediction {
                    bins,
                    most_likely_range: [
                        current_price * (1.0 - range_pct),
                        current_price * (1.0 + range_pct),
                    ],
                    confidence: 0.5,
                }
            };

            // Create sequence prices for order generation (use recent price history)
            let sequence_start = i.saturating_sub(30); // Use up to 30 recent prices
            let sequence_prices = &prices[sequence_start..=i];

            // Calculate bandwidth size from recent price volatility
            let _bandwidth_size = if sequence_prices.len() > 1 {
                let price_std = self.calculate_std_dev(sequence_prices);
                (price_std / current_price).max(0.01) // At least 1% bandwidth
            } else {
                0.02 // Default 2% bandwidth
            };

            // Extract sentiment prediction (column 3 if available)
            let sentiment_pred = if predictions.ncols() > 3 {
                // We have 5 classes for sentiment, extract from predictions
                // Assuming predictions are 5-class probabilities
                let very_bearish = predictions[[i, 3]].clamp(0.0, 1.0);
                let bearish = if predictions.ncols() > 4 {
                    predictions[[i, 4]].clamp(0.0, 1.0)
                } else {
                    0.2
                };
                let neutral = 0.2; // Default neutral
                let bullish = 0.2;
                let very_bullish = 0.2;

                // Normalize to sum to 1.0
                let total = very_bearish + bearish + neutral + bullish + very_bullish;
                let probs = [
                    very_bearish / total,
                    bearish / total,
                    neutral / total,
                    bullish / total,
                    very_bullish / total,
                ];

                // Determine regime based on highest probability
                let max_idx = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(2);

                let regime = match max_idx {
                    0 => "VERY_BEARISH",
                    1 => "BEARISH",
                    2 => "NEUTRAL",
                    3 => "BULLISH",
                    4 => "VERY_BULLISH",
                    _ => "NEUTRAL",
                }
                .to_string();

                crate::output::prediction_types::SentimentPrediction {
                    very_bearish_probability: probs[0],
                    bearish_probability: probs[1],
                    neutral_probability: probs[2],
                    bullish_probability: probs[3],
                    very_bullish_probability: probs[4],
                    confidence: 0.6,
                    training_horizon: "1h".to_string(), // Default for optimization
                    regime,
                }
            } else {
                // Default neutral sentiment
                crate::output::prediction_types::SentimentPrediction {
                    very_bearish_probability: 0.2,
                    bearish_probability: 0.2,
                    neutral_probability: 0.2,
                    bullish_probability: 0.2,
                    very_bullish_probability: 0.2,
                    confidence: 0.5,
                    training_horizon: "1h".to_string(),
                    regime: "NEUTRAL".to_string(),
                }
            };

            // Extract volume prediction (column 4/5 if available)
            let volume_pred = if predictions.ncols() > 4 {
                // We have 5 classes for volume, extract from predictions
                let very_low = if predictions.ncols() > 5 {
                    predictions[[i, 5]].clamp(0.0, 1.0)
                } else {
                    0.2
                };
                let low = 0.2;
                let medium = 0.2;
                let high = 0.2;
                let very_high = 0.2;

                // Normalize to sum to 1.0
                let total = very_low + low + medium + high + very_high;
                let probs = [
                    very_low / total,
                    low / total,
                    medium / total,
                    high / total,
                    very_high / total,
                ];

                // Determine regime based on highest probability
                let max_idx = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(2);

                let regime = match max_idx {
                    0 => "VERY_LOW",
                    1 => "LOW",
                    2 => "MEDIUM",
                    3 => "HIGH",
                    4 => "VERY_HIGH",
                    _ => "MEDIUM",
                }
                .to_string();

                crate::output::prediction_types::VolumePrediction {
                    very_low_probability: probs[0],
                    low_probability: probs[1],
                    medium_probability: probs[2],
                    high_probability: probs[3],
                    very_high_probability: probs[4],
                    confidence: 0.6,
                    training_horizon: "1h".to_string(), // Default for optimization
                    regime,
                }
            } else {
                // Default normal volume
                crate::output::prediction_types::VolumePrediction {
                    very_low_probability: 0.2,
                    low_probability: 0.2,
                    medium_probability: 0.2,
                    high_probability: 0.2,
                    very_high_probability: 0.2,
                    confidence: 0.5,
                    training_horizon: "1h".to_string(),
                    regime: "MEDIUM".to_string(),
                }
            };

            // Create confidence calculator with default settings
            let confidence_calculator =
                crate::output::confidence_calculator::ConfidenceCalculator::new(
                    crate::output::confidence_calculator::ConfidenceConfig::default(),
                );

            // Convert sequence prices to MarketDataRow for SmartOrderConfig
            let sequence_ohlcv = None; // Not available in optimization context

            let config = crate::output::trading_orders::SmartOrderConfig {
                current_price,
                price_levels: &price_levels,
                direction_pred: &direction_pred,
                volatility_pred: &volatility_pred,
                sentiment_pred: &sentiment_pred,
                volume_pred: &volume_pred,
                confidence_calculator: &confidence_calculator,
                min_confidence: 0.3, // Lower threshold for optimization context
                sequence_ohlcv,
            };

            match TradingOrders::generate(config) {
                Ok(orders) => {
                    // Calculate strategy return based on trading orders
                    let strategy_return = self.calculate_return_from_orders(&orders, price_return);
                    returns.push(strategy_return);
                }
                Err(_) => {
                    // Fallback to neutral position on order generation failure
                    returns.push(0.0);
                }
            }
        }

        Ok(returns)
    }

    /// Calculate strategy return from trading orders
    fn calculate_return_from_orders(&self, orders: &TradingOrders, price_return: f64) -> f64 {
        // If no valid trading direction, return neutral
        if orders.direction == "NEUTRAL" || orders.entry_levels.is_empty() {
            return 0.0;
        }

        // Calculate position size based on confidence and risk management
        let base_position_size = orders.total_position_size;

        // Apply risk-reward adjustment
        let risk_adjusted_size = if orders.risk_reward_ratio > 2.0 {
            base_position_size * (1.0 + (orders.risk_reward_ratio - 2.0) * 0.1).min(2.0)
        } else {
            base_position_size * 0.5 // Reduce size for poor risk-reward
        };

        // Calculate directional return
        let directional_return = match orders.direction.as_str() {
            "LONG" => price_return * risk_adjusted_size,
            "SHORT" => -price_return * risk_adjusted_size,
            _ => 0.0,
        };

        // Apply volatility-based scaling
        let volatility_scaling = if orders.atr_multiplier > 2.5 {
            0.8 // Reduce exposure in high volatility
        } else if orders.atr_multiplier < 1.5 {
            1.2 // Increase exposure in low volatility
        } else {
            1.0
        };

        directional_return * volatility_scaling
    }

    /// Detect market regime from price data
    pub fn detect_market_regime(&self, prices: &[f64]) -> Result<MarketRegime> {
        if prices.len() < 20 {
            return Ok(MarketRegime::MediumVolatility);
        }

        // Calculate volatility
        let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

        let volatility = self.calculate_volatility(&returns);

        // Calculate trend
        let trend = self.calculate_trend(prices);

        // Classify regime
        let regime = match (volatility, trend) {
            (v, _) if v > 0.05 => MarketRegime::HighVolatility,
            (v, _) if v < 0.02 => MarketRegime::LowVolatility,
            (_, t) if t > 0.1 => MarketRegime::BullMarket,
            (_, t) if t < -0.1 => MarketRegime::BearMarket,
            (_, t) if t.abs() < 0.05 => MarketRegime::RangeBound,
            _ => MarketRegime::MediumVolatility,
        };

        Ok(regime)
    }

    /// Calculate volatility from returns
    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.02;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Calculate trend from prices
    fn calculate_trend(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let first_half_avg =
            prices[..prices.len() / 2].iter().sum::<f64>() / (prices.len() / 2) as f64;
        let second_half_avg = prices[prices.len() / 2..].iter().sum::<f64>()
            / (prices.len() - prices.len() / 2) as f64;

        (second_half_avg - first_half_avg) / first_half_avg
    }

    /// Calculate standard deviation from prices
    fn calculate_std_dev(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;

        variance.sqrt()
    }
}

impl Default for ObjectiveFunction {
    fn default() -> Self {
        Self::crypto_specific()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_objective_bincode_serialization() {
        // Test that MultiObjective variant can be serialized/deserialized with bincode
        let weights = vec![
            ("mae".to_string(), 0.3),
            ("rmse".to_string(), 0.4),
            ("sharpe".to_string(), 0.3),
        ];

        let metric = OptimizationMetric::MultiObjective { weights };

        // Serialize
        let serialized = bincode::serialize(&metric).expect("Failed to serialize MultiObjective");

        // Deserialize
        let deserialized: OptimizationMetric =
            bincode::deserialize(&serialized).expect("Failed to deserialize MultiObjective");

        // Verify it matches
        match deserialized {
            OptimizationMetric::MultiObjective { weights } => {
                assert_eq!(weights.len(), 3);
                assert!(weights.contains(&("mae".to_string(), 0.3)));
                assert!(weights.contains(&("rmse".to_string(), 0.4)));
                assert!(weights.contains(&("sharpe".to_string(), 0.3)));
            }
            _ => panic!("Deserialized to wrong variant"),
        }
    }
}
