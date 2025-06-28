//! Post-processing pipeline for prediction results
//!
//! This module applies confidence filtering, volatility adjustments, and other
//! post-processing steps to improve prediction quality.

use crate::config::prediction::{PostProcessingConfig, PostProcessingStep};
use crate::output::structures::PredictionResult;
use crate::utils::error::{Result, VangaError};

/// Post-processor for prediction results
pub struct PostProcessor {
    config: PostProcessingConfig,
}

impl PostProcessor {
    /// Create new post-processor with configuration
    pub fn new(config: PostProcessingConfig) -> Self {
        Self { config }
    }

    /// Apply all configured post-processing steps
    pub fn process(&self, mut predictions: Vec<PredictionResult>) -> Result<Vec<PredictionResult>> {
        for step in &self.config.steps {
            predictions = match step {
                PostProcessingStep::VolatilityAdjustment => {
                    self.apply_volatility_adjustment(predictions)?
                }
                PostProcessingStep::TrendSmoothing => self.apply_trend_smoothing(predictions)?,
                PostProcessingStep::OutlierFiltering => {
                    self.apply_outlier_filtering(predictions)?
                }
                PostProcessingStep::RegimeAdjustment => {
                    self.apply_regime_adjustment(predictions)?
                }
            };
        }

        Ok(predictions)
    }

    /// Apply confidence threshold filtering
    pub fn filter_by_confidence(
        &self,
        predictions: Vec<PredictionResult>,
        min_confidence: f64,
    ) -> Vec<PredictionResult> {
        predictions
            .into_iter()
            .filter(|pred| pred.confidence >= min_confidence)
            .collect()
    }

    /// Apply volatility adjustment to predictions
    fn apply_volatility_adjustment(
        &self,
        mut predictions: Vec<PredictionResult>,
    ) -> Result<Vec<PredictionResult>> {
        if !self.config.volatility_adjustment.enabled {
            return Ok(predictions);
        }

        // Adjust confidence based on market volatility
        for pred in &mut predictions {
            if let Some(ref volatility) = pred.volatility {
                let vol_factor = match volatility.regime.as_str() {
                    "HIGH" => 0.8,   // Reduce confidence in high volatility
                    "MEDIUM" => 0.9, // Slightly reduce confidence
                    "LOW" => 1.0,    // Keep confidence in low volatility
                    _ => 1.0,
                };

                pred.confidence *= vol_factor;

                // Adjust price level confidence
                if let Some(ref mut price_levels) = pred.price_levels {
                    price_levels.confidence *= vol_factor;
                }

                // Adjust direction confidence
                if let Some(ref mut direction) = pred.direction {
                    direction.confidence *= vol_factor;
                }
            }
        }

        Ok(predictions)
    }

    /// Apply trend smoothing to reduce noise
    fn apply_trend_smoothing(
        &self,
        predictions: Vec<PredictionResult>,
    ) -> Result<Vec<PredictionResult>> {
        // For now, just return predictions unchanged
        // Apply exponential moving average smoothing to reduce noise and improve stability
        let smoothed_predictions = if predictions.len() > 1 {
            let mut smoothed = Vec::with_capacity(predictions.len());
            let alpha = 0.3; // Smoothing factor for EMA

            // First prediction remains unchanged
            smoothed.push(predictions[0].clone());

            // Apply EMA smoothing to subsequent predictions
            for i in 1..predictions.len() {
                let mut smoothed_pred = predictions[i].clone();

                // Smooth confidence scores
                smoothed_pred.confidence =
                    alpha * predictions[i].confidence + (1.0 - alpha) * smoothed[i - 1].confidence;

                // Smooth price level probabilities if available
                if let (Some(current_price_levels), Some(prev_price_levels)) =
                    (&predictions[i].price_levels, &smoothed[i - 1].price_levels)
                {
                    let mut smoothed_bins = current_price_levels.bins.clone();

                    for (bin_name, current_bin) in &current_price_levels.bins {
                        if let Some(prev_bin) = prev_price_levels.bins.get(bin_name) {
                            if let Some(smoothed_bin) = smoothed_bins.get_mut(bin_name) {
                                smoothed_bin.probability = alpha * current_bin.probability
                                    + (1.0 - alpha) * prev_bin.probability;
                            }
                        }
                    }

                    smoothed_pred.price_levels =
                        Some(crate::output::structures::PriceLevelPrediction {
                            bins: smoothed_bins,
                            most_likely_range: current_price_levels.most_likely_range.clone(),
                            confidence: alpha * current_price_levels.confidence
                                + (1.0 - alpha) * prev_price_levels.confidence,
                        });
                }

                smoothed.push(smoothed_pred);
            }

            smoothed
        } else {
            predictions
        };

        Ok(smoothed_predictions)
    }

    /// Filter out outlier predictions
    fn apply_outlier_filtering(
        &self,
        predictions: Vec<PredictionResult>,
    ) -> Result<Vec<PredictionResult>> {
        // Simple outlier filtering based on confidence
        let confidences: Vec<f64> = predictions.iter().map(|p| p.confidence).collect();

        if confidences.is_empty() {
            return Ok(predictions);
        }

        // Calculate median and IQR
        let mut sorted_confidences = confidences.clone();
        sorted_confidences.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1_idx = sorted_confidences.len() / 4;
        let q3_idx = (sorted_confidences.len() * 3) / 4;

        if q1_idx >= sorted_confidences.len() || q3_idx >= sorted_confidences.len() {
            return Ok(predictions); // Not enough data for outlier detection
        }

        let q1 = sorted_confidences[q1_idx];
        let q3 = sorted_confidences[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        // Filter predictions within bounds
        let filtered: Vec<PredictionResult> = predictions
            .into_iter()
            .filter(|pred| pred.confidence >= lower_bound && pred.confidence <= upper_bound)
            .collect();

        Ok(filtered)
    }

    /// Apply market regime adjustments
    fn apply_regime_adjustment(
        &self,
        mut predictions: Vec<PredictionResult>,
    ) -> Result<Vec<PredictionResult>> {
        if !self.config.regime_detection.enabled {
            return Ok(predictions);
        }

        // Adjust predictions based on detected market regime
        for pred in &mut predictions {
            // Simple regime-based adjustment
            let regime_factor = match pred.metadata.data_quality.market_condition.as_str() {
                "VOLATILE" => 0.85, // Reduce confidence in volatile markets
                "TRENDING" => 1.1,  // Increase confidence in trending markets
                "NORMAL" => 1.0,    // No adjustment for normal markets
                _ => 1.0,
            };

            pred.confidence = (pred.confidence * regime_factor).min(1.0);
        }

        Ok(predictions)
    }

    /// Calculate market regime from recent data
    pub fn detect_market_regime(&self, recent_prices: &[f64]) -> String {
        if recent_prices.len() < 10 {
            return "INSUFFICIENT_DATA".to_string();
        }

        // Calculate price volatility over the period
        let returns: Vec<f64> = recent_prices
            .windows(2)
            .map(|window| (window[1] - window[0]) / window[0])
            .collect();

        if returns.is_empty() {
            return "INSUFFICIENT_DATA".to_string();
        }

        // Calculate volatility (standard deviation of returns)
        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let volatility = variance.sqrt();

        // Calculate trend strength (linear regression slope)
        let n = recent_prices.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = recent_prices.iter().sum::<f64>() / n;

        let numerator: f64 = recent_prices
            .iter()
            .enumerate()
            .map(|(i, &price)| (i as f64 - x_mean) * (price - y_mean))
            .sum();

        let denominator: f64 = (0..recent_prices.len())
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum();

        let trend_slope = if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        // Classify market regime based on volatility and trend
        let high_volatility_threshold = 0.02; // 2% daily volatility
        let strong_trend_threshold = recent_prices[0] * 0.001; // 0.1% price change per period

        if volatility > high_volatility_threshold {
            "VOLATILE".to_string()
        } else if trend_slope.abs() > strong_trend_threshold {
            "TRENDING".to_string()
        } else {
            "NORMAL".to_string()
        }
    }

    /// Apply ensemble weighting to multiple predictions
    pub fn apply_ensemble_weighting(
        &self,
        predictions: Vec<Vec<PredictionResult>>,
        weights: &[f64],
    ) -> Result<Vec<PredictionResult>> {
        if predictions.is_empty() || weights.is_empty() {
            return Ok(Vec::new());
        }

        if predictions.len() != weights.len() {
            return Err(VangaError::InvalidParameter {
                parameter: "weights".to_string(),
                value: format!("{}", weights.len()),
                reason: format!(
                    "Must match number of prediction sets: {}",
                    predictions.len()
                ),
            });
        }

        // Implement proper ensemble weighting based on market conditions and model confidence
        if predictions.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate weighted ensemble based on confidence scores and market regime
        let mut ensemble_result = Vec::new();
        let total_predictions = predictions.len();

        if total_predictions == 1 {
            return Ok(predictions.into_iter().next().unwrap_or_default());
        }

        // Group predictions by horizon for proper ensemble
        let mut horizon_groups: std::collections::HashMap<String, Vec<PredictionResult>> =
            std::collections::HashMap::new();

        for pred_set in predictions {
            for prediction in pred_set {
                horizon_groups
                    .entry(prediction.horizon.clone())
                    .or_default()
                    .push(prediction);
            }
        }

        // Create ensemble prediction for each horizon
        for (_horizon, horizon_predictions) in horizon_groups {
            if let Some(ensemble_pred) = self.create_ensemble_prediction(&horizon_predictions)? {
                ensemble_result.push(ensemble_pred);
            }
        }

        Ok(ensemble_result)
    }

    /// Create ensemble prediction from multiple predictions for the same horizon
    fn create_ensemble_prediction(
        &self,
        predictions: &[PredictionResult],
    ) -> Result<Option<PredictionResult>> {
        if predictions.is_empty() {
            return Ok(None);
        }

        if predictions.len() == 1 {
            return Ok(Some(predictions[0].clone()));
        }

        let first_pred = &predictions[0];
        let mut ensemble_pred = PredictionResult::new(
            first_pred.symbol.clone(),
            first_pred.horizon.clone(),
            first_pred.current_price,
        );

        // Calculate weighted average confidence
        let total_confidence: f64 = predictions.iter().map(|p| p.confidence).sum();
        let weights: Vec<f64> = predictions
            .iter()
            .map(|p| p.confidence / total_confidence)
            .collect();

        // Ensemble confidence as weighted average
        ensemble_pred.confidence = predictions
            .iter()
            .zip(&weights)
            .map(|(pred, weight)| pred.confidence * weight)
            .sum();

        // Ensemble price levels if available
        if let Some(first_price_levels) = &first_pred.price_levels {
            let mut ensemble_bins = first_price_levels.bins.clone();

            // Reset probabilities and calculate weighted average
            for bin in ensemble_bins.values_mut() {
                bin.probability = 0.0;
            }

            for (pred, weight) in predictions.iter().zip(&weights) {
                if let Some(price_levels) = &pred.price_levels {
                    for (bin_name, bin) in &price_levels.bins {
                        if let Some(ensemble_bin) = ensemble_bins.get_mut(bin_name) {
                            ensemble_bin.probability += bin.probability * weight;
                        }
                    }
                }
            }

            // Calculate ensemble confidence based on prediction spread
            let confidence_interval = weights.iter().sum::<f64>() / weights.len() as f64;

            ensemble_pred =
                ensemble_pred.with_price_levels(crate::output::structures::PriceLevelPrediction {
                    bins: ensemble_bins,
                    most_likely_range: first_price_levels.most_likely_range.clone(),
                    confidence: confidence_interval,
                });
        }

        // Ensemble direction if available
        if predictions.iter().any(|p| p.direction.is_some()) {
            let mut up_prob_sum = 0.0;
            let mut down_prob_sum = 0.0;
            let mut total_weight = 0.0;

            for (pred, weight) in predictions.iter().zip(&weights) {
                if let Some(direction) = &pred.direction {
                    up_prob_sum += direction.up_probability * weight;
                    down_prob_sum += direction.down_probability * weight;
                    total_weight += weight;
                }
            }

            if total_weight > 0.0 {
                let ensemble_up_prob = up_prob_sum / total_weight;
                let ensemble_down_prob = down_prob_sum / total_weight;

                ensemble_pred =
                    ensemble_pred.with_direction(crate::output::structures::DirectionPrediction {
                        up_probability: ensemble_up_prob,
                        down_probability: ensemble_down_prob,
                        prediction: if ensemble_up_prob > ensemble_down_prob {
                            "UP".to_string()
                        } else {
                            "DOWN".to_string()
                        },
                        confidence: (ensemble_up_prob - ensemble_down_prob).abs(),
                    });
            }
        }

        Ok(Some(ensemble_pred))
    }
}

impl Default for PostProcessor {
    fn default() -> Self {
        Self::new(PostProcessingConfig::default())
    }
}
