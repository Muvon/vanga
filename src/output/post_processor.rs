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
        // TODO: Implement actual trend smoothing logic
        Ok(predictions)
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

        // For now, just return the first set of predictions
        // TODO: Implement proper ensemble weighting
        Ok(predictions.into_iter().next().unwrap_or_default())
    }
}

impl Default for PostProcessor {
    fn default() -> Self {
        Self::new(PostProcessingConfig::default())
    }
}
