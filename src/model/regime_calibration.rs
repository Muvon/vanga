//! Regime Calibration System - Mathematically Sound Market Regime Handling
//!
//! This module provides empirical regime calibration to replace hardcoded multipliers
//! with statistically sound normalization based on actual loss distributions.

use crate::optimization::objective::MarketRegime;
use crate::utils::error::{Result, VangaError};
use candle_core::Tensor;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Statistical measurements for loss distributions per regime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossStatistics {
    /// Mean loss value for this regime
    pub mean: f64,
    /// Standard deviation of loss values
    pub std_dev: f64,
    /// Number of samples used for calculation
    pub sample_count: usize,
    /// Percentiles for robust statistics [10th, 25th, 50th, 75th, 90th]
    pub percentiles: [f64; 5],
    /// Minimum observed loss
    pub min_loss: f64,
    /// Maximum observed loss
    pub max_loss: f64,
    /// Variance for mathematical operations
    pub variance: f64,
}

impl Default for LossStatistics {
    fn default() -> Self {
        Self {
            mean: 1.0,
            std_dev: 1.0,
            sample_count: 0,
            percentiles: [0.0, 0.0, 1.0, 0.0, 0.0],
            min_loss: f64::INFINITY,
            max_loss: f64::NEG_INFINITY,
            variance: 1.0,
        }
    }
}

impl LossStatistics {
    /// Create new statistics from a collection of loss values
    pub fn from_losses(losses: &[f64]) -> Result<Self> {
        if losses.is_empty() {
            return Ok(Self::default());
        }

        let mut sorted_losses = losses.to_vec();
        sorted_losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = losses.iter().sum::<f64>() / losses.len() as f64;
        let variance = losses.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / losses.len() as f64;
        let std_dev = variance.sqrt();

        let percentiles = [
            Self::percentile(&sorted_losses, 0.1),  // 10th
            Self::percentile(&sorted_losses, 0.25), // 25th
            Self::percentile(&sorted_losses, 0.5),  // 50th (median)
            Self::percentile(&sorted_losses, 0.75), // 75th
            Self::percentile(&sorted_losses, 0.9),  // 90th
        ];

        Ok(Self {
            mean,
            std_dev: if std_dev > 1e-8 { std_dev } else { 1e-8 }, // Prevent division by zero
            sample_count: losses.len(),
            percentiles,
            min_loss: sorted_losses[0],
            max_loss: sorted_losses[sorted_losses.len() - 1],
            variance,
        })
    }

    /// Calculate percentile from sorted data
    fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        if sorted_data.is_empty() {
            return 0.0;
        }

        let index = (p * (sorted_data.len() - 1) as f64).round() as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }

    /// Check if statistics are reliable (enough samples)
    pub fn is_reliable(&self) -> bool {
        self.sample_count >= 10 && self.std_dev > 1e-8
    }

    /// Normalize a loss value using Z-score normalization
    pub fn normalize_loss(&self, loss: f64) -> f64 {
        if !self.is_reliable() {
            return loss; // Return original if not enough data
        }

        // Z-score normalization: (x - μ) / σ
        (loss - self.mean) / self.std_dev
    }

    /// Robust normalization using median and IQR
    pub fn robust_normalize_loss(&self, loss: f64) -> f64 {
        if !self.is_reliable() {
            return loss;
        }

        let median = self.percentiles[2]; // 50th percentile
        let q1 = self.percentiles[1]; // 25th percentile
        let q3 = self.percentiles[3]; // 75th percentile
        let iqr = q3 - q1;

        if iqr > 1e-8 {
            (loss - median) / iqr
        } else {
            loss - median
        }
    }
}

/// Regime calibration system for empirical loss normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeCalibrator {
    /// Statistics per regime type
    regime_stats: HashMap<MarketRegime, LossStatistics>,
    /// Calibration phase settings
    calibration_config: CalibrationConfig,
    /// Whether calibration is complete
    is_calibrated: bool,
    /// Raw loss samples collected during calibration
    calibration_samples: HashMap<MarketRegime, Vec<f64>>,
}

/// Configuration for regime calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Minimum samples needed per regime for reliable statistics
    pub min_samples_per_regime: usize,
    /// Maximum samples to collect (memory limit)
    pub max_samples_per_regime: usize,
    /// Use robust normalization (median/IQR) instead of Z-score
    pub use_robust_normalization: bool,
    /// Enable automatic recalibration when new data is significantly different
    pub enable_adaptive_calibration: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            min_samples_per_regime: 50, // Need at least 50 samples for reliable stats
            max_samples_per_regime: 1000, // Limit memory usage
            use_robust_normalization: true, // More stable for financial data
            enable_adaptive_calibration: false, // Conservative default
        }
    }
}

impl RegimeCalibrator {
    /// Create new regime calibrator
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            regime_stats: HashMap::new(),
            calibration_config: config,
            is_calibrated: false,
            calibration_samples: HashMap::new(),
        }
    }

    /// Add loss sample for a specific regime during calibration phase
    pub fn add_calibration_sample(&mut self, regime: MarketRegime, loss: f64) {
        if !loss.is_finite() {
            log::warn!("🚨 Ignoring non-finite loss sample: {}", loss);
            return;
        }

        let samples = self.calibration_samples.entry(regime).or_default();

        if samples.len() < self.calibration_config.max_samples_per_regime {
            samples.push(loss);

            log::debug!(
                "📊 Calibration sample added for {:?}: {} (total: {})",
                regime,
                loss,
                samples.len()
            );
        }
    }

    /// Finalize calibration by computing statistics from collected samples
    pub fn finalize_calibration(&mut self) -> Result<()> {
        log::info!("🔧 Finalizing regime calibration...");

        for (regime, samples) in &self.calibration_samples {
            if samples.len() >= self.calibration_config.min_samples_per_regime {
                let stats = LossStatistics::from_losses(samples)?;
                self.regime_stats.insert(*regime, stats.clone());

                log::info!(
                    "✅ Regime {:?} calibrated: mean={:.4}, std={:.4}, samples={}",
                    regime,
                    stats.mean,
                    stats.std_dev,
                    stats.sample_count
                );
            } else {
                log::warn!(
                    "⚠️ Insufficient samples for {:?}: {} < {}",
                    regime,
                    samples.len(),
                    self.calibration_config.min_samples_per_regime
                );
            }
        }

        self.is_calibrated = !self.regime_stats.is_empty();

        if self.is_calibrated {
            log::info!(
                "🎯 Regime calibration completed for {} regimes",
                self.regime_stats.len()
            );
        } else {
            log::warn!("⚠️ Regime calibration failed - no regimes have sufficient samples");
        }

        Ok(())
    }

    /// Normalize loss using regime-specific statistics
    pub fn normalize_loss(&self, regime: MarketRegime, loss: f64) -> f64 {
        if !self.is_calibrated {
            log::debug!("📊 Calibration not complete, returning original loss");
            return loss;
        }

        if let Some(stats) = self.regime_stats.get(&regime) {
            let normalized = if self.calibration_config.use_robust_normalization {
                stats.robust_normalize_loss(loss)
            } else {
                stats.normalize_loss(loss)
            };

            log::debug!(
                "🔧 Loss normalized for {:?}: {:.6} -> {:.6}",
                regime,
                loss,
                normalized
            );

            normalized
        } else {
            log::debug!(
                "⚠️ No calibration data for {:?}, returning original loss",
                regime
            );
            loss
        }
    }

    /// Get regime statistics for inspection
    pub fn get_regime_stats(&self, regime: MarketRegime) -> Option<&LossStatistics> {
        self.regime_stats.get(&regime)
    }

    /// Check if calibrator is ready for use
    pub fn is_calibrated(&self) -> bool {
        self.is_calibrated
    }

    /// Get calibration progress (percentage of regimes with sufficient samples)
    pub fn calibration_progress(&self) -> f64 {
        let total_regimes = 5; // MarketRegime has 5 variants: VeryLow, Low, Medium, High, VeryHigh
        let calibrated_regimes = self.regime_stats.len();
        (calibrated_regimes as f64 / total_regimes as f64) * 100.0
    }

    /// Reset calibration (for adaptive recalibration)
    pub fn reset_calibration(&mut self) {
        self.regime_stats.clear();
        self.calibration_samples.clear();
        self.is_calibrated = false;
        log::info!("🔄 Regime calibration reset");
    }
}

/// Epoch-level regime detection for consistent validation
pub struct EpochRegimeDetector;

impl EpochRegimeDetector {
    /// Detect market regime from entire epoch data (not per-batch)
    pub fn detect_epoch_regime(targets: &Array2<f64>) -> Result<MarketRegime> {
        if targets.is_empty() {
            return Ok(MarketRegime::MediumVolatility);
        }

        // Calculate comprehensive statistics from entire epoch
        let data: Vec<f64> = targets.iter().cloned().collect();
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate additional metrics for robust regime detection
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let _median = sorted_data[sorted_data.len() / 2];
        let q1 = sorted_data[sorted_data.len() / 4];
        let q3 = sorted_data[3 * sorted_data.len() / 4];
        let iqr = q3 - q1;

        // Coefficient of variation for volatility assessment
        let cv = if mean.abs() > 1e-8 {
            std_dev / mean.abs()
        } else {
            0.0
        };

        // Skewness for trend detection
        let skewness = if std_dev > 1e-8 {
            data.iter()
                .map(|x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>()
                / data.len() as f64
        } else {
            0.0
        };

        // Mathematical regime classification based on statistical properties
        let regime = if cv > 0.5 {
            // High coefficient of variation indicates high volatility
            MarketRegime::HighVolatility
        } else if cv < 0.2 {
            // Low coefficient of variation indicates low volatility or range-bound
            if iqr < std_dev * 0.5 {
                MarketRegime::RangeBound
            } else {
                MarketRegime::LowVolatility
            }
        } else if skewness > 0.5 {
            // Positive skew indicates bull market tendency
            MarketRegime::BullMarket
        } else if skewness < -0.5 {
            // Negative skew indicates bear market tendency
            MarketRegime::BearMarket
        } else {
            // Balanced distribution indicates medium volatility
            MarketRegime::MediumVolatility
        };

        log::debug!(
            "🔍 Epoch regime detection: mean={:.4}, std={:.4}, cv={:.4}, skew={:.4} -> {:?}",
            mean,
            std_dev,
            cv,
            skewness,
            regime
        );

        Ok(regime)
    }

    /// Detect regime from tensor data
    pub fn detect_regime_from_tensor(targets: &Tensor) -> Result<MarketRegime> {
        // Convert tensor to ndarray for processing
        let shape = targets.shape();
        if shape.dims().len() != 2 {
            return Err(VangaError::ModelError(format!(
                "Expected 2D tensor for regime detection, got shape: {:?}",
                shape
            )));
        }

        let data = targets.to_vec2::<f32>().map_err(|e| {
            VangaError::ModelError(format!("Failed to convert tensor to vec: {}", e))
        })?;

        // Flatten and convert to f64
        let flat_data: Vec<f64> = data.into_iter().flatten().map(|x| x as f64).collect();

        // Create temporary Array2 for processing
        let rows = shape.dims()[0];
        let cols = shape.dims()[1];
        let array = Array2::from_shape_vec((rows, cols), flat_data)
            .map_err(|e| VangaError::ModelError(format!("Failed to create Array2: {}", e)))?;

        Self::detect_epoch_regime(&array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_statistics_creation() {
        let losses = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = LossStatistics::from_losses(&losses).unwrap();

        assert_eq!(stats.mean, 3.0);
        assert!(stats.std_dev > 0.0);
        assert_eq!(stats.sample_count, 5);
        assert!(stats.is_reliable());
    }

    #[test]
    fn test_regime_calibrator() {
        let mut calibrator = RegimeCalibrator::new(CalibrationConfig::default());

        // Add samples
        for i in 0..60 {
            calibrator.add_calibration_sample(MarketRegime::MediumVolatility, i as f64);
        }

        calibrator.finalize_calibration().unwrap();
        assert!(calibrator.is_calibrated());

        let normalized = calibrator.normalize_loss(MarketRegime::MediumVolatility, 30.0);
        assert!(normalized.is_finite());
    }

    #[test]
    fn test_epoch_regime_detection() {
        // Create test data with high volatility pattern
        let data = vec![1.0, 10.0, 2.0, 15.0, 3.0, 20.0];
        let array = Array2::from_shape_vec((2, 3), data).unwrap();

        let regime = EpochRegimeDetector::detect_epoch_regime(&array).unwrap();
        // Should detect high volatility due to large variations
        assert!(matches!(
            regime,
            MarketRegime::HighVolatility | MarketRegime::MediumVolatility
        ));
    }
}
