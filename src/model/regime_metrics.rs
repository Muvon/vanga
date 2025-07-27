//! Comprehensive logging and metrics for the regime calibration system
//!
//! This module provides detailed logging, metrics collection, and validation
//! for the complete regime calibration and dual loss system.

use crate::model::dual_loss_system::{DualLossResult, DualLossSystem};
use crate::model::regime_calibration::RegimeCalibrator;
use crate::optimization::objective::MarketRegime;
use crate::utils::error::{Result, VangaError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive metrics collector for regime calibration system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeMetricsCollector {
    /// Training metrics per epoch
    pub training_metrics: Vec<EpochMetrics>,
    /// Validation metrics per epoch
    pub validation_metrics: Vec<EpochMetrics>,
    /// Regime distribution statistics
    pub regime_distribution: HashMap<MarketRegime, usize>,
    /// Loss consistency metrics
    pub loss_consistency: LossConsistencyMetrics,
    /// Dropout behavior tracking
    pub dropout_behavior: DropoutBehaviorMetrics,
}

/// Metrics for a single epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub training_loss: f32,
    pub evaluation_loss: f32,
    pub loss_ratio: f32,
    pub regime: MarketRegime,
    pub regime_calibrated: bool,
    pub gradient_norm: f64,
    pub learning_rate: f64,
}

/// Loss consistency tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConsistencyMetrics {
    /// Standard deviation of loss ratios (should be low for consistency)
    pub loss_ratio_std: f32,
    /// Maximum loss ratio observed
    pub max_loss_ratio: f32,
    /// Minimum loss ratio observed
    pub min_loss_ratio: f32,
    /// Number of epochs with overfitting detected (ratio > 3.0)
    pub overfitting_epochs: usize,
    /// Average loss ratio across all epochs
    pub avg_loss_ratio: f32,
}

/// Dropout behavior tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropoutBehaviorMetrics {
    /// Number of training batches with dropout applied
    pub training_dropout_applied: usize,
    /// Number of validation batches with dropout applied
    pub validation_dropout_applied: usize,
    /// Total training batches processed
    pub total_training_batches: usize,
    /// Total validation batches processed
    pub total_validation_batches: usize,
}

impl Default for RegimeMetricsCollector {
    fn default() -> Self {
        Self {
            training_metrics: Vec::new(),
            validation_metrics: Vec::new(),
            regime_distribution: HashMap::new(),
            loss_consistency: LossConsistencyMetrics {
                loss_ratio_std: 0.0,
                max_loss_ratio: 0.0,
                min_loss_ratio: f32::INFINITY,
                overfitting_epochs: 0,
                avg_loss_ratio: 0.0,
            },
            dropout_behavior: DropoutBehaviorMetrics {
                training_dropout_applied: 0,
                validation_dropout_applied: 0,
                total_training_batches: 0,
                total_validation_batches: 0,
            },
        }
    }
}

impl RegimeMetricsCollector {
    /// Add epoch metrics from dual loss result
    pub fn add_epoch_metrics(
        &mut self,
        epoch: usize,
        dual_loss_result: &DualLossResult,
        gradient_norm: f64,
        learning_rate: f64,
    ) {
        let metrics = EpochMetrics {
            epoch,
            training_loss: dual_loss_result.training_loss,
            evaluation_loss: dual_loss_result.evaluation_loss,
            loss_ratio: dual_loss_result.loss_ratio(),
            regime: dual_loss_result.regime,
            regime_calibrated: dual_loss_result.regime_calibrated,
            gradient_norm,
            learning_rate,
        };

        // Update regime distribution
        *self
            .regime_distribution
            .entry(dual_loss_result.regime)
            .or_insert(0) += 1;

        // Track overfitting
        if dual_loss_result.is_overfitting(3.0) {
            self.loss_consistency.overfitting_epochs += 1;
        }

        // Update loss ratio statistics
        let ratio = dual_loss_result.loss_ratio();
        if ratio > self.loss_consistency.max_loss_ratio {
            self.loss_consistency.max_loss_ratio = ratio;
        }
        if ratio < self.loss_consistency.min_loss_ratio {
            self.loss_consistency.min_loss_ratio = ratio;
        }

        self.training_metrics.push(metrics);

        log::info!(
            "📊 Epoch {} Metrics: Train={:.6}, Eval={:.6}, Ratio={:.2}x, Regime={:?}, Calibrated={}",
            epoch,
            dual_loss_result.training_loss,
            dual_loss_result.evaluation_loss,
            ratio,
            dual_loss_result.regime,
            dual_loss_result.regime_calibrated
        );
    }

    /// Finalize metrics calculation
    pub fn finalize_metrics(&mut self) {
        if self.training_metrics.is_empty() {
            return;
        }

        // Calculate loss ratio statistics
        let ratios: Vec<f32> = self.training_metrics.iter().map(|m| m.loss_ratio).collect();
        let sum: f32 = ratios.iter().sum();
        let avg = sum / ratios.len() as f32;

        let variance: f32 =
            ratios.iter().map(|r| (r - avg).powi(2)).sum::<f32>() / ratios.len() as f32;
        let std_dev = variance.sqrt();

        self.loss_consistency.avg_loss_ratio = avg;
        self.loss_consistency.loss_ratio_std = std_dev;

        log::info!("📈 Final Loss Consistency Metrics:");
        log::info!("  - Average loss ratio: {:.2}x", avg);
        log::info!("  - Loss ratio std dev: {:.3}", std_dev);
        log::info!(
            "  - Max loss ratio: {:.2}x",
            self.loss_consistency.max_loss_ratio
        );
        log::info!(
            "  - Min loss ratio: {:.2}x",
            self.loss_consistency.min_loss_ratio
        );
        log::info!(
            "  - Overfitting epochs: {}/{}",
            self.loss_consistency.overfitting_epochs,
            self.training_metrics.len()
        );
    }

    /// Log regime distribution
    pub fn log_regime_distribution(&self) {
        log::info!("🔍 Regime Distribution:");
        for (regime, count) in &self.regime_distribution {
            let percentage = (*count as f32 / self.training_metrics.len() as f32) * 100.0;
            log::info!("  - {:?}: {} epochs ({:.1}%)", regime, count, percentage);
        }
    }

    /// Track dropout behavior
    pub fn track_dropout_behavior(&mut self, is_training: bool, dropout_applied: bool) {
        if is_training {
            self.dropout_behavior.total_training_batches += 1;
            if dropout_applied {
                self.dropout_behavior.training_dropout_applied += 1;
            }
        } else {
            self.dropout_behavior.total_validation_batches += 1;
            if dropout_applied {
                self.dropout_behavior.validation_dropout_applied += 1;
            }
        }
    }

    /// Log dropout consistency report
    pub fn log_dropout_consistency(&self) {
        let training_dropout_rate = if self.dropout_behavior.total_training_batches > 0 {
            self.dropout_behavior.training_dropout_applied as f32
                / self.dropout_behavior.total_training_batches as f32
        } else {
            0.0
        };

        let validation_dropout_rate = if self.dropout_behavior.total_validation_batches > 0 {
            self.dropout_behavior.validation_dropout_applied as f32
                / self.dropout_behavior.total_validation_batches as f32
        } else {
            0.0
        };

        log::info!("🔧 Dropout Consistency Report:");
        log::info!(
            "  - Training dropout rate: {:.1}%",
            training_dropout_rate * 100.0
        );
        log::info!(
            "  - Validation dropout rate: {:.1}%",
            validation_dropout_rate * 100.0
        );
        log::info!(
            "  - Consistency: {}",
            if (training_dropout_rate - validation_dropout_rate).abs() < 0.1 {
                "✅ CONSISTENT"
            } else {
                "⚠️ INCONSISTENT"
            }
        );
    }

    /// Check if the system is performing well
    pub fn validate_system_performance(&self) -> Result<()> {
        let mut issues = Vec::new();

        // Check loss ratio consistency
        if self.loss_consistency.loss_ratio_std > 1.0 {
            issues.push(format!(
                "High loss ratio variance: {:.3} (should be < 1.0)",
                self.loss_consistency.loss_ratio_std
            ));
        }

        // Check for excessive overfitting
        let overfitting_rate =
            self.loss_consistency.overfitting_epochs as f32 / self.training_metrics.len() as f32;
        if overfitting_rate > 0.5 {
            issues.push(format!(
                "High overfitting rate: {:.1}% (should be < 50%)",
                overfitting_rate * 100.0
            ));
        }

        // Check regime distribution balance
        let regime_count = self.regime_distribution.len();
        if regime_count < 2 {
            issues.push("Low regime diversity: only 1 regime detected".to_string());
        }

        if issues.is_empty() {
            log::info!("✅ System performance validation: PASSED");
            Ok(())
        } else {
            log::warn!("⚠️ System performance validation: ISSUES DETECTED");
            for issue in &issues {
                log::warn!("  - {}", issue);
            }
            Err(VangaError::ModelError(format!(
                "System performance validation failed: {}",
                issues.join(", ")
            )))
        }
    }

    /// Generate comprehensive report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Regime Calibration System Report\n\n");

        // Summary
        report.push_str("## Summary\n");
        report.push_str(&format!(
            "- Total epochs: {}\n",
            self.training_metrics.len()
        ));
        report.push_str(&format!(
            "- Average loss ratio: {:.2}x\n",
            self.loss_consistency.avg_loss_ratio
        ));
        report.push_str(&format!(
            "- Loss ratio std dev: {:.3}\n",
            self.loss_consistency.loss_ratio_std
        ));
        report.push_str(&format!(
            "- Overfitting epochs: {}\n",
            self.loss_consistency.overfitting_epochs
        ));
        report.push('\n');

        // Regime distribution
        report.push_str("## Regime Distribution\n");
        for (regime, count) in &self.regime_distribution {
            let percentage = (*count as f32 / self.training_metrics.len() as f32) * 100.0;
            report.push_str(&format!(
                "- {:?}: {} epochs ({:.1}%)\n",
                regime, count, percentage
            ));
        }
        report.push('\n');

        // Dropout consistency
        let training_dropout_rate = if self.dropout_behavior.total_training_batches > 0 {
            self.dropout_behavior.training_dropout_applied as f32
                / self.dropout_behavior.total_training_batches as f32
        } else {
            0.0
        };
        let validation_dropout_rate = if self.dropout_behavior.total_validation_batches > 0 {
            self.dropout_behavior.validation_dropout_applied as f32
                / self.dropout_behavior.total_validation_batches as f32
        } else {
            0.0
        };

        report.push_str("## Dropout Consistency\n");
        report.push_str(&format!(
            "- Training dropout rate: {:.1}%\n",
            training_dropout_rate * 100.0
        ));
        report.push_str(&format!(
            "- Validation dropout rate: {:.1}%\n",
            validation_dropout_rate * 100.0
        ));
        report.push_str(&format!(
            "- Consistency: {}\n",
            if (training_dropout_rate - validation_dropout_rate).abs() < 0.1 {
                "CONSISTENT"
            } else {
                "INCONSISTENT"
            }
        ));

        report
    }
}

/// System validation utilities
pub struct SystemValidator;

impl SystemValidator {
    /// Validate regime calibrator state
    pub fn validate_regime_calibrator(calibrator: &RegimeCalibrator) -> Result<()> {
        if !calibrator.is_calibrated() {
            return Err(VangaError::ModelError(
                "Regime calibrator is not calibrated".to_string(),
            ));
        }

        let progress = calibrator.calibration_progress();
        if progress < 50.0 {
            log::warn!("⚠️ Low calibration progress: {:.1}%", progress);
        }

        log::info!(
            "✅ Regime calibrator validation: PASSED ({:.1}% calibrated)",
            progress
        );
        Ok(())
    }

    /// Validate dual loss system state
    pub fn validate_dual_loss_system(dual_loss_system: &DualLossSystem) -> Result<()> {
        if !dual_loss_system.is_ready() {
            return Err(VangaError::ModelError(
                "Dual loss system is not ready".to_string(),
            ));
        }

        let progress = dual_loss_system.get_calibration_progress();
        log::info!(
            "✅ Dual loss system validation: PASSED ({:.1}% calibrated)",
            progress
        );
        Ok(())
    }

    /// Comprehensive system validation
    pub fn validate_complete_system(
        dual_loss_system: &DualLossSystem,
        metrics_collector: &RegimeMetricsCollector,
    ) -> Result<()> {
        // Validate dual loss system
        Self::validate_dual_loss_system(dual_loss_system)?;

        // Validate regime calibrator if available
        if let Some(calibrator) = dual_loss_system.get_regime_calibrator() {
            Self::validate_regime_calibrator(calibrator)?;
        }

        // Validate system performance
        metrics_collector.validate_system_performance()?;

        log::info!("🎉 Complete system validation: PASSED");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::objective::MarketRegime;

    #[test]
    fn test_metrics_collector() {
        let mut collector = RegimeMetricsCollector::default();

        let dual_loss_result = DualLossResult::new(1.0, 1.2, MarketRegime::MediumVolatility, true);

        collector.add_epoch_metrics(0, &dual_loss_result, 0.5, 0.001);
        collector.finalize_metrics();

        assert_eq!(collector.training_metrics.len(), 1);
        assert_eq!(collector.loss_consistency.avg_loss_ratio, 1.2 / 1.0);
    }

    #[test]
    fn test_dropout_tracking() {
        let mut collector = RegimeMetricsCollector::default();

        collector.track_dropout_behavior(true, true); // Training with dropout
        collector.track_dropout_behavior(false, false); // Validation without dropout

        assert_eq!(collector.dropout_behavior.training_dropout_applied, 1);
        assert_eq!(collector.dropout_behavior.validation_dropout_applied, 0);
    }
}
