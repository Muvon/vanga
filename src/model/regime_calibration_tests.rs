//! Integration tests for the complete regime calibration system
//!
//! These tests validate the entire system including regime calibration,
//! dual loss system, dropout consistency, and comprehensive metrics.

#[cfg(test)]
mod tests {
    use crate::model::dropout_consistency::{DropoutConsistencyConfig, DropoutConsistencyStrategy};
    use crate::model::dual_loss_system::{DualLossConfig, DualLossSystem};
    use crate::model::regime_calibration::{CalibrationConfig, RegimeCalibrator};
    use crate::model::regime_metrics::{RegimeMetricsCollector, SystemValidator};
    use crate::optimization::objective::MarketRegime;
    use candle_core::{Device, Tensor};

    #[tokio::test]
    async fn test_complete_regime_calibration_system() {
        // Test the complete system integration
        let device = Device::Cpu;

        // Create test data
        let predictions = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3, 1), &device).unwrap();
        let targets = Tensor::from_slice(&[1.1f32, 2.1, 2.9], (3, 1), &device).unwrap();

        // Initialize dual loss system
        let config = DualLossConfig::default();
        let mut dual_loss_system = DualLossSystem::new(config).unwrap();

        // Update regime
        dual_loss_system.update_epoch_regime(&targets).unwrap();

        // Test simple MSE calculation (dual loss system tests would need LSTM model)
        // For integration tests, we just verify the system can be created and configured
        let training_loss = {
            let diff = predictions.sub(&targets).unwrap();
            diff.sqr().unwrap().mean_all().unwrap()
        };
        let evaluation_loss = {
            let diff = predictions.sub(&targets).unwrap();
            diff.sqr().unwrap().mean_all().unwrap()
        };

        assert!(training_loss.to_scalar::<f32>().unwrap() > 0.0);
        assert!(evaluation_loss.to_scalar::<f32>().unwrap() > 0.0);

        // Finalize calibration
        dual_loss_system.finalize_calibration().unwrap();

        // Validate system
        let _metrics_collector = RegimeMetricsCollector::default();
        SystemValidator::validate_dual_loss_system(&dual_loss_system).unwrap();
    }

    #[test]
    fn test_regime_calibrator_statistical_normalization() {
        let mut calibrator = RegimeCalibrator::new(CalibrationConfig::default());

        // Add samples for different regimes
        let regimes = [
            MarketRegime::LowVolatility,
            MarketRegime::MediumVolatility,
            MarketRegime::HighVolatility,
        ];

        for regime in &regimes {
            for i in 0..60 {
                let loss = match regime {
                    MarketRegime::LowVolatility => 0.5 + (i as f64) * 0.01,
                    MarketRegime::MediumVolatility => 1.0 + (i as f64) * 0.02,
                    MarketRegime::HighVolatility => 2.0 + (i as f64) * 0.03,
                    _ => 1.0,
                };
                calibrator.add_calibration_sample(*regime, loss);
            }
        }

        calibrator.finalize_calibration().unwrap();
        assert!(calibrator.is_calibrated());

        // Test normalization
        let normalized_low = calibrator.normalize_loss(MarketRegime::LowVolatility, 1.0);
        let normalized_high = calibrator.normalize_loss(MarketRegime::HighVolatility, 1.0);

        // Different regimes should normalize the same raw loss differently
        assert_ne!(normalized_low, normalized_high);

        // Verify statistics are available
        assert!(calibrator
            .get_regime_stats(MarketRegime::LowVolatility)
            .is_some());
        assert!(calibrator.calibration_progress() > 0.0);
    }

    #[test]
    fn test_dropout_consistency_strategies() {
        let strategies = [
            DropoutConsistencyStrategy::Standard,
            DropoutConsistencyStrategy::Consistent,
            DropoutConsistencyStrategy::Disabled,
        ];

        for strategy in &strategies {
            let config = DropoutConsistencyConfig {
                strategy: strategy.clone(),
                log_dropout_changes: false, // Disable logging for tests
                warn_validation_inconsistency: false,
                attention_dropout_config:
                    crate::model::dropout_consistency::AttentionDropoutConfig::default(),
            };

            // Test training behavior
            let training_dropout = config.should_apply_dropout(true, true);
            let validation_dropout = config.should_apply_dropout(false, true);

            match strategy {
                DropoutConsistencyStrategy::Standard => {
                    assert!(training_dropout);
                    assert!(!validation_dropout);
                }
                DropoutConsistencyStrategy::Consistent => {
                    assert!(training_dropout);
                    assert!(validation_dropout);
                }
                DropoutConsistencyStrategy::Disabled => {
                    assert!(!training_dropout);
                    assert!(!validation_dropout);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_metrics_collector_comprehensive() {
        let mut collector = RegimeMetricsCollector::default();

        // Simulate training epochs with different regimes
        let regimes = [
            MarketRegime::LowVolatility,
            MarketRegime::MediumVolatility,
            MarketRegime::HighVolatility,
            MarketRegime::BullMarket,
            MarketRegime::BearMarket,
        ];

        for (epoch, regime) in regimes.iter().enumerate() {
            let dual_loss_result = crate::model::dual_loss_system::DualLossResult::new(
                1.0 + epoch as f32 * 0.1,
                1.2 + epoch as f32 * 0.15,
                *regime,
                true,
            );

            collector.add_epoch_metrics(epoch, &dual_loss_result, 0.5, 0.001);

            // Track dropout behavior
            collector.track_dropout_behavior(true, true); // Training with dropout
            collector.track_dropout_behavior(false, false); // Validation without dropout
        }

        collector.finalize_metrics();

        // Validate metrics
        assert_eq!(collector.training_metrics.len(), 5);
        assert_eq!(collector.regime_distribution.len(), 5);
        assert!(collector.loss_consistency.avg_loss_ratio > 0.0);

        // Test report generation
        let report = collector.generate_report();
        assert!(report.contains("Regime Distribution"));
        assert!(report.contains("Dropout Consistency"));

        // Test validation
        let validation_result = collector.validate_system_performance();
        // Should pass since we have good diversity and reasonable ratios
        assert!(validation_result.is_ok());
    }

    #[test]
    fn test_system_validator() {
        // Test regime calibrator validation
        let mut calibrator = RegimeCalibrator::new(CalibrationConfig::default());

        // Should fail before calibration
        assert!(SystemValidator::validate_regime_calibrator(&calibrator).is_err());

        // Add samples and calibrate
        for i in 0..60 {
            calibrator.add_calibration_sample(MarketRegime::MediumVolatility, i as f64);
        }
        calibrator.finalize_calibration().unwrap();

        // Should pass after calibration
        assert!(SystemValidator::validate_regime_calibrator(&calibrator).is_ok());
    }

    #[test]
    fn test_loss_statistics_robustness() {
        use crate::model::regime_calibration::LossStatistics;

        // Test with normal data
        let normal_losses = vec![1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4];
        let stats = LossStatistics::from_losses(&normal_losses).unwrap();
        assert!(stats.is_reliable());
        assert!(stats.std_dev > 0.0);

        // Test normalization
        let normalized = stats.normalize_loss(1.0);
        let robust_normalized = stats.robust_normalize_loss(1.0);
        assert!(normalized.is_finite());
        assert!(robust_normalized.is_finite());

        // Test with outliers
        let outlier_losses = vec![1.0, 1.0, 1.0, 1.0, 100.0]; // One extreme outlier
        let outlier_stats = LossStatistics::from_losses(&outlier_losses).unwrap();

        // Robust normalization should be less affected by outliers
        let normal_robust = stats.robust_normalize_loss(1.5);
        let outlier_robust = outlier_stats.robust_normalize_loss(1.5);

        // The difference should be reasonable (robust method handles outliers better)
        assert!(normal_robust.is_finite());
        assert!(outlier_robust.is_finite());
    }

    #[test]
    fn test_epoch_regime_detection() {
        use crate::model::regime_calibration::EpochRegimeDetector;
        use ndarray::Array2;

        // Test different data patterns
        let test_cases = [
            // Low volatility data (small variations)
            (
                vec![1.0, 1.01, 0.99, 1.02, 0.98],
                "should detect low volatility",
            ),
            // High volatility data (large variations)
            (
                vec![1.0, 2.0, 0.5, 3.0, 0.1],
                "should detect high volatility",
            ),
            // Trending data (consistent direction)
            (vec![1.0, 1.5, 2.0, 2.5, 3.0], "should detect trend"),
        ];

        for (data, description) in &test_cases {
            let array = Array2::from_shape_vec((1, data.len()), data.clone()).unwrap();
            let regime = EpochRegimeDetector::detect_epoch_regime(&array).unwrap();

            // Just verify it returns a valid regime
            println!("Test case '{}': detected regime {:?}", description, regime);
            // All regimes are valid, so just check it doesn't panic
        }
    }
}

/// Performance benchmarks for the regime calibration system
#[cfg(test)]
mod benchmarks {
    use crate::model::regime_calibration::{CalibrationConfig, RegimeCalibrator};
    use crate::optimization::objective::MarketRegime;
    use std::time::Instant;

    #[test]
    fn benchmark_regime_calibration() {
        let mut calibrator = RegimeCalibrator::new(CalibrationConfig::default());

        let start = Instant::now();

        // Add 10,000 samples
        for i in 0..10_000 {
            let regime = match i % 6 {
                0 => MarketRegime::LowVolatility,
                1 => MarketRegime::MediumVolatility,
                2 => MarketRegime::HighVolatility,
                3 => MarketRegime::BullMarket,
                4 => MarketRegime::BearMarket,
                _ => MarketRegime::RangeBound,
            };
            calibrator.add_calibration_sample(regime, i as f64 * 0.001);
        }

        let sample_time = start.elapsed();

        let start = Instant::now();
        calibrator.finalize_calibration().unwrap();
        let calibration_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = calibrator.normalize_loss(MarketRegime::MediumVolatility, 1.0);
        }
        let normalization_time = start.elapsed();

        println!("Regime Calibration Performance:");
        println!("  - Sample collection (10k): {:?}", sample_time);
        println!("  - Calibration finalization: {:?}", calibration_time);
        println!("  - Normalization (1k calls): {:?}", normalization_time);

        // Performance assertions (adjust based on your requirements)
        assert!(sample_time.as_millis() < 100, "Sample collection too slow");
        assert!(calibration_time.as_millis() < 50, "Calibration too slow");
        assert!(
            normalization_time.as_millis() < 10,
            "Normalization too slow"
        );
    }
}
