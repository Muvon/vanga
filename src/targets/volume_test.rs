//! Volume target generation tests
//!
//! Tests the actual volume classification functionality with real market scenarios

#[cfg(test)]
mod tests {
    use super::super::calibration::VolumeParams;
    use super::super::volume::*;
    use polars::prelude::*;

    /// Create test DataFrame from OHLCV data
    fn create_test_dataframe(ohlcv_data: Vec<(f64, f64, f64, f64, f64)>) -> DataFrame {
        let timestamps: Vec<i64> = (0..ohlcv_data.len()).map(|i| i as i64 * 3600).collect();
        let opens: Vec<f64> = ohlcv_data.iter().map(|(o, _, _, _, _)| *o).collect();
        let highs: Vec<f64> = ohlcv_data.iter().map(|(_, h, _, _, _)| *h).collect();
        let lows: Vec<f64> = ohlcv_data.iter().map(|(_, _, l, _, _)| *l).collect();
        let closes: Vec<f64> = ohlcv_data.iter().map(|(_, _, _, c, _)| *c).collect();
        let volumes: Vec<f64> = ohlcv_data.iter().map(|(_, _, _, _, v)| *v).collect();

        DataFrame::new(vec![
            Series::new("timestamp", timestamps),
            Series::new("open", opens),
            Series::new("high", highs),
            Series::new("low", lows),
            Series::new("close", closes),
            Series::new("volume", volumes),
        ])
        .unwrap()
    }

    #[test]
    fn test_calculate_smoothed_volume() {
        // Test with increasing volume trend
        let volumes = vec![1000.0, 1200.0, 1400.0, 1600.0, 1800.0];
        let smoothed = calculate_smoothed_volume(&volumes, 3).unwrap();

        // Should be around the average of the middle values
        assert!(
            smoothed > 1000.0,
            "Smoothed volume should be greater than minimum"
        );
        assert!(
            smoothed < 1800.0,
            "Smoothed volume should be less than maximum"
        );

        println!("Smoothed volume: {:.2}", smoothed);

        // Test with single value
        let single_volume = vec![1500.0];
        let smoothed_single = calculate_smoothed_volume(&single_volume, 1).unwrap();
        assert_eq!(
            smoothed_single, 1500.0,
            "Single volume should return itself"
        );

        // Test with empty volumes
        let empty_volumes = vec![];
        let result = calculate_smoothed_volume(&empty_volumes, 3);
        assert!(result.is_err(), "Empty volumes should return error");
    }

    #[test]
    fn test_calculate_volume_distribution_stats() {
        let volumes = vec![
            1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0,
        ];
        let stats = calculate_volume_distribution_stats(&volumes);

        assert_eq!(stats.mean, 3250.0, "Mean should be calculated correctly");
        assert!(stats.std_dev > 0.0, "Standard deviation should be positive");
        assert!(stats.min < stats.max, "Min should be < max");

        println!(
            "Volume distribution stats: mean={:.1}, std_dev={:.1}, min={:.1}, max={:.1}",
            stats.mean, stats.std_dev, stats.min, stats.max
        );

        // Test with empty volumes
        let empty_stats = calculate_volume_distribution_stats(&[]);
        assert_eq!(empty_stats.mean, 0.0, "Empty volumes should have zero mean");
    }

    #[test]
    fn test_classify_volume_regime() {
        let config = VolumeConfig {
            bandwidth_size: 0.8,
            extreme_multiplier: 2.0,
            smoothing_periods: 3,
        };

        let thresholds = calculate_log_volume_thresholds(&config).unwrap();

        // Low volume scenario
        let low_volumes = vec![500.0, 600.0, 700.0];
        let low_horizon_volumes = vec![400.0, 500.0];
        let low_class =
            classify_volume_regime(&low_volumes, &low_horizon_volumes, &thresholds, &config)
                .unwrap();

        // High volume scenario
        let high_volumes = vec![1000.0, 1200.0, 1400.0];
        let high_horizon_volumes = vec![2000.0, 2500.0, 3000.0];
        let high_class =
            classify_volume_regime(&high_volumes, &high_horizon_volumes, &thresholds, &config)
                .unwrap();

        assert!(
            (0..=4).contains(&low_class),
            "Low volume class should be 0-4, got {}",
            low_class
        );
        assert!(
            (0..=4).contains(&high_class),
            "High volume class should be 0-4, got {}",
            high_class
        );
        assert!(
            high_class >= low_class,
            "High volume class ({}) should be >= low volume class ({})",
            high_class,
            low_class
        );

        println!(
            "Low volume class: {}, High volume class: {}",
            low_class, high_class
        );
    }

    #[test]
    fn test_calculate_log_volume_thresholds() {
        let config = VolumeConfig {
            bandwidth_size: 0.8,
            extreme_multiplier: 2.0,
            smoothing_periods: 3,
        };

        let thresholds = calculate_log_volume_thresholds(&config).unwrap();

        // Thresholds should be in ascending order
        assert!(
            thresholds.very_low_max < thresholds.low_max,
            "Very low max should be < low max"
        );
        assert!(
            thresholds.low_max < thresholds.medium_max,
            "Low max should be < medium max"
        );
        assert!(
            thresholds.medium_max < thresholds.high_max,
            "Medium max should be < high max"
        );

        println!("Volume thresholds: very_low_max={:.3}, low_max={:.3}, medium_max={:.3}, high_max={:.3}",
                 thresholds.very_low_max, thresholds.low_max,
                 thresholds.medium_max, thresholds.high_max);
    }

    #[test]
    fn test_classify_volume_log_ratio() {
        let config = VolumeConfig {
            bandwidth_size: 0.8,
            extreme_multiplier: 2.0,
            smoothing_periods: 3,
        };
        let thresholds = calculate_log_volume_thresholds(&config).unwrap();

        // Test different log ratios
        let very_low_ratio = -1.5;
        let low_ratio = -0.5;
        let medium_ratio = 0.0;
        let high_ratio = 0.5;
        let very_high_ratio = 1.5;

        let very_low_class = classify_volume_log_ratio(very_low_ratio, &thresholds);
        let low_class = classify_volume_log_ratio(low_ratio, &thresholds);
        let medium_class = classify_volume_log_ratio(medium_ratio, &thresholds);
        let high_class = classify_volume_log_ratio(high_ratio, &thresholds);
        let very_high_class = classify_volume_log_ratio(very_high_ratio, &thresholds);

        // Classes should be in ascending order
        assert!(
            very_low_class <= low_class,
            "Very low class should be <= low class"
        );
        assert!(
            low_class <= medium_class,
            "Low class should be <= medium class"
        );
        assert!(
            medium_class <= high_class,
            "Medium class should be <= high class"
        );
        assert!(
            high_class <= very_high_class,
            "High class should be <= very high class"
        );

        // All classes should be valid (0-4)
        for &class in &[
            very_low_class,
            low_class,
            medium_class,
            high_class,
            very_high_class,
        ] {
            assert!(
                (0..=4).contains(&class),
                "Volume class should be 0-4, got {}",
                class
            );
        }

        println!(
            "Volume classes: very_low={}, low={}, medium={}, high={}, very_high={}",
            very_low_class, low_class, medium_class, high_class, very_high_class
        );
    }

    #[test]
    fn test_generate_volume_targets_with_calibrated_params() {
        // Create market data with varying volume patterns
        let df = create_test_dataframe(vec![
            // Low volume period
            (100.0, 101.0, 99.0, 100.5, 500.0),
            (100.5, 101.5, 99.5, 101.0, 600.0),
            (101.0, 102.0, 100.0, 101.5, 700.0),
            (101.5, 102.5, 100.5, 102.0, 800.0),
            // Increasing volume
            (102.0, 103.0, 101.0, 102.5, 1200.0),
            (102.5, 103.5, 101.5, 103.0, 1500.0),
            // High volume period
            (103.0, 104.0, 102.0, 103.5, 2500.0),
            (103.5, 104.5, 102.5, 104.0, 3000.0),
            (104.0, 105.0, 103.0, 104.5, 3500.0),
            // Decreasing volume
            (104.5, 105.5, 103.5, 105.0, 2000.0),
            (105.0, 106.0, 104.0, 105.5, 1500.0),
            (105.5, 106.5, 104.5, 106.0, 1000.0),
        ]);

        let horizons = vec!["1h".to_string()];
        let sequence_indices = vec![0, 3, 6]; // Different volume periods
        let sequence_length = 3;

        let params = VolumeParams {
            bandwidth: 0.8,
            extreme_multiplier: 2.0,
            smoothing_periods: 3,
            balance: Default::default(),
        };

        let result = generate_volume_targets_with_calibrated_params(
            &df,
            &horizons,
            &sequence_indices,
            sequence_length,
            &params,
        );

        assert!(
            result.is_ok(),
            "Volume target generation should succeed: {:?}",
            result.err()
        );
        let targets = result.unwrap();

        assert!(targets.contains_key("1h"), "Should contain 1h horizon");
        let horizon_targets = &targets["1h"];
        assert_eq!(
            horizon_targets.len(),
            sequence_indices.len(),
            "Should have targets for all sequences"
        );

        // Verify all targets are valid volume classes (0-4)
        for (i, &target) in horizon_targets.iter().enumerate() {
            assert!(
                (0..=4).contains(&target),
                "Volume target {} should be 0-4 (VERY_LOW to VERY_HIGH), got {} at sequence {}",
                i,
                target,
                sequence_indices[i]
            );
        }

        println!("Generated volume targets: {:?}", horizon_targets);

        // Expect some progression from low to high volume
        if horizon_targets.len() >= 3 {
            // High volume period should generally have higher class than low volume period
            println!(
                "Low volume class: {}, High volume class: {}",
                horizon_targets[0], horizon_targets[2]
            );
        }
    }

    #[test]
    fn test_volume_class_names() {
        let class_names = get_volume_class_names();
        assert_eq!(class_names.len(), 5, "Should have 5 volume classes");
        assert_eq!(class_names[0], "VERY_LOW", "Class 0 should be VERY_LOW");
        assert_eq!(class_names[1], "LOW", "Class 1 should be LOW");
        assert_eq!(class_names[2], "MEDIUM", "Class 2 should be MEDIUM");
        assert_eq!(class_names[3], "HIGH", "Class 3 should be HIGH");
        assert_eq!(class_names[4], "VERY_HIGH", "Class 4 should be VERY_HIGH");
    }

    #[test]
    fn test_bandwidth_parameter_effect() {
        let sequence_volumes = vec![1000.0, 1200.0, 1400.0];
        let horizon_volumes = vec![1800.0, 2000.0];

        // Small bandwidth - more sensitive to volume changes
        let small_bandwidth_config = VolumeConfig {
            bandwidth_size: 0.4,
            extreme_multiplier: 2.0,
            smoothing_periods: 3,
        };

        // Large bandwidth - less sensitive to volume changes
        let large_bandwidth_config = VolumeConfig {
            bandwidth_size: 1.2,
            extreme_multiplier: 2.0,
            smoothing_periods: 3,
        };

        let small_thresholds = calculate_log_volume_thresholds(&small_bandwidth_config).unwrap();
        let large_thresholds = calculate_log_volume_thresholds(&large_bandwidth_config).unwrap();

        let small_class = classify_volume_regime(
            &sequence_volumes,
            &horizon_volumes,
            &small_thresholds,
            &small_bandwidth_config,
        )
        .unwrap();
        let large_class = classify_volume_regime(
            &sequence_volumes,
            &horizon_volumes,
            &large_thresholds,
            &large_bandwidth_config,
        )
        .unwrap();

        println!(
            "Small bandwidth class: {}, Large bandwidth class: {}",
            small_class, large_class
        );

        // Both should be valid classes
        assert!((0..=4).contains(&small_class));
        assert!((0..=4).contains(&large_class));

        // Small bandwidth might be more sensitive to the volume increase
        // (though this depends on the specific thresholds)
    }

    #[test]
    fn test_calibrate_volume_sensitivity() {
        let volume_data = vec![
            500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0,
            2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0,
        ];

        let result = calibrate_volume_sensitivity(&volume_data, 3, 1, 0.2);
        assert!(
            result.is_ok(),
            "Volume sensitivity calibration should succeed"
        );

        let bandwidth = result.unwrap();
        assert!(bandwidth > 0.0, "Calibrated bandwidth should be positive");
        assert!(bandwidth < 5.0, "Calibrated bandwidth should be reasonable");

        println!("Calibrated volume bandwidth: {:.3}", bandwidth);
    }

    #[test]
    fn test_edge_cases() {
        let config = VolumeConfig {
            bandwidth_size: 0.8,
            extreme_multiplier: 2.0,
            smoothing_periods: 3,
        };

        let thresholds = calculate_log_volume_thresholds(&config).unwrap();

        // Test with minimal data
        let minimal_sequence = vec![1000.0];
        let minimal_horizon = vec![1200.0];
        let result =
            classify_volume_regime(&minimal_sequence, &minimal_horizon, &thresholds, &config);
        assert!(result.is_ok(), "Should handle minimal data gracefully");

        // Test with zero volumes
        let zero_sequence = vec![0.0, 0.0, 0.0];
        let zero_horizon = vec![0.0, 0.0];
        let result = classify_volume_regime(&zero_sequence, &zero_horizon, &thresholds, &config);
        assert!(result.is_ok(), "Should handle zero volumes gracefully");

        // Test with very high volumes
        let high_sequence = vec![1000000.0, 1100000.0, 1200000.0];
        let high_horizon = vec![1500000.0, 1600000.0];
        let result = classify_volume_regime(&high_sequence, &high_horizon, &thresholds, &config);
        assert!(result.is_ok(), "Should handle very high volumes gracefully");

        // Test smoothed volume with insufficient data
        let insufficient_data = vec![1000.0];
        let smoothed = calculate_smoothed_volume(&insufficient_data, 3);
        assert!(
            smoothed.is_ok(),
            "Should handle insufficient data for smoothing"
        );
        assert_eq!(smoothed.unwrap(), 1000.0, "Should return the single value");
    }

    #[test]
    fn test_reconstruct_volume() {
        let params = VolumeParams {
            bandwidth: 0.8,
            extreme_multiplier: 2.0,
            smoothing_periods: 3,
            balance: Default::default(),
        };

        // Test reconstruction with clear high volume signal
        let high_vol_probs = vec![0.05, 0.05, 0.1, 0.2, 0.6]; // Strong VERY_HIGH signal
        let reconstruction = reconstruct_volume(&high_vol_probs, 1000.0, &params).unwrap();

        assert_eq!(
            reconstruction.most_likely_class, 4,
            "Should predict VERY_HIGH class"
        );
        assert!(
            reconstruction.confidence > 0.5,
            "Should have high confidence"
        );
        assert!(
            reconstruction.expected_volume_ratio > 0.0,
            "Should have positive volume ratio"
        );

        // Test reconstruction with low volume signal
        let low_vol_probs = vec![0.6, 0.2, 0.1, 0.05, 0.05]; // Strong VERY_LOW signal
        let reconstruction = reconstruct_volume(&low_vol_probs, 5000.0, &params).unwrap();

        assert_eq!(
            reconstruction.most_likely_class, 0,
            "Should predict VERY_LOW class"
        );
        assert!(
            reconstruction.confidence > 0.5,
            "Should have high confidence"
        );

        // Test reconstruction with unclear probabilities
        let unclear_probs = vec![0.2, 0.2, 0.2, 0.2, 0.2]; // Equal probabilities
        let reconstruction = reconstruct_volume(&unclear_probs, 1000.0, &params).unwrap();

        assert!(
            reconstruction.confidence < 0.3,
            "Should have low confidence for unclear signal"
        );

        println!(
            "High vol reconstruction: class={}, confidence={:.3}, ratio={:.3}",
            reconstruction.most_likely_class,
            reconstruction.confidence,
            reconstruction.expected_volume_ratio
        );
    }

    #[test]
    fn test_volume_regime_consistency() {
        let config = VolumeConfig {
            bandwidth_size: 0.8,
            extreme_multiplier: 2.0,
            smoothing_periods: 3,
        };

        let thresholds = calculate_log_volume_thresholds(&config).unwrap();

        // Test that similar volume patterns produce consistent classifications
        let base_sequence = vec![1000.0, 1100.0, 1200.0];
        let base_horizon = vec![1300.0, 1400.0];
        let base_class =
            classify_volume_regime(&base_sequence, &base_horizon, &thresholds, &config).unwrap();

        // Slightly different but similar pattern
        let similar_sequence = vec![1050.0, 1150.0, 1250.0];
        let similar_horizon = vec![1350.0, 1450.0];
        let similar_class =
            classify_volume_regime(&similar_sequence, &similar_horizon, &thresholds, &config)
                .unwrap();

        // Should be the same or adjacent classes
        let class_diff = (base_class - similar_class).abs();
        assert!(
            class_diff <= 1,
            "Similar volume patterns should produce similar classes, got {} and {}",
            base_class,
            similar_class
        );

        println!(
            "Base class: {}, Similar class: {}",
            base_class, similar_class
        );
    }
}
