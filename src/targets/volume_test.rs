use crate::config::model::TargetsConfig;
use crate::targets::volume::*;

#[test]
fn test_volume_logarithmic_classification() {
    let config = VolumeConfig::default();
    let thresholds = calculate_log_volume_thresholds(&config).unwrap();

    // Test symmetric classification
    let sequence_volume = 1000.0_f64;

    // Test 2x increase (should be symmetric to 0.5x decrease)
    let high_volume = 2000.0_f64;
    let low_volume = 500.0_f64;

    let high_class = classify_volume_log_ratio((high_volume / sequence_volume).ln(), &thresholds);
    let low_class = classify_volume_log_ratio((low_volume / sequence_volume).ln(), &thresholds);

    // Should be symmetric around medium class (2)
    assert!(high_class > 2, "2x volume increase should be above medium");
    assert!(low_class < 2, "0.5x volume decrease should be below medium");

    // Test medium volume (no change)
    let medium_class = classify_volume_log_ratio(0.0, &thresholds); // ln(1.0) = 0
    assert_eq!(medium_class, 2, "No volume change should be medium class");
}

#[test]
fn test_volume_balanced_distribution() {
    let _config = TargetsConfig::default();

    // Create test volume data with various patterns
    let test_volumes = [
        // Very low volume period
        100.0, 120.0, 90.0, 110.0, 95.0, // Low volume period
        200.0, 250.0, 180.0, 220.0, 190.0, // Medium volume period
        500.0, 550.0, 480.0, 520.0, 490.0, // High volume period
        1000.0, 1100.0, 950.0, 1050.0, 980.0, // Very high volume period
        2000.0, 2200.0, 1900.0, 2100.0, 1950.0,
    ];

    let volume_config = VolumeConfig::default();
    let thresholds = calculate_log_volume_thresholds(&volume_config).unwrap();

    // Test classification across different volume regimes
    let mut class_counts = [0usize; 5];

    for i in 0..5 {
        let sequence_start = i * 5;
        let sequence_end = sequence_start + 3;
        let horizon_start = sequence_end;
        let horizon_end = horizon_start + 2;

        if horizon_end <= test_volumes.len() {
            let sequence_volumes = &test_volumes[sequence_start..sequence_end];
            let horizon_volumes = &test_volumes[horizon_start..horizon_end];

            let result = classify_volume_regime(
                sequence_volumes,
                horizon_volumes,
                &thresholds,
                &volume_config,
            );
            if let Ok(class) = result {
                if (0..5).contains(&class) {
                    class_counts[class as usize] += 1;
                }
            }
        }
    }

    // Check that we got some distribution across classes
    let total_classifications = class_counts.iter().sum::<usize>();
    assert!(
        total_classifications > 0,
        "Should have some valid classifications"
    );

    // Check that all classes are in valid range
    for &count in &class_counts {
        assert!(
            count <= total_classifications,
            "Class count should not exceed total"
        );
    }
}

#[test]
fn test_volume_smoothing() {
    let volumes = vec![1000.0, 1200.0, 800.0, 1100.0, 900.0, 1300.0];

    // Test different smoothing periods
    let smoothed_1 = calculate_smoothed_volume(&volumes, 1).unwrap();
    let smoothed_3 = calculate_smoothed_volume(&volumes, 3).unwrap();
    let smoothed_5 = calculate_smoothed_volume(&volumes, 5).unwrap();

    // All should be positive
    assert!(smoothed_1 > 0.0);
    assert!(smoothed_3 > 0.0);
    assert!(smoothed_5 > 0.0);

    // Smoothing should reduce impact of outliers
    // With period 1, it should just be the average
    let expected_avg = volumes.iter().sum::<f64>() / volumes.len() as f64;
    assert!(
        (smoothed_1 - expected_avg).abs() < 1.0,
        "Period 1 should approximate full average"
    );
}

#[test]
fn test_volume_distribution_stats() {
    let volumes = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let stats = calculate_volume_distribution_stats(&volumes);

    assert_eq!(stats.mean, 300.0);
    assert_eq!(stats.min, 100.0);
    assert_eq!(stats.max, 500.0);
    assert!(stats.std_dev > 0.0);

    // Test with empty data
    let empty_volumes: Vec<f64> = vec![];
    let empty_stats = calculate_volume_distribution_stats(&empty_volumes);
    assert_eq!(empty_stats.mean, 0.0);
    assert_eq!(empty_stats.std_dev, 0.0);
}

#[test]
fn test_volume_thresholds_calculation() {
    let config = VolumeConfig {
        bandwidth_size: 0.4,
        extreme_multiplier: 2.0,
        smoothing_periods: 3,
    };

    let result = calculate_log_volume_thresholds(&config);
    assert!(result.is_ok());

    let thresholds = result.unwrap();

    // Check threshold ordering
    assert!(thresholds.very_low_max < thresholds.low_max);
    assert!(thresholds.low_max < thresholds.medium_max);
    assert!(thresholds.medium_max < thresholds.high_max);

    // Check symmetry around zero
    assert!((thresholds.very_low_max + thresholds.high_max).abs() < 1e-10);
    assert!((thresholds.low_max + thresholds.medium_max).abs() < 1e-10);
}

#[test]
fn test_volume_class_names() {
    let class_names = get_volume_class_names();
    assert_eq!(class_names.len(), 5);
    assert_eq!(class_names[0], "VERY_LOW");
    assert_eq!(class_names[1], "LOW");
    assert_eq!(class_names[2], "MEDIUM");
    assert_eq!(class_names[3], "HIGH");
    assert_eq!(class_names[4], "VERY_HIGH");
}

#[test]
fn test_volume_edge_cases() {
    let config = VolumeConfig::default();
    let thresholds = calculate_log_volume_thresholds(&config).unwrap();

    // Test with zero/negative volumes
    let zero_volumes = vec![0.0, 0.0, 0.0];
    let positive_volumes = vec![100.0, 200.0, 150.0];

    let result = classify_volume_regime(&zero_volumes, &positive_volumes, &thresholds, &config);
    assert!(result.is_ok()); // Should handle gracefully with default

    let result = classify_volume_regime(&positive_volumes, &zero_volumes, &thresholds, &config);
    assert!(result.is_ok()); // Should handle gracefully with default

    // Test with empty data
    let empty_volumes: Vec<f64> = vec![];
    let result = classify_volume_regime(&empty_volumes, &positive_volumes, &thresholds, &config);
    assert!(result.is_err(), "Should fail with empty sequence volumes");

    let result = classify_volume_regime(&positive_volumes, &empty_volumes, &thresholds, &config);
    assert!(result.is_err(), "Should fail with empty horizon volumes");
}

#[test]
fn test_volume_reconstruction() {
    let probabilities = vec![0.1, 0.2, 0.4, 0.2, 0.1];
    let sequence_volume = 1000.0;
    let config = VolumeConfig::default();
    let thresholds = calculate_log_volume_thresholds(&config).unwrap();

    let result = reconstruct_volume(&probabilities, sequence_volume, &thresholds);
    assert!(result.is_ok());

    let reconstruction = result.unwrap();
    assert_eq!(reconstruction.probabilities.len(), 5);
    assert_eq!(reconstruction.most_likely_class, 2); // Medium has highest probability
    assert!(reconstruction.confidence > 0.0);
    assert!(reconstruction.confidence <= 1.0);
    assert_eq!(reconstruction.volume_ratio_ranges.len(), 5);
    assert_eq!(reconstruction.volume_ranges.len(), 5);
    assert_eq!(reconstruction.sequence_volume, sequence_volume);

    // Check that volume ranges make sense
    for i in 0..5 {
        let ratio_range = &reconstruction.volume_ratio_ranges[i];
        let volume_range = &reconstruction.volume_ranges[i];

        assert!(ratio_range[0] >= 0.0, "Volume ratio should be non-negative");
        assert!(volume_range[0] >= 0.0, "Volume should be non-negative");

        if !volume_range[1].is_infinite() {
            assert!(
                volume_range[0] <= volume_range[1],
                "Volume range should be ordered"
            );
        }
    }

    // Test with invalid probabilities
    let invalid_probs = vec![0.1, 0.2, 0.3]; // Wrong length
    let result = reconstruct_volume(&invalid_probs, sequence_volume, &thresholds);
    assert!(
        result.is_err(),
        "Should fail with wrong number of probabilities"
    );
}

#[test]
fn test_volume_logarithmic_symmetry() {
    let config = VolumeConfig::default();
    let thresholds = calculate_log_volume_thresholds(&config).unwrap();

    // Test that 2x increase and 0.5x decrease are treated symmetrically
    let log_ratio_2x = (2.0_f64).ln();
    let log_ratio_half = (0.5_f64).ln();

    let class_2x = classify_volume_log_ratio(log_ratio_2x, &thresholds);
    let class_half = classify_volume_log_ratio(log_ratio_half, &thresholds);

    // Should be symmetric around medium (class 2)
    assert_eq!(
        class_2x + class_half,
        4,
        "2x and 0.5x should be symmetric around medium"
    );

    // Test that ln(1.0) = 0 gives medium class
    let class_no_change = classify_volume_log_ratio(0.0, &thresholds);
    assert_eq!(
        class_no_change, 2,
        "No volume change should be medium class"
    );
}
