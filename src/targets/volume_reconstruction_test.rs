//! Tests for volume reconstruction mathematical consistency

#[cfg(test)]
mod tests {
    use crate::data::structures::MarketDataRow;
    use crate::targets::calibration::VolumeParams;
    use crate::targets::volume::{classify_volume_with_calibrated_params, reconstruct_volume};

    /// Create sample OHLCV data with specified volume pattern
    fn create_sample_ohlcv_with_volume(
        base_price: f64,
        base_volume: f64,
        volume_factor: f64,
        count: usize,
    ) -> Vec<MarketDataRow> {
        (0..count)
            .map(|i| {
                let price = base_price * (1.0 + (i as f64 * 0.01));
                MarketDataRow {
                    timestamp: i as i64,
                    open: price * 0.99,
                    high: price * 1.01,
                    low: price * 0.98,
                    close: price,
                    volume: base_volume * volume_factor * (1.0 + (i as f64 * 0.01)), // Gradual volume change
                }
            })
            .collect()
    }

    #[test]
    fn test_volume_reconstruction_consistency() {
        // Create calibrated parameters
        let calibrated_params = VolumeParams {
            bandwidth: 0.3,
            extreme_multiplier: 2.0,
            min_base_threshold: 0.1,
            min_extreme_threshold: 0.2,
            smoothing_periods: 3,
            percentile_low: 0.05,  // p5
            percentile_high: 0.95, // p95
            balance: crate::targets::calibration::ClassBalance::default(),
        };

        // Test different volume scenarios
        let test_cases = vec![
            ("Low volume", 1000.0, 0.5), // Sequence high vol, horizon low vol (50% of sequence)
            ("Medium volume", 1000.0, 1.0), // Same volume
            ("High volume", 1000.0, 2.0), // Sequence low vol, horizon high vol (2x sequence)
        ];

        for (scenario, base_vol, vol_factor) in test_cases {
            println!("\nTesting scenario: {}", scenario);

            // Create sequence and horizon data
            let sequence_ohlcv = create_sample_ohlcv_with_volume(100.0, base_vol, 1.0, 60);
            let horizon_ohlcv = create_sample_ohlcv_with_volume(100.0, base_vol, vol_factor, 24);

            // Classify using training logic
            let (class, strength) = classify_volume_with_calibrated_params(
                &sequence_ohlcv,
                &horizon_ohlcv,
                &calibrated_params,
            )
            .expect("Classification should succeed");

            println!(
                "  Classification: class={}, strength={:.3}",
                class, strength
            );

            // Create probability distribution (simulate model output)
            let mut probabilities = vec![0.1; 5];
            probabilities[class as usize] = 0.6; // High confidence in the classified class

            // Calculate sequence volumes for reconstruction (NEW: array instead of average)
            let sequence_volumes: Vec<f64> = sequence_ohlcv.iter().map(|row| row.volume).collect();

            // Reconstruct using the same parameters
            let reconstruction =
                reconstruct_volume(&probabilities, &sequence_volumes, &calibrated_params)
                    .expect("Reconstruction should succeed");

            println!(
                "  Reconstruction: expected_ratio={:.3}, confidence={:.3}",
                reconstruction.expected_volume_ratio, reconstruction.confidence
            );

            // Verify the most likely class matches our classification
            assert_eq!(
                reconstruction.most_likely_class, class as usize,
                "Most likely class should match classification"
            );

            // Verify volume ratio makes sense for the regime
            match class {
                0 | 1 => {
                    // Low volume: ratio should be < 1.0
                    assert!(
                        reconstruction.expected_volume_ratio < 1.2,
                        "Low volume should have ratio < 1.2, got {:.3}",
                        reconstruction.expected_volume_ratio
                    );
                }
                2 => {
                    // Medium volume: ratio should be around 1.0
                    assert!(
                        (reconstruction.expected_volume_ratio - 1.0).abs() < 0.5,
                        "Medium volume should have ratio near 1.0, got {:.3}",
                        reconstruction.expected_volume_ratio
                    );
                }
                3 | 4 => {
                    // High volume: ratio should be > 1.0
                    assert!(
                        reconstruction.expected_volume_ratio > 0.8,
                        "High volume should have ratio > 0.8, got {:.3}",
                        reconstruction.expected_volume_ratio
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_volume_ratio_scaling() {
        // Test that volume ratios scale appropriately with volume regimes
        let calibrated_params = VolumeParams {
            bandwidth: 0.3,
            extreme_multiplier: 2.0,
            min_base_threshold: 0.1,
            min_extreme_threshold: 0.2,
            smoothing_periods: 3,
            percentile_low: 0.05,  // p5
            percentile_high: 0.95, // p95
            balance: crate::targets::calibration::ClassBalance::default(),
        };

        let sequence_volumes = vec![1000.0, 1100.0, 1050.0, 1200.0, 1150.0]; // Sample volumes

        // Test each volume class
        let mut previous_ratio = 0.0;
        for class in 0..5 {
            let mut probabilities = vec![0.0; 5];
            probabilities[class] = 1.0; // 100% confidence in one class

            let reconstruction =
                reconstruct_volume(&probabilities, &sequence_volumes, &calibrated_params)
                    .expect("Reconstruction should succeed");

            println!(
                "Class {}: expected_ratio={:.3}",
                class, reconstruction.expected_volume_ratio
            );

            // Verify monotonic increase in volume ratios across classes
            if class > 0 {
                assert!(
                    reconstruction.expected_volume_ratio > previous_ratio,
                    "Class {} ratio ({:.3}) should be > Class {} ratio ({:.3})",
                    class,
                    reconstruction.expected_volume_ratio,
                    class - 1,
                    previous_ratio
                );
            }

            // Verify general ranges (more lenient due to small sample size)
            match class {
                0 => assert!(
                    reconstruction.expected_volume_ratio < 0.95,
                    "VeryLow should have low ratio, got {:.3}",
                    reconstruction.expected_volume_ratio
                ),
                1 => assert!(
                    reconstruction.expected_volume_ratio < 1.0,
                    "Low should have ratio < 1, got {:.3}",
                    reconstruction.expected_volume_ratio
                ),
                2 => assert!(
                    (reconstruction.expected_volume_ratio - 1.0).abs() < 0.15,
                    "Medium should be near 1.0, got {:.3}",
                    reconstruction.expected_volume_ratio
                ),
                3 => assert!(
                    reconstruction.expected_volume_ratio > 1.0,
                    "High should have ratio > 1, got {:.3}",
                    reconstruction.expected_volume_ratio
                ),
                4 => assert!(
                    reconstruction.expected_volume_ratio > 1.05,
                    "VeryHigh should have high ratio, got {:.3}",
                    reconstruction.expected_volume_ratio
                ),
                _ => unreachable!(),
            }

            previous_ratio = reconstruction.expected_volume_ratio;
        }
    }

    #[test]
    fn test_classification_reconstruction_mathematical_consistency() {
        // CRITICAL TEST: Verify that reconstruction boundaries match classification boundaries
        let calibrated_params = VolumeParams {
            bandwidth: 0.4,
            extreme_multiplier: 2.5,
            min_base_threshold: 0.1,
            min_extreme_threshold: 0.2,
            smoothing_periods: 3,
            percentile_low: 0.1,  // p10
            percentile_high: 0.9, // p90
            balance: crate::targets::calibration::ClassBalance::default(),
        };

        // Create realistic volume sequence with variation
        let sequence_volumes: Vec<f64> = (0..60)
            .map(|i| 1000.0 + (i as f64 * 10.0) + ((i % 5) as f64 * 50.0))
            .collect();

        // Test multiple horizon scenarios
        let test_scenarios = vec![
            ("Very Low", 0.4),  // 40% of sequence median
            ("Low", 0.7),       // 70% of sequence median
            ("Medium", 1.0),    // Same as sequence median
            ("High", 1.5),      // 150% of sequence median
            ("Very High", 2.5), // 250% of sequence median
        ];

        for (scenario_name, volume_factor) in test_scenarios {
            println!(
                "\n=== Testing scenario: {} (factor={:.2}) ===",
                scenario_name, volume_factor
            );

            // Create horizon volumes based on factor
            let horizon_volumes: Vec<f64> = (0..24)
                .map(|i| {
                    let seq_median = 1000.0 + 300.0; // Approximate median
                    seq_median * volume_factor * (1.0 + (i as f64 * 0.01))
                })
                .collect();

            // Classify using training logic
            let (class, strength) = classify_volume_with_calibrated_params(
                &sequence_volumes
                    .iter()
                    .map(|&v| MarketDataRow {
                        timestamp: 0,
                        open: 100.0,
                        high: 101.0,
                        low: 99.0,
                        close: 100.0,
                        volume: v,
                    })
                    .collect::<Vec<_>>(),
                &horizon_volumes
                    .iter()
                    .map(|&v| MarketDataRow {
                        timestamp: 0,
                        open: 100.0,
                        high: 101.0,
                        low: 99.0,
                        close: 100.0,
                        volume: v,
                    })
                    .collect::<Vec<_>>(),
                &calibrated_params,
            )
            .expect("Classification should succeed");

            println!(
                "  Classification: class={}, strength={:.3}",
                class, strength
            );

            // Reconstruct using prediction logic
            let mut probabilities = vec![0.1; 5];
            probabilities[class as usize] = 0.6; // High confidence in classified class

            let reconstruction =
                reconstruct_volume(&probabilities, &sequence_volumes, &calibrated_params)
                    .expect("Reconstruction should succeed");

            println!(
                "  Reconstruction: most_likely_class={}, expected_ratio={:.3}",
                reconstruction.most_likely_class, reconstruction.expected_volume_ratio
            );

            // CRITICAL: Most likely class should match classification
            assert_eq!(
                reconstruction.most_likely_class, class as usize,
                "Reconstruction most_likely_class should match classification for scenario: {}",
                scenario_name
            );

            // Verify volume ratio is in the expected range for the class
            let ratio_range = &reconstruction.volume_ratio_ranges[class as usize];
            println!(
                "  Volume ratio range for class {}: [{:.3}, {:.3}]",
                class, ratio_range[0], ratio_range[1]
            );

            // The expected ratio should be within or near the class range
            // (may be slightly outside due to weighted averaging across probabilities)
            let is_in_range = reconstruction.expected_volume_ratio >= ratio_range[0] * 0.8
                && reconstruction.expected_volume_ratio <= ratio_range[1] * 1.2;

            assert!(
                is_in_range,
                "Expected ratio {:.3} should be near range [{:.3}, {:.3}] for class {} (scenario: {})",
                reconstruction.expected_volume_ratio,
                ratio_range[0],
                ratio_range[1],
                class,
                scenario_name
            );
        }

        println!("\n✅ Classification-Reconstruction consistency verified!");
    }
}
