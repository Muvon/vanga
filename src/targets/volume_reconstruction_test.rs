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

            // Calculate average sequence volume for reconstruction
            let sequence_volume: f64 = sequence_ohlcv.iter().map(|row| row.volume).sum::<f64>()
                / sequence_ohlcv.len() as f64;

            // Reconstruct using the same parameters
            let reconstruction =
                reconstruct_volume(&probabilities, sequence_volume, &calibrated_params)
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
            balance: crate::targets::calibration::ClassBalance::default(),
        };

        let sequence_volume = 1000.0;

        // Test each volume class
        for class in 0..5 {
            let mut probabilities = vec![0.0; 5];
            probabilities[class] = 1.0; // 100% confidence in one class

            let reconstruction =
                reconstruct_volume(&probabilities, sequence_volume, &calibrated_params)
                    .expect("Reconstruction should succeed");

            println!(
                "Class {}: expected_ratio={:.3}",
                class, reconstruction.expected_volume_ratio
            );

            // Verify monotonic increase in volume ratios
            match class {
                0 => assert!(
                    reconstruction.expected_volume_ratio < 0.7,
                    "VeryLow should have low ratio"
                ),
                1 => assert!(
                    reconstruction.expected_volume_ratio < 1.0,
                    "Low should have ratio < 1"
                ),
                2 => assert!(
                    (reconstruction.expected_volume_ratio - 1.0).abs() < 0.3,
                    "Medium should be near 1.0"
                ),
                3 => assert!(
                    reconstruction.expected_volume_ratio > 1.0,
                    "High should have ratio > 1"
                ),
                4 => assert!(
                    reconstruction.expected_volume_ratio > 1.3,
                    "VeryHigh should have high ratio"
                ),
                _ => unreachable!(),
            }
        }
    }
}
