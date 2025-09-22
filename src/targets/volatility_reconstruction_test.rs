//! Tests for volatility reconstruction mathematical consistency

#[cfg(test)]
mod tests {
    use crate::data::structures::MarketDataRow;
    use crate::targets::calibration::VolatilityParams;
    use crate::targets::volatility::{
        classify_volatility_with_calibrated_params, reconstruct_volatility,
    };

    /// Create sample OHLCV data for testing
    fn create_sample_ohlcv(
        base_price: f64,
        volatility_factor: f64,
        count: usize,
    ) -> Vec<MarketDataRow> {
        (0..count)
            .map(|i| {
                let price = base_price * (1.0 + (i as f64 * 0.01));
                let range = price * volatility_factor;
                MarketDataRow {
                    timestamp: i as i64,
                    open: price - range * 0.3,
                    high: price + range,
                    low: price - range,
                    close: price + range * 0.3,
                    volume: 1000.0 * (1.0 + (i as f64 * 0.1)),
                }
            })
            .collect()
    }

    #[test]
    fn test_volatility_reconstruction_consistency() {
        // Create calibrated parameters
        let calibrated_params = VolatilityParams {
            bandwidth: 0.2,
            extreme_multiplier: 2.0,
            min_volatility_baseline: 0.001,
            volume_weight: 0.1,
            horizon_decay: 0.0,
            balance: crate::targets::calibration::ClassBalance::default(),
        };

        // Test different volatility scenarios
        let test_cases = vec![
            ("Low volatility", 0.01, 0.005), // Sequence high vol, horizon low vol
            ("Medium volatility", 0.01, 0.01), // Same volatility
            ("High volatility", 0.01, 0.02), // Sequence low vol, horizon high vol
        ];

        for (scenario, seq_vol, hor_vol) in test_cases {
            println!("\nTesting scenario: {}", scenario);

            // Create sequence and horizon data
            let sequence_ohlcv = create_sample_ohlcv(100.0, seq_vol, 60);
            let horizon_ohlcv = create_sample_ohlcv(100.0, hor_vol, 24);

            // Classify using training logic
            let (class, strength) = classify_volatility_with_calibrated_params(
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

            // Reconstruct using the same parameters
            let reconstruction =
                reconstruct_volatility(&probabilities, &sequence_ohlcv, &calibrated_params)
                    .expect("Reconstruction should succeed");

            println!(
                "  Reconstruction: expected_atr_ratio={:.3}, stop_multiplier={:.3}",
                reconstruction.expected_atr_ratio, reconstruction.recommended_stop_multiplier
            );

            // Verify mathematical consistency
            // The reconstruction should identify the correct volatility regime
            // For high confidence in a single class, the expected_atr_ratio
            // should match the class's characteristic ratio

            // Verify the most likely class matches our classification
            assert_eq!(
                reconstruction.most_likely_class, class as usize,
                "Most likely class should match classification"
            );

            // Verify stop multiplier is reasonable based on volatility regime
            match class {
                0 | 1 => {
                    // Low volatility: stop multiplier should be < 1.0
                    assert!(
                        reconstruction.recommended_stop_multiplier <= 1.0,
                        "Low volatility should have stop multiplier <= 1.0, got {:.3}",
                        reconstruction.recommended_stop_multiplier
                    );
                }
                2 => {
                    // Medium volatility: stop multiplier should be around 1.0
                    assert!(
                        (reconstruction.recommended_stop_multiplier - 1.0).abs() < 0.3,
                        "Medium volatility should have stop multiplier near 1.0, got {:.3}",
                        reconstruction.recommended_stop_multiplier
                    );
                }
                3 | 4 => {
                    // High volatility: stop multiplier should be > 1.0
                    assert!(
                        reconstruction.recommended_stop_multiplier >= 1.0,
                        "High volatility should have stop multiplier >= 1.0, got {:.3}",
                        reconstruction.recommended_stop_multiplier
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_stop_distance_scaling() {
        // Test that stop distances scale appropriately with volatility regimes
        let calibrated_params = VolatilityParams {
            bandwidth: 0.2,
            extreme_multiplier: 2.0,
            min_volatility_baseline: 0.001,
            volume_weight: 0.1,
            horizon_decay: 0.0,
            balance: crate::targets::calibration::ClassBalance::default(),
        };

        let sequence_ohlcv = create_sample_ohlcv(100.0, 0.01, 60);

        // Test each volatility class
        for class in 0..5 {
            let mut probabilities = vec![0.0; 5];
            probabilities[class] = 1.0; // 100% confidence in one class

            let reconstruction =
                reconstruct_volatility(&probabilities, &sequence_ohlcv, &calibrated_params)
                    .expect("Reconstruction should succeed");

            println!(
                "Class {}: ATR ratio={:.3}, stop multiplier={:.3}",
                class,
                reconstruction.expected_atr_ratio,
                reconstruction.recommended_stop_multiplier
            );

            // Verify monotonic increase in stop distances
            match class {
                0 => assert!(
                    reconstruction.expected_atr_ratio < 0.8,
                    "VeryLow should have low ratio"
                ),
                1 => assert!(
                    reconstruction.expected_atr_ratio < 1.0,
                    "Low should have ratio < 1"
                ),
                2 => assert!(
                    (reconstruction.expected_atr_ratio - 1.0).abs() < 0.2,
                    "Medium should be near 1.0"
                ),
                3 => assert!(
                    reconstruction.expected_atr_ratio > 1.0,
                    "High should have ratio > 1"
                ),
                4 => assert!(
                    reconstruction.expected_atr_ratio > 1.2,
                    "VeryHigh should have high ratio"
                ),
                _ => unreachable!(),
            }
        }
    }
}
