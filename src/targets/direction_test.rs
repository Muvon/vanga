//! Direction target generation tests
//!
//! Tests the actual direction classification functionality with real market scenarios

#[cfg(test)]
mod tests {
    use super::super::calibration::DirectionParams;
    use super::super::direction::*;
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
    fn test_calculate_raw_linear_slope_basic() {
        // Test clear upward trend
        let upward_prices = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let slope = calculate_raw_linear_slope(&upward_prices).unwrap();
        assert!(
            slope > 1.5,
            "Strong upward trend should have slope > 1.5, got {}",
            slope
        );

        // Test clear downward trend
        let downward_prices = vec![108.0, 106.0, 104.0, 102.0, 100.0];
        let slope = calculate_raw_linear_slope(&downward_prices).unwrap();
        assert!(
            slope < -1.5,
            "Strong downward trend should have slope < -1.5, got {}",
            slope
        );

        // Test sideways movement
        let sideways_prices = vec![100.0, 100.1, 99.9, 100.2, 99.8];
        let slope = calculate_raw_linear_slope(&sideways_prices).unwrap();
        assert!(
            slope.abs() < 0.5,
            "Sideways movement should have small slope, got {}",
            slope
        );
    }

    #[test]
    fn test_classify_direction_with_calibrated_params() {
        let params = DirectionParams {
            sensitivity: 1.0,
            extreme_multiplier: 2.0,
            min_base_threshold: 0.01,
            min_extreme_threshold: 0.03,
            base_multiplier: 20.0,
            balance: Default::default(),
        };

        // Test strong upward movement (should be UP or PUMP)
        let strong_up_prices = vec![100.0, 103.0, 106.0, 109.0, 112.0];
        let horizon_prices = vec![112.0, 116.0, 120.0];
        let class =
            classify_direction_with_calibrated_params(&strong_up_prices, &horizon_prices, &params)
                .unwrap();
        assert!(
            class >= 3,
            "Strong upward movement should be UP (3) or PUMP (4), got {}",
            class
        );

        // Test strong downward movement (should be DOWN or DUMP)
        let strong_down_prices = vec![112.0, 109.0, 106.0, 103.0, 100.0];
        let horizon_down_prices = vec![100.0, 96.0, 92.0];
        let class = classify_direction_with_calibrated_params(
            &strong_down_prices,
            &horizon_down_prices,
            &params,
        )
        .unwrap();
        assert!(
            class <= 1,
            "Strong downward movement should be DUMP (0) or DOWN (1), got {}",
            class
        );

        // Test sideways movement (should be SIDEWAYS)
        let sideways_prices = vec![100.0, 100.5, 99.5, 100.2, 99.8];
        let horizon_sideways = vec![100.1, 99.9, 100.3];
        let class =
            classify_direction_with_calibrated_params(&sideways_prices, &horizon_sideways, &params)
                .unwrap();
        assert_eq!(
            class, 2,
            "Sideways movement should be SIDEWAYS (2), got {}",
            class
        );
    }

    #[test]
    fn test_generate_direction_targets_with_calibrated_params() {
        // Create realistic market data with different trends
        let df = create_test_dataframe(vec![
            // Upward trend sequence
            (100.0, 102.0, 99.0, 101.0, 1000.0),
            (101.0, 104.0, 100.0, 103.0, 1100.0),
            (103.0, 106.0, 102.0, 105.0, 1200.0),
            (105.0, 108.0, 104.0, 107.0, 1300.0),
            // Continuation upward
            (107.0, 111.0, 106.0, 110.0, 1400.0),
            (110.0, 114.0, 109.0, 113.0, 1500.0),
            // Sideways movement
            (113.0, 115.0, 111.0, 112.0, 1200.0),
            (112.0, 114.0, 110.0, 113.0, 1100.0),
            (113.0, 115.0, 111.0, 112.0, 1000.0),
            // Downward trend
            (112.0, 113.0, 108.0, 109.0, 1300.0),
            (109.0, 110.0, 105.0, 106.0, 1400.0),
            (106.0, 107.0, 102.0, 103.0, 1500.0),
        ]);

        let horizons = vec!["1h".to_string()];
        let sequence_indices = vec![0, 3, 6]; // Different trend periods
        let sequence_length = 3;

        let params = DirectionParams {
            sensitivity: 0.8,
            extreme_multiplier: 2.0,
            min_base_threshold: 0.01,
            min_extreme_threshold: 0.03,
            base_multiplier: 20.0,
            balance: Default::default(),
        };

        let result = generate_direction_targets_with_calibrated_params(
            &df,
            &horizons,
            &sequence_indices,
            sequence_length,
            &params,
        );

        assert!(
            result.is_ok(),
            "Direction target generation should succeed: {:?}",
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

        // Verify all targets are valid direction classes (0-4)
        for (i, &target) in horizon_targets.iter().enumerate() {
            assert!(
                target >= 0 && target <= 4,
                "Direction target {} should be 0-4 (DUMP to PUMP), got {} at sequence {}",
                i,
                target,
                sequence_indices[i]
            );
        }

        println!("Generated direction targets: {:?}", horizon_targets);
    }

    #[test]
    fn test_direction_class_names() {
        let class_names = get_direction_class_names();
        assert_eq!(class_names.len(), 5, "Should have 5 direction classes");
        assert_eq!(class_names[0], "DUMP", "Class 0 should be DUMP");
        assert_eq!(class_names[1], "DOWN", "Class 1 should be DOWN");
        assert_eq!(class_names[2], "SIDEWAYS", "Class 2 should be SIDEWAYS");
        assert_eq!(class_names[3], "UP", "Class 3 should be UP");
        assert_eq!(class_names[4], "PUMP", "Class 4 should be PUMP");
    }

    #[test]
    fn test_sensitivity_parameter_effect() {
        let base_prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let horizon_prices = vec![104.0, 105.0, 106.0];

        // Low sensitivity - should classify more as sideways
        let low_sensitivity_params = DirectionParams {
            sensitivity: 0.1,
            extreme_multiplier: 2.0,
            min_base_threshold: 0.01,
            min_extreme_threshold: 0.03,
            base_multiplier: 20.0,
            balance: Default::default(),
        };

        // High sensitivity - should classify more as directional
        let high_sensitivity_params = DirectionParams {
            sensitivity: 2.0,
            extreme_multiplier: 2.0,
            min_base_threshold: 0.01,
            min_extreme_threshold: 0.03,
            base_multiplier: 20.0,
            balance: Default::default(),
        };

        let low_class = classify_direction_with_calibrated_params(
            &base_prices,
            &horizon_prices,
            &low_sensitivity_params,
        )
        .unwrap();

        let high_class = classify_direction_with_calibrated_params(
            &base_prices,
            &horizon_prices,
            &high_sensitivity_params,
        )
        .unwrap();

        println!(
            "Low sensitivity class: {}, High sensitivity class: {}",
            low_class, high_class
        );

        // With moderate upward trend, low sensitivity might classify as sideways,
        // while high sensitivity should classify as upward
        if low_class == 2 {
            // SIDEWAYS
            assert!(
                high_class >= 2,
                "High sensitivity should not classify lower than low sensitivity"
            );
        }
    }

    #[test]
    fn test_edge_cases() {
        let params = DirectionParams {
            sensitivity: 1.0,
            extreme_multiplier: 2.0,
            min_base_threshold: 0.01,
            min_extreme_threshold: 0.03,
            base_multiplier: 20.0,
            balance: Default::default(),
        };

        // Test with minimal data
        let minimal_prices = vec![100.0, 100.0];
        let minimal_horizon = vec![100.0];
        let result =
            classify_direction_with_calibrated_params(&minimal_prices, &minimal_horizon, &params);
        assert!(result.is_ok(), "Should handle minimal data gracefully");

        // Test with identical prices (no movement)
        let flat_prices = vec![100.0, 100.0, 100.0, 100.0];
        let flat_horizon = vec![100.0, 100.0];
        let class = classify_direction_with_calibrated_params(&flat_prices, &flat_horizon, &params)
            .unwrap();
        assert_eq!(class, 2, "No movement should be classified as SIDEWAYS");

        // Test with extreme volatility but no trend
        let volatile_prices = vec![100.0, 110.0, 90.0, 105.0, 95.0];
        let volatile_horizon = vec![100.0, 90.0, 110.0];
        let result =
            classify_direction_with_calibrated_params(&volatile_prices, &volatile_horizon, &params);
        assert!(result.is_ok(), "Should handle volatile but trendless data");
    }

    #[test]
    fn test_reconstruct_direction() {
        use crate::data::structures::MarketDataRow;

        // Create test market data
        let sequence_ohlcv = vec![
            MarketDataRow {
                timestamp: 0,
                open: 100.0,
                high: 102.0,
                low: 98.0,
                close: 101.0,
                volume: 1000.0,
            },
            MarketDataRow {
                timestamp: 1,
                open: 101.0,
                high: 103.0,
                low: 99.0,
                close: 102.0,
                volume: 1100.0,
            },
            MarketDataRow {
                timestamp: 2,
                open: 102.0,
                high: 104.0,
                low: 100.0,
                close: 103.0,
                volume: 1200.0,
            },
        ];

        let params = DirectionParams {
            sensitivity: 1.0,
            extreme_multiplier: 2.0,
            min_base_threshold: 0.01,
            min_extreme_threshold: 0.03,
            base_multiplier: 20.0,
            balance: Default::default(),
        };

        // Test reconstruction with clear up probabilities
        let clear_up_probs = vec![0.05, 0.05, 0.1, 0.3, 0.5]; // Strong UP signal
        let reconstruction =
            reconstruct_direction(&clear_up_probs, &sequence_ohlcv, &params).unwrap();

        assert_eq!(
            reconstruction.most_likely_class, 4,
            "Should predict PUMP class"
        );
        assert!(
            reconstruction.confidence > 0.4,
            "Should have high confidence"
        );
        assert!(
            reconstruction.expected_momentum_change > 0.0,
            "Should have positive momentum"
        );

        // Test reconstruction with unclear probabilities
        let unclear_probs = vec![0.2, 0.2, 0.2, 0.2, 0.2]; // Equal probabilities
        let reconstruction =
            reconstruct_direction(&unclear_probs, &sequence_ohlcv, &params).unwrap();

        assert!(
            reconstruction.confidence < 0.3,
            "Should have low confidence for unclear signal"
        );
    }
}
