//! Math consistency verification across different markets and price levels
//!
//! This test ensures that our adaptive target generation produces consistent
//! mathematical behavior across different market conditions and price ranges.

#[cfg(test)]
mod tests {
    use crate::config::model::TargetsConfig;
    use crate::data::structures::MarketDataRow;
    use crate::targets::direction::calculate_raw_linear_slope;
    use crate::targets::price_levels::classify_price_level;
    use crate::targets::volatility::{
        calculate_log_volatility_thresholds, classify_volatility_log_ratio,
        get_sequence_atr_baseline,
    };

    /// Helper function to create test market data
    fn create_test_candles(ohlcv_data: Vec<(f64, f64, f64, f64, f64)>) -> Vec<MarketDataRow> {
        ohlcv_data
            .into_iter()
            .map(|(open, high, low, close, volume)| MarketDataRow {
                timestamp: 0,
                open,
                high,
                low,
                close,
                volume,
            })
            .collect()
    }

    /// Test that direction targets are consistent across different price levels
    #[test]
    fn test_direction_consistency_across_price_levels() {
        // Test same percentage moves at different price levels
        let test_cases = vec![
            // (base_price, description)
            (1.0, "Low price (altcoin)"),
            (100.0, "Medium price"),
            (50000.0, "High price (BTC)"),
        ];

        for (base_price, description) in test_cases {
            // Create identical percentage trends at different price levels
            let sequence_1pct = vec![
                base_price,
                base_price * 1.01,
                base_price * 1.02,
                base_price * 1.03,
                base_price * 1.04,
            ];

            let horizon_2pct = vec![
                base_price * 1.04,
                base_price * 1.06,
                base_price * 1.08,
                base_price * 1.10,
                base_price * 1.12,
            ];

            let seq_slope = calculate_raw_linear_slope(&sequence_1pct).unwrap();
            let hor_slope = calculate_raw_linear_slope(&horizon_2pct).unwrap();

            println!(
                "{}: seq_slope={:.6}, hor_slope={:.6}",
                description, seq_slope, hor_slope
            );

            // With volatility normalization, slopes should be similar across price levels
            // for the same percentage moves
            assert!(
                seq_slope > 0.0,
                "Sequence slope should be positive for {}",
                description
            );
            assert!(
                hor_slope > 0.0,
                "Horizon slope should be positive for {}",
                description
            );

            // The ratio should be consistent (horizon is 2x the percentage change)
            let slope_ratio = hor_slope / seq_slope;
            assert!(
                slope_ratio > 1.5 && slope_ratio < 2.5,
                "Slope ratio should be ~2.0 for {}, got {:.2}",
                description,
                slope_ratio
            );
        }
    }

    /// Test that price level targets are consistent across different price ranges
    #[test]
    fn test_price_level_consistency_across_ranges() {
        // Create default targets config for testing
        let targets_config = TargetsConfig::default();

        let test_cases = vec![
            // (base_price, description)
            (10.0, "Low price range"),
            (1000.0, "Medium price range"),
            (50000.0, "High price range"),
        ];

        for (base_price, description) in test_cases {
            // Create identical percentage distributions at different price levels
            let sequence = create_test_candles(
                (0..10)
                    .map(|i| {
                        let price = base_price * (1.0 + (i as f64 - 4.5) * 0.01); // ±4.5% range
                        (price, price * 1.001, price * 0.999, price, 1000.0)
                    })
                    .collect(),
            );

            // Test targets at different percentile positions
            let test_targets = vec![
                (base_price * 0.94, "Below 10th percentile", 0..=1), // Should be Down
                (base_price * 1.00, "At median", 2..=2),             // Should be Neutral
                (base_price * 1.06, "Above 90th percentile", 3..=4), // Should be Up
            ];

            for (target_price, target_desc, expected_range) in test_targets {
                let target = create_test_candles(vec![(
                    target_price,
                    target_price * 1.001,
                    target_price * 0.999,
                    target_price,
                    1000.0,
                )]);

                let class = classify_price_level(&sequence, &target, &targets_config).unwrap();

                assert!(
                    expected_range.contains(&class),
                    "{} - {}: expected class in {:?}, got {} for target {:.2}",
                    description,
                    target_desc,
                    expected_range,
                    class,
                    target_price
                );
            }
        }
    }

    /// Test that volatility targets are consistent across different ATR levels
    #[test]
    fn test_volatility_consistency_across_atr_levels() {
        let test_cases = vec![
            // (base_price, volatility_multiplier, description)
            (100.0, 0.01, "Low volatility"),
            (100.0, 0.05, "Medium volatility"),
            (100.0, 0.10, "High volatility"),
        ];

        for (base_price, vol_mult, description) in test_cases {
            // Create sequences with different volatility levels but same relative changes
            let sequence = create_test_candles(vec![
                (
                    base_price,
                    base_price * (1.0 + vol_mult),
                    base_price * (1.0 - vol_mult),
                    base_price,
                    1000.0,
                ),
                (
                    base_price,
                    base_price * (1.0 + vol_mult * 1.1),
                    base_price * (1.0 - vol_mult * 1.1),
                    base_price,
                    1100.0,
                ),
                (
                    base_price,
                    base_price * (1.0 + vol_mult * 0.9),
                    base_price * (1.0 - vol_mult * 0.9),
                    base_price,
                    1200.0,
                ),
            ]);

            // Test different volatility scenarios
            let volatility_tests = vec![
                (vol_mult * 0.5, "Lower volatility", 0..=1), // Should be VeryLow or Low
                (vol_mult * 1.0, "Same volatility", 2..=2),  // Should be Medium
                (vol_mult * 2.0, "Higher volatility", 3..=4), // Should be High or VeryHigh
            ];

            let train_atr = get_sequence_atr_baseline(&sequence).unwrap();
            let targets_config = TargetsConfig {
                base_sensitivity: 0.4,
                balance_target: 0.2,
                momentum_weighting: 1.2,
                extreme_multiplier: 2.0,
            };
            let thresholds = calculate_log_volatility_thresholds(&targets_config).unwrap();

            for (target_vol, vol_desc, expected_range) in volatility_tests {
                let target_sequence = create_test_candles(vec![
                    (
                        base_price,
                        base_price * (1.0 + target_vol),
                        base_price * (1.0 - target_vol),
                        base_price,
                        1000.0,
                    ),
                    (
                        base_price,
                        base_price * (1.0 + target_vol * 1.1),
                        base_price * (1.0 - target_vol * 1.1),
                        base_price,
                        1100.0,
                    ),
                ]);

                let target_atr = get_sequence_atr_baseline(&target_sequence).unwrap();
                let class = classify_volatility_log_ratio(train_atr, target_atr, &thresholds);

                println!(
                    "{} - {}: train_atr={:.6}, target_atr={:.6}, class={}",
                    description, vol_desc, train_atr, target_atr, class
                );

                assert!(
                    expected_range.contains(&class),
                    "{} - {}: expected class in {:?}, got {}",
                    description,
                    vol_desc,
                    expected_range,
                    class
                );
            }
        }
    }

    /// Test mathematical properties are preserved across different configurations
    #[test]
    fn test_mathematical_properties_preservation() {
        // Test 1: Direction slope normalization preserves relative ordering
        let price_levels = vec![1.0, 100.0, 10000.0];
        let mut normalized_slopes = Vec::new();

        for base_price in price_levels {
            let prices = vec![
                base_price,
                base_price * 1.01,
                base_price * 1.02,
                base_price * 1.03,
            ];
            let slope = calculate_raw_linear_slope(&prices).unwrap();
            normalized_slopes.push(slope);
        }

        // All normalized slopes should be similar (within 20% of each other)
        let min_slope = normalized_slopes
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_slope = normalized_slopes
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let slope_ratio = max_slope / min_slope;

        assert!(
            slope_ratio < 1.2,
            "Normalized slopes should be similar across price levels, ratio: {:.2}",
            slope_ratio
        );

        // Test 2: Price level percentiles preserve relative positions
        // Create default targets config for testing
        let targets_config = TargetsConfig::default();

        for base_price in [10.0, 1000.0] {
            let sequence = create_test_candles(
                (0..10)
                    .map(|i| {
                        let price = base_price * (1.0 + (i as f64 - 4.5) * 0.02); // ±9% range
                        (price, price, price, price, 1000.0)
                    })
                    .collect(),
            );

            // Test targets at 10th, 50th, and 90th percentiles
            let percentile_targets = vec![
                (base_price * 0.92, 0..=1), // 10th percentile -> Down
                (base_price * 1.00, 2..=2), // 50th percentile -> Neutral
                (base_price * 1.08, 3..=4), // 90th percentile -> Up
            ];

            for (target_price, expected_range) in percentile_targets {
                let target = create_test_candles(vec![(
                    target_price,
                    target_price,
                    target_price,
                    target_price,
                    1000.0,
                )]);

                let class = classify_price_level(&sequence, &target, &targets_config).unwrap();
                assert!(expected_range.contains(&class),
                    "Percentile classification should be consistent at price level {}: target={:.2}, class={}",
                    base_price, target_price, class);
            }
        }

        // Test 3: Volatility log ratios preserve multiplicative relationships
        let base_atr = 0.02;
        let targets_config = TargetsConfig {
            base_sensitivity: 0.4,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 2.0,
        };
        let thresholds = calculate_log_volatility_thresholds(&targets_config).unwrap();

        let volatility_ratios = vec![0.5, 1.0, 2.0, 4.0];
        let mut classifications = Vec::new();

        for ratio in volatility_ratios {
            let target_atr = base_atr * ratio;
            let class = classify_volatility_log_ratio(base_atr, target_atr, &thresholds);
            classifications.push(class);
        }

        // Classifications should be monotonically increasing
        for i in 1..classifications.len() {
            assert!(
                classifications[i] >= classifications[i - 1],
                "Volatility classifications should be monotonic: {:?}",
                classifications
            );
        }
    }

    /// Test edge cases maintain mathematical consistency
    #[test]
    fn test_edge_case_consistency() {
        // Test 1: Very small price movements
        let tiny_sequence = vec![1.0, 1.0001, 1.0002, 1.0001, 1.0];
        let tiny_slope = calculate_raw_linear_slope(&tiny_sequence).unwrap();
        assert!(
            tiny_slope.abs() < 1e-3,
            "Tiny movements should have tiny normalized slopes"
        );

        // Test 2: Zero volatility sequences
        let flat_sequence = create_test_candles(vec![
            (100.0, 100.0, 100.0, 100.0, 1000.0),
            (100.0, 100.0, 100.0, 100.0, 1000.0),
        ]);
        let zero_atr = get_sequence_atr_baseline(&flat_sequence).unwrap();
        assert_eq!(
            zero_atr, 0.005,
            "Zero volatility should use minimum baseline"
        );

        // Test 3: Extreme percentile configurations
        // Create default targets config for testing
        let targets_config = TargetsConfig::default();

        let sequence = create_test_candles(vec![
            (100.0, 100.0, 100.0, 100.0, 1000.0),
            (110.0, 110.0, 110.0, 110.0, 1000.0),
        ]);

        let target = create_test_candles(vec![(105.0, 105.0, 105.0, 105.0, 1000.0)]);

        let class = classify_price_level(&sequence, &target, &targets_config).unwrap();
        assert_eq!(
            class, 2,
            "Target within extreme percentiles should be Neutral"
        );
    }
}
