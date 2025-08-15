//! Math consistency verification across different markets and price levels
//!
//! This test ensures that our adaptive target generation produces consistent
//! mathematical behavior across different market conditions and price ranges.

#[cfg(test)]
mod tests {

    use crate::data::structures::MarketDataRow;
    use crate::targets::calibration::VolatilityParams;
    use crate::targets::direction::calculate_raw_linear_slope;
    use crate::targets::volatility::{
        classify_volatility_with_calibrated_params, get_sequence_atr_baseline,
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

            let train_atr = get_sequence_atr_baseline(&sequence, 0.005).unwrap();

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

                let target_atr = get_sequence_atr_baseline(&target_sequence, 0.005).unwrap();
                let params = VolatilityParams {
                    bandwidth: 0.4,
                    extreme_multiplier: 1.2,
                    volume_weight: 0.5,
                    horizon_decay: 0.95,
                    min_volatility_baseline: 0.005,
                    balance: Default::default(),
                };
                let class = classify_volatility_with_calibrated_params(
                    &sequence,        // sequence_candles first
                    &target_sequence, // horizon_candles second
                    &params,
                )
                .unwrap();

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
}
