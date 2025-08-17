//! Volatility target generation tests
//!
//! Tests the actual volatility classification functionality with real market scenarios

#[cfg(test)]
mod tests {
    use super::super::calibration::VolatilityParams;
    use super::super::volatility::*;
    use crate::data::structures::MarketDataRow;
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

    /// Create test market data from OHLCV tuples
    fn create_test_candles(ohlcv_data: Vec<(f64, f64, f64, f64, f64)>) -> Vec<MarketDataRow> {
        ohlcv_data
            .into_iter()
            .enumerate()
            .map(|(i, (open, high, low, close, volume))| MarketDataRow {
                timestamp: i as i64 * 3600,
                open,
                high,
                low,
                close,
                volume,
            })
            .collect()
    }

    #[test]
    fn test_calculate_simple_atr_different_volatilities() {
        // Low volatility scenario
        let low_vol_candles = create_test_candles(vec![
            (100.0, 100.5, 99.5, 100.2, 1000.0),
            (100.2, 100.7, 99.7, 100.4, 1100.0),
            (100.4, 100.9, 99.9, 100.6, 1200.0),
        ]);

        // High volatility scenario
        let high_vol_candles = create_test_candles(vec![
            (100.0, 105.0, 95.0, 102.0, 1000.0),
            (102.0, 108.0, 96.0, 104.0, 1100.0),
            (104.0, 110.0, 98.0, 106.0, 1200.0),
        ]);

        let low_atr = calculate_simple_atr(&low_vol_candles).unwrap();
        let high_atr = calculate_simple_atr(&high_vol_candles).unwrap();

        assert!(
            low_atr < high_atr,
            "Low volatility ATR ({:.4}) should be less than high volatility ATR ({:.4})",
            low_atr,
            high_atr
        );
        assert!(low_atr >= 0.005, "ATR should respect minimum baseline");
        assert!(
            high_atr > 0.02,
            "High volatility should have significant ATR"
        );

        println!(
            "Low volatility ATR: {:.4}, High volatility ATR: {:.4}",
            low_atr, high_atr
        );
    }

    #[test]
    fn test_calculate_rolling_atr_series() {
        let candles = create_test_candles(vec![
            (100.0, 102.0, 98.0, 101.0, 1000.0),
            (101.0, 104.0, 99.0, 103.0, 1100.0),
            (103.0, 106.0, 101.0, 105.0, 1200.0),
            (105.0, 108.0, 103.0, 107.0, 1300.0),
            (107.0, 110.0, 105.0, 109.0, 1400.0),
            (109.0, 112.0, 107.0, 111.0, 1500.0),
        ]);

        let window = 3;
        let atr_series = calculate_rolling_atr_series(&candles, window).unwrap();

        assert_eq!(
            atr_series.len(),
            candles.len() - window + 1,
            "ATR series should have correct length"
        );

        // All ATR values should be positive and reasonable
        for (i, &atr) in atr_series.iter().enumerate() {
            assert!(
                atr > 0.0,
                "ATR value {} should be positive, got {:.4}",
                i,
                atr
            );
            assert!(
                atr < 20.0,
                "ATR value {} should be reasonable, got {:.4}",
                i,
                atr
            );
        }

        println!("Rolling ATR series: {:?}", atr_series);
    }

    #[test]
    fn test_calculate_atr_distribution_stats() {
        let atr_values = vec![
            0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055,
        ];
        let stats = calculate_atr_distribution_stats(&atr_values);

        assert_eq!(stats.mean, 0.0325, "Mean should be calculated correctly");
        assert!(stats.std_dev > 0.0, "Standard deviation should be positive");
        assert!(stats.median > 0.0, "Median should be positive");
        assert!(
            stats.percentile_25 <= stats.median,
            "Q25 should be <= median"
        );
        assert!(
            stats.median <= stats.percentile_75,
            "Median should be <= Q75"
        );

        println!(
            "ATR distribution stats: mean={:.4}, std_dev={:.4}, median={:.4}",
            stats.mean, stats.std_dev, stats.median
        );
    }

    #[test]
    fn test_classify_volatility_with_calibrated_params() {
        let params = VolatilityParams {
            bandwidth: 0.4,
            extreme_multiplier: 2.0,
            volume_weight: 0.5,
            horizon_decay: 0.95,
            min_volatility_baseline: 0.005,
            balance: Default::default(),
        };

        // Low volatility sequence
        let low_vol_sequence = create_test_candles(vec![
            (100.0, 100.2, 99.8, 100.1, 1000.0),
            (100.1, 100.3, 99.9, 100.2, 1100.0),
            (100.2, 100.4, 100.0, 100.3, 1200.0),
        ]);

        // Low volatility horizon
        let low_vol_horizon = create_test_candles(vec![
            (100.3, 100.5, 100.1, 100.4, 1300.0),
            (100.4, 100.6, 100.2, 100.5, 1400.0),
        ]);

        let (low_class, _) = classify_volatility_with_calibrated_params(
            &low_vol_sequence,
            &low_vol_horizon,
            &params,
        )
        .unwrap();

        // High volatility sequence
        let high_vol_sequence = create_test_candles(vec![
            (100.0, 105.0, 95.0, 102.0, 1000.0),
            (102.0, 108.0, 96.0, 104.0, 1100.0),
            (104.0, 110.0, 98.0, 106.0, 1200.0),
        ]);

        // High volatility horizon
        let high_vol_horizon = create_test_candles(vec![
            (106.0, 115.0, 97.0, 108.0, 1300.0),
            (108.0, 118.0, 98.0, 110.0, 1400.0),
        ]);

        let (high_class, _) = classify_volatility_with_calibrated_params(
            &high_vol_sequence,
            &high_vol_horizon,
            &params,
        )
        .unwrap();

        assert!(
            low_class <= 2,
            "Low volatility should be VeryLow (0), Low (1), or Medium (2), got {}",
            low_class
        );
        assert!(
            high_class >= 2,
            "High volatility should be Medium (2), High (3), or VeryHigh (4), got {}",
            high_class
        );
        assert!(
            low_class < high_class,
            "Low volatility class ({}) should be less than high volatility class ({})",
            low_class,
            high_class
        );

        println!(
            "Low volatility class: {}, High volatility class: {}",
            low_class, high_class
        );
    }

    #[test]
    fn test_generate_volatility_targets_with_calibrated_params() {
        // Create market data with varying volatility regimes
        let df = create_test_dataframe(vec![
            // Low volatility period
            (100.0, 100.5, 99.5, 100.2, 1000.0),
            (100.2, 100.7, 99.7, 100.4, 1100.0),
            (100.4, 100.9, 99.9, 100.6, 1200.0),
            (100.6, 101.1, 100.1, 100.8, 1300.0),
            // Increasing volatility
            (100.8, 102.0, 99.0, 101.5, 1400.0),
            (101.5, 103.5, 98.5, 102.0, 1500.0),
            // High volatility period
            (102.0, 107.0, 97.0, 104.0, 2000.0),
            (104.0, 110.0, 98.0, 106.0, 2200.0),
            (106.0, 113.0, 99.0, 108.0, 2400.0),
            // Decreasing volatility
            (108.0, 111.0, 105.0, 109.0, 1800.0),
            (109.0, 111.5, 106.5, 110.0, 1600.0),
            (110.0, 112.0, 108.0, 111.0, 1400.0),
        ]);

        let horizons = vec!["2h".to_string()]; // Need at least 2 steps for proper calculation
        let sequence_indices = vec![0, 3, 6]; // Different volatility periods
        let sequence_length = 3;

        let params = VolatilityParams {
            bandwidth: 0.4,
            extreme_multiplier: 2.0,
            volume_weight: 0.5,
            horizon_decay: 0.95,
            min_volatility_baseline: 0.005,
            balance: Default::default(),
        };

        let result = generate_volatility_targets_with_calibrated_params(
            &df,
            &horizons,
            &sequence_indices,
            sequence_length,
            &params,
        );

        assert!(
            result.is_ok(),
            "Volatility target generation should succeed: {:?}",
            result.err()
        );
        let (targets, _strengths) = result.unwrap();

        assert!(targets.contains_key("2h"), "Should contain 2h horizon");
        let horizon_targets = &targets["2h"];
        assert_eq!(
            horizon_targets.len(),
            sequence_indices.len(),
            "Should have targets for all sequences"
        );

        // Verify all targets are valid volatility classes (0-4)
        for (i, &target) in horizon_targets.iter().enumerate() {
            assert!(
                (0..=4).contains(&target),
                "Volatility target {} should be 0-4 (VeryLow to VeryHigh), got {} at sequence {}",
                i,
                target,
                sequence_indices[i]
            );
        }

        println!("Generated volatility targets: {:?}", horizon_targets);

        // Note: The actual classification depends on the calibrated parameters and
        // the specific volatility patterns in the data. The important thing is that
        // all targets are valid classes (0-4), which we've already verified above.
    }

    #[test]
    fn test_volatility_class_names() {
        let class_names = get_volatility_class_names();
        assert_eq!(class_names.len(), 5, "Should have 5 volatility classes");
        assert_eq!(class_names[0], "VeryLow", "Class 0 should be VeryLow");
        assert_eq!(class_names[1], "Low", "Class 1 should be Low");
        assert_eq!(class_names[2], "Medium", "Class 2 should be Medium");
        assert_eq!(class_names[3], "High", "Class 3 should be High");
        assert_eq!(class_names[4], "VeryHigh", "Class 4 should be VeryHigh");
    }

    #[test]
    fn test_calculate_average_volume() {
        let candles = create_test_candles(vec![
            (100.0, 102.0, 98.0, 101.0, 1000.0),
            (101.0, 103.0, 99.0, 102.0, 2000.0),
            (102.0, 104.0, 100.0, 103.0, 3000.0),
        ]);

        let avg_volume = calculate_average_volume(&candles);
        assert_eq!(
            avg_volume, 2000.0,
            "Average volume should be calculated correctly"
        );

        // Test empty candles
        let empty_candles = vec![];
        let avg_volume_empty = calculate_average_volume(&empty_candles);
        assert_eq!(
            avg_volume_empty, 0.0,
            "Empty candles should return 0 volume"
        );
    }

    #[test]
    fn test_bandwidth_parameter_effect() {
        let sequence_candles = create_test_candles(vec![
            (100.0, 102.0, 98.0, 101.0, 1000.0),
            (101.0, 103.0, 99.0, 102.0, 1100.0),
            (102.0, 104.0, 100.0, 103.0, 1200.0),
        ]);

        let horizon_candles = create_test_candles(vec![
            (103.0, 107.0, 99.0, 105.0, 1300.0),
            (105.0, 109.0, 101.0, 107.0, 1400.0),
        ]);

        // Low bandwidth - more sensitive to volatility changes
        let low_bandwidth_params = VolatilityParams {
            bandwidth: 0.2,
            extreme_multiplier: 2.0,
            volume_weight: 0.5,
            horizon_decay: 0.95,
            min_volatility_baseline: 0.005,
            balance: Default::default(),
        };

        // High bandwidth - less sensitive to volatility changes
        let high_bandwidth_params = VolatilityParams {
            bandwidth: 0.8,
            extreme_multiplier: 2.0,
            volume_weight: 0.5,
            horizon_decay: 0.95,
            min_volatility_baseline: 0.005,
            balance: Default::default(),
        };

        let (low_bandwidth_class, _) = classify_volatility_with_calibrated_params(
            &sequence_candles,
            &horizon_candles,
            &low_bandwidth_params,
        )
        .unwrap();

        let (high_bandwidth_class, _) = classify_volatility_with_calibrated_params(
            &sequence_candles,
            &horizon_candles,
            &high_bandwidth_params,
        )
        .unwrap();

        println!(
            "Low bandwidth class: {}, High bandwidth class: {}",
            low_bandwidth_class, high_bandwidth_class
        );

        // Both should be valid classes
        assert!((0..=4).contains(&low_bandwidth_class));
        assert!((0..=4).contains(&high_bandwidth_class));
    }

    #[test]
    fn test_edge_cases() {
        let params = VolatilityParams {
            bandwidth: 0.4,
            extreme_multiplier: 2.0,
            volume_weight: 0.5,
            horizon_decay: 0.95,
            min_volatility_baseline: 0.005,
            balance: Default::default(),
        };

        // Test with minimal data
        let minimal_sequence = create_test_candles(vec![(100.0, 100.0, 100.0, 100.0, 1000.0)]);
        let minimal_horizon = create_test_candles(vec![(100.0, 100.0, 100.0, 100.0, 1000.0)]);

        let result = classify_volatility_with_calibrated_params(
            &minimal_sequence,
            &minimal_horizon,
            &params,
        );
        assert!(result.is_ok(), "Should handle minimal data gracefully");

        // Test with zero volatility (identical prices)
        let zero_vol_sequence = create_test_candles(vec![
            (100.0, 100.0, 100.0, 100.0, 1000.0),
            (100.0, 100.0, 100.0, 100.0, 1000.0),
            (100.0, 100.0, 100.0, 100.0, 1000.0),
        ]);
        let zero_vol_horizon = create_test_candles(vec![
            (100.0, 100.0, 100.0, 100.0, 1000.0),
            (100.0, 100.0, 100.0, 100.0, 1000.0),
        ]);

        let (class, _) = classify_volatility_with_calibrated_params(
            &zero_vol_sequence,
            &zero_vol_horizon,
            &params,
        )
        .unwrap();

        assert!(
            class <= 2,
            "Zero volatility should be classified as VeryLow, Low, or Medium, got {}",
            class
        );

        // Test ATR calculation with zero volatility
        let atr = calculate_simple_atr(&zero_vol_sequence).unwrap();
        assert_eq!(
            atr, 0.005,
            "Zero volatility should return minimum baseline ATR"
        );
    }

    #[test]
    fn test_reconstruct_volatility() {
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

        let params = VolatilityParams {
            bandwidth: 0.4,
            extreme_multiplier: 2.0,
            volume_weight: 0.5,
            horizon_decay: 0.95,
            min_volatility_baseline: 0.005,
            balance: Default::default(),
        };

        // Test reconstruction with clear high volatility signal
        let high_vol_probs = vec![0.05, 0.05, 0.1, 0.2, 0.6]; // Strong VeryHigh signal
        let reconstruction =
            reconstruct_volatility(&high_vol_probs, &sequence_ohlcv, &params).unwrap();

        assert_eq!(
            reconstruction.most_likely_class, 4,
            "Should predict VeryHigh class"
        );
        assert!(
            reconstruction.confidence > 0.5,
            "Should have high confidence"
        );
        assert!(
            reconstruction.expected_atr_ratio > 0.0,
            "Should have positive ATR ratio"
        );

        // Test reconstruction with unclear probabilities
        let unclear_probs = vec![0.2, 0.2, 0.2, 0.2, 0.2]; // Equal probabilities
        let reconstruction =
            reconstruct_volatility(&unclear_probs, &sequence_ohlcv, &params).unwrap();

        assert!(
            reconstruction.confidence < 0.3,
            "Should have low confidence for unclear signal"
        );

        println!(
            "High vol reconstruction: class={}, confidence={:.3}, atr_ratio={:.3}",
            reconstruction.most_likely_class,
            reconstruction.confidence,
            reconstruction.expected_atr_ratio
        );
    }
}
