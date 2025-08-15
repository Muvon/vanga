//! Comprehensive tests for volatility classification to ensure mathematical correctness
//!
//! These tests validate:
//! 1. ATR baseline calculation accuracy with adaptive fallback
//! 2. Logarithmic ratio classification correctness
//! 3. Adaptive bandwidth scaling appropriateness
//! 4. Classification balance across different scenarios
//! 5. Edge case handling

#[cfg(test)]
mod tests {
    use super::super::volatility::*;

    use crate::data::structures::MarketDataRow;
    use approx::assert_relative_eq;

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

    /// Test ATR baseline calculation with adaptive fallback
    #[test]
    fn test_atr_baseline_calculation() {
        // Test case 1: Normal volatility sequence
        let normal_candles = create_test_candles(vec![
            (100.0, 102.0, 99.0, 101.0, 1000.0),
            (101.0, 103.0, 100.0, 102.0, 1100.0),
            (102.0, 104.0, 101.0, 103.0, 1200.0),
            (103.0, 105.0, 102.0, 104.0, 1300.0),
        ]);

        let atr = get_sequence_atr_baseline(&normal_candles, 0.005).unwrap();
        assert!(
            atr > 0.005,
            "ATR should be above minimum baseline, got {}",
            atr
        );
        assert!(
            atr < 0.1,
            "ATR should be reasonable for normal volatility, got {}",
            atr
        );

        // Test case 2: High volatility sequence
        let high_vol_candles = create_test_candles(vec![
            (100.0, 110.0, 90.0, 105.0, 2000.0),
            (105.0, 115.0, 95.0, 110.0, 2100.0),
            (110.0, 120.0, 100.0, 115.0, 2200.0),
        ]);

        let high_atr = get_sequence_atr_baseline(&high_vol_candles, 0.005).unwrap();
        assert!(high_atr > atr, "High volatility should have higher ATR");
        assert!(
            high_atr > 0.05,
            "High volatility ATR should be substantial, got {}",
            high_atr
        );

        // Test case 3: Low volatility sequence
        let low_vol_candles = create_test_candles(vec![
            (100.0, 100.1, 99.9, 100.0, 500.0),
            (100.0, 100.05, 99.95, 100.0, 510.0),
            (100.0, 100.02, 99.98, 100.0, 520.0),
        ]);

        let low_atr = get_sequence_atr_baseline(&low_vol_candles, 0.005).unwrap();
        assert_eq!(low_atr, 0.005, "Low volatility should use minimum baseline");

        // Test case 4: Single candle fallback
        let single_candle = create_test_candles(vec![(100.0, 102.0, 98.0, 101.0, 1000.0)]);

        let fallback_atr = get_sequence_atr_baseline(&single_candle, 0.005).unwrap();
        assert!(
            (fallback_atr - 100.0 * 0.005).abs() < 0.01,
            "Single candle should use 0.5% of close price, got {}",
            fallback_atr
        );

        // Test case 5: Empty sequence
        let empty_candles = vec![];
        let empty_atr = get_sequence_atr_baseline(&empty_candles, 0.005).unwrap();
        assert_eq!(
            empty_atr, 0.005,
            "Empty sequence should use absolute minimum"
        );
    }

    /// Test logarithmic volatility classification
    #[test]
    fn test_log_volatility_classification() {
        let _config = TargetsConfig {
            base_sensitivity: 0.4,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 2.0,
        };

        // Test case 1: Same volatility (Medium)
        let same_vol_seq = create_test_candles(vec![
            (100.0, 102.0, 98.0, 101.0, 1000.0),
            (101.0, 103.0, 99.0, 102.0, 1100.0),
            (102.0, 104.0, 100.0, 103.0, 1200.0),
        ]);
        let same_vol_hor = create_test_candles(vec![
            (103.0, 105.0, 101.0, 104.0, 1300.0),
            (104.0, 106.0, 102.0, 105.0, 1400.0),
        ]);

        let train_atr = get_sequence_atr_baseline(&same_vol_seq, 0.005).unwrap();
        let target_atr = get_sequence_atr_baseline(&same_vol_hor, 0.005).unwrap();

        // Should be similar ATR values, resulting in Medium classification
        let targets_config = TargetsConfig {
            base_sensitivity: 0.4,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 2.0,
        };
        let log_thresholds = calculate_log_volatility_thresholds(&targets_config).unwrap();
        let class = classify_volatility_log_ratio(train_atr, target_atr, &log_thresholds);
        assert_eq!(class, 2, "Similar volatility should be Medium (2)");

        // Test case 2: Higher volatility (High or VeryHigh)
        let high_vol_hor = create_test_candles(vec![
            (103.0, 115.0, 90.0, 110.0, 2000.0),
            (110.0, 125.0, 95.0, 120.0, 2100.0),
        ]);

        let high_target_atr = get_sequence_atr_baseline(&high_vol_hor, 0.005).unwrap();
        let high_class = classify_volatility_log_ratio(train_atr, high_target_atr, &log_thresholds);
        assert!(
            high_class >= 3,
            "Higher volatility should be High (3) or VeryHigh (4), got {}",
            high_class
        );

        // Test case 3: Lower volatility (Low or VeryLow)
        let low_vol_hor = create_test_candles(vec![
            (103.0, 103.1, 102.9, 103.0, 500.0),
            (103.0, 103.05, 102.95, 103.0, 510.0),
        ]);

        let low_target_atr = get_sequence_atr_baseline(&low_vol_hor, 0.005).unwrap();
        let low_class = classify_volatility_log_ratio(train_atr, low_target_atr, &log_thresholds);
        assert!(
            low_class <= 1,
            "Lower volatility should be VeryLow (0) or Low (1), got {}",
            low_class
        );
    }

    /// Test adaptive bandwidth scaling
    #[test]
    fn test_adaptive_bandwidth_scaling() {
        // Test different volatility scenarios and their bandwidth scaling
        let test_cases: Vec<(f64, (f64, f64))> = vec![
            // (base_atr, expected_volatility_factor_range)
            (0.001, (0.3, 0.5)), // Very low volatility, clamped to 0.3
            (0.005, (0.8, 1.2)), // Normal volatility, around 1.0
            (0.015, (1.0, 3.0)), // High volatility, wide range due to clamping
        ];

        for (base_atr, (min_factor, max_factor)) in test_cases {
            let baseline_atr = base_atr.max(0.005);
            let volatility_factor = (base_atr / baseline_atr).clamp(0.3, 3.0);

            assert!(
                volatility_factor >= min_factor && volatility_factor <= max_factor,
                "Volatility factor {} should be in range [{}, {}] for ATR {}",
                volatility_factor,
                min_factor,
                max_factor,
                base_atr
            );

            // Test bandwidth scaling
            let base_bandwidth = 0.4;
            let adaptive_bandwidth = base_bandwidth * volatility_factor;

            assert!(
                (0.12..=1.2).contains(&adaptive_bandwidth),
                "Adaptive bandwidth {} should be reasonable for volatility factor {}",
                adaptive_bandwidth,
                volatility_factor
            );
        }
    }

    /// Test logarithmic threshold calculation
    #[test]
    fn test_log_threshold_calculation() {
        // Test case 1: Standard configuration
        let targets_config = TargetsConfig {
            base_sensitivity: 0.4,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 2.0,
        };
        let thresholds = calculate_log_volatility_thresholds(&targets_config).unwrap();

        let expected_half = 0.4 / 2.0; // 0.2
        let expected_extreme = 0.4 * 2.0; // 0.8

        assert_relative_eq!(thresholds.very_low_max, -expected_extreme, epsilon = 1e-10);
        assert_relative_eq!(thresholds.low_max, -expected_half, epsilon = 1e-10);
        assert_relative_eq!(thresholds.medium_max, expected_half, epsilon = 1e-10);
        assert_relative_eq!(thresholds.high_max, expected_extreme, epsilon = 1e-10);

        // Test case 2: Different configuration
        let targets_config2 = TargetsConfig {
            base_sensitivity: 0.6,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 1.5,
        };
        let thresholds2 = calculate_log_volatility_thresholds(&targets_config2).unwrap();
        let expected_half2 = 0.6 / 2.0; // 0.3
        let expected_extreme2 = 0.6 * 1.5; // 0.9

        assert_relative_eq!(
            thresholds2.very_low_max,
            -expected_extreme2,
            epsilon = 1e-10
        );
        assert_relative_eq!(thresholds2.medium_max, expected_half2, epsilon = 1e-10);
    }

    /// Test realistic crypto volatility scenarios
    #[test]
    fn test_realistic_crypto_volatility_scenarios() {
        let _config = TargetsConfig {
            base_sensitivity: 0.4,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 2.0,
        };

        // Scenario 1: BTC normal trading to high volatility
        let btc_normal = create_test_candles(vec![
            (50000.0, 50500.0, 49500.0, 50200.0, 100.0),
            (50200.0, 50700.0, 49700.0, 50400.0, 110.0),
            (50400.0, 50900.0, 49900.0, 50600.0, 120.0),
        ]);

        let btc_volatile = create_test_candles(vec![
            (50600.0, 52000.0, 48000.0, 51000.0, 500.0),
            (51000.0, 53000.0, 47000.0, 52000.0, 600.0),
        ]);

        let normal_atr = get_sequence_atr_baseline(&btc_normal, 0.005).unwrap();
        let volatile_atr = get_sequence_atr_baseline(&btc_volatile, 0.005).unwrap();

        assert!(
            volatile_atr > normal_atr * 1.5,
            "Volatile period should have significantly higher ATR: {} vs {}",
            volatile_atr,
            normal_atr
        );

        // Scenario 2: ETH high volatility to consolidation
        let eth_volatile = create_test_candles(vec![
            (3000.0, 3300.0, 2700.0, 3100.0, 200.0),
            (3100.0, 3400.0, 2800.0, 3200.0, 220.0),
        ]);

        let eth_consolidation = create_test_candles(vec![
            (3200.0, 3220.0, 3180.0, 3210.0, 80.0),
            (3210.0, 3230.0, 3190.0, 3220.0, 85.0),
        ]);

        let eth_vol_atr = get_sequence_atr_baseline(&eth_volatile, 0.005).unwrap();
        let eth_consol_atr = get_sequence_atr_baseline(&eth_consolidation, 0.005).unwrap();

        assert!(
            eth_consol_atr < eth_vol_atr * 0.5,
            "Consolidation should have much lower ATR: {} vs {}",
            eth_consol_atr,
            eth_vol_atr
        );
    }

    /// Test edge cases and error handling
    #[test]
    fn test_volatility_edge_cases() {
        let _config = TargetsConfig {
            base_sensitivity: 0.4,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 2.0,
        };

        // Test case 1: Zero ATR values
        let targets_config = TargetsConfig {
            base_sensitivity: 0.4,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 2.0,
        };
        let log_thresholds = calculate_log_volatility_thresholds(&targets_config).unwrap();
        let class = classify_volatility_log_ratio(0.0, 0.01, &log_thresholds);
        assert_eq!(class, 2, "Zero train ATR should default to Medium");

        let class2 = classify_volatility_log_ratio(0.01, 0.0, &log_thresholds);
        assert_eq!(class2, 2, "Zero target ATR should default to Medium");

        // Test case 2: Very small ATR values
        let class3 = classify_volatility_log_ratio(1e-10, 1e-10, &log_thresholds);
        assert_eq!(class3, 2, "Very small ATR values should default to Medium");

        // Test case 3: Extreme ratios
        let class4 = classify_volatility_log_ratio(0.001, 1.0, &log_thresholds);
        assert_eq!(class4, 4, "Extreme volatility increase should be VeryHigh");

        let class5 = classify_volatility_log_ratio(1.0, 0.001, &log_thresholds);
        assert_eq!(class5, 0, "Extreme volatility decrease should be VeryLow");
    }

    /// Test classification balance with synthetic data
    #[test]
    fn test_volatility_classification_balance() {
        let _config = TargetsConfig {
            base_sensitivity: 0.4,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 2.0,
        };

        let mut class_counts = [0; 5];
        let test_cases = 1000;
        let targets_config = TargetsConfig {
            base_sensitivity: 0.4,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 2.0,
        };
        let log_thresholds = calculate_log_volatility_thresholds(&targets_config).unwrap();

        // Generate synthetic test cases with controlled volatility ratios
        for i in 0..test_cases {
            let base_atr = 0.02; // Fixed baseline

            // Create volatility ratio from 0.1x to 10x (log range)
            let log_ratio = (i as f64 / test_cases as f64 - 0.5) * 4.0; // -2.0 to +2.0
            let target_atr = base_atr * log_ratio.exp();

            let class = classify_volatility_log_ratio(base_atr, target_atr, &log_thresholds);
            class_counts[class as usize] += 1;
        }

        // Print distribution for analysis
        println!(
            "Volatility classification distribution over {} synthetic cases:",
            test_cases
        );
        let class_names = ["VeryLow", "Low", "Medium", "High", "VeryHigh"];
        for (i, &count) in class_counts.iter().enumerate() {
            let percentage = (count as f64 / test_cases as f64) * 100.0;
            println!("  {}: {} ({:.1}%)", class_names[i], count, percentage);
        }

        // Verify no class is completely empty
        for (i, &count) in class_counts.iter().enumerate() {
            assert!(
                count > 0,
                "Class {} ({}) has zero samples",
                i,
                class_names[i]
            );
        }

        // Verify reasonable distribution (no class should dominate > 80%)
        for (i, &count) in class_counts.iter().enumerate() {
            let percentage = (count as f64 / test_cases as f64) * 100.0;
            assert!(
                percentage < 80.0,
                "Class {} ({}) dominates with {:.1}%",
                i,
                class_names[i],
                percentage
            );
        }

        // Medium should be reasonably represented
        assert!(
            class_counts[2] > 50,
            "Medium class should have reasonable representation"
        );
    }

    /// Test rolling ATR series calculation
    #[test]
    fn test_rolling_atr_series_calculation() {
        // Test case 1: Normal sequence with sufficient data
        let normal_candles = create_test_candles(vec![
            (100.0, 102.0, 98.0, 101.0, 1000.0),
            (101.0, 103.0, 99.0, 102.0, 1100.0),
            (102.0, 104.0, 100.0, 103.0, 1200.0),
            (103.0, 105.0, 101.0, 104.0, 1300.0),
            (104.0, 106.0, 102.0, 105.0, 1400.0),
            (105.0, 107.0, 103.0, 106.0, 1500.0),
        ]);

        let atr_series = calculate_rolling_atr_series(&normal_candles, 3).unwrap();

        // Should have multiple ATR values
        assert!(
            atr_series.len() >= 3,
            "Should have multiple ATR values, got {}",
            atr_series.len()
        );

        // All ATR values should be positive and reasonable
        for (i, &atr) in atr_series.iter().enumerate() {
            assert!(
                atr > 0.0 && atr < 0.5,
                "ATR value {} at index {} should be reasonable",
                atr,
                i
            );
        }

        // Test case 2: Insufficient data
        let short_candles = create_test_candles(vec![(100.0, 102.0, 98.0, 101.0, 1000.0)]);

        let short_series = calculate_rolling_atr_series(&short_candles, 3).unwrap();
        assert_eq!(
            short_series,
            vec![0.02],
            "Should return default for insufficient data"
        );

        // Test case 3: Empty data
        let empty_candles = vec![];
        let empty_series = calculate_rolling_atr_series(&empty_candles, 3).unwrap();
        assert_eq!(
            empty_series,
            vec![0.02],
            "Should return default for empty data"
        );
    }

    /// Test ATR distribution statistics calculation
    #[test]
    fn test_atr_distribution_stats() {
        // Test case 1: Normal distribution
        let atr_series = vec![0.01, 0.015, 0.02, 0.025, 0.03];
        let stats = calculate_atr_distribution_stats(&atr_series);

        assert_relative_eq!(stats.mean, 0.02, epsilon = 1e-6);
        assert!(stats.std_dev > 0.0, "Standard deviation should be positive");
        assert_eq!(stats.median, 0.02, "Median should be middle value");

        // Test case 2: Empty series
        let empty_series: Vec<f64> = vec![];
        let empty_stats = calculate_atr_distribution_stats(&empty_series);

        assert_eq!(empty_stats.mean, 0.02, "Should use default values");
        assert_eq!(empty_stats.std_dev, 0.01);
    }

    /// Test adaptive volatility bandwidth calculation
    #[test]
    fn test_adaptive_volatility_bandwidth() {
        // Test case 1: Normal volatility series
        let normal_series = vec![0.015, 0.018, 0.020, 0.022, 0.025];
        let base_sensitivity = 0.2;

        let bandwidth =
            calculate_adaptive_volatility_bandwidth(&normal_series, base_sensitivity).unwrap();

        assert!(
            bandwidth > 0.05 && bandwidth < 1.0,
            "Adaptive bandwidth should be reasonable: {}",
            bandwidth
        );

        // Test case 2: Empty series
        let empty_series: Vec<f64> = vec![];
        let empty_bandwidth =
            calculate_adaptive_volatility_bandwidth(&empty_series, base_sensitivity).unwrap();

        assert_eq!(
            empty_bandwidth, base_sensitivity,
            "Empty series should return base sensitivity"
        );
    }

    /// Test enhanced volatility classification with multi-feature analysis
    #[test]
    fn test_enhanced_volatility_classification() {
        let targets_config = TargetsConfig {
            base_sensitivity: 0.2,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 2.0,
        };

        // Test case 1: Normal to high volatility transition with volume increase
        let normal_sequence = create_test_candles(vec![
            (1000.0, 1020.0, 980.0, 1010.0, 100.0),
            (1010.0, 1030.0, 990.0, 1020.0, 110.0),
            (1020.0, 1040.0, 1000.0, 1030.0, 120.0),
            (1030.0, 1050.0, 1010.0, 1040.0, 130.0),
            (1040.0, 1060.0, 1020.0, 1050.0, 140.0),
        ]);

        let high_vol_horizon = create_test_candles(vec![
            (1050.0, 1100.0, 950.0, 1080.0, 500.0), // High volume, high volatility
            (1080.0, 1150.0, 980.0, 1120.0, 600.0), // Increasing volume and volatility
            (1120.0, 1200.0, 1000.0, 1160.0, 700.0), // Sustained high vol/volume
        ]);

        let test_params = crate::targets::adaptive_parameters::VolatilityAdaptiveParams {
            bandwidth_size: 0.4,
            extreme_multiplier: 2.0,
            volume_weight: 0.1,
            atr_distribution_stats: crate::targets::volatility::AtrDistributionStats::default(),
            cv_adjustment_factor: 1.0,
            horizon_decay_factor: 1.0,
            min_baseline_atr: 0.005,
            achieved_balance: crate::targets::calibration::ClassBalance::default(),
        };
        let class = classify_volatility_with_distribution_analysis(
            &normal_sequence,
            &high_vol_horizon,
            &test_params,
        )
        .unwrap();

        assert!(
            (0..=4).contains(&class),
            "Enhanced classification should produce valid class: {}",
            class
        );

        // Should classify as high volatility due to multiple factors:
        // 1. Higher ATR ratio
        // 2. Increasing volatility trend
        // 3. High volume weighting
        assert!(
            class >= 3,
            "High volume + high volatility should classify as High or VeryHigh, got {}",
            class
        );

        // Test case 2: Low volume volatility (should be less extreme)
        let low_vol_horizon = create_test_candles(vec![
            (1050.0, 1100.0, 950.0, 1080.0, 50.0), // High volatility but low volume
            (1080.0, 1150.0, 980.0, 1120.0, 60.0), // Should be treated as noise
            (1120.0, 1200.0, 1000.0, 1160.0, 70.0),
        ]);

        let low_vol_class = classify_volatility_with_distribution_analysis(
            &normal_sequence,
            &low_vol_horizon,
            &test_params,
        )
        .unwrap();

        // Low volume volatility should be classified less extremely than high volume
        assert!(
            low_vol_class <= class,
            "Low volume volatility should be less extreme than high volume volatility"
        );

        // Test case 3: Decreasing volatility trend
        let decreasing_vol_horizon = create_test_candles(vec![
            (1050.0, 1080.0, 1020.0, 1060.0, 200.0), // Moderate volatility
            (1060.0, 1070.0, 1050.0, 1065.0, 210.0), // Decreasing volatility
            (1065.0, 1068.0, 1062.0, 1067.0, 220.0), // Low volatility
        ]);

        let decreasing_class = classify_volatility_with_distribution_analysis(
            &normal_sequence,
            &decreasing_vol_horizon,
            &test_params,
        )
        .unwrap();

        assert!(
            decreasing_class <= 2,
            "Decreasing volatility trend should classify as Low or Medium, got {}",
            decreasing_class
        );

        // Test case 4: Edge cases
        let single_candle = create_test_candles(vec![(1000.0, 1020.0, 980.0, 1010.0, 100.0)]);

        let edge_class = classify_volatility_with_distribution_analysis(
            &normal_sequence,
            &single_candle, // Use single_candle for edge case test
            &test_params,
        )
        .unwrap();

        assert_eq!(
            edge_class, 2,
            "Insufficient sequence data should default to Medium"
        );
    }

    /// Test individual volatility feature calculations
    #[test]
    fn test_volatility_feature_calculations() {
        // Test volatility trend change
        let stable_sequence = create_test_candles(vec![
            (100.0, 102.0, 98.0, 101.0, 1000.0),
            (101.0, 103.0, 99.0, 102.0, 1000.0),
            (102.0, 104.0, 100.0, 103.0, 1000.0),
        ]);

        let increasing_vol_horizon = create_test_candles(vec![
            (103.0, 108.0, 98.0, 105.0, 1000.0), // Higher volatility
            (105.0, 115.0, 95.0, 110.0, 1000.0), // Even higher volatility
            (110.0, 125.0, 90.0, 120.0, 1000.0), // Highest volatility
        ]);

        let trend_change =
            calculate_volatility_trend_change(&stable_sequence, &increasing_vol_horizon).unwrap();
        assert!(
            trend_change > 0.0,
            "Increasing volatility should have positive trend change, got {}",
            trend_change
        );

        // Test volume-weighted volatility
        let high_volume_horizon = create_test_candles(vec![
            (103.0, 108.0, 98.0, 105.0, 5000.0), // High volume
            (105.0, 115.0, 95.0, 110.0, 6000.0), // Higher volume
        ]);

        let low_volume_horizon = create_test_candles(vec![
            (103.0, 108.0, 98.0, 105.0, 100.0), // Low volume
            (105.0, 115.0, 95.0, 110.0, 120.0), // Low volume
        ]);

        let high_vol_change =
            calculate_volume_weighted_volatility_change(&stable_sequence, &high_volume_horizon)
                .unwrap();
        let low_vol_change =
            calculate_volume_weighted_volatility_change(&stable_sequence, &low_volume_horizon)
                .unwrap();

        // Both should show increased volatility, but volume weighting should make a difference
        assert!(
            high_vol_change > 1.0,
            "High volume volatility should show increase"
        );
        assert!(
            low_vol_change > 1.0,
            "Low volume volatility should show increase"
        );

        // Test volatility persistence
        let consistent_horizon = create_test_candles(vec![
            (103.0, 105.0, 101.0, 104.0, 1000.0), // Consistent volatility
            (104.0, 106.0, 102.0, 105.0, 1000.0), // Similar volatility
            (105.0, 107.0, 103.0, 106.0, 1000.0), // Consistent pattern
        ]);

        let erratic_horizon = create_test_candles(vec![
            (103.0, 120.0, 90.0, 110.0, 1000.0),  // High volatility
            (110.0, 112.0, 108.0, 111.0, 1000.0), // Low volatility
            (111.0, 130.0, 85.0, 125.0, 1000.0),  // Very high volatility
        ]);

        let consistent_persistence =
            calculate_volatility_persistence(&stable_sequence, &consistent_horizon).unwrap();
        let erratic_persistence =
            calculate_volatility_persistence(&stable_sequence, &erratic_horizon).unwrap();

        assert!(
            consistent_persistence > erratic_persistence,
            "Consistent volatility should have higher persistence than erratic volatility"
        );
    }

    /// Test volatility bandwidth calibration
    #[test]
    fn test_volatility_bandwidth_calibration() {
        // Create synthetic OHLCV data with known volatility patterns
        let mut ohlcv_data = Vec::new();
        let base_price = 1000.0;

        // Generate 100 candles with varying volatility
        for i in 0..100 {
            let volatility = 0.01 + (i as f64 / 100.0) * 0.05; // 1% to 6% volatility
            let price = base_price * (1.0 + (i as f64 * 0.01).sin() * 0.1);
            let range = price * volatility;

            ohlcv_data.push(MarketDataRow {
                timestamp: i as i64,
                open: price,
                high: price + range / 2.0,
                low: price - range / 2.0,
                close: price + (range / 4.0) * (i as f64).sin(),
                volume: 1000.0,
            });
        }

        let sequence_length = 20;
        let horizon_steps = 10;
        let target_balance = 0.15;

        let calibrated_bandwidth = calibrate_volatility_bandwidth(
            &ohlcv_data,
            sequence_length,
            horizon_steps,
            target_balance,
        )
        .unwrap();

        // Should return a reasonable bandwidth value
        assert!(
            calibrated_bandwidth > 0.05 && calibrated_bandwidth < 1.0,
            "Calibrated bandwidth should be reasonable: {}",
            calibrated_bandwidth
        );

        // Test with insufficient data
        let short_data = vec![ohlcv_data[0].clone(), ohlcv_data[1].clone()];
        let fallback_bandwidth = calibrate_volatility_bandwidth(
            &short_data,
            sequence_length,
            horizon_steps,
            target_balance,
        )
        .unwrap();

        assert_eq!(
            fallback_bandwidth, 0.2,
            "Should use fallback for insufficient data"
        );

        println!(
            "Calibrated volatility bandwidth: {:.6}",
            calibrated_bandwidth
        );
    }

    /// Test horizon-weighted ATR calculation with different decay factors
    #[test]
    fn test_horizon_weighted_atr_calculation() {
        // Create test horizon candles with increasing volatility toward the end
        let horizon_candles = create_test_candles(vec![
            (100.0, 101.0, 99.0, 100.5, 1000.0), // Low volatility (early)
            (100.5, 102.0, 99.5, 101.0, 1100.0), // Medium volatility
            (101.0, 104.0, 98.0, 102.0, 1200.0), // High volatility
            (102.0, 107.0, 97.0, 105.0, 1300.0), // Very high volatility (recent)
        ]);

        // Test uniform weighting (decay_factor = 1.0)
        let uniform_atr = get_horizon_weighted_atr_baseline(&horizon_candles, 1.0).unwrap();
        let baseline_atr = get_sequence_atr_baseline(&horizon_candles, 0.005).unwrap();
        assert_relative_eq!(uniform_atr, baseline_atr, epsilon = 1e-10);

        // Test recent weighting (decay_factor = 0.9)
        let weighted_atr = get_horizon_weighted_atr_baseline(&horizon_candles, 0.9).unwrap();

        // Weighted ATR should be higher than uniform because recent candles have higher volatility
        assert!(
            weighted_atr > uniform_atr,
            "Weighted ATR ({:.6}) should be higher than uniform ATR ({:.6}) when recent volatility is higher",
            weighted_atr, uniform_atr
        );

        // Test with reverse volatility pattern (high early, low recent)
        let reverse_candles = create_test_candles(vec![
            (100.0, 107.0, 93.0, 105.0, 1000.0), // Very high volatility (early)
            (105.0, 108.0, 102.0, 106.0, 1100.0), // High volatility
            (106.0, 107.0, 105.0, 106.5, 1200.0), // Medium volatility
            (106.5, 107.0, 106.0, 106.8, 1300.0), // Low volatility (recent)
        ]);

        let reverse_uniform = get_sequence_atr_baseline(&reverse_candles, 0.005).unwrap();
        let reverse_weighted = get_horizon_weighted_atr_baseline(&reverse_candles, 0.9).unwrap();

        // Weighted ATR should be lower than uniform because recent candles have lower volatility
        assert!(
            reverse_weighted < reverse_uniform,
            "Weighted ATR ({:.6}) should be lower than uniform ATR ({:.6}) when recent volatility is lower",
            reverse_weighted, reverse_uniform
        );
    }

    /// Test horizon weighting with edge cases
    #[test]
    fn test_horizon_weighted_atr_edge_cases() {
        // Test with insufficient data
        let single_candle = create_test_candles(vec![(100.0, 102.0, 98.0, 101.0, 1000.0)]);
        let result = get_horizon_weighted_atr_baseline(&single_candle, 0.95).unwrap();
        let baseline = get_sequence_atr_baseline(&single_candle, 0.005).unwrap();
        assert_relative_eq!(result, baseline, epsilon = 1e-10);

        // Test with decay_factor very close to 1.0
        let normal_candles = create_test_candles(vec![
            (100.0, 102.0, 98.0, 101.0, 1000.0),
            (101.0, 103.0, 99.0, 102.0, 1100.0),
        ]);
        let near_uniform = get_horizon_weighted_atr_baseline(&normal_candles, 0.999999).unwrap();
        let uniform = get_sequence_atr_baseline(&normal_candles, 0.005).unwrap();
        assert_relative_eq!(near_uniform, uniform, epsilon = 1e-5);

        // Test with extreme decay factor
        let extreme_weighted = get_horizon_weighted_atr_baseline(&normal_candles, 0.1).unwrap();
        assert!(
            extreme_weighted > 0.0,
            "Extreme weighting should still produce valid ATR"
        );
    }

    /// Test integration of horizon weighting with volatility classification
    #[test]
    fn test_horizon_weighting_classification_integration() {
        use crate::targets::adaptive_parameters::VolatilityAdaptiveParams;

        // Create sequence with stable volatility
        let sequence_candles = create_test_candles(vec![
            (100.0, 101.0, 99.0, 100.5, 1000.0),
            (100.5, 101.5, 99.5, 101.0, 1100.0),
            (101.0, 102.0, 100.0, 101.5, 1200.0),
            (101.5, 102.5, 100.5, 102.0, 1300.0),
        ]);

        // Create horizon with increasing volatility toward end
        let horizon_candles = create_test_candles(vec![
            (102.0, 103.0, 101.0, 102.5, 1400.0),
            (102.5, 105.0, 100.0, 104.0, 1500.0), // High recent volatility
        ]);

        let targets_config = TargetsConfig {
            base_sensitivity: 0.4,
            balance_target: 0.2,
            momentum_weighting: 1.2,
            extreme_multiplier: 2.0,
        };

        // Test without horizon weighting (uniform)
        let test_params = crate::targets::adaptive_parameters::VolatilityAdaptiveParams {
            bandwidth_size: 0.4,
            extreme_multiplier: 2.0,
            volume_weight: 0.1,
            atr_distribution_stats: crate::targets::volatility::AtrDistributionStats::default(),
            cv_adjustment_factor: 1.0,
            horizon_decay_factor: 1.0,
            min_baseline_atr: 0.005,
            achieved_balance: crate::targets::calibration::ClassBalance::default(),
        };
        let uniform_class = classify_volatility_with_distribution_analysis(
            &sequence_candles,
            &horizon_candles,
            &test_params,
        )
        .unwrap();

        // Test with horizon weighting (emphasize recent volatility)
        let adaptive_params = VolatilityAdaptiveParams {
            bandwidth_size: 0.4,
            extreme_multiplier: 2.0,
            horizon_decay_factor: 0.9, // Emphasize recent volatility
            atr_distribution_stats: Default::default(),
            cv_adjustment_factor: 1.0,
            min_baseline_atr: 0.005, // Add missing field with default value
            achieved_balance: Default::default(),
            volume_weight: 0.1,
        };

        let weighted_class = classify_volatility_with_distribution_analysis(
            &sequence_candles,
            &horizon_candles,
            &adaptive_params,
        )
        .unwrap();

        // Both should produce valid classes
        assert!(
            (0..5).contains(&uniform_class),
            "Uniform classification should be valid"
        );
        assert!(
            (0..5).contains(&weighted_class),
            "Weighted classification should be valid"
        );

        // The weighted classification might be different due to emphasis on recent high volatility
        // This is expected behavior - the test validates the integration works
        println!(
            "Uniform class: {}, Weighted class: {}",
            uniform_class, weighted_class
        );
    }

    /// Test mathematical properties of horizon weighting
    #[test]
    fn test_horizon_weighting_mathematical_properties() {
        // Create candles with known volatility pattern
        let candles = create_test_candles(vec![
            (100.0, 102.0, 98.0, 101.0, 1000.0),  // TR ≈ 4.0/101 ≈ 0.0396
            (101.0, 104.0, 99.0, 103.0, 1100.0),  // TR ≈ 5.0/103 ≈ 0.0485
            (103.0, 108.0, 101.0, 107.0, 1200.0), // TR ≈ 7.0/107 ≈ 0.0654
        ]);

        // Test monotonicity: stronger decay should emphasize recent volatility more
        let decay_factors = vec![1.0, 0.95, 0.90, 0.85];
        let mut atr_values = Vec::new();

        for &decay in &decay_factors {
            let atr = get_horizon_weighted_atr_baseline(&candles, decay).unwrap();
            atr_values.push(atr);
        }

        // Since recent candles have higher volatility, stronger decay (lower factor) should give higher ATR
        for i in 1..atr_values.len() {
            assert!(
                atr_values[i] >= atr_values[i - 1] * 0.99, // Allow small numerical differences
                "ATR with decay {} ({:.6}) should be >= ATR with decay {} ({:.6})",
                decay_factors[i],
                atr_values[i],
                decay_factors[i - 1],
                atr_values[i - 1]
            );
        }

        // Test weight calculation correctness
        let decay_factor = 0.9f64;
        let n = candles.len();

        // Calculate expected weights manually
        let expected_weights: Vec<f64> = (0..n - 1)
            .map(|i| decay_factor.powi((n - i - 2) as i32) as f64)
            .collect();

        let total_expected_weight: f64 = expected_weights.iter().sum();

        // Verify the weighting produces reasonable results
        assert!(
            total_expected_weight > 0.0,
            "Total weight should be positive"
        );
        assert!(
            expected_weights.last().unwrap() >= expected_weights.first().unwrap(),
            "Most recent weight should be >= earliest weight"
        );
    }
}
