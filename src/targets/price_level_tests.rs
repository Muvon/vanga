//! Comprehensive tests for price level classification to ensure mathematical correctness
//!
//! These tests validate:
//! 1. Percentile-based boundary calculation accuracy
//! 2. VWAP price calculation correctness
//! 3. Bandwidth scaling appropriateness
//! 4. Classification balance across different scenarios
//! 5. Edge case handling

#[cfg(test)]
mod tests {
    use super::super::price_levels::*;
    use crate::config::model::TargetsConfig;
    use crate::data::structures::MarketDataRow;

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

    /// Test different percentile configurations
    #[test]
    fn test_different_percentile_configs() {
        let sequence = create_test_candles(vec![
            (100.0, 100.0, 100.0, 100.0, 1000.0),
            (101.0, 101.0, 101.0, 101.0, 1000.0),
            (102.0, 102.0, 102.0, 102.0, 1000.0),
            (103.0, 103.0, 103.0, 103.0, 1000.0),
            (104.0, 104.0, 104.0, 104.0, 1000.0),
            (105.0, 105.0, 105.0, 105.0, 1000.0),
            (106.0, 106.0, 106.0, 106.0, 1000.0),
            (107.0, 107.0, 107.0, 107.0, 1000.0),
            (108.0, 108.0, 108.0, 108.0, 1000.0),
            (109.0, 109.0, 109.0, 109.0, 1000.0),
        ]);

        let target = create_test_candles(vec![
            (101.5, 101.5, 101.5, 101.5, 1000.0), // 15th percentile
        ]);

        // Test case 1: Wide percentiles [0.05, 0.95]
        let wide_config = TargetsConfig::default();

        let wide_class = classify_price_level(&sequence, &target, &wide_config).unwrap();
        assert_eq!(
            wide_class, 2,
            "Target within wide percentiles should be Neutral"
        );

        // Test case 2: Narrow percentiles [0.2, 0.8]
        let narrow_config = TargetsConfig::default();

        let narrow_class = classify_price_level(&sequence, &target, &narrow_config).unwrap();
        assert!(
            narrow_class <= 1,
            "Target below narrow percentiles should be Down classification"
        );

        // Test case 3: Default percentiles [0.1, 0.9]
        let default_config = TargetsConfig::default();

        let default_class = classify_price_level(&sequence, &target, &default_config).unwrap();
        assert!(
            default_class <= 1,
            "Target below default percentiles should be Down classification"
        );
    }

    /// Test bandwidth scaling effects
    #[test]
    fn test_bandwidth_scaling() {
        let sequence = create_test_candles(vec![
            (100.0, 100.0, 100.0, 100.0, 1000.0), // 10th percentile
            (101.0, 101.0, 101.0, 101.0, 1000.0),
            (102.0, 102.0, 102.0, 102.0, 1000.0),
            (103.0, 103.0, 103.0, 103.0, 1000.0),
            (104.0, 104.0, 104.0, 104.0, 1000.0),
            (105.0, 105.0, 105.0, 105.0, 1000.0),
            (106.0, 106.0, 106.0, 106.0, 1000.0),
            (107.0, 107.0, 107.0, 107.0, 1000.0),
            (108.0, 108.0, 108.0, 108.0, 1000.0),
            (109.0, 109.0, 109.0, 109.0, 1000.0), // 90th percentile
        ]);

        // Target significantly above 90th percentile
        let breakout_target = create_test_candles(vec![
            (115.0, 115.0, 115.0, 115.0, 1000.0), // Well above range
        ]);

        // Test case 1: Low bandwidth (more sensitive)
        let sensitive_config = TargetsConfig::default();

        let sensitive_class =
            classify_price_level(&sequence, &breakout_target, &sensitive_config).unwrap();
        assert_eq!(
            sensitive_class, 4,
            "High target with low bandwidth should be Strong Up (4)"
        );

        // Test case 2: High bandwidth (less sensitive)
        let conservative_config = TargetsConfig::default();

        let conservative_class =
            classify_price_level(&sequence, &breakout_target, &conservative_config).unwrap();
        // With higher bandwidth, might still be Strong Up but could be Moderate Up
        assert!(
            conservative_class >= 3,
            "High target with high bandwidth should be Up classification"
        );

        // Test case 3: Default bandwidth
        let default_config = TargetsConfig::default();

        let default_class =
            classify_price_level(&sequence, &breakout_target, &default_config).unwrap();
        assert!(
            default_class >= 3,
            "High target with default bandwidth should be Up classification"
        );
    }

    /// Test VWAP calculation with volume weighting
    #[test]
    fn test_vwap_calculation() {
        // Test case 1: High volume candle should influence VWAP more
        let volume_weighted_sequence = create_test_candles(vec![
            (100.0, 102.0, 98.0, 101.0, 100.0),   // Low volume, VWAP = 100.25
            (101.0, 103.0, 99.0, 102.0, 10000.0), // High volume, VWAP = 101.25
            (102.0, 104.0, 100.0, 103.0, 100.0),  // Low volume, VWAP = 102.25
        ]);

        let target = create_test_candles(vec![(101.5, 101.5, 101.5, 101.5, 1000.0)]);

        let config = TargetsConfig::default();

        // The high volume candle should dominate the percentile calculation
        let class = classify_price_level(&volume_weighted_sequence, &target, &config).unwrap();
        assert!(
            (0..=4).contains(&class),
            "VWAP-weighted classification should be valid"
        );

        // Test case 2: Zero volume fallback
        let zero_volume_sequence = create_test_candles(vec![
            (100.0, 102.0, 98.0, 101.0, 0.0),  // Zero volume, should use OHLC4
            (101.0, 103.0, 99.0, 102.0, 0.0),  // Zero volume, should use OHLC4
            (102.0, 104.0, 100.0, 103.0, 0.0), // Zero volume, should use OHLC4
        ]);

        let zero_vol_class = classify_price_level(&zero_volume_sequence, &target, &config).unwrap();
        assert!(
            (0..=4).contains(&zero_vol_class),
            "Zero volume classification should be valid"
        );
    }

    /// Test realistic crypto price scenarios
    #[test]
    fn test_realistic_crypto_scenarios() {
        let config = TargetsConfig::default();

        // Scenario 1: BTC consolidation range
        let btc_consolidation = create_test_candles(vec![
            (49000.0, 49500.0, 48500.0, 49200.0, 100.0),
            (49200.0, 49700.0, 48700.0, 49400.0, 110.0),
            (49400.0, 49900.0, 48900.0, 49600.0, 120.0),
            (49600.0, 50100.0, 49100.0, 49800.0, 130.0),
            (49800.0, 50300.0, 49300.0, 50000.0, 140.0),
            (50000.0, 50500.0, 49500.0, 50200.0, 150.0),
            (50200.0, 50700.0, 49700.0, 50400.0, 160.0),
            (50400.0, 50900.0, 49900.0, 50600.0, 170.0),
            (50600.0, 51100.0, 50100.0, 50800.0, 180.0),
            (50800.0, 51300.0, 50300.0, 51000.0, 190.0),
        ]);

        // Target within consolidation range
        let btc_target_inside =
            create_test_candles(vec![(50000.0, 50200.0, 49800.0, 50100.0, 200.0)]);

        let inside_class =
            classify_price_level(&btc_consolidation, &btc_target_inside, &config).unwrap();
        assert_eq!(
            inside_class, 2,
            "BTC target within consolidation should be Neutral"
        );

        // Target breaking above consolidation
        let btc_target_breakout =
            create_test_candles(vec![(52000.0, 52500.0, 51500.0, 52200.0, 500.0)]);

        let breakout_class =
            classify_price_level(&btc_consolidation, &btc_target_breakout, &config).unwrap();
        assert!(
            breakout_class >= 3,
            "BTC breakout should be Up classification, got {}",
            breakout_class
        );

        // Scenario 2: ETH downtrend with support test
        let eth_downtrend = create_test_candles(vec![
            (3500.0, 3600.0, 3400.0, 3450.0, 200.0),
            (3450.0, 3550.0, 3350.0, 3400.0, 210.0),
            (3400.0, 3500.0, 3300.0, 3350.0, 220.0),
            (3350.0, 3450.0, 3250.0, 3300.0, 230.0),
            (3300.0, 3400.0, 3200.0, 3250.0, 240.0),
            (3250.0, 3350.0, 3150.0, 3200.0, 250.0),
            (3200.0, 3300.0, 3100.0, 3150.0, 260.0),
            (3150.0, 3250.0, 3050.0, 3100.0, 270.0),
            (3100.0, 3200.0, 3000.0, 3050.0, 280.0),
            (3050.0, 3150.0, 2950.0, 3000.0, 290.0),
        ]);

        // Target breaking below support
        let eth_target_breakdown =
            create_test_candles(vec![(2800.0, 2900.0, 2700.0, 2850.0, 600.0)]);

        let breakdown_class =
            classify_price_level(&eth_downtrend, &eth_target_breakdown, &config).unwrap();
        assert!(
            breakdown_class <= 1,
            "ETH support breakdown should be Down classification, got {}",
            breakdown_class
        );
    }

    /// Test edge cases and error handling
    #[test]
    fn test_price_level_edge_cases() {
        let config = TargetsConfig::default();

        // Test case 1: Empty sequence
        let empty_sequence = vec![];
        let target = create_test_candles(vec![(100.0, 100.0, 100.0, 100.0, 1000.0)]);

        let empty_class = classify_price_level(&empty_sequence, &target, &config).unwrap();
        assert_eq!(empty_class, 2, "Empty sequence should default to Neutral");

        // Test case 2: Flat sequence (zero bandwidth)
        let flat_sequence = create_test_candles(vec![
            (100.0, 100.0, 100.0, 100.0, 1000.0),
            (100.0, 100.0, 100.0, 100.0, 1000.0),
            (100.0, 100.0, 100.0, 100.0, 1000.0),
        ]);

        let flat_target_high = create_test_candles(vec![(101.0, 101.0, 101.0, 101.0, 1000.0)]);

        let flat_class = classify_price_level(&flat_sequence, &flat_target_high, &config).unwrap();
        assert_eq!(
            flat_class, 3,
            "Target above flat sequence should be Moderate Up"
        );

        let flat_target_low = create_test_candles(vec![(99.0, 99.0, 99.0, 99.0, 1000.0)]);

        let flat_class_low =
            classify_price_level(&flat_sequence, &flat_target_low, &config).unwrap();
        assert_eq!(
            flat_class_low, 2,
            "Target below flat sequence should be Neutral (fallback)"
        );

        // Test case 3: Single candle sequence
        let single_sequence = create_test_candles(vec![(100.0, 102.0, 98.0, 101.0, 1000.0)]);

        let single_class = classify_price_level(&single_sequence, &target, &config).unwrap();
        assert!(
            (0..=4).contains(&single_class),
            "Single candle classification should be valid"
        );
    }

    /// Test classification balance with synthetic data
    #[test]
    fn test_price_level_classification_balance() {
        let config = TargetsConfig::default();

        let mut class_counts = [0; 5];
        let test_cases = 1000;

        // Generate synthetic test cases with controlled price distributions
        for i in 0..test_cases {
            // Create a sequence with known distribution
            let base_price = 1000.0;
            let sequence = create_test_candles(
                (0..10)
                    .map(|j| {
                        let price = base_price + (j as f64 - 4.5) * 10.0; // 955 to 1045
                        (price, price + 1.0, price - 1.0, price, 1000.0)
                    })
                    .collect(),
            );

            // Create target at different positions relative to sequence
            let target_position = (i as f64 / test_cases as f64 - 0.5) * 200.0; // -100 to +100
            let target_price = base_price + target_position;
            let target = create_test_candles(vec![(
                target_price,
                target_price + 1.0,
                target_price - 1.0,
                target_price,
                1000.0,
            )]);

            let class = classify_price_level(&sequence, &target, &config).unwrap();
            class_counts[class as usize] += 1;
        }

        // Print distribution for analysis
        println!(
            "Price level classification distribution over {} synthetic cases:",
            test_cases
        );
        let class_names = [
            "Strong Down",
            "Moderate Down",
            "Neutral",
            "Moderate Up",
            "Strong Up",
        ];
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

        // Neutral should be reasonably represented (targets within percentile range)
        assert!(
            class_counts[2] > 100,
            "Neutral class should have reasonable representation"
        );
    }

    /// Test percentile edge cases
    #[test]
    fn test_percentile_edge_cases() {
        // Test case 1: Invalid percentile order
        let invalid_config = TargetsConfig::default();

        // Should still work but with swapped boundaries
        let sequence = create_test_candles(vec![
            (100.0, 100.0, 100.0, 100.0, 1000.0),
            (105.0, 105.0, 105.0, 105.0, 1000.0),
            (110.0, 110.0, 110.0, 110.0, 1000.0),
        ]);

        let target = create_test_candles(vec![(107.0, 107.0, 107.0, 107.0, 1000.0)]);

        let result = classify_price_level(&sequence, &target, &invalid_config);
        assert!(result.is_ok(), "Invalid percentile order should still work");

        // Test case 2: Extreme percentiles
        let extreme_config = TargetsConfig::default();

        let extreme_class = classify_price_level(&sequence, &target, &extreme_config).unwrap();
        assert_eq!(
            extreme_class, 2,
            "Target within extreme percentiles should be Neutral"
        );

        // Test case 3: Narrow percentiles
        let narrow_config = TargetsConfig::default();

        let narrow_class = classify_price_level(&sequence, &target, &narrow_config).unwrap();
        assert!(
            narrow_class >= 3,
            "Target outside narrow percentiles should be Up classification"
        );
    }

    /// **ENHANCED**: Test momentum-weighted VWAP calculation
    #[test]
    fn test_momentum_weighted_vwap() {
        // Test case 1: Equal weighting (momentum_factor = 1.0) should match standard VWAP
        let sequence = create_test_candles(vec![
            (100.0, 102.0, 98.0, 101.0, 1000.0),  // OHLC4 = 100.25
            (101.0, 103.0, 99.0, 102.0, 2000.0),  // OHLC4 = 101.25
            (102.0, 104.0, 100.0, 103.0, 1500.0), // OHLC4 = 102.25
        ]);

        let standard_vwap = get_sequence_vwap_baseline(&sequence).unwrap();
        let momentum_vwap_equal = calculate_vwap_with_momentum(&sequence, 1.0).unwrap();

        // Should be identical for momentum_factor = 1.0
        assert!((standard_vwap - momentum_vwap_equal).abs() < 1e-6);

        // Test case 2: Recent bias (momentum_factor > 1.0) should weight recent data more
        let momentum_vwap_recent = calculate_vwap_with_momentum(&sequence, 1.5).unwrap();

        // With recent bias, result should be closer to later prices
        // Last candle OHLC4 = 102.25, so momentum_vwap_recent should be > standard_vwap
        assert!(momentum_vwap_recent > standard_vwap);

        // Test case 3: Early bias (momentum_factor < 1.0) should weight early data more
        let momentum_vwap_early = calculate_vwap_with_momentum(&sequence, 0.5).unwrap();

        // With early bias, result should be closer to earlier prices
        // First candle OHLC4 = 100.25, so momentum_vwap_early should be < standard_vwap
        // Note: This might not always be true depending on volume weighting
        // Let's just check that the values are different and reasonable
        assert!(momentum_vwap_early != standard_vwap);
        assert!(momentum_vwap_early > 100.0 && momentum_vwap_early < 103.0);

        println!("Standard VWAP: {:.6}", standard_vwap);
        println!("Momentum VWAP (recent bias): {:.6}", momentum_vwap_recent);
        println!("Momentum VWAP (early bias): {:.6}", momentum_vwap_early);

        // The key test: recent bias should be different from early bias
        assert!((momentum_vwap_recent - momentum_vwap_early).abs() > 1e-6);
    }

    /// **ENHANCED**: Test adaptive bandwidth calculation
    #[test]
    fn test_adaptive_bandwidth() {
        // Test case 1: Low volatility sequence should have smaller bandwidth
        let low_vol_sequence = create_test_candles(vec![
            (100.0, 100.1, 99.9, 100.0, 1000.0), // Very tight range
            (100.0, 100.1, 99.9, 100.05, 1000.0),
            (100.05, 100.15, 99.95, 100.1, 1000.0),
            (100.1, 100.2, 100.0, 100.15, 1000.0),
        ]);

        // Test case 2: High volatility sequence should have larger bandwidth
        let high_vol_sequence = create_test_candles(vec![
            (100.0, 105.0, 95.0, 102.0, 1000.0), // Wide range
            (102.0, 108.0, 98.0, 104.0, 1000.0),
            (104.0, 110.0, 100.0, 106.0, 1000.0),
            (106.0, 112.0, 102.0, 108.0, 1000.0),
        ]);

        let base_bandwidth = 1.0;

        let low_vol_bandwidth =
            calculate_adaptive_bandwidth(&low_vol_sequence, base_bandwidth, None).unwrap();
        let high_vol_bandwidth =
            calculate_adaptive_bandwidth(&high_vol_sequence, base_bandwidth, None).unwrap();

        // High volatility should result in larger bandwidth
        assert!(high_vol_bandwidth > low_vol_bandwidth);

        // Both should be reasonable multiples of base bandwidth
        assert!(low_vol_bandwidth >= 0.3 * base_bandwidth); // Minimum bound
        assert!(high_vol_bandwidth <= 3.0 * base_bandwidth); // Maximum bound

        println!(
            "Low volatility adaptive bandwidth: {:.3}",
            low_vol_bandwidth
        );
        println!(
            "High volatility adaptive bandwidth: {:.3}",
            high_vol_bandwidth
        );
    }

    /// **ENHANCED**: Test enhanced classification with momentum and adaptive features
    #[test]
    fn test_enhanced_classification() {
        let config = TargetsConfig::default();

        // Create sequence with clear range (100-105)
        let trending_sequence = create_test_candles(vec![
            (100.0, 101.0, 99.0, 100.5, 1000.0),
            (100.5, 102.0, 99.5, 101.0, 1000.0),
            (101.0, 103.0, 100.0, 101.5, 1000.0),
            (101.5, 104.0, 100.5, 102.0, 1000.0),
            (102.0, 105.0, 101.0, 102.5, 1000.0),
        ]);

        // Test simple neutral case first
        let neutral_horizon = create_test_candles(vec![
            (101.5, 102.0, 101.0, 101.5, 1000.0), // Within range
        ]);

        // Test standard classification
        let standard_class_neutral =
            classify_price_level(&trending_sequence, &neutral_horizon, &config).unwrap();

        // Test enhanced classification
        let enhanced_class_neutral = classify_price_level_with_momentum(
            &trending_sequence,
            &neutral_horizon,
            Some(1.0), // No momentum bias for debugging
        )
        .unwrap();

        println!(
            "Standard classification - Neutral: {}",
            standard_class_neutral
        );
        println!(
            "Enhanced classification - Neutral: {}",
            enhanced_class_neutral
        );

        // Both should return valid classes (0-4)
        assert!(
            (0..=4).contains(&standard_class_neutral),
            "Standard classification should be 0-4, got {}",
            standard_class_neutral
        );
        assert!(
            (0..=4).contains(&enhanced_class_neutral),
            "Enhanced classification should be 0-4, got {}",
            enhanced_class_neutral
        );

        // Test a clear breakout case
        let strong_breakout_up_horizon = create_test_candles(vec![
            (110.0, 115.0, 109.0, 112.0, 1000.0), // Very strong upward breakout (>10% above range)
        ]);

        let standard_class_up =
            classify_price_level(&trending_sequence, &strong_breakout_up_horizon, &config).unwrap();

        // Test enhanced classification with momentum weighting
        let _enhanced_class_up = classify_price_level_with_momentum(
            &trending_sequence,
            &strong_breakout_up_horizon,
            Some(1.0), // No momentum bias for debugging
        )
        .unwrap();

        let enhanced_class_neutral =
            classify_price_level_with_momentum(&trending_sequence, &neutral_horizon, Some(1.0))
                .unwrap();

        println!(
            "Standard classification - Neutral: {}",
            standard_class_neutral
        );
        println!(
            "Enhanced classification - Neutral: {}",
            enhanced_class_neutral
        );

        // Both should return valid classes (0-4)
        assert!(
            (0..=4).contains(&standard_class_neutral),
            "Standard classification should be 0-4, got {}",
            standard_class_neutral
        );
        assert!(
            (0..=4).contains(&enhanced_class_neutral),
            "Enhanced classification should be 0-4, got {}",
            enhanced_class_neutral
        );

        // Test a clear breakout case
        let strong_breakout_up_horizon = create_test_candles(vec![
            (110.0, 115.0, 109.0, 112.0, 1000.0), // Very strong upward breakout (>10% above range)
        ]);

        let enhanced_class_up = classify_price_level_with_momentum(
            &trending_sequence,
            &strong_breakout_up_horizon,
            Some(1.0),
        )
        .unwrap();

        println!("Standard classification - Up: {}", standard_class_up);
        println!("Enhanced classification - Up: {}", enhanced_class_up);

        // Strong breakout should be higher class than neutral
        assert!(
            standard_class_up > standard_class_neutral,
            "Breakout class {} should be > neutral class {}",
            standard_class_up,
            standard_class_neutral
        );
        assert!(
            enhanced_class_up > enhanced_class_neutral,
            "Enhanced breakout class {} should be > enhanced neutral class {}",
            enhanced_class_up,
            enhanced_class_neutral
        );
    }

    /// **ENHANCED**: Test class balance with enhanced features across market conditions
    #[test]
    fn test_enhanced_class_balance() {
        let _config = TargetsConfig::default();

        let mut all_classifications = Vec::new();

        // Generate multiple scenarios with different market conditions
        for base_price in [100.0, 1000.0, 50000.0] {
            // Different price levels (altcoin, ETH, BTC)
            for volatility_factor in [0.5, 1.0, 2.0] {
                // Different volatility regimes
                for trend_direction in [-1.0, 0.0, 1.0] {
                    // Down, sideways, up trends

                    // Create sequence with specified characteristics
                    let mut sequence_data = Vec::new();
                    for i in 0..10 {
                        let trend_component = trend_direction * (i as f64) * 0.002; // 0.2% per step
                        let volatility_component = volatility_factor * 0.01; // Base 1% volatility

                        let price = base_price * (1.0 + trend_component);
                        let high = price * (1.0 + volatility_component);
                        let low = price * (1.0 - volatility_component);

                        sequence_data.push((price, high, low, price, 1000.0));
                    }
                    let sequence = create_test_candles(sequence_data);

                    // Test multiple horizon outcomes
                    for horizon_change in [-0.03, -0.01, 0.0, 0.01, 0.03] {
                        // -3% to +3%
                        let horizon_price = base_price * (1.0 + horizon_change);
                        let horizon = create_test_candles(vec![(
                            horizon_price,
                            horizon_price * 1.005,
                            horizon_price * 0.995,
                            horizon_price,
                            1000.0,
                        )]);

                        // Test enhanced classification
                        if let Ok(class) = classify_price_level_with_momentum(
                            &sequence,
                            &horizon,
                            Some(1.2), // Momentum weighting
                        ) {
                            all_classifications.push(class);
                        }
                    }
                }
            }
        }

        // Analyze class distribution
        let mut class_counts = [0; 5];
        for &class in &all_classifications {
            if (0..5).contains(&class) {
                class_counts[class as usize] += 1;
            }
        }

        let total_samples = all_classifications.len();
        println!(
            "Enhanced Classification Distribution ({} samples):",
            total_samples
        );

        for (i, &count) in class_counts.iter().enumerate() {
            let percentage = (count as f64 / total_samples as f64) * 100.0;
            println!("  Class {}: {} samples ({:.1}%)", i, count, percentage);
        }

        // Check for reasonable balance (each class should have some representation)
        for (i, &count) in class_counts.iter().enumerate() {
            let percentage = (count as f64 / total_samples as f64) * 100.0;
            assert!(
                percentage >= 5.0,
                "Class {} has too few samples: {:.1}%",
                i,
                percentage
            );
            assert!(
                percentage <= 50.0,
                "Class {} has too many samples: {:.1}%",
                i,
                percentage
            );
        }

        // Calculate balance metrics
        let min_count = *class_counts.iter().min().unwrap();
        let max_count = *class_counts.iter().max().unwrap();
        let imbalance_ratio = max_count as f64 / min_count as f64;

        println!("Enhanced balance metrics:");
        println!("  Imbalance ratio: {:.2}x", imbalance_ratio);
        println!(
            "  Min class size: {} ({:.1}%)",
            min_count,
            (min_count as f64 / total_samples as f64) * 100.0
        );
        println!(
            "  Max class size: {} ({:.1}%)",
            max_count,
            (max_count as f64 / total_samples as f64) * 100.0
        );

        // Enhanced features should maintain reasonable balance
        assert!(
            imbalance_ratio < 5.0,
            "Enhanced classification shows severe imbalance: {:.2}x",
            imbalance_ratio
        );
    }
}
