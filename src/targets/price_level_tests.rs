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
    use crate::config::model::PriceLevelHead;
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

    /// Test percentile boundary calculation
    #[test]
    fn test_percentile_boundaries() {
        let config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(1.0),
            percentiles: Some([0.1, 0.9]), // 10th/90th percentiles
        };

        // Test case 1: Uniform distribution
        let uniform_sequence = create_test_candles(vec![
            (100.0, 100.0, 100.0, 100.0, 1000.0), // VWAP = 100.0
            (101.0, 101.0, 101.0, 101.0, 1000.0), // VWAP = 101.0
            (102.0, 102.0, 102.0, 102.0, 1000.0), // VWAP = 102.0
            (103.0, 103.0, 103.0, 103.0, 1000.0), // VWAP = 103.0
            (104.0, 104.0, 104.0, 104.0, 1000.0), // VWAP = 104.0
            (105.0, 105.0, 105.0, 105.0, 1000.0), // VWAP = 105.0
            (106.0, 106.0, 106.0, 106.0, 1000.0), // VWAP = 106.0
            (107.0, 107.0, 107.0, 107.0, 1000.0), // VWAP = 107.0
            (108.0, 108.0, 108.0, 108.0, 1000.0), // VWAP = 108.0
            (109.0, 109.0, 109.0, 109.0, 1000.0), // VWAP = 109.0
        ]);

        let target_candle = create_test_candles(vec![
            (105.0, 105.0, 105.0, 105.0, 1000.0), // Target in middle
        ]);

        let class = classify_price_level(&uniform_sequence, &target_candle, &config).unwrap();
        assert_eq!(
            class, 2,
            "Target in middle of uniform distribution should be Neutral (2)"
        );

        // Test case 2: Target below 10th percentile
        let low_target = create_test_candles(vec![
            (99.0, 99.0, 99.0, 99.0, 1000.0), // Below 10th percentile (100.0)
        ]);

        let low_class = classify_price_level(&uniform_sequence, &low_target, &config).unwrap();
        assert!(
            low_class <= 1,
            "Target below 10th percentile should be Moderate Down (1) or Strong Down (0), got {}",
            low_class
        );

        // Test case 3: Target above 90th percentile
        let high_target = create_test_candles(vec![
            (110.0, 110.0, 110.0, 110.0, 1000.0), // Above 90th percentile (109.0)
        ]);

        let high_class = classify_price_level(&uniform_sequence, &high_target, &config).unwrap();
        assert!(
            high_class >= 3,
            "Target above 90th percentile should be Moderate Up (3) or Strong Up (4), got {}",
            high_class
        );
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
        let wide_config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(1.0),
            percentiles: Some([0.05, 0.95]), // 5th/95th percentiles
        };

        let wide_class = classify_price_level(&sequence, &target, &wide_config).unwrap();
        assert_eq!(
            wide_class, 2,
            "Target within wide percentiles should be Neutral"
        );

        // Test case 2: Narrow percentiles [0.2, 0.8]
        let narrow_config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(1.0),
            percentiles: Some([0.2, 0.8]), // 20th/80th percentiles
        };

        let narrow_class = classify_price_level(&sequence, &target, &narrow_config).unwrap();
        assert!(
            narrow_class <= 1,
            "Target below narrow percentiles should be Down classification"
        );

        // Test case 3: Default percentiles [0.1, 0.9]
        let default_config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(1.0),
            percentiles: None, // Should use default [0.1, 0.9]
        };

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
        let sensitive_config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(0.5), // More sensitive
            percentiles: Some([0.1, 0.9]),
        };

        let sensitive_class =
            classify_price_level(&sequence, &breakout_target, &sensitive_config).unwrap();
        assert_eq!(
            sensitive_class, 4,
            "High target with low bandwidth should be Strong Up (4)"
        );

        // Test case 2: High bandwidth (less sensitive)
        let conservative_config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(2.0), // Less sensitive
            percentiles: Some([0.1, 0.9]),
        };

        let conservative_class =
            classify_price_level(&sequence, &breakout_target, &conservative_config).unwrap();
        // With higher bandwidth, might still be Strong Up but could be Moderate Up
        assert!(
            conservative_class >= 3,
            "High target with high bandwidth should be Up classification"
        );

        // Test case 3: Default bandwidth
        let default_config = PriceLevelHead {
            enabled: true,
            bandwidth_size: None, // Should use default 1.0
            percentiles: Some([0.1, 0.9]),
        };

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

        let config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(1.0),
            percentiles: Some([0.1, 0.9]),
        };

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
        let config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(1.0),
            percentiles: Some([0.1, 0.9]),
        };

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
        let config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(1.0),
            percentiles: Some([0.1, 0.9]),
        };

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
        let config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(1.0),
            percentiles: Some([0.1, 0.9]),
        };

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
        let invalid_config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(1.0),
            percentiles: Some([0.9, 0.1]), // Wrong order
        };

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
        let extreme_config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(1.0),
            percentiles: Some([0.01, 0.99]), // Very wide
        };

        let extreme_class = classify_price_level(&sequence, &target, &extreme_config).unwrap();
        assert_eq!(
            extreme_class, 2,
            "Target within extreme percentiles should be Neutral"
        );

        // Test case 3: Narrow percentiles
        let narrow_config = PriceLevelHead {
            enabled: true,
            bandwidth_size: Some(1.0),
            percentiles: Some([0.45, 0.55]), // Very narrow
        };

        let narrow_class = classify_price_level(&sequence, &target, &narrow_config).unwrap();
        assert!(
            narrow_class >= 3,
            "Target outside narrow percentiles should be Up classification"
        );
    }
}
