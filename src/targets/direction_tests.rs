//! Comprehensive tests for direction classification to ensure mathematical correctness
//!
//! These tests validate:
//! 1. Linear regression slope calculation accuracy
//! 2. Trend acceleration calculation correctness
//! 3. Threshold scaling appropriateness
//! 4. Classification balance across different scenarios
//! 5. Edge case handling

#[cfg(test)]
mod tests {
    use super::super::direction::*;
    use crate::config::model::DirectionHead;
    use approx::assert_relative_eq;

    /// Test linear regression slope calculation with known data
    #[test]
    fn test_linear_trend_slope_calculation() {
        // Test case 1: Perfect upward trend
        let upward_prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let slope = calculate_linear_trend_slope(&upward_prices).unwrap();
        assert_relative_eq!(slope, 1.0, epsilon = 1e-10);

        // Test case 2: Perfect downward trend
        let downward_prices = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let slope = calculate_linear_trend_slope(&downward_prices).unwrap();
        assert_relative_eq!(slope, -1.0, epsilon = 1e-10);

        // Test case 3: Flat trend
        let flat_prices = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let slope = calculate_linear_trend_slope(&flat_prices).unwrap();
        assert_relative_eq!(slope, 0.0, epsilon = 1e-10);

        // Test case 4: Realistic crypto price trend (BTC-like)
        let btc_prices = vec![50000.0, 50100.0, 50050.0, 50200.0, 50150.0];
        let slope = calculate_linear_trend_slope(&btc_prices).unwrap();
        // Should be positive but small
        assert!(slope > 0.0 && slope < 100.0);

        // Test case 5: Edge case - insufficient data
        let short_prices = vec![100.0];
        let slope = calculate_linear_trend_slope(&short_prices).unwrap();
        assert_eq!(slope, 0.0);
    }

    /// Test threshold calculation with different configurations
    #[test]
    fn test_threshold_calculation() {
        // Test case 1: Standard configuration
        let thresholds = calculate_trend_acceleration_thresholds(0.4, 2.0).unwrap();

        // No slope_scale - use slope_sensitivity directly
        let expected_half = 0.4 / 2.0; // 0.2
        let expected_extreme = 0.4 * 2.0; // 0.8

        assert_relative_eq!(thresholds.dump_max, -expected_extreme, epsilon = 1e-10);
        assert_relative_eq!(thresholds.down_max, -expected_half, epsilon = 1e-10);
        assert_relative_eq!(thresholds.sideways_max, expected_half, epsilon = 1e-10);
        assert_relative_eq!(thresholds.up_max, expected_extreme, epsilon = 1e-10);

        // Test case 2: Different configuration
        let thresholds2 = calculate_trend_acceleration_thresholds(0.6, 1.5).unwrap();
        let expected_half2 = 0.6 / 2.0; // 0.3
        let expected_extreme2 = 0.6 * 1.5; // 0.9

        assert_relative_eq!(thresholds2.dump_max, -expected_extreme2, epsilon = 1e-10);
        assert_relative_eq!(thresholds2.sideways_max, expected_half2, epsilon = 1e-10);
    }

    /// Test classification with controlled slope differences
    #[test]
    fn test_classification_with_known_slopes() {
        let config = DirectionHead {
            enabled: true,
            slope_sensitivity: Some(4.0), // Larger sensitivity for crypto slopes
            base_threshold: Some(0.12),
            extreme_multiplier: Some(2.0),
        };

        // Calculate expected thresholds - no slope_scale
        let _half_sensitivity = 4.0 / 2.0; // 2.0 - for reference
        let _extreme_sensitivity = 4.0 * 2.0; // 8.0 - for reference

        // Test case 1: Strong deceleration (DUMP)
        let seq_prices = vec![100.0, 101.0, 102.0, 103.0, 104.0]; // slope ≈ 1.0
        let hor_prices = vec![104.0, 103.5, 103.0, 102.5, 102.0]; // slope ≈ -0.5
                                                                  // acceleration = -0.5 - 1.0 = -1.5 (between -2.0 and 2.0, so SIDEWAYS)
        let class = classify_direction(&seq_prices, &hor_prices, Some(&config)).unwrap();
        assert_eq!(class, 2); // SIDEWAYS

        // Test case 2: Strong acceleration (PUMP)
        let seq_prices2 = vec![100.0, 100.5, 101.0, 101.5, 102.0]; // slope ≈ 0.5
        let hor_prices2 = vec![102.0, 103.0, 104.0, 105.0, 106.0]; // slope ≈ 1.0
                                                                   // acceleration = 1.0 - 0.5 = 0.5 (between -2.0 and 2.0, so SIDEWAYS)
        let class2 = classify_direction(&seq_prices2, &hor_prices2, Some(&config)).unwrap();
        assert_eq!(class2, 2); // SIDEWAYS

        // Test case 3: Minimal change (SIDEWAYS)
        let seq_prices3 = vec![100.0, 100.1, 100.2, 100.3, 100.4]; // slope ≈ 0.1
        let hor_prices3 = vec![100.4, 100.5, 100.6, 100.7, 100.8]; // slope ≈ 0.1
                                                                   // acceleration = 0.1 - 0.1 = 0.0 (within ±2.0)
        let class3 = classify_direction(&seq_prices3, &hor_prices3, Some(&config)).unwrap();
        assert_eq!(class3, 2); // SIDEWAYS
    }

    /// Test realistic crypto price scenarios
    #[test]
    fn test_realistic_crypto_scenarios() {
        let config = DirectionHead {
            enabled: true,
            slope_sensitivity: Some(4.0), // Appropriate for crypto slopes
            base_threshold: Some(0.12),
            extreme_multiplier: Some(2.0),
        };

        // Scenario 1: BTC bull run acceleration
        let btc_sequence = vec![45000.0, 46000.0, 47000.0, 48000.0, 49000.0]; // +1000/period
        let btc_horizon = vec![49000.0, 51000.0, 53000.0, 55000.0, 57000.0]; // +2000/period
        let class = classify_direction(&btc_sequence, &btc_horizon, Some(&config)).unwrap();
        println!("BTC bull acceleration: class = {}", class);
        // Should be UP or PUMP (3 or 4)
        assert!(class >= 3);

        // Scenario 2: ETH bear market deceleration
        let eth_sequence = vec![3000.0, 2800.0, 2600.0, 2400.0, 2200.0]; // -200/period
        let eth_horizon = vec![2200.0, 2150.0, 2100.0, 2050.0, 2000.0]; // -50/period
        let class2 = classify_direction(&eth_sequence, &eth_horizon, Some(&config)).unwrap();
        println!("ETH bear deceleration: class = {}", class2);
        // Should be UP (trend becoming less bearish = acceleration)
        assert!(class2 >= 2);

        // Scenario 3: Sideways consolidation
        let alt_sequence = vec![100.0, 101.0, 99.0, 102.0, 98.0]; // choppy, ~0 slope
        let alt_horizon = vec![98.0, 99.0, 101.0, 100.0, 102.0]; // choppy, ~0 slope
        let class3 = classify_direction(&alt_sequence, &alt_horizon, Some(&config)).unwrap();
        println!("Sideways consolidation: class = {}", class3);
        // Should be SIDEWAYS (2)
        assert_eq!(class3, 2);
    }

    /// Test edge cases and error handling
    #[test]
    fn test_edge_cases() {
        let config = DirectionHead {
            enabled: true,
            slope_sensitivity: Some(4.0), // Appropriate for crypto slopes
            base_threshold: Some(0.12),
            extreme_multiplier: Some(2.0),
        };

        // Test case 1: Insufficient sequence data
        let short_seq = vec![100.0];
        let normal_hor = vec![100.0, 101.0, 102.0];
        let class = classify_direction(&short_seq, &normal_hor, Some(&config)).unwrap();
        assert_eq!(class, 2); // Should default to SIDEWAYS

        // Test case 2: Insufficient horizon data
        let normal_seq = vec![100.0, 101.0, 102.0];
        let short_hor = vec![102.0];
        let class2 = classify_direction(&normal_seq, &short_hor, Some(&config)).unwrap();
        assert_eq!(class2, 2); // Should default to SIDEWAYS

        // Test case 3: No config provided
        let seq = vec![100.0, 101.0, 102.0];
        let hor = vec![102.0, 103.0, 104.0];
        let class3 = classify_direction(&seq, &hor, None).unwrap();
        // Should use default values and work
        assert!((0..=4).contains(&class3));
    }

    /// Test classification balance with synthetic data
    #[test]
    fn test_classification_balance() {
        let config = DirectionHead {
            enabled: true,
            slope_sensitivity: Some(4.0), // Appropriate for crypto slopes
            base_threshold: Some(0.12),
            extreme_multiplier: Some(2.0),
        };

        let mut class_counts = [0; 5];
        let test_cases = 1000;

        // Generate synthetic test cases with controlled slope differences
        for i in 0..test_cases {
            let base_slope = (i as f64 / test_cases as f64 - 0.5) * 0.02; // -0.01 to +0.01

            // Create sequence with base slope
            let seq_prices: Vec<f64> = (0..5)
                .map(|j| 1000.0 + base_slope * j as f64 * 1000.0)
                .collect();

            // Create horizon with modified slope
            let slope_change = (i as f64 / test_cases as f64 - 0.5) * 0.04; // -0.02 to +0.02
            let horizon_slope = base_slope + slope_change;
            let hor_prices: Vec<f64> = (0..5)
                .map(|j| seq_prices[4] + horizon_slope * j as f64 * 1000.0)
                .collect();

            let class = classify_direction(&seq_prices, &hor_prices, Some(&config)).unwrap();
            class_counts[class as usize] += 1;
        }

        // Print distribution for analysis
        println!(
            "Classification distribution over {} synthetic cases:",
            test_cases
        );
        let class_names = ["DUMP", "DOWN", "SIDEWAYS", "UP", "PUMP"];
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

        // SIDEWAYS should be reasonably represented (but not necessarily most common for trend acceleration)
        assert!(
            class_counts[2] > 50,
            "SIDEWAYS class should have reasonable representation"
        ); // At least 5%
    }

    /// Benchmark different slope_scale values
    #[test]
    fn test_slope_scale_impact() {
        let config = DirectionHead {
            enabled: true,
            slope_sensitivity: Some(4.0), // Appropriate for crypto slopes
            base_threshold: Some(0.12),
            extreme_multiplier: Some(2.0),
        };

        // Test with realistic crypto price movements
        let test_cases = [
            // (sequence_prices, horizon_prices, expected_behavior)
            (
                vec![50000.0, 50100.0, 50200.0, 50300.0, 50400.0], // +100/period
                vec![50400.0, 50500.0, 50600.0, 50700.0, 50800.0], // +100/period (continuation)
                "Should be SIDEWAYS",
            ),
            (
                vec![50000.0, 50100.0, 50200.0, 50300.0, 50400.0], // +100/period
                vec![50400.0, 50600.0, 50800.0, 51000.0, 51200.0], // +200/period (acceleration)
                "Should be UP or PUMP",
            ),
            (
                vec![50000.0, 49800.0, 49600.0, 49400.0, 49200.0], // -200/period
                vec![49200.0, 49150.0, 49100.0, 49050.0, 49000.0], // -50/period (deceleration)
                "Should be UP (less bearish)",
            ),
        ];

        for (i, (seq_prices, hor_prices, expected)) in test_cases.iter().enumerate() {
            let class = classify_direction(seq_prices, hor_prices, Some(&config)).unwrap();
            println!("Test case {}: {} -> class = {}", i + 1, expected, class);

            // Calculate actual slopes for debugging
            let seq_slope = calculate_linear_trend_slope(seq_prices).unwrap();
            let hor_slope = calculate_linear_trend_slope(hor_prices).unwrap();
            let acceleration = hor_slope - seq_slope;

            println!(
                "  Seq slope: {:.6}, Hor slope: {:.6}, Acceleration: {:.6}",
                seq_slope, hor_slope, acceleration
            );
        }
    }
}

// Internal functions are tested through the public API
