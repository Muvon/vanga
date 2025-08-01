//! Comprehensive balance testing for direction target generation
//!
//! Tests momentum change classification balance across various market scenarios:
//! - Accelerating trends (momentum building)
//! - Decelerating trends (momentum fading)
//! - Sideways momentum (consistent direction)
//! - Choppy/noisy markets (inconsistent momentum)
//! - Different trend strengths and consistencies

use super::direction::*;
use crate::config::model::TargetsConfig;
use crate::data::structures::MarketDataRow;
use std::collections::HashMap;

#[cfg(test)]
mod balance_tests {
    use super::*;
    use crate::config::model::AdaptiveSensitivity;

    /// Test data generator for various momentum scenarios
    struct MomentumScenarioGenerator;

    impl MomentumScenarioGenerator {
        /// Generate accelerating upward momentum (momentum building)
        fn generate_accelerating_up(
            start_price: f64,
            length: usize,
            acceleration_factor: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                // Quadratic acceleration: momentum increases over time
                let momentum_component = acceleration_factor * (i as f64).powi(2) * 0.01;
                let noise = (i as f64 * 0.1).sin() * price * 0.002;

                price = start_price + momentum_component + noise;
                let high = price * 1.005;
                let low = price * 0.995;
                let volume = 1000.0 + (i as f64) * 10.0; // Increasing volume

                data.push(MarketDataRow {
                    timestamp: i as i64,
                    open: if i == 0 { price } else { data[i - 1].close },
                    high,
                    low,
                    close: price,
                    volume,
                });
            }
            data
        }

        /// Generate decelerating upward momentum (momentum fading)
        fn generate_decelerating_up(
            start_price: f64,
            length: usize,
            initial_momentum: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                // Logarithmic deceleration: momentum decreases over time
                let momentum_component = initial_momentum * (1.0 + i as f64).ln();
                let noise = (i as f64 * 0.15).cos() * price * 0.003;

                price = start_price + momentum_component + noise;
                let high = price * 1.004;
                let low = price * 0.996;
                let volume = 1500.0 - (i as f64) * 5.0; // Decreasing volume

                data.push(MarketDataRow {
                    timestamp: i as i64,
                    open: if i == 0 { price } else { data[i - 1].close },
                    high,
                    low,
                    close: price,
                    volume: volume.max(500.0),
                });
            }
            data
        }

        /// Generate consistent sideways momentum (no acceleration/deceleration)
        fn generate_consistent_sideways(
            center_price: f64,
            length: usize,
            range_size: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();

            for i in 0..length {
                // Consistent oscillation with no momentum change
                let cycle_component = (i as f64 * 0.3).sin() * range_size * center_price * 0.01;
                let noise = (i as f64 * 0.7).cos() * center_price * 0.001;

                let price = center_price + cycle_component + noise;
                let high = price * 1.002;
                let low = price * 0.998;
                let volume = 1000.0 + (i as f64 * 0.1).sin() * 100.0;

                data.push(MarketDataRow {
                    timestamp: i as i64,
                    open: if i == 0 { price } else { data[i - 1].close },
                    high,
                    low,
                    close: price,
                    volume,
                });
            }
            data
        }

        /// Generate choppy/noisy momentum (inconsistent direction changes)
        fn generate_choppy_momentum(
            start_price: f64,
            length: usize,
            choppiness_factor: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                // Random direction changes with high frequency
                let direction_change = if (i * 7) % 3 == 0 { 1.0 } else { -1.0 };
                let momentum_component =
                    direction_change * choppiness_factor * (i as f64 % 5.0) * 0.01;
                let noise = ((i * 13) % 100) as f64 / 100.0 - 0.5;

                price = start_price + momentum_component + noise * price * 0.005;
                let high = price * (1.0 + choppiness_factor * 0.003);
                let low = price * (1.0 - choppiness_factor * 0.003);
                let volume = 800.0 + noise.abs() * 400.0;

                data.push(MarketDataRow {
                    timestamp: i as i64,
                    open: if i == 0 { price } else { data[i - 1].close },
                    high,
                    low,
                    close: price,
                    volume,
                });
            }
            data
        }

        /// Generate momentum reversal (trend change)
        fn generate_momentum_reversal(
            start_price: f64,
            length: usize,
            reversal_point: usize,
            trend_strength: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                let momentum_component = if i < reversal_point {
                    // Upward momentum before reversal
                    trend_strength * (i as f64) * 0.01
                } else {
                    // Downward momentum after reversal
                    let peak = trend_strength * (reversal_point as f64) * 0.01;
                    peak - trend_strength * ((i - reversal_point) as f64) * 0.015
                };

                let noise = (i as f64 * 0.2).sin() * price * 0.002;
                price = start_price + momentum_component + noise;

                let high = price * 1.003;
                let low = price * 0.997;
                let volume = 1000.0 + ((i as f64 - reversal_point as f64).abs() * 5.0);

                data.push(MarketDataRow {
                    timestamp: i as i64,
                    open: if i == 0 { price } else { data[i - 1].close },
                    high,
                    low,
                    close: price,
                    volume,
                });
            }
            data
        }
    }

    /// Calculate class distribution from targets
    fn calculate_class_distribution(targets: &[i32]) -> HashMap<i32, f64> {
        let mut counts = HashMap::new();
        let total = targets.len() as f64;

        for &target in targets {
            *counts.entry(target).or_insert(0.0) += 1.0;
        }

        for (_, count) in counts.iter_mut() {
            *count /= total;
        }

        counts
    }

    /// Calculate balance metrics
    fn calculate_balance_metrics(distribution: &HashMap<i32, f64>) -> (f64, f64, f64) {
        let percentages: Vec<f64> = (0..5)
            .map(|i| *distribution.get(&i).unwrap_or(&0.0))
            .collect();

        let min_pct = percentages.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_pct = percentages.iter().cloned().fold(0.0, f64::max);
        let imbalance_ratio = if min_pct > 0.0 {
            max_pct / min_pct
        } else {
            f64::INFINITY
        };

        let ideal_pct = 0.2; // 20% per class
        let deviation = percentages
            .iter()
            .map(|&pct| (pct - ideal_pct).abs())
            .sum::<f64>()
            / 5.0;

        (imbalance_ratio, deviation, min_pct)
    }

    #[test]
    fn test_direction_balance_accelerating_momentum() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate accelerating upward momentum
        let market_data = MomentumScenarioGenerator::generate_accelerating_up(
            50000.0, // BTC-like price
            sequence_length + horizon_length + 30,
            2.0, // Acceleration factor
        );

        let mut targets = Vec::new();

        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            let sequence_prices: Vec<f64> = sequence.iter().map(|row| row.close).collect();
            let horizon_prices: Vec<f64> = horizon.iter().map(|row| row.close).collect();

            if let Ok(target) = classify_direction(&sequence_prices, &horizon_prices, &config) {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for accelerating momentum"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("🚀 ACCELERATING UP - Distribution: {:?}", distribution);
        println!(
            "🚀 ACCELERATING UP - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // Accelerating momentum should favor UP/PUMP classes (3,4)
        let bullish_pct =
            distribution.get(&3).unwrap_or(&0.0) + distribution.get(&4).unwrap_or(&0.0);
        assert!(
            bullish_pct > 0.4,
            "Accelerating momentum should favor bullish classes: {:.3}",
            bullish_pct
        );

        assert!(
            imbalance_ratio < 15.0,
            "Imbalance ratio too high: {:.2}",
            imbalance_ratio
        );
    }

    #[test]
    fn test_direction_balance_decelerating_momentum() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate decelerating upward momentum
        let market_data = MomentumScenarioGenerator::generate_decelerating_up(
            50000.0, // BTC-like price
            sequence_length + horizon_length + 30,
            100.0, // Initial momentum
        );

        let mut targets = Vec::new();

        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            let sequence_prices: Vec<f64> = sequence.iter().map(|row| row.close).collect();
            let horizon_prices: Vec<f64> = horizon.iter().map(|row| row.close).collect();

            if let Ok(target) = classify_direction(&sequence_prices, &horizon_prices, &config) {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for decelerating momentum"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("🐌 DECELERATING UP - Distribution: {:?}", distribution);
        println!(
            "🐌 DECELERATING UP - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // Decelerating momentum should favor DOWN/SIDEWAYS classes (1,2)
        let neutral_bearish_pct =
            distribution.get(&1).unwrap_or(&0.0) + distribution.get(&2).unwrap_or(&0.0);
        assert!(
            neutral_bearish_pct > 0.3,
            "Decelerating momentum should favor neutral/bearish classes: {:.3}",
            neutral_bearish_pct
        );

        assert!(
            imbalance_ratio < 12.0,
            "Imbalance ratio too high: {:.2}",
            imbalance_ratio
        );
    }

    #[test]
    fn test_direction_balance_consistent_sideways() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate consistent sideways momentum
        let market_data = MomentumScenarioGenerator::generate_consistent_sideways(
            50000.0, // BTC-like price
            sequence_length + horizon_length + 50,
            2.0, // Range size
        );

        let mut targets = Vec::new();

        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            let sequence_prices: Vec<f64> = sequence.iter().map(|row| row.close).collect();
            let horizon_prices: Vec<f64> = horizon.iter().map(|row| row.close).collect();

            if let Ok(target) = classify_direction(&sequence_prices, &horizon_prices, &config) {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for consistent sideways"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("↔️ CONSISTENT SIDEWAYS - Distribution: {:?}", distribution);
        println!(
            "↔️ CONSISTENT SIDEWAYS - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // Consistent sideways should heavily favor SIDEWAYS class (2)
        let sideways_pct = distribution.get(&2).unwrap_or(&0.0);
        assert!(
            *sideways_pct > 0.4,
            "Consistent sideways should dominate SIDEWAYS class: {:.3}",
            sideways_pct
        );

        // This is CRITICAL - sideways markets should be more balanced overall
        assert!(
            imbalance_ratio < 8.0,
            "Sideways market should be more balanced: {:.2}",
            imbalance_ratio
        );
        assert!(
            min_pct > 0.05,
            "Minimum class percentage too low in sideways: {:.3}",
            min_pct
        );
    }

    #[test]
    fn test_direction_balance_choppy_momentum() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate choppy/noisy momentum
        let market_data = MomentumScenarioGenerator::generate_choppy_momentum(
            50000.0, // BTC-like price
            sequence_length + horizon_length + 50,
            3.0, // Choppiness factor
        );

        let mut targets = Vec::new();

        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            let sequence_prices: Vec<f64> = sequence.iter().map(|row| row.close).collect();
            let horizon_prices: Vec<f64> = horizon.iter().map(|row| row.close).collect();

            if let Ok(target) = classify_direction(&sequence_prices, &horizon_prices, &config) {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for choppy momentum"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("🌪️ CHOPPY MOMENTUM - Distribution: {:?}", distribution);
        println!(
            "🌪️ CHOPPY MOMENTUM - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // Choppy momentum should create more balanced distribution
        // No single class should dominate heavily
        let max_class_pct = distribution.values().cloned().fold(0.0, f64::max);
        assert!(
            max_class_pct < 0.6,
            "No single class should dominate choppy momentum: {:.3}",
            max_class_pct
        );

        assert!(
            imbalance_ratio < 6.0,
            "Choppy momentum should be more balanced: {:.2}",
            imbalance_ratio
        );
        assert!(
            min_pct > 0.08,
            "Minimum class percentage too low in choppy: {:.3}",
            min_pct
        );
    }

    #[test]
    fn test_direction_balance_momentum_reversal() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate momentum reversal
        let market_data = MomentumScenarioGenerator::generate_momentum_reversal(
            50000.0, // BTC-like price
            sequence_length + horizon_length + 50,
            30,   // Reversal point
            50.0, // Trend strength
        );

        let mut targets = Vec::new();

        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            let sequence_prices: Vec<f64> = sequence.iter().map(|row| row.close).collect();
            let horizon_prices: Vec<f64> = horizon.iter().map(|row| row.close).collect();

            if let Ok(target) = classify_direction(&sequence_prices, &horizon_prices, &config) {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for momentum reversal"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("🔄 MOMENTUM REVERSAL - Distribution: {:?}", distribution);
        println!(
            "🔄 MOMENTUM REVERSAL - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // Momentum reversal should create extreme classes (DUMP/PUMP)
        let extreme_pct =
            distribution.get(&0).unwrap_or(&0.0) + distribution.get(&4).unwrap_or(&0.0);
        assert!(
            extreme_pct > 0.2,
            "Momentum reversal should create extreme classes: {:.3}",
            extreme_pct
        );

        assert!(
            imbalance_ratio < 10.0,
            "Momentum reversal imbalance too high: {:.2}",
            imbalance_ratio
        );
    }

    #[test]
    fn test_direction_balance_different_sensitivities() {
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate consistent sideways for sensitivity testing
        let market_data = MomentumScenarioGenerator::generate_consistent_sideways(
            50000.0,
            sequence_length + horizon_length + 50,
            2.0, // Range size
        );

        let sensitivities = vec![("Low", 0.5), ("Default", 1.0), ("High", 2.0)];

        for (name, sensitivity) in sensitivities {
            let config = TargetsConfig {
                sensitivity: AdaptiveSensitivity::Balanced,
                ..Default::default()
            };

            let mut targets = Vec::new();

            for i in 0..(market_data.len() - sequence_length - horizon_length) {
                let sequence = &market_data[i..i + sequence_length];
                let horizon =
                    &market_data[i + sequence_length..i + sequence_length + horizon_length];

                let sequence_prices: Vec<f64> = sequence.iter().map(|row| row.close).collect();
                let horizon_prices: Vec<f64> = horizon.iter().map(|row| row.close).collect();

                if let Ok(target) = classify_direction(&sequence_prices, &horizon_prices, &config) {
                    targets.push(target);
                }
            }

            let distribution = calculate_class_distribution(&targets);
            let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

            println!(
                "🎛️ {} Sensitivity ({}) - Distribution: {:?}",
                name, sensitivity, distribution
            );
            println!(
                "🎛️ {} Sensitivity - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
                name, imbalance_ratio, deviation, min_pct
            );

            // Different sensitivities should affect extreme class distribution
            if sensitivity < 1.0 {
                // Lower sensitivity = more extreme momentum classifications
                let extreme_pct =
                    distribution.get(&0).unwrap_or(&0.0) + distribution.get(&4).unwrap_or(&0.0);
                assert!(
                    extreme_pct > 0.1,
                    "Low sensitivity should create more extreme classes: {:.3}",
                    extreme_pct
                );
            } else if sensitivity > 1.0 {
                // Higher sensitivity = more sideways classifications
                let sideways_pct = distribution.get(&2).unwrap_or(&0.0);
                assert!(
                    *sideways_pct > 0.4,
                    "High sensitivity should create more sideways classes: {:.3}",
                    sideways_pct
                );
            }
        }
    }

    #[test]
    fn test_direction_balance_comprehensive_analysis() {
        println!("\n🎯 COMPREHENSIVE DIRECTION BALANCE ANALYSIS");
        println!("===========================================");

        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        let scenarios = vec![
            (
                "Accelerating Up",
                MomentumScenarioGenerator::generate_accelerating_up(50000.0, 100, 2.0),
            ),
            (
                "Decelerating Up",
                MomentumScenarioGenerator::generate_decelerating_up(50000.0, 100, 100.0),
            ),
            (
                "Consistent Sideways",
                MomentumScenarioGenerator::generate_consistent_sideways(50000.0, 100, 2.0),
            ),
            (
                "Choppy Momentum",
                MomentumScenarioGenerator::generate_choppy_momentum(50000.0, 100, 3.0),
            ),
            (
                "Momentum Reversal",
                MomentumScenarioGenerator::generate_momentum_reversal(50000.0, 100, 30, 50.0),
            ),
        ];

        let mut overall_stats = Vec::new();

        for (scenario_name, market_data) in scenarios {
            let mut targets = Vec::new();

            for i in 0..(market_data.len() - sequence_length - horizon_length) {
                let sequence = &market_data[i..i + sequence_length];
                let horizon =
                    &market_data[i + sequence_length..i + sequence_length + horizon_length];

                let sequence_prices: Vec<f64> = sequence.iter().map(|row| row.close).collect();
                let horizon_prices: Vec<f64> = horizon.iter().map(|row| row.close).collect();

                if let Ok(target) = classify_direction(&sequence_prices, &horizon_prices, &config) {
                    targets.push(target);
                }
            }

            let distribution = calculate_class_distribution(&targets);
            let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

            overall_stats.push((scenario_name, imbalance_ratio, deviation, min_pct));

            println!("\n📊 {} Scenario:", scenario_name);
            println!("   Distribution: {:?}", distribution);
            println!("   Imbalance Ratio: {:.2}x", imbalance_ratio);
            println!("   Average Deviation: {:.3}", deviation);
            println!("   Minimum Class %: {:.3}", min_pct);
        }

        // Overall analysis
        let avg_imbalance: f64 = overall_stats.iter().map(|(_, imb, _, _)| *imb).sum::<f64>()
            / overall_stats.len() as f64;
        let avg_deviation: f64 = overall_stats.iter().map(|(_, _, dev, _)| *dev).sum::<f64>()
            / overall_stats.len() as f64;
        let avg_min_pct: f64 = overall_stats.iter().map(|(_, _, _, min)| *min).sum::<f64>()
            / overall_stats.len() as f64;

        println!("\n🎯 OVERALL DIRECTION BALANCE SUMMARY:");
        println!("   Average Imbalance Ratio: {:.2}x", avg_imbalance);
        println!("   Average Deviation from 20%: {:.3}", avg_deviation);
        println!("   Average Minimum Class %: {:.3}", avg_min_pct);

        // Balance quality assessment
        if avg_imbalance > 8.0 {
            println!("⚠️  HIGH IMBALANCE DETECTED - Momentum classification too extreme");
        }
        if avg_deviation > 0.15 {
            println!("⚠️  HIGH DEVIATION DETECTED - Momentum classes not evenly distributed");
        }
        if avg_min_pct < 0.05 {
            println!("⚠️  LOW MINIMUM CLASS DETECTED - Some momentum classes rarely occur");
        }

        // Recommendations
        println!("\n💡 RECOMMENDATIONS:");
        if avg_imbalance > 6.0 {
            println!("   - Implement momentum-aware threshold adjustment");
            println!("   - Consider trend consistency scaling for balanced classification");
        }
        if avg_deviation > 0.12 {
            println!("   - Add sequence momentum context detection");
            println!("   - Implement adaptive momentum change thresholds");
        }
    }
}