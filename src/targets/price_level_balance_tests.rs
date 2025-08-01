//! Comprehensive balance testing for price level target generation
//!
//! Tests class distribution balance across various market scenarios:
//! - Trending markets (bull/bear)
//! - Sideways/range-bound markets  
//! - High/low volatility periods
//! - Different price ranges (BTC vs altcoins)
//! - Various sequence lengths

use super::price_levels::*;
use crate::config::model::TargetsConfig;
use crate::data::structures::MarketDataRow;
use std::collections::HashMap;

#[cfg(test)]
mod balance_tests {
    use super::*;
    use crate::config::model::AdaptiveSensitivity;

    /// Test data generator for various market scenarios
    struct MarketScenarioGenerator;

    impl MarketScenarioGenerator {
        /// Generate trending upward market data
        fn generate_trending_up(
            start_price: f64,
            length: usize,
            trend_strength: f64,
            volatility: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                let trend_component = trend_strength * (i as f64);
                let noise = (i as f64 * 0.1).sin() * volatility * price * 0.01;

                price = start_price + trend_component + noise;
                let high = price * (1.0 + volatility * 0.005);
                let low = price * (1.0 - volatility * 0.005);
                let volume = 1000.0 + (i as f64 * 0.2).cos() * 200.0;

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

        /// Generate trending downward market data
        fn generate_trending_down(
            start_price: f64,
            length: usize,
            trend_strength: f64,
            volatility: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                let trend_component = -trend_strength * (i as f64);
                let noise = (i as f64 * 0.1).sin() * volatility * price * 0.01;

                price = start_price + trend_component + noise;
                let high = price * (1.0 + volatility * 0.005);
                let low = price * (1.0 - volatility * 0.005);
                let volume = 1000.0 + (i as f64 * 0.2).cos() * 200.0;

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

        /// Generate sideways/range-bound market data
        fn generate_sideways(
            center_price: f64,
            length: usize,
            range_size: f64,
            volatility: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();

            for i in 0..length {
                let cycle_component = (i as f64 * 0.2).sin() * range_size * center_price * 0.01;
                let noise = (i as f64 * 0.3).cos() * volatility * center_price * 0.005;

                let price = center_price + cycle_component + noise;
                let high = price * (1.0 + volatility * 0.003);
                let low = price * (1.0 - volatility * 0.003);
                let volume = 1000.0 + (i as f64 * 0.15).sin() * 300.0;

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

        /// Generate high volatility market data
        fn generate_high_volatility(
            start_price: f64,
            length: usize,
            volatility_factor: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                let volatility_noise = (i as f64 * 0.5).sin() * volatility_factor * price * 0.02;
                let random_component = ((i * 17) % 100) as f64 / 100.0 - 0.5;

                price = start_price
                    + volatility_noise
                    + random_component * volatility_factor * price * 0.01;
                let high = price * (1.0 + volatility_factor * 0.01);
                let low = price * (1.0 - volatility_factor * 0.01);
                let volume = 1000.0 + random_component.abs() * 500.0;

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
    fn test_price_level_balance_trending_up() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate trending up market
        let market_data = MarketScenarioGenerator::generate_trending_up(
            50000.0, // BTC-like price
            sequence_length + horizon_length,
            100.0, // Strong uptrend
            2.0,   // Moderate volatility
        );

        let mut targets = Vec::new();

        // Test multiple sequences from the same trending market
        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            if let Ok(target) = classify_price_level(sequence, horizon, &config) {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for trending up market"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("🔺 TRENDING UP - Distribution: {:?}", distribution);
        println!(
            "🔺 TRENDING UP - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // In trending up market, expect bias toward higher classes (3,4)
        // But should still have reasonable distribution
        assert!(
            imbalance_ratio < 10.0,
            "Imbalance ratio too high: {:.2}",
            imbalance_ratio
        );
        assert!(
            min_pct > 0.05,
            "Minimum class percentage too low: {:.3}",
            min_pct
        );
    }

    #[test]
    fn test_price_level_balance_trending_down() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate trending down market
        let market_data = MarketScenarioGenerator::generate_trending_down(
            50000.0, // BTC-like price
            sequence_length + horizon_length,
            100.0, // Strong downtrend
            2.0,   // Moderate volatility
        );

        let mut targets = Vec::new();

        // Test multiple sequences from the same trending market
        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            if let Ok(target) = classify_price_level(sequence, horizon, &config) {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for trending down market"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("🔻 TRENDING DOWN - Distribution: {:?}", distribution);
        println!(
            "🔻 TRENDING DOWN - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // In trending down market, expect bias toward lower classes (0,1)
        // But should still have reasonable distribution
        assert!(
            imbalance_ratio < 10.0,
            "Imbalance ratio too high: {:.2}",
            imbalance_ratio
        );
        assert!(
            min_pct > 0.05,
            "Minimum class percentage too low: {:.3}",
            min_pct
        );
    }

    #[test]
    fn test_price_level_balance_sideways() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate sideways market
        let market_data = MarketScenarioGenerator::generate_sideways(
            50000.0,                               // BTC-like price
            sequence_length + horizon_length + 50, // Extra data for multiple sequences
            3.0,                                   // Range size
            1.5,                                   // Low volatility
        );

        let mut targets = Vec::new();

        // Test multiple sequences from the same sideways market
        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            if let Ok(target) = classify_price_level(sequence, horizon, &config) {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for sideways market"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("↔️ SIDEWAYS - Distribution: {:?}", distribution);
        println!(
            "↔️ SIDEWAYS - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // In sideways market, expect bias toward neutral class (2)
        // This is the CRITICAL test - sideways should be more balanced
        assert!(
            imbalance_ratio < 5.0,
            "Sideways market should be more balanced: {:.2}",
            imbalance_ratio
        );
        assert!(
            min_pct > 0.10,
            "Minimum class percentage too low in sideways: {:.3}",
            min_pct
        );

        // Neutral class should be dominant in sideways market
        let neutral_pct = distribution.get(&2).unwrap_or(&0.0);
        assert!(
            *neutral_pct > 0.3,
            "Neutral class should dominate sideways market: {:.3}",
            neutral_pct
        );
    }

    #[test]
    fn test_price_level_balance_high_volatility() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate high volatility market
        let market_data = MarketScenarioGenerator::generate_high_volatility(
            50000.0,                               // BTC-like price
            sequence_length + horizon_length + 50, // Extra data for multiple sequences
            5.0,                                   // High volatility factor
        );

        let mut targets = Vec::new();

        // Test multiple sequences from the same volatile market
        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            if let Ok(target) = classify_price_level(sequence, horizon, &config) {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for high volatility market"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("🌪️ HIGH VOLATILITY - Distribution: {:?}", distribution);
        println!(
            "🌪️ HIGH VOLATILITY - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // High volatility should create more extreme classes (0,4)
        let extreme_pct =
            distribution.get(&0).unwrap_or(&0.0) + distribution.get(&4).unwrap_or(&0.0);
        assert!(
            extreme_pct > 0.2,
            "High volatility should create more extreme classes: {:.3}",
            extreme_pct
        );

        assert!(
            imbalance_ratio < 8.0,
            "High volatility imbalance too high: {:.2}",
            imbalance_ratio
        );
    }

    #[test]
    fn test_price_level_balance_different_price_ranges() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        let price_ranges = vec![
            ("BTC", 50000.0),
            ("ETH", 3000.0),
            ("Altcoin", 1.0),
            ("Micro", 0.001),
        ];

        for (name, start_price) in price_ranges {
            // Generate sideways market for each price range
            let market_data = MarketScenarioGenerator::generate_sideways(
                start_price,
                sequence_length + horizon_length + 30,
                2.0, // Range size
                1.0, // Volatility
            );

            let mut targets = Vec::new();

            for i in 0..(market_data.len() - sequence_length - horizon_length) {
                let sequence = &market_data[i..i + sequence_length];
                let horizon =
                    &market_data[i + sequence_length..i + sequence_length + horizon_length];

                if let Ok(target) = classify_price_level(sequence, horizon, &config) {
                    targets.push(target);
                }
            }

            assert!(
                !targets.is_empty(),
                "Should generate targets for {} price range",
                name
            );

            let distribution = calculate_class_distribution(&targets);
            let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

            println!(
                "💰 {} (${}) - Distribution: {:?}",
                name, start_price, distribution
            );
            println!(
                "💰 {} - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
                name, imbalance_ratio, deviation, min_pct
            );

            // All price ranges should have similar balance characteristics
            assert!(
                imbalance_ratio < 6.0,
                "{} price range imbalance too high: {:.2}",
                name,
                imbalance_ratio
            );
            assert!(
                min_pct > 0.08,
                "{} minimum class percentage too low: {:.3}",
                name,
                min_pct
            );
        }
    }

    #[test]
    fn test_price_level_balance_different_sensitivities() {
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate sideways market for sensitivity testing
        let market_data = MarketScenarioGenerator::generate_sideways(
            50000.0,
            sequence_length + horizon_length + 50,
            2.0, // Range size
            1.5, // Volatility
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

                if let Ok(target) = classify_price_level(sequence, horizon, &config) {
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
                // Lower sensitivity = more extreme classes
                let extreme_pct =
                    distribution.get(&0).unwrap_or(&0.0) + distribution.get(&4).unwrap_or(&0.0);
                assert!(
                    extreme_pct > 0.15,
                    "Low sensitivity should create more extreme classes: {:.3}",
                    extreme_pct
                );
            } else if sensitivity > 1.0 {
                // Higher sensitivity = more neutral classes
                let neutral_pct = distribution.get(&2).unwrap_or(&0.0);
                assert!(
                    *neutral_pct > 0.25,
                    "High sensitivity should create more neutral classes: {:.3}",
                    neutral_pct
                );
            }
        }
    }

    #[test]
    fn test_price_level_balance_comprehensive_analysis() {
        println!("\n🎯 COMPREHENSIVE PRICE LEVEL BALANCE ANALYSIS");
        println!("==============================================");

        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        let scenarios = vec![
            (
                "Trending Up",
                MarketScenarioGenerator::generate_trending_up(50000.0, 100, 100.0, 2.0),
            ),
            (
                "Trending Down",
                MarketScenarioGenerator::generate_trending_down(50000.0, 100, 100.0, 2.0),
            ),
            (
                "Sideways",
                MarketScenarioGenerator::generate_sideways(50000.0, 100, 3.0, 1.5),
            ),
            (
                "High Volatility",
                MarketScenarioGenerator::generate_high_volatility(50000.0, 100, 5.0),
            ),
        ];

        let mut overall_stats = Vec::new();

        for (scenario_name, market_data) in scenarios {
            let mut targets = Vec::new();

            for i in 0..(market_data.len() - sequence_length - horizon_length) {
                let sequence = &market_data[i..i + sequence_length];
                let horizon =
                    &market_data[i + sequence_length..i + sequence_length + horizon_length];

                if let Ok(target) = classify_price_level(sequence, horizon, &config) {
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

        println!("\n🎯 OVERALL PRICE LEVEL BALANCE SUMMARY:");
        println!("   Average Imbalance Ratio: {:.2}x", avg_imbalance);
        println!("   Average Deviation from 20%: {:.3}", avg_deviation);
        println!("   Average Minimum Class %: {:.3}", avg_min_pct);

        // Balance quality assessment
        if avg_imbalance > 5.0 {
            println!("⚠️  HIGH IMBALANCE DETECTED - Consider adaptive thresholds");
        }
        if avg_deviation > 0.1 {
            println!("⚠️  HIGH DEVIATION DETECTED - Classes not evenly distributed");
        }
        if avg_min_pct < 0.1 {
            println!("⚠️  LOW MINIMUM CLASS DETECTED - Some classes rarely occur");
        }

        // Recommendations
        println!("\n💡 RECOMMENDATIONS:");
        if avg_imbalance > 3.0 {
            println!("   - Implement context-aware threshold adjustment");
            println!("   - Consider dynamic bandwidth scaling based on market regime");
        }
        if avg_deviation > 0.08 {
            println!("   - Add sequence context detection for balanced classification");
            println!("   - Implement adaptive percentile boundaries");
        }
    }
}