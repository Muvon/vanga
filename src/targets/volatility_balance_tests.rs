//! Comprehensive balance testing for volatility target generation
//!
//! Tests ATR-based volatility regime classification balance across various scenarios:
//! - Low volatility periods (stable markets)
//! - High volatility periods (volatile markets)
//! - Volatility expansion (increasing volatility)
//! - Volatility contraction (decreasing volatility)
//! - Mixed volatility regimes

use crate::config::model::TargetsConfig;
use crate::data::structures::MarketDataRow;
use crate::targets::volatility::classify_volatility_with_distribution_analysis;
use std::collections::HashMap;

#[cfg(test)]
mod balance_tests {
    use super::*;
    use crate::config::model::AdaptiveSensitivity;

    /// Test data generator for various volatility scenarios
    struct VolatilityScenarioGenerator;

    impl VolatilityScenarioGenerator {
        /// Generate low volatility market data (stable/calm periods)
        fn generate_low_volatility(
            start_price: f64,
            length: usize,
            base_volatility: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                // Small, consistent price movements
                let trend_component = (i as f64 * 0.05).sin() * price * 0.002;
                let noise = (i as f64 * 0.3).cos() * base_volatility * price * 0.001;

                price = start_price + trend_component + noise;

                // Low volatility = tight high-low ranges
                let volatility_factor = base_volatility * 0.002;
                let high = price * (1.0 + volatility_factor);
                let low = price * (1.0 - volatility_factor);
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

        /// Generate high volatility market data (volatile/unstable periods)
        fn generate_high_volatility(
            start_price: f64,
            length: usize,
            volatility_factor: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                // Large, erratic price movements
                let volatility_component =
                    (i as f64 * 0.7).sin() * volatility_factor * price * 0.01;
                let noise = ((i * 13) % 100) as f64 / 100.0 - 0.5;
                let random_shock = if (i * 7) % 20 == 0 {
                    noise * volatility_factor * 0.02
                } else {
                    0.0
                };

                price = start_price
                    + volatility_component
                    + noise * price * 0.005
                    + random_shock * price;

                // High volatility = wide high-low ranges
                let volatility_range = volatility_factor * 0.01;
                let high = price * (1.0 + volatility_range);
                let low = price * (1.0 - volatility_range);
                let volume = 1000.0 + noise.abs() * 800.0;

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

        /// Generate volatility expansion (increasing volatility over time)
        fn generate_volatility_expansion(
            start_price: f64,
            length: usize,
            expansion_rate: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                // Volatility increases over time
                let current_volatility = 1.0 + (i as f64 / length as f64) * expansion_rate;
                let volatility_component =
                    (i as f64 * 0.4).sin() * current_volatility * price * 0.005;
                let noise = (i as f64 * 0.6).cos() * current_volatility * price * 0.003;

                price = start_price + volatility_component + noise;

                // Expanding high-low ranges
                let volatility_range = current_volatility * 0.005;
                let high = price * (1.0 + volatility_range);
                let low = price * (1.0 - volatility_range);
                let volume = 1000.0 + current_volatility * 200.0;

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

        /// Generate volatility contraction (decreasing volatility over time)
        fn generate_volatility_contraction(
            start_price: f64,
            length: usize,
            initial_volatility: f64,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                // Volatility decreases over time
                let current_volatility =
                    initial_volatility * (1.0 - (i as f64 / length as f64) * 0.8);
                let volatility_component =
                    (i as f64 * 0.3).sin() * current_volatility * price * 0.008;
                let noise = (i as f64 * 0.5).cos() * current_volatility * price * 0.004;

                price = start_price + volatility_component + noise;

                // Contracting high-low ranges
                let volatility_range = current_volatility * 0.006;
                let high = price * (1.0 + volatility_range);
                let low = price * (1.0 - volatility_range);
                let volume = 1500.0 - current_volatility * 300.0;

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

        /// Generate mixed volatility regimes (alternating periods)
        fn generate_mixed_volatility(
            start_price: f64,
            length: usize,
            regime_length: usize,
        ) -> Vec<MarketDataRow> {
            let mut data: Vec<MarketDataRow> = Vec::new();
            let mut price = start_price;

            for i in 0..length {
                // Alternate between high and low volatility regimes
                let regime_index = (i / regime_length) % 2;
                let current_volatility = if regime_index == 0 { 1.0 } else { 4.0 };

                let volatility_component =
                    (i as f64 * 0.4).sin() * current_volatility * price * 0.003;
                let noise = ((i * 11) % 100) as f64 / 100.0 - 0.5;

                price =
                    start_price + volatility_component + noise * current_volatility * price * 0.002;

                // Regime-dependent high-low ranges
                let volatility_range = current_volatility * 0.004;
                let high = price * (1.0 + volatility_range);
                let low = price * (1.0 - volatility_range);
                let volume = 1000.0 + current_volatility * 150.0;

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
    fn test_volatility_balance_low_volatility() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate low volatility market
        let market_data = VolatilityScenarioGenerator::generate_low_volatility(
            50000.0, // BTC-like price
            sequence_length + horizon_length + 30,
            1.0, // Base volatility
        );

        let mut targets = Vec::new();

        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            if let Ok(target) =
                classify_volatility_with_distribution_analysis(sequence, horizon, &config)
            {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for low volatility"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("😴 LOW VOLATILITY - Distribution: {:?}", distribution);
        println!(
            "😴 LOW VOLATILITY - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // Low volatility should favor VeryLow/Low/Medium classes (0,1,2)
        let low_vol_pct = distribution.get(&0).unwrap_or(&0.0)
            + distribution.get(&1).unwrap_or(&0.0)
            + distribution.get(&2).unwrap_or(&0.0);
        assert!(
            low_vol_pct > 0.6,
            "Low volatility should favor low volatility classes: {:.3}",
            low_vol_pct
        );

        assert!(
            imbalance_ratio < 12.0,
            "Imbalance ratio too high: {:.2}",
            imbalance_ratio
        );
    }

    #[test]
    fn test_volatility_balance_high_volatility() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate high volatility market
        let market_data = VolatilityScenarioGenerator::generate_high_volatility(
            50000.0, // BTC-like price
            sequence_length + horizon_length + 30,
            5.0, // High volatility factor
        );

        let mut targets = Vec::new();

        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            if let Ok(target) =
                classify_volatility_with_distribution_analysis(sequence, horizon, &config)
            {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for high volatility"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("🌪️ HIGH VOLATILITY - Distribution: {:?}", distribution);
        println!(
            "🌪️ HIGH VOLATILITY - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // High volatility should favor Medium/High/VeryHigh classes (2,3,4)
        let high_vol_pct = distribution.get(&2).unwrap_or(&0.0)
            + distribution.get(&3).unwrap_or(&0.0)
            + distribution.get(&4).unwrap_or(&0.0);
        assert!(
            high_vol_pct > 0.6,
            "High volatility should favor high volatility classes: {:.3}",
            high_vol_pct
        );

        assert!(
            imbalance_ratio < 15.0,
            "Imbalance ratio too high: {:.2}",
            imbalance_ratio
        );
    }

    #[test]
    fn test_volatility_balance_expansion() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate volatility expansion
        let market_data = VolatilityScenarioGenerator::generate_volatility_expansion(
            50000.0, // BTC-like price
            sequence_length + horizon_length + 50,
            3.0, // Expansion rate
        );

        let mut targets = Vec::new();

        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            if let Ok(target) =
                classify_volatility_with_distribution_analysis(sequence, horizon, &config)
            {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for volatility expansion"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("📈 VOLATILITY EXPANSION - Distribution: {:?}", distribution);
        println!(
            "📈 VOLATILITY EXPANSION - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // Volatility expansion should favor High/VeryHigh classes (3,4)
        let expanding_vol_pct =
            distribution.get(&3).unwrap_or(&0.0) + distribution.get(&4).unwrap_or(&0.0);
        assert!(
            expanding_vol_pct > 0.3,
            "Volatility expansion should favor high volatility classes: {:.3}",
            expanding_vol_pct
        );

        assert!(
            imbalance_ratio < 10.0,
            "Imbalance ratio too high: {:.2}",
            imbalance_ratio
        );
    }

    #[test]
    fn test_volatility_balance_contraction() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate volatility contraction
        let market_data = VolatilityScenarioGenerator::generate_volatility_contraction(
            50000.0, // BTC-like price
            sequence_length + horizon_length + 50,
            4.0, // Initial volatility
        );

        let mut targets = Vec::new();

        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            if let Ok(target) =
                classify_volatility_with_distribution_analysis(sequence, horizon, &config)
            {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for volatility contraction"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!(
            "📉 VOLATILITY CONTRACTION - Distribution: {:?}",
            distribution
        );
        println!(
            "📉 VOLATILITY CONTRACTION - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // Volatility contraction should favor VeryLow/Low classes (0,1)
        let contracting_vol_pct =
            distribution.get(&0).unwrap_or(&0.0) + distribution.get(&1).unwrap_or(&0.0);
        assert!(
            contracting_vol_pct > 0.3,
            "Volatility contraction should favor low volatility classes: {:.3}",
            contracting_vol_pct
        );

        assert!(
            imbalance_ratio < 10.0,
            "Imbalance ratio too high: {:.2}",
            imbalance_ratio
        );
    }

    #[test]
    fn test_volatility_balance_mixed_regimes() {
        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate mixed volatility regimes
        let market_data = VolatilityScenarioGenerator::generate_mixed_volatility(
            50000.0,                                // BTC-like price
            sequence_length + horizon_length + 100, // Longer for regime changes
            20,                                     // Regime length
        );

        let mut targets = Vec::new();

        for i in 0..(market_data.len() - sequence_length - horizon_length) {
            let sequence = &market_data[i..i + sequence_length];
            let horizon = &market_data[i + sequence_length..i + sequence_length + horizon_length];

            if let Ok(target) =
                classify_volatility_with_distribution_analysis(sequence, horizon, &config)
            {
                targets.push(target);
            }
        }

        assert!(
            !targets.is_empty(),
            "Should generate targets for mixed volatility regimes"
        );

        let distribution = calculate_class_distribution(&targets);
        let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

        println!("🔄 MIXED VOLATILITY - Distribution: {:?}", distribution);
        println!(
            "🔄 MIXED VOLATILITY - Imbalance: {:.2}x, Deviation: {:.3}, Min: {:.3}",
            imbalance_ratio, deviation, min_pct
        );

        // Mixed regimes should create more balanced distribution
        // No single class should dominate heavily
        let max_class_pct = distribution.values().cloned().fold(0.0, f64::max);
        assert!(
            max_class_pct < 0.7,
            "No single class should dominate mixed regimes: {:.3}",
            max_class_pct
        );

        // This is CRITICAL - mixed regimes should be more balanced
        assert!(
            imbalance_ratio < 8.0,
            "Mixed volatility should be more balanced: {:.2}",
            imbalance_ratio
        );
        assert!(
            min_pct > 0.05,
            "Minimum class percentage too low in mixed regimes: {:.3}",
            min_pct
        );
    }

    #[test]
    fn test_volatility_balance_different_sensitivities() {
        let sequence_length = 50;
        let horizon_length = 10;

        // Generate mixed volatility for sensitivity testing
        let market_data = VolatilityScenarioGenerator::generate_mixed_volatility(
            50000.0,
            sequence_length + horizon_length + 80,
            15, // Regime length
        );

        let sensitivities = vec![("Low", 0.2), ("Default", 0.4), ("High", 0.8)];

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

                if let Ok(target) =
                    classify_volatility_with_distribution_analysis(sequence, horizon, &config)
                {
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
            if sensitivity < 0.4 {
                // Lower sensitivity = more extreme volatility classifications
                let extreme_pct =
                    distribution.get(&0).unwrap_or(&0.0) + distribution.get(&4).unwrap_or(&0.0);
                assert!(
                    extreme_pct > 0.15,
                    "Low sensitivity should create more extreme classes: {:.3}",
                    extreme_pct
                );
            } else if sensitivity > 0.4 {
                // Higher sensitivity = more medium classifications
                let medium_pct = distribution.get(&2).unwrap_or(&0.0);
                assert!(
                    *medium_pct > 0.25,
                    "High sensitivity should create more medium classes: {:.3}",
                    medium_pct
                );
            }
        }
    }

    #[test]
    fn test_volatility_balance_comprehensive_analysis() {
        println!("\n🎯 COMPREHENSIVE VOLATILITY BALANCE ANALYSIS");
        println!("============================================");

        let config = TargetsConfig::default();
        let sequence_length = 50;
        let horizon_length = 10;

        let scenarios = vec![
            (
                "Low Volatility",
                VolatilityScenarioGenerator::generate_low_volatility(50000.0, 100, 1.0),
            ),
            (
                "High Volatility",
                VolatilityScenarioGenerator::generate_high_volatility(50000.0, 100, 5.0),
            ),
            (
                "Volatility Expansion",
                VolatilityScenarioGenerator::generate_volatility_expansion(50000.0, 100, 3.0),
            ),
            (
                "Volatility Contraction",
                VolatilityScenarioGenerator::generate_volatility_contraction(50000.0, 100, 4.0),
            ),
            (
                "Mixed Regimes",
                VolatilityScenarioGenerator::generate_mixed_volatility(50000.0, 120, 20),
            ),
        ];

        let mut overall_stats = Vec::new();

        for (scenario_name, market_data) in scenarios {
            let mut targets = Vec::new();

            for i in 0..(market_data.len() - sequence_length - horizon_length) {
                let sequence = &market_data[i..i + sequence_length];
                let horizon =
                    &market_data[i + sequence_length..i + sequence_length + horizon_length];

                if let Ok(target) =
                    classify_volatility_with_distribution_analysis(sequence, horizon, &config)
                {
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

        println!("\n🎯 OVERALL VOLATILITY BALANCE SUMMARY:");
        println!("   Average Imbalance Ratio: {:.2}x", avg_imbalance);
        println!("   Average Deviation from 20%: {:.3}", avg_deviation);
        println!("   Average Minimum Class %: {:.3}", avg_min_pct);

        // Balance quality assessment
        if avg_imbalance > 10.0 {
            println!("⚠️  HIGH IMBALANCE DETECTED - Volatility classification too extreme");
        }
        if avg_deviation > 0.15 {
            println!("⚠️  HIGH DEVIATION DETECTED - Volatility classes not evenly distributed");
        }
        if avg_min_pct < 0.05 {
            println!("⚠️  LOW MINIMUM CLASS DETECTED - Some volatility classes rarely occur");
        }

        // Recommendations
        println!("\n💡 RECOMMENDATIONS:");
        if avg_imbalance > 8.0 {
            println!("   - Implement volatility-aware threshold adjustment");
            println!("   - Consider ATR distribution scaling for balanced classification");
        }
        if avg_deviation > 0.12 {
            println!("   - Add volatility regime context detection");
            println!("   - Implement adaptive log-ratio thresholds");
        }
    }
}