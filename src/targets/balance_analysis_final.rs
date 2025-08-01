//! CORRECTED Balance Analysis - Understanding Current Logic

#[cfg(test)]
mod corrected_analysis {
    use crate::config::model::TargetsConfig;
    use crate::targets::direction::classify_direction;
    use crate::targets::price_levels::classify_price_level;
    use crate::targets::synthetic_data_generators::SyntheticMarketGenerator;
    use crate::targets::volatility::classify_volatility_with_distribution_analysis;
    use std::collections::HashMap;

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
    fn test_all_targets_balance_analysis() {
        println!("\n🎯 COMPREHENSIVE ALL TARGETS BALANCE ANALYSIS");
        println!("=============================================");

        let config = TargetsConfig::default();
        let sequence_length = 60;
        let horizon_length = 10;

        let mut price_targets = Vec::new();
        let mut direction_targets = Vec::new();
        let mut volatility_targets = Vec::new();

        // Test 100 independent sequences
        for i in 0..100 {
            let mut generator = SyntheticMarketGenerator::new(33333 + i as u64);
            let market_data = generator.generate_realistic_crypto_market(
                1000.0 + (i as f64 * 5.0),
                sequence_length + horizon_length,
                1000.0,
            );

            let sequence = &market_data[0..sequence_length];
            let horizon = &market_data[sequence_length..sequence_length + horizon_length];

            // Test Price Levels
            if let Ok(target) = classify_price_level(sequence, horizon, &config) {
                price_targets.push(target);
            }

            // Test Direction
            let sequence_prices: Vec<f64> = sequence.iter().map(|row| row.close).collect();
            let horizon_prices: Vec<f64> = horizon.iter().map(|row| row.close).collect();
            if let Ok(target) = classify_direction(&sequence_prices, &horizon_prices, &config) {
                direction_targets.push(target);
            }

            // Test Volatility
            if let Ok(target) =
                classify_volatility_with_distribution_analysis(sequence, horizon, &config)
            {
                volatility_targets.push(target);
            }
        }

        // Analyze each target type
        let targets = vec![
            ("PRICE LEVELS", price_targets),
            ("DIRECTION", direction_targets),
            ("VOLATILITY", volatility_targets),
        ];

        for (name, target_vec) in targets {
            let distribution = calculate_class_distribution(&target_vec);
            let (imbalance_ratio, deviation, min_pct) = calculate_balance_metrics(&distribution);

            println!("\n📊 {} ANALYSIS:", name);
            println!("   Distribution: {:?}", distribution);
            println!(
                "   Imbalance Ratio: {:.2}x",
                if imbalance_ratio.is_infinite() {
                    f64::MAX
                } else {
                    imbalance_ratio
                }
            );
            println!("   Average Deviation: {:.3}", deviation);
            println!("   Minimum Class %: {:.3}", min_pct);
            println!("   Total Samples: {}", target_vec.len());

            // Count missing classes
            let missing_classes = (0..5).filter(|&i| !distribution.contains_key(&i)).count();
            println!("   Missing Classes: {}/5", missing_classes);

            if missing_classes > 0 {
                let missing: Vec<i32> =
                    (0..5).filter(|&i| !distribution.contains_key(&i)).collect();
                println!("   Missing Class IDs: {:?}", missing);
            }
        }

        println!("\n🔍 ROOT CAUSE ANALYSIS:");
        println!("======================================");
        println!("1. PRICE LEVELS: Uses FIXED percentiles [0.1, 0.9] = 80% neutral zone");
        println!("2. DIRECTION: Uses trend_consistency * base_sensitivity with min thresholds");
        println!("3. VOLATILITY: Uses base_sensitivity / 2.0 for log-ratio thresholds");
        println!();
        println!("🚨 COMMON ISSUE: All targets use FIXED threshold ratios!");
        println!("   - Price: Fixed 10th-90th percentiles");
        println!("   - Direction: Fixed multipliers (20.0, min 0.01/0.03)");
        println!("   - Volatility: Fixed bandwidth divisor (/ 2.0)");
        println!();
        println!("💡 SOLUTION: Make threshold ratios SEQUENCE-ADAPTIVE");
        println!("   - Calculate sequence characteristics ONCE");
        println!("   - Adapt threshold ratios based on sequence properties");
        println!("   - NO magic numbers - all derived from sequence data");
    }
}
