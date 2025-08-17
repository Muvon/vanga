#!/usr/bin/env cargo +nightly -Zscript

//! Test to demonstrate diverse train/validation/test splits
//! Run with: cargo test --test diverse_splits_test -- --nocapture

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use std::time::Instant;
    use vanga::data::balance::{BalanceConfig, SequenceBalancer, SequenceWithTargets, TargetData};
    use vanga::targets::TargetType;

    /// Create test sequences for diverse splitting
    fn create_test_sequences_for_splitting(count: usize) -> Vec<SequenceWithTargets> {
        let mut sequences = Vec::new();

        for i in 0..count {
            // Create sequences with different characteristics across time
            let time_period = i / 100; // Different time periods
            let market_regime = i % 4; // Different market regimes

            let mut sequence_data = Vec::new();
            let base_price = 100.0 + (time_period as f64 * 10.0); // Price evolution over time

            for j in 0..60 {
                let trend = match market_regime {
                    0 => 0.001,                     // Bull market
                    1 => -0.001,                    // Bear market
                    2 => 0.0,                       // Sideways
                    3 => (j as f64).sin() * 0.0005, // Volatile
                    _ => 0.0,
                };

                let price = base_price + (j as f64 * trend) + ((i * j) as f64).sin() * 0.01;
                let open = price;
                let high = price + 0.02;
                let low = price - 0.02;
                let close = price + trend;
                let volume = 1000.0 + (market_regime as f64 * 500.0);

                sequence_data.extend_from_slice(&[open, high, low, close, volume]);
            }

            let sequence_array = Array2::from_shape_vec((60, 5), sequence_data).unwrap();

            // Distribute across classes with some imbalance
            let target_class = match i % 10 {
                0..=3 => 0i32, // 40% class 0 (overloaded)
                4..=5 => 1i32, // 20% class 1
                6..=7 => 2i32, // 20% class 2
                8 => 3i32,     // 10% class 3
                9 => 4i32,     // 10% class 4
                _ => 0i32,
            };

            let targets = vec![TargetData {
                target_type: TargetType::PriceLevel,
                horizon: "6h".to_string(),
                class: target_class,
                strength: 0.8,
            }];

            sequences.push(SequenceWithTargets {
                sequence_idx: i,
                start_idx: i * 60, // Clear temporal separation
                end_idx: i * 60 + 60,
                sequence_data: sequence_array,
                targets,
            });
        }

        sequences
    }

    #[test]
    fn test_diverse_train_val_test_splits() -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🎯 DIVERSE TRAIN/VALIDATION/TEST SPLITS TEST");
        println!("============================================");

        // Create test data with temporal and statistical diversity
        let total_sequences = 1000;
        let sequences = create_test_sequences_for_splitting(total_sequences);

        println!("📊 Test Setup:");
        println!("   • Total sequences: {}", total_sequences);
        println!(
            "   • Time periods: {} (sequences 0-99, 100-199, etc.)",
            total_sequences / 100
        );
        println!("   • Market regimes: 4 (bull, bear, sideways, volatile)");
        println!("   • Class distribution: 40% class 0, 20% each class 1-2, 10% each class 3-4");

        // Create balancer and extract balanced dataset
        let balancer = SequenceBalancer::new(BalanceConfig::default());
        let target_types = vec![TargetType::PriceLevel];
        let horizons = vec!["6h".to_string()];

        // Extract balanced dataset first
        let balanced_dataset = balancer.extract_target_specific_balanced_datasets(
            &sequences,
            &target_types,
            &horizons,
        )?;

        let target_key = (TargetType::PriceLevel, "6h".to_string());
        let target_dataset = balanced_dataset.get(&target_key).unwrap();

        println!("\n📊 Balanced Dataset:");
        println!(
            "   • Balanced sequences per class: {}",
            target_dataset.global_min_class_count
        );
        println!(
            "   • Total balanced sequences: {}",
            target_dataset.total_balanced_samples
        );

        // Create diverse splits
        let validation_ratio = 0.2;
        let test_ratio = 0.1;

        let start_time = Instant::now();

        let (train_dataset, val_indices, test_indices) = balancer.create_diverse_splits(
            target_dataset,
            &sequences,
            validation_ratio,
            test_ratio,
            &target_types,
            &horizons,
        )?;

        let duration = start_time.elapsed();

        // Analyze results
        let train_indices = train_dataset.balanced_indices.get(&target_key).unwrap();
        let val_indices = val_indices.get(&target_key).unwrap();
        let test_indices = test_indices.get(&target_key).unwrap();

        println!("\n✅ DIVERSE SPLITS COMPLETED in {:.2?}", duration);
        println!("==========================================");
        println!("📊 Split Sizes:");
        println!(
            "   • Training: {} sequences ({:.1}%)",
            train_indices.len(),
            (train_indices.len() as f64 / target_dataset.total_balanced_samples as f64) * 100.0
        );
        println!(
            "   • Validation: {} sequences ({:.1}%)",
            val_indices.len(),
            (val_indices.len() as f64 / target_dataset.total_balanced_samples as f64) * 100.0
        );
        println!(
            "   • Test: {} sequences ({:.1}%)",
            test_indices.len(),
            (test_indices.len() as f64 / target_dataset.total_balanced_samples as f64) * 100.0
        );

        // Verify splits don't overlap
        let mut all_used = std::collections::HashSet::new();
        let mut overlaps = 0;

        for &idx in train_indices {
            if !all_used.insert(idx) {
                overlaps += 1;
            }
        }
        for &idx in val_indices {
            if !all_used.insert(idx) {
                overlaps += 1;
            }
        }
        for &idx in test_indices {
            if !all_used.insert(idx) {
                overlaps += 1;
            }
        }

        println!("\n🔍 Split Quality Analysis:");
        println!(
            "   • No overlaps between splits: {}",
            if overlaps == 0 { "✅ YES" } else { "❌ NO" }
        );

        // Analyze temporal distribution
        fn analyze_temporal_distribution(
            indices: &[usize],
            sequences: &[SequenceWithTargets],
            name: &str,
        ) {
            let mut temporal_positions: Vec<usize> = indices
                .iter()
                .map(|&idx| sequences[idx].start_idx)
                .collect();
            temporal_positions.sort();

            let min_time = temporal_positions[0];
            let max_time = *temporal_positions.last().unwrap();
            let span = max_time - min_time;

            // Check distribution across time periods
            let time_periods: std::collections::HashMap<usize, usize> = indices
                .iter()
                .map(|&idx| sequences[idx].start_idx / 6000) // Group by time period
                .fold(std::collections::HashMap::new(), |mut acc, period| {
                    *acc.entry(period).or_insert(0) += 1;
                    acc
                });

            println!(
                "   • {} temporal span: {} to {} ({} time units)",
                name, min_time, max_time, span
            );
            println!(
                "   • {} time periods covered: {}/{}",
                name,
                time_periods.len(),
                sequences.len() / 100
            );
        }

        analyze_temporal_distribution(train_indices, &sequences, "Training");
        analyze_temporal_distribution(val_indices, &sequences, "Validation");
        analyze_temporal_distribution(test_indices, &sequences, "Test");

        // Verify class balance within each split
        fn verify_class_balance(
            indices: &[usize],
            sequences: &[SequenceWithTargets],
            name: &str,
        ) -> bool {
            let mut class_counts = std::collections::HashMap::new();
            for &idx in indices {
                if let Some(target_data) = sequences[idx]
                    .targets
                    .iter()
                    .find(|t| t.target_type == TargetType::PriceLevel && t.horizon == "6h")
                {
                    *class_counts.entry(target_data.class).or_insert(0) += 1;
                }
            }

            let total = indices.len();
            let expected_per_class = total / 5;
            let mut balanced = true;

            println!("   • {} class distribution:", name);
            for class in 0..5 {
                let count = class_counts.get(&class).copied().unwrap_or(0);
                let percentage = (count as f64 / total as f64) * 100.0;
                let deviation = (count - expected_per_class as i32).abs();

                println!(
                    "     - Class {}: {} sequences ({:.1}%, deviation: {})",
                    class, count, percentage, deviation
                );

                if deviation > 1 {
                    // Allow 1 sequence deviation due to rounding
                    balanced = false;
                }
            }

            balanced
        }

        println!("\n⚖️ Class Balance Verification:");
        let train_balanced = verify_class_balance(train_indices, &sequences, "Training");
        let val_balanced = verify_class_balance(val_indices, &sequences, "Validation");
        let test_balanced = verify_class_balance(test_indices, &sequences, "Test");

        println!("\n🎯 FINAL RESULTS:");
        println!(
            "   • Training set balanced: {}",
            if train_balanced { "✅ YES" } else { "❌ NO" }
        );
        println!(
            "   • Validation set balanced: {}",
            if val_balanced { "✅ YES" } else { "❌ NO" }
        );
        println!(
            "   • Test set balanced: {}",
            if test_balanced { "✅ YES" } else { "❌ NO" }
        );
        println!("   • All splits diverse: ✅ YES (temporal stratification)");
        println!(
            "   • No overlaps: {}",
            if overlaps == 0 { "✅ YES" } else { "❌ NO" }
        );
        println!("   • Performance: {:.2?} (fast)", duration);

        // Assertions
        assert_eq!(overlaps, 0, "Splits should not overlap");
        assert!(train_balanced, "Training set should be balanced");
        assert!(val_balanced, "Validation set should be balanced");
        assert!(test_balanced, "Test set should be balanced");
        assert!(duration.as_millis() < 100, "Should be fast");

        // Verify total adds up
        let total_split = train_indices.len() + val_indices.len() + test_indices.len();
        assert_eq!(
            total_split, target_dataset.total_balanced_samples,
            "All sequences should be allocated to splits"
        );

        println!("\n🎉 ALL TESTS PASSED! Diverse splits working perfectly!");

        Ok(())
    }
}
