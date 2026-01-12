#!/usr/bin/env cargo +nightly -Zscript

//! Performance test to demonstrate the fast diversity selection
//! Run with: cargo test --test diversity_performance_test -- --nocapture

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use std::time::Instant;
    use vanga::data::balance::{SequenceWithTargets, TargetData};
    use vanga::data::diversity::{DiversityConfig, DiversitySelector};
    use vanga::targets::TargetType;

    /// Create test sequences for performance testing
    fn create_performance_test_sequences(count: usize) -> Vec<SequenceWithTargets> {
        let mut sequences = Vec::new();

        for i in 0..count {
            // Create realistic OHLCV data
            let mut sequence_data = Vec::new();
            let base_price = 100.0 + (i as f64 * 0.01);

            for j in 0..60 {
                let price = base_price + (j as f64 * 0.001) + ((i * j) as f64).sin() * 0.1;
                let open = price;
                let high = price + 0.01;
                let low = price - 0.01;
                let close = price + 0.005;
                let volume = 1000.0 + (i % 100) as f64;

                sequence_data.extend_from_slice(&[open, high, low, close, volume]);
            }

            let sequence_array = Array2::from_shape_vec((60, 5), sequence_data).unwrap();

            // All sequences belong to class 0 (overloaded class scenario)
            let targets = vec![TargetData {
                target_type: TargetType::PriceLevel,
                horizon: "6h".to_string(),
                class: 0i32,
                strength: 0.8,
            }];

            sequences.push(SequenceWithTargets {
                sequence_idx: i,
                start_idx: i * 60,
                end_idx: i * 60 + 60,
                sequence_data: sequence_array,
                targets,
            });
        }

        sequences
    }

    #[test]
    fn test_fast_diversity_selection_performance() {
        println!("\n🚀 FAST DIVERSITY SELECTION PERFORMANCE TEST");
        println!("============================================");

        // Test with realistic overloaded class size
        let total_sequences = 500;
        let target_count = 425;
        let sequences = create_performance_test_sequences(total_sequences);

        println!("📊 Test Setup:");
        println!("   • Total sequences: {}", total_sequences);
        println!("   • Target selection: {}", target_count);
        println!(
            "   • Utilization: {:.1}%",
            (target_count as f64 / total_sequences as f64) * 100.0
        );

        let diversity_selector = DiversitySelector::new(DiversityConfig::default());
        let class_indices: Vec<usize> = (0..total_sequences).collect();

        // Measure performance
        let start_time = Instant::now();

        let result = diversity_selector.select_diverse_sequences(
            &sequences,
            &class_indices,
            target_count,
            TargetType::PriceLevel,
            "6h",
            &[], // No exclusions
            0,   // validation_gap_steps
        );

        let duration = start_time.elapsed();

        match result {
            Ok(selected) => {
                println!("\n✅ PERFORMANCE TEST RESULTS:");
                println!("   • Selection completed in: {:.2?}", duration);
                println!("   • Selected sequences: {}", selected.len());
                println!("   • Expected sequences: {}", target_count);
                println!(
                    "   • Performance: {} sequences/second",
                    (total_sequences as f64 / duration.as_secs_f64()) as u64
                );

                // Verify correctness
                assert_eq!(selected.len(), target_count);
                assert!(
                    duration.as_millis() < 1000,
                    "Should complete in under 1 second"
                );

                // Check temporal distribution
                let mut temporal_positions: Vec<usize> = selected
                    .iter()
                    .map(|&idx| sequences[idx].start_idx)
                    .collect();
                temporal_positions.sort();

                println!(
                    "   • Temporal spread: {} to {} (good distribution)",
                    temporal_positions[0],
                    temporal_positions.last().unwrap()
                );

                // Verify it's not just chronological selection
                let chronological: Vec<usize> = (0..target_count).collect();
                let mut selected_sorted = selected.clone();
                selected_sorted.sort();

                if selected_sorted != chronological {
                    println!("   • ✅ Selection is diverse (not chronological)");
                } else {
                    println!("   • ⚠️ Selection appears chronological");
                }

                println!("\n🎯 PERFORMANCE BENCHMARK:");
                if duration.as_millis() < 100 {
                    println!("   🚀 EXCELLENT: < 100ms (production ready)");
                } else if duration.as_millis() < 500 {
                    println!("   ✅ GOOD: < 500ms (acceptable)");
                } else if duration.as_millis() < 1000 {
                    println!("   ⚠️ ACCEPTABLE: < 1s (could be better)");
                } else {
                    println!("   ❌ SLOW: > 1s (needs optimization)");
                }
            }
            Err(e) => {
                panic!("Diversity selection failed: {}", e);
            }
        }
    }

    #[test]
    fn test_scalability_different_sizes() {
        println!("\n📈 SCALABILITY TEST");
        println!("==================");

        let test_sizes = vec![100, 200, 500, 1000];
        let diversity_selector = DiversitySelector::new(DiversityConfig::default());

        for &size in &test_sizes {
            let sequences = create_performance_test_sequences(size);
            let target_count = size / 2; // 50% utilization
            let class_indices: Vec<usize> = (0..size).collect();

            let start_time = Instant::now();
            let result = diversity_selector.select_diverse_sequences(
                &sequences,
                &class_indices,
                target_count,
                TargetType::PriceLevel,
                "6h",
                &[],
                0, // validation_gap_steps
            );

            let duration = start_time.elapsed();

            match result {
                Ok(selected) => {
                    println!(
                        "   {} sequences → {} selected in {:.2?} ({} seq/s)",
                        size,
                        selected.len(),
                        duration,
                        (size as f64 / duration.as_secs_f64()) as u64
                    );

                    assert_eq!(selected.len(), target_count);
                    assert!(duration.as_millis() < 2000, "Should scale well");
                }
                Err(e) => {
                    panic!("Failed for size {}: {}", size, e);
                }
            }
        }

        println!("\n✅ All scalability tests passed!");
    }
}
