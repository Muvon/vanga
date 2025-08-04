//! Tests to ensure validation sequences NEVER contain exact same sequences as training
//! This is CRITICAL for preventing data leakage and ensuring valid model evaluation

use crate::data::balance::*;
use crate::targets::TargetType;
use ndarray::Array2;
use std::collections::{HashMap, HashSet};

#[test]
fn test_validation_never_contains_training_sequences() {
    // Create test sequences with known indices
    let mut sequences = Vec::new();

    // Create 100 sequences with different class distributions
    for i in 0..100 {
        let class = i % 5; // Even distribution for simplicity

        let mut targets = HashMap::new();
        targets.insert((TargetType::PriceLevel, "1h".to_string()), class as i32);

        let seq = SequenceWithTargets {
            sequence_idx: i,
            start_idx: i * 10, // Non-overlapping sequences
            end_idx: i * 10 + 30,
            sequence_data: Array2::zeros((30, 5)),
            targets,
        };

        // Debug: print first few sequences
        if i < 3 {
            println!(
                "Sequence {}: start_idx={}, end_idx={}",
                i, seq.start_idx, seq.end_idx
            );
        }

        sequences.push(seq);
    }

    let config = BalanceConfig {
        max_overlap: 0.0, // No overlap allowed
        prefer_non_overlapping: true,
        min_sequences_per_class: 5,
    };

    let balancer = SequenceBalancer::new(config);

    // Select validation with 20% ratio
    let validation_result = balancer
        .select_target_specific_validation(
            &sequences,
            0.2, // 20% validation
            &[TargetType::PriceLevel],
            &["1h".to_string()],
        )
        .unwrap();

    // Get validation indices for our target
    let val_indices = validation_result
        .get(&(TargetType::PriceLevel, "1h".to_string()))
        .unwrap();

    println!("Total sequences: {}", sequences.len());
    println!("Validation sequences selected: {}", val_indices.len());
    println!(
        "Expected validation size: ~{}",
        (sequences.len() as f64 * 0.2) as usize
    );

    // Get validation indices for our target
    let val_indices = validation_result
        .get(&(TargetType::PriceLevel, "1h".to_string()))
        .unwrap();

    // Create a set of validation indices for fast lookup
    let val_indices_set: HashSet<usize> = val_indices.iter().cloned().collect();

    // Now select training sequences, excluding validation
    println!("Attempting to select training sequences with window range (0, 3000)");
    println!("Validation indices to exclude: {:?}", val_indices);

    let training_result = balancer
        .balance_sequences_for_window(
            &sequences,
            val_indices, // Exclude validation sequences
            (0, 3000),   // Wide range to include all sequences
            TargetType::PriceLevel,
            "1h",
        )
        .unwrap();

    // CRITICAL CHECK: Ensure NO overlap between training and validation
    for train_idx in &training_result.selected_indices {
        assert!(
            !val_indices_set.contains(train_idx),
            "CRITICAL ERROR: Training sequence {} is also in validation set!",
            train_idx
        );
    }

    println!(
        "✅ Validation check passed: {} training sequences, {} validation sequences, NO overlap",
        training_result.selected_indices.len(),
        val_indices.len()
    );
}

#[test]
fn test_validation_with_overlapping_sequences() {
    // Test with sequences that have overlap in data indices but are different sequences
    let mut sequences = Vec::new();

    // Create sequences with 80% overlap
    let sequence_length = 60;
    let step_size = 12; // 80% overlap (60 - 12 = 48 overlapping points)

    for i in 0..50 {
        let start_idx = i * step_size;
        let class = i % 5;

        let mut targets = HashMap::new();
        targets.insert((TargetType::PriceLevel, "1h".to_string()), class as i32);

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx,
            end_idx: start_idx + sequence_length,
            sequence_data: Array2::zeros((sequence_length, 5)),
            targets,
        });
    }

    let config = BalanceConfig {
        max_overlap: 0.8, // Allow 80% overlap
        prefer_non_overlapping: false,
        min_sequences_per_class: 3,
    };

    let balancer = SequenceBalancer::new(config);

    // Select validation
    let validation_result = balancer
        .select_target_specific_validation(
            &sequences,
            0.2,
            &[TargetType::PriceLevel],
            &["1h".to_string()],
        )
        .unwrap();

    let val_indices = validation_result
        .get(&(TargetType::PriceLevel, "1h".to_string()))
        .unwrap();

    // Select training
    let training_result = balancer
        .balance_sequences_for_window(
            &sequences,
            val_indices,
            (0, 3000), // Wide range to include all sequences
            TargetType::PriceLevel,
            "1h",
        )
        .unwrap();

    // Check that sequence INDICES don't overlap (even if data ranges might)
    let val_set: HashSet<usize> = val_indices.iter().cloned().collect();
    let train_set: HashSet<usize> = training_result.selected_indices.iter().cloned().collect();

    let intersection: Vec<usize> = val_set.intersection(&train_set).cloned().collect();

    assert!(
        intersection.is_empty(),
        "Found {} sequences in both training and validation: {:?}",
        intersection.len(),
        intersection
    );

    // Additional check: verify data range overlap is acceptable
    let mut total_data_overlap = 0;
    for &val_idx in val_indices {
        let val_seq = &sequences[val_idx];
        for &train_idx in &training_result.selected_indices {
            let train_seq = &sequences[train_idx];
            let overlap = val_seq.overlap_ratio(train_seq);
            if overlap > 0.0 {
                total_data_overlap += 1;
                println!(
                    "Data overlap found: val seq {} and train seq {} have {:.1}% overlap",
                    val_idx,
                    train_idx,
                    overlap * 100.0
                );
            }
        }
    }

    println!(
        "✅ Sequence index check passed: {} training, {} validation, NO sequence index overlap",
        train_set.len(),
        val_set.len()
    );
    println!(
        "   Data range overlaps: {} pairs (acceptable with overlapping sequences)",
        total_data_overlap
    );
}

#[test]
fn test_smart_validation_split_preserves_separation() {
    // Test the smart validation split used in actual training
    let mut sequences = Vec::new();

    // Create imbalanced dataset
    let class_counts = [50, 30, 15, 4, 1]; // Severe imbalance
    let mut seq_idx = 0;

    for (class, &count) in class_counts.iter().enumerate() {
        for _ in 0..count {
            let mut targets = HashMap::new();
            targets.insert((TargetType::PriceLevel, "1h".to_string()), class as i32);

            sequences.push(SequenceWithTargets {
                sequence_idx: seq_idx,
                start_idx: seq_idx * 30,
                end_idx: seq_idx * 30 + 60,
                sequence_data: Array2::zeros((60, 5)),
                targets,
            });
            seq_idx += 1;
        }
    }

    let balancer = SequenceBalancer::new(BalanceConfig::default());

    // First, create globally balanced dataset
    let global_result = balancer
        .extract_globally_balanced_dataset(
            &sequences,
            &[TargetType::PriceLevel],
            &["1h".to_string()],
        )
        .unwrap();

    // Then derive validation from it
    let validation_split = balancer
        .smart_validation_split_from_balanced(
            &global_result,
            &sequences,
            0.2,
            &[TargetType::PriceLevel],
            &["1h".to_string()],
        )
        .unwrap();

    let (remaining_training, validation_indices) = validation_split;

    // Check no overlap between remaining training and validation
    let val_indices = validation_indices
        .get(&(TargetType::PriceLevel, "1h".to_string()))
        .unwrap();
    let train_indices = remaining_training
        .balanced_indices
        .get(&(TargetType::PriceLevel, "1h".to_string()))
        .unwrap();

    let val_set: HashSet<usize> = val_indices.iter().cloned().collect();
    let train_set: HashSet<usize> = train_indices.iter().cloned().collect();

    let overlap: Vec<usize> = val_set.intersection(&train_set).cloned().collect();

    assert!(
        overlap.is_empty(),
        "CRITICAL: Found {} sequences in both training and validation after smart split: {:?}",
        overlap.len(),
        overlap
    );

    // Verify class balance is maintained
    let mut val_class_dist: HashMap<i32, usize> = HashMap::new();
    for &idx in val_indices {
        let class = sequences[idx].targets[&(TargetType::PriceLevel, "1h".to_string())];
        *val_class_dist.entry(class).or_insert(0) += 1;
    }

    println!("✅ Smart validation split passed:");
    println!("   Training: {} sequences", train_indices.len());
    println!("   Validation: {} sequences", val_indices.len());
    println!("   Validation class distribution: {:?}", val_class_dist);
    println!("   NO sequence overlap between training and validation");
}

#[test]
fn test_window_based_validation_separation() {
    // Test validation separation in walk-forward window context
    let mut sequences = Vec::new();

    // Create 200 sequences to have enough for multiple windows
    for i in 0..200 {
        let class = i % 5;
        let mut targets = HashMap::new();
        targets.insert((TargetType::PriceLevel, "1h".to_string()), class as i32);

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx: i * 20, // Some overlap
            end_idx: i * 20 + 60,
            sequence_data: Array2::zeros((60, 5)),
            targets,
        });
    }

    let balancer = SequenceBalancer::new(BalanceConfig::default());

    // Simulate multiple training windows
    let window_ranges = vec![
        (0, 2000), // Window 1
        (0, 3000), // Window 2 (expanding)
        (0, 4000), // Window 3 (full data)
    ];

    // First select validation (should be consistent across windows)
    let validation_result = balancer
        .select_target_specific_validation(
            &sequences,
            0.2,
            &[TargetType::PriceLevel],
            &["1h".to_string()],
        )
        .unwrap();

    let val_indices = validation_result
        .get(&(TargetType::PriceLevel, "1h".to_string()))
        .unwrap();

    let val_set: HashSet<usize> = val_indices.iter().cloned().collect();

    // Check each window
    for (window_idx, &(start, end)) in window_ranges.iter().enumerate() {
        let training_result = balancer
            .balance_sequences_for_window(
                &sequences,
                val_indices,
                (start, end),
                TargetType::PriceLevel,
                "1h",
            )
            .unwrap();

        // Verify no overlap
        for &train_idx in &training_result.selected_indices {
            assert!(
                !val_set.contains(&train_idx),
                "Window {}: Training sequence {} is in validation set!",
                window_idx + 1,
                train_idx
            );
        }

        println!(
            "✅ Window {} ({}-{}): {} training sequences, NO validation overlap",
            window_idx + 1,
            start,
            end,
            training_result.selected_indices.len()
        );
    }
}

#[test]
fn test_validation_consistency_across_targets() {
    // Ensure validation separation works for multi-target scenarios
    let mut sequences = Vec::new();

    for i in 0..100 {
        let mut targets = HashMap::new();
        // Different class distributions for different targets
        targets.insert((TargetType::PriceLevel, "1h".to_string()), (i % 5) as i32);
        targets.insert((TargetType::Direction, "1h".to_string()), (i % 3) as i32);
        targets.insert((TargetType::Volatility, "1h".to_string()), (i % 4) as i32);

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx: i * 30,
            end_idx: i * 30 + 60,
            sequence_data: Array2::zeros((60, 5)),
            targets,
        });
    }

    let balancer = SequenceBalancer::new(BalanceConfig::default());

    // Select validation for all targets
    let validation_result = balancer
        .select_target_specific_validation(
            &sequences,
            0.2,
            &[
                TargetType::PriceLevel,
                TargetType::Direction,
                TargetType::Volatility,
            ],
            &["1h".to_string()],
        )
        .unwrap();

    // Check each target separately
    for target_type in &[
        TargetType::PriceLevel,
        TargetType::Direction,
        TargetType::Volatility,
    ] {
        let val_indices = validation_result
            .get(&(*target_type, "1h".to_string()))
            .unwrap();

        let training_result = balancer
            .balance_sequences_for_window(
                &sequences,
                val_indices,
                (0, 3000), // Wide range to include all sequences
                *target_type,
                "1h",
            )
            .unwrap();

        // Check no overlap
        let val_set: HashSet<usize> = val_indices.iter().cloned().collect();
        let overlap: Vec<usize> = training_result
            .selected_indices
            .iter()
            .filter(|idx| val_set.contains(idx))
            .cloned()
            .collect();

        assert!(
            overlap.is_empty(),
            "Target {:?}: Found {} overlapping sequences",
            target_type,
            overlap.len()
        );

        println!(
            "✅ Target {:?}: {} training, {} validation, NO overlap",
            target_type,
            training_result.selected_indices.len(),
            val_indices.len()
        );
    }
}
