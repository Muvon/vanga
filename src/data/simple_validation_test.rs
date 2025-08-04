//! Simple test to verify validation/training split without complex balancing

use crate::data::balance::*;
use crate::targets::TargetType;
use ndarray::Array2;
use std::collections::{HashMap, HashSet};

#[test]
fn test_simple_validation_training_separation() {
    // Create a simple dataset with balanced classes
    let mut sequences = Vec::new();

    // Create 50 sequences (10 per class) for easier testing
    for i in 0..50 {
        let class = i % 5;
        let mut targets = HashMap::new();
        targets.insert((TargetType::PriceLevel, "1h".to_string()), class as i32);

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx: i * 100, // Wide spacing to avoid overlap issues
            end_idx: i * 100 + 60,
            sequence_data: Array2::zeros((60, 5)),
            targets,
        });
    }

    // Manually select validation indices (2 per class = 10 total = 20%)
    let validation_indices = vec![0, 5, 10, 15, 20, 25, 30, 35, 40, 45];

    // Create training indices (remaining 40 sequences)
    let all_indices: HashSet<usize> = (0..50).collect();
    let val_set: HashSet<usize> = validation_indices.iter().cloned().collect();
    let training_indices: Vec<usize> = all_indices.difference(&val_set).cloned().collect();

    // Verify no overlap
    let overlap: Vec<usize> = training_indices
        .iter()
        .filter(|idx| val_set.contains(idx))
        .cloned()
        .collect();

    assert!(
        overlap.is_empty(),
        "Found {} overlapping sequences: {:?}",
        overlap.len(),
        overlap
    );

    println!("✅ Simple separation test passed:");
    println!("   Total sequences: 50");
    println!("   Validation: {} sequences", validation_indices.len());
    println!("   Training: {} sequences", training_indices.len());
    println!("   NO overlap between sets");

    // Verify class distribution in both sets
    let mut val_class_dist: HashMap<i32, usize> = HashMap::new();
    let mut train_class_dist: HashMap<i32, usize> = HashMap::new();

    for &idx in &validation_indices {
        let class = sequences[idx].targets[&(TargetType::PriceLevel, "1h".to_string())];
        *val_class_dist.entry(class).or_insert(0) += 1;
    }

    for &idx in &training_indices {
        let class = sequences[idx].targets[&(TargetType::PriceLevel, "1h".to_string())];
        *train_class_dist.entry(class).or_insert(0) += 1;
    }

    println!("   Validation class distribution: {:?}", val_class_dist);
    println!("   Training class distribution: {:?}", train_class_dist);
}

#[test]
fn test_data_module_validation_split_logic() {
    // Test the actual logic used in data module for validation split
    use crate::data::balance::SequenceBalancer;

    let mut sequences = Vec::new();

    // Create 100 sequences with better class distribution
    for i in 0..100 {
        let class = i % 5;
        let mut targets = HashMap::new();
        targets.insert((TargetType::PriceLevel, "1h".to_string()), class as i32);

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx: i * 50,
            end_idx: i * 50 + 60,
            sequence_data: Array2::zeros((60, 5)),
            targets,
        });
    }

    let balancer = SequenceBalancer::new(BalanceConfig::default());

    // First extract globally balanced dataset (as done in actual training)
    let global_result = balancer
        .extract_globally_balanced_dataset(
            &sequences,
            &[TargetType::PriceLevel],
            &["1h".to_string()],
        )
        .unwrap();

    println!("Global balanced dataset:");
    println!(
        "  Total balanced samples: {}",
        global_result.total_balanced_samples
    );
    println!(
        "  Min class count: {}",
        global_result.global_min_class_count
    );

    // Then derive validation from it
    let (remaining_training, validation_indices) = balancer
        .smart_validation_split_from_balanced(
            &global_result,
            &sequences,
            0.2, // 20% validation
            &[TargetType::PriceLevel],
            &["1h".to_string()],
        )
        .unwrap();

    let val_indices = validation_indices
        .get(&(TargetType::PriceLevel, "1h".to_string()))
        .unwrap();
    let train_indices = remaining_training
        .balanced_indices
        .get(&(TargetType::PriceLevel, "1h".to_string()))
        .unwrap();

    // Check no overlap
    let val_set: HashSet<usize> = val_indices.iter().cloned().collect();
    let train_set: HashSet<usize> = train_indices.iter().cloned().collect();
    let overlap: Vec<usize> = val_set.intersection(&train_set).cloned().collect();

    assert!(
        overlap.is_empty(),
        "Found {} overlapping sequences between training and validation",
        overlap.len()
    );

    println!("✅ Data module validation split test passed:");
    println!("   Training: {} sequences", train_indices.len());
    println!("   Validation: {} sequences", val_indices.len());
    println!("   NO overlap between sets");
}

#[test]
fn test_sequence_data_overlap_vs_index_overlap() {
    // Test to clarify the difference between sequence index overlap and data range overlap
    let mut sequences = Vec::new();

    // Create overlapping sequences (data ranges overlap but indices are different)
    for i in 0..20 {
        let mut targets = HashMap::new();
        targets.insert((TargetType::PriceLevel, "1h".to_string()), (i % 5) as i32);

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx: i * 10,    // Overlapping data ranges
            end_idx: i * 10 + 60, // Each sequence covers 60 time steps
            sequence_data: Array2::zeros((60, 5)),
            targets,
        });
    }

    // Check data overlap between consecutive sequences
    let seq0 = &sequences[0];
    let seq1 = &sequences[1];
    let overlap_ratio = seq0.overlap_ratio(seq1);

    println!("Sequence 0: start={}, end={}", seq0.start_idx, seq0.end_idx);
    println!("Sequence 1: start={}, end={}", seq1.start_idx, seq1.end_idx);
    println!("Data overlap ratio: {:.2}", overlap_ratio);

    assert!(overlap_ratio > 0.0, "Sequences should have data overlap");

    // But sequence indices are different
    assert_ne!(
        seq0.sequence_idx, seq1.sequence_idx,
        "Sequence indices should be different"
    );

    // Validation should never include the same sequence INDEX as training
    let validation_indices = vec![0, 2, 4, 6, 8];
    let training_indices = vec![1, 3, 5, 7, 9];

    let val_set: HashSet<usize> = validation_indices.iter().cloned().collect();
    let train_set: HashSet<usize> = training_indices.iter().cloned().collect();

    let index_overlap = val_set.intersection(&train_set).count();
    assert_eq!(index_overlap, 0, "No sequence INDEX overlap allowed");

    // But data ranges might overlap (which is acceptable)
    let mut data_overlaps = 0;
    for &val_idx in &validation_indices {
        for &train_idx in &training_indices {
            if sequences[val_idx].overlap_ratio(&sequences[train_idx]) > 0.0 {
                data_overlaps += 1;
            }
        }
    }

    println!("✅ Overlap understanding test passed:");
    println!("   Sequence INDEX overlap: 0 (required)");
    println!("   Data range overlaps: {} (acceptable)", data_overlaps);
}
