use super::balance::*;
use super::diversity::*;
use crate::targets::TargetType;
use ndarray::Array2;

#[test]
fn test_balance_with_validation_gap_maintains_perfect_balance() {
    let mut sequences = Vec::new();

    for class in 0..5 {
        for i in 0..600 {
            let seq_idx = class * 600 + i;
            let start_idx = seq_idx * 10;
            let end_idx = start_idx + 60;

            let sequence_data = Array2::zeros((60, 50));

            let target_data = TargetData {
                target_type: TargetType::PriceLevel,
                horizon: "30m".to_string(),
                class: class as i32,
                strength: 0.5,
            };

            sequences.push(SequenceWithTargets {
                sequence_idx: seq_idx,
                start_idx,
                end_idx,
                sequence_data,
                targets: vec![target_data],
            });
        }
    }

    let balance_config = BalanceConfig {
        max_overlap: 0.3,
        prefer_non_overlapping: true,
        min_sequences_per_class: 10,
    };
    let balancer = SequenceBalancer::new(balance_config);

    let result =
        balancer.balance_sequences_for_window(&sequences, TargetType::PriceLevel, "30m", &[], None);

    assert!(result.is_ok(), "Balance should succeed");
    let balanced = result.unwrap();

    let mut class_counts = [0usize; 5];
    for &idx in &balanced.selected_indices {
        let seq = &sequences[idx];
        if let Some(class) = seq.get_target_class(TargetType::PriceLevel, "30m") {
            class_counts[class as usize] += 1;
        }
    }

    let min_count = *class_counts.iter().min().unwrap();
    let max_count = *class_counts.iter().max().unwrap();

    println!("Class distribution: {:?}", class_counts);
    println!(
        "Min: {}, Max: {}, Diff: {}",
        min_count,
        max_count,
        max_count - min_count
    );

    assert_eq!(
        min_count, max_count,
        "PERFECT BALANCE REQUIRED: All classes must have exactly {} samples, but got {:?}",
        min_count, class_counts
    );

    assert_eq!(
        min_count, balanced.sequences_per_class,
        "Each class should have exactly {} samples",
        balanced.sequences_per_class
    );
}

#[test]
fn test_diversity_selector_returns_exact_count() {
    let mut sequences = Vec::new();

    for i in 0..1000 {
        let start_idx = i * 10;
        let end_idx = start_idx + 60;

        let sequence_data = Array2::zeros((60, 50));

        let target_data = TargetData {
            target_type: TargetType::PriceLevel,
            horizon: "30m".to_string(),
            class: 0,
            strength: 0.5,
        };

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx,
            end_idx,
            sequence_data,
            targets: vec![target_data],
        });
    }

    let diversity_config = DiversityConfig::default();
    let selector = DiversitySelector::new(diversity_config);

    let indices: Vec<usize> = (0..1000).collect();
    let target_count = 545;

    let result = selector.select_diverse_sequences(
        &sequences,
        &indices,
        target_count,
        TargetType::PriceLevel,
        "30m",
        &[],
        0,
    );

    assert!(result.is_ok(), "Selection should succeed");
    let selected = result.unwrap();

    assert_eq!(
        selected.len(),
        target_count,
        "Diversity selector MUST return exactly {} sequences when gap=0, but returned {}",
        target_count,
        selected.len()
    );
}

#[test]
fn test_validation_gap_does_not_affect_training_balance() {
    use std::collections::HashSet;

    let mut sequences = Vec::new();

    for class in 0..5 {
        for i in 0..600 {
            let seq_idx = class * 600 + i;
            // Create sequences with realistic spacing for gap=10
            // Each sequence is 30 long, spaced 20 apart (overlap=10, gap=20 between non-overlapping)
            let start_idx = seq_idx * 20;
            let end_idx = start_idx + 30;

            let sequence_data = Array2::zeros((30, 50));

            let target_data = TargetData {
                target_type: TargetType::PriceLevel,
                horizon: "30m".to_string(),
                class: class as i32,
                strength: 0.5 + (class as f64 * 0.1),
            };

            sequences.push(SequenceWithTargets {
                sequence_idx: seq_idx,
                start_idx,
                end_idx,
                sequence_data,
                targets: vec![target_data],
            });
        }
    }

    let balance_config = BalanceConfig {
        max_overlap: 0.3,
        prefer_non_overlapping: true,
        min_sequences_per_class: 10,
    };
    let balancer = SequenceBalancer::new(balance_config);

    let class_indices: Vec<usize> = (0..600).collect();

    let result = balancer.create_diverse_class_splits(&sequences, &class_indices, 480, 60, 60, 10);

    if let Err(e) = &result {
        eprintln!("ERROR: {:?}", e);
    }
    assert!(result.is_ok(), "Split should succeed");
    let (train_indices, val_indices, test_indices) = result.unwrap();
    let train: Vec<usize> = train_indices;
    let val: Vec<usize> = val_indices;
    let test: Vec<usize> = test_indices;

    assert_eq!(train.len(), 480, "TRAIN must have exactly 480 samples");
    assert_eq!(val.len(), 60, "VAL must have exactly 60 samples");
    assert_eq!(test.len(), 60, "TEST must have exactly 60 samples");

    // Verify no overlap
    let train_set: HashSet<usize> = train.iter().cloned().collect();
    let val_overlap: Vec<_> = val.iter().filter(|&&i| train_set.contains(&i)).collect();
    assert!(
        val_overlap.is_empty(),
        "VAL and TRAIN should not overlap, but found {}",
        val_overlap.len()
    );

    let test_overlap: Vec<_> = test.iter().filter(|&&i| train_set.contains(&i)).collect();
    assert!(
        test_overlap.is_empty(),
        "TEST and TRAIN should not overlap, but found {}",
        test_overlap.len()
    );
}
