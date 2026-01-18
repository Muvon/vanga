use crate::data::balance::{
    BalanceConfig, GloballyBalancedDataset, SequenceBalancer, SequenceWithTargets, TargetData,
};
use crate::targets::TargetType;
use ndarray::Array2;
use std::collections::HashMap;

#[test]
fn test_truncation_before_split_maintains_perfect_balance() {
    // Create test sequences with balanced classes
    let num_sequences = 1000; // 200 per class
    let sequence_length = 30;
    let features = 10;

    let mut sequences = Vec::new();
    for i in 0..num_sequences {
        let class = (i % 5) as i32; // Perfect balance: 200 per class
        let strength = 0.5 + (i as f64 / num_sequences as f64) * 0.5;

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx: i * 10,
            end_idx: (i + 1) * 10,
            sequence_data: Array2::zeros((sequence_length, features)),
            targets: vec![TargetData {
                target_type: TargetType::PriceLevel,
                horizon: "1h".to_string(),
                class,
                strength,
            }],
        });
    }

    // Create balanced dataset
    let mut balanced_indices = HashMap::new();
    let target_key = (TargetType::PriceLevel, "1h".to_string());
    let all_indices: Vec<usize> = (0..num_sequences).collect();
    balanced_indices.insert(target_key.clone(), all_indices);

    let mut class_distribution = HashMap::new();
    let mut per_class = HashMap::new();
    for class in 0..5 {
        per_class.insert(class, 200);
    }
    class_distribution.insert(target_key.clone(), per_class);

    let mut dataset = GloballyBalancedDataset {
        balanced_indices,
        class_distribution,
        global_min_class_count: 200,
        total_balanced_samples: 1000,
        overloaded_classes: HashMap::new(),
    };

    // Simulate truncation to 500 samples (100 per class)
    let target_count = 500;
    let samples_to_keep = (target_count / 5) * 5; // 500

    // Group indices by class for balanced truncation
    let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
    for &idx in dataset.balanced_indices.get(&target_key).unwrap() {
        if let Some(seq) = sequences.iter().find(|s| s.sequence_idx == idx) {
            if let Some(class) = seq.get_target_class(TargetType::PriceLevel, "1h") {
                class_indices.entry(class).or_default().push(idx);
            }
        }
    }

    // Calculate samples per class (must be exact for perfect balance)
    let samples_per_class = samples_to_keep / 5; // 100 per class

    // Select balanced indices using stride-based sampling for diversity
    let mut truncated_indices = Vec::with_capacity(samples_to_keep);

    for class in 0..5 {
        if let Some(indices) = class_indices.get(&class) {
            let class_stride = indices.len() as f64 / samples_per_class as f64;

            for i in 0..samples_per_class {
                let idx_in_class =
                    ((i as f64 * class_stride).round() as usize).min(indices.len() - 1);
                truncated_indices.push(indices[idx_in_class]);
            }
        }
    }

    // Update dataset with truncated indices
    dataset
        .balanced_indices
        .insert(target_key.clone(), truncated_indices.clone());
    dataset.total_balanced_samples = samples_to_keep;
    dataset.global_min_class_count = samples_per_class;

    // Update class distribution
    let mut new_distribution = HashMap::new();
    for class in 0..5 {
        new_distribution.insert(class, samples_per_class);
    }
    dataset
        .class_distribution
        .insert(target_key.clone(), new_distribution);

    // Verify truncation maintains perfect balance
    assert_eq!(
        dataset.total_balanced_samples, 500,
        "Total samples should be 500"
    );
    assert_eq!(
        dataset.global_min_class_count, 100,
        "Each class should have 100 samples"
    );

    let class_dist = dataset.class_distribution.get(&target_key).unwrap();
    for class in 0..5 {
        assert_eq!(
            *class_dist.get(&class).unwrap(),
            100,
            "Class {} should have exactly 100 samples",
            class
        );
    }

    // Now split the truncated dataset into train/val/test
    let balancer = SequenceBalancer::new(BalanceConfig {
        max_overlap: 0.3,
        prefer_non_overlapping: true,
        min_sequences_per_class: 10,
    });

    let validation_ratio = 0.2;
    let test_ratio = 0.1;

    let (train_dataset, val_indices, test_indices) = balancer
        .create_diverse_splits(
            &dataset,
            &sequences,
            validation_ratio,
            test_ratio,
            &[TargetType::PriceLevel],
            &["1h".to_string()],
            0, // No validation gap for this test
        )
        .unwrap();

    // Verify splits maintain perfect balance
    let train_indices = train_dataset.balanced_indices.get(&target_key).unwrap();
    let val_indices = val_indices.get(&target_key).unwrap();
    let test_indices = test_indices.get(&target_key).unwrap();

    // Expected: 70% train, 20% val, 10% test of 500 samples
    // 350 train, 100 val, 50 test
    let expected_train = 350;
    let expected_val = 100;
    let expected_test = 50;

    assert_eq!(
        train_indices.len(),
        expected_train,
        "Train should have {} samples",
        expected_train
    );
    assert_eq!(
        val_indices.len(),
        expected_val,
        "Val should have {} samples",
        expected_val
    );
    assert_eq!(
        test_indices.len(),
        expected_test,
        "Test should have {} samples",
        expected_test
    );

    // Verify each split has perfect balance (70 train, 20 val, 10 test per class)
    let train_per_class = expected_train / 5; // 70
    let val_per_class = expected_val / 5; // 20
    let test_per_class = expected_test / 5; // 10

    // Count classes in train split
    let mut train_class_counts = HashMap::new();
    for &idx in train_indices {
        if let Some(seq) = sequences.iter().find(|s| s.sequence_idx == idx) {
            if let Some(class) = seq.get_target_class(TargetType::PriceLevel, "1h") {
                *train_class_counts.entry(class).or_insert(0) += 1;
            }
        }
    }

    for class in 0..5 {
        assert_eq!(
            *train_class_counts.get(&class).unwrap(),
            train_per_class,
            "Train class {} should have exactly {} samples",
            class,
            train_per_class
        );
    }

    // Count classes in val split
    let mut val_class_counts = HashMap::new();
    for &idx in val_indices {
        if let Some(seq) = sequences.iter().find(|s| s.sequence_idx == idx) {
            if let Some(class) = seq.get_target_class(TargetType::PriceLevel, "1h") {
                *val_class_counts.entry(class).or_insert(0) += 1;
            }
        }
    }

    for class in 0..5 {
        assert_eq!(
            *val_class_counts.get(&class).unwrap(),
            val_per_class,
            "Val class {} should have exactly {} samples",
            class,
            val_per_class
        );
    }

    // Count classes in test split
    let mut test_class_counts = HashMap::new();
    for &idx in test_indices {
        if let Some(seq) = sequences.iter().find(|s| s.sequence_idx == idx) {
            if let Some(class) = seq.get_target_class(TargetType::PriceLevel, "1h") {
                *test_class_counts.entry(class).or_insert(0) += 1;
            }
        }
    }

    for class in 0..5 {
        assert_eq!(
            *test_class_counts.get(&class).unwrap(),
            test_per_class,
            "Test class {} should have exactly {} samples",
            class,
            test_per_class
        );
    }

    println!("✅ Truncation before split maintains perfect balance!");
    println!("   Truncated: 1000 → 500 samples (100 per class)");
    println!(
        "   Train: {} samples ({} per class)",
        expected_train, train_per_class
    );
    println!(
        "   Val: {} samples ({} per class)",
        expected_val, val_per_class
    );
    println!(
        "   Test: {} samples ({} per class)",
        expected_test, test_per_class
    );
}
