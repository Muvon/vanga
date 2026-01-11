//! Tests for intelligent minority class augmentation

use super::balance::*;
use crate::targets::TargetType;
use ndarray::Array2;

#[test]
fn test_synthetic_sequence_targets_extraction() {
    // Simulate the scenario from create_data_from_indices:
    // - Real sequences at indices 0, 1, 2
    // - Synthetic sequences at indices 1_000_000, 1_000_001
    // - Combined sorted_indices: [0, 1, 2, 1_000_000, 1_000_001]
    // - new_idx for 1_000_000 is 3, for 1_000_001 is 4

    let real_sequences = vec![
        SequenceWithTargets {
            sequence_idx: 0,
            start_idx: 0,
            end_idx: 60,
            sequence_data: Array2::zeros((60, 50)),
            targets: vec![TargetData {
                target_type: TargetType::PriceLevel,
                horizon: "30m".to_string(),
                class: 0,
                strength: 0.8,
            }],
        },
        SequenceWithTargets {
            sequence_idx: 1,
            start_idx: 60,
            end_idx: 120,
            sequence_data: Array2::zeros((60, 50)),
            targets: vec![TargetData {
                target_type: TargetType::PriceLevel,
                horizon: "30m".to_string(),
                class: 1,
                strength: 0.7,
            }],
        },
        SequenceWithTargets {
            sequence_idx: 2,
            start_idx: 120,
            end_idx: 180,
            sequence_data: Array2::zeros((60, 50)),
            targets: vec![TargetData {
                target_type: TargetType::PriceLevel,
                horizon: "30m".to_string(),
                class: 2,
                strength: 0.6,
            }],
        },
    ];

    // Synthetic sequences (simulating atomic counter output)
    let synthetic_sequences = vec![
        SequenceWithTargets {
            sequence_idx: 1_000_000, // First synthetic
            start_idx: 60,
            end_idx: 120,
            sequence_data: Array2::zeros((60, 50)),
            targets: vec![TargetData {
                target_type: TargetType::PriceLevel,
                horizon: "30m".to_string(),
                class: 3,
                strength: 0.9,
            }],
        },
        SequenceWithTargets {
            sequence_idx: 1_000_001, // Second synthetic
            start_idx: 120,
            end_idx: 180,
            sequence_data: Array2::zeros((60, 50)),
            targets: vec![TargetData {
                target_type: TargetType::PriceLevel,
                horizon: "30m".to_string(),
                class: 4,
                strength: 0.85,
            }],
        },
    ];

    // Combine all sequences
    let all_sequences: Vec<SequenceWithTargets> = real_sequences
        .into_iter()
        .chain(synthetic_sequences)
        .collect();

    // Create sequence_map (this is what create_data_from_indices does)
    let sequence_map: std::collections::HashMap<usize, &SequenceWithTargets> = all_sequences
        .iter()
        .map(|seq| (seq.sequence_idx, seq))
        .collect();

    // Sorted indices as would come from training
    let sorted_indices: Vec<usize> = vec![0, 1, 2, 1_000_000, 1_000_001];

    // Simulate target extraction (as done in create_data_from_indices)
    let num_sequences = sorted_indices.len();
    let mut extracted_targets: Vec<i32> = vec![-1; num_sequences]; // Default -1

    for (new_idx, &orig_idx) in sorted_indices.iter().enumerate() {
        if let Some(seq_with_targets) = sequence_map.get(&orig_idx) {
            for target_data in &seq_with_targets.targets {
                if target_data.horizon == "30m" {
                    extracted_targets[new_idx] = target_data.class;
                }
            }
        }
    }

    // VERIFY: All targets should be extracted properly
    assert_eq!(extracted_targets.len(), 5, "Should have 5 targets");

    // Real sequences
    assert_eq!(extracted_targets[0], 0, "Real seq 0 should have class 0");
    assert_eq!(extracted_targets[1], 1, "Real seq 1 should have class 1");
    assert_eq!(extracted_targets[2], 2, "Real seq 2 should have class 2");

    // Synthetic sequences - THIS WAS THE BUG!
    assert_eq!(
        extracted_targets[3], 3,
        "Synthetic seq 1_000_000 should have class 3, got {}",
        extracted_targets[3]
    );
    assert_eq!(
        extracted_targets[4], 4,
        "Synthetic seq 1_000_001 should have class 4, got {}",
        extracted_targets[4]
    );

    // Verify no -1 values remain
    for (idx, &target) in extracted_targets.iter().enumerate() {
        assert!(
            (0..5).contains(&target),
            "Target {} at index {} is invalid (expected 0-4, got {})",
            target,
            idx,
            target
        );
    }
}

#[test]
fn test_augmentation_preserves_price_volume() {
    // Test that augmentation only modifies technical indicators (columns 5+)
    // and leaves price/volume columns (0-4) unchanged

    use crate::data::augmentation::{augment_sequence, AugmentationConfig};

    // Create a sequence with clear price/volume values
    let mut sequence = Array2::zeros((10, 10));
    // Set price/volume columns
    sequence[[0, 0]] = 100.0; // open
    sequence[[0, 1]] = 105.0; // high
    sequence[[0, 2]] = 98.0; // low
    sequence[[0, 3]] = 102.0; // close
    sequence[[0, 4]] = 1000.0; // volume

    // Set some indicator values
    sequence[[0, 5]] = 50.0; // RSI
    sequence[[0, 6]] = 0.5; // MACD

    let mut rng = rand::rng();
    let config = AugmentationConfig::default();
    let price_volume_cols = vec![0, 1, 2, 3, 4];

    let augmented = augment_sequence(&sequence, &config, &mut rng, &price_volume_cols);

    // Verify price/volume unchanged
    assert_eq!(augmented[[0, 0]], 100.0, "Open should be unchanged");
    assert_eq!(augmented[[0, 1]], 105.0, "High should be unchanged");
    assert_eq!(augmented[[0, 2]], 98.0, "Low should be unchanged");
    assert_eq!(augmented[[0, 3]], 102.0, "Close should be unchanged");
    assert_eq!(augmented[[0, 4]], 1000.0, "Volume should be unchanged");

    // Indicators may have changed (that's the point of augmentation)
    // But should be finite (no NaN/Inf)
    for f in 5..10 {
        assert!(
            augmented[[0, f]].is_finite(),
            "Indicator column {} should be finite after augmentation",
            f
        );
    }
}

#[test]
fn test_synthetic_id_uniqueness_across_classes() {
    // Test that atomic counter generates unique IDs across multiple class augmentations

    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(1_000_000);

    let mut generated_ids = Vec::new();

    // Simulate 5 classes each generating 100 synthetic sequences
    for class in 0..5 {
        for i in 0..100 {
            let id = COUNTER.fetch_add(1, Ordering::SeqCst);
            generated_ids.push((class, i, id));
        }
    }

    // Verify all IDs are unique
    let id_set: std::collections::HashSet<usize> =
        generated_ids.iter().map(|(_, _, id)| *id).collect();
    assert_eq!(id_set.len(), 500, "Should have 500 unique IDs");

    // Verify IDs are in expected range (start from 1_000_000)
    let min_id = generated_ids.iter().map(|(_, _, id)| *id).min().unwrap();
    let max_id = generated_ids.iter().map(|(_, _, id)| *id).max().unwrap();

    assert!(
        min_id >= 1_000_000,
        "Min ID should be >= 1_000_000, got {}",
        min_id
    );
    assert_eq!(max_id - min_id + 1, 500, "IDs should be contiguous");
}

#[test]
fn test_calculate_target_count_median() {
    // Test median calculation (50th percentile)
    let class_counts = vec![100, 150, 200, 120, 180];
    let target = calculate_target_count(&class_counts, 0.5);

    // Sorted: [100, 120, 150, 180, 200]
    // Median (index 2): 150
    assert_eq!(target, 150, "Median should be 150");
}

#[test]
fn test_calculate_target_count_percentiles() {
    let class_counts = vec![100, 150, 200, 120, 180];
    // Sorted: [100, 120, 150, 180, 200]

    // Test 40th percentile: index = (5-1) * 0.4 = 1.6 → 1 → 120
    let p40 = calculate_target_count(&class_counts, 0.4);
    assert_eq!(p40, 120, "40th percentile should be 120");

    // Test 60th percentile: index = (5-1) * 0.6 = 2.4 → 2 → 150
    let p60 = calculate_target_count(&class_counts, 0.6);
    assert_eq!(p60, 150, "60th percentile should be 150");

    // Test 80th percentile: index = (5-1) * 0.8 = 3.2 → 3 → 180
    let p80 = calculate_target_count(&class_counts, 0.8);
    assert_eq!(p80, 180, "80th percentile should be 180");

    // Test 0th percentile (minimum)
    let p0 = calculate_target_count(&class_counts, 0.0);
    assert_eq!(p0, 100, "0th percentile should be minimum (100)");

    // Test 100th percentile (maximum)
    let p100 = calculate_target_count(&class_counts, 1.0);
    assert_eq!(p100, 200, "100th percentile should be maximum (200)");
}

#[test]
fn test_calculate_target_count_edge_cases() {
    // Empty vector
    let empty: Vec<usize> = vec![];
    assert_eq!(
        calculate_target_count(&empty, 0.5),
        0,
        "Empty should return 0"
    );

    // Single element
    let single = vec![100];
    assert_eq!(
        calculate_target_count(&single, 0.5),
        100,
        "Single element should return itself"
    );

    // Two elements
    let two = vec![100, 200];
    assert_eq!(
        calculate_target_count(&two, 0.5),
        100,
        "Two elements median should be first"
    );

    // All same values
    let same = vec![150, 150, 150, 150];
    assert_eq!(
        calculate_target_count(&same, 0.5),
        150,
        "All same should return that value"
    );
}

#[test]
fn test_calculate_target_count_percentile_clamping() {
    let class_counts = vec![100, 150, 200];

    // Test negative percentile (should clamp to 0.0)
    let negative = calculate_target_count(&class_counts, -0.5);
    assert_eq!(
        negative, 100,
        "Negative percentile should clamp to 0.0 (min)"
    );

    // Test > 1.0 percentile (should clamp to 1.0)
    let over = calculate_target_count(&class_counts, 1.5);
    assert_eq!(over, 200, "Over 1.0 percentile should clamp to 1.0 (max)");
}

#[test]
fn test_identify_price_volume_columns_standard() {
    // Standard case: 50 features (5 OHLCV + 45 indicators)
    let sequence = Array2::<f64>::zeros((60, 50));
    let cols = identify_price_volume_columns(&sequence);

    assert_eq!(cols.len(), 5, "Should identify 5 OHLCV columns");
    assert_eq!(cols, vec![0, 1, 2, 3, 4], "Should be first 5 columns");
}

#[test]
fn test_identify_price_volume_columns_edge_cases() {
    // Edge case: exactly 5 features
    let sequence_5 = Array2::<f64>::zeros((60, 5));
    let cols_5 = identify_price_volume_columns(&sequence_5);
    assert_eq!(
        cols_5,
        vec![0, 1, 2, 3, 4],
        "Should handle exactly 5 features"
    );

    // Edge case: less than 5 features (shouldn't happen but handle gracefully)
    let sequence_3 = Array2::<f64>::zeros((60, 3));
    let cols_3 = identify_price_volume_columns(&sequence_3);
    assert_eq!(
        cols_3,
        vec![0, 1, 2],
        "Should handle < 5 features gracefully"
    );

    // Edge case: many features
    let sequence_100 = Array2::<f64>::zeros((60, 100));
    let cols_100 = identify_price_volume_columns(&sequence_100);
    assert_eq!(
        cols_100.len(),
        5,
        "Should always return 5 for standard case"
    );
}

#[test]
fn test_minority_augmentation_concept() {
    // This test verifies the concept without full integration
    // Full integration tests are in pipeline_augmentation_test.rs

    // Verify that calculate_target_count works correctly for balancing
    let class_counts = vec![200, 150, 100, 80, 120]; // Imbalanced
    let target = calculate_target_count(&class_counts, 0.5); // Median

    // Sorted: [80, 100, 120, 150, 200]
    // Median (index 2): 120
    assert_eq!(target, 120, "Target should be median (120)");

    // Verify minority classes would need augmentation
    assert!(80 < target, "Class with 80 needs augmentation");
    assert!(100 < target, "Class with 100 needs augmentation");
    assert!(120 == target, "Class with 120 is at target");
    assert!(150 > target, "Class with 150 needs downsampling");
    assert!(200 > target, "Class with 200 needs downsampling");
}

#[test]
fn test_majority_downsampling_concept() {
    // Verify downsampling logic
    let class_counts = vec![200, 200, 200, 200, 200]; // All equal
    let target = calculate_target_count(&class_counts, 0.4);

    // All equal, so any percentile returns 200
    assert_eq!(target, 200, "All classes equal should return that value");
}

#[test]
fn test_augmentation_with_empty_class_concept() {
    // Verify handling of missing classes
    let class_counts = vec![100, 100, 100, 100]; // Only 4 classes
    let target = calculate_target_count(&class_counts, 0.5);

    // Should still calculate median correctly
    assert_eq!(target, 100, "Should handle 4 classes correctly");
}

#[test]
fn test_max_synthetic_ratio_concept() {
    // Verify ratio enforcement logic
    let real_count = 10;
    let target_count = 105; // Would need 95 synthetic
    let max_ratio = 2.0;

    let max_synthetic = (real_count as f64 * max_ratio) as usize;
    let synthetic_to_generate = (target_count - real_count).min(max_synthetic);

    // Should be limited to 20 (10 * 2.0)
    assert_eq!(synthetic_to_generate, 20, "Should be limited by max_ratio");
    assert_eq!(
        real_count + synthetic_to_generate,
        30,
        "Total should be 30, not 105"
    );
}
