//! Comprehensive tests for overlap optimizer

use super::overlap_optimizer::{count_classes, *};
use ndarray::{Array2, Array3};

#[test]
fn test_calculate_optimal_overlap_normal_case() {
    // 1000 data points, 60 sequence length, 30 baseline overlap, 500 target samples
    let result = calculate_optimal_overlap(1000, 60, 30, 500);
    assert!(result.is_ok());

    let overlap = result.unwrap();
    assert!(overlap >= 30, "Overlap should be at least baseline");
    assert!(overlap < 60, "Overlap should be less than sequence length");

    // Verify the formula produces reasonable results
    let estimated_sequences = (1000 - 60) / (60 - overlap) + 1;
    assert!(
        estimated_sequences >= 450 && estimated_sequences <= 550,
        "Estimated sequences {} should be close to target 500",
        estimated_sequences
    );
}

#[test]
fn test_calculate_optimal_overlap_edge_cases() {
    // Target equals minimum possible sequences (no overlap needed)
    let result = calculate_optimal_overlap(1000, 60, 30, 15);
    assert!(result.is_ok());
    let overlap = result.unwrap();
    assert!(overlap <= 30, "Should use baseline or less for low targets");

    // Target requires maximum overlap
    let result = calculate_optimal_overlap(1000, 60, 30, 900);
    assert!(result.is_ok());
    let overlap = result.unwrap();
    assert!(overlap > 50, "Should use high overlap for high targets");
}

#[test]
fn test_calculate_optimal_overlap_impossible_target() {
    // Target exceeds maximum possible sequences
    let result = calculate_optimal_overlap(1000, 60, 30, 100000);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("impossible to achieve"));
}

#[test]
fn test_calculate_optimal_overlap_disabled() {
    // count = 0 should return baseline
    let result = calculate_optimal_overlap(1000, 60, 30, 0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 30);
}

#[test]
fn test_calculate_optimal_overlap_invalid_inputs() {
    // Sequence length >= data points
    let result = calculate_optimal_overlap(100, 100, 30, 50);
    assert!(result.is_err());

    let result = calculate_optimal_overlap(100, 150, 30, 50);
    assert!(result.is_err());
}

#[test]
fn test_find_optimal_overlap_with_mock_calibration() {
    // Mock calibration function that returns predictable results
    let mock_calibration = |overlap: usize| -> Result<usize, crate::utils::error::VangaError> {
        // Simulate: more overlap = more samples
        let samples = 100 + overlap * 10;
        Ok(samples)
    };

    let result = find_optimal_overlap_with_calibration(
        mock_calibration,
        1000, // data_points
        30,   // baseline_overlap
        60,   // sequence_length
        500,  // target_samples
        0.05, // tolerance
    );

    assert!(result.is_ok());
    let (overlap, samples) = result.unwrap();
    assert!(overlap >= 30, "Should be at least baseline");
    assert!(
        samples >= 475,
        "Should reach at least 95% of target (500 * 0.95)"
    );
}

#[test]
fn test_find_optimal_overlap_disabled() {
    let mock_calibration =
        |_overlap: usize| -> Result<usize, crate::utils::error::VangaError> { Ok(200) };

    let result = find_optimal_overlap_with_calibration(
        mock_calibration,
        1000, // data_points
        30,   // baseline_overlap
        60,   // sequence_length
        0,    // target_samples (disabled)
        0.05, // tolerance
    );

    assert!(result.is_ok());
    let (overlap, samples) = result.unwrap();
    assert_eq!(overlap, 30, "Should return baseline when disabled");
    assert_eq!(samples, 200);
}

#[test]
fn test_truncate_balanced_sequences_perfect_balance() {
    // Create perfectly balanced dataset: 10 samples, 5 classes, 2 per class
    let sequences = Array3::<f64>::zeros((10, 60, 50));
    let targets = Array2::from_shape_vec(
        (10, 5),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, // Class 0
            1.0, 0.0, 0.0, 0.0, 0.0, // Class 0
            0.0, 1.0, 0.0, 0.0, 0.0, // Class 1
            0.0, 1.0, 0.0, 0.0, 0.0, // Class 1
            0.0, 0.0, 1.0, 0.0, 0.0, // Class 2
            0.0, 0.0, 1.0, 0.0, 0.0, // Class 2
            0.0, 0.0, 0.0, 1.0, 0.0, // Class 3
            0.0, 0.0, 0.0, 1.0, 0.0, // Class 3
            0.0, 0.0, 0.0, 0.0, 1.0, // Class 4
            0.0, 0.0, 0.0, 0.0, 1.0, // Class 4
        ],
    )
    .unwrap();

    // Truncate to 5 samples (1 per class)
    let result = truncate_balanced_sequences(&sequences, &targets, 5);
    assert!(result.is_ok());

    let (trunc_seq, trunc_targets) = result.unwrap();
    assert_eq!(trunc_seq.shape()[0], 5);
    assert_eq!(trunc_targets.shape()[0], 5);

    // Verify perfect balance maintained
    let counts = count_classes(&trunc_targets);
    assert_eq!(counts.len(), 5);
    for i in 0..5 {
        assert_eq!(counts[&i], 1, "Each class should have exactly 1 sample");
    }
}

#[test]
fn test_truncate_balanced_sequences_larger_dataset() {
    // Create 100 samples, 5 classes, 20 per class
    let sequences = Array3::<f64>::zeros((100, 60, 50));
    let mut target_vec = Vec::new();
    for class_idx in 0..5 {
        for _ in 0..20 {
            let mut row = vec![0.0; 5];
            row[class_idx] = 1.0;
            target_vec.extend(row);
        }
    }
    let targets = Array2::from_shape_vec((100, 5), target_vec).unwrap();

    // Truncate to 50 samples (10 per class)
    let result = truncate_balanced_sequences(&sequences, &targets, 50);
    assert!(result.is_ok());

    let (trunc_seq, trunc_targets) = result.unwrap();
    assert_eq!(trunc_seq.shape()[0], 50);
    assert_eq!(trunc_targets.shape()[0], 50);

    // Verify perfect balance
    let counts = count_classes(&trunc_targets);
    assert_eq!(counts.len(), 5);
    for i in 0..5 {
        assert_eq!(counts[&i], 10, "Each class should have exactly 10 samples");
    }
}

#[test]
fn test_truncate_balanced_sequences_invalid_inputs() {
    let sequences = Array3::<f64>::zeros((10, 60, 50));
    let targets = Array2::<f64>::zeros((10, 5));

    // Target count exceeds available samples
    let result = truncate_balanced_sequences(&sequences, &targets, 20);
    assert!(result.is_err());

    // Target count not divisible by num_classes
    let result = truncate_balanced_sequences(&sequences, &targets, 7);
    assert!(result.is_err());
}

#[test]
fn test_truncate_balanced_sequences_imbalanced_input() {
    // Create imbalanced dataset
    let sequences = Array3::<f64>::zeros((10, 60, 50));
    let targets = Array2::from_shape_vec(
        (10, 5),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, // Class 0
            1.0, 0.0, 0.0, 0.0, 0.0, // Class 0
            1.0, 0.0, 0.0, 0.0, 0.0, // Class 0 (imbalanced!)
            0.0, 1.0, 0.0, 0.0, 0.0, // Class 1
            0.0, 1.0, 0.0, 0.0, 0.0, // Class 1
            0.0, 0.0, 1.0, 0.0, 0.0, // Class 2
            0.0, 0.0, 1.0, 0.0, 0.0, // Class 2
            0.0, 0.0, 0.0, 1.0, 0.0, // Class 3
            0.0, 0.0, 0.0, 0.0, 1.0, // Class 4
            0.0, 0.0, 0.0, 0.0, 1.0, // Class 4
        ],
    )
    .unwrap();

    // Should fail because input is not perfectly balanced
    let result = truncate_balanced_sequences(&sequences, &targets, 5);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("not perfectly balanced"));
}

#[test]
fn test_count_classes_helper() {
    let targets = Array2::from_shape_vec(
        (15, 5),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, // Class 0
            1.0, 0.0, 0.0, 0.0, 0.0, // Class 0
            1.0, 0.0, 0.0, 0.0, 0.0, // Class 0
            0.0, 1.0, 0.0, 0.0, 0.0, // Class 1
            0.0, 1.0, 0.0, 0.0, 0.0, // Class 1
            0.0, 1.0, 0.0, 0.0, 0.0, // Class 1
            0.0, 0.0, 1.0, 0.0, 0.0, // Class 2
            0.0, 0.0, 1.0, 0.0, 0.0, // Class 2
            0.0, 0.0, 1.0, 0.0, 0.0, // Class 2
            0.0, 0.0, 0.0, 1.0, 0.0, // Class 3
            0.0, 0.0, 0.0, 1.0, 0.0, // Class 3
            0.0, 0.0, 0.0, 1.0, 0.0, // Class 3
            0.0, 0.0, 0.0, 0.0, 1.0, // Class 4
            0.0, 0.0, 0.0, 0.0, 1.0, // Class 4
            0.0, 0.0, 0.0, 0.0, 1.0, // Class 4
        ],
    )
    .unwrap();

    let counts = count_classes(&targets);
    assert_eq!(counts.len(), 5);
    assert_eq!(counts[&0], 3);
    assert_eq!(counts[&1], 3);
    assert_eq!(counts[&2], 3);
    assert_eq!(counts[&3], 3);
    assert_eq!(counts[&4], 3);
}

#[test]
fn test_temporal_diversity_preservation() {
    // Create dataset with temporal markers (use sequence values to track position)
    let mut sequences = Array3::<f64>::zeros((20, 60, 50));
    for i in 0..20 {
        sequences[[i, 0, 0]] = i as f64; // Mark temporal position
    }

    // Create perfectly balanced targets (4 per class)
    let mut target_vec = Vec::new();
    for class_idx in 0..5 {
        for _ in 0..4 {
            let mut row = vec![0.0; 5];
            row[class_idx] = 1.0;
            target_vec.extend(row);
        }
    }
    let targets = Array2::from_shape_vec((20, 5), target_vec).unwrap();

    // Truncate to 10 samples (2 per class)
    let result = truncate_balanced_sequences(&sequences, &targets, 10);
    assert!(result.is_ok());

    let (trunc_seq, _) = result.unwrap();

    // Verify temporal diversity: selected sequences should be spread across time
    let mut temporal_positions = Vec::new();
    for i in 0..10 {
        temporal_positions.push(trunc_seq[[i, 0, 0]] as usize);
    }

    // Check that positions are reasonably spread (not all clustered)
    temporal_positions.sort_unstable();
    let min_pos = temporal_positions[0];
    let max_pos = temporal_positions[temporal_positions.len() - 1];
    let spread = max_pos - min_pos;

    assert!(
        spread >= 10,
        "Temporal spread {} should be at least 10 (half of original 20)",
        spread
    );
}

#[test]
fn test_balance_overhead_calculation() {
    // Test that balance_overhead correctly calculates raw target
    let target_balanced = 5000;
    let balance_overhead = 0.3;

    // Expected: 5000 * 1.3 = 6500 raw sequences needed
    let raw_target = (target_balanced as f64 * (1.0 + balance_overhead)) as usize;
    assert_eq!(raw_target, 6500);

    // Test with different overhead values
    let overhead_20 = 0.2;
    let raw_target_20 = (target_balanced as f64 * (1.0 + overhead_20)) as usize;
    assert_eq!(raw_target_20, 6000);

    let overhead_50 = 0.5;
    let raw_target_50 = (target_balanced as f64 * (1.0 + overhead_50)) as usize;
    assert_eq!(raw_target_50, 7500);

    // Test that this accounts for balancing loss
    // If we generate 6500 raw sequences and lose 30% during balancing,
    // we should end up with ~5000 balanced samples
    let simulated_balanced = (raw_target as f64 / (1.0 + balance_overhead)) as usize;
    assert_eq!(simulated_balanced, target_balanced);
}
