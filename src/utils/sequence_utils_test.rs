use crate::utils::sequence_utils::*;

#[test]
fn test_calculate_step_size() {
    // No overlap
    assert_eq!(calculate_step_size(0.0, 60), 60);

    // 50% overlap
    assert_eq!(calculate_step_size(0.5, 60), 30);

    // 80% overlap
    assert_eq!(calculate_step_size(0.8, 60), 12);

    // 90% overlap
    assert_eq!(calculate_step_size(0.9, 60), 6);

    // Maximum overlap
    assert_eq!(calculate_step_size(1.0, 60), 1);

    // Edge case: very high overlap
    assert_eq!(calculate_step_size(0.99, 60), 1);
}

#[test]
fn test_calculate_sequence_indices() {
    let indices = calculate_sequence_indices(1000, 60, 12, 24).unwrap();

    // Should start at 0
    assert_eq!(indices[0], 0);

    // Should advance by step_size
    assert_eq!(indices[1], 12);
    assert_eq!(indices[2], 24);

    // Last index should leave room for sequence + horizon
    let last_idx = *indices.last().unwrap();
    assert!(last_idx + 60 + 24 <= 1000);
}

#[test]
fn test_validate_sequence_overlap() {
    assert!(validate_sequence_overlap(0.0).is_ok());
    assert!(validate_sequence_overlap(0.5).is_ok());
    assert!(validate_sequence_overlap(0.99).is_ok());

    assert!(validate_sequence_overlap(-0.1).is_err());
    assert!(validate_sequence_overlap(1.0).is_err());
    assert!(validate_sequence_overlap(1.1).is_err());
}

#[test]
fn test_calculate_expected_sequence_count() {
    // With step_size=12, sequence_length=60, horizon=24
    // Available length = 1000 - 60 - 24 + 1 = 917
    // Count = ceil(917 / 12) = 77
    assert_eq!(calculate_expected_sequence_count(1000, 60, 12, 24), 77);

    // Insufficient data
    assert_eq!(calculate_expected_sequence_count(50, 60, 12, 24), 0);
}

#[test]
fn test_target_sequence_synchronization() {
    // Test case: 1000 data points, 60 sequence length, 24 horizon, 80% overlap
    let total_data = 1000;
    let seq_len = 60;
    let horizon = 24;
    let overlap = 0.8;

    let step_size = calculate_step_size(overlap, seq_len);
    assert_eq!(step_size, 12); // 20% of 60 = 12

    let indices = calculate_sequence_indices(total_data, seq_len, step_size, horizon).unwrap();

    // Verify indices are properly spaced
    for i in 1..indices.len() {
        assert_eq!(indices[i] - indices[i - 1], step_size);
    }

    // Verify last sequence has room for sequence + horizon
    let last_idx = *indices.last().unwrap();
    assert!(last_idx + seq_len + horizon <= total_data);
}

#[test]
fn test_different_overlap_configurations() {
    let seq_len = 60;
    let total_data = 1000;
    let horizon = 24;

    // Test various overlap configurations - calculate expected values dynamically
    let overlaps = vec![0.0, 0.5, 0.8, 0.9, 0.95, 0.99];

    for overlap in overlaps {
        let step_size = calculate_step_size(overlap, seq_len);

        // Verify step size is reasonable
        assert!(
            step_size >= 1,
            "Step size must be at least 1 for overlap {}",
            overlap
        );
        assert!(
            step_size <= seq_len,
            "Step size cannot exceed sequence length for overlap {}",
            overlap
        );

        // For no overlap, step size should equal sequence length
        if overlap == 0.0 {
            assert_eq!(step_size, seq_len);
        }

        // For high overlap, step size should be small
        if overlap >= 0.9 {
            assert!(
                step_size <= seq_len / 10,
                "High overlap should result in small step size"
            );
        }

        let indices = calculate_sequence_indices(total_data, seq_len, step_size, horizon).unwrap();
        assert!(
            !indices.is_empty(),
            "No indices generated for overlap {}",
            overlap
        );

        // Verify step consistency
        if indices.len() > 1 {
            assert_eq!(indices[1] - indices[0], step_size);
        }

        println!(
            "Overlap {}: step_size={}, sequences={}",
            overlap,
            step_size,
            indices.len()
        );
    }
}

#[test]
fn test_edge_cases() {
    // Minimum sequence length - ceil(0.2 * 5) = ceil(1.0) = 1
    let result = calculate_step_size(0.8, 5);
    println!("calculate_step_size(0.8, 5) = {}", result);
    assert_eq!(result, 1);

    // Maximum overlap
    assert_eq!(calculate_step_size(0.99, 100), 1);

    // No overlap
    assert_eq!(calculate_step_size(0.0, 100), 100);

    // Insufficient data
    let result = calculate_sequence_indices(50, 60, 12, 24);
    assert!(result.is_err());
}

#[test]
fn test_validation_functions() {
    // Valid overlap values
    assert!(validate_sequence_overlap(0.0).is_ok());
    assert!(validate_sequence_overlap(0.5).is_ok());
    assert!(validate_sequence_overlap(0.99).is_ok());

    // Invalid overlap values
    assert!(validate_sequence_overlap(-0.1).is_err());
    assert!(validate_sequence_overlap(1.0).is_err());
    assert!(validate_sequence_overlap(1.1).is_err());
}
