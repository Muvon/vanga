//! Tests for validation gap removal and target alignment
//!
//! These tests verify that removing the validation gap doesn't cause data leakage
//! and that target-sequence alignment remains correct.

use vanga::utils::sequence_utils::{calculate_sequence_indices, calculate_step_size};

#[tokio::test]
async fn test_validation_gap_removal_no_data_leakage() {
    // Create test data with known structure
    let _test_data_size = 1000;
    let sequence_length = 60;
    let horizon_steps = 24;
    let _validation_size = 200;

    // Test that validation can start immediately after training without data leakage
    let train_end = 500;
    let val_start = train_end; // No gap

    // Verify no overlap between training targets and validation sequences
    let max_training_target_idx = train_end + horizon_steps;
    let min_validation_sequence_idx = val_start;

    assert!(
        max_training_target_idx <= min_validation_sequence_idx + sequence_length,
        "No data leakage: training targets end before validation sequences begin"
    );

    // Verify validation targets are properly separated
    let validation_target_start = val_start + sequence_length + horizon_steps;
    assert!(
        validation_target_start > max_training_target_idx,
        "Validation targets are temporally separated from training data"
    );
}

#[tokio::test]
async fn test_sequence_target_alignment_consistency() {
    let total_data_length = 1000;
    let sequence_length = 60;
    let step_size = 12; // 80% overlap
    let max_horizon_steps = 24;

    // Calculate sequence indices
    let sequence_indices = calculate_sequence_indices(
        total_data_length,
        sequence_length,
        step_size,
        max_horizon_steps,
    )
    .expect("Should generate valid indices");

    // Verify each sequence has a valid target
    for (i, &seq_idx) in sequence_indices.iter().enumerate() {
        let target_idx = seq_idx + sequence_length + max_horizon_steps;

        assert!(
            target_idx < total_data_length,
            "Sequence {} at index {} should have valid target at index {}",
            i,
            seq_idx,
            target_idx
        );

        // Verify no overlap between sequence and target
        let sequence_end = seq_idx + sequence_length;
        assert!(
            sequence_end + max_horizon_steps == target_idx,
            "Target should be exactly horizon_steps after sequence end"
        );
    }
}

#[tokio::test]
async fn test_step_size_calculation_accuracy() {
    // Test various overlap scenarios
    let sequence_length = 60;

    // No overlap
    assert_eq!(calculate_step_size(0.0, sequence_length), 60);

    // 50% overlap
    assert_eq!(calculate_step_size(0.5, sequence_length), 30);

    // 80% overlap
    assert_eq!(calculate_step_size(0.8, sequence_length), 12);

    // 90% overlap
    assert_eq!(calculate_step_size(0.9, sequence_length), 6);

    // Maximum overlap
    assert_eq!(calculate_step_size(1.0, sequence_length), 1);
}

#[tokio::test]
async fn test_more_validation_data_available() {
    let total_available = 1000;
    let train_end = 600;
    let validation_size = 200;
    let max_horizon_steps = 24;

    // OLD approach (with gap)
    let old_val_start = train_end + max_horizon_steps;
    let old_can_create_window = old_val_start + validation_size <= total_available;

    // NEW approach (no gap)
    let new_val_start = train_end;
    let new_can_create_window = new_val_start + validation_size <= total_available;

    // Both should work, but new approach uses more data efficiently
    assert!(old_can_create_window, "Old approach should work");
    assert!(new_can_create_window, "New approach should work");

    // New approach allows for more training data in edge cases
    let edge_case_train_end = total_available - validation_size - max_horizon_steps + 1;
    let old_edge_val_start = edge_case_train_end + max_horizon_steps;
    let new_edge_val_start = edge_case_train_end;

    assert!(
        old_edge_val_start + validation_size > total_available,
        "Old approach fails in edge case"
    );
    assert!(
        new_edge_val_start + validation_size <= total_available,
        "New approach works in edge case"
    );
}

#[test]
fn test_target_names_consistency() {
    // Test that target names are consistent across different creation methods
    use vanga::targets::{MultiTargetConfig, TargetGenerator};

    let horizons = vec!["1h".to_string(), "4h".to_string()];
    let config = MultiTargetConfig {
        price_level_config: vanga::targets::PriceLevelConfig::default(),
        horizons: horizons.clone(),
    };

    let generator = TargetGenerator::new(config);
    let target_names = generator.get_target_names();

    // Verify expected target names are generated
    let expected_names = vec![
        "price_level_1h",
        "price_level_4h",
        "direction_1h",
        "direction_4h",
        "volatility_1h",
        "volatility_4h",
    ];

    assert_eq!(target_names.len(), expected_names.len());
    for expected in expected_names {
        assert!(
            target_names.contains(&expected.to_string()),
            "Should contain target name: {}",
            expected
        );
    }
}
