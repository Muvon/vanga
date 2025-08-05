use crate::api::trainer::validate_prepared_targets_balance;
use crate::targets::{PreparedTargets, TargetType};
use crate::utils::error::VangaError;

#[test]
fn test_perfect_balance_validation_success() {
    // Create perfectly balanced PreparedTargets: 6 samples per class for each target
    let mut targets = PreparedTargets::new(30);

    // Add horizon
    let horizon = "1h".to_string();

    // Create perfectly balanced data: 6 samples per class (0,1,2,3,4)
    let balanced_data = vec![
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
    ];

    targets
        .price_levels
        .insert(horizon.clone(), balanced_data.clone());
    targets
        .directions
        .insert(horizon.clone(), balanced_data.clone());
    targets
        .volatility
        .insert(horizon.clone(), balanced_data.clone());

    // Set valid indices for all samples
    targets.valid_indices = (0..30).collect();

    // Should pass validation
    let result = validate_prepared_targets_balance(&targets, "TEST");
    assert!(result.is_ok());
}

#[test]
fn test_perfect_balance_validation_failure() {
    // Create imbalanced PreparedTargets
    let mut targets = PreparedTargets::new(25);

    let horizon = "1h".to_string();

    // Create imbalanced data: Class 0: 10 samples, Class 1: 5, Class 2: 5, Class 3: 3, Class 4: 2
    let imbalanced_data = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 10 samples of class 0
        1, 1, 1, 1, 1, // 5 samples of class 1
        2, 2, 2, 2, 2, // 5 samples of class 2
        3, 3, 3, // 3 samples of class 3
        4, 4, // 2 samples of class 4
    ];

    targets
        .price_levels
        .insert(horizon.clone(), imbalanced_data.clone());
    targets
        .directions
        .insert(horizon.clone(), imbalanced_data.clone());
    targets
        .volatility
        .insert(horizon.clone(), imbalanced_data.clone());

    targets.valid_indices = (0..25).collect();

    // Should fail validation
    let result = validate_prepared_targets_balance(&targets, "TEST");
    assert!(result.is_err());

    if let Err(VangaError::DataError(msg)) = result {
        assert!(msg.contains("PERFECT BALANCE VALIDATION FAILED"));
        assert!(msg.contains("IMBALANCE"));
    } else {
        panic!("Expected DataError with balance validation failure");
    }
}

#[test]
fn test_empty_targets_validation() {
    let targets = PreparedTargets::new(0);

    let result = validate_prepared_targets_balance(&targets, "EMPTY");
    assert!(result.is_err());

    if let Err(VangaError::DataError(msg)) = result {
        assert!(msg.contains("no valid indices"));
    } else {
        panic!("Expected DataError for empty targets");
    }
}

#[test]
fn test_invalid_class_validation() {
    let mut targets = PreparedTargets::new(5);

    let horizon = "1h".to_string();

    // Create data with invalid class values
    let invalid_data = vec![0, 1, 2, 5, 4]; // Class 5 is invalid (should be 0-4)

    targets.price_levels.insert(horizon.clone(), invalid_data);
    targets.valid_indices = (0..5).collect();

    let result = validate_prepared_targets_balance(&targets, "INVALID");
    assert!(result.is_err());

    if let Err(VangaError::DataError(msg)) = result {
        assert!(msg.contains("Invalid class 5"));
    } else {
        panic!("Expected DataError for invalid class");
    }
}
