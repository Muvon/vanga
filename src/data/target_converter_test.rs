use crate::data::target_converter::*;
use crate::targets::PreparedTargets;

fn create_test_targets() -> PreparedTargets {
    let mut targets = PreparedTargets::new(100);
    targets
        .price_levels
        .insert("1h".to_string(), vec![0, 1, 2, 3, 4]);
    targets
        .direction
        .insert("1h".to_string(), vec![0, 1, 2, 3, 4]); // Use full 5-class range
    targets
        .volatility
        .insert("1h".to_string(), vec![4, 3, 2, 1, 0]); // Use full 5-class range
    targets.valid_indices = vec![0, 1, 2, 3, 4];
    targets
}

#[test]
fn test_target_conversion() {
    let converter = TargetConverter::new();
    let targets = create_test_targets();

    let result = converter.convert_to_training_array(&targets, &targets.valid_indices, "1h");
    assert!(result.is_ok());

    let training_array = result.unwrap();
    assert_eq!(training_array.shape()[0], 5); // 5 samples
    assert_eq!(training_array.shape()[1], 15); // 5 (price) + 5 (direction) + 5 (volatility)
}

#[test]
fn test_target_validation() {
    let converter = TargetConverter::new();
    let targets = create_test_targets();

    let result = converter.validate_targets(&targets, "1h");
    assert!(result.is_ok());
}
