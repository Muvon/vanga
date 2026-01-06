use crate::model::calibration::temperature::*;
use ndarray::Array2;

#[test]
fn test_temperature_scaling_creation() {
    let temp_scaling = AdaptiveTemperatureScaling::new();

    assert_eq!(temp_scaling.temperatures, [1.0; 5]);
    assert!(!temp_scaling.is_optimized);
    assert!(temp_scaling.ece_history.is_empty());
}

#[test]
fn test_temperature_scaling_default() {
    let temp_scaling = AdaptiveTemperatureScaling::default();

    assert_eq!(temp_scaling.temperatures, [1.0; 5]);
    assert_eq!(temp_scaling.per_class_ece, [0.0; 5]);
}

#[test]
fn test_optimize_with_empty_data() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    let logits = Array2::zeros((0, 5));
    let targets = Array2::zeros((0, 5));

    let result = temp_scaling.optimize_temperatures(&logits, &targets);
    assert!(result.is_ok());
    assert!(!temp_scaling.is_optimized);
}

#[test]
fn test_optimize_with_invalid_dimensions() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    let logits = Array2::zeros((10, 3));
    let targets = Array2::zeros((10, 5));

    let result = temp_scaling.optimize_temperatures(&logits, &targets);
    assert!(result.is_err());
}

#[test]
fn test_optimize_temperatures_nll_minimization() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    // Create overconfident predictions
    let mut logits_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..50 {
        let class = i % 5;

        // Very high logits (overconfident)
        let mut logit_row = vec![-5.0; 5];
        logit_row[class] = 5.0;
        logits_data.extend_from_slice(&logit_row);

        let mut target_row = vec![0.0; 5];
        target_row[class] = 1.0;
        targets_data.extend_from_slice(&target_row);
    }

    let logits = Array2::from_shape_vec((50, 5), logits_data).unwrap();
    let targets = Array2::from_shape_vec((50, 5), targets_data).unwrap();

    let result = temp_scaling.optimize_temperatures(&logits, &targets);
    assert!(result.is_ok());
    assert!(temp_scaling.is_optimized);

    // Temperatures should be adjusted (not all 1.0)
    let all_one = temp_scaling
        .temperatures
        .iter()
        .all(|&t| (t - 1.0).abs() < 0.01);
    assert!(!all_one, "Temperatures should be optimized, not all 1.0");

    // All temperatures should be in reasonable range [0.1, 5.0]
    for &temp in &temp_scaling.temperatures {
        assert!(temp >= 0.1 && temp <= 5.0);
    }
}

#[test]
fn test_apply_to_logits_before_optimization() {
    let temp_scaling = AdaptiveTemperatureScaling::new();

    let logits = Array2::from_shape_vec(
        (2, 5),
        vec![1.0, 0.5, 0.0, -0.5, -1.0, 2.0, 1.0, 0.0, -1.0, -2.0],
    )
    .unwrap();

    let result = temp_scaling.apply_to_logits(&logits).unwrap();

    // Should return unchanged when not optimized
    assert_eq!(logits, result);
}

#[test]
fn test_apply_to_logits_after_optimization() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    // Optimize first
    let logits = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![5.0, -2.0, -2.0, -2.0, -2.0])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let targets = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![1.0, 0.0, 0.0, 0.0, 0.0])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    temp_scaling
        .optimize_temperatures(&logits, &targets)
        .unwrap();

    // Apply to new logits
    let test_logits = Array2::from_shape_vec(
        (2, 5),
        vec![5.0, -2.0, -2.0, -2.0, -2.0, 3.0, -1.0, -1.0, -1.0, -1.0],
    )
    .unwrap();

    let result = temp_scaling.apply_to_logits(&test_logits).unwrap();

    // Result should be different (temperature scaled)
    assert_ne!(test_logits, result);

    // Result should still be valid probabilities (sum to 1.0)
    for row in result.outer_iter() {
        let sum: f64 = row.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Row sum: {}", sum);
    }
}

#[test]
fn test_get_temperatures() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    assert_eq!(temp_scaling.get_temperatures(), [1.0; 5]);

    // Manually set temperatures
    temp_scaling.temperatures = [0.5, 1.0, 1.5, 2.0, 2.5];

    assert_eq!(temp_scaling.get_temperatures(), [0.5, 1.0, 1.5, 2.0, 2.5]);
}

#[test]
fn test_get_per_class_ece() {
    let temp_scaling = AdaptiveTemperatureScaling::new();

    assert_eq!(temp_scaling.get_per_class_ece(), [0.0; 5]);
}

#[test]
fn test_get_latest_ece() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    assert!(temp_scaling.get_latest_ece().is_none());

    // Add some ECE values
    temp_scaling.ece_history.push(0.1);
    temp_scaling.ece_history.push(0.05);

    assert_eq!(temp_scaling.get_latest_ece(), Some(0.05));
}

#[test]
fn test_reset() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    // Optimize first
    let logits = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![5.0, -2.0, -2.0, -2.0, -2.0])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let targets = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![1.0, 0.0, 0.0, 0.0, 0.0])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    temp_scaling
        .optimize_temperatures(&logits, &targets)
        .unwrap();
    assert!(temp_scaling.is_optimized);

    // Reset
    temp_scaling.reset();

    assert!(!temp_scaling.is_optimized);
    assert_eq!(temp_scaling.temperatures, [1.0; 5]);
    assert!(temp_scaling.ece_history.is_empty());
    assert_eq!(temp_scaling.per_class_ece, [0.0; 5]);
}

// Note: calculate_nll is private, tested indirectly through optimize_temperatures

#[test]
fn test_temperature_bounds() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    // Create data
    let logits = Array2::from_shape_vec(
        (50, 5),
        (0..50)
            .flat_map(|_| vec![5.0, -2.0, -2.0, -2.0, -2.0])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let targets = Array2::from_shape_vec(
        (50, 5),
        (0..50)
            .flat_map(|_| vec![1.0, 0.0, 0.0, 0.0, 0.0])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    temp_scaling
        .optimize_temperatures(&logits, &targets)
        .unwrap();

    // All temperatures should be within [0.1, 5.0]
    for &temp in &temp_scaling.temperatures {
        assert!(temp >= 0.1, "Temperature {} below minimum 0.1", temp);
        assert!(temp <= 5.0, "Temperature {} above maximum 5.0", temp);
    }
}

#[test]
fn test_ece_history_tracking() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    let logits = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![5.0, -2.0, -2.0, -2.0, -2.0])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let targets = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![1.0, 0.0, 0.0, 0.0, 0.0])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    temp_scaling
        .optimize_temperatures(&logits, &targets)
        .unwrap();

    // ECE history should have one entry after optimization
    assert_eq!(temp_scaling.ece_history.len(), 1);
    assert!(temp_scaling.ece_history[0] >= 0.0);
}

#[test]
fn test_per_class_ece_calculation() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    // Create data with different calibration per class
    let mut logits_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..100 {
        let class = i % 5;

        // Different confidence per class
        let confidence = match class {
            0 => 5.0,
            1 => 3.0,
            2 => 1.0,
            3 => 0.5,
            4 => 0.1,
            _ => 1.0,
        };

        let mut logit_row = vec![-1.0; 5];
        logit_row[class] = confidence;
        logits_data.extend_from_slice(&logit_row);

        let mut target_row = vec![0.0; 5];
        target_row[class] = 1.0;
        targets_data.extend_from_slice(&target_row);
    }

    let logits = Array2::from_shape_vec((100, 5), logits_data).unwrap();
    let targets = Array2::from_shape_vec((100, 5), targets_data).unwrap();

    temp_scaling
        .optimize_temperatures(&logits, &targets)
        .unwrap();

    // Per-class ECE should be calculated
    let per_class_ece = temp_scaling.get_per_class_ece();

    for &ece in &per_class_ece {
        assert!(ece >= 0.0);
    }
}
