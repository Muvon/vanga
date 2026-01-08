use crate::model::calibration::temperature::*;
use ndarray::Array2;

#[test]
fn test_temperature_scaling_creation() {
    let temp_scaling = AdaptiveTemperatureScaling::new();

    assert_eq!(temp_scaling.temperature, 1.0);
    assert!(!temp_scaling.is_optimized);
    assert!(temp_scaling.ece_history.is_empty());
}

#[test]
fn test_temperature_scaling_default() {
    let temp_scaling = AdaptiveTemperatureScaling::default();

    assert_eq!(temp_scaling.temperature, 1.0);
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
fn test_optimize_temperature_nll_minimization() {
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

    // Temperature should be adjusted (not 1.0)
    assert!(
        (temp_scaling.temperature - 1.0).abs() > 0.01,
        "Temperature should be optimized, not 1.0"
    );

    // Temperature should be in extended range [0.05, 5.0]
    assert!(
        temp_scaling.temperature >= 0.05 && temp_scaling.temperature <= 5.0,
        "Temperature should be in range [0.05, 5.0], got {}",
        temp_scaling.temperature
    );

    // Verify NLL improved (decreased)
    let predictions_initial = temp_scaling
        .apply_temperature_to_logits(&logits, 1.0)
        .unwrap();
    let nll_initial = temp_scaling
        .calculate_nll(&predictions_initial, &targets)
        .unwrap();

    let predictions_optimized = temp_scaling
        .apply_temperature_to_logits(&logits, temp_scaling.temperature)
        .unwrap();
    let nll_optimized = temp_scaling
        .calculate_nll(&predictions_optimized, &targets)
        .unwrap();

    assert!(
        nll_optimized <= nll_initial,
        "NLL should not increase after optimization: initial={}, optimized={}",
        nll_initial,
        nll_optimized
    );
}

#[test]
fn test_apply_temperature_to_logits() {
    let temp_scaling = AdaptiveTemperatureScaling::new();

    // Create some test logits
    let logits = Array2::from_shape_vec(
        (3, 5),
        vec![
            1.0, 0.5, 0.3, 0.2, 0.1, 2.0, 1.0, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.1, 0.05,
        ],
    )
    .unwrap();

    // Apply temperature scaling with T=2.0
    let predictions = temp_scaling
        .apply_temperature_to_logits(&logits, 2.0)
        .unwrap();

    // Check that predictions sum to 1.0 (valid probability distribution)
    for i in 0..3 {
        let sum: f64 = predictions.row(i).iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Predictions should sum to 1.0, got {} for row {}",
            sum,
            i
        );

        // All probabilities should be positive
        for j in 0..5 {
            assert!(
                predictions[[i, j]] > 0.0,
                "Probability should be positive, got {} at [{}, {}]",
                predictions[[i, j]],
                i,
                j
            );
        }
    }
}

#[test]
fn test_temperature_temperatures_method() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    // Default should return [1.0; 5]
    assert_eq!(temp_scaling.temperatures(), [1.0; 5]);

    // After setting temperature, should return [T; 5]
    temp_scaling.temperature = 1.5;
    assert_eq!(temp_scaling.temperatures(), [1.5; 5]);
}

#[test]
fn test_reset() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    // Set some values
    temp_scaling.temperature = 2.0;
    temp_scaling.is_optimized = true;
    temp_scaling.ece_history.push(0.1);
    temp_scaling.per_class_ece = [0.1; 5];

    // Reset
    temp_scaling.reset();

    // Verify reset values
    assert_eq!(temp_scaling.temperature, 1.0);
    assert!(!temp_scaling.is_optimized);
    assert!(temp_scaling.ece_history.is_empty());
    assert_eq!(temp_scaling.per_class_ece, [0.0; 5]);
}

#[test]
fn test_getters() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    assert_eq!(temp_scaling.get_temperature(), 1.0);
    assert_eq!(temp_scaling.get_per_class_ece(), [0.0; 5]);
    assert!(temp_scaling.get_latest_ece().is_none());

    temp_scaling.per_class_ece = [0.1, 0.2, 0.3, 0.4, 0.5];
    temp_scaling.ece_history.push(0.25);

    assert_eq!(temp_scaling.get_temperature(), 1.0);
    assert_eq!(temp_scaling.get_per_class_ece(), [0.1, 0.2, 0.3, 0.4, 0.5]);
    assert_eq!(temp_scaling.get_latest_ece(), Some(0.25));
}

#[test]
fn test_apply_to_logits_without_optimization() {
    let temp_scaling = AdaptiveTemperatureScaling::new();

    let logits = Array2::from_shape_vec(
        (2, 5),
        vec![1.0, 0.5, 0.3, 0.2, 0.1, 1.0, 0.5, 0.3, 0.2, 0.1],
    )
    .unwrap();

    // Without optimization, should return original logits
    let result = temp_scaling.apply_to_logits(&logits).unwrap();
    assert_eq!(result, logits);
}

#[test]
fn test_high_temperature_softens_predictions() {
    let temp_scaling = AdaptiveTemperatureScaling::new();

    // Create highly confident predictions
    let logits = Array2::from_shape_vec((1, 5), vec![10.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

    // Apply with high temperature (T=5.0)
    let soft_predictions = temp_scaling
        .apply_temperature_to_logits(&logits, 5.0)
        .unwrap();

    // Apply with temperature T=1.0
    let sharp_predictions = temp_scaling
        .apply_temperature_to_logits(&logits, 1.0)
        .unwrap();

    // High temperature should produce softer predictions
    let soft_max_prob = soft_predictions[[0, 0]];
    let sharp_max_prob = sharp_predictions[[0, 0]];

    assert!(
        soft_max_prob < sharp_max_prob,
        "High temperature should produce softer predictions: soft={}, sharp={}",
        soft_max_prob,
        sharp_max_prob
    );

    // But first class should still be the most probable
    assert!(
        soft_predictions[[0, 0]] > soft_predictions[[0, 1]]
            && soft_predictions[[0, 0]] > soft_predictions[[0, 2]]
            && soft_predictions[[0, 0]] > soft_predictions[[0, 3]]
            && soft_predictions[[0, 0]] > soft_predictions[[0, 4]],
        "First class should still be most probable"
    );
}

#[test]
fn test_temperature_no_improvement_stays_at_1() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    // Create well-calibrated predictions (not overconfident, not underconfident)
    // These are already close to optimal, so temperature optimization shouldn't improve much
    let logits = Array2::from_shape_vec(
        (30, 5),
        (0..30)
            .flat_map(|i| {
                let class = i % 5;
                let mut row = vec![0.0; 5];
                row[class] = 1.5; // Moderate confidence
                row
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let targets = Array2::from_shape_vec(
        (30, 5),
        (0..30)
            .flat_map(|i| {
                let class = i % 5;
                let mut row = vec![0.0; 5];
                row[class] = 1.0;
                row
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let result = temp_scaling.optimize_temperatures(&logits, &targets);
    assert!(result.is_ok());

    // If NLL increased at the best found temperature, should stay at 1.0
    // Otherwise should use the optimized temperature
    // In either case, we should verify the logic is consistent
    if temp_scaling.temperature != 1.0 {
        // Verify NLL actually improved
        let predictions_optimized = temp_scaling
            .apply_temperature_to_logits(&logits, temp_scaling.temperature)
            .unwrap();
        let nll_optimized = temp_scaling
            .calculate_nll(&predictions_optimized, &targets)
            .unwrap();

        let predictions_baseline = temp_scaling
            .apply_temperature_to_logits(&logits, 1.0)
            .unwrap();
        let nll_baseline = temp_scaling
            .calculate_nll(&predictions_baseline, &targets)
            .unwrap();

        assert!(
            nll_optimized < nll_baseline,
            "If T != 1.0, NLL should improve: optimized={}, baseline={}",
            nll_optimized,
            nll_baseline
        );
    }
}

#[test]
fn test_optimize_temperature_no_improvement() {
    let mut temp_scaling = AdaptiveTemperatureScaling::new();

    // Create well-calibrated predictions where no temperature improvement is possible
    // Using perfectly balanced logits that already give good probabilities
    let mut logits_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..100 {
        let class = i % 5;

        // Well-calibrated: moderate logits with some noise
        let mut logit_row = vec![-0.5; 5];
        logit_row[class] = 1.5;
        // Add small random noise to other classes
        for j in 0..5 {
            if j != class {
                logit_row[j] = -0.5 + (j as f64 * 0.05);
            }
        }
        logits_data.extend_from_slice(&logit_row);

        // One-hot targets
        let mut target_row = vec![0.0; 5];
        target_row[class] = 1.0;
        targets_data.extend_from_slice(&target_row);
    }

    let logits = Array2::from_shape_vec((100, 5), logits_data).unwrap();
    let targets = Array2::from_shape_vec((100, 5), targets_data).unwrap();

    // Get baseline NLL at T=1.0
    let predictions_baseline = temp_scaling
        .apply_temperature_to_logits(&logits, 1.0)
        .unwrap();
    let nll_baseline = temp_scaling
        .calculate_nll(&predictions_baseline, &targets)
        .unwrap();

    let result = temp_scaling.optimize_temperatures(&logits, &targets);
    assert!(result.is_ok());
    assert!(temp_scaling.is_optimized);

    // Get optimized NLL
    let predictions_optimized = temp_scaling
        .apply_temperature_to_logits(&logits, temp_scaling.temperature)
        .unwrap();
    let nll_optimized = temp_scaling
        .calculate_nll(&predictions_optimized, &targets)
        .unwrap();

    // Verify temperature is within valid range
    assert!(
        temp_scaling.temperature >= 0.05 && temp_scaling.temperature <= 5.0,
        "Temperature should be in [0.05, 5.0], got {}",
        temp_scaling.temperature
    );

    // If NLL increased, temperature should be 1.0
    if nll_optimized >= nll_baseline {
        assert_eq!(
            temp_scaling.temperature, 1.0,
            "If NLL increased, should keep T=1.0"
        );
    }
}
