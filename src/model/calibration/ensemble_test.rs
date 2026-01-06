use crate::model::calibration::ensemble::*;
use ndarray::{Array2, Array3};

#[test]
fn test_ensemble_calibrator_creation() {
    let calibrator = EnsembleCalibrator::new();

    assert!(!calibrator.is_calibrated);
    assert!(calibrator.ece_history.is_empty());
    assert!(calibrator.per_class_ece_history.is_empty());
    assert!(calibrator.reliability_diagram.is_none());
}

#[test]
fn test_ensemble_calibrator_default() {
    let calibrator = EnsembleCalibrator::default();

    assert!(!calibrator.is_calibrated);
    assert_eq!(calibrator.temperature_scaling.temperatures, [1.0; 5]);
    assert_eq!(calibrator.label_smoothing.epsilons, [0.0; 5]);
    assert!(!calibrator.mixup.is_calibrated);
}

#[test]
fn test_calibration_with_empty_data() {
    let mut calibrator = EnsembleCalibrator::new();

    let logits = Array2::zeros((0, 5));
    let targets = Array2::zeros((0, 5));

    let result = calibrator.calibrate_from_validation(&logits, &targets);
    assert!(result.is_ok());
    assert!(!calibrator.is_calibrated);
}

#[test]
fn test_calibration_with_invalid_dimensions() {
    let mut calibrator = EnsembleCalibrator::new();

    // Wrong number of classes
    let logits = Array2::zeros((10, 3));
    let targets = Array2::zeros((10, 5));

    let result = calibrator.calibrate_from_validation(&logits, &targets);
    assert!(result.is_err());
}

#[test]
fn test_calibration_with_mismatched_samples() {
    let mut calibrator = EnsembleCalibrator::new();

    let logits = Array2::zeros((10, 5));
    let targets = Array2::zeros((15, 5));

    let result = calibrator.calibrate_from_validation(&logits, &targets);
    assert!(result.is_err());
}

#[test]
fn test_full_calibration_pipeline() {
    let mut calibrator = EnsembleCalibrator::new();

    // Create realistic validation data
    let mut logits_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..100 {
        let class = i % 5;

        // Create logits with some noise
        let mut logit_row = vec![0.0; 5];
        logit_row[class] = 2.0 + (i as f64 * 0.01);
        for j in 0..5 {
            if j != class {
                logit_row[j] = -0.5 + (j as f64 * 0.1);
            }
        }
        logits_data.extend_from_slice(&logit_row);

        // Create one-hot targets
        let mut target_row = vec![0.0; 5];
        target_row[class] = 1.0;
        targets_data.extend_from_slice(&target_row);
    }

    let logits = Array2::from_shape_vec((100, 5), logits_data).unwrap();
    let targets = Array2::from_shape_vec((100, 5), targets_data).unwrap();

    let result = calibrator.calibrate_from_validation(&logits, &targets);
    assert!(result.is_ok());
    assert!(calibrator.is_calibrated);

    // Check that all components are calibrated
    assert!(calibrator.temperature_scaling.is_optimized);
    assert!(calibrator.label_smoothing.is_calibrated);
    assert!(calibrator.mixup.is_calibrated);

    // Check ECE history
    assert!(!calibrator.ece_history.is_empty());
    assert!(!calibrator.per_class_ece_history.is_empty());

    // Check reliability diagram
    assert!(calibrator.reliability_diagram.is_some());
}

#[test]
fn test_apply_to_logits_before_calibration() {
    let calibrator = EnsembleCalibrator::new();

    let logits = Array2::from_shape_vec(
        (2, 5),
        vec![1.0, 0.5, 0.0, -0.5, -1.0, 2.0, 1.0, 0.0, -1.0, -2.0],
    )
    .unwrap();

    let result = calibrator.apply_to_logits(&logits).unwrap();

    // Should return unchanged logits when not calibrated
    assert_eq!(logits, result);
}

#[test]
fn test_apply_label_smoothing_before_calibration() {
    let calibrator = EnsembleCalibrator::new();

    let targets = Array2::from_shape_vec(
        (2, 5),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();

    let result = calibrator.apply_label_smoothing(&targets).unwrap();

    // Should return unchanged targets when not calibrated
    assert_eq!(targets, result);
}

#[test]
fn test_apply_mixup_before_calibration() {
    let calibrator = EnsembleCalibrator::new();

    let sequences = Array3::zeros((4, 10, 20));
    let targets = Array2::zeros((4, 5));
    let mut rng_state = 12345u64;

    let result = calibrator
        .apply_mixup(&sequences, &targets, &mut rng_state)
        .unwrap();

    // Should return unchanged data when not calibrated
    assert_eq!(sequences, result.0);
    assert_eq!(targets, result.1);
}

#[test]
fn test_calibration_metrics() {
    let mut calibrator = EnsembleCalibrator::new();

    // Create simple validation data
    let logits = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![2.0, 0.0, 0.0, 0.0, 0.0])
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

    calibrator
        .calibrate_from_validation(&logits, &targets)
        .unwrap();

    let metrics = calibrator.get_calibration_metrics();

    assert!(metrics.is_calibrated);
    assert!(metrics.overall_ece >= 0.0);
    assert_eq!(metrics.temperatures.len(), 5);
    assert_eq!(metrics.label_smoothing_epsilons.len(), 5);
    assert!(metrics.mixup_alpha >= 0.0);
}

#[test]
fn test_reset_calibration() {
    let mut calibrator = EnsembleCalibrator::new();

    // Calibrate first
    let logits = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![2.0, 0.0, 0.0, 0.0, 0.0])
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

    calibrator
        .calibrate_from_validation(&logits, &targets)
        .unwrap();
    assert!(calibrator.is_calibrated);

    // Reset
    calibrator.reset();

    assert!(!calibrator.is_calibrated);
    assert!(calibrator.ece_history.is_empty());
    assert!(calibrator.per_class_ece_history.is_empty());
    assert!(calibrator.reliability_diagram.is_none());
    assert_eq!(calibrator.temperature_scaling.temperatures, [1.0; 5]);
}

#[test]
fn test_is_active() {
    let mut calibrator = EnsembleCalibrator::new();

    assert!(!calibrator.is_active());

    // Calibrate
    let logits = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![2.0, 0.0, 0.0, 0.0, 0.0])
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

    calibrator
        .calibrate_from_validation(&logits, &targets)
        .unwrap();

    assert!(calibrator.is_active());
}

#[test]
fn test_temperature_scaling_improves_calibration() {
    let mut calibrator = EnsembleCalibrator::new();

    // Create overconfident predictions
    let mut logits_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..50 {
        let class = i % 5;

        // Very high logits (overconfident)
        let mut logit_row = vec![-5.0; 5];
        logit_row[class] = 5.0;
        logits_data.extend_from_slice(&logit_row);

        // One-hot targets
        let mut target_row = vec![0.0; 5];
        target_row[class] = 1.0;
        targets_data.extend_from_slice(&target_row);
    }

    let logits = Array2::from_shape_vec((50, 5), logits_data).unwrap();
    let targets = Array2::from_shape_vec((50, 5), targets_data).unwrap();

    // Calibrate
    calibrator
        .calibrate_from_validation(&logits, &targets)
        .unwrap();

    // Apply temperature scaling
    let calibrated_logits = calibrator.apply_to_logits(&logits).unwrap();

    // Temperature scaling should improve calibration for overconfident predictions
    assert!(calibrator.is_calibrated);

    // Calibrated logits should be different from original
    assert_ne!(logits, calibrated_logits);
}

#[test]
fn test_mixup_with_small_batch() {
    let mut calibrator = EnsembleCalibrator::new();

    // Calibrate first
    let logits = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![2.0, 0.0, 0.0, 0.0, 0.0])
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

    calibrator
        .calibrate_from_validation(&logits, &targets)
        .unwrap();

    // Try mixup with batch size 1 (should return unchanged)
    let sequences = Array3::zeros((1, 10, 20));
    let targets_small = Array2::zeros((1, 5));
    let mut rng_state = 12345u64;

    let result = calibrator
        .apply_mixup(&sequences, &targets_small, &mut rng_state)
        .unwrap();

    assert_eq!(sequences, result.0);
    assert_eq!(targets_small, result.1);
}

#[test]
fn test_calibration_metrics_summary() {
    let mut calibrator = EnsembleCalibrator::new();

    let logits = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![2.0, 0.0, 0.0, 0.0, 0.0])
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

    calibrator
        .calibrate_from_validation(&logits, &targets)
        .unwrap();

    let metrics = calibrator.get_calibration_metrics();
    let summary = metrics.summary();

    assert!(summary.contains("Calibration"));
    assert!(summary.contains("ECE"));
}

#[test]
fn test_concurrent_calibration_safety() {
    // Test that calibration is safe to call multiple times
    let mut calibrator = EnsembleCalibrator::new();

    let logits = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![2.0, 0.0, 0.0, 0.0, 0.0])
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

    // First calibration
    calibrator
        .calibrate_from_validation(&logits, &targets)
        .unwrap();

    // Second calibration (should overwrite)
    calibrator
        .calibrate_from_validation(&logits, &targets)
        .unwrap();

    // Both should succeed
    assert!(calibrator.is_calibrated);
    assert_eq!(calibrator.ece_history.len(), 2);
}

#[test]
fn test_per_class_ece_tracking() {
    let mut calibrator = EnsembleCalibrator::new();

    // Create data with varying per-class calibration
    let mut logits_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..100 {
        let class = i % 5;

        // Different confidence levels per class
        let confidence = match class {
            0 => 5.0, // Very confident
            1 => 3.0, // Confident
            2 => 1.0, // Moderate
            3 => 0.5, // Low
            4 => 0.1, // Very low
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

    calibrator
        .calibrate_from_validation(&logits, &targets)
        .unwrap();

    // Check that per-class ECE is tracked
    assert!(!calibrator.per_class_ece_history.is_empty());
    let per_class_ece = calibrator.per_class_ece_history.last().unwrap();

    // All classes should have ECE values
    for &ece in per_class_ece.iter() {
        assert!(ece >= 0.0);
    }
}
