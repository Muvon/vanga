use crate::model::bias_correction::*;
use ndarray::Array2;

#[test]
fn test_bias_corrector_default_creation() {
    let corrector = LinearBiasCorrector::default();

    assert!(corrector.config.enabled);
    assert!(!corrector.is_calibrated);
    assert_eq!(corrector.class_bias_factors, [1.0; 5]);
    assert_eq!(corrector.confidence_scaling, 1.0);
    assert!(corrector.validation_stats.is_none());
}

#[test]
fn test_bias_corrector_with_config() {
    let mut config = BiasCorrection::default();
    config.enabled = false;
    config.smoothing_factor = 0.2;
    config.correction_bounds = [0.3, 3.0];

    let corrector = LinearBiasCorrector::new(config.clone());

    assert!(!corrector.config.enabled);
    assert_eq!(corrector.config.smoothing_factor, 0.2);
    assert_eq!(corrector.config.correction_bounds, [0.3, 3.0]);
}

#[test]
fn test_probability_renormalization() {
    let config = BiasCorrection::default();
    let corrector = LinearBiasCorrector::new(config);

    // Create predictions that don't sum to 1.0
    let mut predictions = Array2::from_shape_vec(
        (3, 5),
        vec![
            0.1, 0.2, 0.3, 0.4, 0.5, // Sum = 1.5
            0.2, 0.4, 0.6, 0.8, 1.0, // Sum = 3.0
            0.05, 0.05, 0.05, 0.05, 0.05, // Sum = 0.25
        ],
    )
    .unwrap();

    corrector
        .renormalize_probabilities(&mut predictions)
        .unwrap();

    // Check that each row sums to approximately 1.0
    for (i, row) in predictions.axis_iter(ndarray::Axis(0)).enumerate() {
        let sum: f64 = row.sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Row {} sum: {} (should be 1.0)",
            i,
            sum
        );
    }
}

#[test]
fn test_calibration_with_insufficient_samples() {
    let mut config = BiasCorrection::default();
    config.min_samples = 100;

    let mut corrector = LinearBiasCorrector::new(config);

    // Create small validation set (less than min_samples)
    let predictions = Array2::from_shape_vec(
        (10, 5),
        vec![0.2; 50], // All equal probabilities
    )
    .unwrap();

    let targets = Array2::from_shape_vec(
        (10, 5),
        vec![0.2; 50], // All equal probabilities
    )
    .unwrap();

    // Should succeed but not calibrate due to insufficient samples
    let result = corrector.calibrate_from_validation(&predictions, &targets);
    assert!(result.is_ok());
    assert!(!corrector.is_calibrated);
}

#[test]
fn test_calibration_with_sufficient_samples() {
    let mut config = BiasCorrection::default();
    config.min_samples = 10; // Lower threshold for testing

    let mut corrector = LinearBiasCorrector::new(config);

    // Create biased predictions (model over-predicts class 0, under-predicts class 4)
    let mut pred_data = vec![];
    let mut target_data = vec![];

    for _ in 0..20 {
        // Biased predictions: favor class 0
        pred_data.extend_from_slice(&[0.5, 0.2, 0.15, 0.1, 0.05]);
        // Balanced targets
        target_data.extend_from_slice(&[0.2, 0.2, 0.2, 0.2, 0.2]);
    }

    let predictions = Array2::from_shape_vec((20, 5), pred_data).unwrap();
    let targets = Array2::from_shape_vec((20, 5), target_data).unwrap();

    let result = corrector.calibrate_from_validation(&predictions, &targets);
    assert!(result.is_ok());
    assert!(corrector.is_calibrated);

    // Check that bias factors are calculated
    // Class 0: predicted=0.5, actual=0.2, factor should be 0.2/0.5 = 0.4
    // Class 4: predicted=0.05, actual=0.2, factor should be 0.2/0.05 = 4.0
    assert!(corrector.class_bias_factors[0] < 1.0); // Should reduce class 0
    assert!(corrector.class_bias_factors[4] > 1.0); // Should increase class 4
}

#[test]
fn test_apply_correction_when_disabled() {
    let mut config = BiasCorrection::default();
    config.enabled = false;

    let corrector = LinearBiasCorrector::new(config);

    let predictions = Array2::from_shape_vec(
        (2, 5),
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    .unwrap();

    let corrected = corrector.apply_correction(&predictions).unwrap();

    // Should return unchanged predictions when disabled
    assert_eq!(predictions, corrected);
}

#[test]
fn test_apply_correction_when_not_calibrated() {
    let config = BiasCorrection::default(); // enabled but not calibrated
    let corrector = LinearBiasCorrector::new(config);

    let predictions = Array2::from_shape_vec(
        (2, 5),
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    .unwrap();

    let corrected = corrector.apply_correction(&predictions).unwrap();

    // Should return unchanged predictions when not calibrated
    assert_eq!(predictions, corrected);
}

#[test]
fn test_apply_correction_with_calibration() {
    let mut config = BiasCorrection::default();
    config.min_samples = 5;

    let mut corrector = LinearBiasCorrector::new(config);

    // Calibrate with biased data
    let cal_predictions =
        Array2::from_shape_vec((10, 5), vec![0.4, 0.3, 0.2, 0.1, 0.0; 10].concat()).unwrap();

    let cal_targets =
        Array2::from_shape_vec((10, 5), vec![0.2, 0.2, 0.2, 0.2, 0.2; 10].concat()).unwrap();

    corrector
        .calibrate_from_validation(&cal_predictions, &cal_targets)
        .unwrap();
    assert!(corrector.is_calibrated);

    // Apply correction to new predictions
    let test_predictions = Array2::from_shape_vec(
        (2, 5),
        vec![0.4, 0.3, 0.2, 0.1, 0.0, 0.5, 0.25, 0.15, 0.1, 0.0],
    )
    .unwrap();

    let corrected = corrector.apply_correction(&test_predictions).unwrap();

    // Check that probabilities still sum to 1.0 after correction
    for row in corrected.axis_iter(ndarray::Axis(0)) {
        let sum: f64 = row.sum();
        assert!((sum - 1.0).abs() < 1e-10, "Row sum: {}", sum);
    }

    // Corrected predictions should be different from original
    assert_ne!(test_predictions, corrected);
}

#[test]
fn test_correction_bounds() {
    let mut config = BiasCorrection::default();
    config.min_samples = 5;
    config.correction_bounds = [0.5, 2.0]; // Limit correction factors

    let mut corrector = LinearBiasCorrector::new(config);

    // Create extreme bias scenario
    let cal_predictions = Array2::from_shape_vec(
        (10, 5),
        vec![0.9, 0.025, 0.025, 0.025, 0.025; 10].concat(), // Extreme bias toward class 0
    )
    .unwrap();

    let cal_targets = Array2::from_shape_vec(
        (10, 5),
        vec![0.2, 0.2, 0.2, 0.2, 0.2; 10].concat(), // Balanced targets
    )
    .unwrap();

    corrector
        .calibrate_from_validation(&cal_predictions, &cal_targets)
        .unwrap();

    // Check that correction factors are within bounds
    for &factor in &corrector.class_bias_factors {
        assert!(factor >= 0.5, "Factor {} below lower bound", factor);
        assert!(factor <= 2.0, "Factor {} above upper bound", factor);
    }
}

#[test]
fn test_correction_metrics() {
    let mut config = BiasCorrection::default();
    config.min_samples = 5;

    let mut corrector = LinearBiasCorrector::new(config);

    // Before calibration, no metrics should be available
    assert!(corrector.get_correction_metrics().is_none());

    // Calibrate
    let predictions =
        Array2::from_shape_vec((10, 5), vec![0.3, 0.3, 0.2, 0.15, 0.05; 10].concat()).unwrap();

    let targets =
        Array2::from_shape_vec((10, 5), vec![0.2, 0.2, 0.2, 0.2, 0.2; 10].concat()).unwrap();

    corrector
        .calibrate_from_validation(&predictions, &targets)
        .unwrap();

    // After calibration, metrics should be available
    let metrics = corrector.get_correction_metrics();
    assert!(metrics.is_some());

    let metrics = metrics.unwrap();
    assert!(metrics.is_calibrated);
    assert_eq!(metrics.total_samples, 10);
    assert!(metrics.validation_accuracy >= 0.0 && metrics.validation_accuracy <= 1.0);
    assert!(metrics.correction_strength >= 0.0);
}

#[test]
fn test_reset_calibration() {
    let mut config = BiasCorrection::default();
    config.min_samples = 5;

    let mut corrector = LinearBiasCorrector::new(config);

    // Calibrate
    let predictions =
        Array2::from_shape_vec((10, 5), vec![0.3, 0.3, 0.2, 0.15, 0.05; 10].concat()).unwrap();

    let targets =
        Array2::from_shape_vec((10, 5), vec![0.2, 0.2, 0.2, 0.2, 0.2; 10].concat()).unwrap();

    corrector
        .calibrate_from_validation(&predictions, &targets)
        .unwrap();
    assert!(corrector.is_calibrated);

    // Reset calibration
    corrector.reset_calibration();

    assert!(!corrector.is_calibrated);
    assert_eq!(corrector.class_bias_factors, [1.0; 5]);
    assert_eq!(corrector.confidence_scaling, 1.0);
    assert!(corrector.validation_stats.is_none());
}

#[test]
fn test_is_active() {
    let mut config = BiasCorrection::default();
    config.min_samples = 5;

    let mut corrector = LinearBiasCorrector::new(config);

    // Not active when not calibrated
    assert!(!corrector.is_active());

    // Calibrate
    let predictions = Array2::from_shape_vec((10, 5), vec![0.2; 50]).unwrap();

    let targets = Array2::from_shape_vec((10, 5), vec![0.2; 50]).unwrap();

    corrector
        .calibrate_from_validation(&predictions, &targets)
        .unwrap();

    // Should be active when enabled and calibrated
    assert!(corrector.is_active());

    // Disable and check
    corrector.config.enabled = false;
    assert!(!corrector.is_active());
}

#[tokio::test]
async fn test_confidence_calculation() {
    let config = BiasCorrection::default();
    let corrector = LinearBiasCorrector::new(config);

    // Create predictions with known confidence levels
    let predictions = Array2::from_shape_vec(
        (3, 5),
        vec![
            0.8, 0.05, 0.05, 0.05, 0.05, // High confidence (0.8)
            0.4, 0.3, 0.15, 0.1, 0.05, // Medium confidence (0.4)
            0.21, 0.20, 0.20, 0.20, 0.19, // Low confidence (0.21)
        ],
    )
    .unwrap();

    let avg_confidence = corrector
        .calculate_average_confidence(&predictions)
        .unwrap();

    // Expected average: (0.8 + 0.4 + 0.21) / 3 = 0.47
    let expected = (0.8 + 0.4 + 0.21) / 3.0;
    assert!((avg_confidence - expected).abs() < 1e-10);
}

#[tokio::test]
async fn test_accuracy_calculation() {
    let config = BiasCorrection::default();
    let corrector = LinearBiasCorrector::new(config);

    // Create predictions and targets where 2 out of 3 are correct
    let predictions = Array2::from_shape_vec(
        (3, 5),
        vec![
            0.8, 0.05, 0.05, 0.05, 0.05, // Predicts class 0
            0.1, 0.1, 0.6, 0.1, 0.1, // Predicts class 2
            0.1, 0.1, 0.1, 0.1, 0.6, // Predicts class 4
        ],
    )
    .unwrap();

    let targets = Array2::from_shape_vec(
        (3, 5),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, // Actual class 0 (correct)
            0.0, 1.0, 0.0, 0.0, 0.0, // Actual class 1 (incorrect, predicted 2)
            0.0, 0.0, 0.0, 0.0, 1.0, // Actual class 4 (correct)
        ],
    )
    .unwrap();

    let accuracy = corrector
        .calculate_accuracy(&predictions, &targets)
        .unwrap();

    // Expected accuracy: 2/3 = 0.6667
    assert!((accuracy - 2.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_logit_adjustment_bounds() {
    // Create corrector with extreme bias factors to test bounds
    let corrector = LinearBiasCorrector {
        class_bias_factors: [0.1, 0.5, 1.0, 2.0, 5.0], // Extreme values
        is_calibrated: true,
        ..Default::default()
    };

    // Calculate adjustments with high strength
    let adjustments = corrector.calculate_ordinal_aware_adjustments(1.0);

    // All adjustments should be bounded to [-0.2, 0.2]
    for &adj in &adjustments {
        assert!(
            (-0.2..=0.2).contains(&adj),
            "Adjustment {} is outside bounds [-0.2, 0.2]",
            adj
        );
    }
}

#[test]
fn test_confidence_scaling_bounds() {
    let config = BiasCorrection::default();
    let mut corrector = LinearBiasCorrector::new(config);

    // Create mock validation data
    let predictions = Array2::from_shape_vec(
        (10, 5),
        vec![
            0.9, 0.025, 0.025, 0.025, 0.025, // High confidence
            0.2, 0.2, 0.2, 0.2, 0.2, // Low confidence
            0.8, 0.05, 0.05, 0.05, 0.05, 0.7, 0.075, 0.075, 0.075, 0.075, 0.6, 0.1, 0.1, 0.1, 0.1,
            0.5, 0.125, 0.125, 0.125, 0.125, 0.4, 0.15, 0.15, 0.15, 0.15, 0.3, 0.175, 0.175, 0.175,
            0.175, 0.25, 0.1875, 0.1875, 0.1875, 0.1875, 0.2, 0.2, 0.2, 0.2, 0.2,
        ],
    )
    .unwrap();

    let targets = Array2::from_shape_vec(
        (10, 5),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();

    // Calibrate
    corrector
        .calibrate_from_validation(&predictions, &targets)
        .unwrap();

    // Confidence scaling should be bounded to [0.5, 2.0]
    assert!(
        corrector.confidence_scaling >= 0.5 && corrector.confidence_scaling <= 2.0,
        "Confidence scaling {} is outside bounds [0.5, 2.0]",
        corrector.confidence_scaling
    );
}
