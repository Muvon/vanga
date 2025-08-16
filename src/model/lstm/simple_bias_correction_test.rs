use crate::model::lstm::LSTMModel;
use ndarray::Array2;

#[test]
fn test_simple_bias_correction_calibration() {
    let mut model = LSTMModel::new(crate::model::lstm::LSTMConfig::default()).unwrap();

    // Create biased validation predictions (favor class 0)
    let val_predictions =
        Array2::from_shape_vec((100, 5), vec![0.5, 0.2, 0.15, 0.1, 0.05; 100].concat()).unwrap();

    // Create balanced validation targets
    let val_targets =
        Array2::from_shape_vec((100, 5), vec![0.2, 0.2, 0.2, 0.2, 0.2; 100].concat()).unwrap();

    // Calibrate bias correction
    let result = model.calibrate_simple_bias_correction(&val_predictions, &val_targets);
    assert!(result.is_ok());

    // Check that correction factors were calculated
    assert!(model.bias_correction_factors.is_some());
    let factors = model.bias_correction_factors.unwrap();

    // Class 0 should be reduced (factor < 1.0), class 4 should be increased (factor > 1.0)
    assert!(
        factors[0] < 1.0,
        "Class 0 factor should be < 1.0, got {}",
        factors[0]
    );
    assert!(
        factors[4] > 1.0,
        "Class 4 factor should be > 1.0, got {}",
        factors[4]
    );
}

#[test]
fn test_simple_bias_correction_application() {
    let model = LSTMModel::new(crate::model::lstm::LSTMConfig::default()).unwrap();

    let correction_factors = [0.5, 1.0, 1.0, 1.0, 2.0]; // Reduce class 0, increase class 4

    let mut predictions = Array2::from_shape_vec(
        (2, 5),
        vec![0.4, 0.3, 0.2, 0.1, 0.0, 0.5, 0.25, 0.15, 0.1, 0.0],
    )
    .unwrap();

    let result = model.apply_simple_bias_correction(&mut predictions, &correction_factors);
    assert!(result.is_ok());

    // Check that probabilities still sum to 1.0
    for row in predictions.axis_iter(ndarray::Axis(0)) {
        let sum: f64 = row.sum();
        assert!((sum - 1.0).abs() < 1e-10, "Row sum: {}", sum);
    }

    // Check that class 0 was reduced and class 4 was increased
    assert!(predictions[[0, 0]] < 0.4, "Class 0 should be reduced");
    assert!(predictions[[0, 4]] > 0.0, "Class 4 should be increased");
}

#[test]
fn test_insufficient_validation_samples() {
    let mut model = LSTMModel::new(crate::model::lstm::LSTMConfig::default()).unwrap();

    // Create small validation set (less than 50 samples)
    let val_predictions = Array2::from_shape_vec((10, 5), vec![0.2; 50]).unwrap();

    let val_targets = Array2::from_shape_vec((10, 5), vec![0.2; 50]).unwrap();

    // Should succeed but not calibrate
    let result = model.calibrate_simple_bias_correction(&val_predictions, &val_targets);
    assert!(result.is_ok());
    assert!(model.bias_correction_factors.is_none());
}
