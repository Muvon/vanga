//! Integration test for bias correction system
//!
//! Tests the complete bias correction pipeline from configuration loading
//! through training calibration to inference application.

#[cfg(test)]
mod tests {
    use crate::model::bias_correction::{BiasCorrection, LinearBiasCorrector};
    use crate::model::lstm::{LSTMConfig, LSTMModel};
    use ndarray::Array2;

    #[test]
    fn test_bias_correction_integration() {
        // Create bias correction config
        let bias_config = BiasCorrection {
            enabled: true,
            min_samples: 10, // Lower for testing
            ..Default::default()
        };

        // Create LSTM model with bias correction
        let lstm_config = LSTMConfig {
            input_size: 10,
            hidden_sizes: vec![32],
            output_size: 5,
            sequence_length: 20,
            learning_rate: 0.001,
            num_layers: 1,
        };

        let model = LSTMModel::new_with_bias_config(lstm_config, bias_config).unwrap();

        // Verify bias corrector is initialized
        assert!(model.bias_corrector.is_some());
        let corrector = model.bias_corrector.as_ref().unwrap();
        assert!(corrector.config.enabled);
        assert!(!corrector.is_calibrated); // Not calibrated yet

        println!("✅ Bias correction integration test passed");
    }

    #[test]
    fn test_linear_bias_corrector_calibration() {
        let mut corrector = LinearBiasCorrector::new(BiasCorrection {
            min_samples: 5, // Lower for testing
            ..Default::default()
        });

        // Create mock validation data (biased toward class 0)
        let val_predictions = Array2::from_shape_vec(
            (10, 5),
            vec![
                0.8, 0.1, 0.05, 0.03, 0.02, // Heavily biased toward class 0
                0.7, 0.15, 0.08, 0.04, 0.03, 0.75, 0.12, 0.06, 0.04, 0.03, 0.8, 0.1, 0.05, 0.03,
                0.02, 0.7, 0.15, 0.08, 0.04, 0.03, 0.75, 0.12, 0.06, 0.04, 0.03, 0.8, 0.1, 0.05,
                0.03, 0.02, 0.7, 0.15, 0.08, 0.04, 0.03, 0.75, 0.12, 0.06, 0.04, 0.03, 0.8, 0.1,
                0.05, 0.03, 0.02,
            ],
        )
        .unwrap();

        // Create balanced target data
        let val_targets = Array2::from_shape_vec(
            (10, 5),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, // Class 0
                0.0, 1.0, 0.0, 0.0, 0.0, // Class 1
                0.0, 0.0, 1.0, 0.0, 0.0, // Class 2
                0.0, 0.0, 0.0, 1.0, 0.0, // Class 3
                0.0, 0.0, 0.0, 0.0, 1.0, // Class 4
                1.0, 0.0, 0.0, 0.0, 0.0, // Class 0
                0.0, 1.0, 0.0, 0.0, 0.0, // Class 1
                0.0, 0.0, 1.0, 0.0, 0.0, // Class 2
                0.0, 0.0, 0.0, 1.0, 0.0, // Class 3
                0.0, 0.0, 0.0, 0.0, 1.0, // Class 4
            ],
        )
        .unwrap();

        // Calibrate bias correction
        let result = corrector.calibrate_from_validation(&val_predictions, &val_targets);
        assert!(result.is_ok());
        assert!(corrector.is_calibrated);

        // Check that correction factors were calculated
        // Class 0 should be reduced (factor < 1.0), others should be increased (factor > 1.0)
        assert!(
            corrector.class_bias_factors[0] < 1.0,
            "Class 0 factor should be < 1.0, got {}",
            corrector.class_bias_factors[0]
        );
        assert!(
            corrector.class_bias_factors[1] > 1.0,
            "Class 1 factor should be > 1.0, got {}",
            corrector.class_bias_factors[1]
        );

        println!("✅ Linear bias corrector calibration test passed");
        println!("📊 Correction factors: {:?}", corrector.class_bias_factors);
    }

    #[test]
    fn test_bias_correction_application() {
        let mut corrector = LinearBiasCorrector::new(BiasCorrection {
            min_samples: 5,
            ..Default::default()
        });

        // Set up calibrated corrector
        corrector.class_bias_factors = [0.5, 1.0, 1.0, 1.0, 2.0]; // Reduce class 0, increase class 4
        corrector.is_calibrated = true;

        // Create test predictions
        let raw_predictions = Array2::from_shape_vec(
            (2, 5),
            vec![
                0.4, 0.3, 0.2, 0.05, 0.05, // First sample
                0.5, 0.25, 0.15, 0.05, 0.05, // Second sample
            ],
        )
        .unwrap();

        // Apply correction
        let corrected = corrector.apply_correction(&raw_predictions).unwrap();

        // Check that probabilities still sum to 1.0
        for row in corrected.outer_iter() {
            let sum: f64 = row.sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Probabilities should sum to 1.0, got {}",
                sum
            );
        }

        // Check that class 0 was reduced and class 4 was increased
        assert!(
            corrected[[0, 0]] < raw_predictions[[0, 0]],
            "Class 0 should be reduced"
        );
        assert!(
            corrected[[0, 4]] > raw_predictions[[0, 4]],
            "Class 4 should be increased"
        );

        println!("✅ Bias correction application test passed");
        println!("📊 Original: {:?}", raw_predictions.row(0));
        println!("📊 Corrected: {:?}", corrected.row(0));
    }

    #[test]
    fn test_bias_correction_dimension_validation() {
        let mut corrector = LinearBiasCorrector::new(BiasCorrection {
            min_samples: 5,
            ..Default::default()
        });

        // Test with wrong number of classes in predictions
        let wrong_predictions = Array2::from_shape_vec(
            (10, 3), // Only 3 classes instead of 5
            vec![
                0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5,
                0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2,
            ],
        )
        .unwrap();

        let correct_targets = Array2::from_shape_vec(
            (10, 5),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0,
            ],
        )
        .unwrap();

        // Should fail with dimension error
        let result = corrector.calibrate_from_validation(&wrong_predictions, &correct_targets);
        assert!(result.is_err());

        // Test with wrong number of classes in targets
        let correct_predictions = Array2::from_shape_vec(
            (10, 5),
            vec![
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                0.2, 0.2,
            ],
        )
        .unwrap();

        let wrong_targets = Array2::from_shape_vec(
            (10, 3), // Only 3 classes instead of 5
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
            ],
        )
        .unwrap();

        // Should fail with dimension error
        let result = corrector.calibrate_from_validation(&correct_predictions, &wrong_targets);
        assert!(result.is_err());

        // Test apply_correction with wrong dimensions
        corrector.is_calibrated = true; // Mark as calibrated for testing
        let wrong_pred_for_apply = Array2::from_shape_vec(
            (5, 3), // Only 3 classes instead of 5
            vec![
                0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2,
            ],
        )
        .unwrap();

        let result = corrector.apply_correction(&wrong_pred_for_apply);
        assert!(result.is_err());

        println!("✅ Dimension validation test passed");
    }
}
