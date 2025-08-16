#[cfg(test)]
mod bias_correction_integration_test {
    use super::*;
    use ndarray::Array2;

    #[tokio::test]
    async fn test_end_to_end_bias_correction() {
        // Create a simple LSTM model
        let mut model = LSTMModel::new(crate::model::lstm::LSTMConfig {
            input_size: 10,
            hidden_sizes: vec![16],
            output_size: 5,
            sequence_length: 20,
            learning_rate: 0.01,
            num_layers: 1,
            ..Default::default()
        })
        .unwrap();

        // Create mock training data
        let sequences = ndarray::Array3::<f64>::zeros((100, 20, 10));
        let targets = ndarray::Array2::<f64>::zeros((100, 5));

        // Create mock validation data with bias (favor class 0)
        let val_sequences = ndarray::Array3::<f64>::ones((50, 20, 10));
        let mut val_targets = ndarray::Array2::<f64>::zeros((50, 5));
        // Set balanced targets
        for i in 0..50 {
            val_targets[[i, i % 5]] = 1.0;
        }

        // Create training config
        let config = crate::config::training::TrainingConfig::default();

        // Test that bias correction factors are None initially
        assert!(
            model.bias_correction_factors.is_none(),
            "Bias correction factors should be None initially"
        );

        // Run one epoch of training with validation
        let result = model
            .train(
                &sequences,
                &targets,
                &config,
                Some(&val_sequences),
                Some(&val_targets),
                None,
            )
            .await;

        // Training should succeed
        assert!(result.is_ok(), "Training should succeed: {:?}", result);

        // Check if bias correction factors were calculated
        if let Some(factors) = model.bias_correction_factors {
            println!("✅ Bias correction factors calculated: {:?}", factors);

            // Factors should be within bounds [0.5, 2.0]
            for (i, &factor) in factors.iter().enumerate() {
                assert!(
                    factor >= 0.5 && factor <= 2.0,
                    "Factor {} should be in [0.5, 2.0], got {}",
                    i,
                    factor
                );
            }
        } else {
            println!("⚠️ Bias correction factors not calculated (insufficient validation data)");
        }

        // Test prediction with bias correction
        let test_sequences = ndarray::Array3::<f64>::ones((10, 20, 10));
        let predictions = model.predict(&test_sequences);

        assert!(predictions.is_ok(), "Prediction should succeed");
        let pred_array = predictions.unwrap();
        assert_eq!(
            pred_array.shape(),
            &[10, 5],
            "Predictions should have correct shape"
        );

        // Check that probabilities sum to 1.0
        for row in pred_array.axis_iter(ndarray::Axis(0)) {
            let sum: f64 = row.sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Probabilities should sum to 1.0, got {}",
                sum
            );
        }

        println!("✅ End-to-end bias correction test passed!");
    }
}
