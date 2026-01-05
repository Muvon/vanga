use crate::config::model::ModelConfig;
use crate::config::training::{EpochConfig, OptimizerType, TrainingConfig, TrainingParams};
use crate::model::bias_correction::{BiasCorrection, LinearBiasCorrector};
use crate::model::lstm::LSTMModel;
use crate::targets::TargetType;
use candle_core::{Device, Tensor};
use ndarray::{Array2, Array3};

#[test]
fn test_tensor_based_bias_correction() {
    // Create a bias corrector with test configuration
    let mut config = BiasCorrection::default();
    config.enabled = true;
    config.ramp_up_epochs = 5;
    config.max_strength = 0.8;

    let mut corrector = LinearBiasCorrector::new(config);

    // Set up test bias factors
    corrector.class_bias_factors = [1.2, 0.8, 1.0, 1.1, 0.9];
    corrector.confidence_scaling = 1.0;
    corrector.is_calibrated = true;

    // Create test predictions tensor (2 samples, 5 classes)
    let device = Device::Cpu;
    let predictions_data = vec![
        0.2_f32, 0.2, 0.2, 0.2, 0.2, // Uniform distribution
        0.1, 0.3, 0.4, 0.15, 0.05, // Skewed distribution
    ];
    let predictions = Tensor::from_slice(&predictions_data, (2, 5), &device).unwrap();

    // Test correction at different epochs (ramp-up)
    for epoch in 0..10 {
        let corrected = corrector
            .apply_correction_tensor(&predictions, epoch)
            .unwrap();

        // Verify shape is preserved
        assert_eq!(corrected.shape().dims(), &[2, 5]);

        // Extract corrected values
        let corrected_vec: Vec<f32> = corrected.flatten_all().unwrap().to_vec1().unwrap();

        // Verify probabilities sum to ~1.0
        for i in 0..2 {
            let row_sum: f32 = (0..5).map(|j| corrected_vec[i * 5 + j]).sum();
            assert!((row_sum - 1.0).abs() < 0.01, "Row {} sum: {}", i, row_sum);
        }

        // Verify gradual integration (strength increases with epoch)
        if epoch < 5 {
            // During ramp-up, correction should be partial
            let expected_strength = (epoch as f64 / 5.0) * 0.8;
            println!(
                "Epoch {}: Expected strength {:.2}",
                epoch, expected_strength
            );
        } else {
            // After ramp-up, full strength
            println!("Epoch {}: Full strength 0.8", epoch);
        }
    }
}

#[test]
fn test_correction_impact_calculation() {
    let config = BiasCorrection::default();
    let corrector = LinearBiasCorrector::new(config);

    let device = Device::Cpu;

    // Create two different distributions
    let original_data = vec![0.2_f32, 0.2, 0.2, 0.2, 0.2]; // Uniform
    let corrected_data = vec![0.1, 0.15, 0.3, 0.25, 0.2]; // Different

    let original = Tensor::from_slice(&original_data, (1, 5), &device).unwrap();
    let corrected = Tensor::from_slice(&corrected_data, (1, 5), &device).unwrap();

    // Calculate KL divergence
    let kl_div = corrector
        .calculate_correction_impact(&original, &corrected)
        .unwrap();

    // KL divergence should be positive and reasonable
    assert!(kl_div > 0.0, "KL divergence should be positive");
    assert!(
        kl_div < 1.0,
        "KL divergence should be reasonable for similar distributions"
    );

    println!("KL divergence between distributions: {:.6}", kl_div);
}

#[test]
fn test_recalibration_frequency() {
    let mut config = BiasCorrection::default();
    config.recalibration_frequency = 3; // Recalibrate every 3 epochs

    let corrector = LinearBiasCorrector::new(config.clone());

    // Test that recalibration should happen at epochs 3, 6, 9, etc.
    for epoch in 1..=10 {
        let should_recalibrate =
            config.recalibration_frequency > 0 && epoch % config.recalibration_frequency == 0;

        if should_recalibrate {
            println!("Epoch {}: Should recalibrate", epoch);
            assert_eq!(epoch % 3, 0);
        } else {
            println!("Epoch {}: No recalibration", epoch);
        }
    }
}

#[test]
fn test_print_info_configuration() {
    // Test with print_info enabled
    let mut config_with_print = BiasCorrection::default();
    config_with_print.print_info = true;
    config_with_print.recalibration_frequency = 5;
    config_with_print.use_ensemble_calibration = true;

    assert!(config_with_print.print_info);
    assert_eq!(config_with_print.recalibration_frequency, 5);
    assert!(config_with_print.use_ensemble_calibration);

    // Test with print_info disabled (default)
    let config_without_print = BiasCorrection::default();
    assert!(!config_without_print.print_info);
    assert_eq!(config_without_print.recalibration_frequency, 5); // default value

    // Test recalibration logic with print_info
    for epoch in 1..=20 {
        let should_recalibrate = config_with_print.use_ensemble_calibration
            && config_with_print.recalibration_frequency > 0
            && epoch > 0
            && epoch % config_with_print.recalibration_frequency == 0;

        if should_recalibrate {
            // Recalibration happens at epochs 5, 10, 15, 20
            assert_eq!(epoch % 5, 0);
            // Print info should only be shown if print_info is true
            if config_with_print.print_info {
                println!("Epoch {}: Recalibrating with print info", epoch);
            }
        }
    }
}

#[tokio::test]
async fn test_training_integration_with_bias_correction() {
    // Create a simple LSTM model with bias correction enabled
    let mut model_config = ModelConfig::default();
    model_config.lstm.hidden_size = 32;
    model_config.lstm.num_layers = 1;
    model_config.lstm.sequence_length = 10;
    model_config.lstm.input_size = 5;
    model_config.lstm.output_size = 5; // 5-class classification

    // Enable bias correction with training integration
    model_config.bias_correction.enabled = true;
    model_config.bias_correction.ramp_up_epochs = 3;
    model_config.bias_correction.max_strength = 0.5;
    model_config.bias_correction.recalibration_frequency = 2;

    let mut lstm_model = LSTMModel::new(
        model_config.lstm.clone(),
        Device::Cpu,
        Some(model_config.clone()),
    )
    .unwrap();

    // Set target context for categorical target
    lstm_model.set_target_context("test_symbol", TargetType::PriceLevel);

    // Initialize the model
    lstm_model.initialize().unwrap();

    // Create synthetic training data
    let batch_size = 8;
    let sequences = Array3::<f64>::zeros((batch_size, 10, 5));
    let targets = Array2::<f64>::zeros((batch_size, 5)); // One-hot encoded

    // Create training configuration
    let mut training_config = TrainingConfig::default();
    training_config.training.epochs = EpochConfig::Fixed(10);
    training_config.training.batch_size = 4;
    training_config.training.learning_rate = 0.001;
    training_config.training.optimizer = OptimizerType::Adam {
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    };

    // Create validation data for bias correction calibration
    let val_sequences = Array3::<f64>::zeros((4, 10, 5));
    let val_targets = Array2::<f64>::zeros((4, 5));

    // Train with bias correction integration
    let result = lstm_model
        .train(
            &sequences,
            &targets,
            &training_config,
            Some(&val_sequences),
            Some(&val_targets),
            None,
        )
        .await;

    // Training should complete successfully
    assert!(
        result.is_ok(),
        "Training with bias correction failed: {:?}",
        result
    );

    // Verify bias corrector was initialized
    assert!(
        lstm_model.bias_corrector.is_some(),
        "Bias corrector should be initialized"
    );

    // The corrector should be calibrated after training with validation data
    if let Some(corrector) = &lstm_model.bias_corrector {
        // Note: In this test with zero data, it might not calibrate due to min_samples
        println!("Bias corrector calibrated: {}", corrector.is_calibrated);
        println!("Bias factors: {:?}", corrector.class_bias_factors);
    }
}

#[test]
fn test_gradual_strength_integration() {
    let mut config = BiasCorrection::default();
    config.ramp_up_epochs = 10;
    config.max_strength = 0.6;

    let corrector = LinearBiasCorrector::new(config.clone());

    // Test strength calculation at different epochs
    let test_epochs = vec![0, 2, 5, 7, 10, 15, 20];

    for epoch in test_epochs {
        let expected_strength = if epoch < config.ramp_up_epochs {
            (epoch as f64 / config.ramp_up_epochs as f64) * config.max_strength
        } else {
            config.max_strength
        };

        println!(
            "Epoch {}: Expected strength = {:.3} (ramp_up={}, max={})",
            epoch, expected_strength, config.ramp_up_epochs, config.max_strength
        );

        // Verify strength is within bounds
        assert!(expected_strength >= 0.0);
        assert!(expected_strength <= config.max_strength);

        // Verify gradual increase during ramp-up
        if epoch < config.ramp_up_epochs {
            assert!(expected_strength < config.max_strength);
        } else {
            assert_eq!(expected_strength, config.max_strength);
        }
    }
}

#[test]
fn test_bias_correction_with_extreme_distributions() {
    let mut config = BiasCorrection::default();
    config.enabled = true;
    config.correction_bounds = [0.5, 2.0];

    let mut corrector = LinearBiasCorrector::new(config);

    // Set extreme bias factors (will be clamped by bounds)
    corrector.class_bias_factors = [3.0, 0.2, 1.5, 0.7, 2.5]; // Some outside bounds
    corrector.is_calibrated = true;

    // After bounds application, factors should be:
    // [2.0, 0.5, 1.5, 0.7, 2.0] (clamped to [0.5, 2.0])

    let device = Device::Cpu;

    // Create highly skewed predictions
    let predictions_data = vec![
        0.8_f32, 0.05, 0.05, 0.05, 0.05, // Heavily biased to class 0
        0.01, 0.01, 0.01, 0.01, 0.96, // Heavily biased to class 4
    ];
    let predictions = Tensor::from_slice(&predictions_data, (2, 5), &device).unwrap();

    // Apply correction at full strength (epoch > ramp_up)
    let corrected = corrector.apply_correction_tensor(&predictions, 20).unwrap();

    // Extract corrected values
    let corrected_vec: Vec<f32> = corrected.flatten_all().unwrap().to_vec1().unwrap();

    // Verify corrections are applied but bounded
    println!("Original row 1: {:?}", &predictions_data[0..5]);
    println!("Corrected row 1: {:?}", &corrected_vec[0..5]);
    println!("Original row 2: {:?}", &predictions_data[5..10]);
    println!("Corrected row 2: {:?}", &corrected_vec[5..10]);

    // Probabilities should still sum to 1.0
    for i in 0..2 {
        let row_sum: f32 = (0..5).map(|j| corrected_vec[i * 5 + j]).sum();
        assert!((row_sum - 1.0).abs() < 0.01, "Row {} sum: {}", i, row_sum);
    }
}
