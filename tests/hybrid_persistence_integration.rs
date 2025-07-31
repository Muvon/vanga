//! Integration test for hybrid LSTM+SmartCore model persistence and inference
//!
//! This test verifies the complete pipeline:
//! 1. Train hybrid LSTM+SmartCore model
//! 2. Save the complete model (LSTM + SmartCore)
//! 3. Load the complete model
//! 4. Make predictions using the loaded hybrid model

use vanga::config::model::XGBoostConfig;
use vanga::config::TrainingConfig;
use vanga::model::lstm::{LSTMConfig, LSTMModel};
use vanga::utils::error::Result;

use candle_core::Device;
use ndarray::{Array2, Array3};
use tempfile::tempdir;

/// Test complete hybrid model persistence and inference pipeline
#[tokio::test]
async fn test_hybrid_model_persistence_and_inference() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🚀 Starting hybrid model persistence and inference test...");

    // Step 1: Create hybrid model configuration
    let lstm_config = LSTMConfig {
        input_size: 8,
        hidden_sizes: vec![16, 8], // Multi-layer LSTM
        output_size: 3,            // 3-class classification
        sequence_length: 10,
        learning_rate: 0.01,
        num_layers: 2,
    };

    let xgb_config = XGBoostConfig {
        enabled: true,
        feature_dim: 8,   // Matches LSTM final hidden size
        n_estimators: 10, // Small for testing
        max_depth: 3,
        objective: "RandomForest".to_string(),
        eval_metric: "multiclass_accuracy".to_string(),
        save_feature_importance: true,
        importance_method: "permutation".to_string(),
    };

    println!("✅ Configuration created");

    // Step 2: Create synthetic training data
    let batch_size = 50;
    let sequence_length = 10;
    let input_size = 8;
    let num_classes = 3;

    // Create sequences with patterns
    let mut sequences_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..batch_size {
        // Create sequence with pattern based on sample index
        for seq_idx in 0..sequence_length {
            for feature_idx in 0..input_size {
                let value =
                    (i as f64 * 0.1 + seq_idx as f64 * 0.05 + feature_idx as f64 * 0.02).sin();
                sequences_data.push(value);
            }
        }

        // Create one-hot encoded targets based on pattern
        let class = i % num_classes;
        let mut target_row = vec![0.0; num_classes];
        target_row[class] = 1.0;
        targets_data.extend(target_row);
    }

    let sequences =
        Array3::from_shape_vec((batch_size, sequence_length, input_size), sequences_data).unwrap();

    let targets = Array2::from_shape_vec((batch_size, num_classes), targets_data).unwrap();

    println!(
        "✅ Synthetic data created: sequences={:?}, targets={:?}",
        sequences.shape(),
        targets.shape()
    );

    // Step 3: Create and train hybrid model
    let mut model = LSTMModel::new(lstm_config)?;

    // Enable XGBoost hybrid mode
    model.xgboost_model = Some(vanga::model::xgboost::XGBoostRegressor::new(
        xgb_config,
        Device::Cpu,
    ));

    println!("🔄 Training hybrid LSTM+SmartCore model...");

    // Create training configuration
    let training_config = TrainingConfig::default_for_testing()?;

    // Train the model (this should train both LSTM and SmartCore)
    model
        .train(&sequences, &targets, &training_config, None, None, None)
        .await?;

    println!("✅ Hybrid model training completed");

    // Verify XGBoost model was trained
    assert!(
        model.xgboost_model.is_some(),
        "XGBoost model should be present after training"
    );
    assert!(
        model.xgboost_model.as_ref().unwrap().is_trained(),
        "XGBoost model should be trained"
    );

    // Step 4: Make predictions with original model
    println!("🔮 Making predictions with original hybrid model...");
    let original_predictions = model.predict(&sequences).await?;
    println!(
        "✅ Original predictions shape: {:?}",
        original_predictions.shape()
    );

    // Step 5: Save the complete hybrid model
    let temp_dir = tempdir().unwrap();
    let model_path = temp_dir.path().join("test_hybrid_model");

    println!("💾 Saving hybrid model to: {}", model_path.display());
    model.save(&model_path)?;

    // Verify all expected files were created
    assert!(
        model_path.with_extension("safetensors").exists(),
        "LSTM weights should be saved"
    );
    assert!(
        model_path.with_extension("config").exists(),
        "LSTM config should be saved"
    );
    assert!(
        std::path::Path::new(&format!("{}.smartcore.meta", model_path.to_string_lossy())).exists(),
        "SmartCore metadata should be saved"
    );

    println!("✅ Hybrid model saved successfully");

    // Step 6: Load the complete hybrid model
    println!("📂 Loading hybrid model from: {}", model_path.display());
    let loaded_model = LSTMModel::load(&model_path)?;

    // Verify the loaded model has XGBoost component
    assert!(
        loaded_model.xgboost_model.is_some(),
        "Loaded model should have XGBoost component"
    );
    println!("✅ Hybrid model loaded successfully");

    // Step 7: Make predictions with loaded model
    println!("🔮 Making predictions with loaded hybrid model...");
    let loaded_predictions = loaded_model.predict(&sequences).await?;
    println!(
        "✅ Loaded predictions shape: {:?}",
        loaded_predictions.shape()
    );

    // Step 8: Verify predictions are consistent
    assert_eq!(
        original_predictions.shape(),
        loaded_predictions.shape(),
        "Prediction shapes should match"
    );

    // Check that predictions are reasonable (not all zeros or identical)
    let original_sum: f64 = original_predictions.iter().sum();
    let loaded_sum: f64 = loaded_predictions.iter().sum();

    assert!(
        original_sum.abs() > 0.1,
        "Original predictions should not be all zeros"
    );
    assert!(
        loaded_sum.abs() > 0.1,
        "Loaded predictions should not be all zeros"
    );

    println!("📊 Original predictions sum: {:.4}", original_sum);
    println!("📊 Loaded predictions sum: {:.4}", loaded_sum);

    // Step 9: Test with new data (inference-only)
    println!("🔮 Testing inference with new data...");

    // Create smaller test batch
    let test_batch_size = 10;
    let mut test_sequences_data = Vec::new();

    for i in 0..test_batch_size {
        for seq_idx in 0..sequence_length {
            for _feature_idx in 0..input_size {
                let value = (i as f64 * 0.15 + seq_idx as f64 * 0.08).cos(); // Different pattern
                test_sequences_data.push(value);
            }
        }
    }

    let test_sequences = Array3::from_shape_vec(
        (test_batch_size, sequence_length, input_size),
        test_sequences_data,
    )
    .unwrap();

    let test_predictions = loaded_model.predict(&test_sequences).await?;
    assert_eq!(test_predictions.shape(), &[test_batch_size, num_classes]);

    let test_sum: f64 = test_predictions.iter().sum();
    assert!(
        test_sum.abs() > 0.1,
        "Test predictions should not be all zeros"
    );

    println!(
        "✅ Inference test completed - predictions sum: {:.4}",
        test_sum
    );

    println!("🎉 Hybrid model persistence and inference test completed successfully!");
    println!("✅ All components working:");
    println!("   - LSTM training ✅");
    println!("   - SmartCore training ✅");
    println!("   - Hybrid prediction ✅");
    println!("   - Model persistence ✅");
    println!("   - Model loading ✅");
    println!("   - Inference pipeline ✅");

    Ok(())
}

/// Test that pure LSTM models still work (backward compatibility)
#[tokio::test]
async fn test_pure_lstm_persistence() -> Result<()> {
    println!("🚀 Testing pure LSTM model persistence (no XGBoost)...");

    let lstm_config = LSTMConfig {
        input_size: 5,
        hidden_sizes: vec![8],
        output_size: 2,
        sequence_length: 8,
        learning_rate: 0.01,
        num_layers: 1,
    };

    let mut model = LSTMModel::new(lstm_config)?;
    // Note: NOT setting xgboost_model, so it remains None

    // Create simple training data with multiple classes
    let mut sequences_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..20 {
        // Create sequence
        for seq_idx in 0..8 {
            for feature_idx in 0..5 {
                let value =
                    (i as f64 * 0.1 + seq_idx as f64 * 0.05 + feature_idx as f64 * 0.02).sin();
                sequences_data.push(value);
            }
        }

        // Create targets with 2 classes
        let class = i % 2; // Alternate between 0 and 1
        let mut target_row = vec![0.0; 2];
        target_row[class] = 1.0;
        targets_data.extend(target_row);
    }

    let sequences = Array3::from_shape_vec((20, 8, 5), sequences_data).unwrap();
    let targets = Array2::from_shape_vec((20, 2), targets_data).unwrap();

    let mut training_config = TrainingConfig::default_for_testing()?;
    // Disable XGBoost for pure LSTM test
    training_config.model.xgboost.enabled = false;

    model
        .train(&sequences, &targets, &training_config, None, None, None)
        .await?;

    // Verify no XGBoost model
    assert!(
        model.xgboost_model.is_none(),
        "Pure LSTM should not have XGBoost model"
    );

    // Save and load
    let temp_dir = tempdir().unwrap();
    let model_path = temp_dir.path().join("pure_lstm_model");

    model.save(&model_path)?;
    let loaded_model = LSTMModel::load(&model_path)?;

    // Verify loaded model is still pure LSTM
    assert!(
        loaded_model.xgboost_model.is_none(),
        "Loaded pure LSTM should not have XGBoost model"
    );

    // Test predictions
    let predictions = loaded_model.predict(&sequences).await?;
    assert_eq!(predictions.shape(), &[20, 2]);

    println!("✅ Pure LSTM persistence test completed");
    Ok(())
}
