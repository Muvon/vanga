//! Integration tests for SmartCore backend with VANGA hybrid model
//!
//! These tests verify that the SmartCore backend integrates properly
//! with the complete VANGA hybrid LSTM+ML training pipeline.

#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]

use vanga::config::model::XGBoostConfig;
use vanga::model::{SmartCoreRegressor, XGBoostRegressor};
use vanga::utils::error::Result;

use candle_core::{DType, Device, Tensor};

/// Test hybrid model integration with SmartCore backend
#[tokio::test]
async fn test_hybrid_model_integration() -> Result<()> {
    // Configure hybrid model with SmartCore backend
    let xgb_config = XGBoostConfig {
        enabled: true,
        feature_dim: 32, // Typical LSTM output dimension
        n_estimators: 50,
        max_depth: 6,
        save_feature_importance: true,
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut hybrid_classifier = XGBoostRegressor::new(xgb_config, device.clone());

    // Simulate LSTM feature extraction output
    let batch_size = 100;
    let lstm_feature_dim = 32;
    let num_price_levels = 5;

    // Create realistic LSTM features (simulating h_n from LSTM)
    let mut lstm_features_data = Vec::new();
    for i in 0..batch_size {
        for j in 0..lstm_feature_dim {
            // Simulate LSTM hidden state patterns
            let feature = (i as f32 * 0.01 + j as f32 * 0.02).tanh()
                + 0.1 * (i as f32 * j as f32 * 0.001).sin();
            lstm_features_data.push(feature);
        }
    }
    let lstm_features =
        Tensor::from_vec(lstm_features_data, (batch_size, lstm_feature_dim), &device)?
            .to_dtype(DType::F32)?;

    // Create price level targets (one-hot encoded)
    let mut price_targets_data = Vec::new();
    for i in 0..batch_size {
        // Simulate price level classification based on features
        let level = (i * 3 + 7) % num_price_levels;
        let mut target_row = vec![0.0f32; num_price_levels];
        target_row[level] = 1.0;
        price_targets_data.extend(target_row);
    }
    let price_targets =
        Tensor::from_vec(price_targets_data, (batch_size, num_price_levels), &device)?
            .to_dtype(DType::F32)?;

    // Phase 1: Train hybrid classifier on LSTM features
    println!("Phase 1: Training hybrid classifier on LSTM features...");
    hybrid_classifier.train(&lstm_features, &price_targets, None, None)?;

    // Verify training success
    assert!(hybrid_classifier.is_trained());
    println!("✅ Hybrid classifier training completed");

    // Phase 2: Test inference pipeline
    println!("Phase 2: Testing inference pipeline...");

    // Create new test batch (simulating real inference)
    let test_batch_size = 20;
    let test_features_data: Vec<f32> = (0..test_batch_size * lstm_feature_dim)
        .map(|i| (i as f32 * 0.005).sin() + 0.1)
        .collect();
    let test_features = Tensor::from_vec(
        test_features_data,
        (test_batch_size, lstm_feature_dim),
        &device,
    )?
    .to_dtype(DType::F32)?;

    // Make predictions
    let predictions = hybrid_classifier.predict(&test_features)?;

    // Verify prediction format
    assert_eq!(
        predictions.shape().dims(),
        &[test_batch_size, num_price_levels]
    );
    println!("✅ Predictions shape correct: {:?}", predictions.shape());

    // Verify predictions are probability distributions
    let pred_data = predictions.to_vec2::<f32>()?;
    for (i, row) in pred_data.iter().enumerate() {
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.1,
            "Row {} predictions don't sum to ~1.0: sum={:.3}",
            i,
            sum
        );

        // Check for valid probabilities
        for &prob in row {
            assert!((0.0..=1.0).contains(&prob), "Invalid probability: {}", prob);
        }
    }
    println!("✅ Predictions are valid probability distributions");

    // Phase 3: Test feature importance extraction
    println!("Phase 3: Testing feature importance extraction...");
    let importance = hybrid_classifier.get_feature_importance().unwrap();

    assert_eq!(importance.len(), lstm_feature_dim);
    let total_importance: f32 = importance.values().sum();
    assert!(
        (total_importance - 1.0).abs() < 0.01,
        "Feature importance should sum to 1.0"
    );

    // Show top important features
    let mut sorted_features: Vec<_> = importance.iter().collect();
    sorted_features.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("Top 5 most important LSTM features:");
    for (i, (feature, score)) in sorted_features.iter().take(5).enumerate() {
        println!("  {}. {}: {:.4}", i + 1, feature, score);
    }

    println!("✅ Hybrid model integration test completed successfully");
    Ok(())
}

/// Test multi-target hybrid model (price levels + direction)
#[tokio::test]
async fn test_multi_target_hybrid_model() -> Result<()> {
    let device = Device::Cpu;
    let batch_size = 80;
    let lstm_feature_dim = 24;

    // Create LSTM features
    let lstm_features_data: Vec<f32> = (0..batch_size * lstm_feature_dim)
        .map(|i| (i as f32 * 0.02).cos() + 0.05 * (i as f32).sin())
        .collect();
    let lstm_features =
        Tensor::from_vec(lstm_features_data, (batch_size, lstm_feature_dim), &device)?
            .to_dtype(DType::F32)?;

    // Test 1: Price Level Classification (5 classes)
    println!("Testing price level classification...");
    let mut price_config = XGBoostConfig::default();
    price_config.enabled = true;
    price_config.feature_dim = lstm_feature_dim;
    price_config.n_estimators = 30;

    let mut price_classifier = XGBoostRegressor::new(price_config, device.clone());

    let num_price_levels = 5;
    let mut price_targets_data = Vec::new();
    for i in 0..batch_size {
        let level = (i * 2 + 3) % num_price_levels;
        let mut target_row = vec![0.0f32; num_price_levels];
        target_row[level] = 1.0;
        price_targets_data.extend(target_row);
    }
    let price_targets =
        Tensor::from_vec(price_targets_data, (batch_size, num_price_levels), &device)?
            .to_dtype(DType::F32)?;

    price_classifier.train(&lstm_features, &price_targets, None, None)?;
    let price_predictions = price_classifier.predict(&lstm_features)?;
    assert_eq!(
        price_predictions.shape().dims(),
        &[batch_size, num_price_levels]
    );

    // Test 2: Direction Classification (3 classes: up, down, sideways)
    println!("Testing direction classification...");
    let mut direction_config = XGBoostConfig::default();
    direction_config.enabled = true;
    direction_config.feature_dim = lstm_feature_dim;
    direction_config.n_estimators = 25;

    let mut direction_classifier = XGBoostRegressor::new(direction_config, device.clone());

    let num_directions = 3;
    let mut direction_targets_data = Vec::new();
    for i in 0..batch_size {
        let direction = i % num_directions;
        let mut target_row = vec![0.0f32; num_directions];
        target_row[direction] = 1.0;
        direction_targets_data.extend(target_row);
    }
    let direction_targets = Tensor::from_vec(
        direction_targets_data,
        (batch_size, num_directions),
        &device,
    )?
    .to_dtype(DType::F32)?;

    direction_classifier.train(&lstm_features, &direction_targets, None, None)?;
    let direction_predictions = direction_classifier.predict(&lstm_features)?;
    assert_eq!(
        direction_predictions.shape().dims(),
        &[batch_size, num_directions]
    );

    // Verify both models work independently
    assert!(price_classifier.is_trained());
    assert!(direction_classifier.is_trained());
    assert!(price_classifier.get_feature_importance().is_some());
    assert!(direction_classifier.get_feature_importance().is_some());

    println!("✅ Multi-target hybrid model test completed");
    Ok(())
}

/// Test SmartCore backend with different data patterns
#[tokio::test]
async fn test_smartcore_data_patterns() -> Result<()> {
    let device = Device::Cpu;
    let mut config = XGBoostConfig::default();
    config.enabled = true;
    config.feature_dim = 8;
    config.n_estimators = 20;
    config.save_feature_importance = true;

    // Test 1: Linear separable data
    println!("Testing linear separable data...");
    let mut regressor1 = SmartCoreRegressor::new(config.clone(), device.clone());

    let batch_size = 60;
    let feature_dim = 8;
    let num_classes = 2;

    let mut features_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..batch_size {
        // Create linearly separable pattern
        let separator = if i < batch_size / 2 { 1.0 } else { -1.0 };
        let mut feature_row = vec![separator; feature_dim];
        // Add some noise
        for j in 1..feature_dim {
            feature_row[j] += (i as f32 * j as f32 * 0.01).sin() * 0.1;
        }
        features_data.extend(feature_row);

        let class = if separator > 0.0 { 0 } else { 1 };
        let mut target_row = vec![0.0f32; num_classes];
        target_row[class] = 1.0;
        targets_data.extend(target_row);
    }

    let features1 = Tensor::from_vec(features_data, (batch_size, feature_dim), &device)?
        .to_dtype(DType::F32)?;
    let targets1 =
        Tensor::from_vec(targets_data, (batch_size, num_classes), &device)?.to_dtype(DType::F32)?;

    regressor1.train(&features1, &targets1, None, None)?;
    let predictions1 = regressor1.predict(&features1)?;

    // Should achieve high accuracy on linearly separable data
    let pred_data1 = predictions1.to_vec2::<f32>()?;
    let target_data1 = targets1.to_vec2::<f32>()?;
    let mut correct = 0;
    for (pred_row, target_row) in pred_data1.iter().zip(target_data1.iter()) {
        let pred_class = pred_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let true_class = target_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        if pred_class == true_class {
            correct += 1;
        }
    }
    let accuracy1 = correct as f32 / batch_size as f32;
    println!("Linear separable accuracy: {:.2}%", accuracy1 * 100.0);
    assert!(
        accuracy1 > 0.7,
        "Should achieve >70% accuracy on linear data"
    );

    // Test 2: Non-linear pattern (XOR-like)
    println!("Testing non-linear pattern...");
    let mut regressor2 = SmartCoreRegressor::new(config, device.clone());

    let mut features_data2 = Vec::new();
    let mut targets_data2 = Vec::new();

    for i in 0..batch_size {
        // Create XOR-like pattern
        let x1 = if i % 4 < 2 { 1.0 } else { -1.0 };
        let x2 = if (i % 4) % 2 == 0 { 1.0 } else { -1.0 };
        let mut feature_row = vec![x1, x2];
        // Add more features with noise
        for j in 2..feature_dim {
            feature_row.push((i as f32 * j as f32 * 0.05).sin());
        }
        features_data2.extend(feature_row);

        // XOR target: class 0 if x1 and x2 have same sign, class 1 otherwise
        let class = if (x1 > 0.0) == (x2 > 0.0) { 0 } else { 1 };
        let mut target_row = vec![0.0f32; num_classes];
        target_row[class] = 1.0;
        targets_data2.extend(target_row);
    }

    let features2 = Tensor::from_vec(features_data2, (batch_size, feature_dim), &device)?
        .to_dtype(DType::F32)?;
    let targets2 = Tensor::from_vec(targets_data2, (batch_size, num_classes), &device)?
        .to_dtype(DType::F32)?;

    regressor2.train(&features2, &targets2, None, None)?;
    let predictions2 = regressor2.predict(&features2)?;

    // Non-linear pattern should still be learnable by Random Forest
    let pred_data2 = predictions2.to_vec2::<f32>()?;
    let target_data2 = targets2.to_vec2::<f32>()?;
    let mut correct2 = 0;
    for (pred_row, target_row) in pred_data2.iter().zip(target_data2.iter()) {
        let pred_class = pred_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let true_class = target_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        if pred_class == true_class {
            correct2 += 1;
        }
    }
    let accuracy2 = correct2 as f32 / batch_size as f32;
    println!("Non-linear pattern accuracy: {:.2}%", accuracy2 * 100.0);
    assert!(
        accuracy2 > 0.5,
        "Should achieve >50% accuracy on non-linear data"
    );

    println!("✅ Data pattern tests completed");
    Ok(())
}

/// Test SmartCore backend robustness with edge cases
#[tokio::test]
async fn test_smartcore_robustness() -> Result<()> {
    let device = Device::Cpu;

    // Test 1: Small dataset
    println!("Testing small dataset handling...");
    let mut config = XGBoostConfig::default();
    config.enabled = true;
    config.feature_dim = 4;
    config.n_estimators = 5; // Reduced for small dataset
    config.max_depth = 2;

    let mut regressor = SmartCoreRegressor::new(config.clone(), device.clone());

    let batch_size = 10; // Very small dataset
    let feature_dim = 4;
    let num_classes = 2;

    let features_data: Vec<f32> = (0..batch_size * feature_dim)
        .map(|i| (i as f32 * 0.5).sin())
        .collect();
    let features = Tensor::from_vec(features_data, (batch_size, feature_dim), &device)?
        .to_dtype(DType::F32)?;

    let mut targets_data = Vec::new();
    for i in 0..batch_size {
        let class = i % num_classes;
        let mut target_row = vec![0.0f32; num_classes];
        target_row[class] = 1.0;
        targets_data.extend(target_row);
    }
    let targets =
        Tensor::from_vec(targets_data, (batch_size, num_classes), &device)?.to_dtype(DType::F32)?;

    // Should handle small dataset gracefully
    regressor.train(&features, &targets, None, None)?;
    let predictions = regressor.predict(&features)?;
    assert_eq!(predictions.shape().dims(), &[batch_size, num_classes]);

    // Test 2: Single class dataset (edge case) - SmartCore requires at least 2 classes
    println!("Testing single class dataset...");
    let mut single_class_targets_data = Vec::new();
    for i in 0..batch_size {
        let mut target_row = vec![0.0f32; num_classes];
        // Create a dataset with mostly class 0 but some class 1 to satisfy SmartCore requirements
        let class = if i < batch_size - 2 { 0 } else { 1 }; // Last 2 samples are class 1
        target_row[class] = 1.0;
        single_class_targets_data.extend(target_row);
    }
    let single_class_targets = Tensor::from_vec(
        single_class_targets_data,
        (batch_size, num_classes),
        &device,
    )?
    .to_dtype(DType::F32)?;

    let mut single_class_regressor = SmartCoreRegressor::new(config, device.clone());

    // Should handle imbalanced dataset gracefully
    single_class_regressor.train(&features, &single_class_targets, None, None)?;
    let single_class_predictions = single_class_regressor.predict(&features)?;
    assert_eq!(
        single_class_predictions.shape().dims(),
        &[batch_size, num_classes]
    );

    // Most predictions should favor class 0 (since it's the majority class)
    let pred_data = single_class_predictions.to_vec2::<f32>()?;
    let class_0_predictions = pred_data.iter().filter(|row| row[0] >= row[1]).count();
    assert!(
        class_0_predictions >= batch_size / 2,
        "Should predict class 0 for majority of samples"
    );

    println!("✅ Robustness tests completed");
    Ok(())
}

/// Test configuration compatibility between XGBoost and SmartCore
#[tokio::test]
async fn test_config_compatibility() {
    // Test that XGBoostConfig works with SmartCore backend
    let mut config = XGBoostConfig::default();

    // Set SmartCore parameters
    config.enabled = true;
    config.feature_dim = 16;
    config.n_estimators = 100;
    config.max_depth = 8;
    config.objective = "RandomForest".to_string();
    config.eval_metric = "multiclass_accuracy".to_string();
    config.save_feature_importance = true;
    config.importance_method = "permutation".to_string();

    let device = Device::Cpu;

    // Should create SmartCore regressor with XGBoost config
    let smartcore_regressor = SmartCoreRegressor::new(config.clone(), device.clone());
    assert_eq!(smartcore_regressor.get_config().feature_dim, 16);
    assert_eq!(smartcore_regressor.get_config().n_estimators, 100);
    assert_eq!(smartcore_regressor.get_config().max_depth, 8);

    // Should create XGBoost wrapper with same config
    let xgb_regressor = XGBoostRegressor::new(config, device);
    assert_eq!(xgb_regressor.get_config().feature_dim, 16);
    assert_eq!(xgb_regressor.get_config().n_estimators, 100);
    assert_eq!(xgb_regressor.get_config().max_depth, 8);

    println!("✅ Configuration compatibility verified");
}
