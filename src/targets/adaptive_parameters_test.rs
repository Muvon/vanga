//! Tests for adaptive parameter optimization and storage consistency
//!
//! This test suite verifies:
//! - Parameter calibration produces balanced class distributions
//! - Model persistence correctly saves/loads adaptive parameters
//! - Target generation consistency between training and prediction
//! - Cross-target parameter coordination

use crate::config::model::TargetsConfig;
use crate::data::structures::MarketDataRow;
use crate::model::lstm::config::{LSTMConfig, ModelState};
use crate::model::lstm::core::LSTMModel;
use crate::targets::adaptive_parameters::*;
use crate::targets::unified_calibrator::*;
use crate::targets::*;
use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Create sample OHLCV data for testing
fn create_sample_ohlcv_data(length: usize) -> Vec<MarketDataRow> {
    let mut data = Vec::new();
    let mut price = 50000.0;

    for i in 0..length {
        // Create realistic price movement with volatility
        let change_pct = (i as f64 * 0.1).sin() * 0.02 + (i as f64 * 0.05).cos() * 0.01;
        price *= 1.0 + change_pct;

        let high = price * (1.0 + 0.005);
        let low = price * (1.0 - 0.005);
        let volume = 1000.0 + (i as f64 * 0.2).sin().abs() * 500.0;

        data.push(MarketDataRow {
            timestamp: i as i64 * 3600, // Hourly data
            open: price,
            high,
            low,
            close: price,
            volume,
        });
    }

    data
}

/// Create sample DataFrame from OHLCV data
fn create_sample_dataframe(ohlcv_data: &[MarketDataRow]) -> Result<DataFrame> {
    let timestamps: Vec<i64> = ohlcv_data.iter().map(|row| row.timestamp).collect();
    let opens: Vec<f64> = ohlcv_data.iter().map(|row| row.open).collect();
    let highs: Vec<f64> = ohlcv_data.iter().map(|row| row.high).collect();
    let lows: Vec<f64> = ohlcv_data.iter().map(|row| row.low).collect();
    let closes: Vec<f64> = ohlcv_data.iter().map(|row| row.close).collect();
    let volumes: Vec<f64> = ohlcv_data.iter().map(|row| row.volume).collect();

    let df = df! {
        "timestamp" => timestamps,
        "open" => opens,
        "high" => highs,
        "low" => lows,
        "close" => closes,
        "volume" => volumes,
    }
    .map_err(|e| {
        crate::utils::error::VangaError::DataError(format!("DataFrame creation failed: {}", e))
    })?;

    Ok(df)
}

#[tokio::test]
async fn test_adaptive_parameter_calibration() -> Result<()> {
    // Create sample data
    let ohlcv_data = create_sample_ohlcv_data(1000);
    let sequence_length = 60;
    let horizon_steps = 24;
    let sequence_indices: Vec<usize> = (0..900).step_by(10).collect();

    // Create base configuration
    let base_config = TargetsConfig::default();

    // Test unified calibration
    let adaptive_params = calibrate_adaptive_parameters(
        base_config,
        &ohlcv_data,
        sequence_length,
        horizon_steps,
        &sequence_indices,
    )
    .await?;

    // Verify all target types have parameters
    assert!(adaptive_params.direction.base_sensitivity > 0.0);
    assert!(adaptive_params.price_levels.bandwidth_size > 0.0);
    assert!(adaptive_params.volatility.bandwidth_size > 0.0);
    assert!(adaptive_params.sentiment.body_sensitivity > 0.0);
    assert!(adaptive_params.volume.bandwidth_size > 0.0);

    // Verify balance scores are reasonable (should be better than random)
    assert!(adaptive_params.direction.achieved_balance.balance_score < 5.0); // Lower is better
    assert!(adaptive_params.price_levels.achieved_balance.balance_score < 5.0);
    assert!(adaptive_params.volatility.achieved_balance.balance_score < 5.0);
    assert!(adaptive_params.sentiment.achieved_balance.balance_score < 5.0);
    assert!(adaptive_params.volume.achieved_balance.balance_score < 5.0);

    // Verify class distributions are more balanced than default
    let direction_balance = &adaptive_params.direction.achieved_balance;
    let max_class_pct = direction_balance
        .class_percentages
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let min_class_pct = direction_balance
        .class_percentages
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    // Should be closer to 20% per class than completely imbalanced
    assert!(
        *max_class_pct < 50.0,
        "Max class percentage too high: {}",
        max_class_pct
    );
    assert!(
        *min_class_pct > 5.0,
        "Min class percentage too low: {}",
        min_class_pct
    );

    println!("✅ Adaptive parameter calibration test passed");
    println!(
        "   Direction sensitivity: {:.6}",
        adaptive_params.direction.base_sensitivity
    );
    println!(
        "   Price level bandwidth: {:.4}",
        adaptive_params.price_levels.bandwidth_size
    );
    println!(
        "   Volatility bandwidth: {:.4}",
        adaptive_params.volatility.bandwidth_size
    );
    println!(
        "   Sentiment body sensitivity: {:.3}",
        adaptive_params.sentiment.body_sensitivity
    );
    println!(
        "   Volume bandwidth: {:.3}",
        adaptive_params.volume.bandwidth_size
    );

    Ok(())
}

#[tokio::test]
async fn test_model_persistence_with_adaptive_parameters() -> Result<()> {
    // Create a model with adaptive parameters
    let config = LSTMConfig::default();
    let mut model = LSTMModel::new(config)?;

    // Create sample adaptive parameters
    let adaptive_params = AdaptiveTargetParameters {
        direction: DirectionAdaptiveParams {
            base_sensitivity: 0.012345,
            extreme_multiplier: 2.5,
            achieved_balance: ClassDistributionBalance {
                class_percentages: vec![18.5, 19.2, 20.8, 21.1, 20.4],
                imbalance_ratio: 1.14,
                balance_score: 1.23,
            },
            calibration_metadata: CalibrationMetadata {
                calibration_success: true,
                optimization_time_ms: 1500,
                total_sequences_analyzed: 900,
                parameter_search_space_size: 48,
            },
        },
        price_levels: PriceLevelAdaptiveParams {
            bandwidth_size: 1.234,
            percentile_bounds: [0.15, 0.85],
            momentum_factor: 1.2,
            achieved_balance: ClassDistributionBalance {
                class_percentages: vec![19.8, 20.1, 19.9, 20.3, 19.9],
                imbalance_ratio: 1.02,
                balance_score: 0.45,
            },
            calibration_metadata: CalibrationMetadata {
                calibration_success: true,
                optimization_time_ms: 2100,
                total_sequences_analyzed: 500,
                parameter_search_space_size: 40,
            },
        },
        volatility: VolatilityAdaptiveParams {
            bandwidth_size: 0.456,
            extreme_multiplier: 2.2,
            achieved_balance: ClassDistributionBalance {
                class_percentages: vec![20.5, 19.8, 20.2, 19.7, 19.8],
                imbalance_ratio: 1.04,
                balance_score: 0.67,
            },
            calibration_metadata: CalibrationMetadata {
                calibration_success: true,
                optimization_time_ms: 1800,
                total_sequences_analyzed: 900,
                parameter_search_space_size: 36,
            },
        },
    };

    // Set adaptive parameters in model
    model.adaptive_target_parameters = Some(adaptive_params.clone());

    // Save model to temporary file
    let temp_path = std::env::temp_dir().join("test_adaptive_model");
    model.save(&temp_path)?;

    // Load model from file
    let loaded_model = LSTMModel::load(&temp_path)?;

    // Verify adaptive parameters were preserved
    assert!(loaded_model.adaptive_target_parameters.is_some());
    let loaded_params = loaded_model.adaptive_target_parameters.unwrap();

    // Verify direction parameters
    assert!(
        (loaded_params.direction.base_sensitivity - adaptive_params.direction.base_sensitivity)
            .abs()
            < 1e-6
    );
    assert!(
        (loaded_params.direction.extreme_multiplier - adaptive_params.direction.extreme_multiplier)
            .abs()
            < 1e-6
    );
    assert_eq!(
        loaded_params
            .direction
            .achieved_balance
            .class_percentages
            .len(),
        5
    );

    // Verify price level parameters
    assert!(
        (loaded_params.price_levels.bandwidth_size - adaptive_params.price_levels.bandwidth_size)
            .abs()
            < 1e-6
    );
    assert_eq!(
        loaded_params.price_levels.percentile_bounds,
        adaptive_params.price_levels.percentile_bounds
    );
    assert!(
        (loaded_params.price_levels.momentum_factor - adaptive_params.price_levels.momentum_factor)
            .abs()
            < 1e-6
    );

    // Verify volatility parameters
    assert!(
        (loaded_params.volatility.bandwidth_size - adaptive_params.volatility.bandwidth_size).abs()
            < 1e-6
    );
    assert!(
        (loaded_params.volatility.extreme_multiplier
            - adaptive_params.volatility.extreme_multiplier)
            .abs()
            < 1e-6
    );

    // Verify metadata
    assert!(
        loaded_params
            .direction
            .calibration_metadata
            .calibration_success
    );
    assert!(
        loaded_params
            .price_levels
            .calibration_metadata
            .calibration_success
    );
    assert!(
        loaded_params
            .volatility
            .calibration_metadata
            .calibration_success
    );

    // Clean up
    std::fs::remove_file(temp_path.with_extension("safetensors")).ok();
    std::fs::remove_file(temp_path.with_extension("config")).ok();

    println!("✅ Model persistence with adaptive parameters test passed");

    Ok(())
}

#[tokio::test]
async fn test_target_generation_consistency() -> Result<()> {
    // Create sample data
    let ohlcv_data = create_sample_ohlcv_data(500);
    let df = create_sample_dataframe(&ohlcv_data)?;
    let sequence_length = 60;
    let horizon_steps = 24;
    let sequence_indices: Vec<usize> = (0..400).step_by(5).collect();
    let horizons = vec!["1h".to_string()];

    // Create base configuration
    let base_config = TargetsConfig::default();

    // Calibrate adaptive parameters
    let adaptive_params = calibrate_adaptive_parameters(
        base_config.clone(),
        &ohlcv_data,
        sequence_length,
        horizon_steps,
        &sequence_indices,
    )
    .await?;

    // Generate targets WITHOUT adaptive parameters (calibration mode)
    let targets_without_adaptive = generate_direction_targets(
        &df,
        &horizons,
        &base_config,
        &sequence_indices,
        sequence_length,
    )?;

    // Generate targets WITH adaptive parameters (prediction mode)
    let targets_with_adaptive = generate_direction_targets_with_adaptive_params(
        &df,
        &horizons,
        &base_config,
        &sequence_indices,
        sequence_length,
        Some(&adaptive_params.direction),
    )?;

    // Verify both methods produce targets
    assert!(targets_without_adaptive.contains_key("1h"));
    assert!(targets_with_adaptive.contains_key("1h"));

    let targets_1 = &targets_without_adaptive["1h"];
    let targets_2 = &targets_with_adaptive["1h"];

    // Both should have same length
    assert_eq!(targets_1.len(), targets_2.len());

    // Calculate class distributions
    let dist_1 = calculate_class_distribution_balance(targets_1);
    let dist_2 = calculate_class_distribution_balance(targets_2);

    // Adaptive parameters should produce more balanced distribution
    assert!(
        dist_2.balance_score <= dist_1.balance_score + 0.5,
        "Adaptive parameters should produce better or similar balance: {} vs {}",
        dist_2.balance_score,
        dist_1.balance_score
    );

    println!("✅ Target generation consistency test passed");
    println!(
        "   Without adaptive - Balance score: {:.3}",
        dist_1.balance_score
    );
    println!(
        "   With adaptive - Balance score: {:.3}",
        dist_2.balance_score
    );

    Ok(())
}

#[tokio::test]
async fn test_multi_target_generator_with_adaptive_params() -> Result<()> {
    // Create sample data
    let ohlcv_data = create_sample_ohlcv_data(300);
    let df = create_sample_dataframe(&ohlcv_data)?;
    let sequence_length = 30;
    let sequence_indices: Vec<usize> = (0..200).step_by(3).collect();

    // Create target generator
    let config = MultiTargetConfig {
        price_level_config: PriceLevelConfig {
            bandwidth_size: 1.0,
        },
        horizons: vec!["1h".to_string()],
    };
    let generator = TargetGenerator::new(config);

    // Create sample adaptive parameters
    let adaptive_params = AdaptiveTargetParameters {
        direction: DirectionAdaptiveParams {
            base_sensitivity: 0.015,
            extreme_multiplier: 2.0,
            achieved_balance: ClassDistributionBalance {
                class_percentages: vec![20.0, 20.0, 20.0, 20.0, 20.0],
                imbalance_ratio: 1.0,
                balance_score: 0.0,
            },
            calibration_metadata: CalibrationMetadata {
                calibration_success: true,
                optimization_time_ms: 1000,
                total_sequences_analyzed: 100,
                parameter_search_space_size: 20,
            },
        },
        price_levels: PriceLevelAdaptiveParams {
            bandwidth_size: 1.2,
            percentile_bounds: [0.2, 0.8],
            momentum_factor: 1.1,
            achieved_balance: ClassDistributionBalance {
                class_percentages: vec![20.0, 20.0, 20.0, 20.0, 20.0],
                imbalance_ratio: 1.0,
                balance_score: 0.0,
            },
            calibration_metadata: CalibrationMetadata {
                calibration_success: true,
                optimization_time_ms: 1200,
                total_sequences_analyzed: 100,
                parameter_search_space_size: 25,
            },
        },
        volatility: VolatilityAdaptiveParams {
            bandwidth_size: 0.5,
            extreme_multiplier: 2.5,
            achieved_balance: ClassDistributionBalance {
                class_percentages: vec![20.0, 20.0, 20.0, 20.0, 20.0],
                imbalance_ratio: 1.0,
                balance_score: 0.0,
            },
            calibration_metadata: CalibrationMetadata {
                calibration_success: true,
                optimization_time_ms: 1100,
                total_sequences_analyzed: 100,
                parameter_search_space_size: 18,
            },
        },
    };

    // Generate targets with adaptive parameters
    let prepared_targets = generator
        .generate_all_targets_with_adaptive_params(
            &df,
            None, // No model config
            &sequence_indices,
            sequence_length,
            Some(&adaptive_params),
        )
        .await?;

    // Verify all target types were generated
    assert!(prepared_targets.price_levels.contains_key("1h"));
    assert!(prepared_targets.directions.contains_key("1h"));
    assert!(prepared_targets.volatility.contains_key("1h"));

    // Verify target lengths match sequence count
    let expected_length = sequence_indices.len();
    assert_eq!(prepared_targets.price_levels["1h"].len(), expected_length);
    assert_eq!(prepared_targets.directions["1h"].len(), expected_length);
    assert_eq!(prepared_targets.volatility["1h"].len(), expected_length);

    // Verify targets validation passes
    prepared_targets.validate()?;

    // Calculate statistics
    let stats = prepared_targets.calculate_statistics();
    assert!(stats.price_level_stats.contains_key("1h"));
    assert!(stats.direction_stats.contains_key("1h"));
    assert!(stats.volatility_stats.contains_key("1h"));

    println!("✅ Multi-target generator with adaptive parameters test passed");
    println!(
        "   Generated {} sequences across all target types",
        expected_length
    );

    Ok(())
}

#[test]
fn test_class_distribution_balance_calculation() {
    // Test perfectly balanced distribution
    let perfect_targets = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4]; // 2 of each class
    let perfect_balance = calculate_class_distribution_balance(&perfect_targets);

    assert_eq!(perfect_balance.class_percentages.len(), 5);
    for &pct in &perfect_balance.class_percentages {
        assert!((pct - 20.0).abs() < 1e-6); // Should be exactly 20%
    }
    assert!((perfect_balance.imbalance_ratio - 1.0).abs() < 1e-6); // Perfect balance
    assert!(perfect_balance.balance_score < 0.1); // Very low score for perfect balance

    // Test imbalanced distribution
    let imbalanced_targets = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 2]; // 80% class 0, 10% each for 1,2
    let imbalanced_balance = calculate_class_distribution_balance(&imbalanced_targets);

    assert!(imbalanced_balance.imbalance_ratio > 4.0); // High imbalance
    assert!(imbalanced_balance.balance_score > 10.0); // High score for poor balance

    println!("✅ Class distribution balance calculation test passed");
    println!(
        "   Perfect balance score: {:.3}",
        perfect_balance.balance_score
    );
    println!(
        "   Imbalanced score: {:.3}",
        imbalanced_balance.balance_score
    );
}

#[test]
fn test_adaptive_parameter_serialization() {
    // Create sample adaptive parameters
    let params = AdaptiveTargetParameters {
        direction: DirectionAdaptiveParams {
            base_sensitivity: 0.123456,
            extreme_multiplier: 2.5,
            achieved_balance: ClassDistributionBalance {
                class_percentages: vec![18.5, 19.2, 20.8, 21.1, 20.4],
                imbalance_ratio: 1.14,
                balance_score: 1.23,
            },
            calibration_metadata: CalibrationMetadata {
                calibration_success: true,
                optimization_time_ms: 1500,
                total_sequences_analyzed: 900,
                parameter_search_space_size: 48,
            },
        },
        price_levels: PriceLevelAdaptiveParams {
            bandwidth_size: 1.234,
            percentile_bounds: [0.15, 0.85],
            momentum_factor: 1.2,
            achieved_balance: ClassDistributionBalance {
                class_percentages: vec![19.8, 20.1, 19.9, 20.3, 19.9],
                imbalance_ratio: 1.02,
                balance_score: 0.45,
            },
            calibration_metadata: CalibrationMetadata {
                calibration_success: true,
                optimization_time_ms: 2100,
                total_sequences_analyzed: 500,
                parameter_search_space_size: 40,
            },
        },
        volatility: VolatilityAdaptiveParams {
            bandwidth_size: 0.456,
            extreme_multiplier: 2.2,
            achieved_balance: ClassDistributionBalance {
                class_percentages: vec![20.5, 19.8, 20.2, 19.7, 19.8],
                imbalance_ratio: 1.04,
                balance_score: 0.67,
            },
            calibration_metadata: CalibrationMetadata {
                calibration_success: true,
                optimization_time_ms: 1800,
                total_sequences_analyzed: 900,
                parameter_search_space_size: 36,
            },
        },
    };

    // Test serialization
    let serialized = bincode::serialize(&params).expect("Serialization should succeed");
    assert!(!serialized.is_empty());

    // Test deserialization
    let deserialized: AdaptiveTargetParameters =
        bincode::deserialize(&serialized).expect("Deserialization should succeed");

    // Verify all fields are preserved
    assert!(
        (deserialized.direction.base_sensitivity - params.direction.base_sensitivity).abs() < 1e-10
    );
    assert!(
        (deserialized.price_levels.bandwidth_size - params.price_levels.bandwidth_size).abs()
            < 1e-10
    );
    assert!(
        (deserialized.volatility.bandwidth_size - params.volatility.bandwidth_size).abs() < 1e-10
    );

    assert_eq!(
        deserialized
            .direction
            .achieved_balance
            .class_percentages
            .len(),
        5
    );
    assert_eq!(
        deserialized
            .price_levels
            .achieved_balance
            .class_percentages
            .len(),
        5
    );
    assert_eq!(
        deserialized
            .volatility
            .achieved_balance
            .class_percentages
            .len(),
        5
    );

    assert!(
        deserialized
            .direction
            .calibration_metadata
            .calibration_success
    );
    assert!(
        deserialized
            .price_levels
            .calibration_metadata
            .calibration_success
    );
    assert!(
        deserialized
            .volatility
            .calibration_metadata
            .calibration_success
    );

    println!("✅ Adaptive parameter serialization test passed");
    println!("   Serialized size: {} bytes", serialized.len());
}
