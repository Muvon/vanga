//! Tests for sentiment target generation
//!
//! This module tests the actual sentiment generation methods to ensure
//! they work correctly with real market data scenarios.

use crate::config::model::TargetsConfig;
use crate::data::structures::MarketDataRow;
use crate::targets::adaptive_parameters::SentimentAdaptiveParams;
use crate::targets::sentiment::{
    calculate_sequence_sentiment_score, calculate_sequence_sentiment_score_with_weighting,
    classify_sentiment, generate_sentiment_targets,
    generate_sentiment_targets_with_adaptive_params, get_sentiment_class_names,
    reconstruct_sentiment, SentimentConfig,
};
use polars::prelude::*;

/// Create test OHLCV data for sentiment analysis
fn create_test_ohlcv_data() -> Vec<MarketDataRow> {
    vec![
        // Strong bullish candle (green body, high volume)
        MarketDataRow::new(1640995200, 50000.0, 52000.0, 49500.0, 51500.0, 1000.0),
        // Bearish candle (red body, moderate volume)
        MarketDataRow::new(1640998800, 51500.0, 51800.0, 50000.0, 50200.0, 800.0),
        // Doji-like candle (small body, neutral sentiment)
        MarketDataRow::new(1641002400, 50200.0, 50400.0, 49800.0, 50100.0, 600.0),
        // Strong bearish candle (large red body, high volume)
        MarketDataRow::new(1641006000, 50100.0, 50200.0, 48000.0, 48500.0, 1200.0),
        // Moderate bullish candle (green body, average volume)
        MarketDataRow::new(1641009600, 48500.0, 49800.0, 48200.0, 49200.0, 700.0),
    ]
}

/// Create test DataFrame from OHLCV data
fn create_test_dataframe() -> DataFrame {
    let ohlcv_data = create_test_ohlcv_data();

    let timestamps: Vec<i64> = ohlcv_data.iter().map(|row| row.timestamp).collect();
    let opens: Vec<f64> = ohlcv_data.iter().map(|row| row.open).collect();
    let highs: Vec<f64> = ohlcv_data.iter().map(|row| row.high).collect();
    let lows: Vec<f64> = ohlcv_data.iter().map(|row| row.low).collect();
    let closes: Vec<f64> = ohlcv_data.iter().map(|row| row.close).collect();
    let volumes: Vec<f64> = ohlcv_data.iter().map(|row| row.volume).collect();

    DataFrame::new(vec![
        Series::new("timestamp", timestamps),
        Series::new("open", opens),
        Series::new("high", highs),
        Series::new("low", lows),
        Series::new("close", closes),
        Series::new("volume", volumes),
    ])
    .expect("Failed to create test DataFrame")
}

#[test]
fn test_sentiment_class_names() {
    let class_names = get_sentiment_class_names();

    assert_eq!(class_names.len(), 5);
    assert_eq!(class_names[0], "STRONG_PANIC");
    assert_eq!(class_names[1], "MODERATE_PANIC");
    assert_eq!(class_names[2], "NEUTRAL");
    assert_eq!(class_names[3], "MODERATE_GREED");
    assert_eq!(class_names[4], "STRONG_GREED");
}

#[test]
fn test_calculate_sequence_sentiment_score() {
    let ohlcv_data = create_test_ohlcv_data();

    // Test basic sentiment score calculation
    let score = calculate_sequence_sentiment_score(&ohlcv_data);

    // Score should be a valid number
    assert!(score.is_finite());
    println!("Sequence sentiment score: {:.6}", score);
}

#[test]
fn test_calculate_sequence_sentiment_score_with_weighting() {
    let ohlcv_data = create_test_ohlcv_data();

    // Test with different weightings
    let score_uniform = calculate_sequence_sentiment_score_with_weighting(&ohlcv_data, 1.0);
    let score_weighted = calculate_sequence_sentiment_score_with_weighting(&ohlcv_data, 0.5);

    // Both scores should be valid
    assert!(score_uniform.is_finite());
    assert!(score_weighted.is_finite());

    println!("Uniform weighting score: {:.6}", score_uniform);
    println!("Weighted score (0.5): {:.6}", score_weighted);
}

#[test]
fn test_classify_sentiment() {
    let sequence_data = create_test_ohlcv_data();
    let horizon_data = create_test_ohlcv_data(); // Use same data for horizon
    let targets_config = TargetsConfig::default();
    let sentiment_config = SentimentConfig::default();

    // Test sentiment classification
    let result = classify_sentiment(
        &sequence_data,
        &horizon_data,
        &targets_config,
        &sentiment_config,
        None,
    );

    assert!(result.is_ok());
    let class = result.unwrap();

    // Should be a valid sentiment class (0-4)
    assert!((0..=4).contains(&class));
    println!(
        "Classified sentiment class: {} ({})",
        class,
        get_sentiment_class_names()[class as usize]
    );
}

#[test]
fn test_classify_sentiment_edge_cases() {
    let targets_config = TargetsConfig::default();
    let sentiment_config = SentimentConfig::default();

    // Test with single candle
    let single_candle = vec![MarketDataRow::new(
        1640995200, 50000.0, 50100.0, 49900.0, 50050.0, 500.0,
    )];
    let result = classify_sentiment(
        &single_candle,
        &single_candle,
        &targets_config,
        &sentiment_config,
        None,
    );
    assert!(result.is_ok());

    // Test with empty data
    let empty_data: Vec<MarketDataRow> = vec![];
    let result = classify_sentiment(
        &empty_data,
        &empty_data,
        &targets_config,
        &sentiment_config,
        None,
    );
    assert!(result.is_err()); // Should return error for empty data
}

#[test]
fn test_generate_sentiment_targets() {
    let df = create_test_dataframe();
    let horizons = vec!["1h".to_string()];
    let config = TargetsConfig::default();
    let sequence_indices = vec![0, 1]; // Test with first two sequences
    let sequence_length = 3;

    let result =
        generate_sentiment_targets(&df, &horizons, &config, &sequence_indices, sequence_length);

    assert!(result.is_ok());
    let targets = result.unwrap();

    // Should have targets for the specified horizon
    assert!(targets.contains_key("1h"));
    let horizon_targets = &targets["1h"];

    // Should have targets for each sequence
    assert_eq!(horizon_targets.len(), sequence_indices.len());

    // All targets should be valid classes (0-4) or -1 for invalid
    for &target in horizon_targets {
        assert!((-1..=4).contains(&target));
    }

    println!("Generated sentiment targets: {:?}", horizon_targets);
}

#[test]
fn test_generate_sentiment_targets_with_adaptive_params() {
    let df = create_test_dataframe();
    let horizons = vec!["1h".to_string()];
    let config = TargetsConfig::default();
    let sequence_indices = vec![0, 1];
    let sequence_length = 3;

    // Create adaptive parameters
    // Create adaptive parameters
    let adaptive_params = SentimentAdaptiveParams {
        body_sensitivity: 0.8,
        volume_weight: 1.2,
        consistency_factor: 0.9,
        horizon_decay_factor: 1.0,
        achieved_balance: crate::targets::adaptive_parameters::ClassDistributionBalance {
            class_percentages: [0.20, 0.18, 0.22, 0.20, 0.20],
            imbalance_ratio: 1.22,
            total_samples: 100,
            balance_score: 0.85,
        },
    };

    let result = generate_sentiment_targets_with_adaptive_params(
        &df,
        &horizons,
        &config,
        &sequence_indices,
        sequence_length,
        Some(&adaptive_params),
    );

    assert!(result.is_ok());
    let targets = result.unwrap();

    assert!(targets.contains_key("1h"));
    let horizon_targets = &targets["1h"];
    assert_eq!(horizon_targets.len(), sequence_indices.len());

    println!("Adaptive sentiment targets: {:?}", horizon_targets);
}

#[test]
fn test_reconstruct_sentiment() {
    // Test with sample probabilities (should sum to 1.0)
    let probabilities = vec![0.1, 0.2, 0.4, 0.2, 0.1]; // Neutral-leaning distribution
    let sequence_sentiment = 0.15; // Sample sentiment score
    let thresholds = [-0.8, -0.3, 0.3, 0.8]; // Sample thresholds

    let result = reconstruct_sentiment(&probabilities, sequence_sentiment, &thresholds);

    assert!(result.is_ok());
    let reconstruction = result.unwrap();

    // Should have sentiment ranges for each class
    assert_eq!(reconstruction.sentiment_ranges.len(), 5);

    // Check that we have valid ranges (some might be infinite for extreme classes)
    assert!(reconstruction.probabilities.len() == 5);
    assert!(reconstruction.confidence >= 0.0 && reconstruction.confidence <= 1.0);
    assert!(reconstruction.most_likely_class < 5);

    println!(
        "Reconstructed sentiment ranges: {:?}",
        reconstruction.sentiment_ranges
    );
    println!(
        "Most likely class: {} with confidence: {:.3}",
        reconstruction.most_likely_class, reconstruction.confidence
    );
}
#[test]
fn test_reconstruct_sentiment_edge_cases() {
    let sequence_sentiment = 0.0;
    let thresholds = [-0.8, -0.3, 0.3, 0.8];

    // Test with invalid probabilities (wrong length)
    let invalid_probs = vec![0.5, 0.5]; // Only 2 classes instead of 5
    let result = reconstruct_sentiment(&invalid_probs, sequence_sentiment, &thresholds);
    assert!(result.is_err());

    // Test with probabilities that don't sum to 1.0 (should still work)
    let unnormalized_probs = vec![0.2, 0.4, 0.8, 0.4, 0.2];
    let result = reconstruct_sentiment(&unnormalized_probs, sequence_sentiment, &thresholds);
    assert!(result.is_ok());
}

#[test]
fn test_sentiment_consistency_across_calls() {
    let df = create_test_dataframe();
    let horizons = vec!["1h".to_string()];
    let config = TargetsConfig::default();
    let sequence_indices = vec![0, 1];
    let sequence_length = 3;

    // Generate targets multiple times with same parameters
    let result1 =
        generate_sentiment_targets(&df, &horizons, &config, &sequence_indices, sequence_length);
    let result2 =
        generate_sentiment_targets(&df, &horizons, &config, &sequence_indices, sequence_length);

    assert!(result1.is_ok());
    assert!(result2.is_ok());

    let targets1 = result1.unwrap();
    let targets2 = result2.unwrap();

    // Results should be consistent (same inputs = same outputs)
    assert_eq!(targets1["1h"], targets2["1h"]);

    println!("Consistent sentiment targets: {:?}", targets1["1h"]);
}

#[test]
fn test_sentiment_with_different_horizons() {
    let df = create_test_dataframe();
    let horizons = vec!["1h".to_string(), "2h".to_string()];
    let config = TargetsConfig::default();
    let sequence_indices = vec![0]; // Only one sequence to avoid boundary issues
    let sequence_length = 2;

    let result =
        generate_sentiment_targets(&df, &horizons, &config, &sequence_indices, sequence_length);

    assert!(result.is_ok());
    let targets = result.unwrap();

    // Should have targets for both horizons
    assert!(targets.contains_key("1h"));
    assert!(targets.contains_key("2h"));

    println!("1h sentiment targets: {:?}", targets["1h"]);
    println!("2h sentiment targets: {:?}", targets["2h"]);
}

#[test]
fn test_sentiment_config_default() {
    let config = SentimentConfig::default();

    // Verify default values are reasonable
    assert!(config.body_sensitivity > 0.0);
    assert!(config.volume_weight > 0.0);
    assert!(config.consistency_factor > 0.0);

    println!("Default sentiment config: {:?}", config);
}
