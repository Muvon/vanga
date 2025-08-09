use crate::config::model::TargetsConfig;
use crate::targets::sentiment::*;

#[test]
fn test_sentiment_calibration_with_realistic_data() {
    // Create realistic crypto data with various volume-price patterns
    let mut test_data = Vec::new();

    // Generate 100 candles with different volume-price correlation patterns
    for i in 0..100 {
        let base_price = 100.0 + (i as f64 * 0.1);
        let price_change = (i as f64 - 50.0) * 0.002; // Range from -0.1 to +0.1
        let volume = 1000.0 + (i as f64 * 10.0); // Increasing volume

        test_data.push(MarketDataRow {
            open: base_price,
            high: base_price + price_change.abs(),
            low: base_price - price_change.abs(),
            close: base_price + price_change,
            volume,
        });
    }

    // Test calibration
    let sensitivity = calibrate_sentiment_sensitivity(&test_data, 10, 5, 0.2).unwrap();
    println!("Calibrated sensitivity: {:.6}", sensitivity);

    // Debug: Test the sentiment changes directly
    let mut sentiment_changes = Vec::new();
    for i in 0..(test_data.len() - 15) {
        let sequence_ohlcv = &test_data[i..i + 10];
        let horizon_ohlcv = &test_data[i + 10..i + 15];

        let seq_sentiment = calculate_sequence_sentiment_score(sequence_ohlcv);
        let hor_sentiment = calculate_sequence_sentiment_score(horizon_ohlcv);
        let sentiment_change = hor_sentiment - seq_sentiment;
        sentiment_changes.push(sentiment_change);
    }

    sentiment_changes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sentiment_changes.len();
    let p20 = sentiment_changes[(n as f64 * 0.20) as usize];
    let p40 = sentiment_changes[(n as f64 * 0.40) as usize];
    let p60 = sentiment_changes[(n as f64 * 0.60) as usize];
    let p80 = sentiment_changes[(n as f64 * 0.80) as usize];

    println!(
        "Sentiment changes range: {:.6} to {:.6}",
        sentiment_changes[0],
        sentiment_changes[n - 1]
    );
    println!(
        "Percentiles: 20th={:.6}, 40th={:.6}, 60th={:.6}, 80th={:.6}",
        p20, p40, p60, p80
    );
    println!("Neutral range: {:.6}", p60 - p40);

    // Test classification with calibrated sensitivity
    let config = SentimentConfig {
        body_sensitivity: sensitivity,
        volume_weight: 1.0,
        consistency_factor: 1.0,
    };

    let targets_config = TargetsConfig::default();

    // Test a few classifications
    let sequence = &test_data[0..10];
    let horizon = &test_data[10..15];

    let class = classify_sentiment(sequence, horizon, &targets_config, &config).unwrap();
    println!("Test classification: {}", class);
}

#[test]
fn test_sentiment_classification_balanced_distribution() {
    // Create test OHLCV data with various sentiment patterns
    let test_data = vec![
        // Strong panic: Large red bodies, lower wicks
        MarketDataRow {
            open: 100.0,
            high: 102.0,
            low: 90.0,
            close: 92.0,
            volume: 1000.0,
        },
        MarketDataRow {
            open: 92.0,
            high: 94.0,
            low: 85.0,
            close: 87.0,
            volume: 1200.0,
        },
        // Moderate panic: Medium red bodies
        MarketDataRow {
            open: 100.0,
            high: 103.0,
            low: 95.0,
            close: 97.0,
            volume: 800.0,
        },
        MarketDataRow {
            open: 97.0,
            high: 99.0,
            low: 93.0,
            close: 95.0,
            volume: 900.0,
        },
        // Neutral: Small bodies, balanced wicks
        MarketDataRow {
            open: 100.0,
            high: 102.0,
            low: 98.0,
            close: 101.0,
            volume: 500.0,
        },
        MarketDataRow {
            open: 101.0,
            high: 103.0,
            low: 99.0,
            close: 100.0,
            volume: 600.0,
        },
        // Moderate greed: Medium green bodies
        MarketDataRow {
            open: 100.0,
            high: 108.0,
            low: 99.0,
            close: 105.0,
            volume: 800.0,
        },
        MarketDataRow {
            open: 105.0,
            high: 110.0,
            low: 103.0,
            close: 108.0,
            volume: 900.0,
        },
        // Strong greed: Large green bodies, upper wicks
        MarketDataRow {
            open: 100.0,
            high: 120.0,
            low: 98.0,
            close: 115.0,
            volume: 1500.0,
        },
        MarketDataRow {
            open: 115.0,
            high: 125.0,
            low: 112.0,
            close: 122.0,
            volume: 1600.0,
        },
    ];

    let config = TargetsConfig::default();
    let sentiment_config = SentimentConfig::default();

    // Test classification for different sentiment patterns
    let sequence_data = &test_data[0..5];
    let horizon_data = &test_data[5..10];

    let result = classify_sentiment(sequence_data, horizon_data, &config, &sentiment_config);
    assert!(result.is_ok());

    let class = result.unwrap();
    assert!(
        (0..5).contains(&class),
        "Sentiment class should be in range 0-4, got {}",
        class
    );
}

#[test]
fn test_sentiment_metrics_calculation() {
    let test_candle = MarketDataRow {
        open: 100.0,
        high: 110.0,
        low: 95.0,
        close: 108.0,
        volume: 1000.0,
    };

    let avg_volume = 800.0;
    let result = calculate_single_candle_metrics(&test_candle, avg_volume);
    assert!(result.is_ok());

    let metrics = result.unwrap();

    // Body ratio should be positive (bullish)
    assert!(
        metrics.body_ratio > 0.0,
        "Body ratio should be positive for green candle"
    );

    // Body size should be reasonable
    assert!(metrics.body_size > 0.0, "Body size should be positive");
    assert!(metrics.body_size < 1.0, "Body size should be less than 1.0");

    // Volume confirmation should reflect higher than average volume
    assert!(
        metrics.volume_confirmation > 0.0,
        "Volume confirmation should be positive for above-average volume"
    );

    // Sentiment score should combine all metrics
    assert!(
        metrics.sentiment_score != 0.0,
        "Sentiment score should not be zero"
    );
}

#[test]
fn test_sentiment_adaptive_thresholds() {
    let sequence_scores = vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4, -0.2];

    let result = calculate_sentiment_consistency(&sequence_scores);
    assert!(result.is_ok());

    let consistency = result.unwrap();
    assert!(
        consistency > 0.0,
        "Sentiment consistency should be positive"
    );
    assert!(
        consistency >= 0.05,
        "Sentiment consistency should be at least minimum threshold"
    );
}

#[test]
fn test_sentiment_class_names() {
    // Test corrected class names (kept original PANIC/GREED terminology)
    let class_names = get_sentiment_class_names();
    assert_eq!(class_names.len(), 5);
    assert_eq!(class_names[0], "STRONG_PANIC");
    assert_eq!(class_names[1], "MODERATE_PANIC");
    assert_eq!(class_names[2], "NEUTRAL");
    assert_eq!(class_names[3], "MODERATE_GREED");
    assert_eq!(class_names[4], "STRONG_GREED");
}

#[test]
fn test_sentiment_edge_cases() {
    let config = TargetsConfig::default();
    let sentiment_config = SentimentConfig::default();

    // Test with empty data
    let empty_data: Vec<MarketDataRow> = vec![];
    let test_data = vec![MarketDataRow {
        open: 100.0,
        high: 102.0,
        low: 98.0,
        close: 101.0,
        volume: 500.0,
    }];

    let result = classify_sentiment(&empty_data, &test_data, &config, &sentiment_config);
    assert!(result.is_err(), "Should fail with empty sequence data");

    let result = classify_sentiment(&test_data, &empty_data, &config, &sentiment_config);
    assert!(result.is_err(), "Should fail with empty horizon data");

    // Test with invalid candle (high < low)
    let invalid_candle = MarketDataRow {
        open: 100.0,
        high: 95.0, // Invalid: high < low
        low: 105.0,
        close: 98.0,
        volume: 500.0,
    };

    let result = calculate_single_candle_metrics(&invalid_candle, 500.0);
    assert!(result.is_err(), "Should fail with invalid candle data");
}

#[test]
fn test_sentiment_volume_confirmation() {
    let high_volume_candle = MarketDataRow {
        open: 100.0,
        high: 110.0,
        low: 98.0,
        close: 108.0,
        volume: 2000.0, // High volume
    };

    let low_volume_candle = MarketDataRow {
        open: 100.0,
        high: 110.0,
        low: 98.0,
        close: 108.0,
        volume: 200.0, // Low volume
    };

    let avg_volume = 1000.0;

    let high_vol_result = calculate_single_candle_metrics(&high_volume_candle, avg_volume).unwrap();
    let low_vol_result = calculate_single_candle_metrics(&low_volume_candle, avg_volume).unwrap();

    // High volume should have higher volume confirmation
    assert!(
        high_vol_result.volume_confirmation > low_vol_result.volume_confirmation,
        "High volume candle should have higher volume confirmation"
    );

    // High volume should result in higher sentiment score (all else equal)
    assert!(
        high_vol_result.sentiment_score.abs() > low_vol_result.sentiment_score.abs(),
        "High volume should amplify sentiment score"
    );
}

#[test]
fn test_sentiment_reconstruction() {
    let probabilities = vec![0.1, 0.2, 0.4, 0.2, 0.1];
    let sequence_sentiment = 0.0;
    let thresholds = [0.3, 0.1, 0.1, 0.3]; // [panic_extreme, panic_moderate, greed_moderate, greed_extreme]

    let result = reconstruct_sentiment(&probabilities, sequence_sentiment, &thresholds);
    assert!(result.is_ok());

    let reconstruction = result.unwrap();
    assert_eq!(reconstruction.probabilities.len(), 5);
    assert_eq!(reconstruction.most_likely_class, 2); // Neutral has highest probability
    assert!(reconstruction.confidence > 0.0);
    assert!(reconstruction.confidence <= 1.0);
    assert_eq!(reconstruction.sentiment_ranges.len(), 5);

    // Test with invalid probabilities
    let invalid_probs = vec![0.1, 0.2, 0.3]; // Wrong length
    let result = reconstruct_sentiment(&invalid_probs, sequence_sentiment, &thresholds);
    assert!(
        result.is_err(),
        "Should fail with wrong number of probabilities"
    );
}
