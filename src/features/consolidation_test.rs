use super::technical::*;
use crate::config::features::TechnicalIndicatorsConfig;
use polars::prelude::*;

/// Helper to create test OHLCV data
fn create_test_ohlcv_data(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let open: Vec<f64> = (0..len).map(|i| 100.0 + i as f64).collect();
    let high: Vec<f64> = (0..len).map(|i| 105.0 + i as f64).collect();
    let low: Vec<f64> = (0..len).map(|i| 95.0 + i as f64).collect();
    let close: Vec<f64> = (0..len).map(|i| 100.0 + i as f64).collect();
    (open, high, low, close)
}

/// Helper to create test DataFrame with OHLCV data
fn create_test_dataframe(len: usize) -> DataFrame {
    let (open, high, low, close) = create_test_ohlcv_data(len);
    let volume: Vec<f64> = vec![1000.0; len];
    let timestamp: Vec<i64> = (0..len).map(|i| i as i64).collect();

    DataFrame::new(vec![
        Series::new("timestamp".into(), timestamp).into_column(),
        Series::new("open".into(), open).into_column(),
        Series::new("high".into(), high).into_column(),
        Series::new("low".into(), low).into_column(),
        Series::new("close".into(), close).into_column(),
        Series::new("volume".into(), volume).into_column(),
    ])
    .unwrap()
}

#[test]
fn test_range_tightness_calculation() {
    let high = vec![105.0, 110.0, 108.0, 112.0, 115.0];
    let low = vec![95.0, 98.0, 102.0, 108.0, 113.0];
    let close = vec![100.0, 105.0, 105.0, 110.0, 114.0];

    let result = super::technical::calculate_range_tightness(&high, &low, &close);

    assert_eq!(result.len(), 5);
    // First value: (105 - 95) / 100 = 0.1
    assert!((result[0] - 0.1).abs() < 1e-6);
    // Second value: (110 - 98) / 105 ≈ 0.114
    assert!((result[1] - 0.114285).abs() < 1e-4);
    // Fourth value: (112 - 108) / 110 ≈ 0.036 (tight range = consolidation)
    assert!((result[3] - 0.036363).abs() < 1e-4);
}

#[test]
fn test_range_tightness_consolidation_detection() {
    // Simulate consolidation: tight range
    let high = vec![101.0, 101.5, 101.2, 101.3, 101.1];
    let low = vec![99.0, 99.5, 99.8, 99.7, 99.9];
    let close = vec![100.0, 100.5, 100.0, 100.2, 100.1];

    let result = super::technical::calculate_range_tightness(&high, &low, &close);

    // All values should be small (< 0.03) indicating tight consolidation
    for &val in &result {
        assert!(
            val < 0.03,
            "Range tightness {} should indicate consolidation",
            val
        );
    }
}

#[test]
fn test_bollinger_squeeze_calculation() {
    let close = vec![100.0, 102.0, 101.0, 103.0, 102.5, 104.0, 103.5, 105.0];
    let period = 5;

    let result = super::technical::calculate_bollinger_squeeze(&close, period, 2.0);

    assert_eq!(result.len(), 8);
    // First 4 values should be NaN (warmup period)
    for value in result.iter().take(4) {
        assert!(value.is_nan());
    }
    // Values after warmup should be valid
    for value in result.iter().skip(4) {
        assert!(!value.is_nan());
        assert!(*value > 0.0);
    }
}

#[test]
fn test_bollinger_squeeze_detects_consolidation() {
    // Simulate tight consolidation (low volatility)
    let close_consolidation = vec![100.0, 100.1, 100.2, 100.1, 100.0, 100.1, 100.2];
    let result_consolidation =
        super::technical::calculate_bollinger_squeeze(&close_consolidation, 5, 2.0);

    // Simulate high volatility (trending)
    let close_trending = vec![100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0];
    let result_trending = super::technical::calculate_bollinger_squeeze(&close_trending, 5, 2.0);

    // Consolidation should have smaller squeeze values than trending
    let consolidation_avg = result_consolidation[4..]
        .iter()
        .filter(|x| !x.is_nan())
        .sum::<f64>()
        / 3.0;
    let trending_avg = result_trending[4..]
        .iter()
        .filter(|x| !x.is_nan())
        .sum::<f64>()
        / 3.0;

    assert!(
        consolidation_avg < trending_avg,
        "Consolidation squeeze ({}) should be smaller than trending squeeze ({})",
        consolidation_avg,
        trending_avg
    );
}

#[test]
fn test_choppiness_index_calculation() {
    let high = vec![
        105.0, 110.0, 108.0, 112.0, 115.0, 118.0, 116.0, 120.0, 122.0, 125.0, 128.0, 130.0, 129.0,
        133.0, 136.0, 138.0, 137.0, 140.0, 143.0, 145.0,
    ];
    let low = vec![
        95.0, 98.0, 102.0, 108.0, 110.0, 113.0, 111.0, 115.0, 118.0, 120.0, 123.0, 125.0, 124.0,
        128.0, 131.0, 133.0, 132.0, 135.0, 138.0, 140.0,
    ];
    let close = vec![
        100.0, 105.0, 105.0, 110.0, 113.0, 116.0, 114.0, 118.0, 120.0, 123.0, 126.0, 128.0, 127.0,
        131.0, 134.0, 136.0, 135.0, 138.0, 141.0, 143.0,
    ];

    let result = super::technical::calculate_choppiness_index(&high, &low, &close, 14);

    assert_eq!(result.len(), 20);
    // First values should be NaN (warmup period)
    for value in result.iter().take(14) {
        assert!(value.is_nan());
    }
    // Values after warmup should be valid and in range [0, 100]
    for value in result.iter().skip(14) {
        assert!(!value.is_nan());
        assert!(*value >= 0.0 && *value <= 100.0);
    }
}

#[test]
fn test_choppiness_consolidation_vs_trending() {
    // Consolidation: choppy sideways movement
    let high_chop = vec![
        102.0, 101.0, 103.0, 100.0, 102.0, 101.0, 103.0, 100.0, 102.0, 101.0, 103.0, 100.0, 102.0,
        101.0, 103.0, 100.0, 102.0, 101.0, 103.0, 100.0,
    ];
    let low_chop = vec![
        98.0, 99.0, 97.0, 100.0, 98.0, 99.0, 97.0, 100.0, 98.0, 99.0, 97.0, 100.0, 98.0, 99.0,
        97.0, 100.0, 98.0, 99.0, 97.0, 100.0,
    ];
    let close_chop = vec![
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    ];

    // Trending: smooth upward movement
    let high_trend: Vec<f64> = (0..20).map(|i| 105.0 + i as f64 * 2.0).collect();
    let low_trend: Vec<f64> = (0..20).map(|i| 95.0 + i as f64 * 2.0).collect();
    let close_trend: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();

    let chop_consolidation =
        super::technical::calculate_choppiness_index(&high_chop, &low_chop, &close_chop, 14);
    let chop_trending =
        super::technical::calculate_choppiness_index(&high_trend, &low_trend, &close_trend, 14);

    // Choppiness for consolidation should be higher (>61.8) than trending (<38.2)
    if let (Some(&chop_val), Some(&trend_val)) = (chop_consolidation.last(), chop_trending.last()) {
        if !chop_val.is_nan() && !trend_val.is_nan() {
            assert!(
                chop_val > trend_val,
                "Consolidation choppiness ({}) should be higher than trending choppiness ({})",
                chop_val,
                trend_val
            );
        }
    }
}

#[test]
fn test_price_efficiency_ratio_calculation() {
    let close = vec![
        100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0,
    ];

    let result = super::technical::calculate_price_efficiency_ratio(&close, 5);

    assert_eq!(result.len(), 10);
    // First 4 values should be NaN (warmup period)
    for value in result.iter().take(4) {
        assert!(value.is_nan());
    }
    // Values after warmup should be valid and in range [0, 1]
    for value in result.iter().skip(4) {
        assert!(!value.is_nan());
        assert!(*value >= 0.0 && *value <= 1.0);
    }
}

#[test]
fn test_price_efficiency_trending_vs_choppy() {
    // Efficient trending movement
    let close_efficient: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();

    // Choppy sideways movement
    let close_choppy = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 103.0, 97.0, 102.0, 98.0, 101.0, 99.0, 103.0, 97.0, 102.0,
        98.0, 101.0, 99.0, 103.0, 97.0, 100.0,
    ];

    let efficiency_trending =
        super::technical::calculate_price_efficiency_ratio(&close_efficient, 10);
    let efficiency_choppy = super::technical::calculate_price_efficiency_ratio(&close_choppy, 10);

    // Trending should have higher efficiency than choppy
    if let (Some(&trend_val), Some(&chop_val)) =
        (efficiency_trending.last(), efficiency_choppy.last())
    {
        if !trend_val.is_nan() && !chop_val.is_nan() {
            assert!(
                trend_val > chop_val,
                "Trending efficiency ({}) should be higher than choppy efficiency ({})",
                trend_val,
                chop_val
            );
        }
    }
}

#[test]
fn test_directional_movement_balance_calculation() {
    let high = vec![
        105.0, 110.0, 108.0, 112.0, 115.0, 118.0, 116.0, 120.0, 122.0, 125.0, 128.0, 130.0, 129.0,
        133.0, 136.0, 138.0, 137.0, 140.0, 143.0, 145.0,
    ];
    let low = vec![
        95.0, 98.0, 102.0, 108.0, 110.0, 113.0, 111.0, 115.0, 118.0, 120.0, 123.0, 125.0, 124.0,
        128.0, 131.0, 133.0, 132.0, 135.0, 138.0, 140.0,
    ];
    let close = vec![
        100.0, 105.0, 105.0, 110.0, 113.0, 116.0, 114.0, 118.0, 120.0, 123.0, 126.0, 128.0, 127.0,
        131.0, 134.0, 136.0, 135.0, 138.0, 141.0, 143.0,
    ];

    let result = super::technical::calculate_directional_movement_balance(&high, &low, &close, 14);

    assert_eq!(result.len(), 20);
    // First values should be NaN (warmup period)
    for value in result.iter().take(14) {
        assert!(value.is_nan());
    }
    // Values after warmup should be valid and in range [0, 1]
    for value in result.iter().skip(14) {
        assert!(!value.is_nan());
        assert!(*value >= 0.0 && *value <= 1.0);
    }
}

#[test]
fn test_directional_balance_trending_vs_sideways() {
    // Strong uptrend (unbalanced directional movement)
    let high_trend: Vec<f64> = (0..20).map(|i| 105.0 + i as f64 * 2.0).collect();
    let low_trend: Vec<f64> = (0..20).map(|i| 95.0 + i as f64 * 2.0).collect();
    let close_trend: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();

    // Sideways (balanced directional movement)
    let high_sideways = vec![
        102.0, 101.0, 103.0, 100.0, 102.0, 101.0, 103.0, 100.0, 102.0, 101.0, 103.0, 100.0, 102.0,
        101.0, 103.0, 100.0, 102.0, 101.0, 103.0, 100.0,
    ];
    let low_sideways = vec![
        98.0, 99.0, 97.0, 100.0, 98.0, 99.0, 97.0, 100.0, 98.0, 99.0, 97.0, 100.0, 98.0, 99.0,
        97.0, 100.0, 98.0, 99.0, 97.0, 100.0,
    ];
    let close_sideways = vec![
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    ];

    let balance_trend = super::technical::calculate_directional_movement_balance(
        &high_trend,
        &low_trend,
        &close_trend,
        14,
    );
    let balance_sideways = super::technical::calculate_directional_movement_balance(
        &high_sideways,
        &low_sideways,
        &close_sideways,
        14,
    );

    // Trending should have higher balance (more one-directional)
    // Sideways should have lower balance (more balanced)
    if let (Some(&trend_val), Some(&sideways_val)) = (balance_trend.last(), balance_sideways.last())
    {
        if !trend_val.is_nan() && !sideways_val.is_nan() {
            assert!(
                trend_val > sideways_val,
                "Trending balance ({}) should be higher than sideways balance ({})",
                trend_val,
                sideways_val
            );
        }
    }
}

#[test]
fn test_add_consolidation_features_integration() {
    let df = create_test_dataframe(50);
    let config = TechnicalIndicatorsConfig::default();

    let open = extract_numeric_column(&df, "open").unwrap();
    let high = extract_numeric_column(&df, "high").unwrap();
    let low = extract_numeric_column(&df, "low").unwrap();
    let close = extract_numeric_column(&df, "close").unwrap();
    let volume = extract_numeric_column(&df, "volume").unwrap();

    let ohlcv = super::technical::OhlcvData {
        open: &open,
        high: &high,
        low: &low,
        close: &close,
        volume: &volume,
    };

    let result = super::technical::add_consolidation_features(df, &ohlcv, &config);

    assert!(result.is_ok());
    let result_df = result.unwrap();

    // Check that all NEW consolidation features were added (no ADX - it already exists in trend indicators)
    assert!(result_df.column("range_tightness").is_ok());
    assert!(result_df.column("bb_squeeze_20").is_ok());
    assert!(result_df.column("choppiness_14").is_ok());
    assert!(result_df.column("choppiness_20").is_ok());
    assert!(result_df.column("price_efficiency_10").is_ok());
    assert!(result_df.column("price_efficiency_20").is_ok());
    assert!(result_df.column("dm_balance_14").is_ok());
    assert!(result_df.column("dm_balance_20").is_ok());
}

#[test]
fn test_consolidation_features_no_nan_in_valid_range() {
    let df = create_test_dataframe(100);
    let config = TechnicalIndicatorsConfig::default();

    let open = extract_numeric_column(&df, "open").unwrap();
    let high = extract_numeric_column(&df, "high").unwrap();
    let low = extract_numeric_column(&df, "low").unwrap();
    let close = extract_numeric_column(&df, "close").unwrap();
    let volume = extract_numeric_column(&df, "volume").unwrap();

    let ohlcv = super::technical::OhlcvData {
        open: &open,
        high: &high,
        low: &low,
        close: &close,
        volume: &volume,
    };

    let result_df = super::technical::add_consolidation_features(df, &ohlcv, &config).unwrap();

    // Check range_tightness has no NaN values (it's calculated for all rows)
    let range_tightness = extract_numeric_column(&result_df, "range_tightness").unwrap();
    for val in range_tightness {
        assert!(!val.is_nan(), "range_tightness should not have NaN values");
    }

    // Check that features have valid values after warmup period
    let choppiness_14 = extract_numeric_column(&result_df, "choppiness_14").unwrap();
    let valid_count = choppiness_14.iter().filter(|x| !x.is_nan()).count();
    assert!(
        valid_count > 50,
        "Choppiness should have valid values after warmup"
    );
}
