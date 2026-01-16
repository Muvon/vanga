// Tests for data preprocessor module
use crate::config::training::{DataConfig, MissingDataStrategy};
use crate::data::preprocessor::{DataPreprocessor, ReplacementStrategy};
use polars::prelude::*;

fn make_df_with_missing() -> DataFrame {
    let ts = Series::new("timestamp".into(), &["2024-01-01T00:00:00Z"]).into_column();
    let open = Series::new("open".into(), &[Some(42000.0), None]).into_column();
    let high = Series::new("high".into(), &[42500.0, 42600.0]).into_column();
    let low = Series::new("low".into(), &[41800.0, 41900.0]).into_column();
    let close = Series::new("close".into(), &[42300.0, 42400.0]).into_column();
    let volume = Series::new("volume".into(), &[1000.0, 1200.0]).into_column();
    DataFrame::new(
        vec![ts, open, high, low, close, volume]
            .into_iter()
            .map(|s| s.into_column())
            .collect(),
    )
    .unwrap()
}

#[tokio::test]
async fn process_for_training_forward_fill() {
    let mut df = make_df_with_missing();
    let config = DataConfig::default();
    let pre = DataPreprocessor::new();
    let df2 = pre.process_for_training(df, &config, None).await.unwrap();
    assert_eq!(df2.height(), 2);
    assert!(df2.column("open").unwrap().null_count() == 0);
}

#[tokio::test]
async fn process_for_training_drop_missing() {
    let mut df = make_df_with_missing();
    let config = DataConfig::default();
    let pre = DataPreprocessor::new();
    let df2 = pre.process_for_training(df, &config, None).await.unwrap();
    assert!(df2.height() < 2); // Should drop the row with missing open
}

#[test]
fn test_interpolation_strategy_selection() {
    let preprocessor = DataPreprocessor;

    // Test that interpolation is selected for price-derived features
    assert!(matches!(
        preprocessor.get_replacement_strategy("price_return"),
        ReplacementStrategy::Interpolate
    ));
    assert!(matches!(
        preprocessor.get_replacement_strategy("momentum_5"),
        ReplacementStrategy::Interpolate
    ));
    assert!(matches!(
        preprocessor.get_replacement_strategy("volatility_10"),
        ReplacementStrategy::Interpolate
    ));

    // Test that median is selected for technical indicators
    assert!(matches!(
        preprocessor.get_replacement_strategy("rsi_14"),
        ReplacementStrategy::Median
    ));
    assert!(matches!(
        preprocessor.get_replacement_strategy("sma_20"),
        ReplacementStrategy::Median
    ));

    // Test that cap is selected for volume and other features
    assert!(matches!(
        preprocessor.get_replacement_strategy("volume"),
        ReplacementStrategy::Cap
    ));
    assert!(matches!(
        preprocessor.get_replacement_strategy("some_other_feature"),
        ReplacementStrategy::Cap
    ));
}

#[test]
fn test_interpolation_logic() {
    let preprocessor = DataPreprocessor;

    // Create test series: [1.0, 2.0, OUTLIER, 4.0, 5.0]
    let values = vec![Some(1.0), Some(2.0), Some(100.0), Some(4.0), Some(5.0)];
    let series = Series::new("test".into(), values)
        .into_column()
        .f64()
        .unwrap()
        .clone();

    // Test interpolation at index 2 (outlier value 100.0)
    let interpolated = preprocessor.interpolate_outlier_value(&series, 2, 100.0, 3.0);

    // Should interpolate between 2.0 and 4.0, giving 3.0
    assert!((interpolated - 3.0).abs() < 0.001);
}

#[test]
fn test_extract_recent_clean_data() {
    let preprocessor = DataPreprocessor::new();

    // Create test data with NaN in middle but clean at end
    let data = vec![
        1.0,
        2.0,
        f64::NAN,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0, // 10 rows
    ];
    let df = DataFrame::new(
        vec![
            Series::new("price".into(), data),
            Series::new("volume".into(), vec![100.0; 10]),
        ]
        .into_iter()
        .map(|s| s.into_column())
        .collect(),
    )
    .unwrap();

    // Extract 5 most recent clean rows
    let result = preprocessor.extract_recent_clean_data(df, 5).unwrap();

    // Should get rows 5-9 (6.0, 7.0, 8.0, 9.0, 10.0)
    assert_eq!(result.height(), 5);
    let price_col = result.column("price").unwrap().f64().unwrap();
    assert_eq!(price_col.get(0).unwrap(), 6.0);
    assert_eq!(price_col.get(4).unwrap(), 10.0);
}

#[test]
fn test_extract_recent_clean_data_insufficient() {
    let preprocessor = DataPreprocessor::new();

    // Create test data with only 3 rows
    let df = DataFrame::new(
        vec![Series::new("price".into(), vec![1.0, 2.0, 3.0])]
            .into_iter()
            .map(|s| s.into_column())
            .collect(),
    )
    .unwrap();

    // Try to extract 5 rows - should fail
    let result = preprocessor.extract_recent_clean_data(df, 5);
    assert!(result.is_err());
}
