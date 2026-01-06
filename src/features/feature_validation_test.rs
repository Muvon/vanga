use crate::config::features::TechnicalIndicatorsConfig;
use crate::features::technical::generate_technical_indicators;
use polars::prelude::*;

#[tokio::test]
async fn test_fractal_and_gaps_fixed() {
    let df = CsvReadOptions::default().with_has_header(true).try_into_reader_with_file_path(Some("data/BTCUSDT.csv".into()))
        .expect("Failed to load data")
        .finish()
        .expect("Failed to parse CSV");


    let mut config = TechnicalIndicatorsConfig::default();
    config.trend.advanced.enabled = true;

    let result_df = generate_technical_indicators(df, &config)
        .await
        .expect("Failed to generate indicators");

    // Test fractal_dimension has variation
    let fractal_col = result_df
        .column("fractal_dimension")
        .expect("fractal_dimension missing");
    let fractal_series = fractal_col.f64().expect("Should be f64");
    let values: Vec<Option<f64>> = (0..fractal_series.len())
        .map(|i| fractal_series.get(i))
        .collect();
    let unique: std::collections::HashSet<_> = values
        .iter()
        .filter_map(|x| x.map(|v| (v * 1000.0) as i64))
        .collect();

    assert!(
        unique.len() > 100,
        "fractal_dimension should have variation, got {} unique values",
        unique.len()
    );

    // Test price_gaps captures small movements
    let gaps_col = result_df.column("price_gaps").expect("price_gaps missing");
    let gaps_series = gaps_col.f64().expect("Should be f64");
    let values: Vec<Option<f64>> = (0..gaps_series.len()).map(|i| gaps_series.get(i)).collect();
    let unique: std::collections::HashSet<_> = values
        .iter()
        .filter_map(|x| x.map(|v| (v * 100000.0) as i64))
        .collect();

    assert!(
        unique.len() > 100,
        "price_gaps should capture small movements, got {} unique values",
        unique.len()
    );
}
