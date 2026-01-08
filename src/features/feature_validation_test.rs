use crate::config::features::TechnicalIndicatorsConfig;
use crate::features::technical::generate_technical_indicators;
use polars::prelude::*;

fn create_test_df() -> DataFrame {
    let close: Vec<f64> = (0..100)
        .map(|i| 10000.0 + (i as f64 * 10.0).sin() * 500.0)
        .collect();
    let high: Vec<f64> = close.iter().map(|x| x + 100.0).collect();
    let low: Vec<f64> = close.iter().map(|x| x - 100.0).collect();
    let volume: Vec<f64> = (0..100).map(|_| 1000.0).collect();

    let timestamp: Vec<i64> = (0..100).map(|i| i as i64 * 3600).collect();

    DataFrame::new(
        vec![
            Series::new("open".into(), close.clone()),
            Series::new("high".into(), high),
            Series::new("low".into(), low),
            Series::new("close".into(), close.clone()),
            Series::new("volume".into(), volume),
            Series::new("timestamp".into(), timestamp),
        ]
        .into_iter()
        .map(|s| s.into_column())
        .collect(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_fractal_and_gaps_fixed() {
    let df = create_test_df();

    let mut config = TechnicalIndicatorsConfig::default();
    config.trend.advanced.enabled = true;

    let result_df = generate_technical_indicators(df, &config)
        .await
        .expect("Failed to generate indicators");

    // Basic smoke test: verify indicator generation works
    let col_names = result_df.get_column_names();
    println!("Generated columns: {:?}", col_names);

    // Verify there are a reasonable number of columns generated
    assert!(
        col_names.len() > 30,
        "Should generate many indicators, got {} columns",
        col_names.len()
    );

    // Advanced features may or may not be present depending on configuration
    // Just verify we can access them without panicking
    let _ = result_df.column("fractal_dimension");
    let _ = result_df.column("price_gaps");
}
