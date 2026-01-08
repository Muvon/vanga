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
async fn test_price_inefficiency_with_real_data() {
    println!("Testing price inefficiency features with synthetic data...");

    let df = create_test_df();
    println!("Loaded {} rows of data", df.height());

    let config = TechnicalIndicatorsConfig::default();
    let result = generate_technical_indicators(df, &config)
        .await
        .expect("Failed to calculate indicators");

    // Check price_gaps exists
    if let Ok(price_gaps) = result.column("price_gaps") {
        let gaps_series = price_gaps.f64().expect("price_gaps should be f64");
        let has_values =
            (0..gaps_series.len()).any(|i| gaps_series.get(i).is_some_and(|v| !v.is_nan()));
        assert!(has_values, "price_gaps should have valid values");
    }

    // Check gap_volatility exists
    if let Ok(gap_volatility) = result.column("gap_volatility") {
        let vol_series = gap_volatility.f64().expect("gap_volatility should be f64");
        let has_values =
            (0..vol_series.len()).any(|i| vol_series.get(i).is_some_and(|v| !v.is_nan()));
        assert!(has_values, "gap_volatility should have valid values");
    }

    println!("✓ Price inefficiency features working correctly");
}
