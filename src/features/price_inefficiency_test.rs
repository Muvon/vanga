use crate::config::features::TechnicalIndicatorsConfig;
use crate::features::technical::generate_technical_indicators;
use polars::prelude::*;

#[tokio::test]
async fn test_price_inefficiency_with_real_data() {
    println!("Testing price inefficiency features with real BTCUSDT data...");

    // Load real data
    let df = CsvReader::from_path("data/BTCUSDT.csv")
        .expect("Failed to load BTCUSDT.csv")
        .has_header(true)
        .finish()
        .expect("Failed to parse CSV");

    println!("Loaded {} rows of data", df.height());

    // Calculate technical indicators
    let config = TechnicalIndicatorsConfig::default();
    let result = generate_technical_indicators(df, &config)
        .await
        .expect("Failed to calculate indicators");

    // Check price_gaps (body-to-range ratio)
    let price_gaps = result
        .column("price_gaps")
        .expect("price_gaps column missing")
        .f64()
        .expect("price_gaps should be f64");

    let non_zero_gaps = price_gaps
        .into_iter()
        .filter(|v| v.is_some() && v.unwrap().abs() > 0.01)
        .count();

    let total_gaps = price_gaps.len();
    let non_zero_pct = (non_zero_gaps as f64 / total_gaps as f64) * 100.0;

    println!(
        "price_gaps: {}/{} non-zero values ({:.2}%)",
        non_zero_gaps, total_gaps, non_zero_pct
    );

    // Check gap_volatility (wick imbalance)
    let gap_volatility = result
        .column("gap_volatility")
        .expect("gap_volatility column missing")
        .f64()
        .expect("gap_volatility should be f64");

    let non_zero_vol = gap_volatility
        .into_iter()
        .filter(|v| v.is_some() && v.unwrap().abs() > 0.01)
        .count();

    let total_vol = gap_volatility.len();
    let non_zero_vol_pct = (non_zero_vol as f64 / total_vol as f64) * 100.0;

    println!(
        "gap_volatility (wick imbalance): {}/{} non-zero values ({:.2}%)",
        non_zero_vol, total_vol, non_zero_vol_pct
    );

    // Get some sample values
    let sample_gaps: Vec<f64> = price_gaps.into_iter().filter_map(|v| v).take(20).collect();

    let sample_vol: Vec<f64> = gap_volatility
        .into_iter()
        .filter_map(|v| v)
        .take(20)
        .collect();

    println!("Sample price_gaps values: {:?}", sample_gaps);
    println!("Sample wick imbalance values: {:?}", sample_vol);

    // Assertions
    assert!(
        non_zero_pct > 90.0,
        "price_gaps should have >90% non-zero values, got {:.2}%",
        non_zero_pct
    );

    assert!(
        non_zero_vol_pct > 90.0,
        "wick imbalance should have >90% non-zero values, got {:.2}%",
        non_zero_vol_pct
    );

    println!("✓ Price inefficiency features working correctly with real data");
}
