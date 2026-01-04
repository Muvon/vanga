// Comprehensive mathematical validation tests for technical indicators
use crate::config::features::TechnicalIndicatorsConfig;
use crate::features::technical::generate_technical_indicators;
use approx::assert_relative_eq;
use polars::prelude::*;

/// Helper to create test DataFrame
fn create_test_df(
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
) -> DataFrame {
    let timestamp_len = close.len();
    DataFrame::new(vec![
        Series::new("open", open),
        Series::new("high", high),
        Series::new("low", low),
        Series::new("close", close),
        Series::new("volume", volume),
        Series::new(
            "timestamp",
            (0..timestamp_len)
                .map(|i| i as i64 * 3600)
                .collect::<Vec<_>>(),
        ),
    ])
    .unwrap()
}

#[tokio::test]
async fn test_sma_calculation_through_api() {
    let close = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let df = create_test_df(
        close.clone(),
        close.iter().map(|x| x + 5.0).collect(),
        close.iter().map(|x| x - 5.0).collect(),
        close.clone(),
        vec![1000.0; 5],
    );

    let mut config = TechnicalIndicatorsConfig::default();
    config.moving_averages.sma_periods = vec![3];

    let result = generate_technical_indicators(df, &config).await.unwrap();

    // Check SMA_3 column exists
    assert!(result.column("sma_3").is_ok());

    let sma_series = result.column("sma_3").unwrap().f64().unwrap();

    // First 2 values should be NaN (warmup period)
    assert!(sma_series.get(0).unwrap().is_nan());
    assert!(sma_series.get(1).unwrap().is_nan());

    // SMA(3) at index 2: (10+20+30)/3 = 20.0
    assert_relative_eq!(sma_series.get(2).unwrap(), 20.0, epsilon = 1e-10);

    // SMA(3) at index 3: (20+30+40)/3 = 30.0
    assert_relative_eq!(sma_series.get(3).unwrap(), 30.0, epsilon = 1e-10);

    // SMA(3) at index 4: (30+40+50)/3 = 40.0
    assert_relative_eq!(sma_series.get(4).unwrap(), 40.0, epsilon = 1e-10);
}

#[tokio::test]
async fn test_obv_calculation_through_api() {
    let close = vec![100.0, 105.0, 103.0, 108.0, 108.0];
    let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 1000.0];

    let df = create_test_df(
        close.clone(),
        close.iter().map(|x| x + 5.0).collect(),
        close.iter().map(|x| x - 5.0).collect(),
        close.clone(),
        volume.clone(),
    );

    let mut config = TechnicalIndicatorsConfig::default();
    config.volume.obv = true;

    let result = generate_technical_indicators(df, &config).await.unwrap();

    assert!(result.column("obv").is_ok());
    let obv_series = result.column("obv").unwrap().f64().unwrap();

    // OBV[0] = volume[0]
    assert_relative_eq!(obv_series.get(0).unwrap(), 1000.0, epsilon = 1e-10);

    // OBV[1] = OBV[0] + volume[1] (price up)
    assert_relative_eq!(obv_series.get(1).unwrap(), 2500.0, epsilon = 1e-10);

    // OBV[2] = OBV[1] - volume[2] (price down)
    assert_relative_eq!(obv_series.get(2).unwrap(), 1300.0, epsilon = 1e-10);

    // OBV[3] = OBV[2] + volume[3] (price up)
    assert_relative_eq!(obv_series.get(3).unwrap(), 3100.0, epsilon = 1e-10);

    // OBV[4] = OBV[3] (price unchanged)
    assert_relative_eq!(obv_series.get(4).unwrap(), 3100.0, epsilon = 1e-10);
}

#[tokio::test]
async fn test_mfi_calculation_through_api() {
    let high = vec![105.0, 110.0, 108.0, 112.0, 115.0, 113.0];
    let low = vec![95.0, 100.0, 98.0, 102.0, 105.0, 103.0];
    let close = vec![100.0, 105.0, 103.0, 108.0, 110.0, 108.0];
    let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 2000.0, 1600.0];

    let df = create_test_df(close.clone(), high, low, close, volume);

    let mut config = TechnicalIndicatorsConfig::default();
    config.volume.mfi_periods = vec![3];

    let result = generate_technical_indicators(df, &config).await.unwrap();

    assert!(result.column("mfi_3").is_ok());
    let mfi_series = result.column("mfi_3").unwrap().f64().unwrap();

    // First 3 values should be NaN (warmup period)
    assert!(mfi_series.get(0).unwrap().is_nan());
    assert!(mfi_series.get(1).unwrap().is_nan());
    assert!(mfi_series.get(2).unwrap().is_nan());

    // MFI should be between 0 and 100
    for i in 3..mfi_series.len() {
        let val = mfi_series.get(i).unwrap();
        assert!(
            (0.0..=100.0).contains(&val),
            "MFI at index {} = {} is out of range [0, 100]",
            i,
            val
        );
    }
}

#[tokio::test]
async fn test_advanced_indicators_range() {
    // Create sufficient data for advanced indicators
    let mut close: Vec<f64> = Vec::new();
    for i in 0..200 {
        close.push(100.0 + (i as f64 * 0.1).sin() * 10.0 + i as f64 * 0.05);
    }

    let df = create_test_df(
        close.clone(),
        close.iter().map(|x| x + 5.0).collect(),
        close.iter().map(|x| x - 5.0).collect(),
        close.clone(),
        vec![1000.0; 200],
    );

    let mut config = TechnicalIndicatorsConfig::default();
    config.trend.advanced.enabled = true;
    config.trend.advanced.hurst_window = 50;
    config.trend.advanced.fractal_window = 50;
    config.trend.advanced.regime_window = 30;
    config.trend.advanced.clustering_window = 30;
    config.trend.advanced.reversion_window = 30;

    let result = generate_technical_indicators(df, &config).await.unwrap();

    // Test Hurst Exponent
    if let Ok(hurst_col) = result.column("hurst_exponent") {
        let hurst_series = hurst_col.f64().unwrap();
        for i in 0..hurst_series.len() {
            if let Some(val) = hurst_series.get(i) {
                if !val.is_nan() {
                    assert!(
                        (0.1..=0.9).contains(&val),
                        "Hurst exponent at index {} = {} is out of range [0.1, 0.9]",
                        i,
                        val
                    );
                }
            }
        }
    }

    // Test Fractal Dimension
    if let Ok(fractal_col) = result.column("fractal_dimension") {
        let fractal_series = fractal_col.f64().unwrap();
        for i in 0..fractal_series.len() {
            if let Some(val) = fractal_series.get(i) {
                if !val.is_nan() {
                    assert!(
                        (1.0..=2.0).contains(&val),
                        "Fractal dimension at index {} = {} is out of range [1.0, 2.0]",
                        i,
                        val
                    );
                }
            }
        }
    }

    // Test Regime Indicator
    if let Ok(regime_col) = result.column("regime_indicator") {
        let regime_series = regime_col.f64().unwrap();
        for i in 0..regime_series.len() {
            if let Some(val) = regime_series.get(i) {
                if !val.is_nan() {
                    assert!(
                        (0.0..=3.0).contains(&val),
                        "Regime indicator at index {} = {} is out of range [0.0, 3.0]",
                        i,
                        val
                    );
                }
            }
        }
    }

    // Test Volatility Clustering
    if let Ok(clustering_col) = result.column("volatility_clustering") {
        let clustering_series = clustering_col.f64().unwrap();
        for i in 0..clustering_series.len() {
            if let Some(val) = clustering_series.get(i) {
                if !val.is_nan() {
                    assert!(
                        (-1.0..=1.0).contains(&val),
                        "Volatility clustering at index {} = {} is out of range [-1.0, 1.0]",
                        i,
                        val
                    );
                }
            }
        }
    }

    // Test Mean Reversion Strength
    if let Ok(reversion_col) = result.column("mean_reversion_strength") {
        let reversion_series = reversion_col.f64().unwrap();
        for i in 0..reversion_series.len() {
            if let Some(val) = reversion_series.get(i) {
                if !val.is_nan() {
                    assert!(
                        (0.0..=1.0).contains(&val),
                        "Mean reversion strength at index {} = {} is out of range [0.0, 1.0]",
                        i,
                        val
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn test_price_gaps_basis_points() {
    // Create data with known gaps
    let open = vec![100.0, 100.5, 99.8, 100.2, 100.0];
    let close = vec![100.3, 100.1, 100.0, 100.5, 100.3];
    let high = vec![101.0, 101.0, 100.5, 101.0, 101.0];
    let low = vec![99.0, 99.5, 99.0, 99.5, 99.0];
    let volume = vec![1000.0; 5];

    let df = create_test_df(open.clone(), high, low, close.clone(), volume);

    let config = TechnicalIndicatorsConfig::default();
    let result = generate_technical_indicators(df, &config).await.unwrap();

    // Check price_gaps column exists
    assert!(result.column("price_gaps").is_ok());

    let gaps_series = result.column("price_gaps").unwrap().f64().unwrap();

    // Gap[1] = (open[1] - close[0]) / close[0] * 10000 = (100.5 - 100.3) / 100.3 * 10000 ≈ 19.94 bps
    let gap_1 = gaps_series.get(1).unwrap();
    assert_relative_eq!(gap_1, 19.94, epsilon = 0.1);

    // Gap[2] = (open[2] - close[1]) / close[1] * 10000 = (99.8 - 100.1) / 100.1 * 10000 ≈ -29.97 bps
    let gap_2 = gaps_series.get(2).unwrap();
    assert_relative_eq!(gap_2, -29.97, epsilon = 0.1);

    // Gap[3] = (open[3] - close[2]) / close[2] * 10000 = (100.2 - 100.0) / 100.0 * 10000 = 20.0 bps
    let gap_3 = gaps_series.get(3).unwrap();
    assert_relative_eq!(gap_3, 20.0, epsilon = 0.1);
}

#[tokio::test]
async fn test_vwap_calculation() {
    let close = vec![100.0, 110.0, 105.0, 115.0];
    let volume = vec![1000.0, 2000.0, 1500.0, 2500.0];

    let df = create_test_df(
        close.clone(),
        close.iter().map(|x| x + 5.0).collect(),
        close.iter().map(|x| x - 5.0).collect(),
        close.clone(),
        volume,
    );

    let config = TechnicalIndicatorsConfig::default();
    let result = generate_technical_indicators(df, &config).await.unwrap();

    // Check VWAP column exists
    assert!(result.column("vwap").is_ok());

    let vwap_series = result.column("vwap").unwrap().f64().unwrap();

    // VWAP is cumulative, so it should be calculated from start
    // VWAP[0] = (100*1000) / 1000 = 100.0
    assert_relative_eq!(vwap_series.get(0).unwrap(), 100.0, epsilon = 1e-5);

    // VWAP[1] = (100*1000 + 110*2000) / (1000+2000) = 106.67
    assert_relative_eq!(vwap_series.get(1).unwrap(), 106.666666, epsilon = 1e-5);
}

#[tokio::test]
async fn test_all_indicators_no_panic() {
    // Comprehensive test with all indicators enabled
    let mut close: Vec<f64> = Vec::new();
    for i in 0..100 {
        close.push(100.0 + (i as f64 * 0.2).sin() * 10.0);
    }

    let df = create_test_df(
        close.clone(),
        close.iter().map(|x| x + 5.0).collect(),
        close.iter().map(|x| x - 5.0).collect(),
        close.clone(),
        vec![1000.0; 100],
    );

    let mut config = TechnicalIndicatorsConfig::default();
    config.moving_averages.sma_periods = vec![5, 10, 20];
    config.moving_averages.ema_periods = vec![5, 10, 20];
    config.trend.macd.enabled = true;
    config.volatility.bollinger_bands.enabled = true;
    config.momentum.rsi_periods = vec![14];
    config.momentum.stochastic = true;
    config.momentum.williams_r = true;
    config.momentum.cci_periods = vec![20];
    config.volume.volume_sma_periods = vec![10];
    config.volume.obv = true;
    config.volume.mfi_periods = vec![14];
    config.volatility.atr_periods = vec![14];
    config.volatility.keltner_channels = true;
    config.trend.advanced.enabled = true;

    // This should not panic
    let result = generate_technical_indicators(df, &config).await;
    assert!(
        result.is_ok(),
        "Indicator generation failed: {:?}",
        result.err()
    );

    let result_df = result.unwrap();

    // Verify we have more columns than input
    assert!(
        result_df.width() > 6,
        "Expected more columns after indicator generation"
    );
}

#[tokio::test]
async fn test_indicators_with_real_data_patterns() {
    // Test with realistic crypto price patterns
    let mut close: Vec<f64> = Vec::new();
    let mut volume: Vec<f64> = Vec::new();

    // Simulate a realistic crypto pattern: uptrend with volatility
    for i in 0..150 {
        let trend = i as f64 * 0.5;
        let volatility = (i as f64 * 0.3).sin() * 15.0;
        let noise = (i as f64 * 1.7).cos() * 5.0;
        close.push(10000.0 + trend + volatility + noise);
        volume.push(1000.0 + (i as f64 * 0.1).sin().abs() * 500.0);
    }

    let df = create_test_df(
        close.clone(),
        close.iter().map(|x| x + 50.0).collect(),
        close.iter().map(|x| x - 50.0).collect(),
        close,
        volume,
    );

    let mut config = TechnicalIndicatorsConfig::default();
    config.moving_averages.sma_periods = vec![20, 50];
    config.moving_averages.ema_periods = vec![12, 26];
    config.trend.macd.enabled = true;
    config.momentum.rsi_periods = vec![14];
    config.volume.obv = true;
    config.trend.advanced.enabled = true;

    let result = generate_technical_indicators(df, &config).await.unwrap();

    // Verify all expected columns exist
    assert!(result.column("sma_20").is_ok());
    assert!(result.column("sma_50").is_ok());
    assert!(result.column("ema_12").is_ok());
    assert!(result.column("ema_26").is_ok());
    assert!(result.column("macd").is_ok());
    assert!(result.column("rsi_14").is_ok());
    assert!(result.column("obv").is_ok());
    assert!(result.column("hurst_exponent").is_ok());
    assert!(result.column("fractal_dimension").is_ok());

    // Verify RSI is in valid range
    let rsi_series = result.column("rsi_14").unwrap().f64().unwrap();
    for i in 0..rsi_series.len() {
        if let Some(val) = rsi_series.get(i) {
            if !val.is_nan() {
                assert!(
                    (0.0..=100.0).contains(&val),
                    "RSI at index {} = {} is out of range [0, 100]",
                    i,
                    val
                );
            }
        }
    }
}
