use crate::config::features::LiquidityFeaturesConfig;
use crate::features::liquidity::generate_liquidity_features;
use polars::prelude::*;

fn make_df(rows: Vec<(f64, f64, f64, f64, f64)>) -> DataFrame {
    let opens: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let highs: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let lows: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let closes: Vec<f64> = rows.iter().map(|r| r.3).collect();
    let volumes: Vec<f64> = rows.iter().map(|r| r.4).collect();

    DataFrame::new(vec![
        Series::new("open".into(), opens).into_column(),
        Series::new("high".into(), highs).into_column(),
        Series::new("low".into(), lows).into_column(),
        Series::new("close".into(), closes).into_column(),
        Series::new("volume".into(), volumes).into_column(),
    ])
    .expect("failed to build test DataFrame")
}

fn small_config() -> LiquidityFeaturesConfig {
    LiquidityFeaturesConfig {
        enabled: true,
        sweep_lookbacks: vec![5],
        atr_period: 3,
        cvd_windows: vec![5],
    }
}

fn get_col(df: &DataFrame, name: &str) -> Vec<f64> {
    // Mirror the .column().f64() pattern used elsewhere in the feature tests
    // (e.g. technical_math_test.rs). f64() on a Column gives a ChunkedArray
    // we can iterate via into_no_null_iter once we know there are no nulls
    // (our generators never produce nulls — they emit 0.0 or sentinel f64s).
    let ca = df.column(name).unwrap().f64().unwrap().clone();
    ca.into_no_null_iter().collect()
}

#[tokio::test]
async fn wick_ratios_are_bounded_and_sum_to_one() {
    // Mix of bullish, bearish, doji, marubozu candles.
    let rows = vec![
        (100.0, 110.0, 90.0, 105.0, 1000.0), // bullish, balanced wicks
        (105.0, 106.0, 95.0, 96.0, 1100.0),  // bearish, big upper wick
        (96.0, 100.0, 92.0, 99.0, 900.0),    // bullish recovery, lower wick
        (99.0, 99.0, 99.0, 99.0, 800.0),     // doji-like (zero range)
        (99.0, 105.0, 99.0, 105.0, 1200.0),  // bull marubozu
        (105.0, 105.0, 99.0, 99.0, 1300.0),  // bear marubozu
    ];
    let df = make_df(rows);
    let out = generate_liquidity_features(df, &small_config())
        .await
        .unwrap();

    let upper = get_col(&out, "liq_upper_wick_ratio");
    let lower = get_col(&out, "liq_lower_wick_ratio");
    let body = get_col(&out, "liq_body_ratio");
    let asym = get_col(&out, "liq_wick_asymmetry");

    for i in 0..upper.len() {
        assert!(
            (0.0..=1.0).contains(&upper[i]),
            "upper[{}] out of [0,1]: {}",
            i,
            upper[i]
        );
        assert!((0.0..=1.0).contains(&lower[i]));
        assert!((0.0..=1.0).contains(&body[i]));
        assert!((-1.0..=1.0).contains(&asym[i]));
        // Components must add to ~1 on candles with positive range (the doji at i=3
        // is the only zero-range candle and the epsilon path produces (0,0,0)).
        if i != 3 {
            let total = upper[i] + lower[i] + body[i];
            assert!(
                (total - 1.0).abs() < 1e-6,
                "wick + body should sum to 1 at i={}: got {}",
                i,
                total
            );
        }
    }

    // Bear marubozu: no wicks, all body, negative skew? body=1, others 0.
    assert!(body[5] > 0.99, "bear marubozu body should ~= 1");
    assert!(upper[5] < 0.01 && lower[5] < 0.01);
}

#[tokio::test]
async fn liquidity_sweep_up_detected_when_high_breaks_then_close_fails() {
    // Five-bar uptrending high plateau at 110, then a sweep candle prints 115
    // but closes back at 108 (below the prior high of 110). Up-sweep expected.
    let rows = vec![
        (100.0, 110.0, 95.0, 108.0, 1000.0),
        (108.0, 110.0, 100.0, 109.0, 1000.0),
        (109.0, 110.0, 105.0, 108.0, 1000.0),
        (108.0, 110.0, 106.0, 109.0, 1000.0),
        (109.0, 110.0, 105.0, 109.0, 1000.0),
        (109.0, 115.0, 108.0, 108.0, 1500.0), // SWEEP: high > 110, close < 110
        (108.0, 110.0, 104.0, 105.0, 1000.0),
    ];
    let df = make_df(rows);
    let out = generate_liquidity_features(df, &small_config())
        .await
        .unwrap();

    let sweep_up = get_col(&out, "liq_sweep_up_5");
    let sweep_down = get_col(&out, "liq_sweep_down_5");
    let strength = get_col(&out, "liq_sweep_strength_5");

    // Pre-window bars must not fire (no prior 5-bar window yet).
    for v in &sweep_up[..5] {
        assert_eq!(*v, 0.0);
    }
    for v in &sweep_down[..5] {
        assert_eq!(*v, 0.0);
    }

    assert_eq!(sweep_up[5], 1.0, "expected up-sweep at bar 5");
    assert_eq!(sweep_down[5], 0.0, "down-sweep must not fire concurrently");
    assert!(
        strength[5] > 0.0,
        "up-sweep strength should be positive, got {}",
        strength[5]
    );
}

#[tokio::test]
async fn liquidity_sweep_down_detected_and_signed_negative() {
    // Plateau at 100 low, then bar prints 90 low and closes at 102 (above prior low).
    let rows = vec![
        (110.0, 115.0, 100.0, 108.0, 1000.0),
        (108.0, 112.0, 100.0, 105.0, 1000.0),
        (105.0, 108.0, 100.0, 104.0, 1000.0),
        (104.0, 107.0, 100.0, 103.0, 1000.0),
        (103.0, 106.0, 100.0, 102.0, 1000.0),
        (102.0, 105.0, 90.0, 102.0, 1500.0), // SWEEP: low < 100, close > 100
    ];
    let df = make_df(rows);
    let out = generate_liquidity_features(df, &small_config())
        .await
        .unwrap();

    let sweep_down = get_col(&out, "liq_sweep_down_5");
    let strength = get_col(&out, "liq_sweep_strength_5");

    assert_eq!(sweep_down[5], 1.0);
    assert!(
        strength[5] < 0.0,
        "down-sweep strength should be signed negative, got {}",
        strength[5]
    );
}

#[tokio::test]
async fn cvd_slope_positive_when_closes_trend_to_highs() {
    // Steady rally where every candle closes near its high → CVD should climb,
    // so the rolling slope is positive once warm-up is done.
    let mut rows = Vec::new();
    let mut price = 100.0;
    for _ in 0..30 {
        let open = price;
        let close = price + 1.0;
        let high = close + 0.05; // close very near high
        let low = open - 0.05;
        rows.push((open, high, low, close, 1000.0));
        price = close;
    }
    let df = make_df(rows);
    let out = generate_liquidity_features(df, &small_config())
        .await
        .unwrap();

    let cvd_slope = get_col(&out, "liq_cvd_slope_5");

    // After warm-up the slope should be positive on most bars.
    let post_warmup = &cvd_slope[10..];
    let positive_count = post_warmup.iter().filter(|v| **v > 0.0).count();
    assert!(
        positive_count >= post_warmup.len() * 8 / 10,
        "expected mostly-positive CVD slope on uptrend, got only {}/{} positive",
        positive_count,
        post_warmup.len()
    );
}

#[tokio::test]
async fn cvd_price_divergence_flags_price_up_flow_down() {
    // Construct a sequence where price drifts up while CVD drifts down (closes
    // near LOW even though high crosses higher → most volume marked as sell).
    let mut rows = Vec::new();
    for i in 0..20 {
        let price = 100.0 + (i as f64) * 0.5; // price climbing
        let high = price + 1.0;
        let low = price - 1.0;
        let open = price;
        let close = low + 0.05; // close pinned to low → strong sell estimate
        rows.push((open, high, low, close, 1000.0));
    }
    let df = make_df(rows);
    let out = generate_liquidity_features(df, &small_config())
        .await
        .unwrap();

    let div = get_col(&out, "liq_cvd_price_div_5");

    // The last bar should show a clearly positive divergence (price rose while CVD fell).
    assert!(
        div[19] > 0.5,
        "expected strong positive price/CVD divergence on last bar, got {}",
        div[19]
    );
}

#[tokio::test]
async fn disabled_config_returns_input_unchanged() {
    let rows = vec![
        (100.0, 110.0, 90.0, 105.0, 1000.0),
        (105.0, 115.0, 95.0, 110.0, 1100.0),
    ];
    let df = make_df(rows);
    let before_cols = df.get_column_names_owned();
    let cfg = LiquidityFeaturesConfig {
        enabled: false,
        sweep_lookbacks: vec![5],
        atr_period: 3,
        cvd_windows: vec![5],
    };
    let out = generate_liquidity_features(df, &cfg).await.unwrap();
    assert_eq!(out.get_column_names_owned(), before_cols);
}
