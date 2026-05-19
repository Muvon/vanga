use crate::config::features::RegimeFeaturesConfig;
use crate::features::regime::generate_regime_features;
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

fn cfg() -> RegimeFeaturesConfig {
    RegimeFeaturesConfig {
        enabled: true,
        range_position_windows: vec![5],
        squeeze_periods: vec![5],
        bb_std_dev: 2.0,
        kc_atr_mult: 1.5,
        range_compression_short: Some(5),
        range_compression_long: Some(10),
    }
}

fn get_col(df: &DataFrame, name: &str) -> Vec<f64> {
    let ca = df.column(name).unwrap().f64().unwrap().clone();
    ca.into_no_null_iter().collect()
}

#[tokio::test]
async fn position_in_range_extremes() {
    // 5 bars setting range [100, 110], then a 6th bar closing right at the bottom.
    let rows = vec![
        (105.0, 110.0, 100.0, 108.0, 1000.0),
        (108.0, 110.0, 100.0, 105.0, 1000.0),
        (105.0, 110.0, 100.0, 102.0, 1000.0),
        (102.0, 110.0, 100.0, 106.0, 1000.0),
        (106.0, 110.0, 100.0, 100.0, 1000.0), // close at low of window
        (100.0, 110.0, 100.0, 110.0, 1000.0), // close at high of window
    ];
    let df = make_df(rows);
    let out = generate_regime_features(df, &cfg()).await.unwrap();
    let pos = get_col(&out, "regime_position_in_range_5");

    assert!(
        (pos[4] - 0.0).abs() < 1e-6,
        "close at window low should give 0.0, got {}",
        pos[4]
    );
    assert!(
        (pos[5] - 1.0).abs() < 1e-6,
        "close at window high should give 1.0, got {}",
        pos[5]
    );
    for v in pos {
        assert!((0.0..=1.0).contains(&v));
    }
}

#[tokio::test]
async fn squeeze_fires_when_bb_inside_kc_and_duration_counts() {
    // 10 bars of tight close-prices (low std) but with wider H/L (so ATR-proxy
    // > stdev). BB should be inside KC -> squeeze.
    let mut rows = Vec::new();
    for i in 0..10 {
        let close = 100.0 + (i as f64) * 0.01; // very tight close-to-close
        let open = close;
        let high = close + 1.0; // wide bar range
        let low = close - 1.0;
        rows.push((open, high, low, close, 1000.0));
    }
    let df = make_df(rows);
    let out = generate_regime_features(df, &cfg()).await.unwrap();

    let on = get_col(&out, "regime_squeeze_on_5");
    let dur = get_col(&out, "regime_squeeze_duration_5");

    // After the period warmup, most bars should be in squeeze.
    let post = &on[5..];
    assert!(
        post.iter().filter(|v| **v > 0.5).count() >= post.len() / 2,
        "expected squeeze to dominate on tight-close wide-range bars: {:?}",
        post
    );

    // Duration is monotonic non-decreasing while squeeze stays on.
    for i in 6..on.len() {
        if on[i] > 0.5 && on[i - 1] > 0.5 {
            assert!(
                dur[i] >= dur[i - 1],
                "duration should not drop while squeeze stays on: i={} dur={:?}",
                i,
                dur
            );
        }
        if on[i] < 0.5 {
            assert_eq!(dur[i], 0.0, "duration must reset when squeeze ends");
        }
    }
}

#[tokio::test]
async fn squeeze_does_not_fire_on_volatile_closes() {
    // Wide close-to-close swings (std large) but small bar range → BB outside KC.
    let mut rows = Vec::new();
    for i in 0..10 {
        let base = 100.0 + ((i % 2) as f64) * 10.0; // closes alternate 100 / 110
        rows.push((base, base + 0.5, base - 0.5, base, 1000.0));
    }
    let df = make_df(rows);
    let out = generate_regime_features(df, &cfg()).await.unwrap();
    let on = get_col(&out, "regime_squeeze_on_5");

    // Post-warmup the squeeze should be off most of the time.
    let post = &on[5..];
    let on_count = post.iter().filter(|v| **v > 0.5).count();
    assert!(
        on_count <= post.len() / 4,
        "squeeze should rarely fire on volatile closes: {:?}",
        post
    );
}

#[tokio::test]
async fn range_compression_drops_when_recent_window_is_tighter() {
    // 10 bars of wide range, then 5 bars of tight range. Compression at the
    // end should be << 1.0.
    let mut rows = Vec::new();
    for _ in 0..10 {
        rows.push((100.0, 120.0, 80.0, 100.0, 1000.0)); // wide
    }
    for _ in 0..5 {
        rows.push((100.0, 101.0, 99.0, 100.0, 1000.0)); // tight
    }
    let df = make_df(rows);
    let out = generate_regime_features(df, &cfg()).await.unwrap();
    let comp = get_col(&out, "regime_range_compression_5_10");

    // The last bar's compression should be much lower than 1.0.
    assert!(
        comp[14] < 0.5,
        "expected strong compression at end (< 0.5), got {}",
        comp[14]
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
    let mut c = cfg();
    c.enabled = false;
    let out = generate_regime_features(df, &c).await.unwrap();
    assert_eq!(out.get_column_names_owned(), before_cols);
}
