use crate::config::features::TechnicalIndicatorsConfig;
use crate::features::technical::generate_technical_indicators;
use polars::prelude::*;

#[tokio::test]
async fn test_generate_technical_indicators_basic() {
    let df = DataFrame::new(
        vec![
            Series::new("open".into(), &[42000.0, 42100.0, 42200.0]),
            Series::new("high".into(), &[42500.0, 42600.0, 42700.0]),
            Series::new("low".into(), &[41800.0, 41900.0, 42000.0]),
            Series::new("close".into(), &[42300.0, 42400.0, 42500.0]),
            Series::new("volume".into(), &[1000.0, 1200.0, 1300.0]),
        ]
        .into_iter()
        .map(|s| s.into_column())
        .collect(),
    )
    .unwrap();
    let config = TechnicalIndicatorsConfig::default();
    let df2 = generate_technical_indicators(df.clone(), &config)
        .await
        .unwrap();
    assert!(df2.get_column_names().len() > df.get_column_names().len());
    assert!(
        df2.column("sma_5").is_ok() || df2.get_column_names().iter().any(|n| n.contains("sma"))
    );
}

fn causal_test_frame(rows: usize) -> DataFrame {
    let close: Vec<f64> = (0..rows)
        .map(|i| 100.0 + (i as f64 / 7.0).sin() * 5.0 + i as f64 * 0.03)
        .collect();
    df!(
        "timestamp" => (0..rows).map(|i| 1704067200_i64 + i as i64 * 3600).collect::<Vec<_>>(),
        "open" => close.clone(),
        "high" => close.iter().map(|v| v + 1.0).collect::<Vec<_>>(),
        "low" => close.iter().map(|v| v - 1.0).collect::<Vec<_>>(),
        "close" => close,
        "volume" => (0..rows).map(|i| 1000.0 + (i as f64).sin() * 100.0).collect::<Vec<_>>()
    )
    .unwrap()
}

#[tokio::test]
async fn indicator_groups_coexist_and_regime_is_causal() {
    let config = TechnicalIndicatorsConfig::default();
    let mut frame = causal_test_frame(300);
    for name in ["open", "high", "low", "close"] {
        let values: Vec<f64> = frame
            .column(name)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .enumerate()
            .map(|(index, value)| {
                if index < 200 {
                    value
                } else {
                    value * if index % 2 == 0 { 1.2 } else { 0.8 }
                }
            })
            .collect();
        frame.with_column(Series::new(name.into(), values)).unwrap();
    }
    let prefix = crate::features::technical::generate_technical_indicators_with_window(
        frame.head(Some(200)),
        &config,
        17,
    )
    .await
    .unwrap();
    let full =
        crate::features::technical::generate_technical_indicators_with_window(frame, &config, 17)
            .await
            .unwrap();
    for name in ["sma_5", "ema_5", "rsi_14"] {
        assert!(full.column(name).is_ok(), "missing {name}");
    }
    for name in full.get_column_names() {
        if name.starts_with("dema_") || name.starts_with("tema_") || name.starts_with("zlema_") {
            assert!(
                full.column(name)
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(299)
                    .unwrap()
                    .is_finite(),
                "{name} never leaves warmup"
            );
        }
    }
    assert!(full.column("regime_indicator").is_ok());
    for name in full.get_column_names() {
        if name.contains("regime") {
            let a = prefix.column(name).unwrap().f64().unwrap();
            let b = full.column(name).unwrap().f64().unwrap();
            for (left, right) in a.into_iter().zip(b.into_iter()) {
                assert!(
                    left == right || left.unwrap().is_nan() && right.unwrap().is_nan(),
                    "noncausal {name}: {left:?} != {right:?}"
                );
            }
        }
    }
    println!(
        "B1: SMA + EMA + RSI coexist; {} input features",
        full.width() - 1
    );
}
