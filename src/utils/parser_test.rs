use crate::utils::parser::*;

#[test]
fn test_detect_timeframe_minutes_1h() {
    // Create DataFrame with 1-hour intervals (3600 seconds)
    let timestamps = vec![
        1609459200i64, // 2021-01-01 00:00:00
        1609462800i64, // 2021-01-01 01:00:00
        1609466400i64, // 2021-01-01 02:00:00
        1609470000i64, // 2021-01-01 03:00:00
    ];
    let df = DataFrame::new(
        vec![Series::new("timestamp".into(), timestamps)]
            .into_iter()
            .map(|s| s.into_column())
            .collect(),
    )
    .unwrap();

    let result = detect_timeframe_minutes(&df).unwrap();
    assert_eq!(result, 60); // 1 hour = 60 minutes
}

#[test]
fn test_detect_timeframe_minutes_5m() {
    // Create DataFrame with 5-minute intervals (300 seconds)
    let timestamps = vec![
        1609459200i64, // 2021-01-01 00:00:00
        1609459500i64, // 2021-01-01 00:05:00
        1609459800i64, // 2021-01-01 00:10:00
        1609460100i64, // 2021-01-01 00:15:00
    ];
    let df = DataFrame::new(
        vec![Series::new("timestamp".into(), timestamps)]
            .into_iter()
            .map(|s| s.into_column())
            .collect(),
    )
    .unwrap();

    let result = detect_timeframe_minutes(&df).unwrap();
    assert_eq!(result, 5); // 5 minutes
}

#[test]
fn test_detect_timeframe_minutes_15m() {
    // Create DataFrame with 15-minute intervals (900 seconds)
    let timestamps = vec![
        1609459200i64, // 2021-01-01 00:00:00
        1609460100i64, // 2021-01-01 00:15:00
        1609461000i64, // 2021-01-01 00:30:00
        1609461900i64, // 2021-01-01 00:45:00
    ];
    let df = DataFrame::new(
        vec![Series::new("timestamp".into(), timestamps)]
            .into_iter()
            .map(|s| s.into_column())
            .collect(),
    )
    .unwrap();

    let result = detect_timeframe_minutes(&df).unwrap();
    assert_eq!(result, 15); // 15 minutes
}

#[test]
fn test_detect_timeframe_minutes_6m_microseconds() {
    // Create DataFrame with 6-minute intervals in microseconds
    // Simulates Polars Datetime(Microseconds) format
    let timestamps = vec![
        1609459200000000i64, // 2021-01-01 00:00:00 in microseconds
        1609459560000000i64, // 2021-01-01 00:06:00 in microseconds (360 seconds later)
        1609459920000000i64, // 2021-01-01 00:12:00 in microseconds
        1609460280000000i64, // 2021-01-01 00:18:00 in microseconds
    ];
    let df = DataFrame::new(
        vec![Series::new("timestamp".into(), timestamps)]
            .into_iter()
            .map(|s| s.into_column())
            .collect(),
    )
    .unwrap();

    let result = detect_timeframe_minutes(&df).unwrap();
    assert_eq!(result, 6); // 6 minutes (360000000 µs diff / 60000000 = 6)
}

#[test]
fn test_parse_horizon_to_steps_hours() {
    // 1-hour timeframe
    assert_eq!(parse_horizon_to_steps("1h", 60).unwrap(), 1);
    assert_eq!(parse_horizon_to_steps("4h", 60).unwrap(), 4);
    assert_eq!(parse_horizon_to_steps("8h", 60).unwrap(), 8);
    assert_eq!(parse_horizon_to_steps("16h", 60).unwrap(), 16);

    // 5-minute timeframe
    assert_eq!(parse_horizon_to_steps("1h", 5).unwrap(), 12); // 60/5 = 12 steps
    assert_eq!(parse_horizon_to_steps("4h", 5).unwrap(), 48); // 240/5 = 48 steps
}

#[test]
fn test_parse_horizon_to_steps_days() {
    // 1-hour timeframe
    assert_eq!(parse_horizon_to_steps("1d", 60).unwrap(), 24); // 24 hours
    assert_eq!(parse_horizon_to_steps("2d", 60).unwrap(), 48); // 48 hours

    // 5-minute timeframe
    assert_eq!(parse_horizon_to_steps("1d", 5).unwrap(), 288); // 1440/5 = 288 steps
}

#[test]
fn test_parse_horizon_to_steps_minutes() {
    // 1-minute timeframe
    assert_eq!(parse_horizon_to_steps("5m", 1).unwrap(), 5);
    assert_eq!(parse_horizon_to_steps("15m", 1).unwrap(), 15);
    assert_eq!(parse_horizon_to_steps("30m", 1).unwrap(), 30);

    // 5-minute timeframe
    assert_eq!(parse_horizon_to_steps("15m", 5).unwrap(), 3); // 15/5 = 3 steps
    assert_eq!(parse_horizon_to_steps("30m", 5).unwrap(), 6); // 30/5 = 6 steps
}

#[test]
fn test_parse_horizon_to_steps_invalid() {
    assert!(parse_horizon_to_steps("invalid", 60).is_err());
    assert!(parse_horizon_to_steps("16", 60).is_err());
    assert!(parse_horizon_to_steps("h16", 60).is_err());
    assert!(parse_horizon_to_steps("", 60).is_err());
}

#[test]
fn test_parse_horizon_to_steps_too_small() {
    // Horizon smaller than timeframe should error
    assert!(parse_horizon_to_steps("5m", 60).is_err()); // 5 minutes < 60 minute timeframe
    assert!(parse_horizon_to_steps("30m", 60).is_err()); // 30 minutes < 60 minute timeframe
}

#[test]
fn nonintegral_horizon_reports_both_minute_counts() {
    let error = parse_horizon_to_steps("90m", 60).unwrap_err().to_string();
    assert!(error.contains("90") && error.contains("60"), "{error}");
    assert!(parse_horizon_to_steps("1h", 0).is_err());
}

#[test]
fn ingestion_rejects_each_nonfinite_ohlcv_column() {
    for name in ["open", "high", "low", "close", "volume"] {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut frame = df!(
                "timestamp" => [1704067200_i64, 1704070800],
                "open" => [10.0, 10.0], "high" => [11.0, 11.0],
                "low" => [9.0, 9.0], "close" => [10.0, 10.0], "volume" => [100.0, 100.0]
            )
            .unwrap();
            frame
                .with_column(Series::new(name.into(), [10.0, value]))
                .unwrap();
            let error = crate::utils::market_data::extract_ohlcv_data(&frame)
                .unwrap_err()
                .to_string();
            assert!(error.contains(name) && error.contains("row 1"), "{error}");
        }
    }
}
