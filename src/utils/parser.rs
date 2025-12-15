use crate::utils::error::Result;
use polars::prelude::*;

/// Detect timeframe (candle interval) from DataFrame timestamps
pub fn detect_timeframe_minutes(df: &DataFrame) -> Result<usize> {
    let timestamp_col = df.column("timestamp").map_err(|_| {
        crate::utils::error::VangaError::DataError(
            "No 'timestamp' column found in DataFrame".to_string(),
        )
    })?;

    log::debug!("Timestamp column type: {:?}", timestamp_col.dtype());

    // Handle different timestamp types
    let timestamps_i64 = match timestamp_col.dtype() {
        DataType::Int64 => {
            log::debug!("Using existing i64 timestamps");
            timestamp_col
                .i64()
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to get i64 from timestamp column: {}",
                        e
                    ))
                })?
                .clone()
        }
        DataType::Datetime(_, _) => {
            log::debug!("Converting datetime to i64");
            timestamp_col
                .cast(&DataType::Int64)
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to convert datetime to i64: {}",
                        e
                    ))
                })?
                .i64()
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to get i64 from datetime: {}",
                        e
                    ))
                })?
                .clone()
        }
        DataType::Utf8 => {
            log::debug!("Parsing string timestamps to datetime");
            // Try to parse as datetime string
            timestamp_col
                .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to parse timestamp strings: {}",
                        e
                    ))
                })?
                .cast(&DataType::Int64)
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to convert datetime to i64: {}",
                        e
                    ))
                })?
                .i64()
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to get i64 from datetime: {}",
                        e
                    ))
                })?
                .clone()
        }
        _ => {
            return Err(crate::utils::error::VangaError::DataError(format!(
                "Unsupported timestamp type: {:?}. Expected Int64, Datetime, or String",
                timestamp_col.dtype()
            )));
        }
    };

    if timestamps_i64.len() < 2 {
        return Err(crate::utils::error::VangaError::DataError(
            "Need at least 2 rows to detect timeframe".to_string(),
        ));
    }

    let mut differences = Vec::new();
    let mut zero_diffs = 0;
    for i in 1..timestamps_i64.len().min(100) {
        if let (Some(curr), Some(prev)) = (timestamps_i64.get(i), timestamps_i64.get(i - 1)) {
            let diff = (curr - prev).abs();
            if diff > 0 {
                differences.push(diff);
            } else {
                zero_diffs += 1;
            }
        }
    }

    log::debug!(
        "Found {} non-zero differences, {} zero differences",
        differences.len(),
        zero_diffs
    );

    if differences.is_empty() {
        let first_val = timestamps_i64.get(0);
        let last_val = timestamps_i64.get(timestamps_i64.len().min(100) - 1);
        return Err(crate::utils::error::VangaError::DataError(format!(
            "All timestamps are identical (first={:?}, last={:?}). Data may not be sorted.",
            first_val, last_val
        )));
    }

    differences.sort_unstable();
    let median_diff = differences[differences.len() / 2];

    // Detect unit based on magnitude and convert to minutes
    // Polars stores Datetime(Microseconds) as microseconds since epoch
    // Different timestamp formats:
    // - Nanoseconds: 360000000000 ns = 360 seconds = 6 minutes
    // - Microseconds: 360000000 µs = 360 seconds = 6 minutes
    // - Milliseconds: 360000 ms = 360 seconds = 6 minutes
    // - Seconds: 360 s = 6 minutes
    let timeframe_minutes = if median_diff >= 1_000_000_000 {
        // Nanoseconds (>= 1 billion = at least ~16 minutes in nanoseconds)
        (median_diff / 60_000_000_000) as usize
    } else if median_diff >= 1_000_000 {
        // Microseconds (>= 1 million = at least ~16 seconds in microseconds)
        (median_diff / 60_000_000) as usize
    } else if median_diff >= 10_000 {
        // Milliseconds (>= 10,000 = at least 10 seconds in milliseconds)
        (median_diff / 60_000) as usize
    } else {
        // Seconds (< 10,000 = reasonable for seconds)
        (median_diff / 60) as usize
    };

    if timeframe_minutes == 0 {
        return Err(crate::utils::error::VangaError::DataError(format!(
            "Detected timeframe too small: {} raw units",
            median_diff
        )));
    }

    log::info!(
        "🕐 Detected timeframe: {} minutes (median diff: {} units)",
        timeframe_minutes,
        median_diff
    );

    Ok(timeframe_minutes)
}

/// Parse horizon string to number of steps based on detected timeframe
pub fn parse_horizon_to_steps(horizon: &str, timeframe_minutes: usize) -> Result<usize> {
    let horizon_minutes = if let Some(num_str) = horizon.strip_suffix('h') {
        let hours = num_str.parse::<usize>().map_err(|_| {
            crate::utils::error::VangaError::DataError(format!(
                "Invalid horizon format: {}",
                horizon
            ))
        })?;
        hours * 60
    } else if let Some(num_str) = horizon.strip_suffix('d') {
        let days = num_str.parse::<usize>().map_err(|_| {
            crate::utils::error::VangaError::DataError(format!(
                "Invalid horizon format: {}",
                horizon
            ))
        })?;
        days * 24 * 60
    } else if let Some(num_str) = horizon.strip_suffix('m') {
        num_str.parse::<usize>().map_err(|_| {
            crate::utils::error::VangaError::DataError(format!(
                "Invalid horizon format: {}",
                horizon
            ))
        })?
    } else {
        return Err(crate::utils::error::VangaError::DataError(format!(
            "Unsupported horizon format: {}",
            horizon
        )));
    };

    let steps = horizon_minutes / timeframe_minutes;
    if steps == 0 {
        return Err(crate::utils::error::VangaError::DataError(format!(
            "Horizon {} is smaller than timeframe ({} min)",
            horizon, timeframe_minutes
        )));
    }

    Ok(steps)
}
