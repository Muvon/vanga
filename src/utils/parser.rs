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
        DataType::String => {
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

    // Determine raw-units-per-minute from the KNOWN time unit instead of guessing
    // by magnitude. Polars stores the TimeUnit in the Datetime dtype, and strings
    // are cast to Datetime(Microseconds) above — both are exact. Only raw Int64
    // epoch columns carry no unit metadata, so those fall back to the magnitude
    // heuristic below.
    let known_per_minute: Option<i64> = match timestamp_col.dtype() {
        DataType::Datetime(TimeUnit::Nanoseconds, _) => Some(60_000_000_000),
        DataType::Datetime(TimeUnit::Microseconds, _) => Some(60_000_000),
        DataType::Datetime(TimeUnit::Milliseconds, _) => Some(60_000),
        DataType::String => Some(60_000_000),
        _ => None,
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

    // Convert the median diff to minutes. When the unit is known (Datetime/String)
    // we divide deterministically. Magnitude guessing only applies to raw Int64
    // epoch columns, where the unit is genuinely unknown — and even then the ranges
    // overlap for long timeframes, so prefer typed timestamps.
    let timeframe_minutes = match known_per_minute {
        Some(per_minute) => (median_diff / per_minute) as usize,
        None => {
            // Raw Int64 epoch column: guess unit from magnitude (ms/s are typical).
            if median_diff >= 1_000_000_000 {
                (median_diff / 60_000_000_000) as usize // nanoseconds
            } else if median_diff >= 1_000_000 {
                (median_diff / 60_000_000) as usize // microseconds
            } else if median_diff >= 10_000 {
                (median_diff / 60_000) as usize // milliseconds
            } else {
                (median_diff / 60) as usize // seconds
            }
        }
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
