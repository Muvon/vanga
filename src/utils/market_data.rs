use crate::data::structures::MarketDataRow;
use crate::utils::error::Result;
use polars::frame::DataFrame;

/// Extract OHLCV data from DataFrame for ATR calculations
pub fn extract_ohlcv_data(df: &DataFrame) -> Result<Vec<MarketDataRow>> {
    let timestamp_col = df.column("timestamp").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!(
            "Failed to extract 'timestamp' column: {}",
            e
        ))
    })?;
    let open_col = df.column("open").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!(
            "Failed to extract 'open' column: {}",
            e
        ))
    })?;
    let high_col = df.column("high").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!(
            "Failed to extract 'high' column: {}",
            e
        ))
    })?;
    let low_col = df.column("low").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!("Failed to extract 'low' column: {}", e))
    })?;
    let close_col = df.column("close").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!(
            "Failed to extract 'close' column: {}",
            e
        ))
    })?;
    let volume_col = df.column("volume").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!(
            "Failed to extract 'volume' column: {}",
            e
        ))
    })?;

    let mut ohlcv_data = Vec::new();

    for i in 0..df.height() {
        // Extract timestamp
        let timestamp = match timestamp_col.get(i).map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to get timestamp at row {}: {}",
                i, e
            ))
        })? {
            polars::prelude::AnyValue::Datetime(dt, _, _) => dt / 1_000_000, // Convert microseconds to seconds
            polars::prelude::AnyValue::Int64(ts) => ts,
            polars::prelude::AnyValue::Utf8(s) => chrono::DateTime::parse_from_rfc3339(s)
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to parse timestamp '{}': {}",
                        s, e
                    ))
                })?
                .timestamp(),
            _ => {
                return Err(crate::utils::error::VangaError::DataError(format!(
                    "Unsupported timestamp type at row {}",
                    i
                )))
            }
        };

        // Extract OHLCV values
        let open = extract_f64_value(
            open_col.get(i).map_err(|e| {
                crate::utils::error::VangaError::DataError(format!(
                    "Failed to get open at row {}: {}",
                    i, e
                ))
            })?,
            "open",
            i,
        )?;

        let high = extract_f64_value(
            high_col.get(i).map_err(|e| {
                crate::utils::error::VangaError::DataError(format!(
                    "Failed to get high at row {}: {}",
                    i, e
                ))
            })?,
            "high",
            i,
        )?;

        let low = extract_f64_value(
            low_col.get(i).map_err(|e| {
                crate::utils::error::VangaError::DataError(format!(
                    "Failed to get low at row {}: {}",
                    i, e
                ))
            })?,
            "low",
            i,
        )?;

        let close = extract_f64_value(
            close_col.get(i).map_err(|e| {
                crate::utils::error::VangaError::DataError(format!(
                    "Failed to get close at row {}: {}",
                    i, e
                ))
            })?,
            "close",
            i,
        )?;

        let volume = extract_f64_value(
            volume_col.get(i).map_err(|e| {
                crate::utils::error::VangaError::DataError(format!(
                    "Failed to get volume at row {}: {}",
                    i, e
                ))
            })?,
            "volume",
            i,
        )?;

        ohlcv_data.push(MarketDataRow::new(
            timestamp, open, high, low, close, volume,
        ));
    }

    if ohlcv_data.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "No valid OHLCV data found".to_string(),
        ));
    }

    Ok(ohlcv_data)
}

/// Helper function to extract f64 value from AnyValue
fn extract_f64_value(
    value: polars::prelude::AnyValue,
    column_name: &str,
    row: usize,
) -> Result<f64> {
    match value {
        polars::prelude::AnyValue::Float64(f) => Ok(f),
        polars::prelude::AnyValue::Float32(f) => Ok(f as f64),
        polars::prelude::AnyValue::Int64(i) => Ok(i as f64),
        polars::prelude::AnyValue::Int32(i) => Ok(i as f64),
        polars::prelude::AnyValue::Int16(i) => Ok(i as f64),
        polars::prelude::AnyValue::Int8(i) => Ok(i as f64),
        polars::prelude::AnyValue::UInt64(i) => Ok(i as f64),
        polars::prelude::AnyValue::UInt32(i) => Ok(i as f64),
        polars::prelude::AnyValue::UInt16(i) => Ok(i as f64),
        polars::prelude::AnyValue::UInt8(i) => Ok(i as f64),
        polars::prelude::AnyValue::Null => Err(crate::utils::error::VangaError::DataError(format!(
            "NULL value in {} column at row {} (CSV row {}). Check your data file for missing values.",
            column_name, row, row + 1
        ))),
        polars::prelude::AnyValue::Utf8(s) => {
            // Try to parse string as number
            s.parse::<f64>().map_err(|_| {
                crate::utils::error::VangaError::DataError(format!(
                    "Cannot parse {} value '{}' as number at row {} (CSV row {})",
                    column_name, s, row, row + 1
                ))
            })
        }
        _ => Err(crate::utils::error::VangaError::DataError(format!(
            "Unsupported {} type at row {} (CSV row {}): {:?}",
            column_name, row, row + 1, value
        ))),
    }
}

/// Extract close prices from DataFrame
pub fn extract_close_prices(df: &DataFrame) -> Result<Vec<f64>> {
    let close_series = df.column("close").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!(
            "Failed to extract 'close' column: {}",
            e
        ))
    })?;

    let values: Vec<f64> = close_series
        .f64()
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to convert close prices to f64: {}",
                e
            ))
        })?
        .into_no_null_iter()
        .collect();

    if values.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "No valid close prices found".to_string(),
        ));
    }

    Ok(values)
}
