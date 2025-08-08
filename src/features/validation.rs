// Validation utilities for technical indicators
use crate::utils::{Result, VangaError};

/// Validate input data for technical indicators
pub fn validate_ohlcv_data(
    open: Option<&[f64]>,
    high: Option<&[f64]>,
    low: Option<&[f64]>,
    close: &[f64],
    volume: Option<&[f64]>,
) -> Result<()> {
    if close.is_empty() {
        return Err(VangaError::FeatureError(
            "Close prices cannot be empty".to_string(),
        ));
    }

    let len = close.len();

    // Validate close prices
    for (i, &price) in close.iter().enumerate() {
        if !price.is_finite() {
            return Err(VangaError::FeatureError(format!(
                "Close price at index {} is not finite: {}",
                i, price
            )));
        }
        if price <= 0.0 {
            return Err(VangaError::FeatureError(format!(
                "Close price at index {} is not positive: {}",
                i, price
            )));
        }
    }

    // Validate high prices if provided
    if let Some(high) = high {
        if high.len() != len {
            return Err(VangaError::FeatureError(format!(
                "High prices length {} doesn't match close length {}",
                high.len(),
                len
            )));
        }
        for (i, (&h, &c)) in high.iter().zip(close.iter()).enumerate() {
            if !h.is_finite() {
                return Err(VangaError::FeatureError(format!(
                    "High price at index {} is not finite: {}",
                    i, h
                )));
            }
            if h <= 0.0 {
                return Err(VangaError::FeatureError(format!(
                    "High price at index {} is not positive: {}",
                    i, h
                )));
            }
            if h < c {
                return Err(VangaError::FeatureError(format!(
                    "High price {} is less than close price {} at index {}",
                    h, c, i
                )));
            }
        }
    }

    // Validate low prices if provided
    if let Some(low) = low {
        if low.len() != len {
            return Err(VangaError::FeatureError(format!(
                "Low prices length {} doesn't match close length {}",
                low.len(),
                len
            )));
        }
        for (i, (&l, &c)) in low.iter().zip(close.iter()).enumerate() {
            if !l.is_finite() {
                return Err(VangaError::FeatureError(format!(
                    "Low price at index {} is not finite: {}",
                    i, l
                )));
            }
            if l <= 0.0 {
                return Err(VangaError::FeatureError(format!(
                    "Low price at index {} is not positive: {}",
                    i, l
                )));
            }
            if l > c {
                return Err(VangaError::FeatureError(format!(
                    "Low price {} is greater than close price {} at index {}",
                    l, c, i
                )));
            }
        }
    }

    // Validate open prices if provided
    if let Some(open) = open {
        if open.len() != len {
            return Err(VangaError::FeatureError(format!(
                "Open prices length {} doesn't match close length {}",
                open.len(),
                len
            )));
        }
        for (i, &o) in open.iter().enumerate() {
            if !o.is_finite() {
                return Err(VangaError::FeatureError(format!(
                    "Open price at index {} is not finite: {}",
                    i, o
                )));
            }
            if o <= 0.0 {
                return Err(VangaError::FeatureError(format!(
                    "Open price at index {} is not positive: {}",
                    i, o
                )));
            }
        }
    }

    // Validate volume if provided
    if let Some(volume) = volume {
        if volume.len() != len {
            return Err(VangaError::FeatureError(format!(
                "Volume length {} doesn't match close length {}",
                volume.len(),
                len
            )));
        }
        for (i, &v) in volume.iter().enumerate() {
            if !v.is_finite() {
                return Err(VangaError::FeatureError(format!(
                    "Volume at index {} is not finite: {}",
                    i, v
                )));
            }
            if v < 0.0 {
                return Err(VangaError::FeatureError(format!(
                    "Volume at index {} is negative: {}",
                    i, v
                )));
            }
        }
    }

    // Cross-validate high/low if both provided
    if let (Some(high), Some(low)) = (high, low) {
        for (i, (&h, &l)) in high.iter().zip(low.iter()).enumerate() {
            if h < l {
                return Err(VangaError::FeatureError(format!(
                    "High price {} is less than low price {} at index {}",
                    h, l, i
                )));
            }
        }
    }

    Ok(())
}

/// Validate period parameter for indicators
pub fn validate_period(period: usize, data_len: usize, indicator_name: &str) -> Result<()> {
    if period == 0 {
        return Err(VangaError::FeatureError(format!(
            "{}: period cannot be zero",
            indicator_name
        )));
    }

    if period > data_len {
        return Err(VangaError::FeatureError(format!(
            "{}: period {} is larger than data length {}",
            indicator_name, period, data_len
        )));
    }

    // Warn if period is too large relative to data
    if period > data_len / 2 {
        log::warn!(
            "{}: period {} is more than half of data length {} - results may be unreliable",
            indicator_name,
            period,
            data_len
        );
    }

    Ok(())
}

/// Validate multiple periods
pub fn validate_periods(periods: &[usize], data_len: usize, indicator_name: &str) -> Result<()> {
    if periods.is_empty() {
        return Err(VangaError::FeatureError(format!(
            "{}: no periods specified",
            indicator_name
        )));
    }

    for &period in periods {
        validate_period(period, data_len, indicator_name)?;
    }

    Ok(())
}

/// Validate MACD parameters
pub fn validate_macd_params(
    fast: usize,
    slow: usize,
    signal: usize,
    data_len: usize,
) -> Result<()> {
    if fast >= slow {
        return Err(VangaError::FeatureError(format!(
            "MACD: fast period {} must be less than slow period {}",
            fast, slow
        )));
    }

    validate_period(fast, data_len, "MACD fast")?;
    validate_period(slow, data_len, "MACD slow")?;
    validate_period(signal, data_len, "MACD signal")?;

    Ok(())
}

/// Validate Bollinger Bands parameters
pub fn validate_bollinger_params(period: usize, std_dev: f64, data_len: usize) -> Result<()> {
    validate_period(period, data_len, "Bollinger Bands")?;

    if !std_dev.is_finite() || std_dev <= 0.0 {
        return Err(VangaError::FeatureError(format!(
            "Bollinger Bands: standard deviation must be positive and finite, got {}",
            std_dev
        )));
    }

    if std_dev > 5.0 {
        log::warn!(
            "Bollinger Bands: standard deviation {} is unusually high",
            std_dev
        );
    }

    Ok(())
}

/// Validate Stochastic parameters
pub fn validate_stochastic_params(k_period: usize, d_period: usize, data_len: usize) -> Result<()> {
    validate_period(k_period, data_len, "Stochastic %K")?;
    validate_period(d_period, data_len, "Stochastic %D")?;

    Ok(())
}

/// Check for sufficient data variation to avoid constant values
pub fn check_data_variation(data: &[f64], indicator_name: &str, min_variation: f64) -> Result<()> {
    if data.len() < 2 {
        return Ok(()); // Can't check variation with less than 2 points
    }

    let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    if range < min_variation {
        log::warn!(
            "{}: data has very low variation (range: {:.6}), indicator may produce constant values",
            indicator_name,
            range
        );
    }

    Ok(())
}

/// Sanitize indicator output to handle edge cases
pub fn sanitize_indicator_output(
    mut values: Vec<f64>,
    indicator_name: &str,
    default_value: f64,
    valid_range: Option<(f64, f64)>,
) -> Vec<f64> {
    let mut sanitized_count = 0;

    for value in values.iter_mut() {
        let mut needs_sanitization = false;

        // Skip NaN values - they are intentional for warm-up periods
        if value.is_nan() {
            continue;
        }

        // Check for other non-finite values (infinity)
        if !value.is_finite() {
            needs_sanitization = true;
        }

        // Check for values outside valid range
        if let Some((min_val, max_val)) = valid_range {
            if *value < min_val || *value > max_val {
                needs_sanitization = true;
            }
        }

        if needs_sanitization {
            *value = default_value;
            sanitized_count += 1;
        }
    }

    if sanitized_count > 0 {
        log::warn!(
            "{}: sanitized {} out of {} values to default value {}",
            indicator_name,
            sanitized_count,
            values.len(),
            default_value
        );
    }

    values
}

/// Create default values for failed indicator calculations
pub fn create_default_values(len: usize, default_value: f64) -> Vec<f64> {
    vec![default_value; len]
}

/// Log indicator calculation statistics
pub fn log_indicator_stats(values: &[f64], indicator_name: &str) {
    if values.is_empty() {
        log::warn!("{}: no values calculated", indicator_name);
        return;
    }

    let finite_values: Vec<f64> = values.iter().filter(|&&x| x.is_finite()).copied().collect();

    if finite_values.is_empty() {
        log::warn!("{}: all values are non-finite", indicator_name);
        return;
    }

    let min_val = finite_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = finite_values
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean_val = finite_values.iter().sum::<f64>() / finite_values.len() as f64;
    let non_finite_count = values.len() - finite_values.len();

    log::debug!(
        "{}: {} values, range: [{:.4}, {:.4}], mean: {:.4}, non-finite: {}",
        indicator_name,
        values.len(),
        min_val,
        max_val,
        mean_val,
        non_finite_count
    );
}
