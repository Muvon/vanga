// Volatility features
use crate::config::features::VolatilityFeaturesConfig;
use crate::utils::error::Result;
use polars::prelude::*;

pub async fn generate_volatility_features(
    mut df: DataFrame,
    config: &VolatilityFeaturesConfig,
) -> Result<DataFrame> {
    log::info!("Generating volatility features with config: {:?}", config);

    // Extract price data
    let close_prices = crate::features::technical::extract_numeric_column(&df, "close")?;
    let high_prices = crate::features::technical::extract_numeric_column(&df, "high")?;
    let low_prices = crate::features::technical::extract_numeric_column(&df, "low")?;

    // Calculate realized volatility (close-to-close) with NaN handling
    let returns: Vec<f64> = close_prices
        .windows(2)
        .map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 && w[0].is_finite() && w[1].is_finite() {
                (w[1] / w[0]).ln()
            } else {
                f64::NAN
            }
        })
        .collect();

    let mut realized_vol = calculate_rolling_volatility(&returns, 24)?; // 24-period rolling
                                                                        // Pad to match original DataFrame length (returns is 1 shorter than close_prices)
    realized_vol.insert(0, f64::NAN);

    // Calculate range-based volatility (Parkinson estimator)
    let range_vol: Vec<f64> = high_prices
        .iter()
        .zip(low_prices.iter())
        .map(|(h, l)| {
            if *h > 0.0 && *l > 0.0 {
                ((h / l).ln()).powi(2) / (4.0 * (2.0_f64).ln())
            } else {
                0.0
            }
        })
        .collect();

    // Calculate volatility of volatility
    let mut vol_of_vol = calculate_rolling_volatility(&realized_vol[1..], 12)?; // Skip the NaN we added
    vol_of_vol.insert(0, f64::NAN); // Pad to match DataFrame length

    // Calculate GARCH-like volatility (simplified)
    let mut garch_vol = calculate_garch_volatility(&returns)?;
    garch_vol.insert(0, f64::NAN); // Pad to match DataFrame length

    // Add volatility features to DataFrame one by one
    df = df
        .with_column(Series::new("realized_volatility", realized_vol))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add realized_volatility column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("range_volatility", range_vol))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add range_volatility column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("volatility_of_volatility", vol_of_vol))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add volatility_of_volatility column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("garch_volatility", garch_vol))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add garch_volatility column: {}",
                e
            ))
        })?
        .clone();

    Ok(df)
}

/// Calculate rolling volatility
fn calculate_rolling_volatility(returns: &[f64], window: usize) -> Result<Vec<f64>> {
    let mut volatility = vec![0.0; returns.len()];

    for i in window..returns.len() {
        let window_returns = &returns[i - window..i];
        let mean = window_returns.iter().sum::<f64>() / window as f64;
        let variance = window_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (window - 1) as f64;
        volatility[i] = variance.sqrt() * (24.0_f64).sqrt(); // Annualized
    }

    Ok(volatility)
}

/// Calculate GARCH-like volatility (simplified)
fn calculate_garch_volatility(returns: &[f64]) -> Result<Vec<f64>> {
    let mut volatility = vec![f64::NAN; returns.len()];

    if returns.is_empty() {
        return Ok(volatility);
    }

    // GARCH(1,1) parameters
    let omega = 0.000_001;
    let alpha = 0.1;
    let beta = 0.85;

    // Find first valid return to initialize
    let mut first_valid_idx = None;
    for (i, &ret) in returns.iter().enumerate() {
        if ret.is_finite() {
            first_valid_idx = Some(i);
            break;
        }
    }

    let start_idx = match first_valid_idx {
        Some(idx) => idx,
        None => {
            log::warn!("No valid returns found for GARCH calculation");
            return Ok(volatility);
        }
    };

    // Initialize with first valid squared return
    let mut variance = returns[start_idx].powi(2);
    if !variance.is_finite() || variance < 0.0 {
        variance = 0.000_001; // Fallback to small positive value
    }

    volatility[start_idx] = variance.sqrt();

    // Calculate GARCH volatility for remaining valid returns
    for i in (start_idx + 1)..returns.len() {
        let prev_return = returns[i - 1];

        // Skip if previous return is not finite
        if !prev_return.is_finite() {
            continue;
        }

        variance = omega + alpha * prev_return.powi(2) + beta * variance;

        // Ensure variance is positive and finite
        if !variance.is_finite() || variance < 0.0 {
            variance = 0.000_001; // Reset to small positive value
        }

        let vol = variance.sqrt() * (24.0_f64).sqrt(); // Annualized

        // Only set if result is finite
        if vol.is_finite() {
            volatility[i] = vol;
        }
    }

    Ok(volatility)
}
