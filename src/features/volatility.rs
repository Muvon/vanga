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

    // Calculate realized volatility (close-to-close)
    let returns: Vec<f64> = close_prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    let realized_vol = calculate_rolling_volatility(&returns, 24)?; // 24-period rolling

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
    let vol_of_vol = calculate_rolling_volatility(&realized_vol, 12)?; // 12-period rolling

    // Calculate GARCH-like volatility (simplified)
    let garch_vol = calculate_garch_volatility(&returns)?;

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
    let mut volatility = vec![0.0; returns.len()];

    if returns.is_empty() {
        return Ok(volatility);
    }

    // GARCH(1,1) parameters
    let omega = 0.000001;
    let alpha = 0.1;
    let beta = 0.85;

    // Initialize with first squared return
    let mut variance = if !returns.is_empty() {
        returns[0].powi(2)
    } else {
        0.0
    };
    volatility[0] = variance.sqrt();

    for i in 1..returns.len() {
        variance = omega + alpha * returns[i - 1].powi(2) + beta * variance;
        volatility[i] = variance.sqrt() * (24.0_f64).sqrt(); // Annualized
    }

    Ok(volatility)
}
