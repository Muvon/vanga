//! Volatility target generation for cryptocurrency market regime classification
//!
//! This module implements volatility regime classification:
//! - 0: Low volatility regime
//! - 1: Medium volatility regime
//! - 2: High volatility regime

use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Configuration for volatility target generation
#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    pub volatility_periods: Vec<usize>,
    pub regime_thresholds: (f64, f64), // (low_percentile, high_percentile)
    pub smoothing_window: usize,
    pub use_garch_features: bool,
    pub min_periods: usize,
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            volatility_periods: vec![24, 48, 168], // 1d, 2d, 1w
            regime_thresholds: (0.33, 0.67),       // 33rd and 67th percentiles
            smoothing_window: 12,
            use_garch_features: false,
            min_periods: 100,
        }
    }
}

/// Volatility regime classes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolatilityRegime {
    Low = 0,
    Medium = 1,
    High = 2,
}

/// Generate volatility targets for multiple horizons
pub fn generate_volatility_targets(
    df: &DataFrame,
    horizons: &[String],
    config: &VolatilityConfig,
) -> Result<HashMap<String, Vec<i32>>> {
    let close_prices = extract_close_prices(df)?;
    let high_prices = extract_high_prices(df)?;
    let low_prices = extract_low_prices(df)?;

    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let volatility_targets = calculate_volatility_targets(
            &close_prices,
            &high_prices,
            &low_prices,
            horizon_steps,
            config,
        )?;
        targets.insert(horizon.clone(), volatility_targets);
    }

    Ok(targets)
}

/// Calculate volatility targets for a specific horizon
fn calculate_volatility_targets(
    close_prices: &[f64],
    _high_prices: &[f64],
    _low_prices: &[f64],
    horizon_steps: usize,
    config: &VolatilityConfig,
) -> Result<Vec<i32>> {
    if close_prices.len() < config.min_periods + horizon_steps {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient data for volatility target generation".to_string(),
        ));
    }

    // Calculate realized volatility
    let realized_vol = calculate_realized_volatility(close_prices, config.volatility_periods[0])?;

    // Calculate forward-looking volatility
    let forward_vol =
        calculate_forward_volatility(close_prices, horizon_steps, config.volatility_periods[0])?;

    // Determine regime thresholds
    let thresholds = calculate_regime_thresholds(&realized_vol, config)?;

    // Classify volatility regimes
    let mut targets = vec![-1; close_prices.len()];

    for i in config.min_periods..(close_prices.len() - horizon_steps) {
        if i >= forward_vol.len() {
            break;
        }

        let future_volatility = forward_vol[i];
        let regime = classify_volatility_regime(future_volatility, &thresholds);
        targets[i] = regime as i32;
    }

    Ok(targets)
}

/// Calculate realized volatility using close-to-close returns
fn calculate_realized_volatility(prices: &[f64], window: usize) -> Result<Vec<f64>> {
    if prices.len() < window + 1 {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient data for realized volatility".to_string(),
        ));
    }

    let mut volatility = vec![f64::NAN; prices.len()];

    // Calculate log returns
    let mut returns = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        returns.push((prices[i] / prices[i - 1]).ln());
    }

    // Calculate rolling volatility
    for i in window..returns.len() {
        let window_returns = &returns[i - window..i];
        let mean_return = window_returns.iter().sum::<f64>() / window as f64;

        let variance = window_returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / (window - 1) as f64;

        volatility[i + 1] = variance.sqrt() * (24.0_f64).sqrt(); // Annualized
    }

    Ok(volatility)
}

/// Calculate forward-looking volatility
fn calculate_forward_volatility(
    prices: &[f64],
    horizon_steps: usize,
    vol_window: usize,
) -> Result<Vec<f64>> {
    if prices.len() < vol_window + horizon_steps {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient data for forward volatility".to_string(),
        ));
    }

    let mut forward_vol = vec![f64::NAN; prices.len()];

    let end_range = prices.len() - horizon_steps - vol_window;
    #[allow(clippy::needless_range_loop)]
    for i in 0..end_range {
        let future_start = i + horizon_steps;
        let future_end = future_start + vol_window;

        if future_end <= prices.len() {
            let future_prices = &prices[future_start..future_end];
            let vol = calculate_window_volatility(future_prices)?;
            forward_vol[i] = vol;
        }
    }

    Ok(forward_vol)
}

/// Calculate volatility for a specific window
fn calculate_window_volatility(prices: &[f64]) -> Result<f64> {
    if prices.len() < 2 {
        return Ok(0.0);
    }

    let mut returns = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        returns.push((prices[i] / prices[i - 1]).ln());
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;

    Ok(variance.sqrt() * (24.0_f64).sqrt())
}

/// Calculate regime thresholds based on historical volatility distribution
fn calculate_regime_thresholds(
    volatility: &[f64],
    config: &VolatilityConfig,
) -> Result<(f64, f64)> {
    let valid_vol: Vec<f64> = volatility
        .iter()
        .filter(|&&v| !v.is_nan() && v.is_finite())
        .copied()
        .collect();

    if valid_vol.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "No valid volatility data for thresholds".to_string(),
        ));
    }

    let mut sorted_vol = valid_vol.clone();
    sorted_vol.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let low_idx = (config.regime_thresholds.0 * (sorted_vol.len() - 1) as f64) as usize;
    let high_idx = (config.regime_thresholds.1 * (sorted_vol.len() - 1) as f64) as usize;

    Ok((sorted_vol[low_idx], sorted_vol[high_idx]))
}

/// Classify volatility into regime
fn classify_volatility_regime(volatility: f64, thresholds: &(f64, f64)) -> VolatilityRegime {
    if volatility <= thresholds.0 {
        VolatilityRegime::Low
    } else if volatility <= thresholds.1 {
        VolatilityRegime::Medium
    } else {
        VolatilityRegime::High
    }
}

/// Extract close prices from DataFrame
fn extract_close_prices(df: &DataFrame) -> Result<Vec<f64>> {
    let close_series = df.column("close").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!("Failed to get close column: {}", e))
    })?;

    let values: Vec<f64> = close_series
        .f64()
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to convert close to f64: {}",
                e
            ))
        })?
        .into_no_null_iter()
        .collect();

    Ok(values)
}

/// Extract high prices from DataFrame
fn extract_high_prices(df: &DataFrame) -> Result<Vec<f64>> {
    let high_series = df.column("high").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!("Failed to get high column: {}", e))
    })?;

    let values: Vec<f64> = high_series
        .f64()
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to convert high to f64: {}",
                e
            ))
        })?
        .into_no_null_iter()
        .collect();

    Ok(values)
}

/// Extract low prices from DataFrame
fn extract_low_prices(df: &DataFrame) -> Result<Vec<f64>> {
    let low_series = df.column("low").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!("Failed to get low column: {}", e))
    })?;

    let values: Vec<f64> = low_series
        .f64()
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to convert low to f64: {}",
                e
            ))
        })?
        .into_no_null_iter()
        .collect();

    Ok(values)
}

/// Parse horizon string to number of steps
fn parse_horizon_to_steps(horizon: &str) -> Result<usize> {
    match horizon {
        "1h" => Ok(1),
        "4h" => Ok(4),
        "1d" => Ok(24),
        "7d" => Ok(168),
        _ => {
            if let Some(num_str) = horizon.strip_suffix('h') {
                num_str.parse::<usize>().map_err(|_| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Invalid horizon format: {}",
                        horizon
                    ))
                })
            } else if let Some(num_str) = horizon.strip_suffix('d') {
                num_str.parse::<usize>().map(|d| d * 24).map_err(|_| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Invalid horizon format: {}",
                        horizon
                    ))
                })
            } else {
                Err(crate::utils::error::VangaError::DataError(format!(
                    "Unsupported horizon format: {}",
                    horizon
                )))
            }
        }
    }
}
