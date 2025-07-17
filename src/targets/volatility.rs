//! Volatility target generation for cryptocurrency market regime classification
//!
//! This module implements volatility regime classification:
//! - 0: Low volatility regime
//! - 1: Medium volatility regime
//! - 2: High volatility regime

use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Type alias for volatility targets with regime boundaries
type VolatilityTargetsWithBoundaries = (HashMap<String, Vec<i32>>, (f64, f64));

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

/// Generate volatility targets with consistent regime boundaries
/// Ensures training and validation use the same volatility thresholds
pub fn generate_volatility_targets_with_consistent_boundaries(
    df: &DataFrame,
    horizons: &[String],
    config: &VolatilityConfig,
    train_val_split_idx: Option<usize>,
) -> Result<VolatilityTargetsWithBoundaries> {
    let close_prices = extract_close_prices(df)?;
    let high_prices = extract_high_prices(df)?;
    let low_prices = extract_low_prices(df)?;

    // Calculate proper volatility using high/low prices for better accuracy
    let volatility = if config.use_garch_features {
        // Use range-based volatility for higher precision
        calculate_range_based_volatility(
            &close_prices,
            &high_prices,
            &low_prices,
            config.volatility_periods[0],
        )?
    } else {
        // Use close-to-close volatility as fallback
        calculate_realized_volatility(&close_prices, config.volatility_periods[0])?
    };

    // Calculate regime boundaries from training data only
    let train_volatility = if let Some(split_idx) = train_val_split_idx {
        &volatility[..split_idx.min(volatility.len())]
    } else {
        &volatility
    };

    let regime_boundaries = calculate_volatility_regime_boundaries(train_volatility, config)?;

    // Apply same boundaries to entire dataset for all horizons
    let mut all_targets = HashMap::new();
    for horizon in horizons {
        let targets = apply_volatility_boundaries(&volatility, &regime_boundaries)?;
        all_targets.insert(horizon.clone(), targets);
    }

    Ok((all_targets, regime_boundaries))
}

/// Calculate volatility regime boundaries from training data
fn calculate_volatility_regime_boundaries(
    train_volatility: &[f64],
    config: &VolatilityConfig,
) -> Result<(f64, f64)> {
    if train_volatility.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Empty volatility data for regime boundary calculation".to_string(),
        ));
    }

    let mut sorted_vol = train_volatility.to_vec();
    sorted_vol.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let (low_percentile, high_percentile) = config.regime_thresholds;

    let low_idx = (sorted_vol.len() as f64 * low_percentile) as usize;
    let high_idx = (sorted_vol.len() as f64 * high_percentile) as usize;

    let low_threshold = sorted_vol[low_idx.min(sorted_vol.len() - 1)];
    let high_threshold = sorted_vol[high_idx.min(sorted_vol.len() - 1)];

    Ok((low_threshold, high_threshold))
}

/// Apply volatility boundaries to generate targets
fn apply_volatility_boundaries(volatility: &[f64], boundaries: &(f64, f64)) -> Result<Vec<i32>> {
    let (low_threshold, high_threshold) = *boundaries;

    let targets = volatility
        .iter()
        .map(|&vol| {
            if vol <= low_threshold {
                0 // Low volatility
            } else if vol >= high_threshold {
                2 // High volatility
            } else {
                1 // Medium volatility
            }
        })
        .collect();

    Ok(targets)
}

/// Generate volatility targets for multiple horizons
pub fn generate_volatility_targets(
    df: &DataFrame,
    horizons: &[String],
    config: &VolatilityConfig,
) -> Result<HashMap<String, Vec<i32>>> {
    let close_prices = extract_close_prices(df)?;
    let high_prices = extract_high_prices(df).ok();
    let low_prices = extract_low_prices(df).ok();

    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let volatility_targets = calculate_volatility_targets_with_optimal_method(
            &close_prices,
            high_prices.as_deref(),
            low_prices.as_deref(),
            horizon_steps,
            config,
        )?;
        targets.insert(horizon.clone(), volatility_targets);
    }

    Ok(targets)
}

/// Calculate volatility targets for a specific horizon with optimal method selection
fn calculate_volatility_targets_with_optimal_method(
    close_prices: &[f64],
    high_prices: Option<&[f64]>,
    low_prices: Option<&[f64]>,
    horizon_steps: usize,
    config: &VolatilityConfig,
) -> Result<Vec<i32>> {
    if close_prices.len() < config.min_periods + horizon_steps {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient data for volatility target generation".to_string(),
        ));
    }

    // Calculate realized volatility using optimal method
    let realized_vol = calculate_optimal_volatility(
        close_prices,
        high_prices,
        low_prices,
        config.volatility_periods[0],
        config.use_garch_features,
    )?;

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

/// Calculate volatility targets for a specific horizon (legacy function for backward compatibility)
#[allow(dead_code)]
fn calculate_volatility_targets(
    close_prices: &[f64],
    high_prices: &[f64],
    low_prices: &[f64],
    horizon_steps: usize,
    config: &VolatilityConfig,
) -> Result<Vec<i32>> {
    // Use the new optimal method with explicit high/low prices
    calculate_volatility_targets_with_optimal_method(
        close_prices,
        Some(high_prices),
        Some(low_prices),
        horizon_steps,
        config,
    )
}

/// Calculate realized volatility using close-to-close returns
/// For better accuracy in crypto markets, consider using calculate_range_based_volatility instead
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
pub fn parse_horizon_to_steps(horizon: &str) -> Result<usize> {
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

/// Calculate range-based volatility using Yang-Zhang estimator
/// More accurate than close-to-close for cryptocurrency markets with high intraday volatility
fn calculate_range_based_volatility(
    close_prices: &[f64],
    high_prices: &[f64],
    low_prices: &[f64],
    window: usize,
) -> Result<Vec<f64>> {
    if close_prices.len() != high_prices.len() || close_prices.len() != low_prices.len() {
        return Err(crate::utils::error::VangaError::DataError(
            "Price arrays must have equal length for range-based volatility".to_string(),
        ));
    }

    if close_prices.len() < window + 1 {
        return Err(crate::utils::error::VangaError::DataError(format!(
            "Insufficient data for range-based volatility calculation: need {}, got {}",
            window + 1,
            close_prices.len()
        )));
    }

    let mut volatility = vec![f64::NAN; close_prices.len()];

    #[allow(clippy::needless_range_loop)]
    for i in window..close_prices.len() {
        let mut range_variance = 0.0;

        for j in (i - window)..i {
            // Yang-Zhang range-based volatility estimator
            if high_prices[j] > 0.0 && low_prices[j] > 0.0 && close_prices[j] > 0.0 {
                let log_hl = (high_prices[j] / low_prices[j]).ln();
                range_variance += log_hl * log_hl;
            }
        }

        // Calculate range-based volatility
        let range_vol = (range_variance / window as f64).sqrt();

        // Annualize for 24/7 crypto trading (8760 hours per year)
        volatility[i] = range_vol * (8760.0_f64).sqrt();
    }

    Ok(volatility)
}

/// Choose optimal volatility calculation method based on available data and configuration
fn calculate_optimal_volatility(
    close_prices: &[f64],
    high_prices: Option<&[f64]>,
    low_prices: Option<&[f64]>,
    window: usize,
    use_range_based: bool,
) -> Result<Vec<f64>> {
    match (high_prices, low_prices, use_range_based) {
        (Some(highs), Some(lows), true) => {
            // Use range-based volatility for better accuracy
            calculate_range_based_volatility(close_prices, highs, lows, window)
        }
        _ => {
            // Fallback to close-to-close volatility
            calculate_realized_volatility(close_prices, window)
        }
    }
}
