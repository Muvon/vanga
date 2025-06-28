//! Price level target generation for cryptocurrency forecasting
//!
//! This module implements quantile-based price level classification for LSTM training.
//! Price levels are calculated using dynamic quantiles to create balanced target distributions.

use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Configuration for price level target generation
#[derive(Debug, Clone)]
pub struct PriceLevelConfig {
    pub bins: u32,
    pub quantile_method: QuantileMethod,
    pub lookback_window: usize,
    pub min_price_change: f64,
}

#[derive(Debug, Clone)]
pub enum QuantileMethod {
    Fixed,
    Rolling { window: usize },
    Adaptive { min_samples: usize },
}

impl Default for PriceLevelConfig {
    fn default() -> Self {
        Self {
            bins: 5,
            quantile_method: QuantileMethod::Rolling { window: 1000 },
            lookback_window: 100,
            min_price_change: 0.001, // 0.1% minimum change
        }
    }
}

/// Generate price level targets for multiple horizons
pub fn generate_price_level_targets(
    df: &DataFrame,
    horizons: &[String],
    config: &PriceLevelConfig,
) -> Result<HashMap<String, Vec<i32>>> {
    let close_prices = extract_close_prices(df)?;
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let price_targets = calculate_price_level_targets(&close_prices, horizon_steps, config)?;
        targets.insert(horizon.clone(), price_targets);
    }

    Ok(targets)
}

/// Calculate price level targets for a specific horizon
fn calculate_price_level_targets(
    prices: &[f64],
    horizon_steps: usize,
    config: &PriceLevelConfig,
) -> Result<Vec<i32>> {
    if prices.len() < horizon_steps + config.lookback_window {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient data for price level target generation".to_string(),
        ));
    }

    let mut targets = vec![-1; prices.len()];

    for i in config.lookback_window..(prices.len() - horizon_steps) {
        let current_price = prices[i];
        let future_price = prices[i + horizon_steps];
        let price_change = (future_price - current_price) / current_price;

        // Skip if price change is too small
        if price_change.abs() < config.min_price_change {
            targets[i] = config.bins as i32 / 2; // Neutral class
            continue;
        }

        // Calculate quantiles for price level classification
        let quantiles = calculate_quantiles(
            &prices[i.saturating_sub(config.lookback_window)..=i],
            config.bins,
            &config.quantile_method,
        )?;

        // Classify future price into quantile bins
        targets[i] = classify_price_to_level(future_price, &quantiles);
    }

    Ok(targets)
}

/// Calculate dynamic quantiles for price classification
fn calculate_quantiles(prices: &[f64], bins: u32, method: &QuantileMethod) -> Result<Vec<f64>> {
    if prices.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Empty price data for quantile calculation".to_string(),
        ));
    }

    let mut sorted_prices = prices.to_vec();
    sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut quantiles = Vec::new();

    match method {
        QuantileMethod::Fixed => {
            for i in 1..bins {
                let percentile = i as f64 / bins as f64;
                let index = (percentile * (sorted_prices.len() - 1) as f64) as usize;
                quantiles.push(sorted_prices[index]);
            }
        }
        QuantileMethod::Rolling { window } => {
            let effective_window = (*window).min(prices.len());
            let recent_prices = &sorted_prices[sorted_prices.len() - effective_window..];

            for i in 1..bins {
                let percentile = i as f64 / bins as f64;
                let index = (percentile * (recent_prices.len() - 1) as f64) as usize;
                quantiles.push(recent_prices[index]);
            }
        }
        QuantileMethod::Adaptive { min_samples } => {
            if sorted_prices.len() < *min_samples {
                return calculate_quantiles(prices, bins, &QuantileMethod::Fixed);
            }

            // Use adaptive quantiles based on price volatility
            let volatility = calculate_price_volatility(&sorted_prices);
            let adaptive_factor = 1.0 + volatility * 0.5;

            for i in 1..bins {
                let base_percentile = i as f64 / bins as f64;
                let adaptive_percentile = (base_percentile * adaptive_factor).clamp(0.05, 0.95);
                let index = (adaptive_percentile * (sorted_prices.len() - 1) as f64) as usize;
                quantiles.push(sorted_prices[index]);
            }
        }
    }

    Ok(quantiles)
}

/// Classify a price into a quantile level
fn classify_price_to_level(price: f64, quantiles: &[f64]) -> i32 {
    for (i, &threshold) in quantiles.iter().enumerate() {
        if price <= threshold {
            return i as i32;
        }
    }
    quantiles.len() as i32
}

/// Calculate price volatility for adaptive quantiles
fn calculate_price_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    variance.sqrt()
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
