//! Direction target generation for cryptocurrency price movement prediction
//!
//! This module implements directional classification for price movements:
//! - 0: Down (significant decrease)
//! - 1: Sideways (minimal change)
//! - 2: Up (significant increase)

use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Configuration for direction target generation
#[derive(Debug, Clone)]
pub struct DirectionConfig {
    pub up_threshold: f64,
    pub down_threshold: f64,
    pub use_adaptive_thresholds: bool,
    pub volatility_window: usize,
    pub min_confidence: f64,
}

impl Default for DirectionConfig {
    fn default() -> Self {
        Self {
            up_threshold: 0.02,    // 2% increase
            down_threshold: -0.02, // 2% decrease
            use_adaptive_thresholds: true,
            volatility_window: 100,
            min_confidence: 0.6,
        }
    }
}

/// Direction classes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Down = 0,
    Sideways = 1,
    Up = 2,
}

/// Generate direction targets for multiple horizons
pub fn generate_direction_targets(
    df: &DataFrame,
    horizons: &[String],
    config: &DirectionConfig,
) -> Result<HashMap<String, Vec<i32>>> {
    let close_prices = extract_close_prices(df)?;
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let direction_targets = calculate_direction_targets(&close_prices, horizon_steps, config)?;
        targets.insert(horizon.clone(), direction_targets);
    }

    Ok(targets)
}

/// Calculate direction targets for a specific horizon
fn calculate_direction_targets(
    prices: &[f64],
    horizon_steps: usize,
    config: &DirectionConfig,
) -> Result<Vec<i32>> {
    if prices.len() < horizon_steps + config.volatility_window {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient data for direction target generation".to_string(),
        ));
    }

    let mut targets = vec![-1; prices.len()];

    // Calculate adaptive thresholds if enabled
    let thresholds = if config.use_adaptive_thresholds {
        calculate_adaptive_thresholds(prices, config)?
    } else {
        vec![(config.down_threshold, config.up_threshold); prices.len()]
    };

    for i in config.volatility_window..(prices.len() - horizon_steps) {
        let current_price = prices[i];
        let future_price = prices[i + horizon_steps];
        let price_change = (future_price - current_price) / current_price;

        let (down_threshold, up_threshold) = thresholds[i];

        // Classify direction based on thresholds
        let direction = if price_change <= down_threshold {
            Direction::Down
        } else if price_change >= up_threshold {
            Direction::Up
        } else {
            Direction::Sideways
        };

        targets[i] = direction as i32;
    }

    Ok(targets)
}

/// Calculate adaptive thresholds based on local volatility
fn calculate_adaptive_thresholds(
    prices: &[f64],
    config: &DirectionConfig,
) -> Result<Vec<(f64, f64)>> {
    let mut thresholds = Vec::with_capacity(prices.len());

    for i in 0..prices.len() {
        let start_idx = i.saturating_sub(config.volatility_window);
        let end_idx = (i + 1).min(prices.len());
        let window_prices = &prices[start_idx..end_idx];

        let volatility = calculate_local_volatility(window_prices);

        // Adapt thresholds based on volatility
        let volatility_factor = 1.0 + volatility * 2.0;
        let adaptive_up = config.up_threshold * volatility_factor;
        let adaptive_down = config.down_threshold * volatility_factor;

        thresholds.push((adaptive_down, adaptive_up));
    }

    Ok(thresholds)
}

/// Calculate local volatility for threshold adaptation
fn calculate_local_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

    if returns.is_empty() {
        return 0.0;
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    variance.sqrt()
}

/// Generate direction confidence scores
pub fn calculate_direction_confidence(
    prices: &[f64],
    targets: &[i32],
    config: &DirectionConfig,
) -> Result<Vec<f64>> {
    let mut confidence_scores = vec![0.0; targets.len()];

    for i in config.volatility_window..targets.len() {
        if targets[i] == -1 {
            continue;
        }

        let window_start = i.saturating_sub(config.volatility_window);
        let window_prices = &prices[window_start..=i];

        // Calculate confidence based on price momentum and volatility
        let momentum = calculate_momentum(window_prices, 10);
        let volatility = calculate_local_volatility(window_prices);

        // Higher momentum and lower volatility = higher confidence
        let base_confidence = momentum.abs() / (1.0 + volatility);
        confidence_scores[i] = base_confidence.clamp(0.0, 1.0);
    }

    Ok(confidence_scores)
}

/// Calculate price momentum over a window
fn calculate_momentum(prices: &[f64], window: usize) -> f64 {
    if prices.len() < window + 1 {
        return 0.0;
    }

    let start_price = prices[prices.len() - window - 1];
    let end_price = prices[prices.len() - 1];

    (end_price - start_price) / start_price
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
