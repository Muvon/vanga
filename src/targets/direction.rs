//! Direction target generation for cryptocurrency price movement prediction
//!
//! This module implements directional classification for price movements:
//! - 0: Down (significant decrease)
//! - 1: Sideways (minimal change)
//! - 2: Up (significant increase)

use crate::config::model::DirectionHead;
use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Configuration for direction target generation
#[derive(Debug, Clone)]
pub struct DirectionConfig {
    pub threshold: f64, // Single threshold for symmetric up/down
    pub up_threshold: f64,
    pub down_threshold: f64,
    pub use_adaptive_thresholds: bool,
    pub volatility_window: usize,
    pub min_confidence: f64,
    /// Multiplier for extreme thresholds (pump/dump detection)
    pub extreme_multiplier: f64,
}

impl Default for DirectionConfig {
    fn default() -> Self {
        Self {
            threshold: 0.02,       // 2% symmetric threshold
            up_threshold: 0.02,    // 2% increase
            down_threshold: -0.02, // 2% decrease
            use_adaptive_thresholds: true,
            volatility_window: 100,
            min_confidence: 0.6,
            extreme_multiplier: 2.5, // 5% for pump/dump vs 2% for up/down
        }
    }
}

impl DirectionConfig {
    /// Create configuration with custom extreme multiplier
    pub fn with_extreme_multiplier(multiplier: f64) -> Self {
        Self {
            extreme_multiplier: multiplier,
            ..Default::default()
        }
    }
}

/// Direction classes (5-class system)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Dump = 0,     // Extreme down (< extreme_down_threshold)
    Down = 1,     // Moderate down (extreme_down ≤ x < down_threshold)
    Sideways = 2, // Minimal change (down_threshold ≤ x < up_threshold)
    Up = 3,       // Moderate up (up_threshold ≤ x < extreme_up_threshold)
    Pump = 4,     // Extreme up (≥ extreme_up_threshold)
}

/// Generate direction targets with volatility-adaptive thresholds
/// Uses market volatility to adjust classification boundaries for better accuracy
pub fn generate_direction_targets_with_adaptive_thresholds(
    prices: &[f64],
    horizon_steps: usize,
    config: &DirectionConfig,
    train_val_split_idx: Option<usize>,
) -> Result<(Vec<i32>, (f64, f64))> {
    if prices.len() < horizon_steps + 1 {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient price data for direction target generation".to_string(),
        ));
    }

    // Calculate thresholds from training data only
    let train_prices = if let Some(split_idx) = train_val_split_idx {
        &prices[..split_idx.min(prices.len())]
    } else {
        prices
    };

    let thresholds = calculate_volatility_adaptive_thresholds_from_training(train_prices, config)?;

    // Apply same thresholds to entire dataset
    let targets = apply_direction_thresholds(prices, horizon_steps, &thresholds, config)?;

    Ok((targets, thresholds))
}

/// Calculate volatility-adaptive thresholds from training data
/// Adjusts thresholds based on market volatility for better accuracy
fn calculate_volatility_adaptive_thresholds_from_training(
    train_prices: &[f64],
    config: &DirectionConfig,
) -> Result<(f64, f64)> {
    if config.use_adaptive_thresholds {
        // Calculate rolling volatility
        let volatility_window = config.volatility_window.min(train_prices.len() / 2);
        let mut volatilities = Vec::new();

        for i in volatility_window..train_prices.len() {
            let window = &train_prices[i.saturating_sub(volatility_window)..i];
            let returns: Vec<f64> = window.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

            if !returns.is_empty() {
                let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns
                    .iter()
                    .map(|r| (r - mean_return).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                volatilities.push(variance.sqrt());
            }
        }

        if volatilities.is_empty() {
            return Ok((config.down_threshold, config.up_threshold));
        }

        // Calculate average volatility for adaptive scaling
        let avg_volatility = if volatilities.is_empty() {
            0.01
        } else {
            volatilities.iter().sum::<f64>() / volatilities.len() as f64
        };

        // Use calculate_volatility helper for consistency
        let market_volatility = calculate_volatility(train_prices, config.volatility_window);
        let combined_volatility = (avg_volatility + market_volatility) / 2.0;

        // Adjust thresholds based on combined volatility
        let volatility_multiplier = (combined_volatility * 100.0).clamp(0.5, 3.0); // Scale and clamp
        let adaptive_threshold = config.threshold * volatility_multiplier;

        // Apply min/max constraints
        let final_threshold = adaptive_threshold
            .max(config.threshold * 0.5) // At least 50% of base threshold
            .min(config.threshold * 2.0); // At most 200% of base threshold

        Ok((-final_threshold, final_threshold))
    } else {
        Ok((config.down_threshold, config.up_threshold))
    }
}

/// Apply direction thresholds to generate targets (5-class system)
fn apply_direction_thresholds(
    prices: &[f64],
    horizon_steps: usize,
    thresholds: &(f64, f64),
    config: &DirectionConfig,
) -> Result<Vec<i32>> {
    let (down_threshold, up_threshold) = *thresholds;
    let mut targets = vec![-1; prices.len()];

    // Calculate extreme thresholds
    let extreme_down = down_threshold * config.extreme_multiplier;
    let extreme_up = up_threshold * config.extreme_multiplier;

    for i in 0..(prices.len().saturating_sub(horizon_steps)) {
        let current_price = prices[i];
        let future_price = prices[i + horizon_steps];
        let price_change = (future_price - current_price) / current_price;

        targets[i] = if price_change <= extreme_down {
            0 // Dump
        } else if price_change <= down_threshold {
            1 // Down
        } else if price_change < up_threshold {
            2 // Sideways
        } else if price_change < extreme_up {
            3 // Up
        } else {
            4 // Pump
        };
    }

    Ok(targets)
}

/// Calculate volatility for adaptive thresholds
fn calculate_volatility(prices: &[f64], window: usize) -> f64 {
    if prices.len() < window + 1 {
        return 0.01; // Default volatility
    }

    let mut returns = Vec::new();
    for i in 1..=window.min(prices.len() - 1) {
        let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
        returns.push(ret);
    }

    if returns.is_empty() {
        return 0.01;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

    variance.sqrt()
}

/// Generate direction targets for multiple horizons (ENHANCED: supports both configs)
pub fn generate_direction_targets(
    df: &DataFrame,
    horizons: &[String],
    config: Option<&DirectionConfig>,
    model_config: Option<&DirectionHead>,
) -> Result<HashMap<String, Vec<i32>>> {
    let close_prices = extract_close_prices(df)?;
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;

        let direction_targets = if let Some(model_cfg) = model_config {
            // Use model config (NEW: eliminates hardcoded values)
            let bandwidth_size = model_cfg.bandwidth_size.unwrap_or(1.0);
            let market_volatility = calculate_market_volatility(&close_prices)?;
            let base_threshold = market_volatility * model_cfg.base_threshold_factor;
            let adaptive_threshold = base_threshold / bandwidth_size;

            let thresholds = [
                -adaptive_threshold * model_cfg.extreme_multiplier, // Dump
                -adaptive_threshold,                                // Down
                adaptive_threshold,                                 // Up
                adaptive_threshold * model_cfg.extreme_multiplier,  // Pump
            ];

            apply_direction_classification(&close_prices, horizon_steps, &thresholds)?
        } else if let Some(legacy_cfg) = config {
            // Legacy path for backward compatibility
            calculate_direction_targets(&close_prices, horizon_steps, legacy_cfg)?
        } else {
            return Err(crate::utils::error::VangaError::config(
                "Either DirectionConfig or DirectionHead must be provided",
            ));
        };

        targets.insert(horizon.clone(), direction_targets);
    }

    Ok(targets)
}

/// Calculate market volatility for adaptive threshold scaling
fn calculate_market_volatility(prices: &[f64]) -> Result<f64> {
    if prices.len() < 2 {
        return Ok(0.01); // Default minimum volatility
    }

    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

    if returns.is_empty() {
        return Ok(0.01);
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    Ok(variance.sqrt().max(0.01)) // Ensure minimum volatility
}

/// Apply direction classification using threshold array
fn apply_direction_classification(
    prices: &[f64],
    horizon_steps: usize,
    thresholds: &[f64; 4], // [dump, down, up, pump]
) -> Result<Vec<i32>> {
    if prices.len() < horizon_steps + 1 {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient data for direction classification".to_string(),
        ));
    }

    let mut targets = vec![-1; prices.len()];

    for i in 0..(prices.len() - horizon_steps) {
        let current_price = prices[i];
        let future_price = prices[i + horizon_steps];
        let price_change = (future_price - current_price) / current_price;

        let direction = if price_change <= thresholds[0] {
            Direction::Dump as i32
        } else if price_change <= thresholds[1] {
            Direction::Down as i32
        } else if price_change < thresholds[2] {
            Direction::Sideways as i32
        } else if price_change < thresholds[3] {
            Direction::Up as i32
        } else {
            Direction::Pump as i32
        };

        targets[i] = direction;
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

        // Classify direction based on thresholds (5-class system)
        let extreme_down = down_threshold * config.extreme_multiplier;
        let extreme_up = up_threshold * config.extreme_multiplier;

        let direction = if price_change <= extreme_down {
            Direction::Dump as i32
        } else if price_change <= down_threshold {
            Direction::Down as i32
        } else if price_change < up_threshold {
            Direction::Sideways as i32
        } else if price_change < extreme_up {
            Direction::Up as i32
        } else {
            Direction::Pump as i32
        };

        targets[i] = direction;
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
