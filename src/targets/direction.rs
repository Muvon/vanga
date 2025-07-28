//! Direction target generation for cryptocurrency price movement prediction
//!
//! This module implements directional classification for price movements:
//! - 0: Down (significant decrease)
//! - 1: Sideways (minimal change)
//! - 2: Up (significant increase)

use crate::config::model::DirectionHead;
use crate::utils::error::Result;
use crate::utils::parser::parse_horizon_to_steps;
use polars::prelude::*;
use std::collections::HashMap;

// DEPRECATED: DirectionConfig has been removed in favor of DirectionHead in src/config/model.rs
// All direction configuration is now handled through model_config.output_heads.direction

/// Direction classes (5-class system)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Dump = 0,     // Extreme down movement
    Down = 1,     // Significant down movement
    Sideways = 2, // Minimal movement
    Up = 3,       // Significant up movement
    Pump = 4,     // Extreme up movement
}

/// Generate direction targets for multiple horizons using DirectionHead configuration
pub fn generate_direction_targets(
    df: &DataFrame,
    horizons: &[String],
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

            // Calculate thresholds from DirectionHead with bandwidth sensitivity
            let base_threshold = model_cfg.base_threshold_factor * market_volatility;
            let adaptive_threshold = base_threshold / bandwidth_size; // Higher bandwidth = more sensitive (lower thresholds)
            let extreme_threshold = adaptive_threshold * model_cfg.extreme_multiplier;

            apply_direction_classification(
                &close_prices,
                horizon_steps,
                adaptive_threshold,
                extreme_threshold,
            )?
        } else {
            // Fallback to default configuration
            let default_threshold = 0.02; // 2%
            let extreme_threshold = default_threshold * 2.5; // 5%
            apply_direction_classification(
                &close_prices,
                horizon_steps,
                default_threshold,
                extreme_threshold,
            )?
        };

        targets.insert(horizon.clone(), direction_targets);
    }

    Ok(targets)
}

/// Apply direction classification using thresholds
fn apply_direction_classification(
    prices: &[f64],
    horizon_steps: usize,
    base_threshold: f64,
    extreme_threshold: f64,
) -> Result<Vec<i32>> {
    if prices.len() < horizon_steps + 1 {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient price data for direction classification".to_string(),
        ));
    }

    let mut targets = vec![-1; prices.len()];

    for i in 0..prices.len().saturating_sub(horizon_steps) {
        let current_price = prices[i];
        let future_price = prices[i + horizon_steps];
        let price_change = (future_price - current_price) / current_price;

        // 5-class system: DUMP(-2), DOWN(-1), SIDEWAYS(0), UP(1), PUMP(2)
        let target = if price_change <= -extreme_threshold {
            0 // DUMP
        } else if price_change <= -base_threshold {
            1 // DOWN
        } else if price_change >= extreme_threshold {
            4 // PUMP
        } else if price_change >= base_threshold {
            3 // UP
        } else {
            2 // SIDEWAYS
        };

        targets[i] = target;
    }

    Ok(targets)
}

/// Calculate market volatility for adaptive thresholds
fn calculate_market_volatility(prices: &[f64]) -> Result<f64> {
    if prices.len() < 2 {
        return Ok(0.02); // Default 2% volatility
    }

    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

    if returns.is_empty() {
        return Ok(0.02);
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    Ok(variance.sqrt().max(0.005)) // Minimum 0.5% volatility
}

/// Extract close prices from DataFrame
fn extract_close_prices(df: &DataFrame) -> Result<Vec<f64>> {
    let close_series = df.column("close").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!("Missing 'close' column: {}", e))
    })?;

    let close_prices: Vec<f64> = close_series
        .f64()
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to convert 'close' column to f64: {}",
                e
            ))
        })?
        .into_no_null_iter()
        .collect();

    if close_prices.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "No valid close prices found".to_string(),
        ));
    }

    Ok(close_prices)
}
