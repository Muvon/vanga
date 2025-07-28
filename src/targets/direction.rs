//! Direction target generation for cryptocurrency price movement prediction
//!
//! This module implements directional classification for price movements:
//! - 0: Down (significant decrease)
//! - 1: Sideways (minimal change)
//! - 2: Up (significant increase)

use crate::config::model::DirectionHead;
use crate::utils::error::Result;
use crate::utils::parser::parse_horizon_to_steps;
use crate::utils::market_data::extract_close_prices;
use polars::prelude::*;
use std::collections::HashMap;

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
    sequence_indices: &[usize],
    sequence_length: usize,
) -> Result<HashMap<String, Vec<i32>>> {
    let close_prices = extract_close_prices(df)?;
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let mut horizon_targets = vec![-1; sequence_indices.len()];

        for (seq_position, &seq_idx) in sequence_indices.iter().enumerate() {
            let target_idx = seq_idx + sequence_length + horizon_steps;

            if target_idx < close_prices.len() && seq_idx + sequence_length <= close_prices.len() {
                let current_price = close_prices[seq_idx + sequence_length - 1];
                let target_price = close_prices[target_idx];

                let target_class = calculate_direction_class(
                    current_price,
                    target_price,
                    &close_prices,
                    model_config,
                )?;

                horizon_targets[seq_position] = target_class;
            }
        }

        log_direction_distribution(&horizon_targets, horizon);
        targets.insert(horizon.clone(), horizon_targets);
    }

    Ok(targets)
}

/// Calculate direction class for a single price pair
fn calculate_direction_class(
    current_price: f64,
    target_price: f64,
    all_prices: &[f64],
    model_config: Option<&DirectionHead>,
) -> Result<i32> {
    let price_change = (target_price - current_price) / current_price;

    let (base_threshold, extreme_threshold) = if let Some(model_cfg) = model_config {
        let market_volatility = calculate_market_volatility(all_prices)?;
        let bandwidth_size = model_cfg.bandwidth_size.unwrap_or(1.0);
        let base_threshold = model_cfg.base_threshold_factor * market_volatility;
        let adaptive_threshold = base_threshold / bandwidth_size;
        let extreme_threshold = adaptive_threshold * model_cfg.extreme_multiplier;
        (adaptive_threshold, extreme_threshold)
    } else {
        let market_volatility = calculate_market_volatility(all_prices)?;
        let default_threshold = 0.02 * market_volatility;
        let extreme_threshold = default_threshold * 2.0;
        (default_threshold, extreme_threshold)
    };

    // 5-class system: DUMP(0), DOWN(1), SIDEWAYS(2), UP(3), PUMP(4)
    let class = if price_change <= -extreme_threshold {
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

    Ok(class)
}

/// Log direction class distribution
fn log_direction_distribution(targets: &[i32], horizon: &str) {
    let class_names = ["DUMP", "DOWN", "SIDEWAYS", "UP", "PUMP"];
    let mut class_counts = [0usize; 5];
    let mut valid_targets = 0;

    for &target in targets {
        if (0..5).contains(&target) {
            class_counts[target as usize] += 1;
            valid_targets += 1;
        }
    }

    if valid_targets == 0 {
        log::warn!(
            "📊 Direction Analysis [{}]: No valid targets found",
            horizon
        );
        return;
    }

    let total_samples = valid_targets as f64;
    let class_percentages: Vec<String> = class_counts
        .iter()
        .enumerate()
        .map(|(i, &count)| {
            let percentage = (count as f64 / total_samples) * 100.0;
            format!("{}:{:.1}%", class_names[i], percentage)
        })
        .collect();

    let min_class_size = class_counts.iter().filter(|&&c| c > 0).min().unwrap_or(&0);
    let max_class_size = class_counts.iter().max().unwrap_or(&0);
    let imbalance_ratio = if *min_class_size > 0 {
        *max_class_size as f64 / *min_class_size as f64
    } else {
        f64::INFINITY
    };

    log::info!(
        "📊 Direction Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}]",
        horizon,
        valid_targets,
        imbalance_ratio,
        class_percentages.join(", ")
    );
}

/// Calculate market volatility for adaptive thresholds
fn calculate_market_volatility(prices: &[f64]) -> Result<f64> {
    if prices.len() < 2 {
        return Ok(0.02); // Default 2% volatility
    }

    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    Ok(variance.sqrt().max(0.005)) // Minimum 0.5% volatility
}

