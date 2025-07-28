//! Direction target generation for cryptocurrency price movement prediction
//!
//! This module implements directional classification for price movements:
//! - 0: Down (significant decrease)
//! - 1: Sideways (minimal change)
//! - 2: Up (significant increase)

use crate::config::model::DirectionHead;
use crate::utils::error::Result;
use crate::utils::market_data::extract_close_prices;
use crate::utils::parser::parse_horizon_to_steps;
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

/// Generate direction targets for multiple horizons using sequence-to-horizon momentum analysis
///
/// FLOW:
/// 1. Extract close prices from DataFrame
/// 2. For each sequence position:
///    - Get INPUT sequence prices (for momentum baseline)
///    - Get HORIZON sequence prices (from sequence end to target horizon)
///    - Classify direction using sequence-to-horizon comparison
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
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            // Check boundaries - need both sequence and horizon data
            if target_end_idx <= close_prices.len() && sequence_end_idx <= close_prices.len() {
                // Get INPUT sequence prices (for momentum baseline)
                let sequence_prices = &close_prices[seq_idx..sequence_end_idx];

                // Get HORIZON sequence prices (from sequence end to target horizon)
                let horizon_prices = &close_prices[sequence_end_idx..target_end_idx];

                // Only classify if we have enough horizon data for momentum calculation
                if horizon_prices.len() >= 2 {
                    let target_class = classify_direction(
                        sequence_prices,
                        horizon_prices, // Now using horizon sequence, not single price
                        model_config,
                    )?;

                    horizon_targets[seq_position] = target_class;
                }
            }
        }

        log_direction_distribution(&horizon_targets, horizon);
        targets.insert(horizon.clone(), horizon_targets);
    }

    Ok(targets)
}

/// Classify direction using sequence-to-horizon momentum calculation (same pattern as volatility)
///
/// FLOW:
/// 1. Get momentum baseline from INPUT sequence (training window)
/// 2. Get momentum from sequence END to target horizon (prediction period)
/// 3. Compare horizon momentum vs sequence baseline (same logic as volatility)
/// 4. Apply bandwidth sensitivity and classify into 5 classes
fn classify_direction(
    sequence_prices: &[f64], // Input sequence for baseline
    horizon_prices: &[f64],  // From sequence end to target horizon
    model_config: Option<&DirectionHead>,
) -> Result<i32> {
    // Step 1: Get momentum baseline from INPUT sequence (like volatility's sequence baseline)
    let sequence_baseline = get_sequence_momentum_baseline(sequence_prices)?;

    // Step 2: Get momentum from sequence END to target horizon (the actual prediction period)
    let horizon_momentum = get_horizon_momentum(horizon_prices)?;

    // Step 3: Same logic as volatility classification
    let bandwidth_size = model_config.and_then(|c| c.bandwidth_size).unwrap_or(1.0);

    // Same factors as volatility (hardcoded values)
    let base_threshold_factor = 0.5; // 50% of baseline
    let extreme_multiplier = 2.0; // 2x for extreme

    // Calculate thresholds (identical to volatility)
    let base_threshold = base_threshold_factor * sequence_baseline;
    let adaptive_threshold = base_threshold / bandwidth_size;
    let extreme_threshold = adaptive_threshold * extreme_multiplier;

    // Step 4: Compare horizon momentum to sequence baseline (like momentum ratio in volatility)
    let momentum_ratio = horizon_momentum / sequence_baseline;

    // 5-class system (same logic as volatility)
    let class = if momentum_ratio <= (1.0 - extreme_threshold) {
        0 // DUMP: Much below sequence baseline
    } else if momentum_ratio <= (1.0 - adaptive_threshold) {
        1 // DOWN: Below sequence baseline
    } else if momentum_ratio >= (1.0 + extreme_threshold) {
        4 // PUMP: Much above sequence baseline
    } else if momentum_ratio >= (1.0 + adaptive_threshold) {
        3 // UP: Above sequence baseline
    } else {
        2 // SIDEWAYS: Around sequence baseline
    };

    Ok(class)
}

/// Log direction class distribution with momentum-based analysis
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
            "📊 Direction Momentum Analysis [{}]: No valid targets found",
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
        "📊 Direction Momentum Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}]",
        horizon,
        valid_targets,
        imbalance_ratio,
        class_percentages.join(", ")
    );
}

/// Get momentum baseline from sequence prices (same pattern as volatility's ATR baseline)
fn get_sequence_momentum_baseline(sequence_prices: &[f64]) -> Result<f64> {
    if sequence_prices.len() < 2 {
        return Ok(0.02); // Minimal fallback
    }

    let max_price = sequence_prices
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_price = sequence_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let avg_price = sequence_prices.iter().sum::<f64>() / sequence_prices.len() as f64;

    if avg_price <= 0.0 {
        return Ok(0.02); // Fallback for invalid prices
    }

    // Baseline = (max - min) / avg (normalized range - symbol agnostic)
    Ok((max_price - min_price) / avg_price)
}

/// Get horizon momentum (same calculation as baseline)
fn get_horizon_momentum(horizon_prices: &[f64]) -> Result<f64> {
    // Same calculation as baseline (without bandwidth_size)
    get_sequence_momentum_baseline(horizon_prices)
}
