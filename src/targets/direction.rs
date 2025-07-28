//! Direction target generation for cryptocurrency price movement prediction
//!
//! This module implements momentum-based directional classification using sequence-to-horizon analysis:
//! - 0: DUMP (extreme downward momentum - much below sequence baseline)
//! - 1: DOWN (moderate downward momentum - below sequence baseline)
//! - 2: SIDEWAYS (minimal momentum - around sequence baseline)
//! - 3: UP (moderate upward momentum - above sequence baseline)
//! - 4: PUMP (extreme upward momentum - much above sequence baseline)
//!
//! **Key Features:**
//! - Uses (max-min)/avg momentum calculation from input sequence as baseline
//! - Compares horizon period momentum against sequence baseline
//! - Adaptive bandwidth_size for symbol-specific sensitivity
//! - Symbol-agnostic through percentage-based calculations
//! - Same architecture pattern as volatility.rs for consistency

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

/// Classify direction using adaptive percentile-based approach
///
/// FLOW:
/// 1. Calculate price changes within the input sequence
/// 2. Build percentile distribution from sequence price changes
/// 3. Use bandwidth_size to define middle class (SIDEWAYS) boundaries
/// 4. Apply extreme_multiplier for DUMP/PUMP classification
/// 5. Compare horizon price change against adaptive percentile thresholds
fn classify_direction(
    sequence_prices: &[f64], // Input sequence for adaptive baseline
    horizon_prices: &[f64],  // From sequence end to target horizon
    model_config: Option<&DirectionHead>,
) -> Result<i32> {
    if sequence_prices.len() < 2 || horizon_prices.is_empty() {
        return Ok(2); // Default to SIDEWAYS for insufficient data
    }

    // Step 1: Calculate actual price change from sequence to horizon
    let sequence_avg = sequence_prices.iter().sum::<f64>() / sequence_prices.len() as f64;
    let horizon_avg = horizon_prices.iter().sum::<f64>() / horizon_prices.len() as f64;

    if sequence_avg <= 0.0 {
        return Ok(2); // Default to SIDEWAYS for invalid data
    }

    let actual_price_change = horizon_avg - sequence_avg;

    // Step 2: Build adaptive baseline from sequence price changes
    let sequence_changes: Vec<f64> = sequence_prices.windows(2).map(|w| w[1] - w[0]).collect();

    if sequence_changes.is_empty() {
        return Ok(2); // Default to SIDEWAYS
    }

    // Step 3: Calculate adaptive percentile thresholds
    let adaptive_thresholds =
        calculate_adaptive_percentile_thresholds(&sequence_changes, model_config)?;

    // Step 4: Classify using adaptive thresholds
    let class = if actual_price_change <= adaptive_thresholds.extreme_down {
        0 // DUMP: Much below sequence distribution
    } else if actual_price_change <= adaptive_thresholds.moderate_down {
        1 // DOWN: Below sequence distribution
    } else if actual_price_change >= adaptive_thresholds.extreme_up {
        4 // PUMP: Much above sequence distribution
    } else if actual_price_change >= adaptive_thresholds.moderate_up {
        3 // UP: Above sequence distribution
    } else {
        2 // SIDEWAYS: Within sequence distribution
    };

    // Debug logging with actual values
    log::debug!(
        "🎯 Adaptive Direction: price_change={:.6}, thresholds=[down:{:.6}, moderate_down:{:.6}, moderate_up:{:.6}, up:{:.6}]",
        actual_price_change,
        adaptive_thresholds.extreme_down,
        adaptive_thresholds.moderate_down,
        adaptive_thresholds.moderate_up,
        adaptive_thresholds.extreme_up
    );

    Ok(class)
}

/// Adaptive percentile thresholds for direction classification
#[derive(Debug)]
struct AdaptiveDirectionThresholds {
    extreme_down: f64,  // DUMP threshold
    moderate_down: f64, // DOWN threshold
    moderate_up: f64,   // UP threshold
    extreme_up: f64,    // PUMP threshold
}

/// Calculate adaptive percentile thresholds from sequence price changes
///
/// LOGIC:
/// - bandwidth_size controls the middle class (SIDEWAYS) definition
/// - Smaller bandwidth_size = narrower SIDEWAYS = more UP/DOWN classifications
/// - Larger bandwidth_size = wider SIDEWAYS = more SIDEWAYS classifications
/// - extreme_multiplier extends the thresholds for DUMP/PUMP classes
fn calculate_adaptive_percentile_thresholds(
    sequence_changes: &[f64],
    model_config: Option<&DirectionHead>,
) -> Result<AdaptiveDirectionThresholds> {
    if sequence_changes.is_empty() {
        return Ok(AdaptiveDirectionThresholds {
            extreme_down: -0.01,
            moderate_down: -0.005,
            moderate_up: 0.005,
            extreme_up: 0.01,
        });
    }

    // Get config parameters
    let bandwidth_size = model_config.and_then(|c| c.bandwidth_size).unwrap_or(1.0);
    let extreme_multiplier = model_config
        .and_then(|c| c.extreme_multiplier)
        .unwrap_or(2.0);

    // Sort sequence changes for percentile calculation
    let mut sorted_changes = sequence_changes.to_vec();
    sorted_changes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate base percentiles (bandwidth_size controls middle class width)
    // bandwidth_size = 1.0 means 40% middle class (20th-80th percentiles)
    // bandwidth_size = 0.5 means 20% middle class (40th-60th percentiles)
    // bandwidth_size = 2.0 means 80% middle class (10th-90th percentiles)
    let base_percentile = bandwidth_size * 0.3; // FIXED: Direct scaling
    let base_percentile = base_percentile.clamp(0.05, 0.45); // Clamp to reasonable range

    // Calculate percentile positions
    let len = sorted_changes.len();
    let down_idx = ((base_percentile * len as f64) as usize).min(len - 1);
    let up_idx = (((1.0 - base_percentile) * len as f64) as usize).min(len - 1);

    // Get base thresholds from percentiles
    let moderate_down = sorted_changes[down_idx];
    let moderate_up = sorted_changes[up_idx];

    // FIXED: Extreme thresholds should be FURTHER from zero
    let extreme_down = moderate_down * extreme_multiplier; // FIXED: More negative
    let extreme_up = moderate_up * extreme_multiplier; // FIXED: More positive

    let thresholds = AdaptiveDirectionThresholds {
        extreme_down,
        moderate_down,
        moderate_up,
        extreme_up,
    };

    log::debug!(
        "📊 FIXED Thresholds: bandwidth_size={}, middle_class={:.1}%, percentiles=[{:.1}%, {:.1}%], thresholds=[{:.6}, {:.6}, {:.6}, {:.6}]",
        bandwidth_size, (1.0 - 2.0 * base_percentile) * 100.0, base_percentile * 100.0, (1.0 - base_percentile) * 100.0,
        extreme_down, moderate_down, moderate_up, extreme_up
    );

    Ok(thresholds)
}

/// Log direction class distribution with adaptive percentile analysis
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
            "📊 Adaptive Direction Analysis [{}]: No valid targets found",
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
        "📊 Adaptive Direction Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}]",
        horizon,
        valid_targets,
        imbalance_ratio,
        class_percentages.join(", ")
    );
}
