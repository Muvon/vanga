//! Direction target generation for cryptocurrency price movement prediction
//!
//! This module implements trend acceleration-based directional classification:
//! - 0: DUMP (strong trend deceleration/reversal - negative acceleration)
//! - 1: DOWN (moderate trend deceleration - slight negative acceleration)
//! - 2: SIDEWAYS (trend continuation - minimal acceleration change)
//! - 3: UP (moderate trend acceleration - slight positive acceleration)
//! - 4: PUMP (strong trend acceleration - positive acceleration)
//!
//! ## Mathematical Approach
//!
//! **Trend Acceleration Detection:**
//! 1. Calculate linear regression slope for sequence prices (recent trend)
//! 2. Calculate linear regression slope for horizon prices (future trend)
//! 3. Compute trend acceleration: `horizon_slope - sequence_slope`
//! 4. Classify based on acceleration magnitude using slope_sensitivity thresholds
//!
//! **Key Features:**
//! - Uses linear regression slopes to measure trend strength
//! - Compares sequence trend vs horizon trend (acceleration/deceleration)
//! - Uses absolute slope differences (no price normalization needed)
//! - Detects trend momentum changes rather than just price changes
//! - Complementary to price_levels (range) and volatility (risk) targets
//!
//! ## Configuration Parameters
//!
//! ### `slope_sensitivity` (default: 0.4)
//! Controls the sensitivity of slope acceleration thresholds for trend momentum detection.
//! Higher values = less sensitive (wider thresholds), lower values = more sensitive (tighter thresholds).
//!
//! **Threshold Calculation:**
//! ```text
//! half_sensitivity = slope_sensitivity / 2.0
//! extreme_sensitivity = slope_sensitivity * extreme_multiplier
//!
//! DUMP:     acceleration <= -extreme_sensitivity  (e.g., <= -8.0)
//! DOWN:     -extreme_sensitivity < acceleration <= -half_sensitivity  (e.g., -8.0 to -2.0)
//! SIDEWAYS: -half_sensitivity < acceleration <= half_sensitivity  (e.g., -2.0 to +2.0)
//! UP:       half_sensitivity < acceleration <= extreme_sensitivity  (e.g., +2.0 to +8.0)
//! PUMP:     acceleration > extreme_sensitivity  (e.g., > +8.0)
//! ```
//!
//! **Recommended Values:**
//! - **Conservative (0.2-0.3)**: More sensitive, detects subtle momentum changes
//! - **Standard (0.4-0.6)**: Balanced sensitivity for most crypto pairs
//! - **Aggressive (0.8-1.2)**: Less sensitive, only major momentum shifts
//!
//! ### `extreme_multiplier` (default: 2.0)
//! Multiplier for extreme class boundaries (DUMP/PUMP vs DOWN/UP).
//! Higher values = fewer extreme classifications, more moderate classifications.
//!
//! ## Usage Examples
//!
//! ```rust
//! use crate::config::model::DirectionHead;
//!
//! // Conservative: Detects subtle momentum changes
//! let conservative_config = DirectionHead {
//!     enabled: true,
//!     slope_sensitivity: Some(0.3),
//!     base_threshold: Some(0.12),
//!     extreme_multiplier: Some(2.0),
//! };
//!
//! // Standard: Balanced for most crypto trading
//! let standard_config = DirectionHead {
//!     enabled: true,
//!     slope_sensitivity: Some(0.4),
//!     base_threshold: Some(0.12),
//!     extreme_multiplier: Some(2.0),
//! };
//!
//! // Aggressive: Only major momentum shifts
//! let aggressive_config = DirectionHead {
//!     enabled: true,
//!     slope_sensitivity: Some(0.8),
//!     base_threshold: Some(0.12),
//!     extreme_multiplier: Some(1.5),
//! };
//! ```
//!
//! ## Target Differentiation Strategy
//!
//! **Direction vs Other Targets:**
//! - **DIRECTION**: "How is trend momentum changing?" (acceleration/deceleration)
//! - **PRICE_LEVELS**: "Where will price be?" (range/breakout analysis)
//! - **VOLATILITY**: "How volatile will it be?" (risk assessment)
//!
//! Each target serves a different purpose in the multi-target prediction system,
//! providing complementary information for comprehensive market analysis.

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

/// Generate direction targets for multiple horizons using trend acceleration analysis
///
/// FLOW:
/// 1. Extract close prices from DataFrame
/// 2. For each sequence position:
///    - Get INPUT sequence prices (for trend baseline calculation)
///    - Get HORIZON sequence prices (for trend comparison)
///    - Calculate linear regression slopes for both periods
///    - Classify based on trend acceleration (slope change)
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

/// Classify direction using trend acceleration approach
///
/// This is the main classification function that determines the directional class
/// based on the acceleration/deceleration of price trends between sequence and horizon periods.
///
/// ## Algorithm
/// 1. **Trend Analysis**: Calculate linear regression slopes for both periods
///    - `sequence_slope`: Trend strength in the input sequence (recent history)
///    - `horizon_slope`: Trend strength in the prediction horizon (future)
/// 2. **Acceleration Calculation**: `trend_acceleration = horizon_slope - sequence_slope`
/// 3. **Classification**: Compare acceleration against slope_sensitivity-based thresholds
///
/// ## Parameters
/// - `sequence_prices`: Input sequence prices for establishing trend baseline
/// - `horizon_prices`: Prices from sequence end to target horizon
/// - `model_config`: Optional DirectionHead configuration (uses defaults if None)
///
/// ## Returns
/// Direction class as i32:
/// - 0: DUMP (strong deceleration/reversal)
/// - 1: DOWN (moderate deceleration)
/// - 2: SIDEWAYS (trend continuation)
/// - 3: UP (moderate acceleration)
/// - 4: PUMP (strong acceleration)
///
/// ## Configuration
/// Uses `slope_sensitivity` from model_config to determine threshold sensitivity:
/// - Default: 0.4 (balanced sensitivity)
/// - Lower values: More sensitive to momentum changes
/// - Higher values: Less sensitive, only major shifts
///
/// ## Example
/// ```rust
/// let sequence = vec![100.0, 101.0, 102.0, 103.0, 104.0]; // +1.0/period trend
/// let horizon = vec![104.0, 106.0, 108.0, 110.0, 112.0];  // +2.0/period trend
/// // Acceleration = 2.0 - 1.0 = 1.0 (positive acceleration)
/// // With default config: likely UP or PUMP class
/// ```
pub fn classify_direction(
    sequence_prices: &[f64], // Input sequence for trend baseline
    horizon_prices: &[f64],  // From sequence end to target horizon
    model_config: Option<&DirectionHead>,
) -> Result<i32> {
    if sequence_prices.len() < 2 || horizon_prices.len() < 2 {
        return Ok(2); // Default to SIDEWAYS for insufficient data
    }

    // Get config parameters
    let slope_sensitivity = model_config
        .and_then(|c| c.slope_sensitivity)
        .unwrap_or(0.4);
    let extreme_multiplier = model_config
        .and_then(|c| c.extreme_multiplier)
        .unwrap_or(2.0);

    // Calculate trend acceleration thresholds
    let thresholds =
        calculate_trend_acceleration_thresholds(slope_sensitivity, extreme_multiplier)?;

    // Classify using trend acceleration approach
    classify_direction_trend_acceleration(sequence_prices, horizon_prices, &thresholds)
}

/// Trend acceleration thresholds for direction classification
///
/// This struct defines the boundary values used to classify trend acceleration
/// into the 5-class direction system. The thresholds are calculated based on
/// `slope_sensitivity` and `extreme_multiplier` parameters.
///
/// ## Threshold Structure
/// ```text
/// DUMP:     acceleration <= dump_max (most negative)
/// DOWN:     dump_max < acceleration <= down_max (moderate negative)
/// SIDEWAYS: down_max < acceleration <= sideways_max (minimal change)
/// UP:       sideways_max < acceleration <= up_max (moderate positive)
/// PUMP:     acceleration > up_max (most positive)
/// ```
///
/// ## Field Meanings
/// - `dump_max`: Maximum acceleration for DUMP class (strong deceleration)
/// - `down_max`: Maximum acceleration for DOWN class (moderate deceleration)
/// - `sideways_max`: Maximum acceleration for SIDEWAYS class (trend continuation)
/// - `up_max`: Maximum acceleration for UP class (moderate acceleration)
/// - Values above `up_max` are classified as PUMP (strong acceleration)
#[derive(Debug)]
pub struct TrendAccelerationThresholds {
    pub dump_max: f64,     // DUMP threshold (most negative acceleration)
    pub down_max: f64,     // DOWN threshold (moderate negative acceleration)
    pub sideways_max: f64, // SIDEWAYS threshold (minimal acceleration)
    pub up_max: f64,       // UP threshold (moderate positive acceleration)
                           // Above up_max = PUMP (strong positive acceleration)
}

/// Calculate linear regression slope for trend analysis
///
/// Uses least squares method to find the best-fit line slope
/// Returns slope normalized per time unit (price change per period)
pub fn calculate_linear_trend_slope(prices: &[f64]) -> Result<f64> {
    if prices.len() < 2 {
        return Ok(0.0); // No trend for insufficient data
    }

    let n = prices.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;

    // Calculate sums for least squares regression
    for (i, &price) in prices.iter().enumerate() {
        let x = i as f64;
        sum_x += x;
        sum_y += price;
        sum_xy += x * price;
        sum_x2 += x * x;
    }

    // Calculate slope using least squares formula: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    let denominator = n * sum_x2 - sum_x * sum_x;

    if denominator.abs() < 1e-10 {
        return Ok(0.0); // Avoid division by zero
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    Ok(slope)
}

/// Calculate trend acceleration thresholds for direction classification
///
/// This function computes the threshold boundaries used to classify trend acceleration
/// into the 5-class direction system (DUMP, DOWN, SIDEWAYS, UP, PUMP).
///
/// ## Parameters
/// - `slope_sensitivity`: Controls threshold sensitivity (default: 0.4)
///   - Lower values = more sensitive (tighter thresholds)
///   - Higher values = less sensitive (wider thresholds)
/// - `extreme_multiplier`: Multiplier for extreme boundaries (default: 2.0)
///   - Controls the ratio between moderate and extreme classifications
///
/// ## Threshold Logic
/// ```text
/// half_sensitivity = slope_sensitivity / 2.0
/// extreme_sensitivity = slope_sensitivity * extreme_multiplier
///
/// Classification boundaries:
/// DUMP:     acceleration <= -extreme_sensitivity
/// DOWN:     -extreme_sensitivity < acceleration <= -half_sensitivity
/// SIDEWAYS: -half_sensitivity < acceleration <= +half_sensitivity
/// UP:       +half_sensitivity < acceleration <= +extreme_sensitivity
/// PUMP:     acceleration > +extreme_sensitivity
/// ```
///
/// ## Example
/// With slope_sensitivity=4.0 and extreme_multiplier=2.0:
/// - DUMP: acceleration <= -8.0 (strong deceleration)
/// - DOWN: -8.0 < acceleration <= -2.0 (moderate deceleration)
/// - SIDEWAYS: -2.0 < acceleration <= +2.0 (trend continuation)
/// - UP: +2.0 < acceleration <= +8.0 (moderate acceleration)
/// - PUMP: acceleration > +8.0 (strong acceleration)
pub fn calculate_trend_acceleration_thresholds(
    slope_sensitivity: f64,
    extreme_multiplier: f64,
) -> Result<TrendAccelerationThresholds> {
    // Use slope_sensitivity directly - it should be configured appropriately for slope differences
    // No magic scaling factor needed - let the user configure slope_sensitivity properly
    let half_sensitivity = slope_sensitivity / 2.0;
    let extreme_sensitivity = slope_sensitivity * extreme_multiplier;

    let thresholds = TrendAccelerationThresholds {
        dump_max: -extreme_sensitivity, // Most negative acceleration
        down_max: -half_sensitivity,    // Moderate negative acceleration
        sideways_max: half_sensitivity, // Minimal acceleration (around 0)
        up_max: extreme_sensitivity,    // Moderate positive acceleration
                                        // Above up_max = PUMP (strong positive acceleration)
    };

    log::debug!(
        "🎯 Trend Acceleration Thresholds: slope_sensitivity={}, extreme_factor={}, thresholds=[{:.6}, {:.6}, {:.6}, {:.6}]",
        slope_sensitivity, extreme_multiplier,
        thresholds.dump_max, thresholds.down_max, thresholds.sideways_max, thresholds.up_max
    );

    Ok(thresholds)
}

/// Classify direction using trend acceleration approach
fn classify_direction_trend_acceleration(
    sequence_prices: &[f64],
    horizon_prices: &[f64],
    thresholds: &TrendAccelerationThresholds,
) -> Result<i32> {
    // Step 1: Calculate linear regression slopes
    let sequence_trend = calculate_linear_trend_slope(sequence_prices)?;
    let horizon_trend = calculate_linear_trend_slope(horizon_prices)?;

    // Step 2: Calculate trend acceleration (change in slope) - NO NORMALIZATION
    let trend_acceleration = horizon_trend - sequence_trend;

    // Step 3: Classify using absolute acceleration thresholds
    let class = if trend_acceleration <= thresholds.dump_max {
        0 // DUMP: Strong deceleration/reversal
    } else if trend_acceleration <= thresholds.down_max {
        1 // DOWN: Moderate deceleration
    } else if trend_acceleration <= thresholds.sideways_max {
        2 // SIDEWAYS: Trend continuation
    } else if trend_acceleration <= thresholds.up_max {
        3 // UP: Moderate acceleration
    } else {
        4 // PUMP: Strong acceleration
    };

    log::debug!(
        "🎯 Trend Acceleration: seq_slope={:.6}, horizon_slope={:.6}, acceleration={:.6} → class={} (thresholds: [{:.6}, {:.6}, {:.6}, {:.6}])",
        sequence_trend, horizon_trend, trend_acceleration, class,
        thresholds.dump_max, thresholds.down_max, thresholds.sideways_max, thresholds.up_max
    );

    Ok(class)
}

/// Log direction class distribution with trend acceleration analysis
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
            "📊 Trend Acceleration Direction Analysis [{}]: No valid targets found",
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
        "📊 Trend Acceleration Direction Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}]",
        horizon,
        valid_targets,
        imbalance_ratio,
        class_percentages.join(", ")
    );
}
