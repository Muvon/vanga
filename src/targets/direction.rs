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

    // ADAPTIVE CALIBRATION: Auto-calibrate slope sensitivity if not provided or if using default
    let should_calibrate = model_config
        .and_then(|c| c.slope_sensitivity)
        .map(|s| s >= 0.4) // Calibrate if using old default values
        .unwrap_or(true); // Calibrate if no config provided

    let calibrated_sensitivity = if should_calibrate {
        // Use first horizon for calibration
        let first_horizon_steps = parse_horizon_to_steps(&horizons[0])?;
        calibrate_slope_sensitivity(&close_prices, sequence_length, first_horizon_steps, 0.15)?
    } else {
        model_config
            .and_then(|c| c.slope_sensitivity)
            .unwrap_or(0.02)
    };

    log::info!(
        "🎯 Direction targets using slope_sensitivity: {:.6} (calibrated: {})",
        calibrated_sensitivity,
        should_calibrate
    );

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
                    // Create config with calibrated sensitivity
                    let calibrated_config = model_config.map(|c| DirectionHead {
                        enabled: c.enabled,
                        slope_sensitivity: Some(calibrated_sensitivity),
                        base_threshold: c.base_threshold,
                        extreme_multiplier: c.extreme_multiplier,
                    });

                    let target_class = classify_direction(
                        sequence_prices,
                        horizon_prices, // Now using horizon sequence, not single price
                        calibrated_config.as_ref(),
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

/// Calculate linear regression slope for trend analysis with sequence volatility normalization
///
/// Uses least squares method to find the best-fit line slope, normalized by sequence price volatility
/// to ensure consistent trend detection across different price levels and market conditions.
///
/// ## Normalization Strategy
/// - Uses sequence price standard deviation for volatility-based normalization
/// - Fallback to price mean for low-volatility periods (< 0.1% of mean)
/// - Ensures slope values are comparable across different price ranges and symbols
///
/// Returns slope normalized by sequence volatility (dimensionless trend strength)
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

    // ADAPTIVE NORMALIZATION: Use sequence price volatility for normalization
    let price_mean = sum_y / n;
    let price_variance = prices.iter().map(|x| (x - price_mean).powi(2)).sum::<f64>() / n;
    let price_std = price_variance.sqrt();

    // Normalize by volatility, fallback to mean for low-volatility periods
    let normalization_factor = if price_std > price_mean * 0.001 {
        price_std // Use volatility for normal market conditions
    } else {
        price_mean.max(1e-8) // Fallback to mean for extremely low volatility
    };

    let normalized_slope = slope / normalization_factor;

    log::trace!(
        "🎯 Slope Calculation: raw_slope={:.6}, price_mean={:.2}, price_std={:.4}, norm_factor={:.4}, normalized_slope={:.6}",
        slope, price_mean, price_std, normalization_factor, normalized_slope
    );

    Ok(normalized_slope)
}

/// Calculate adaptive slope sensitivity based on actual price data
///
/// This function analyzes the distribution of normalized slope differences in the data
/// to automatically determine appropriate slope_sensitivity thresholds that will
/// produce balanced class distributions.
///
/// # Algorithm
/// 1. Calculate normalized slopes for all sequences and horizons
/// 2. Compute slope acceleration distribution
/// 3. Use percentiles to set balanced thresholds
/// 4. Return calibrated slope_sensitivity value
pub fn calibrate_slope_sensitivity(
    close_prices: &[f64],
    sequence_length: usize,
    horizon_steps: usize,
    target_balance: f64, // Target percentage for extreme classes (e.g., 0.15 for 15%)
) -> Result<f64> {
    if close_prices.len() < sequence_length + horizon_steps + 10 {
        return Ok(0.02); // Default fallback for insufficient data
    }

    let mut accelerations = Vec::new();

    // Sample accelerations from the data
    for i in 0..(close_prices.len() - sequence_length - horizon_steps) {
        let sequence_prices = &close_prices[i..i + sequence_length];
        let horizon_prices =
            &close_prices[i + sequence_length..i + sequence_length + horizon_steps];

        if sequence_prices.len() >= 2 && horizon_prices.len() >= 2 {
            let seq_slope = calculate_linear_trend_slope(sequence_prices)?;
            let hor_slope = calculate_linear_trend_slope(horizon_prices)?;
            let acceleration = hor_slope - seq_slope;

            if acceleration.is_finite() {
                accelerations.push(acceleration.abs()); // Use absolute values for threshold calculation
            }
        }
    }

    if accelerations.is_empty() {
        return Ok(0.02); // Default fallback
    }

    // Sort accelerations to find percentiles
    accelerations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = accelerations.len();

    // Find the percentile that corresponds to target_balance for extreme classes
    // We want target_balance% in each extreme class, so (1.0 - 2*target_balance) in middle classes
    let extreme_percentile = 1.0 - target_balance;
    let extreme_idx = ((n as f64) * extreme_percentile) as usize;
    let extreme_threshold = accelerations[extreme_idx.min(n - 1)];

    // The slope_sensitivity should be set so that extreme_threshold becomes the extreme boundary
    // With extreme_multiplier = 2.0: extreme_boundary = slope_sensitivity * 2.0
    // So: slope_sensitivity = extreme_threshold / 2.0
    let calibrated_sensitivity = extreme_threshold / 2.0;

    // Ensure reasonable bounds
    let final_sensitivity = calibrated_sensitivity.clamp(0.001, 0.5);

    log::info!(
        "🎯 Calibrated slope_sensitivity: {:.6} (from {} samples, extreme_threshold: {:.6})",
        final_sensitivity,
        n,
        extreme_threshold
    );

    Ok(final_sensitivity)
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
