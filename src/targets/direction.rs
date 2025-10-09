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

use crate::data::structures::MarketDataRow;
use crate::targets::TargetResult;
use crate::utils::error::Result;
use crate::utils::market_data::extract_ohlcv_data;
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

/// Generate direction targets with per-horizon calibrated parameters
///
/// When calibrated_params is provided, uses the pre-calibrated parameters for each horizon
/// for consistent target generation between training and prediction.
pub fn generate_direction_targets_with_calibrated_params(
    df: &DataFrame,
    horizons: &[String],
    sequence_indices: &[usize],
    sequence_length: usize,
    calibrated_params: &std::collections::HashMap<
        String,
        crate::targets::calibration::DirectionParams,
    >,
) -> Result<TargetResult> {
    // CRITICAL FIX: Use OHLCV data instead of just close prices for richer momentum analysis
    let ohlcv_data = extract_ohlcv_data(df)?;
    let mut targets = HashMap::new();
    let mut strengths = HashMap::new();

    log::info!("🎯 Generating direction targets with per-horizon calibrated parameters");

    for horizon in horizons {
        // Get parameters for this specific horizon
        let params = calibrated_params.get(horizon).ok_or_else(|| {
            crate::utils::error::VangaError::ConfigError(format!(
                "No calibrated direction parameters found for horizon: {}",
                horizon
            ))
        })?;

        log::debug!(
            "  Horizon {}: sensitivity={:.6}, extreme_mult={:.2}",
            horizon,
            params.sensitivity,
            params.extreme_multiplier
        );
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let mut horizon_targets = vec![-1; sequence_indices.len()];
        let mut horizon_strengths = vec![0.5; sequence_indices.len()];

        log::debug!(
            "Processing horizon {} with {} steps",
            horizon,
            horizon_steps
        );

        for (seq_position, &seq_idx) in sequence_indices.iter().enumerate() {
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            // Check boundaries - need both sequence and horizon data
            if target_end_idx <= ohlcv_data.len() && sequence_end_idx <= ohlcv_data.len() {
                // Get INPUT sequence OHLCV (for momentum baseline)
                let sequence_ohlcv = &ohlcv_data[seq_idx..sequence_end_idx];

                // Get HORIZON sequence OHLCV (from sequence end to target horizon)
                let horizon_ohlcv = &ohlcv_data[sequence_end_idx..target_end_idx];

                log::debug!(
                    "Sequence {}: seq_idx={}, seq_end={}, target_end={}, seq_len={}, hor_len={}",
                    seq_position,
                    seq_idx,
                    sequence_end_idx,
                    target_end_idx,
                    sequence_ohlcv.len(),
                    horizon_ohlcv.len()
                );

                // Only classify if we have enough horizon data for momentum calculation
                if horizon_ohlcv.len() >= 2 {
                    // Use per-horizon calibrated parameters with OHLCV data
                    let (target_class, strength) = classify_direction_with_calibrated_params(
                        sequence_ohlcv,
                        horizon_ohlcv,
                        params,
                    )?;

                    horizon_targets[seq_position] = target_class;
                    horizon_strengths[seq_position] = strength;
                }
            }
        }

        log_direction_distribution(&horizon_targets, horizon);
        targets.insert(horizon.clone(), horizon_targets);
        strengths.insert(horizon.clone(), horizon_strengths);
    }

    Ok((targets, strengths))
}

/// Classify direction using OHLCV data for momentum analysis
///
/// ## Algorithm
/// 1. **Extract OHLC4 prices**: Use (O+H+L+C)/4 for better price representation
/// 2. **Sequence Momentum**: Calculate overall trend momentum in the sequence
/// 3. **Horizon Momentum**: Calculate overall trend momentum in the horizon
/// 4. **Momentum Change**: Measure the change in momentum (acceleration/deceleration)
/// 5. **Adaptive Thresholds**: Set thresholds based on sequence trend consistency
/// 6. **Classification**: Classify based on momentum change magnitude and direction
pub fn classify_direction_with_calibrated_params(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::DirectionParams,
) -> Result<(i32, f64)> {
    if sequence_ohlcv.len() < 2 || horizon_ohlcv.len() < 2 {
        return Ok((2, 0.5)); // Default to SIDEWAYS with neutral strength for insufficient data
    }

    // Extract OHLC4 prices for richer representation
    let sequence_prices: Vec<f64> = sequence_ohlcv
        .iter()
        .map(|row| (row.open + row.high + row.low + row.close) / 4.0)
        .collect();

    let horizon_prices: Vec<f64> = horizon_ohlcv
        .iter()
        .map(|row| (row.open + row.high + row.low + row.close) / 4.0)
        .collect();

    // Step 1: Calculate momentum change between sequence and horizon
    let (sequence_momentum, horizon_momentum, momentum_change) =
        calculate_directional_momentum_change(&sequence_prices, &horizon_prices)?;

    // Step 2: Calculate sequence trend consistency for adaptive thresholds
    let trend_consistency = calculate_sequence_trend_consistency(&sequence_prices)?;

    // Step 3: Set adaptive thresholds based on trend consistency
    let base_multiplier = calibrated_params.base_multiplier;
    let extreme_multiplier = calibrated_params.extreme_multiplier;

    let base_threshold = trend_consistency * calibrated_params.sensitivity * base_multiplier;
    let extreme_threshold = base_threshold * extreme_multiplier;

    // Use calibrated minimum thresholds
    let min_base = calibrated_params.min_base_threshold;
    let min_extreme = calibrated_params.min_extreme_threshold;

    let final_base_threshold = base_threshold.max(min_base);
    let final_extreme_threshold = extreme_threshold.max(min_extreme);

    // Step 4: Classify based on momentum change magnitude and direction
    let class = if momentum_change <= -final_extreme_threshold {
        0 // DUMP
    } else if momentum_change <= -final_base_threshold {
        1 // DOWN
    } else if momentum_change.abs() <= final_base_threshold {
        2 // SIDEWAYS
    } else if momentum_change <= final_extreme_threshold {
        3 // UP
    } else {
        4 // PUMP
    };

    // Step 5: Calculate classification strength based on distance from boundaries
    let strength = calculate_direction_strength(
        momentum_change,
        final_base_threshold,
        final_extreme_threshold,
        class,
    );

    log::debug!(
        "🎯 Direction: seq_mom={:.6}, hor_mom={:.6}, change={:.6}, consistency={:.6}, base={:.6}, extreme={:.6} → class={} ({}) strength={:.3}",
        sequence_momentum, horizon_momentum, momentum_change, trend_consistency, final_base_threshold, final_extreme_threshold, class,
        ["DUMP", "DOWN", "SIDEWAYS", "UP", "PUMP"][class as usize], strength
    );

    Ok((class, strength))
}

/// Calculate raw linear regression slope without normalization
///
/// Uses least squares method to find the best-fit line slope in raw price units per time step.
/// This is used internally for momentum calculations where we need raw slope values.
///
/// Returns raw slope in price units per time step
pub fn calculate_raw_linear_slope(prices: &[f64]) -> Result<f64> {
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

/// Calculate sequence trend consistency for adaptive threshold setting
///
/// Measures how consistent the trend is within the sequence to set appropriate
/// thresholds for detecting meaningful momentum changes. More volatile sequences
/// need larger thresholds to filter out noise.
fn calculate_sequence_trend_consistency(prices: &[f64]) -> Result<f64> {
    if prices.len() < 3 {
        return Ok(0.01); // Default consistency for short sequences
    }

    let mut momentum_changes = Vec::new();

    // Calculate momentum between consecutive segments
    let segment_size = (prices.len() / 3).max(2);

    // Ensure we have enough data for the calculation
    if prices.len() < segment_size * 2 {
        return Ok(0.01); // Return minimum consistency for insufficient data
    }

    for i in 0..(prices.len() - segment_size * 2) {
        let seg1_start = prices[i];
        let seg1_end = prices[i + segment_size];
        let seg2_start = seg1_end;
        let seg2_end = prices[i + segment_size * 2];

        if seg1_start != 0.0 && seg2_start != 0.0 {
            let seg1_momentum = (seg1_end - seg1_start) / seg1_start;
            let seg2_momentum = (seg2_end - seg2_start) / seg2_start;
            let momentum_change = seg2_momentum - seg1_momentum;

            if momentum_change.is_finite() {
                momentum_changes.push(momentum_change);
            }
        }
    }

    if momentum_changes.is_empty() {
        return Ok(0.01);
    }

    // Calculate standard deviation of momentum changes (trend consistency)
    let mean = momentum_changes.iter().sum::<f64>() / momentum_changes.len() as f64;
    let variance = momentum_changes
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / momentum_changes.len() as f64;
    let std_dev = variance.sqrt();

    Ok(std_dev.max(0.005)) // Minimum consistency threshold
}

/// Calculate classification strength for direction based on distance from boundaries
///
/// Strength represents how confident/strong the classification is:
/// - 1.0 = Very strong (deep in the center of the class range)
/// - 0.5 = Moderate (near class boundaries)
/// - 0.0 = Very weak (just barely in the class)
fn calculate_direction_strength(
    momentum_change: f64,
    base_threshold: f64,
    extreme_threshold: f64,
    class: i32,
) -> f64 {
    match class {
        0 => {
            // DUMP: momentum_change <= -extreme_threshold
            // The more negative beyond extreme, the stronger
            let distance_beyond = (-momentum_change - extreme_threshold).max(0.0);
            let max_distance = extreme_threshold; // Reasonable max distance
            (distance_beyond / max_distance).clamp(0.1, 1.0) // At least 0.1 strength
        }
        1 => {
            // DOWN: -extreme_threshold < momentum_change <= -base_threshold
            let range_center = -(extreme_threshold + base_threshold) / 2.0;
            let range_half_width = (extreme_threshold - base_threshold) / 2.0;
            let distance_from_center = (momentum_change - range_center).abs();
            let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
            strength.max(0.1) // At least 0.1 strength
        }
        2 => {
            // SIDEWAYS: -base_threshold < momentum_change <= base_threshold
            // Closer to zero = stronger sideways signal
            let distance_from_zero = momentum_change.abs();
            let strength = 1.0 - (distance_from_zero / base_threshold).min(1.0);
            strength.max(0.1) // At least 0.1 strength
        }
        3 => {
            // UP: base_threshold < momentum_change <= extreme_threshold
            let range_center = (base_threshold + extreme_threshold) / 2.0;
            let range_half_width = (extreme_threshold - base_threshold) / 2.0;
            let distance_from_center = (momentum_change - range_center).abs();
            let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
            strength.max(0.1) // At least 0.1 strength
        }
        4 => {
            // PUMP: momentum_change > extreme_threshold
            // The more positive beyond extreme, the stronger
            let distance_beyond = (momentum_change - extreme_threshold).max(0.0);
            let max_distance = extreme_threshold; // Reasonable max distance
            (distance_beyond / max_distance).clamp(0.1, 1.0) // At least 0.1 strength
        }
        _ => 0.5, // Default neutral strength
    }
}

/// Calculate directional momentum change between sequence and horizon
///
/// Direction classification should detect TREND CHANGES and MOMENTUM SHIFTS,
/// not just movement magnitude. This function analyzes how the directional
/// momentum changes from the sequence period to the horizon period.
///
/// **ENHANCED VERSION**: Normalizes momentum change by sequence volatility
/// to make the signal more stable and comparable across different market conditions.
fn calculate_directional_momentum_change(
    sequence_prices: &[f64],
    horizon_prices: &[f64],
) -> Result<(f64, f64, f64)> {
    if sequence_prices.len() < 2 || horizon_prices.len() < 2 {
        return Ok((0.0, 0.0, 0.0));
    }

    // Calculate sequence momentum (trend strength and direction)
    let seq_start = sequence_prices[0];
    let seq_end = sequence_prices[sequence_prices.len() - 1];

    // Avoid division by zero - use small epsilon if needed
    let sequence_momentum = if seq_start.abs() < 1e-10 {
        0.0 // No momentum if starting from near-zero
    } else {
        (seq_end - seq_start) / seq_start
    };

    // Calculate horizon momentum (trend strength and direction)
    let hor_start = horizon_prices[0]; // This is same as seq_end
    let hor_end = horizon_prices[horizon_prices.len() - 1];

    // Avoid division by zero - use small epsilon if needed
    let horizon_momentum = if hor_start.abs() < 1e-10 {
        0.0 // No momentum if starting from near-zero
    } else {
        (hor_end - hor_start) / hor_start
    };

    // Calculate raw momentum change
    let raw_momentum_change = horizon_momentum - sequence_momentum;

    // ENHANCEMENT: Normalize by sequence volatility for stability
    // Calculate coefficient of variation (CV) for sequence
    let sequence_mean = sequence_prices.iter().sum::<f64>() / sequence_prices.len() as f64;
    let sequence_variance = sequence_prices
        .iter()
        .map(|&p| (p - sequence_mean).powi(2))
        .sum::<f64>()
        / sequence_prices.len() as f64;
    let sequence_std = sequence_variance.sqrt();

    // Coefficient of variation (volatility as percentage of price)
    let sequence_cv = if sequence_mean.abs() > 1e-10 {
        sequence_std / sequence_mean
    } else {
        0.01 // Default 1% volatility for edge cases
    };

    // Normalize momentum change by volatility context
    // Higher volatility = larger normalization factor = smaller normalized signal
    // This makes momentum changes comparable across different volatility regimes
    let volatility_normalization_factor = 1.0 + sequence_cv;
    let normalized_momentum_change = raw_momentum_change / volatility_normalization_factor;

    log::debug!(
        "🎯 Direction Momentum: seq_cv={:.4}, raw_change={:.6}, normalized_change={:.6} (factor={:.3})",
        sequence_cv,
        raw_momentum_change,
        normalized_momentum_change,
        volatility_normalization_factor
    );

    Ok((
        sequence_momentum,
        horizon_momentum,
        normalized_momentum_change,
    ))
}

/// Classify direction using momentum change analysis
///
/// This is the correct directional approach that focuses on TREND CHANGES:
/// - DUMP: Strong momentum reversal from positive to negative
/// - DOWN: Moderate momentum weakening or slight reversal
/// - SIDEWAYS: Momentum continuation with minimal change
/// - UP: Moderate momentum strengthening or slight acceleration
/// - PUMP: Strong momentum acceleration or reversal from negative to positive
///
/// Key insight: Direction is about momentum CHANGE, not absolute movement
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
            "📊 Momentum-Based Direction Analysis [{}]: No valid targets found",
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
        "📊 Momentum-Based Direction Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}] (BEFORE balanced selection)",
        horizon,
        valid_targets,
        imbalance_ratio,
        class_percentages.join(", ")
    );
}

// ============================================================================
// PREDICTION RECONSTRUCTION METHODS
// ============================================================================

/// Reconstruction result for direction predictions

#[derive(Debug, Clone)]
pub struct DirectionReconstruction {
    /// Class probabilities from model [DUMP, DOWN, SIDEWAYS, UP, PUMP]
    pub probabilities: Vec<f64>,
    /// Representative momentum change values used for each class
    pub momentum_changes: Vec<f64>,
    /// Trend acceleration percentages for each class
    pub trend_accelerations: Vec<f64>,
    /// Most likely direction class index
    pub most_likely_class: usize,
    /// Confidence (probability of most likely class)
    pub confidence: f64,
    /// Expected momentum change (weighted average across classes)
    pub expected_momentum_change: f64,
    /// Expected trend acceleration percentage (weighted average)
    pub expected_trend_acceleration: f64,
    /// Breakout probability (DUMP + PUMP)
    pub breakout_probability: f64,
    /// Upward bias probability (UP + PUMP)
    pub upward_probability: f64,
    /// Downward bias probability (DUMP + DOWN)
    pub downward_probability: f64,
    /// Sequence trend consistency used for thresholds (persistence proxy)
    pub trend_consistency: f64,
    /// Base threshold used for classification
    pub base_threshold: f64,
    /// Extreme threshold used for classification
    pub extreme_threshold: f64,
    // -------------------- Enhanced reconstruction metrics --------------------
    /// 10th percentile of momentum change (discrete CDF over class reps)
    pub momentum_ci_10: f64,
    /// 90th percentile of momentum change (discrete CDF over class reps)
    pub momentum_ci_90: f64,
    /// Normalized magnitude of expected momentum change in [0,1]
    pub directional_magnitude: f64,
    /// Normalized geometric distance to nearest boundary (0 at boundary, 1 at center)
    pub class_margin: f64,
    /// Normalized certainty from entropy (1 - H/ln(5))
    pub entropy_norm: f64,
    /// Distribution skew toward up vs down: (UP+PUMP) - (DOWN+DUMP)
    pub directional_skew: f64,
    /// Estimated horizon momentum = sequence_momentum + expected_momentum_change
    pub horizon_momentum_estimate: f64,
    /// Exposed persistence score (alias of trend_consistency)
    pub persistence_score: f64,
}

/// Reconstruct direction predictions from model probabilities
///
/// This method reverses the training classification logic to convert
/// raw model probabilities back to meaningful momentum changes and trend accelerations.
///
/// # Arguments
/// * `probabilities` - 5-element array of class probabilities [DUMP, DOWN, SIDEWAYS, UP, PUMP]
/// * `sequence_ohlcv` - OHLCV data for the input sequence (same as used in training)
/// * `config` - Optional configuration (uses defaults if None)
///
/// # Returns
/// * `DirectionReconstruction` - Complete reconstruction with momentum values and metrics
pub fn reconstruct_direction(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::DirectionParams,
) -> Result<DirectionReconstruction> {
    // Validate inputs
    if probabilities.len() != 5 {
        return Err(crate::utils::error::VangaError::DataError(
            "Direction reconstruction requires exactly 5 class probabilities".to_string(),
        ));
    }

    if sequence_ohlcv.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Sequence OHLCV data is required for direction reconstruction".to_string(),
        ));
    }

    // Extract sequence prices for momentum calculation
    let sequence_prices: Vec<f64> = sequence_ohlcv
        .iter()
        .map(|row| (row.open + row.high + row.low + row.close) / 4.0) // OHLC4
        .collect();

    // Calculate trend consistency (same as training)
    let trend_consistency = calculate_sequence_trend_consistency(&sequence_prices)?;

    // Use calibrated parameters
    let base_multiplier = calibrated_params.base_multiplier; // Use calibrated value, not hardcoded
    let extreme_multiplier = calibrated_params.extreme_multiplier;

    let base_threshold = trend_consistency * calibrated_params.sensitivity * base_multiplier;
    let extreme_threshold = base_threshold * extreme_multiplier;

    // Apply same minimum thresholds as training (from calibrated parameters)
    let min_base = calibrated_params.min_base_threshold;
    let min_extreme = calibrated_params.min_extreme_threshold;
    let final_base_threshold = base_threshold.max(min_base);
    let final_extreme_threshold = extreme_threshold.max(min_extreme);

    // CRITICAL: Calculate volatility normalization factor (same as training)
    // This ensures reconstruction uses the SAME normalized values as classification
    let sequence_mean = sequence_prices.iter().sum::<f64>() / sequence_prices.len() as f64;
    let sequence_variance = sequence_prices
        .iter()
        .map(|&p| (p - sequence_mean).powi(2))
        .sum::<f64>()
        / sequence_prices.len() as f64;
    let sequence_std = sequence_variance.sqrt();
    let sequence_cv = if sequence_mean.abs() > 1e-10 {
        sequence_std / sequence_mean
    } else {
        0.01 // Default 1% volatility
    };
    let volatility_normalization_factor = 1.0 + sequence_cv;

    // Representative momentum change for each class (midpoint of each class range)
    // IMPORTANT: These are NORMALIZED momentum changes (same as training)
    // The thresholds are applied to normalized values, so representatives must also be normalized
    let normalized_momentum_changes = [
        // DUMP: midpoint of (-∞, -extreme_threshold]
        -final_extreme_threshold - (final_extreme_threshold - final_base_threshold) / 2.0,
        // DOWN: midpoint of (-extreme_threshold, -base_threshold]
        -(final_extreme_threshold + final_base_threshold) / 2.0,
        // SIDEWAYS: midpoint of (-base_threshold, +base_threshold]
        0.0,
        // UP: midpoint of (+base_threshold, +extreme_threshold]
        (final_base_threshold + final_extreme_threshold) / 2.0,
        // PUMP: midpoint of (+extreme_threshold, +∞)
        final_extreme_threshold + (final_extreme_threshold - final_base_threshold) / 2.0,
    ];

    // Convert normalized momentum changes back to raw momentum changes for interpretation
    // This allows users to understand the actual momentum change magnitude
    let raw_momentum_changes: Vec<f64> = normalized_momentum_changes
        .iter()
        .map(|&normalized| normalized * volatility_normalization_factor)
        .collect();

    // Convert raw momentum changes to trend acceleration percentages
    let trend_accelerations: Vec<f64> = raw_momentum_changes
        .iter()
        .map(|&change| change * 100.0) // Convert to percentage
        .collect();

    // Find most likely class and confidence
    let (most_likely_class, confidence) = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &prob)| (idx, prob))
        .unwrap_or((2, 0.2)); // Default to SIDEWAYS

    // Expected values (weighted averages using RAW momentum changes)
    let expected_momentum_change: f64 = probabilities
        .iter()
        .zip(raw_momentum_changes.iter())
        .map(|(&prob, &change)| prob * change)
        .sum();

    let expected_trend_acceleration: f64 = probabilities
        .iter()
        .zip(trend_accelerations.iter())
        .map(|(&prob, &accel)| prob * accel)
        .sum();

    // Directional aggregates
    let breakout_probability = probabilities[0] + probabilities[4]; // DUMP + PUMP
    let upward_probability = probabilities[3] + probabilities[4]; // UP + PUMP
    let downward_probability = probabilities[0] + probabilities[1]; // DUMP + DOWN

    // Additional reconstruction metrics (using RAW momentum changes for interpretation)
    // 1) Credible interval over momentum change (10% and 90% quantiles)
    let mut cdf = 0.0;
    let mut q10 = raw_momentum_changes[0];
    let mut q90 = raw_momentum_changes[4];
    for (i, &p) in probabilities.iter().enumerate() {
        cdf += p;
        if cdf >= 0.10 && q10 == raw_momentum_changes[0] {
            q10 = raw_momentum_changes[i];
        }
        if cdf >= 0.90 {
            q90 = raw_momentum_changes[i];
            break;
        }
    }

    // 2) Directional magnitude normalized to [0,1]
    let max_abs = (final_extreme_threshold * volatility_normalization_factor * 1.5)
        .abs()
        .max(1e-12);
    let directional_magnitude = (expected_momentum_change.abs() / max_abs).min(1.0);

    let boundaries = [
        0.5 * (raw_momentum_changes[0] + raw_momentum_changes[1]),
        0.5 * (raw_momentum_changes[1] + raw_momentum_changes[2]),
        0.5 * (raw_momentum_changes[2] + raw_momentum_changes[3]),
        0.5 * (raw_momentum_changes[3] + raw_momentum_changes[4]),
    ];
    // Find interval containing expected_momentum_change
    let (left_bound, right_bound) = if expected_momentum_change <= boundaries[0] {
        (f64::NEG_INFINITY, boundaries[0])
    } else if expected_momentum_change <= boundaries[1] {
        (boundaries[0], boundaries[1])
    } else if expected_momentum_change <= boundaries[2] {
        (boundaries[1], boundaries[2])
    } else if expected_momentum_change <= boundaries[3] {
        (boundaries[2], boundaries[3])
    } else {
        (boundaries[3], f64::INFINITY)
    };
    let class_margin = if left_bound.is_finite() && right_bound.is_finite() {
        let center = 0.5 * (left_bound + right_bound);
        let half_interval = 0.5 * (right_bound - left_bound).max(1e-12);
        (1.0 - ((expected_momentum_change - center).abs() / half_interval)).clamp(0.0, 1.0)
    } else {
        // Outside inner intervals: margin degrades toward 0 near open ends
        0.0
    };

    // 4) Entropy (normalized) and skew
    let ln5 = 5_f64.ln();
    let entropy = probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    let entropy_norm = (1.0 - (entropy / ln5)).clamp(0.0, 1.0);
    let directional_skew =
        (probabilities[3] + probabilities[4]) - (probabilities[0] + probabilities[1]);

    // 5) Horizon momentum estimate = sequence momentum + expected Δmomentum
    let seq_start = sequence_prices.first().copied().unwrap_or(0.0);
    let seq_end = sequence_prices.last().copied().unwrap_or(0.0);
    let sequence_momentum = if seq_start.abs() < 1e-12 {
        0.0
    } else {
        (seq_end - seq_start) / seq_start
    };
    let horizon_momentum_estimate = sequence_momentum + expected_momentum_change;

    Ok(DirectionReconstruction {
        probabilities: probabilities.to_vec(),
        momentum_changes: raw_momentum_changes, // Use RAW values for user interpretation
        trend_accelerations,
        most_likely_class,
        confidence,
        expected_momentum_change,
        expected_trend_acceleration,
        breakout_probability,
        upward_probability,
        downward_probability,
        trend_consistency,
        base_threshold: final_base_threshold,
        extreme_threshold: final_extreme_threshold,
        // New fields
        momentum_ci_10: q10,
        momentum_ci_90: q90,
        directional_magnitude,
        class_margin,
        entropy_norm,
        directional_skew,
        horizon_momentum_estimate,
        persistence_score: trend_consistency,
    })
}

/// Convert class probabilities to expected momentum change values
///
/// This method calculates the expected momentum change for each class based on
/// the same mathematical logic used in training target generation.
/// Get direction class names in order
pub fn get_direction_class_names() -> Vec<&'static str> {
    vec!["DUMP", "DOWN", "SIDEWAYS", "UP", "PUMP"]
}
