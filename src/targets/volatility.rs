//! Volatility target generation for cryptocurrency market regime classification
//!
//! This module implements logarithmic ratio-based volatility regime classification:
//! - 0: VeryLow (target_atr << train_atr, extreme decrease)
//! - 1: Low (target_atr < train_atr, moderate decrease)
//! - 2: Medium (target_atr ≈ train_atr, similar volatility)
//! - 3: High (target_atr > train_atr, moderate increase)
//! - 4: VeryHigh (target_atr >> train_atr, extreme increase)
//!
//! ## Mathematical Approach: Logarithmic Ratio Classification
//!
//! **Why Logarithmic Space?**
//! Volatility ratios are naturally multiplicative and asymmetric. A 2x increase (ratio=2.0)
//! should be treated equally to a 0.5x decrease (ratio=0.5), but in linear space:
//! - 2.0 - 1.0 = +1.0 (increase)
//! - 0.5 - 1.0 = -0.5 (decrease) ← Asymmetric!
//!
//! In logarithmic space, ratios become symmetric:
//! - ln(2.0) = +0.693 (increase)
//! - ln(0.5) = -0.693 (decrease) ← Perfectly symmetric!
//!
//! **Algorithm:**
//! 1. **ATR Calculation**: Compute Average True Range for both periods
//!    - `train_atr`: ATR from input sequence (baseline volatility)
//!    - `target_atr`: ATR from horizon period (volatility to classify)
//! 2. **Logarithmic Ratio**: `log_ratio = ln(target_atr / train_atr)`
//! 3. **Symmetric Classification**: Apply bandwidth_size-based thresholds to log_ratio
//!
//! ## Configuration Parameters
//!
//! ### `bandwidth_size` (default: 0.4)
//! Controls the sensitivity of volatility regime detection. In volatility context,
//! this parameter defines the logarithmic threshold boundaries for classification.
//!
//! **Threshold Calculation:**
//! ```text
//! half_bandwidth = bandwidth_size / 2.0
//! extreme_bandwidth = bandwidth_size * extreme_multiplier
//!
//! VeryLow:  log_ratio <= -extreme_bandwidth  (e.g., <= -0.8)
//! Low:      -extreme_bandwidth < log_ratio <= -half_bandwidth  (e.g., -0.8 to -0.2)
//! Medium:   -half_bandwidth < log_ratio <= +half_bandwidth  (e.g., -0.2 to +0.2)
//! High:     +half_bandwidth < log_ratio <= +extreme_bandwidth  (e.g., +0.2 to +0.8)
//! VeryHigh: log_ratio > +extreme_bandwidth  (e.g., > +0.8)
//! ```
//!
//! **Ratio Interpretation:**
//! With bandwidth_size=0.4 and extreme_multiplier=2.0:
//! - VeryLow: target_atr ≤ 0.45 × train_atr (55%+ decrease)
//! - Low: 0.45 × train_atr < target_atr ≤ 0.82 × train_atr (18-55% decrease)
//! - Medium: 0.82 × train_atr < target_atr ≤ 1.22 × train_atr (±18% change)
//! - High: 1.22 × train_atr < target_atr ≤ 2.23 × train_atr (22-123% increase)
//! - VeryHigh: target_atr > 2.23 × train_atr (123%+ increase)
//!
//! **Recommended Values:**
//! - **Sensitive (0.2-0.3)**: Detects subtle volatility regime changes
//! - **Standard (0.4-0.6)**: Balanced for most crypto volatility patterns
//! - **Conservative (0.8-1.2)**: Only major volatility regime shifts
//!
//! ### `extreme_multiplier` (default: 2.0)
//! Multiplier for extreme class boundaries (VeryLow/VeryHigh vs Low/High).
//! Higher values = fewer extreme classifications, more moderate classifications.
//!
//! ## Usage Examples
//!
//! ```rust
//! use crate::config::model::VolatilityHead;
//!
//! // Sensitive: Detects subtle volatility changes
//! let sensitive_config = VolatilityHead {
//!     enabled: true,
//!     bandwidth_size: Some(0.3),
//!     base_threshold: Some(0.15),
//!     extreme_multiplier: Some(2.0),
//! };
//!
//! // Standard: Balanced volatility regime detection
//! let standard_config = VolatilityHead {
//!     enabled: true,
//!     bandwidth_size: Some(0.4),
//!     base_threshold: Some(0.15),
//!     extreme_multiplier: Some(2.0),
//! };
//!
//! // Conservative: Only major volatility shifts
//! let conservative_config = VolatilityHead {
//!     enabled: true,
//!     bandwidth_size: Some(0.8),
//!     base_threshold: Some(0.15),
//!     extreme_multiplier: Some(1.5),
//! };
//! ```
//!
//! ## Target Differentiation Strategy
//!
//! **Volatility vs Other Targets:**
//! - **VOLATILITY**: "How volatile will it be?" (risk assessment)
//! - **DIRECTION**: "How is trend momentum changing?" (acceleration/deceleration)
//! - **PRICE_LEVELS**: "Where will price be?" (range/breakout analysis)
//!
//! Volatility provides risk assessment complementary to directional and price predictions,
//! enabling comprehensive market regime analysis for position sizing and risk management.

use crate::data::structures::MarketDataRow;
use crate::utils::error::Result;
use crate::utils::market_data::extract_ohlcv_data;
use crate::utils::parser::parse_horizon_to_steps;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generate volatility targets with optional adaptive parameters
///
/// When adaptive_params is provided, uses the pre-calibrated parameters for consistent
/// target generation between training and prediction. When None, performs calibration.
pub fn generate_volatility_targets_with_calibrated_params(
    df: &DataFrame,
    horizons: &[String],
    sequence_indices: &[usize],
    sequence_length: usize,
    calibrated_params: &crate::targets::calibration::VolatilityParams,
) -> Result<HashMap<String, Vec<i32>>> {
    let ohlcv_data = extract_ohlcv_data(df)?;
    let mut targets = HashMap::new();

    // Use pre-calibrated adaptive parameters
    let calibrated_bandwidth = calibrated_params.bandwidth;
    log::info!(
        "🎯 Using pre-calibrated volatility bandwidth: {:.6}",
        calibrated_bandwidth
    );

    let extreme_multiplier = calibrated_params.extreme_multiplier;

    log::info!(
        "🎯 Volatility targets using bandwidth: {:.6}, extreme_multiplier: {:.2}",
        calibrated_bandwidth,
        extreme_multiplier
    );

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let mut horizon_targets = vec![-1; sequence_indices.len()];

        for (seq_position, &seq_idx) in sequence_indices.iter().enumerate() {
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            // Check boundaries - need both sequence and horizon data
            if target_end_idx <= ohlcv_data.len() && sequence_end_idx <= ohlcv_data.len() {
                // Get INPUT sequence candles (for baseline volatility)
                let sequence_candles = &ohlcv_data[seq_idx..sequence_end_idx];

                // Get HORIZON candles (from sequence end to target horizon)
                let horizon_candles = &ohlcv_data[sequence_end_idx..target_end_idx];

                // Only classify if we have sufficient data for ATR calculation
                if sequence_candles.len() >= 2 && horizon_candles.len() >= 2 {
                    // Use enhanced classification with calibrated adaptive parameters
                    let volatility_class = classify_volatility_with_calibrated_params(
                        sequence_candles,
                        horizon_candles,
                        calibrated_params,
                    )?;

                    horizon_targets[seq_position] = volatility_class;
                }
            }
        }

        log_volatility_distribution(&horizon_targets, horizon);
        targets.insert(horizon.clone(), horizon_targets);
    }

    Ok(targets)
}

/// Calculate rolling ATR series for distribution analysis
///
/// This function computes a rolling Average True Range series from the sequence
/// to analyze volatility distribution patterns. Essential for sequence-adaptive
/// threshold calculation and balanced volatility regime classification.
///
/// ## Algorithm
/// 1. Calculate ATR for each rolling window in the sequence
/// 2. Normalize each ATR by the corresponding price level
/// 3. Return series of normalized ATR values for statistical analysis
///
/// ## Parameters
/// - `candles`: OHLCV data sequence for ATR calculation
/// - `window`: Rolling window size for ATR calculation (minimum 2)
///
/// ## Returns
/// Vector of normalized ATR values representing volatility distribution
pub fn calculate_rolling_atr_series(candles: &[MarketDataRow], window: usize) -> Result<Vec<f64>> {
    if candles.len() < window.max(2) {
        return Ok(vec![0.02]); // Default 2% volatility for insufficient data
    }

    let mut atr_series = Vec::new();
    let effective_window = window.max(2);

    // Calculate rolling ATR for each possible window position
    for i in 0..=(candles.len() - effective_window) {
        let window_candles = &candles[i..i + effective_window];
        let window_atr = get_sequence_atr_baseline(window_candles, 0.005)?;

        if window_atr.is_finite() && window_atr > 0.0 {
            atr_series.push(window_atr);
        }
    }

    // Ensure we have at least some data points
    if atr_series.is_empty() {
        atr_series.push(0.02); // Default fallback
    }

    log::trace!(
        "🎯 Rolling ATR Series: {} candles, window={}, {} ATR values, range=[{:.6}, {:.6}]",
        candles.len(),
        effective_window,
        atr_series.len(),
        atr_series
            .iter()
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f64::INFINITY),
        atr_series
            .iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f64::NEG_INFINITY)
    );

    Ok(atr_series)
}

/// Calculate statistical distribution metrics for ATR series analysis
///
/// Computes comprehensive statistics for the ATR distribution to enable
/// adaptive threshold calculation and volatility regime detection.
///
/// ## Returns
/// Statistical metrics including mean, standard deviation, and percentiles
/// for adaptive volatility classification.
pub fn calculate_atr_distribution_stats(atr_series: &[f64]) -> AtrDistributionStats {
    if atr_series.is_empty() {
        return AtrDistributionStats {
            mean: 0.02,
            std_dev: 0.01,
            median: 0.02,
            percentile_25: 0.015,
            percentile_75: 0.025,
            coefficient_of_variation: 0.5,
        };
    }

    let mean = atr_series.iter().sum::<f64>() / atr_series.len() as f64;
    let variance =
        atr_series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / atr_series.len() as f64;
    let std_dev = variance.sqrt();

    // Calculate percentiles
    let mut sorted_series = atr_series.to_vec();
    sorted_series.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = sorted_series.len();
    let median = if len % 2 == 0 {
        (sorted_series[len / 2 - 1] + sorted_series[len / 2]) / 2.0
    } else {
        sorted_series[len / 2]
    };

    let percentile_25 = sorted_series[len / 4];
    let percentile_75 = sorted_series[3 * len / 4];

    let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

    AtrDistributionStats {
        mean,
        std_dev,
        median,
        percentile_25,
        percentile_75,
        coefficient_of_variation,
    }
}

/// Statistical distribution metrics for ATR series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtrDistributionStats {
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub percentile_25: f64,
    pub percentile_75: f64,
    pub coefficient_of_variation: f64,
}

impl Default for AtrDistributionStats {
    fn default() -> Self {
        Self {
            mean: 0.02,
            std_dev: 0.01,
            median: 0.02,
            percentile_25: 0.015,
            percentile_75: 0.025,
            coefficient_of_variation: 0.5,
        }
    }
}

/// Calculate adaptive volatility bandwidth based on ATR distribution analysis
///
/// This function analyzes the ATR distribution to determine optimal bandwidth
/// for volatility classification thresholds. Ensures balanced class distribution
/// across different market volatility regimes.
///
/// ## Algorithm
/// 1. Analyze ATR distribution characteristics
/// 2. Calculate volatility coefficient of variation
/// 3. Scale base sensitivity by volatility characteristics
/// 4. Apply bounds to ensure reasonable threshold ranges
///
/// ## Parameters
/// - `atr_series`: Rolling ATR series for distribution analysis
/// - `base_sensitivity`: Base sensitivity parameter for scaling
///
/// ## Returns
/// Adaptive bandwidth value optimized for current volatility regime
pub fn calculate_adaptive_volatility_bandwidth(
    atr_series: &[f64],
    base_sensitivity: f64,
) -> Result<f64> {
    if atr_series.is_empty() {
        return Ok(base_sensitivity);
    }

    let stats = calculate_atr_distribution_stats(atr_series);

    // Calculate coefficient of variation for ATR distribution
    let cv = if stats.mean > 1e-8 {
        stats.std_dev / stats.mean
    } else {
        0.5 // Default CV for edge cases
    };

    // Scale bandwidth based on ATR distribution characteristics
    // Higher CV (more volatile ATR) = wider bandwidth (less sensitive)
    // Lower CV (stable ATR) = narrower bandwidth (more sensitive)
    let cv_multiplier = (cv / 0.5).clamp(0.4, 2.5); // Normalize around 0.5 CV baseline
    let adaptive_bandwidth = base_sensitivity * cv_multiplier;

    // Apply reasonable bounds
    let final_bandwidth = adaptive_bandwidth.clamp(0.05, 1.0);

    log::debug!(
        "🎯 Adaptive Volatility Bandwidth: atr_mean={:.6}, atr_std={:.6}, cv={:.4}, cv_mult={:.2}, bandwidth={:.4}",
        stats.mean, stats.std_dev, cv, cv_multiplier, final_bandwidth
    );

    Ok(final_bandwidth)
}
///
/// Computes Average True Range for the sequence, using calibrated baseline
/// instead of hardcoded values. Provides adaptive fallback based on sequence price volatility.
pub fn get_sequence_atr_baseline(
    sequence_candles: &[MarketDataRow],
    min_volatility_baseline: f64,
) -> Result<f64> {
    if sequence_candles.len() < 2 {
        // ADAPTIVE FALLBACK: Use calibrated minimum baseline instead of hardcoded 0.005
        let fallback_atr = if !sequence_candles.is_empty() {
            sequence_candles[0].close * min_volatility_baseline
        } else {
            min_volatility_baseline // Use calibrated minimum for edge cases
        };
        return Ok(fallback_atr);
    }

    let mut true_ranges = Vec::new();

    // Calculate ATR for each candle in the sequence
    for i in 1..sequence_candles.len() {
        let current = &sequence_candles[i];
        let previous = &sequence_candles[i - 1];

        // True Range: max(high-low, |high-prev_close|, |low-prev_close|)
        let hl = current.high - current.low;
        let hc = (current.high - previous.close).abs();
        let lc = (current.low - previous.close).abs();

        let true_range = hl.max(hc).max(lc);
        if true_range.is_finite() && true_range > 0.0 {
            true_ranges.push(true_range / current.close); // Normalize by price
        }
    }

    if true_ranges.is_empty() {
        // ADAPTIVE FALLBACK: Use sequence price range as volatility estimate
        let prices: Vec<f64> = sequence_candles.iter().map(|c| c.close).collect();
        let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let price_range = (max_price - min_price) / min_price;
        return Ok(price_range.max(min_volatility_baseline)); // Use calibrated minimum
    }

    // Average True Range of the sequence - this is our adaptive baseline
    let sequence_atr = true_ranges.iter().sum::<f64>() / true_ranges.len() as f64;

    log::trace!(
        "🎯 Sequence ATR Baseline: {} candles, {} true_ranges, atr={:.6}",
        sequence_candles.len(),
        true_ranges.len(),
        sequence_atr
    );

    Ok(sequence_atr.max(min_volatility_baseline)) // Use calibrated minimum volatility baseline
}

// Removed unused function calculate_proportional_atr_window - no longer needed in simplified approach

/// Calculate horizon ATR with adaptive exponential weighting toward recent steps
///
/// This function computes ATR for horizon candles using exponential decay weighting
/// where recent time steps (closer to the prediction point) receive higher weights.
/// This provides more responsive volatility classification that emphasizes recent
/// market conditions over historical patterns.
///
/// ## Algorithm
/// 1. Calculate True Range for each horizon candle (standard ATR calculation)
/// 2. Apply exponential decay weights: `weight = decay_factor^(steps_from_end)`
/// 3. Compute weighted average of normalized true ranges
/// 4. Recent steps get higher weights, earlier steps get progressively lower weights
///
/// ## Parameters
/// - `horizon_candles`: OHLCV data for the horizon period
/// - `decay_factor`: Exponential decay factor (from adaptive calibration)
///   - Values < 1.0: Emphasize recent volatility (e.g., 0.95, 0.90)
///   - Value = 1.0: Uniform weighting (equivalent to get_sequence_atr_baseline)
///   - Typical calibrated range: 0.85 - 1.0
///
/// ## Returns
/// Weighted ATR value emphasizing recent volatility patterns
///
/// ## Mathematical Foundation
/// For horizon with N candles, weights are calculated as:
/// ```text
/// candle[i] weight = decay_factor^(N - i - 1)
///
/// Example with decay_factor=0.95, N=4:
/// candle[0]: weight = 0.95^3 = 0.857 (earliest, lowest weight)
/// candle[1]: weight = 0.95^2 = 0.903
/// candle[2]: weight = 0.95^1 = 0.950
/// candle[3]: weight = 0.95^0 = 1.000 (most recent, highest weight)
/// ```
///
/// This ensures recent volatility patterns have stronger influence on classification
/// while maintaining mathematical consistency with the logarithmic ratio approach.
pub fn get_horizon_weighted_atr_baseline(
    horizon_candles: &[MarketDataRow],
    decay_factor: f64,
    min_volatility_baseline: f64,
) -> Result<f64> {
    // Fallback to uniform weighting for insufficient data or uniform decay factor
    if horizon_candles.len() < 2 || (decay_factor - 1.0).abs() < f64::EPSILON {
        return get_sequence_atr_baseline(horizon_candles, min_volatility_baseline);
    }

    let mut weighted_true_ranges = Vec::new();
    let mut total_weight = 0.0;

    // Calculate weighted True Range for each candle pair
    for i in 1..horizon_candles.len() {
        let current = &horizon_candles[i];
        let previous = &horizon_candles[i - 1];

        // Standard True Range calculation
        let hl = current.high - current.low;
        let hc = (current.high - previous.close).abs();
        let lc = (current.low - previous.close).abs();
        let true_range = hl.max(hc).max(lc);

        if true_range.is_finite() && true_range > 0.0 {
            // Exponential weighting: recent steps get higher weights
            // steps_from_end = 0 for most recent, increases for older candles
            let steps_from_end = horizon_candles.len() - i - 1;
            let weight = decay_factor.powi(steps_from_end as i32);

            // Normalize by current price and apply weight
            let normalized_tr = true_range / current.close;
            weighted_true_ranges.push(normalized_tr * weight);
            total_weight += weight;
        }
    }

    // Fallback to uniform calculation if no valid true ranges
    if weighted_true_ranges.is_empty() || total_weight == 0.0 {
        return get_sequence_atr_baseline(horizon_candles, min_volatility_baseline);
    }

    // Calculate weighted average ATR
    let weighted_atr = weighted_true_ranges.iter().sum::<f64>() / total_weight;

    log::trace!(
        "🎯 Horizon Weighted ATR: {} candles, decay_factor={:.3}, {} weighted_ranges, weighted_atr={:.6}, total_weight={:.3}",
        horizon_candles.len(),
        decay_factor,
        weighted_true_ranges.len(),
        weighted_atr,
        total_weight
    );

    Ok(weighted_atr.max(min_volatility_baseline)) // Use calibrated minimum volatility baseline
}

/// Logarithmic volatility thresholds for regime classification
///
/// This struct defines the boundary values used to classify volatility ratios
/// in logarithmic space into the 5-class volatility regime system.
/// All threshold values are in log space (natural logarithm).
///
/// ## Threshold Structure (Log Space)
/// ```text
/// VeryLow:  log_ratio <= very_low_max (most negative)
/// Low:      very_low_max < log_ratio <= low_max (moderate negative)
/// Medium:   low_max < log_ratio <= medium_max (around zero)
/// High:     medium_max < log_ratio <= high_max (moderate positive)
/// VeryHigh: log_ratio > high_max (most positive)
/// ```
///
/// ## Field Meanings
/// - `very_low_max`: Maximum log ratio for VeryLow class (extreme volatility decrease)
/// - `low_max`: Maximum log ratio for Low class (moderate volatility decrease)
/// - `medium_max`: Maximum log ratio for Medium class (similar volatility)
/// - `high_max`: Maximum log ratio for High class (moderate volatility increase)
/// - Values above `high_max` are classified as VeryHigh (extreme volatility increase)
///
/// ## Conversion to Ratio Space
/// To convert log thresholds back to ratio space: `ratio = exp(log_threshold)`
/// - Example: log_threshold = -0.693 → ratio = exp(-0.693) = 0.5 (50% of baseline)
#[derive(Debug, Clone)]
pub struct LogVolatilityThresholds {
    pub very_low_max: f64, // VeryLow threshold (log space)
    pub low_max: f64,      // Low threshold (log space)
    pub medium_max: f64,   // Medium threshold (log space)
    pub high_max: f64,     // High threshold (log space)
                           // Above high_max = VeryHigh
}

/// Simplified volatility classification using ATR momentum approach
///
/// SIMPLIFIED APPROACH: Following the successful sentiment pattern with 2 simple features:
/// - **ATR Momentum**: How volatility is changing (horizon_atr - sequence_atr) / sequence_atr
/// - **Volume Conviction**: Volume ratio adds conviction to volatility changes
///
/// This replaces the complex 4-feature approach with a simple, effective method that
/// mirrors the successful sentiment classification pattern.
///
/// ## Why Simplified?
/// - **Sentiment Success**: Simple momentum + volume approach fixed sentiment performance
/// - **ML Learning**: Simple features are easier for LSTM to learn patterns from
/// - **Research-Based**: ATR momentum is a proven volatility measurement technique
/// - **Volume Integration**: High volume volatility changes are more meaningful
pub fn classify_volatility_with_calibrated_params(
    sequence_candles: &[MarketDataRow],
    horizon_candles: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::VolatilityParams,
) -> Result<i32> {
    if sequence_candles.len() < 2 || horizon_candles.len() < 2 {
        return Ok(2); // Default to Medium for insufficient data
    }

    // Calculate ATR for both periods (same as before)
    let sequence_atr = calculate_simple_atr_with_params(
        sequence_candles,
        calibrated_params.min_volatility_baseline,
    )?;
    let horizon_atr = calculate_simple_atr_with_params(
        horizon_candles,
        calibrated_params.min_volatility_baseline,
    )?;

    // Ensure minimum baseline to avoid division by zero
    let baseline_atr = sequence_atr.max(calibrated_params.min_volatility_baseline);

    // SIMPLE FEATURE 1: ATR momentum (identical to price momentum formula)
    let atr_momentum = (horizon_atr - baseline_atr) / baseline_atr;

    // SIMPLE FEATURE 2: Volume conviction (identical to sentiment approach)
    let sequence_volume = calculate_average_volume(sequence_candles);
    let horizon_volume = calculate_average_volume(horizon_candles);
    let volume_conviction = if sequence_volume > 0.0 {
        (horizon_volume / sequence_volume).ln().clamp(-2.0, 2.0) // Log scale, clamped
    } else {
        0.0 // Neutral if no volume data
    };

    // Combine ATR momentum with volume conviction (momentum is primary signal)
    let volatility_score = atr_momentum + (volume_conviction * calibrated_params.volume_weight);

    // Use adaptive thresholds for classification (same structure as sentiment)
    let moderate_threshold = calibrated_params.bandwidth; // Now represents ATR momentum threshold
    let extreme_threshold = calibrated_params.bandwidth * calibrated_params.extreme_multiplier;

    // Classify based on combined volatility score
    let class = if volatility_score <= -extreme_threshold {
        0 // VeryLow: Large ATR decrease
    } else if volatility_score <= -moderate_threshold {
        1 // Low: Moderate ATR decrease
    } else if volatility_score < moderate_threshold {
        2 // Medium: Small ATR change in either direction
    } else if volatility_score < extreme_threshold {
        3 // High: Moderate ATR increase
    } else {
        4 // VeryHigh: Large ATR increase
    };

    log::debug!(
        "🎯 Simple ATR Momentum: seq_atr={:.6}, hor_atr={:.6}, momentum={:.4}, vol_conviction={:.4}, score={:.4}, thresholds=[{:.4}, {:.4}, {:.4}, {:.4}] → class={} ({})",
        baseline_atr, horizon_atr, atr_momentum, volume_conviction, volatility_score,
        -extreme_threshold, -moderate_threshold, moderate_threshold, extreme_threshold,
        class, ["VeryLow", "Low", "Medium", "High", "VeryHigh"][class as usize]
    );

    Ok(class)
}

/// Calibrate volatility thresholds for balanced class distribution using ATR ratios
///
/// This function analyzes historical ATR ratios (horizon/sequence) to find optimal
/// thresholds that achieve balanced class distribution (approximately 20% per class).
///
/// ## CORRECTED ALGORITHM
/// 1. Calculate ATR ratios for all sequence→horizon pairs in historical data
/// 2. Sort ratio values to find percentile boundaries
/// 3. Calculate thresholds that map percentiles to balanced 5-class system
/// 4. Return calibrated base_sensitivity parameter for ratio-based threshold calculation
///
/// ## Parameters
/// - `ohlcv_data`: Historical OHLCV data for volatility analysis
/// - `sequence_length`: Length of input sequences
/// - `horizon_steps`: Number of steps in prediction horizon
/// - `target_balance`: Target percentage for each class (e.g., 0.2 for 20%)
/// - `min_baseline`: Minimum volatility baseline to test during calibration
///
/// ## Returns
/// Calibrated base_sensitivity parameter for balanced volatility classification
pub fn calibrate_volatility_bandwidth(
    ohlcv_data: &[MarketDataRow],
    sequence_length: usize,
    horizon_steps: usize,
    target_balance: f64,
    min_baseline: f64,
) -> Result<f64> {
    if ohlcv_data.len() < sequence_length + horizon_steps + 10 {
        return Ok(0.3); // Default threshold for ratio-based approach
    }

    let mut volatility_ratios = Vec::new();

    // Sample ATR ratios from historical sequence→horizon pairs
    for i in 0..(ohlcv_data.len() - sequence_length - horizon_steps) {
        let sequence_candles = &ohlcv_data[i..i + sequence_length];
        let horizon_candles = &ohlcv_data[i + sequence_length..i + sequence_length + horizon_steps];

        if sequence_candles.len() >= 2 && horizon_candles.len() >= 2 {
            // Use the min_baseline parameter passed for calibration testing
            let sequence_atr = calculate_simple_atr_with_params(sequence_candles, min_baseline)
                .unwrap_or(min_baseline);
            let horizon_atr = calculate_simple_atr_with_params(horizon_candles, min_baseline)
                .unwrap_or(min_baseline);

            // Calculate ratio with the test baseline
            let baseline_atr = sequence_atr.max(min_baseline);
            let volatility_ratio = horizon_atr / baseline_atr;

            if volatility_ratio.is_finite() && volatility_ratio > 0.0 {
                volatility_ratios.push(volatility_ratio);
            }
        }
    }

    if volatility_ratios.is_empty() {
        return Ok(0.3); // Default fallback
    }

    // Sort ratios to find percentiles for balanced distribution
    volatility_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = volatility_ratios.len();

    // For 5-class system with target_balance per class (e.g., 20% each):
    // Find percentile boundaries around neutral (1.0 for ratios)
    let p20_idx = ((n as f64) * target_balance) as usize;
    let p40_idx = ((n as f64) * (2.0 * target_balance)) as usize;
    let p60_idx = ((n as f64) * (3.0 * target_balance)) as usize;
    let p80_idx = ((n as f64) * (4.0 * target_balance)) as usize;

    let p20_value = volatility_ratios[p20_idx.min(n - 1)];
    let p40_value = volatility_ratios[p40_idx.min(n - 1)];
    let p60_value = volatility_ratios[p60_idx.min(n - 1)];
    let p80_value = volatility_ratios[p80_idx.min(n - 1)];

    // Calculate threshold distances from neutral (1.0 for ratios)
    let threshold_40 = (1.0 - p40_value).abs();
    let threshold_60 = (p60_value - 1.0).abs();

    // Use the moderate threshold as base sensitivity
    let moderate_threshold = (threshold_40 + threshold_60) / 2.0;

    // Ensure reasonable bounds for ratio-based approach
    let final_sensitivity = moderate_threshold.clamp(0.1, 0.8);

    log::info!(
        "🎯 Calibrated volatility sensitivity: {:.4} (from {} samples)",
        final_sensitivity,
        n
    );
    log::info!(
        "🎯 Volatility ratio percentiles: P20={:.3}, P40={:.3}, P60={:.3}, P80={:.3}",
        p20_value,
        p40_value,
        p60_value,
        p80_value
    );

    Ok(final_sensitivity)
}

/// Comprehensive volatility parameter calibration following sentiment success pattern
///
/// COMPREHENSIVE APPROACH: Tests multiple parameter combinations like sentiment's 360 tests:
/// - **ATR Sensitivity**: How sensitive to ATR momentum changes (like sentiment's body_sensitivity)
/// - **Volume Weight**: How much volume conviction affects classification
/// - **Extreme Multiplier**: Boundary between moderate and extreme classes
/// - **Min Baseline**: Minimum volatility baseline for stability
///
/// This replaces the simple bandwidth calibration with comprehensive testing that
/// mirrors the successful sentiment calibration approach.
///
/// ## Algorithm
/// 1. Collect all ATR momentum and volume data for testing
/// 2. Test different parameter combinations (sensitivity × volume_weight × extreme_multiplier × min_baseline)
/// 3. Find optimal parameters that achieve balanced 20% per class distribution
/// 4. Return best parameters with balance quality score
pub fn calibrate_volatility_comprehensive(
    ohlcv_data: &[MarketDataRow],
    sequence_length: usize,
    horizon_steps: usize,
    target_balance: f64,
    min_baseline: f64,
) -> Result<(f64, f64, f64)> {
    if ohlcv_data.len() < sequence_length + horizon_steps + 50 {
        return Ok((0.02, 0.1, 2.0)); // Default parameters
    }

    // Collect all ATR momentum and volume data for testing
    let mut test_data = Vec::new();

    for i in 0..(ohlcv_data.len() - sequence_length - horizon_steps) {
        let sequence_ohlcv = &ohlcv_data[i..i + sequence_length];
        let horizon_ohlcv = &ohlcv_data[i + sequence_length..i + sequence_length + horizon_steps];

        if sequence_ohlcv.len() >= 3 && horizon_ohlcv.len() >= 3 {
            // Use the min_baseline parameter for calibration testing
            let sequence_atr = calculate_simple_atr_with_params(sequence_ohlcv, min_baseline)
                .unwrap_or(min_baseline);
            let horizon_atr = calculate_simple_atr_with_params(horizon_ohlcv, min_baseline)
                .unwrap_or(min_baseline);
            let sequence_volume = calculate_average_volume(sequence_ohlcv);
            let horizon_volume = calculate_average_volume(horizon_ohlcv);

            let baseline_atr = sequence_atr.max(min_baseline);
            if baseline_atr > 0.0 && sequence_volume > 0.0 {
                let atr_momentum = (horizon_atr - baseline_atr) / baseline_atr;
                let volume_conviction = (horizon_volume / sequence_volume).ln().clamp(-2.0, 2.0);

                if atr_momentum.is_finite() && volume_conviction.is_finite() {
                    test_data.push((atr_momentum, volume_conviction));
                }
            }
        }
    }

    if test_data.len() < 100 {
        return Ok((0.02, 0.1, 2.0)); // Need sufficient data for calibration
    }

    // Test different parameter combinations to find optimal balance
    let sensitivity_candidates = vec![
        0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08,
    ];
    let volume_weight_candidates = vec![0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
    let extreme_multiplier_candidates = vec![1.5, 1.8, 2.0, 2.2, 2.5, 3.0];

    let mut best_sensitivity = 0.02;
    let mut best_volume_weight = 0.1;
    let mut best_extreme_multiplier = 2.0;
    let mut best_balance_score = f64::INFINITY;

    log::info!(
        "🔍 Testing {} parameter combinations for optimal volatility calibration...",
        sensitivity_candidates.len()
            * volume_weight_candidates.len()
            * extreme_multiplier_candidates.len()
    );

    for &sensitivity in &sensitivity_candidates {
        for &volume_weight in &volume_weight_candidates {
            for &extreme_multiplier in &extreme_multiplier_candidates {
                // Test this parameter combination
                let balance_score = test_volatility_parameter_combination(
                    &test_data,
                    sensitivity,
                    volume_weight,
                    extreme_multiplier,
                    target_balance,
                );

                if balance_score < best_balance_score {
                    best_balance_score = balance_score;
                    best_sensitivity = sensitivity;
                    best_volume_weight = volume_weight;
                    best_extreme_multiplier = extreme_multiplier;
                }
            }
        }
    }

    log::info!(
        "🎯 Optimal volatility parameters: sensitivity={:.4}, volume_weight={:.3}, extreme_multiplier={:.2}, balance_score={:.4}",
        best_sensitivity, best_volume_weight, best_extreme_multiplier, best_balance_score
    );

    Ok((
        best_sensitivity,
        best_volume_weight,
        best_extreme_multiplier,
    ))
}

/// Test a specific volatility parameter combination and return balance quality score
fn test_volatility_parameter_combination(
    test_data: &[(f64, f64)], // (atr_momentum, volume_conviction) pairs
    sensitivity: f64,
    volume_weight: f64,
    extreme_multiplier: f64,
    target_balance: f64,
) -> f64 {
    let mut class_counts = [0usize; 5];
    let extreme_threshold = sensitivity * extreme_multiplier;

    // Classify all test samples with these parameters
    for &(atr_momentum, volume_conviction) in test_data {
        let volatility_score = atr_momentum + (volume_conviction * volume_weight);

        let class = if volatility_score <= -extreme_threshold {
            0 // VeryLow
        } else if volatility_score <= -sensitivity {
            1 // Low
        } else if volatility_score < sensitivity {
            2 // Medium
        } else if volatility_score < extreme_threshold {
            3 // High
        } else {
            4 // VeryHigh
        };

        class_counts[class] += 1;
    }

    // Calculate balance quality score (lower is better)
    let total_samples = test_data.len() as f64;
    let mut balance_score = 0.0;

    for count in &class_counts {
        let percentage = (*count as f64) / total_samples;
        let deviation = (percentage - target_balance).abs();
        balance_score += deviation * deviation; // Squared deviation penalty
    }

    // Add penalty for empty classes
    let empty_classes = class_counts.iter().filter(|&&count| count == 0).count();
    balance_score += (empty_classes as f64) * 0.1; // 10% penalty per empty class

    balance_score
}

/// Calculate simple ATR for sequence→horizon comparison
///
/// CORRECTED APPROACH: Simple, direct ATR calculation without complex windowing or weighting.
/// Used for both sequence (baseline) and horizon (target) measurements.
///
/// ## Core Algorithm
/// 1. **True Range**: Calculate max(high-low, |high-prev_close|, |low-prev_close|) for each candle
/// 2. **Price Normalization**: Divide each true range by current close price
/// 3. **Average**: Simple average of all normalized true ranges
///
/// ## Result
/// - **Normalized ATR**: Average true range as percentage of price
/// - **Comparable**: Same calculation for sequence and horizon enables direct ratio comparison
/// - **Stable**: Simple average without complex weighting schemes
///
/// ## Usage
/// - **Sequence**: Establishes volatility baseline/context
/// - **Horizon**: Measures target volatility for comparison
/// - **Ratio**: horizon_atr / sequence_atr shows volatility change
pub fn calculate_simple_atr(candles: &[MarketDataRow]) -> Result<f64> {
    calculate_simple_atr_with_params(candles, 0.005) // Default minimum for backward compatibility
}

/// Calculate simple ATR with configurable minimum baseline
pub fn calculate_simple_atr_with_params(
    candles: &[MarketDataRow],
    min_baseline: f64,
) -> Result<f64> {
    if candles.len() < 2 {
        return Ok(min_baseline * 2.0); // Default volatility for insufficient data
    }

    let mut true_ranges = Vec::new();

    // Calculate True Range for each candle pair
    for i in 1..candles.len() {
        let current = &candles[i];
        let previous = &candles[i - 1];

        // Standard True Range calculation
        let high_low = current.high - current.low;
        let high_close = (current.high - previous.close).abs();
        let low_close = (current.low - previous.close).abs();
        let true_range = high_low.max(high_close).max(low_close);

        if true_range.is_finite() && true_range > 0.0 && current.close > 0.0 {
            // Normalize by current price for percentage-based comparison
            let normalized_tr = true_range / current.close;
            true_ranges.push(normalized_tr);
        }
    }

    if true_ranges.is_empty() {
        // Fallback: use price range as volatility estimate
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let price_range = (max_price - min_price) / min_price;
        return Ok(price_range.max(min_baseline));
    }

    // Simple average of normalized true ranges
    let simple_atr = true_ranges.iter().sum::<f64>() / true_ranges.len() as f64;

    Ok(simple_atr.max(min_baseline))
}

/// Calculate volatility trend change (similar to direction's momentum change approach)
///
/// This function analyzes how volatility is changing over time, capturing volatility
/// acceleration/deceleration patterns that are crucial for LSTM learning.
///
/// ## Algorithm
/// 1. Calculate volatility trend in sequence period (using ATR slope)
/// 2. Calculate volatility trend in horizon period
/// 3. Return the change in volatility trend (acceleration/deceleration)
///
/// ## Returns
/// - Positive: Volatility is accelerating (trend becoming more volatile)
/// - Negative: Volatility is decelerating (trend becoming less volatile)
/// - Zero: Volatility trend is stable
pub fn calculate_volatility_trend_change(
    sequence_candles: &[MarketDataRow],
    horizon_candles: &[MarketDataRow],
) -> Result<f64> {
    if sequence_candles.len() < 3 || horizon_candles.len() < 3 {
        return Ok(0.0); // No trend change for insufficient data
    }

    // Calculate rolling ATR values for trend analysis
    let sequence_atr_values = calculate_rolling_atr_values(sequence_candles)?;
    let horizon_atr_values = calculate_rolling_atr_values(horizon_candles)?;

    if sequence_atr_values.len() < 2 || horizon_atr_values.len() < 2 {
        return Ok(0.0);
    }

    // Calculate volatility trend (slope) for each period
    let sequence_trend = calculate_atr_trend_slope(&sequence_atr_values)?;
    let horizon_trend = calculate_atr_trend_slope(&horizon_atr_values)?;

    // Return trend change (similar to direction's momentum change)
    Ok(horizon_trend - sequence_trend)
}

/// Calculate volume-weighted volatility change (similar to price_level's VWAP approach)
///
/// This function weights volatility by volume to distinguish between high-volume
/// volatility (meaningful) and low-volume volatility (noise).
///
/// ## Algorithm
/// 1. Calculate volume-weighted ATR for sequence period
/// 2. Calculate volume-weighted ATR for horizon period
/// 3. Return the ratio change with volume weighting
pub fn calculate_volume_weighted_volatility_change(
    sequence_candles: &[MarketDataRow],
    horizon_candles: &[MarketDataRow],
) -> Result<f64> {
    if sequence_candles.len() < 2 || horizon_candles.len() < 2 {
        return Ok(1.0); // Neutral change for insufficient data
    }

    // Use a default min_baseline for these helper functions
    let min_baseline = 0.005;
    let sequence_vw_atr = calculate_volume_weighted_atr(sequence_candles, min_baseline)?;
    let horizon_vw_atr = calculate_volume_weighted_atr(horizon_candles, min_baseline)?;

    if sequence_vw_atr <= 0.0 {
        return Ok(1.0);
    }

    Ok(horizon_vw_atr / sequence_vw_atr)
}

/// Calculate volatility persistence (regime stability measure)
///
/// This function measures how stable the volatility regime is, helping to
/// distinguish between temporary volatility spikes and regime changes.
///
/// ## Algorithm
/// 1. Calculate volatility consistency within sequence
/// 2. Calculate volatility consistency within horizon
/// 3. Return persistence score (higher = more stable regime)
pub fn calculate_volatility_persistence(
    sequence_candles: &[MarketDataRow],
    horizon_candles: &[MarketDataRow],
) -> Result<f64> {
    if sequence_candles.len() < 3 || horizon_candles.len() < 3 {
        return Ok(0.5); // Neutral persistence for insufficient data
    }

    let sequence_atr_values = calculate_rolling_atr_values(sequence_candles)?;
    let horizon_atr_values = calculate_rolling_atr_values(horizon_candles)?;

    let sequence_consistency = calculate_atr_consistency(&sequence_atr_values);
    let horizon_consistency = calculate_atr_consistency(&horizon_atr_values);

    // Return average consistency as persistence score
    Ok((sequence_consistency + horizon_consistency) / 2.0)
}

/// Combine multiple volatility features into enhanced score
///
/// This function combines all volatility features similar to how direction
/// and price_level targets use multiple features for robust classification.
pub fn combine_volatility_features(
    volatility_ratio: f64,
    trend_change: f64,
    volume_weighted_change: f64,
    persistence: f64,
) -> f64 {
    // Weight the features (can be tuned based on performance)
    let ratio_weight = 0.4; // Basic ratio (existing approach)
    let trend_weight = 0.3; // Trend change (new, similar to direction)
    let volume_weight = 0.2; // Volume weighting (new, similar to price_level)
    let persistence_weight = 0.1; // Persistence (new, regime stability)

    // Normalize trend_change and combine
    let normalized_trend = (trend_change * 10.0).tanh(); // Normalize to [-1, 1] range
    let normalized_volume = volume_weighted_change.ln(); // Log space for ratios

    // Combine features into enhanced score
    let base_score = volatility_ratio.ln(); // Log space for ratios
    let trend_component = normalized_trend * trend_weight / ratio_weight;
    let volume_component = normalized_volume * volume_weight / ratio_weight;
    let persistence_component = (persistence - 0.5) * persistence_weight / ratio_weight;

    let enhanced_score = base_score + trend_component + volume_component + persistence_component;

    // Convert back to ratio space
    enhanced_score.exp()
}

/// Classify enhanced volatility score using adaptive thresholds
pub fn classify_enhanced_volatility_score(
    enhanced_score: f64,
    moderate_threshold: f64,
    extreme_threshold: f64,
) -> i32 {
    if enhanced_score <= (1.0 - extreme_threshold) {
        0 // VeryLow
    } else if enhanced_score <= (1.0 - moderate_threshold) {
        1 // Low
    } else if enhanced_score < (1.0 + moderate_threshold) {
        2 // Medium
    } else if enhanced_score < (1.0 + extreme_threshold) {
        3 // High
    } else {
        4 // VeryHigh
    }
}

/// Helper: Calculate rolling ATR values for trend analysis
fn calculate_rolling_atr_values(candles: &[MarketDataRow]) -> Result<Vec<f64>> {
    if candles.len() < 2 {
        return Ok(vec![]);
    }

    let mut atr_values = Vec::new();

    // Calculate ATR for each possible window
    for i in 1..candles.len() {
        let window_candles = &candles[0..=i];
        if let Ok(atr) = calculate_simple_atr(window_candles) {
            atr_values.push(atr);
        }
    }

    Ok(atr_values)
}

/// Helper: Calculate trend slope from ATR values
fn calculate_atr_trend_slope(atr_values: &[f64]) -> Result<f64> {
    if atr_values.len() < 2 {
        return Ok(0.0);
    }

    let n = atr_values.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;

    for (i, &atr) in atr_values.iter().enumerate() {
        let x = i as f64;
        sum_x += x;
        sum_y += atr;
        sum_xy += x * atr;
        sum_x2 += x * x;
    }

    let denominator = n * sum_x2 - sum_x * sum_x;
    if denominator.abs() < 1e-10 {
        return Ok(0.0);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    Ok(slope)
}

/// Helper: Calculate volume-weighted ATR with configurable minimum baseline
fn calculate_volume_weighted_atr(candles: &[MarketDataRow], min_baseline: f64) -> Result<f64> {
    if candles.len() < 2 {
        return Ok(min_baseline); // Use calibrated minimum
    }

    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for i in 1..candles.len() {
        let current = &candles[i];
        let previous = &candles[i - 1];

        let high_low = current.high - current.low;
        let high_close = (current.high - previous.close).abs();
        let low_close = (current.low - previous.close).abs();
        let true_range = high_low.max(high_close).max(low_close);

        if true_range.is_finite() && true_range > 0.0 && current.close > 0.0 {
            let normalized_tr = true_range / current.close;
            let weight = current.volume; // Volume weighting

            weighted_sum += normalized_tr * weight;
            total_weight += weight;
        }
    }

    if total_weight <= 0.0 {
        return calculate_simple_atr_with_params(candles, min_baseline);
    }

    Ok((weighted_sum / total_weight).max(min_baseline))
}

/// Helper: Calculate ATR consistency (lower = more consistent)
fn calculate_atr_consistency(atr_values: &[f64]) -> f64 {
    if atr_values.len() < 2 {
        return 0.5; // Neutral consistency
    }

    let mean = atr_values.iter().sum::<f64>() / atr_values.len() as f64;
    let variance =
        atr_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / atr_values.len() as f64;

    let coefficient_of_variation = if mean > 0.0 {
        variance.sqrt() / mean
    } else {
        1.0
    };

    // Convert to consistency score (higher = more consistent)
    (1.0 / (1.0 + coefficient_of_variation)).clamp(0.0, 1.0)
}

/// Calculate enhanced score midpoints for each volatility class using proper mathematical boundaries
///
/// This function calculates the actual enhanced score boundaries for each class and finds
/// the mathematical midpoints, eliminating magic numbers and making it fully adaptive.
///
/// # Arguments
/// * `sequence_ohlcv` - Sequence OHLCV data for feature calculation
/// * `adaptive_params` - Adaptive parameters for threshold calculation
/// * `moderate_threshold` - Moderate threshold for classification
/// * `extreme_threshold` - Extreme threshold for classification
///
/// # Returns
/// * `Vec<f64>` - Enhanced score midpoints for each class [VeryLow, Low, Medium, High, VeryHigh]
fn calculate_enhanced_score_midpoints(
    sequence_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::VolatilityParams,
    moderate_threshold: f64,
    extreme_threshold: f64,
) -> Result<Vec<f64>> {
    // Calculate sequence-based features (available during prediction)
    let sequence_atr = calculate_simple_atr_with_params(
        sequence_ohlcv,
        calibrated_params.min_volatility_baseline,
    )?;
    let baseline_atr = sequence_atr.max(calibrated_params.min_volatility_baseline);

    // Calculate sequence-only features for neutral approximation
    let sequence_trend = calculate_sequence_volatility_trend(sequence_ohlcv)?;
    let sequence_volume_weighted =
        calculate_volume_weighted_atr(sequence_ohlcv, calibrated_params.min_volatility_baseline)?;
    let sequence_persistence = calculate_sequence_volatility_persistence(sequence_ohlcv)?;

    // Calculate actual enhanced score boundaries (same as classify_enhanced_volatility_score)
    let boundary_ratios = [
        1.0 - extreme_threshold,  // VeryLow/Low boundary
        1.0 - moderate_threshold, // Low/Medium boundary
        1.0 + moderate_threshold, // Medium/High boundary
        1.0 + extreme_threshold,  // High/VeryHigh boundary
    ];

    // Calculate enhanced scores for boundaries
    let mut boundary_enhanced_scores = Vec::new();
    for &ratio in &boundary_ratios {
        // Use neutral features for boundary calculation (no magic scaling)
        let trend_change = sequence_trend * (ratio - 1.0); // Proportional to ratio change
        let volume_change = ratio * (sequence_volume_weighted / baseline_atr);
        let persistence = sequence_persistence;

        let enhanced_score =
            combine_volatility_features(ratio, trend_change, volume_change, persistence);
        boundary_enhanced_scores.push(enhanced_score);
    }

    // Calculate mathematical midpoints between boundaries
    let enhanced_score_midpoints = vec![
        // VeryLow: midpoint between 0 and first boundary
        boundary_enhanced_scores[0] * 0.5,
        // Low: midpoint between first and second boundary
        (boundary_enhanced_scores[0] + boundary_enhanced_scores[1]) * 0.5,
        // Medium: midpoint between second and third boundary (around 1.0)
        (boundary_enhanced_scores[1] + boundary_enhanced_scores[2]) * 0.5,
        // High: midpoint between third and fourth boundary
        (boundary_enhanced_scores[2] + boundary_enhanced_scores[3]) * 0.5,
        // VeryHigh: extrapolate beyond fourth boundary using same spacing as High class
        boundary_enhanced_scores[3] + (boundary_enhanced_scores[3] - boundary_enhanced_scores[2]),
    ];

    Ok(enhanced_score_midpoints)
}

/// Calculate volatility trend from sequence data only (for prediction)
///
/// This approximates the trend change calculation using only sequence data
/// by analyzing the volatility trend within the sequence itself.
fn calculate_sequence_volatility_trend(sequence_ohlcv: &[MarketDataRow]) -> Result<f64> {
    if sequence_ohlcv.len() < 3 {
        return Ok(0.0);
    }

    let atr_values = calculate_rolling_atr_values(sequence_ohlcv)?;
    if atr_values.len() < 2 {
        return Ok(0.0);
    }

    // Calculate trend slope within sequence
    calculate_atr_trend_slope(&atr_values)
}

/// Calculate volatility persistence from sequence data only (for prediction)
///
/// This approximates the persistence calculation using only sequence data
/// by analyzing the consistency within the sequence itself.
fn calculate_sequence_volatility_persistence(sequence_ohlcv: &[MarketDataRow]) -> Result<f64> {
    if sequence_ohlcv.len() < 3 {
        return Ok(0.5);
    }

    let atr_values = calculate_rolling_atr_values(sequence_ohlcv)?;
    if atr_values.is_empty() {
        return Ok(0.5);
    }

    Ok(calculate_atr_consistency(&atr_values))
}

/// Log volatility class distribution with logarithmic ratio analysis
fn log_volatility_distribution(targets: &[i32], horizon: &str) {
    let class_names = ["VeryLow", "Low", "Medium", "High", "VeryHigh"];
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
            "📊 Log Ratio Volatility Analysis [{}]: No valid targets found",
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

    // Calculate imbalance ratio
    let min_class_size = class_counts.iter().filter(|&&c| c > 0).min().unwrap_or(&0);
    let max_class_size = class_counts.iter().max().unwrap_or(&0);
    let imbalance_ratio = if *min_class_size > 0 {
        *max_class_size as f64 / *min_class_size as f64
    } else {
        f64::INFINITY
    };

    log::info!(
        "📊 Log Ratio Volatility Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}] (BEFORE balanced selection)",
        horizon,
        valid_targets,
        imbalance_ratio,
        class_percentages.join(", ")
    );
}

// ============================================================================
// PREDICTION RECONSTRUCTION METHODS
// ============================================================================

/// Reconstruction result for volatility predictions
#[derive(Debug, Clone)]
pub struct VolatilityReconstruction {
    /// Class probabilities from model [VeryLow, Low, Medium, High, VeryHigh]
    pub probabilities: Vec<f64>,
    /// Expected ATR ratios for each class (target_atr / train_atr)
    pub atr_ratios: Vec<f64>,
    /// Expected volatility change percentages for each class
    pub volatility_changes: Vec<f64>,
    /// Most likely volatility regime class index
    pub most_likely_class: usize,
    /// Confidence (probability of most likely class)
    pub confidence: f64,
    /// Expected ATR ratio (weighted average)
    pub expected_atr_ratio: f64,
    /// Expected volatility change percentage (weighted average)
    pub expected_volatility_change: f64,
    /// Low volatility probability (VeryLow + Low)
    pub low_volatility_probability: f64,
    /// High volatility probability (High + VeryHigh)
    pub high_volatility_probability: f64,
    /// Extreme volatility probability (VeryLow + VeryHigh)
    pub extreme_volatility_probability: f64,
    /// Log space thresholds used for classification
    pub log_thresholds: LogVolatilityThresholds,
    /// Training ATR baseline for ratio calculations
    pub train_atr: f64,
}

/// Reconstruct volatility predictions from model probabilities using enhanced multi-feature approach
///
/// This method reverses the enhanced classification logic to convert raw model probabilities
/// back to meaningful volatility metrics. Uses the same 4-feature approach as training:
/// 1. Basic ATR ratio, 2. Trend change, 3. Volume-weighted change, 4. Persistence
///
/// # Arguments
/// * `probabilities` - 5-element array of class probabilities [VeryLow, Low, Medium, High, VeryHigh]
/// * `sequence_ohlcv` - OHLCV data for the input sequence (same as used in training)
/// * `adaptive_params` - Adaptive parameters used during training (for threshold calculation)
///
/// # Returns
/// * `VolatilityReconstruction` - Complete reconstruction with enhanced volatility metrics
pub fn reconstruct_volatility(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::VolatilityParams,
) -> Result<VolatilityReconstruction> {
    // Validate inputs
    if probabilities.len() != 5 {
        return Err(crate::utils::error::VangaError::DataError(
            "Volatility reconstruction requires exactly 5 class probabilities".to_string(),
        ));
    }

    if sequence_ohlcv.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Sequence OHLCV data is required for volatility reconstruction".to_string(),
        ));
    }

    // Calculate sequence ATR baseline (same as training)
    let sequence_atr = calculate_simple_atr_with_params(
        sequence_ohlcv,
        calibrated_params.min_volatility_baseline,
    )?;
    let baseline_atr = sequence_atr.max(calibrated_params.min_volatility_baseline); // Same minimum as training

    if baseline_atr <= 0.0 {
        return Err(crate::utils::error::VangaError::DataError(
            "Invalid sequence ATR baseline for volatility reconstruction".to_string(),
        ));
    }

    // Use calibrated parameters for threshold calculation (same as training)
    let moderate_threshold = calibrated_params.bandwidth;
    let extreme_threshold = calibrated_params.bandwidth * calibrated_params.extreme_multiplier;

    // Calculate enhanced score midpoints for each class using same logic as training
    // These represent the enhanced scores that would classify to each class
    let enhanced_score_midpoints = calculate_enhanced_score_midpoints(
        sequence_ohlcv,
        calibrated_params,
        moderate_threshold,
        extreme_threshold,
    )?;

    // Convert enhanced scores back to basic ATR ratios for compatibility
    let class_ratio_midpoints: Vec<f64> = enhanced_score_midpoints.clone();

    // Convert enhanced scores to actual ATR ratios for compatibility
    let atr_ratios: Vec<f64> = class_ratio_midpoints.clone();

    // Convert ATR ratios to volatility change percentages
    let volatility_changes: Vec<f64> = atr_ratios
        .iter()
        .map(|&ratio| (ratio - 1.0) * 100.0) // Convert to percentage change
        .collect();

    // Find most likely class
    let most_likely_class = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(2); // Default to medium

    let confidence = probabilities[most_likely_class];

    // Calculate expected ATR ratio (weighted average)
    let expected_atr_ratio = probabilities
        .iter()
        .zip(class_ratio_midpoints.iter())
        .map(|(prob, ratio)| prob * ratio)
        .sum::<f64>();

    // Calculate expected volatility change percentage
    let expected_volatility_change = (expected_atr_ratio - 1.0) * 100.0;

    // Calculate volatility regime probabilities
    let low_volatility_probability = probabilities[0] + probabilities[1]; // VeryLow + Low
    let high_volatility_probability = probabilities[3] + probabilities[4]; // High + VeryHigh
    let extreme_volatility_probability = probabilities[0] + probabilities[4]; // VeryLow + VeryHigh

    // Create simplified thresholds structure for compatibility
    let log_thresholds = LogVolatilityThresholds {
        very_low_max: (1.0 - extreme_threshold).ln(),
        low_max: (1.0 - moderate_threshold).ln(),
        medium_max: (1.0 + moderate_threshold).ln(),
        high_max: (1.0 + extreme_threshold).ln(),
    };

    Ok(VolatilityReconstruction {
        probabilities: probabilities.to_vec(),
        atr_ratios,
        volatility_changes,
        most_likely_class,
        confidence,
        expected_atr_ratio,
        expected_volatility_change,
        low_volatility_probability,
        high_volatility_probability,
        extreme_volatility_probability,
        log_thresholds,
        train_atr: baseline_atr,
    })
}

/// Calculate average volume for volume conviction factor
///
/// SIMPLE APPROACH: Returns average volume for volume conviction calculation
/// Higher volume = higher conviction in volatility moves
pub fn calculate_average_volume(candles: &[MarketDataRow]) -> f64 {
    if candles.is_empty() {
        return 0.0; // No volume if no data
    }

    // Calculate average volume
    candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64
}

/// Get volatility regime class names in order
pub fn get_volatility_class_names() -> Vec<&'static str> {
    vec!["VeryLow", "Low", "Medium", "High", "VeryHigh"]
}
