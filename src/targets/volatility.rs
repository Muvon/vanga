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

use crate::config::model::VolatilityHead;
use crate::data::structures::MarketDataRow;
use crate::utils::error::Result;
use crate::utils::market_data::extract_ohlcv_data;
use crate::utils::parser::parse_horizon_to_steps;
use polars::prelude::*;
use std::collections::HashMap;

/// Generate volatility targets for multiple horizons using logarithmic ratio approach
///
/// This is the main function that generates volatility regime classifications for
/// cryptocurrency market data using ATR-based logarithmic ratio analysis.
///
/// ## Algorithm
/// 1. **Extract OHLCV Data**: Get Open, High, Low, Close, Volume from DataFrame
/// 2. **For Each Sequence Position**:
///    - Calculate `train_atr`: ATR from input sequence (baseline volatility)
///    - Calculate `target_atr`: ATR from horizon period (volatility to classify)
///    - Compute `log_ratio = ln(target_atr / train_atr)` for symmetric classification
///    - Apply bandwidth_size-based thresholds to classify volatility regime
///
/// ## Parameters
/// - `df`: Market data DataFrame with OHLCV columns
/// - `horizons`: Prediction horizons (e.g., ["1h", "4h", "1d"])
/// - `model_config`: Optional VolatilityHead configuration
/// - `sequence_indices`: Starting indices for each sequence
/// - `sequence_length`: Length of input sequences for ATR calculation
///
/// ## Returns
/// HashMap mapping horizon strings to vectors of volatility class integers:
/// - 0: VeryLow (extreme volatility decrease)
/// - 1: Low (moderate volatility decrease)
/// - 2: Medium (similar volatility)
/// - 3: High (moderate volatility increase)
/// - 4: VeryHigh (extreme volatility increase)
///
/// ## Configuration
/// Uses `bandwidth_size` from model_config to control regime sensitivity:
/// - Default: 0.4 (balanced regime detection)
/// - Lower values: More sensitive to volatility changes
/// - Higher values: Less sensitive, only major regime shifts
///
/// ## Mathematical Foundation
/// The logarithmic approach ensures symmetric treatment of volatility ratios:
/// - 2x volatility increase: ln(2.0) = +0.693
/// - 0.5x volatility decrease: ln(0.5) = -0.693 (perfectly symmetric)
///   This prevents bias toward volatility increases in linear space.
pub fn generate_volatility_targets(
    df: &DataFrame,
    horizons: &[String],
    model_config: Option<&VolatilityHead>,
    sequence_indices: &[usize],
    sequence_length: usize,
) -> Result<HashMap<String, Vec<i32>>> {
    let ohlcv_data = extract_ohlcv_data(df)?;
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let mut horizon_targets = vec![-1; sequence_indices.len()];

        for (seq_position, &seq_idx) in sequence_indices.iter().enumerate() {
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            // Check boundaries - need both sequence and horizon data
            if target_end_idx <= ohlcv_data.len() && sequence_end_idx <= ohlcv_data.len() {
                // Get INPUT sequence candles (for adaptive thresholds)
                let sequence_candles = &ohlcv_data[seq_idx..sequence_end_idx];

                // Get HORIZON candles (from sequence end to target horizon)
                let horizon_candles = &ohlcv_data[sequence_end_idx..target_end_idx];

                // Get ATR values for train (sequence) and target (horizon)
                let train_atr = get_sequence_atr_baseline(sequence_candles)?;
                let target_atr = get_sequence_atr_baseline(horizon_candles)?;

                // Calculate logarithmic ratio thresholds
                let bandwidth_size = model_config.and_then(|c| c.bandwidth_size).unwrap_or(0.4);
                let extreme_multiplier = model_config
                    .and_then(|c| c.extreme_multiplier)
                    .unwrap_or(2.0);

                let log_thresholds =
                    calculate_log_volatility_thresholds(bandwidth_size, extreme_multiplier)?;

                // Classify using logarithmic ratio approach
                let volatility_class =
                    classify_volatility_log_ratio(train_atr, target_atr, &log_thresholds);
                horizon_targets[seq_position] = volatility_class;
            }
        }

        log_volatility_distribution(&horizon_targets, horizon);
        targets.insert(horizon.clone(), horizon_targets);
    }

    Ok(targets)
}

/// Calculate ATR baseline for a sequence of candles
fn get_sequence_atr_baseline(sequence_candles: &[MarketDataRow]) -> Result<f64> {
    if sequence_candles.len() < 2 {
        return Ok(0.02); // Minimal fallback
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
        return Ok(0.02);
    }

    // Average True Range of the sequence - this is our baseline
    Ok(true_ranges.iter().sum::<f64>() / true_ranges.len() as f64)
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
#[derive(Debug)]
struct LogVolatilityThresholds {
    very_low_max: f64, // VeryLow threshold (log space)
    low_max: f64,      // Low threshold (log space)
    medium_max: f64,   // Medium threshold (log space)
    high_max: f64,     // High threshold (log space)
                       // Above high_max = VeryHigh
}

/// Calculate logarithmic ratio thresholds for volatility classification
///
/// This function computes the threshold boundaries used to classify volatility ratios
/// into the 5-class volatility regime system (VeryLow, Low, Medium, High, VeryHigh).
/// Uses logarithmic space for mathematically symmetric ratio classification.
///
/// ## Parameters
/// - `bandwidth_size`: Controls threshold sensitivity in log space (default: 0.4)
///   - Lower values = more sensitive (tighter thresholds)
///   - Higher values = less sensitive (wider thresholds)
/// - `extreme_multiplier`: Multiplier for extreme boundaries (default: 2.0)
///   - Controls the ratio between moderate and extreme classifications
///
/// ## Logarithmic Threshold Logic
/// ```text
/// half_bandwidth = bandwidth_size / 2.0
/// extreme_bandwidth = bandwidth_size * extreme_multiplier
///
/// Classification boundaries (in log space):
/// VeryLow:  log_ratio <= -extreme_bandwidth
/// Low:      -extreme_bandwidth < log_ratio <= -half_bandwidth
/// Medium:   -half_bandwidth < log_ratio <= +half_bandwidth
/// High:     +half_bandwidth < log_ratio <= +extreme_bandwidth
/// VeryHigh: log_ratio > +extreme_bandwidth
/// ```
///
/// ## Ratio Space Interpretation
/// With bandwidth_size=0.4 and extreme_multiplier=2.0:
/// - VeryLow: target_atr ≤ 0.45 × train_atr (55%+ volatility decrease)
/// - Low: 0.45 × train_atr < target_atr ≤ 0.82 × train_atr (18-55% decrease)
/// - Medium: 0.82 × train_atr < target_atr ≤ 1.22 × train_atr (±18% change)
/// - High: 1.22 × train_atr < target_atr ≤ 2.23 × train_atr (22-123% increase)
/// - VeryHigh: target_atr > 2.23 × train_atr (123%+ volatility increase)
///
/// ## Mathematical Symmetry
/// The logarithmic approach ensures that multiplicative changes are treated symmetrically:
/// - 2x increase: ln(2.0) = +0.693
/// - 0.5x decrease: ln(0.5) = -0.693 (perfectly symmetric)
fn calculate_log_volatility_thresholds(
    bandwidth_size: f64,
    extreme_multiplier: f64,
) -> Result<LogVolatilityThresholds> {
    let half_bandwidth = bandwidth_size / 2.0;
    let extreme_bandwidth = bandwidth_size * extreme_multiplier;

    let thresholds = LogVolatilityThresholds {
        very_low_max: -extreme_bandwidth, // Most negative in log space
        low_max: -half_bandwidth,         // Negative side of medium
        medium_max: half_bandwidth,       // Positive side of medium
        high_max: extreme_bandwidth,      // Most positive before very high
    };

    // Convert log thresholds back to ratio ranges for logging
    let very_low_ratio = (-extreme_bandwidth).exp();
    let low_ratio = (-half_bandwidth).exp();
    let medium_high_ratio = half_bandwidth.exp();
    let high_ratio = extreme_bandwidth.exp();

    log::debug!(
        "🎯 Log Volatility Thresholds: bandwidth_size={}, extreme_factor={}, log_thresholds=[{:.4}, {:.4}, {:.4}, {:.4}], ratio_ranges=[{:.3}, {:.3}, {:.3}, {:.3}]",
        bandwidth_size, extreme_multiplier,
        thresholds.very_low_max, thresholds.low_max, thresholds.medium_max, thresholds.high_max,
        very_low_ratio, low_ratio, medium_high_ratio, high_ratio
    );

    Ok(thresholds)
}

/// Classify volatility using logarithmic ratio approach
fn classify_volatility_log_ratio(
    train_atr: f64,
    target_atr: f64,
    thresholds: &LogVolatilityThresholds,
) -> i32 {
    // Handle edge cases
    if train_atr <= 0.0 || target_atr <= 0.0 {
        return 2; // Default to medium for invalid ATR values
    }

    // Calculate log ratio (symmetric around 0)
    let log_ratio = (target_atr / train_atr).ln();

    // Classify using log space thresholds
    if log_ratio <= thresholds.very_low_max {
        0 // VeryLow
    } else if log_ratio <= thresholds.low_max {
        1 // Low
    } else if log_ratio <= thresholds.medium_max {
        2 // Medium (balanced around ln(1.0) = 0)
    } else if log_ratio <= thresholds.high_max {
        3 // High
    } else {
        4 // VeryHigh
    }
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
        "📊 Log Ratio Volatility Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}]",
        horizon,
        valid_targets,
        imbalance_ratio,
        class_percentages.join(", ")
    );
}
