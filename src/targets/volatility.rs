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

use crate::config::model::TargetsConfig;
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
    targets_config: &TargetsConfig, // Use new unified config
    sequence_indices: &[usize],
    sequence_length: usize,
) -> Result<HashMap<String, Vec<i32>>> {
    let ohlcv_data = extract_ohlcv_data(df)?;
    let mut targets = HashMap::new();

    // ALWAYS ADAPTIVE: Auto-calibrate bandwidth using unified config
    let first_horizon_steps = parse_horizon_to_steps(&horizons[0])?;
    let calibrated_bandwidth = calibrate_volatility_bandwidth(
        &ohlcv_data,
        sequence_length,
        first_horizon_steps,
        targets_config.balance_target,
    )?;

    log::info!(
        "🎯 Volatility targets using calibrated bandwidth: {:.6} (was base: {:.6})",
        calibrated_bandwidth,
        targets_config.base_sensitivity
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
                    // Create adaptive config with calibrated bandwidth
                    let adaptive_targets_config = TargetsConfig {
                        base_sensitivity: calibrated_bandwidth,
                        balance_target: targets_config.balance_target,
                        momentum_weighting: targets_config.momentum_weighting,
                        extreme_multiplier: targets_config.extreme_multiplier,
                    };

                    // Use enhanced classification with calibrated bandwidth
                    let volatility_class = classify_volatility_with_distribution_analysis(
                        sequence_candles,
                        horizon_candles,
                        &adaptive_targets_config,
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
        let window_atr = get_sequence_atr_baseline(window_candles)?;

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
        atr_series.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        atr_series.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
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

    AtrDistributionStats {
        mean,
        std_dev,
        median,
        percentile_25,
        percentile_75,
    }
}

/// Statistical distribution metrics for ATR series
#[derive(Debug, Clone)]
pub struct AtrDistributionStats {
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub percentile_25: f64,
    pub percentile_75: f64,
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
/// Computes Average True Range for the sequence, using sequence-specific baseline
/// instead of hardcoded values. Provides adaptive fallback based on sequence price volatility.
pub fn get_sequence_atr_baseline(sequence_candles: &[MarketDataRow]) -> Result<f64> {
    if sequence_candles.len() < 2 {
        // ADAPTIVE FALLBACK: Use 0.5% of first candle's close price as minimum baseline
        let fallback_atr = if !sequence_candles.is_empty() {
            sequence_candles[0].close * 0.005 // 0.5% minimum volatility assumption
        } else {
            0.005 // Absolute minimum for edge cases
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
        return Ok(price_range.max(0.005)); // At least 0.5% volatility
    }

    // Average True Range of the sequence - this is our adaptive baseline
    let sequence_atr = true_ranges.iter().sum::<f64>() / true_ranges.len() as f64;

    log::trace!(
        "🎯 Sequence ATR Baseline: {} candles, {} true_ranges, atr={:.6}",
        sequence_candles.len(),
        true_ranges.len(),
        sequence_atr
    );

    Ok(sequence_atr.max(0.005)) // Ensure minimum 0.5% volatility baseline
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

/// Classify volatility using sequence-adaptive distribution analysis
///
/// This is the enhanced classification function that provides superior volatility
/// regime detection through ATR distribution analysis and adaptive thresholds.
/// Designed for production use with balanced class distribution across market conditions.
///
/// ## Key Features
/// - **ATR Distribution Analysis**: Analyzes rolling ATR patterns for adaptive thresholds
/// - **Sequence-Adaptive Bandwidth**: Automatically adjusts to volatility characteristics
/// - **Logarithmic Symmetry**: Maintains mathematical consistency in ratio space
/// - **Balanced Classification**: Targets ~20% distribution per class
///
/// ## Algorithm
/// 1. Calculate rolling ATR series from sequence data
/// 2. Analyze ATR distribution statistics
/// 3. Calculate adaptive bandwidth based on distribution characteristics
/// 4. Apply logarithmic ratio classification with adaptive thresholds
/// 5. Return balanced volatility regime classification
///
/// ## Parameters
/// - `sequence_candles`: Input sequence OHLCV data for baseline analysis
/// - `horizon_candles`: Horizon period OHLCV data for classification
/// - `targets_config`: Unified targets configuration for adaptive thresholds
///
/// ## Returns
/// Volatility class [0-4]: VeryLow, Low, Medium, High, VeryHigh
pub fn classify_volatility_with_distribution_analysis(
    sequence_candles: &[MarketDataRow],
    horizon_candles: &[MarketDataRow],
    targets_config: &TargetsConfig,
) -> Result<i32> {
    if sequence_candles.len() < 2 || horizon_candles.len() < 2 {
        return Ok(2); // Default to Medium for insufficient data
    }

    // Step 1: Calculate rolling ATR series for distribution analysis
    let rolling_window = (sequence_candles.len() / 3).clamp(3, 10); // Adaptive window size
    let atr_series = calculate_rolling_atr_series(sequence_candles, rolling_window)?;

    // Step 2: Calculate sequence baseline ATR (mean of distribution)
    let sequence_atr_stats = calculate_atr_distribution_stats(&atr_series);
    let baseline_atr = sequence_atr_stats.mean;

    // Step 3: Calculate horizon ATR for comparison
    let horizon_atr = get_sequence_atr_baseline(horizon_candles)?;

    // Step 4: Calculate adaptive thresholds using calibrated sensitivity
    let log_thresholds = calculate_log_volatility_thresholds(targets_config)?;

    // Step 5: Classify using enhanced logarithmic ratio approach
    let volatility_class =
        classify_volatility_log_ratio(baseline_atr, horizon_atr, &log_thresholds);

    log::debug!(
        "🎯 Volatility Distribution Analysis: seq_atr={:.6}, hor_atr={:.6}, calibrated_sensitivity={:.4}, class={}",
        baseline_atr, horizon_atr, targets_config.base_sensitivity, volatility_class
    );

    Ok(volatility_class)
}

/// Calibrate volatility bandwidth based on actual market data distribution
///
/// This function analyzes the distribution of ATR ratios in the data to automatically
/// determine appropriate bandwidth_size thresholds that will produce balanced
/// class distributions across different market conditions.
///
/// ## Algorithm
/// 1. Calculate ATR ratios for all sequences and horizons in the data
/// 2. Compute log ratio distribution
/// 3. Use percentiles to set balanced thresholds
/// 4. Return calibrated bandwidth_size value
///
/// ## Parameters
/// - `ohlcv_data`: Market data for calibration analysis
/// - `sequence_length`: Length of input sequences
/// - `horizon_steps`: Horizon period length
/// - `target_balance`: Target percentage for extreme classes (e.g., 0.15 for 15%)
///
/// ## Returns
/// Calibrated bandwidth_size optimized for balanced class distribution
pub fn calibrate_volatility_bandwidth(
    ohlcv_data: &[MarketDataRow],
    sequence_length: usize,
    horizon_steps: usize,
    target_balance: f64,
) -> Result<f64> {
    if ohlcv_data.len() < sequence_length + horizon_steps + 10 {
        return Ok(0.2); // Default fallback for insufficient data
    }

    let mut log_ratios = Vec::new();

    // Sample ATR ratios from the data
    for i in 0..(ohlcv_data.len() - sequence_length - horizon_steps) {
        let sequence_candles = &ohlcv_data[i..i + sequence_length];
        let horizon_candles = &ohlcv_data[i + sequence_length..i + sequence_length + horizon_steps];

        if sequence_candles.len() >= 2 && horizon_candles.len() >= 2 {
            let seq_atr = get_sequence_atr_baseline(sequence_candles)?;
            let hor_atr = get_sequence_atr_baseline(horizon_candles)?;

            if seq_atr > 0.0 && hor_atr > 0.0 {
                let atr_ratio = hor_atr / seq_atr;
                let log_ratio = atr_ratio.ln();

                if log_ratio.is_finite() {
                    log_ratios.push(log_ratio.abs()); // Use absolute values for threshold calculation
                }
            }
        }
    }

    if log_ratios.is_empty() {
        return Ok(0.2); // Default fallback
    }

    // Sort log ratios to find percentiles
    log_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = log_ratios.len();

    // Find the percentile that corresponds to target_balance for extreme classes
    // We want target_balance% in each extreme class, so (1.0 - 2*target_balance) in middle classes
    let extreme_percentile = 1.0 - target_balance;
    let extreme_idx = ((n as f64) * extreme_percentile) as usize;
    let extreme_threshold = log_ratios[extreme_idx.min(n - 1)];

    // The bandwidth_size should be set so that extreme_threshold becomes the extreme boundary
    // With extreme_multiplier = 2.0: extreme_boundary = bandwidth_size * 2.0
    // So: bandwidth_size = extreme_threshold / 2.0
    let calibrated_bandwidth = extreme_threshold / 2.0;

    // Ensure reasonable bounds
    let final_bandwidth = calibrated_bandwidth.clamp(0.05, 1.0);

    log::info!(
        "🎯 Calibrated volatility bandwidth: {:.6} (from {} samples, extreme_threshold: {:.6})",
        final_bandwidth,
        n,
        extreme_threshold
    );

    Ok(final_bandwidth)
}

/// Calculate ATR baseline for a sequence of candles with adaptive fallback
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
pub fn calculate_log_volatility_thresholds(
    targets_config: &TargetsConfig,
) -> Result<LogVolatilityThresholds> {
    let half_bandwidth = targets_config.base_sensitivity / 2.0;
    let extreme_bandwidth = targets_config.base_sensitivity * targets_config.extreme_multiplier;

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
        "🎯 Log Volatility Thresholds: base_sensitivity={}, extreme_multiplier={}, log_thresholds=[{:.4}, {:.4}, {:.4}, {:.4}], ratio_ranges=[{:.3}, {:.3}, {:.3}, {:.3}]",
        targets_config.base_sensitivity, targets_config.extreme_multiplier,
        thresholds.very_low_max, thresholds.low_max, thresholds.medium_max, thresholds.high_max,
        very_low_ratio, low_ratio, medium_high_ratio, high_ratio
    );

    Ok(thresholds)
}

/// Classify volatility using logarithmic ratio approach
pub fn classify_volatility_log_ratio(
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

/// Reconstruct volatility predictions from model probabilities
///
/// This method reverses the training classification logic to convert
/// raw model probabilities back to meaningful ATR ratios and volatility changes.
///
/// # Arguments
/// * `probabilities` - 5-element array of class probabilities [VeryLow, Low, Medium, High, VeryHigh]
/// * `sequence_ohlcv` - OHLCV data for the input sequence (same as used in training)
/// * `config` - Optional configuration (uses defaults if None)
///
/// # Returns
/// * `VolatilityReconstruction` - Complete reconstruction with ATR ratios and volatility metrics
pub fn reconstruct_volatility(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    config: Option<&TargetsConfig>,
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

    // Use same configuration as training
    let targets_config = config.cloned().unwrap_or_default();
    let log_thresholds = calculate_log_volatility_thresholds(&targets_config)?;

    // Calculate training ATR baseline (same as training)
    let train_atr = get_sequence_atr_baseline(sequence_ohlcv)?;
    if train_atr <= 0.0 {
        return Err(crate::utils::error::VangaError::DataError(
            "Invalid training ATR baseline for volatility reconstruction".to_string(),
        ));
    }

    // Calculate representative ATR ratios for each class (reverse of classification)
    // Use midpoints of log space boundaries, then convert to ratio space
    let log_midpoints = [
        log_thresholds.very_low_max - 0.2, // VeryLow: below very_low_max
        (log_thresholds.very_low_max + log_thresholds.low_max) / 2.0, // Low: between very_low_max and low_max
        (log_thresholds.low_max + log_thresholds.medium_max) / 2.0, // Medium: between low_max and medium_max
        (log_thresholds.medium_max + log_thresholds.high_max) / 2.0, // High: between medium_max and high_max
        log_thresholds.high_max + 0.2,                               // VeryHigh: above high_max
    ];

    // Convert log space midpoints to ATR ratios
    let atr_ratios: Vec<f64> = log_midpoints
        .iter()
        .map(|&log_val| log_val.exp()) // e^log_val = ratio
        .collect();

    // Convert ATR ratios to volatility change percentages
    let volatility_changes: Vec<f64> = atr_ratios
        .iter()
        .map(|&ratio| (ratio - 1.0) * 100.0) // (ratio - 1) * 100 = percentage change
        .collect();

    // Find most likely class and confidence
    let (most_likely_class, confidence) = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &prob)| (idx, prob))
        .unwrap_or((2, 0.2)); // Default to Medium

    // Calculate expected values (weighted averages)
    let expected_atr_ratio: f64 = probabilities
        .iter()
        .zip(atr_ratios.iter())
        .map(|(&prob, &ratio)| prob * ratio)
        .sum();

    let expected_volatility_change: f64 = probabilities
        .iter()
        .zip(volatility_changes.iter())
        .map(|(&prob, &change)| prob * change)
        .sum();

    // Calculate regime probabilities
    let low_volatility_probability = probabilities[0] + probabilities[1]; // VeryLow + Low
    let high_volatility_probability = probabilities[3] + probabilities[4]; // High + VeryHigh
    let extreme_volatility_probability = probabilities[0] + probabilities[4]; // VeryLow + VeryHigh

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
        train_atr,
    })
}

/// Convert class probabilities to expected ATR ratios
///
/// This method calculates the expected ATR ratio for each class based on
/// the same mathematical logic used in training target generation.
///
/// # Arguments
/// * `probabilities` - 5-element array of class probabilities
/// * `sequence_ohlcv` - OHLCV data for ATR baseline calculation
/// * `config` - Optional configuration
///
/// # Returns
/// * `Vec<f64>` - Expected ATR ratio for each class [VeryLow, Low, Medium, High, VeryHigh]
pub fn probabilities_to_atr_ratios(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    config: Option<&TargetsConfig>,
) -> Result<Vec<f64>> {
    if probabilities.len() != 5 {
        return Err(crate::utils::error::VangaError::DataError(
            "Expected 5 class probabilities for volatility reconstruction".to_string(),
        ));
    }

    let reconstruction = reconstruct_volatility(probabilities, sequence_ohlcv, config)?;
    Ok(reconstruction.atr_ratios)
}

/// Calculate volatility change percentages from probabilities
///
/// # Arguments
/// * `probabilities` - 5-element array of class probabilities
/// * `sequence_ohlcv` - OHLCV data for ATR baseline calculation
/// * `config` - Optional configuration
///
/// # Returns
/// * `Vec<f64>` - Volatility change percentage for each class
pub fn probabilities_to_volatility_changes(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    config: Option<&TargetsConfig>,
) -> Result<Vec<f64>> {
    if probabilities.len() != 5 {
        return Err(crate::utils::error::VangaError::DataError(
            "Expected 5 class probabilities for volatility reconstruction".to_string(),
        ));
    }

    let reconstruction = reconstruct_volatility(probabilities, sequence_ohlcv, config)?;
    Ok(reconstruction.volatility_changes)
}

/// Get volatility regime class names in order
pub fn get_volatility_class_names() -> Vec<&'static str> {
    vec!["VeryLow", "Low", "Medium", "High", "VeryHigh"]
}
