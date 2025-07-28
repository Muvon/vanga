//! Volatility target generation for cryptocurrency market regime classification
//!
//! This module implements volatility regime classification for risk assessment:
//! - 0: VeryLow (minimal volatility)
//! - 1: Low (below average volatility)
//! - 2: Medium (average volatility)
//! - 3: High (above average volatility)
//! - 4: VeryHigh (extreme volatility)

use crate::config::model::VolatilityHead;
use crate::data::structures::MarketDataRow;
use crate::utils::error::Result;
use crate::utils::market_data::extract_ohlcv_data;
use crate::utils::parser::parse_horizon_to_steps;
use polars::prelude::*;
use std::collections::HashMap;

// DEPRECATED: VolatilityConfig has been removed in favor of VolatilityHead in src/config/model.rs
// All volatility configuration is now handled through model_config.output_heads.volatility

/// Generate volatility targets for multiple horizons using sequence-to-horizon ATR
///
/// FLOW:
/// 1. Extract OHLCV data from DataFrame
/// 2. For each sequence position:
///    - Get INPUT sequence candles (for ATR baseline)
///    - Get HORIZON sequence candles (from sequence end to target horizon)
///    - Classify volatility using sequence-to-horizon comparison
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
                // Get INPUT sequence candles (for ATR baseline)
                let sequence_candles = &ohlcv_data[seq_idx..sequence_end_idx];

                // Get HORIZON sequence candles (from sequence end to target horizon)
                let horizon_candles = &ohlcv_data[sequence_end_idx..target_end_idx];

                // Only classify if we have enough horizon data for ATR calculation
                if horizon_candles.len() >= 2 {
                    let target_class = classify_volatility(
                        sequence_candles,
                        horizon_candles, // Now using horizon sequence, not single candle
                        model_config,
                    )?;

                    horizon_targets[seq_position] = target_class;
                }
            }
        }

        log_volatility_distribution(&horizon_targets, horizon);
        targets.insert(horizon.clone(), horizon_targets);
    }

    Ok(targets)
}

/// Get ATR baseline from sequence candles (same pattern as direction's market volatility)
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

/// Classify volatility using sequence-to-horizon ATR calculation
///
/// FLOW:
/// 1. Get ATR baseline from INPUT sequence (training window)
/// 2. Get ATR from sequence END to target horizon (prediction period)
/// 3. Compare horizon ATR vs sequence baseline (same logic as direction)
/// 4. Apply bandwidth sensitivity and classify into 5 classes
fn classify_volatility(
    sequence_candles: &[MarketDataRow],
    horizon_candles: &[MarketDataRow], // From sequence end to target horizon
    model_config: Option<&VolatilityHead>,
) -> Result<i32> {
    // Step 1: Get ATR baseline from INPUT sequence (like market volatility in direction)
    let sequence_baseline = get_sequence_atr_baseline(sequence_candles)?;

    // Step 2: Get ATR from sequence END to target horizon (the actual prediction period)
    let horizon_atr = get_sequence_atr_baseline(horizon_candles)?;

    // Step 3: Get config parameters with crypto-tuned defaults
    let bandwidth_size = model_config.and_then(|c| c.bandwidth_size).unwrap_or(1.0);
    let base_threshold = model_config.and_then(|c| c.base_threshold).unwrap_or(0.15); // 15% ATR change
    let extreme_multiplier = model_config
        .and_then(|c| c.extreme_multiplier)
        .unwrap_or(1.8);

    // FIXED: Use percentage-based ATR change (not ratio)
    let atr_change = (horizon_atr - sequence_baseline) / sequence_baseline;

    // Calculate adaptive thresholds
    let adaptive_threshold = base_threshold / bandwidth_size;
    let extreme_threshold = adaptive_threshold * extreme_multiplier;

    // Debug logging with threshold values
    log::debug!(
        "🎯 Volatility Classification: atr_change={:.3}, adaptive_threshold={:.3}, extreme_threshold={:.3}, bandwidth_size={}",
        atr_change, adaptive_threshold, extreme_threshold, bandwidth_size
    );

    // FIXED: 5-class system with proper percentage thresholds
    let class = if atr_change <= -extreme_threshold {
        0 // VeryLow: Much below sequence baseline
    } else if atr_change <= -adaptive_threshold {
        1 // Low: Below sequence baseline
    } else if atr_change >= extreme_threshold {
        4 // VeryHigh: Much above sequence baseline
    } else if atr_change >= adaptive_threshold {
        3 // High: Above sequence baseline
    } else {
        2 // Medium: Around sequence baseline (sideways equivalent)
    };

    Ok(class)
}

/// Log volatility class distribution
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
            "📊 Volatility Analysis [{}]: No valid targets found",
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
        "📊 Volatility Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}]",
        horizon,
        valid_targets,
        imbalance_ratio,
        class_percentages.join(", ")
    );
}

/// Apply volatility classification using model config
fn apply_volatility_classification(
    prices: &[f64],
    horizon_steps: usize,
    adaptive_percentiles: &[f64; 4],
) -> Result<Vec<i32>> {
    let volatility_window = 24; // 24-hour rolling window for volatility calculation

    if prices.len() < volatility_window + horizon_steps {
        return Err(crate::utils::error::VangaError::DataError(format!(
            "Insufficient data for volatility target generation: need {}, got {}",
            volatility_window + horizon_steps,
            prices.len()
        )));
    }

    let mut targets = vec![-1; prices.len()];

    // Calculate current volatility series for threshold determination
    let current_volatility = calculate_realized_volatility(prices, volatility_window)?;

    // Calculate regime boundaries from current volatility for consistent classification
    let regime_boundaries =
        calculate_percentile_boundaries(&current_volatility, adaptive_percentiles)?;

    // For each valid position, calculate future volatility and classify
    for (i, target) in targets
        .iter_mut()
        .enumerate()
        .take(prices.len() - horizon_steps)
        .skip(volatility_window)
    {
        // Calculate future volatility window starting at horizon
        let future_start = i + horizon_steps;
        let future_end = (future_start + volatility_window).min(prices.len());

        if future_end - future_start < volatility_window / 2 {
            // Skip if insufficient future data for reliable volatility calculation
            continue;
        }

        // Calculate future volatility for this horizon
        let future_prices = &prices[future_start..future_end];
        let future_volatility = calculate_future_volatility(future_prices)?;

        // Classify future volatility using current regime boundaries
        let volatility_class =
            classify_volatility_regime_legacy(future_volatility, &regime_boundaries);
        *target = volatility_class;
    }

    Ok(targets)
}

/// Calculate volatility targets using legacy config (for backward compatibility)
/// DEPRECATED: This function is kept for backward compatibility only
#[allow(dead_code)]
fn calculate_volatility_targets_legacy(
    prices: &[f64],
    horizon_steps: usize,
    bandwidth_size: f64,
) -> Result<Vec<i32>> {
    // Use default percentiles for legacy path
    let base_percentiles = [0.20, 0.40, 0.60, 0.80];
    let sensitivity = 1.0 / bandwidth_size;
    let center = 0.5;

    // Apply bandwidth sensitivity to percentiles
    let adaptive_percentiles: [f64; 4] = base_percentiles.map(|p| {
        let distance = p - center;
        center + (distance * sensitivity)
    });

    apply_volatility_classification(prices, horizon_steps, &adaptive_percentiles)
}

/// Calculate future volatility for horizon-specific prediction
fn calculate_future_volatility(prices: &[f64]) -> Result<f64> {
    if prices.len() < 2 {
        return Ok(0.0); // Default for insufficient data
    }

    // Calculate returns for the future period
    let mut returns = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        if prices[i] > 0.0 && prices[i - 1] > 0.0 {
            returns.push((prices[i] / prices[i - 1]).ln());
        } else {
            returns.push(0.0);
        }
    }

    if returns.is_empty() {
        return Ok(0.0);
    }

    // Calculate standard deviation of returns (volatility)
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    Ok(variance.sqrt())
}

/// Classify volatility into regime using boundaries (legacy single-value version)
fn classify_volatility_regime_legacy(volatility: f64, boundaries: &[f64; 4]) -> i32 {
    if volatility <= boundaries[0] {
        0 // VeryLow
    } else if volatility <= boundaries[1] {
        1 // Low
    } else if volatility <= boundaries[2] {
        2 // Medium
    } else if volatility <= boundaries[3] {
        3 // High
    } else {
        4 // VeryHigh
    }
}

/// Volatility regime classes (5-class system)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolatilityRegime {
    VeryLow = 0,  // <20th percentile
    Low = 1,      // 20th-40th percentile
    Medium = 2,   // 40th-60th percentile
    High = 3,     // 60th-80th percentile
    VeryHigh = 4, // >80th percentile
}

/// Calculate realized volatility using rolling window
fn calculate_realized_volatility(prices: &[f64], window: usize) -> Result<Vec<f64>> {
    if prices.len() < window {
        return Err(crate::utils::error::VangaError::DataError(format!(
            "Insufficient data for volatility calculation: need {}, got {}",
            window,
            prices.len()
        )));
    }

    let mut volatilities = Vec::new();

    for i in window..prices.len() {
        let window_prices = &prices[i - window..i];
        let volatility = calculate_future_volatility(window_prices)?;
        volatilities.push(volatility);
    }

    Ok(volatilities)
}

/// Calculate percentile boundaries for classification
fn calculate_percentile_boundaries(values: &[f64], percentiles: &[f64; 4]) -> Result<[f64; 4]> {
    if values.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Cannot calculate percentiles from empty data".to_string(),
        ));
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut boundaries = [0.0; 4];
    for (i, &percentile) in percentiles.iter().enumerate() {
        let index = ((sorted_values.len() - 1) as f64 * percentile) as usize;
        boundaries[i] = sorted_values[index.min(sorted_values.len() - 1)];
    }

    Ok(boundaries)
}
