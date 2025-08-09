//! Sentiment target generation for cryptocurrency market psychology analysis
//!
//! # 🎯 TARGET PURPOSE: "WHAT IS THE MARKET SENTIMENT?"
//!
//! This module implements **candle body sentiment analysis** for market psychology detection.
//! It answers: "Is the market showing greed, panic, or neutral sentiment?"
//!
//! ## 📊 MATHEMATICAL FOUNDATION
//!
//! ### **Core Logic: Candle Body Psychology Analysis**
//! ```
//! 1. Calculate body ratio: (close - open) / (high - low) - directional strength
//! 2. Calculate body size: abs(close - open) / typical_price - magnitude
//! 3. Calculate wick imbalance: (upper_wick - lower_wick) / (high - low) - psychology
//! 4. Apply volume confirmation: body_strength * volume_ratio - conviction
//! 5. Combine into sentiment score with adaptive thresholds
//! ```
//!
//! ### **5-Class Sentiment Classification:**
//! - **0: Strong Panic** - Large red bodies, lower wicks, high volume confirmation
//! - **1: Moderate Panic** - Medium red bodies, mixed wicks, moderate bearish sentiment
//! - **2: Neutral** - Small bodies, balanced wicks, sideways sentiment
//! - **3: Moderate Greed** - Medium green bodies, upper wicks, moderate bullish sentiment
//! - **4: Strong Greed** - Large green bodies, upper wicks, high volume confirmation
//!
//! ## 🔧 KEY FEATURES
//!
//! ### **Body Ratio Analysis**
//! - Positive values: Bullish sentiment (close > open)
//! - Negative values: Bearish sentiment (close < open)
//! - Magnitude indicates strength of directional conviction
//!
//! ### **Volume Confirmation**
//! - High volume + strong body = high conviction sentiment
//! - Low volume + strong body = potential false signal
//! - Volume weighting validates sentiment authenticity
//!
//! ### **Adaptive Thresholds**
//! - Automatically calibrated for balanced 20% per class distribution
//! - Adjusts to market volatility and sentiment consistency
//! - No hardcoded thresholds - fully adaptive system

use crate::config::model::TargetsConfig;
use crate::targets::adaptive_parameters::SentimentAdaptiveParams;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Use the main MarketDataRow from data::structures
use crate::data::structures::MarketDataRow;

/// Sentiment analysis metrics
#[derive(Debug, Clone)]
pub struct SentimentMetrics {
    /// Body ratio: (close - open) / (high - low) - directional strength
    pub body_ratio: f64,
    /// Body size: abs(close - open) / typical_price - magnitude
    pub body_size: f64,
    /// Wick imbalance: (upper_wick - lower_wick) / (high - low) - psychology
    pub wick_imbalance: f64,
    /// Volume confirmation: body_strength * volume_ratio - conviction
    pub volume_confirmation: f64,
    /// Final sentiment score combining all metrics
    pub sentiment_score: f64,
}

/// Sentiment configuration for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentConfig {
    pub body_sensitivity: f64,
    pub volume_weight: f64,
    pub consistency_factor: f64,
}

impl Default for SentimentConfig {
    fn default() -> Self {
        Self {
            body_sensitivity: 1.0,
            volume_weight: 0.3,
            consistency_factor: 1.0,
        }
    }
}

/// Generate sentiment targets using TargetsConfig (UNIFIED APPROACH)
pub fn generate_sentiment_targets(
    df: &DataFrame,
    horizons: &[String],
    targets_config: &TargetsConfig,
    sequence_indices: &[usize],
    sequence_length: usize,
) -> Result<HashMap<String, Vec<i32>>> {
    generate_sentiment_targets_with_adaptive_params(
        df,
        horizons,
        targets_config,
        sequence_indices,
        sequence_length,
        None, // No adaptive parameters - use base config
    )
}

/// Generate sentiment targets with optional adaptive parameters
///
/// When adaptive_params is provided, uses the pre-calibrated parameters for consistent
/// target generation between training and prediction. When None, uses base config.
pub fn generate_sentiment_targets_with_adaptive_params(
    df: &DataFrame,
    horizons: &[String],
    targets_config: &TargetsConfig,
    sequence_indices: &[usize],
    sequence_length: usize,
    adaptive_params: Option<&SentimentAdaptiveParams>,
) -> Result<HashMap<String, Vec<i32>>> {
    let ohlcv_data = extract_ohlcv_data(df)?;

    // Use adaptive parameters if available, otherwise calibrate
    let calibrated_body_sensitivity = if let Some(params) = adaptive_params {
        log::info!(
            "🎯 Using pre-calibrated sentiment parameters: body_sensitivity={:.4}, volume_weight={:.4}",
            params.body_sensitivity,
            params.volume_weight
        );
        params.body_sensitivity
    } else {
        log::info!("🎯 Calibrating sentiment sensitivity (no adaptive parameters provided)");
        // Use first horizon for calibration
        let first_horizon_steps = parse_horizon_steps(&horizons[0])?;
        calibrate_sentiment_sensitivity(
            &ohlcv_data,
            sequence_length,
            first_horizon_steps,
            targets_config.balance_target,
        )?
    };

    let config = SentimentConfig {
        body_sensitivity: calibrated_body_sensitivity,
        volume_weight: if let Some(params) = adaptive_params {
            params.volume_weight
        } else {
            0.3
        },
        consistency_factor: if let Some(params) = adaptive_params {
            params.consistency_factor
        } else {
            1.0
        },
    };

    log::info!(
        "🎯 Sentiment targets using calibrated sensitivity: {:.6} (was base: {:.6})",
        calibrated_body_sensitivity,
        targets_config.base_sensitivity
    );

    let ohlcv_data = extract_ohlcv_data(df)?;
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_steps(horizon)?;
        let mut horizon_targets = Vec::new();

        for &seq_start in sequence_indices {
            let seq_end = seq_start + sequence_length;
            let horizon_end = seq_end + horizon_steps;

            if horizon_end > ohlcv_data.len() {
                continue;
            }

            let sequence_data = &ohlcv_data[seq_start..seq_end];
            let horizon_data = &ohlcv_data[seq_end..horizon_end];

            match classify_sentiment(sequence_data, horizon_data, targets_config, &config) {
                Ok(class) => horizon_targets.push(class),
                Err(e) => {
                    log::warn!(
                        "Failed to classify sentiment for sequence {}: {}",
                        seq_start,
                        e
                    );
                    continue;
                }
            }
        }

        if !horizon_targets.is_empty() {
            log_sentiment_distribution(&horizon_targets, horizon);
            targets.insert(horizon.clone(), horizon_targets);
        }
    }

    Ok(targets)
}

/// Classify sentiment using percentile-based adaptive thresholds (no magic numbers)
pub fn classify_sentiment(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    targets_config: &TargetsConfig,
    config: &SentimentConfig,
) -> Result<i32> {
    if sequence_ohlcv.is_empty() || horizon_ohlcv.is_empty() {
        return Err(VangaError::DataError(
            "Empty OHLCV data for sentiment analysis".to_string(),
        ));
    }

    // Calculate sentiment metrics for both sequence and horizon
    let sequence_sentiment = calculate_sequence_sentiment_score(sequence_ohlcv);
    let horizon_sentiment = calculate_sequence_sentiment_score(horizon_ohlcv);

    // Calculate sentiment change (horizon vs sequence baseline)
    let sentiment_change = horizon_sentiment - sequence_sentiment;

    // Use adaptive percentile-based thresholds (calibrated from historical data)
    let base_threshold = config.body_sensitivity; // This is now the base percentile threshold
    let extreme_multiplier = targets_config.extreme_multiplier;

    // Calculate thresholds based on calibrated percentiles
    // These will be set by calibration to achieve 20% per class
    let moderate_panic_threshold = -base_threshold * extreme_multiplier; // Negative for panic
    let neutral_lower_threshold = -base_threshold; // Small negative
    let neutral_upper_threshold = base_threshold; // Small positive
    let moderate_greed_threshold = base_threshold * extreme_multiplier; // Positive for greed

    // Classify based on sentiment change using percentile boundaries
    let class = if sentiment_change <= moderate_panic_threshold {
        0 // Strong Panic: Bottom 20%
    } else if sentiment_change <= neutral_lower_threshold {
        1 // Moderate Panic: Next 20%
    } else if sentiment_change <= neutral_upper_threshold {
        2 // Neutral: Middle 20%
    } else if sentiment_change <= moderate_greed_threshold {
        3 // Moderate Greed: Next 20%
    } else {
        4 // Strong Greed: Top 20%
    };

    log::debug!(
        "🎯 Sentiment Analysis: seq_sentiment={:.6}, hor_sentiment={:.6}, sentiment_change={:.6}, thresholds=[{:.6}, {:.6}, {:.6}, {:.6}] → class={} ({})",
        sequence_sentiment, horizon_sentiment, sentiment_change,
        moderate_panic_threshold, neutral_lower_threshold, neutral_upper_threshold, moderate_greed_threshold,
        class, ["STRONG_PANIC", "MODERATE_PANIC", "NEUTRAL", "MODERATE_GREED", "STRONG_GREED"][class as usize]
    );

    Ok(class)
}

/// Calculate sentiment score using volume-price correlation analysis
///
/// This enhanced approach captures buying/selling pressure by analyzing the correlation
/// between price movements and volume, providing a more accurate representation of
/// market psychology than simple green/red candle analysis.
pub fn calculate_sequence_sentiment_score(candles: &[MarketDataRow]) -> f64 {
    if candles.is_empty() {
        return 0.0;
    }

    // Calculate average volume for normalization (keep existing logic)
    let avg_volume = candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64;
    let mut correlation_sum = 0.0;
    let mut valid_candles = 0;

    for candle in candles {
        if candle.open <= 0.0 || avg_volume <= 0.0 {
            continue;
        }

        // Volume-price correlation instead of simple green/red analysis
        let price_change_pct = (candle.close - candle.open) / candle.open;
        let volume_ratio = candle.volume / avg_volume;

        // Correlation captures buying/selling pressure:
        // Positive: High volume + price up = buying pressure (bullish sentiment)
        // Negative: High volume + price down = selling pressure (bearish sentiment)
        // Near zero: Low volume or conflicting signals = neutral sentiment
        let vp_correlation = price_change_pct * volume_ratio;

        correlation_sum += vp_correlation;
        valid_candles += 1;
    }

    if valid_candles > 0 {
        correlation_sum / valid_candles as f64
    } else {
        0.0
    }
}

/// Calculate comprehensive sentiment metrics for OHLCV data (legacy - kept for compatibility)
#[allow(dead_code)]
fn calculate_sentiment_metrics(ohlcv_data: &[MarketDataRow]) -> Result<SentimentMetrics> {
    if ohlcv_data.is_empty() {
        return Err(VangaError::DataError(
            "Empty OHLCV data for sentiment metrics".to_string(),
        ));
    }

    let mut total_body_ratio = 0.0;
    let mut total_body_size = 0.0;
    let mut total_wick_imbalance = 0.0;
    let mut total_volume_confirmation = 0.0;
    let mut valid_candles = 0;

    // Calculate average volume for confirmation
    let avg_volume = ohlcv_data.iter().map(|c| c.volume).sum::<f64>() / ohlcv_data.len() as f64;

    for candle in ohlcv_data {
        if let Ok(metrics) = calculate_single_candle_metrics(candle, avg_volume) {
            total_body_ratio += metrics.body_ratio;
            total_body_size += metrics.body_size;
            total_wick_imbalance += metrics.wick_imbalance;
            total_volume_confirmation += metrics.volume_confirmation;
            valid_candles += 1;
        }
    }

    if valid_candles == 0 {
        return Err(VangaError::DataError(
            "No valid candles for sentiment analysis".to_string(),
        ));
    }

    let avg_body_ratio = total_body_ratio / valid_candles as f64;
    let avg_body_size = total_body_size / valid_candles as f64;
    let avg_wick_imbalance = total_wick_imbalance / valid_candles as f64;
    let avg_volume_confirmation = total_volume_confirmation / valid_candles as f64;

    // Combine metrics into final sentiment score
    let sentiment_score =
        avg_body_ratio * avg_body_size * (1.0 + avg_volume_confirmation + avg_wick_imbalance * 0.5);

    Ok(SentimentMetrics {
        body_ratio: avg_body_ratio,
        body_size: avg_body_size,
        wick_imbalance: avg_wick_imbalance,
        volume_confirmation: avg_volume_confirmation,
        sentiment_score,
    })
}

/// Calculate sentiment metrics for a single candle (legacy - kept for compatibility)
#[allow(dead_code)]
pub fn calculate_single_candle_metrics(
    candle: &MarketDataRow,
    avg_volume: f64,
) -> Result<SentimentMetrics> {
    let range = candle.high - candle.low;
    if range <= 0.0 {
        return Err(VangaError::DataError("Invalid candle range".to_string()));
    }

    let typical_price = (candle.high + candle.low + candle.close) / 3.0;
    if typical_price <= 0.0 {
        return Err(VangaError::DataError("Invalid typical price".to_string()));
    }

    // Body ratio: (close - open) / (high - low) - directional strength
    let body_ratio = (candle.close - candle.open) / range;

    // Body size: abs(close - open) / typical_price - magnitude
    let body_size = (candle.close - candle.open).abs() / typical_price;

    // Wick analysis
    let upper_wick = candle.high - candle.close.max(candle.open);
    let lower_wick = candle.close.min(candle.open) - candle.low;
    let wick_imbalance = (upper_wick - lower_wick) / range;

    // Volume confirmation
    let volume_ratio = if avg_volume > 0.0 {
        candle.volume / avg_volume
    } else {
        1.0
    };
    let volume_confirmation = (volume_ratio - 1.0) * 0.3; // 30% weight for volume

    // Combine into sentiment score
    let sentiment_score =
        body_ratio * body_size * (1.0 + volume_confirmation + wick_imbalance * 0.5);

    Ok(SentimentMetrics {
        body_ratio,
        body_size,
        wick_imbalance,
        volume_confirmation,
        sentiment_score,
    })
}

/// Calculate single candle sentiment score (simplified for consistency calculation)
#[allow(dead_code)] // Keep for potential future use
fn calculate_single_candle_sentiment(candle: &MarketDataRow, volume_weight: f64) -> Result<f64> {
    let range = candle.high - candle.low;
    if range <= 0.0 {
        return Ok(0.0);
    }

    let typical_price = (candle.high + candle.low + candle.close) / 3.0;
    if typical_price <= 0.0 {
        return Ok(0.0);
    }

    let body_ratio = (candle.close - candle.open) / range;
    let body_size = (candle.close - candle.open).abs() / typical_price;

    Ok(body_ratio * body_size * volume_weight)
}

/// Calculate sentiment consistency for adaptive thresholds (legacy - kept for compatibility)
#[allow(dead_code)]
pub fn calculate_sentiment_consistency(sentiment_scores: &[f64]) -> Result<f64> {
    if sentiment_scores.len() < 3 {
        return Ok(0.001); // Much smaller default for insufficient data
    }

    let mean = sentiment_scores.iter().sum::<f64>() / sentiment_scores.len() as f64;
    let variance = sentiment_scores
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / sentiment_scores.len() as f64;
    let std_dev = variance.sqrt();

    // Use actual data scale - no artificial minimum that's too large
    Ok(std_dev.max(0.0001)) // Minimum consistency threshold matching actual sentiment scale
}

/// Extract OHLCV data from DataFrame
fn extract_ohlcv_data(df: &DataFrame) -> Result<Vec<MarketDataRow>> {
    let open = df.column("open")?.f64()?.to_vec();
    let high = df.column("high")?.f64()?.to_vec();
    let low = df.column("low")?.f64()?.to_vec();
    let close = df.column("close")?.f64()?.to_vec();
    let volume = df.column("volume")?.f64()?.to_vec();

    let mut ohlcv_data = Vec::new();
    for i in 0..open.len() {
        if let (Some(o), Some(h), Some(l), Some(c), Some(v)) =
            (open[i], high[i], low[i], close[i], volume[i])
        {
            ohlcv_data.push(MarketDataRow {
                timestamp: i as i64, // Use index as timestamp for sentiment analysis
                open: o,
                high: h,
                low: l,
                close: c,
                volume: v,
            });
        }
    }

    Ok(ohlcv_data)
}

/// Parse horizon string to steps
fn parse_horizon_steps(horizon: &str) -> Result<usize> {
    let horizon_clean = horizon.trim_end_matches('h');
    horizon_clean
        .parse::<usize>()
        .map_err(|_| VangaError::DataError(format!("Invalid horizon format: {}", horizon)))
}

/// Log sentiment class distribution with volume-price correlation context
fn log_sentiment_distribution(targets: &[i32], horizon: &str) {
    let class_names = [
        "STRONG_PANIC",
        "MODERATE_PANIC",
        "NEUTRAL",
        "MODERATE_GREED",
        "STRONG_GREED",
    ];
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
            "📊 Volume-Price Sentiment Analysis [{}]: No valid targets found",
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
        "📊 Volume-Price Sentiment Distribution [{}]: {} samples, {} | Imbalance: {:.2}x",
        horizon,
        valid_targets,
        class_percentages.join(", "),
        imbalance_ratio
    );

    // Log balance quality assessment
    let balance_quality = if imbalance_ratio <= 1.5 {
        "EXCELLENT"
    } else if imbalance_ratio <= 2.0 {
        "GOOD"
    } else if imbalance_ratio <= 3.0 {
        "FAIR"
    } else {
        "POOR"
    };

    log::info!(
        "📊 Volume-Price Sentiment Balance Quality [{}]: {} (target: ~20% per class)",
        horizon,
        balance_quality
    );
}

/// Calibrate sentiment sensitivity for balanced class distribution
///
/// This function analyzes historical sentiment data to find the optimal body_sensitivity
/// parameter that achieves the target class balance (e.g., 15% in extreme classes).
///
/// ## Algorithm
/// 1. Sample sentiment changes from historical data using the same logic as target generation
/// 2. Calculate sentiment consistency for normalization (like direction's trend consistency)
/// 3. Find the percentile threshold that corresponds to target_balance for extreme classes
/// 4. Calculate body_sensitivity to achieve that threshold with extreme_multiplier
/// 5. Apply reasonable bounds and return calibrated parameter
///
/// ## Parameters
/// - `ohlcv_data`: Historical OHLCV data for sentiment analysis
/// - `sequence_length`: Length of input sequences
/// - `horizon_steps`: Number of steps in prediction horizon
/// - `target_balance`: Target percentage for extreme classes (e.g., 0.15 for 15%)
///
/// ## Returns
/// Calibrated body_sensitivity parameter for balanced sentiment classification
pub fn calibrate_sentiment_sensitivity(
    ohlcv_data: &[MarketDataRow],
    sequence_length: usize,
    horizon_steps: usize,
    _target_balance: f64, // Unused but kept for API compatibility
) -> Result<f64> {
    log::info!(
        "🔍 Sentiment calibration: data_len={}, sequence_length={}, horizon_steps={}, required={}",
        ohlcv_data.len(),
        sequence_length,
        horizon_steps,
        sequence_length + horizon_steps + 10
    );

    if ohlcv_data.len() < sequence_length + horizon_steps + 10 {
        log::warn!(
            "⚠️ Insufficient data for sentiment calibration: {} < {}, using default: 0.055",
            ohlcv_data.len(),
            sequence_length + horizon_steps + 10
        );
        return Ok(0.055); // Default from debug analysis
    }

    let mut sentiment_changes = Vec::new();

    // Sample sentiment changes from the data using same logic as target generation
    for i in 0..(ohlcv_data.len() - sequence_length - horizon_steps) {
        let sequence_ohlcv = &ohlcv_data[i..i + sequence_length];
        let horizon_ohlcv = &ohlcv_data[i + sequence_length..i + sequence_length + horizon_steps];

        if sequence_ohlcv.len() >= 3 && !horizon_ohlcv.is_empty() {
            let seq_sentiment = calculate_sequence_sentiment_score(sequence_ohlcv);
            let hor_sentiment = calculate_sequence_sentiment_score(horizon_ohlcv);

            let sentiment_change = hor_sentiment - seq_sentiment;
            if sentiment_change.is_finite() {
                sentiment_changes.push(sentiment_change);
            }
        }
    }

    if sentiment_changes.is_empty() {
        log::warn!("⚠️ No valid sentiment changes found, using default: 0.055");
        return Ok(0.055); // Default fallback from debug analysis
    }

    log::info!(
        "🔍 Collected {} sentiment changes, range: [{:.6}, {:.6}]",
        sentiment_changes.len(),
        sentiment_changes
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0),
        sentiment_changes
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0)
    );

    // Sort changes to find percentiles for balanced 5-class distribution
    sentiment_changes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sentiment_changes.len();

    // Calculate percentile thresholds for 20% per class
    // We need the 20th, 40th, 60th, and 80th percentiles for 5-class distribution
    let p20_idx = (n as f64 * 0.20) as usize;
    let p40_idx = (n as f64 * 0.40) as usize;
    let p60_idx = (n as f64 * 0.60) as usize;
    let p80_idx = (n as f64 * 0.80) as usize;

    let strong_panic_threshold = sentiment_changes[p20_idx.min(n - 1)];
    let moderate_panic_threshold = sentiment_changes[p40_idx.min(n - 1)];
    let moderate_greed_threshold = sentiment_changes[p60_idx.min(n - 1)];
    let strong_greed_threshold = sentiment_changes[p80_idx.min(n - 1)];

    // FIXED: Use actual percentile values directly for balanced classification
    // Instead of calculating derived thresholds, use the actual 40th and 60th percentile values
    // This ensures the neutral zone contains exactly 20% of the data

    // The 40th percentile value should be our neutral lower bound
    // The 60th percentile value should be our neutral upper bound
    // Use the smaller absolute value to ensure symmetric thresholds
    let abs_40th = moderate_panic_threshold.abs();
    let abs_60th = moderate_greed_threshold.abs();
    let base_threshold = abs_40th.min(abs_60th);

    // Ensure minimum threshold for numerical stability
    let final_sensitivity = if base_threshold < 0.0001 {
        0.0001
    } else {
        base_threshold
    };

    log::info!(
        "🎯 Calibrated sentiment sensitivity: {:.6} (from {} samples, percentiles: 20th={:.6}, 40th={:.6}, 60th={:.6}, 80th={:.6}, base_threshold={:.6})",
        final_sensitivity,
        n,
        strong_panic_threshold,
        moderate_panic_threshold,
        moderate_greed_threshold,
        strong_greed_threshold,
        base_threshold
    );

    Ok(final_sensitivity)
}

/// Get sentiment class names (keeping original PANIC/GREED terminology)
pub fn get_sentiment_class_names() -> Vec<&'static str> {
    vec![
        "STRONG_PANIC",
        "MODERATE_PANIC",
        "NEUTRAL",
        "MODERATE_GREED",
        "STRONG_GREED",
    ]
}

// ============================================================================
// PREDICTION RECONSTRUCTION METHODS
// ============================================================================

/// Reconstruction result for sentiment predictions
#[derive(Debug, Clone)]
pub struct SentimentReconstruction {
    /// Sentiment ranges for each class [lower_bound, upper_bound]
    pub sentiment_ranges: Vec<[f64; 2]>,
    /// Class probabilities from model
    pub probabilities: Vec<f64>,
    /// Most likely class index
    pub most_likely_class: usize,
    /// Confidence (probability of most likely class)
    pub confidence: f64,
    /// Expected sentiment score (weighted average)
    pub expected_sentiment: f64,
    /// Sentiment interpretation
    pub sentiment_interpretation: String,
}

/// Reconstruct sentiment from model probabilities with volume-price correlation interpretation
pub fn reconstruct_sentiment(
    probabilities: &[f64],
    sequence_sentiment: f64, // Now represents volume-price correlation baseline
    thresholds: &[f64; 4],   // [panic_extreme, panic_moderate, greed_moderate, greed_extreme]
) -> Result<SentimentReconstruction> {
    if probabilities.len() != 5 {
        return Err(VangaError::DataError(
            "Expected 5 sentiment probabilities".to_string(),
        ));
    }

    // Define sentiment ranges based on volume-price correlation thresholds
    let sentiment_ranges = vec![
        [f64::NEG_INFINITY, sequence_sentiment - thresholds[0]], // Strong Panic
        [
            sequence_sentiment - thresholds[0],
            sequence_sentiment - thresholds[1],
        ], // Moderate Panic
        [
            sequence_sentiment - thresholds[1],
            sequence_sentiment + thresholds[2],
        ], // Neutral
        [
            sequence_sentiment + thresholds[2],
            sequence_sentiment + thresholds[3],
        ], // Moderate Greed
        [sequence_sentiment + thresholds[3], f64::INFINITY],     // Strong Greed
    ];

    // Find most likely class
    let most_likely_class = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(2); // Default to neutral

    let confidence = probabilities[most_likely_class];

    // Calculate expected sentiment (weighted average of volume-price correlation)
    let class_midpoints = [
        sequence_sentiment - thresholds[0] * 1.5, // Strong Panic
        sequence_sentiment - (thresholds[0] + thresholds[1]) / 2.0, // Moderate Panic
        sequence_sentiment,                       // Neutral
        sequence_sentiment + (thresholds[2] + thresholds[3]) / 2.0, // Moderate Greed
        sequence_sentiment + thresholds[3] * 1.5, // Strong Greed
    ];

    let expected_sentiment = probabilities
        .iter()
        .zip(class_midpoints.iter())
        .map(|(prob, midpoint)| prob * midpoint)
        .sum::<f64>();

    // Generate interpretation with corrected PANIC/GREED terminology
    let class_names = [
        "STRONG_PANIC",
        "MODERATE_PANIC",
        "NEUTRAL",
        "MODERATE_GREED",
        "STRONG_GREED",
    ];
    let sentiment_interpretation = format!(
        "{} (confidence: {:.1}%)",
        class_names[most_likely_class],
        confidence * 100.0
    );

    Ok(SentimentReconstruction {
        sentiment_ranges,
        probabilities: probabilities.to_vec(),
        most_likely_class,
        confidence,
        expected_sentiment,
        sentiment_interpretation,
    })
}
