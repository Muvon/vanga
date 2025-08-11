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
    /// Controls the influence of the candle body ratio on sentiment calculation.
    /// Higher values increase sensitivity to price movement direction.
    /// Valid range: positive real number (typically 0.1 to 2.0). Default is 1.0.
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

/// Generate sentiment targets with optional adaptive parameters
///
/// When adaptive_params is provided, uses the pre-calibrated parameters for consistent
/// target generation between training and prediction. When None, uses base config.
pub fn generate_sentiment_targets_with_adaptive_params(
    df: &DataFrame,
    horizons: &[String],
    sequence_indices: &[usize],
    sequence_length: usize,
    adaptive_params: &SentimentAdaptiveParams, // Now mandatory
) -> Result<HashMap<String, Vec<i32>>> {
    // Use pre-calibrated parameters (always available)
    log::info!(
        "🎯 Using calibrated sentiment parameters: body_sensitivity={:.6}, volume_weight={:.3}, consistency_factor={:.3}",
        adaptive_params.body_sensitivity,
        adaptive_params.volume_weight,
        adaptive_params.consistency_factor
    );

    let config = SentimentConfig {
        body_sensitivity: adaptive_params.body_sensitivity,
        volume_weight: adaptive_params.volume_weight,
        consistency_factor: adaptive_params.consistency_factor,
    };

    log::info!(
        "🎯 Sentiment targets using calibrated sensitivity: {:.6}",
        adaptive_params.body_sensitivity
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

            match classify_sentiment(sequence_data, horizon_data, &config, adaptive_params) {
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

/// Classify sentiment using momentum-based adaptive thresholds with optional adaptive parameters
///
/// IMPROVED: This function now uses candle momentum shifts for clearer classification
/// with stronger signals that are easier for ML models to learn.
pub fn classify_sentiment(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    config: &SentimentConfig,
    adaptive_params: &SentimentAdaptiveParams,
) -> Result<i32> {
    if sequence_ohlcv.is_empty() || horizon_ohlcv.is_empty() {
        return Err(VangaError::DataError(
            "Empty OHLCV data for sentiment analysis".to_string(),
        ));
    }

    // Calculate sentiment metrics for both sequence and horizon using adaptive weighting
    let sequence_sentiment = calculate_sequence_sentiment_score_with_weighting(
        sequence_ohlcv,
        adaptive_params.horizon_decay_factor,
    );

    let horizon_sentiment = calculate_sequence_sentiment_score_with_weighting(
        horizon_ohlcv,
        adaptive_params.horizon_decay_factor,
    );

    // Calculate sentiment momentum shift (horizon vs sequence baseline)
    // This now represents the change in bullish/bearish balance
    let momentum_shift = horizon_sentiment - sequence_sentiment;

    // IMPROVED THRESHOLDS: Using momentum shift for clearer classification
    // The momentum shift is now in a stronger [-2, 2] range (difference of two [-1, 1] values)

    // Use calibrated sensitivity scaled appropriately for momentum range
    let base_threshold = config.body_sensitivity;
    let extreme_multiplier = adaptive_params.extreme_multiplier;

    // Create balanced thresholds for momentum shifts
    // These thresholds are designed for the [-2, 2] momentum shift range
    let strong_panic_threshold = -base_threshold * extreme_multiplier; // Strong bearish shift
    let moderate_panic_threshold = -base_threshold; // Moderate bearish shift
    let moderate_greed_threshold = base_threshold; // Moderate bullish shift
    let strong_greed_threshold = base_threshold * extreme_multiplier; // Strong bullish shift

    // Classify based on momentum shift with clear boundaries
    let class = if momentum_shift <= strong_panic_threshold {
        0 // Strong Panic: Large shift toward bearish sentiment
    } else if momentum_shift <= moderate_panic_threshold {
        1 // Moderate Panic: Moderate shift toward bearish
    } else if momentum_shift < moderate_greed_threshold {
        2 // Neutral: Small or no momentum shift
    } else if momentum_shift < strong_greed_threshold {
        3 // Moderate Greed: Moderate shift toward bullish
    } else {
        4 // Strong Greed: Large shift toward bullish sentiment
    };

    log::debug!(
        "🎯 Sentiment Momentum: seq={:.4}, hor={:.4}, shift={:.4}, thresholds=[{:.4}, {:.4}, {:.4}, {:.4}] → class={} ({})",
        sequence_sentiment, horizon_sentiment, momentum_shift,
        strong_panic_threshold, moderate_panic_threshold, moderate_greed_threshold, strong_greed_threshold,
        class, ["STRONG_PANIC", "MODERATE_PANIC", "NEUTRAL", "MODERATE_GREED", "STRONG_GREED"][class as usize]
    );

    Ok(class)
}

/// Calculate sentiment score for a sequence using candle momentum analysis
/// with optional horizon weighting for recent emphasis
///
/// ## Parameters
/// - `candles`: OHLCV data for sentiment analysis
/// - `horizon_decay_factor`: Optional decay factor for recent emphasis (1.0 = uniform, <1.0 = recent emphasis)
///
/// ## Weighting Strategy
/// When horizon_decay_factor < 1.0, recent candles get higher weights:
/// - weight[i] = decay_factor^(n-1-i) where i=0 is oldest, i=n-1 is newest
/// - This emphasizes recent sentiment changes for better horizon prediction
///
/// ## NEW MOMENTUM-BASED APPROACH
/// Focuses on the balance between bullish and bearish candles, weighted by their body strength.
/// This creates a stronger signal that's easier for ML models to learn.
///
/// ## Core Algorithm
/// 1. **Candle Direction**: Classify each candle as bullish (close > open) or bearish
/// 2. **Body Strength**: abs(close - open) / range as weight for each candle
/// 3. **Weighted Balance**: Sum of (bullish_weights - bearish_weights) / total_weights
/// 4. **Result Range**: [-1, 1] where -1 = all strong bearish, +1 = all strong bullish
///
/// ## Advantages
/// - Strong, clear signal in [-1, 1] range
/// - Volume-independent (pure price action)
/// - Captures market momentum shifts effectively
/// - Better differentiation between sentiment states
pub fn calculate_sequence_sentiment_score(candles: &[MarketDataRow]) -> f64 {
    calculate_sequence_sentiment_score_with_weighting(candles, 1.0) // Default: uniform weighting
}

/// Calculate weighted sentiment score with optional horizon decay for recent emphasis
pub fn calculate_sequence_sentiment_score_with_weighting(
    candles: &[MarketDataRow],
    horizon_decay_factor: f64,
) -> f64 {
    if candles.is_empty() {
        return 0.0;
    }

    let mut bullish_weight_sum = 0.0;
    let mut bearish_weight_sum = 0.0;
    let mut total_weight_sum = 0.0;
    let n = candles.len();

    for (i, candle) in candles.iter().enumerate() {
        let range = candle.high - candle.low;
        if range <= 0.0 {
            continue; // Skip invalid candles
        }

        // MOMENTUM-BASED SENTIMENT CALCULATION (volume-independent)

        // Calculate body size and direction
        let body = candle.close - candle.open;
        let body_strength = body.abs() / range; // [0, 1] range

        // Apply time decay weight if specified
        let time_weight = if horizon_decay_factor < 1.0 {
            horizon_decay_factor.powf((n - 1 - i) as f64)
        } else {
            1.0 // Uniform weighting when decay_factor >= 1.0
        };

        // Combine body strength with time weight
        let candle_weight = body_strength * time_weight;

        // Accumulate bullish or bearish weights
        if body > 0.0 {
            // Bullish candle (green)
            bullish_weight_sum += candle_weight;
        } else if body < 0.0 {
            // Bearish candle (red)
            bearish_weight_sum += candle_weight;
        }
        // Doji candles (body == 0) contribute no directional weight

        total_weight_sum += candle_weight;
    }

    if total_weight_sum > 0.0 {
        // Calculate momentum balance: difference between bullish and bearish strength
        // Normalized to [-1, 1] range
        (bullish_weight_sum - bearish_weight_sum) / total_weight_sum
    } else {
        0.0 // No valid candles or all dojis
    }
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

/// Log sentiment class distribution
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
            "📊 Sentiment Analysis [{}]: No valid targets found",
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
        "📊 Sentiment Distribution [{}]: {} samples, {} | Imbalance: {:.2}x",
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
        "📊 Sentiment Balance Quality [{}]: {} (target: ~20% per class)",
        horizon,
        balance_quality
    );
}

/// Calibrate sentiment sensitivity for balanced class distribution
///
/// This function analyzes historical sentiment momentum shifts to find the optimal body_sensitivity
/// parameter that achieves balanced class distribution (approximately 20% per class).
///
/// ## Algorithm
/// 1. Sample momentum shifts from historical data using the same logic as target generation
/// 2. Sort momentum shifts to find percentile boundaries
/// 3. Calculate sensitivity that maps percentiles to the 5-class system
/// 4. Apply reasonable bounds and return calibrated parameter
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
    target_balance: f64,
) -> Result<f64> {
    if ohlcv_data.len() < sequence_length + horizon_steps + 10 {
        return Ok(0.3); // Default for momentum-based approach
    }

    let mut momentum_shifts = Vec::new();

    // Sample momentum shifts from the data using same logic as target generation
    for i in 0..(ohlcv_data.len() - sequence_length - horizon_steps) {
        let sequence_ohlcv = &ohlcv_data[i..i + sequence_length];
        let horizon_ohlcv = &ohlcv_data[i + sequence_length..i + sequence_length + horizon_steps];

        if sequence_ohlcv.len() >= 3 && horizon_ohlcv.len() >= 3 {
            let seq_sentiment = calculate_sequence_sentiment_score(sequence_ohlcv);
            let hor_sentiment = calculate_sequence_sentiment_score(horizon_ohlcv);

            let momentum_shift = hor_sentiment - seq_sentiment;
            if momentum_shift.is_finite() {
                momentum_shifts.push(momentum_shift.abs()); // Use absolute for threshold calculation
            }
        }
    }

    if momentum_shifts.is_empty() {
        return Ok(0.3); // Default fallback for momentum approach
    }

    // Sort shifts to find percentiles for balanced distribution
    momentum_shifts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = momentum_shifts.len();

    // For 5-class system with target_balance in extreme classes:
    // - Classes 0 & 4: target_balance each (e.g., 15% each)
    // - Classes 1 & 3: (0.5 - target_balance) / 2 each (e.g., 17.5% each)
    // - Class 2: Remaining (e.g., 35%)

    // Find the percentile that separates moderate from extreme classes
    let moderate_percentile = 0.5 - target_balance; // e.g., 0.35 for 15% extreme
    let moderate_idx = ((n as f64) * moderate_percentile) as usize;
    let moderate_threshold = momentum_shifts[moderate_idx.min(n - 1)];

    // Find the percentile for extreme classes
    let extreme_percentile = 1.0 - target_balance; // e.g., 0.85 for 15% extreme
    let extreme_idx = ((n as f64) * extreme_percentile) as usize;
    let extreme_threshold = momentum_shifts[extreme_idx.min(n - 1)];

    // The base sensitivity should be the moderate threshold
    // This ensures the moderate classes capture the right percentage
    let base_sensitivity = moderate_threshold;

    // Ensure reasonable bounds for momentum-based approach
    let final_sensitivity = base_sensitivity.clamp(0.1, 0.8);

    log::info!(
        "🎯 Calibrated sentiment sensitivity: {:.4} (from {} samples, moderate: {:.4}, extreme: {:.4})",
        final_sensitivity,
        n,
        moderate_threshold,
        extreme_threshold
    );

    Ok(final_sensitivity)
}

/// Get sentiment class names in order
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

/// Reconstruct sentiment from model probabilities
pub fn reconstruct_sentiment(
    probabilities: &[f64],
    sequence_sentiment: f64,
    thresholds: &[f64; 4], // [panic_extreme, panic_moderate, greed_moderate, greed_extreme]
) -> Result<SentimentReconstruction> {
    if probabilities.len() != 5 {
        return Err(VangaError::DataError(
            "Expected 5 sentiment probabilities".to_string(),
        ));
    }

    // Define sentiment ranges based on thresholds
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

    // Calculate expected sentiment (weighted average)
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

    // Generate interpretation
    let class_names = get_sentiment_class_names();
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
