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

/// Classify sentiment using sequence→horizon bullish strength ratio for strong ML signals
///
/// CORRECTED APPROACH: Uses sequence as baseline, horizon as target measurement.
/// Creates strong ratio-based signals that are easier for LSTM to learn.
///
/// ## Algorithm
/// 1. Calculate bullish strength for sequence (baseline context)
/// 2. Calculate bullish strength for horizon (target measurement)
/// 3. Compute bullish ratio = horizon_strength / sequence_strength
/// 4. Classify using adaptive thresholds on ratio values
///
/// ## Signal Strength
/// - **Strong Signal**: Direct ratio in [0, ∞] range
/// - **Clear Meaning**: >1 = more bullish, <1 = more bearish, =1 = same sentiment
/// - **ML Friendly**: Consistent, predictable patterns for LSTM learning
pub fn classify_sentiment(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    _config: &SentimentConfig, // Kept for API compatibility, will be removed in future version
    adaptive_params: &SentimentAdaptiveParams,
) -> Result<i32> {
    if sequence_ohlcv.is_empty() || horizon_ohlcv.is_empty() {
        return Err(VangaError::DataError(
            "Empty sequence or horizon OHLCV data for sentiment analysis".to_string(),
        ));
    }

    // Calculate bullish strength for both sequence (baseline) and horizon (target)
    let sequence_bullish_strength = calculate_bullish_strength(sequence_ohlcv);
    let horizon_bullish_strength = calculate_bullish_strength(horizon_ohlcv);

    // Calculate bullish ratio: horizon/sequence (with adaptive minimum baseline)
    let baseline_strength = sequence_bullish_strength.max(adaptive_params.min_baseline_strength);
    let bullish_ratio = horizon_bullish_strength / baseline_strength;

    // Use adaptive thresholds calibrated for balanced distribution on ratio values
    let moderate_threshold = adaptive_params.body_sensitivity; // Now represents ratio threshold
    let extreme_threshold = adaptive_params.body_sensitivity * adaptive_params.extreme_multiplier;

    // Classify based on bullish ratio with clear boundaries
    // bullish_ratio range: [0, ∞] where 1.0 = same sentiment
    let class = if bullish_ratio <= (1.0 - extreme_threshold) {
        0 // Strong Panic: Much less bullish than sequence
    } else if bullish_ratio <= (1.0 - moderate_threshold) {
        1 // Moderate Panic: Somewhat less bullish than sequence
    } else if bullish_ratio < (1.0 + moderate_threshold) {
        2 // Neutral: Similar bullish strength to sequence
    } else if bullish_ratio < (1.0 + extreme_threshold) {
        3 // Moderate Greed: Somewhat more bullish than sequence
    } else {
        4 // Strong Greed: Much more bullish than sequence
    };

    log::debug!(
        "🎯 Sentiment Ratio: seq_bullish={:.4}, hor_bullish={:.4}, ratio={:.4}, thresholds=[{:.4}, {:.4}, {:.4}, {:.4}] → class={} ({})",
        sequence_bullish_strength, horizon_bullish_strength, bullish_ratio,
        1.0 - extreme_threshold, 1.0 - moderate_threshold, 1.0 + moderate_threshold, 1.0 + extreme_threshold,
        class, ["STRONG_PANIC", "MODERATE_PANIC", "NEUTRAL", "MODERATE_GREED", "STRONG_GREED"][class as usize]
    );

    Ok(class)
}

/// Calculate bullish strength from candles for sequence→horizon comparison
///
/// CORRECTED APPROACH: Calculate bullish vs bearish strength for any set of candles.
/// Used for both sequence (baseline) and horizon (target) measurements.
///
/// ## Core Algorithm
/// 1. **Body Strength**: Calculate abs(close - open) / range for each candle
/// 2. **Bullish Accumulation**: Sum body strength for bullish candles (close > open)
/// 3. **Total Strength**: Sum all body strengths regardless of direction
/// 4. **Bullish Ratio**: bullish_strength / total_strength
///
/// ## Result Range
/// - **[0.0, 1.0]** where:
///   - **0.0** = All strong bearish candles
///   - **0.5** = Balanced bullish/bearish strength (neutral)
///   - **1.0** = All strong bullish candles
///
/// ## Usage
/// - **Sequence**: Establishes bullish strength baseline/context
/// - **Horizon**: Measures target bullish strength for comparison
/// - **Ratio**: horizon_strength / sequence_strength shows sentiment change
pub fn calculate_bullish_strength(candles: &[MarketDataRow]) -> f64 {
    if candles.is_empty() {
        return 0.5; // Neutral if no data
    }

    let mut bullish_strength = 0.0;
    let mut total_strength = 0.0;

    for candle in candles {
        let range = candle.high - candle.low;
        if range <= 0.0 {
            continue; // Skip invalid candles (no price movement)
        }

        // Calculate body strength (magnitude of price movement within range)
        let body = candle.close - candle.open;
        let body_strength = body.abs() / range; // [0, 1] normalized by range

        // Accumulate bullish strength only for bullish candles
        if body > 0.0 {
            bullish_strength += body_strength;
        }

        // Always accumulate total strength (bullish + bearish)
        total_strength += body_strength;
    }

    if total_strength > 0.0 {
        // Return bullish ratio: [0, 1] where 0.5 = neutral
        bullish_strength / total_strength
    } else {
        0.5 // Neutral if no valid candles (all dojis)
    }
}

/// Legacy function for backward compatibility - now uses simplified approach
pub fn calculate_sequence_sentiment_score(candles: &[MarketDataRow]) -> f64 {
    // Convert [0, 1] bullish strength to [-1, 1] for legacy compatibility
    let bullish_strength = calculate_bullish_strength(candles);
    (bullish_strength - 0.5) * 2.0 // Map [0, 1] to [-1, 1]
}

/// Legacy function for backward compatibility - now uses simplified approach
/// Calculate sentiment score with optional horizon decay weighting
///
/// Note: The horizon_decay_factor parameter is kept for API compatibility but not used
/// in the simplified ratio-based approach. Will be removed in future version.
pub fn calculate_sequence_sentiment_score_with_weighting(
    candles: &[MarketDataRow],
    _horizon_decay_factor: f64, // Kept for API compatibility, not used in ratio-based approach
) -> f64 {
    calculate_sequence_sentiment_score(candles)
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

/// Reconstruct sentiment from model probabilities using ratio-based approach
///
/// This method reverses the new ratio-based classification logic to convert
/// raw model probabilities back to meaningful bullish strength ratios and sentiment metrics.
///
/// # Arguments
/// * `probabilities` - 5-element array of class probabilities [Strong Panic, Moderate Panic, Neutral, Moderate Greed, Strong Greed]
/// * `sequence_ohlcv` - OHLCV data for the input sequence (same as used in training)
/// * `adaptive_params` - Adaptive parameters used during training (for threshold calculation)
///
/// # Returns
/// * `SentimentReconstruction` - Complete reconstruction with bullish ratios and sentiment metrics
pub fn reconstruct_sentiment(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    adaptive_params: &crate::targets::adaptive_parameters::SentimentAdaptiveParams,
) -> Result<SentimentReconstruction> {
    if probabilities.len() != 5 {
        return Err(VangaError::DataError(
            "Expected 5 sentiment probabilities".to_string(),
        ));
    }

    if sequence_ohlcv.is_empty() {
        return Err(VangaError::DataError(
            "Sequence OHLCV data is required for sentiment reconstruction".to_string(),
        ));
    }

    // Calculate sequence bullish strength baseline (same as training)
    let sequence_bullish_strength = calculate_bullish_strength(sequence_ohlcv);
    let baseline_strength = sequence_bullish_strength.max(adaptive_params.min_baseline_strength); // Same minimum as training

    // Use adaptive parameters for threshold calculation (same as training)
    let moderate_threshold = adaptive_params.body_sensitivity;
    let extreme_threshold = adaptive_params.body_sensitivity * adaptive_params.extreme_multiplier;

    // Define bullish ratio ranges for each class (reverse of classification logic)
    let ratio_ranges = [
        [0.0, 1.0 - extreme_threshold], // Strong Panic: Much less bullish
        [1.0 - extreme_threshold, 1.0 - moderate_threshold], // Moderate Panic: Somewhat less bullish
        [1.0 - moderate_threshold, 1.0 + moderate_threshold], // Neutral: Similar bullish strength
        [1.0 + moderate_threshold, 1.0 + extreme_threshold], // Moderate Greed: Somewhat more bullish
        [1.0 + extreme_threshold, f64::INFINITY],            // Strong Greed: Much more bullish
    ];
    // Calculate representative bullish ratios for each class (midpoints)
    // These ranges are used to determine expected values for reconstruction
    let class_ratio_midpoints = [
        (1.0 - extreme_threshold) * 0.75, // Strong Panic
        (1.0 - extreme_threshold + 1.0 - moderate_threshold) / 2.0, // Moderate Panic
        1.0,                              // Neutral (same as sequence)
        (1.0 + moderate_threshold + 1.0 + extreme_threshold) / 2.0, // Moderate Greed
        (1.0 + extreme_threshold) * 1.25, // Strong Greed
    ];

    // Convert ratio ranges to sentiment ranges (for compatibility)
    let sentiment_ranges: Vec<[f64; 2]> = ratio_ranges
        .iter()
        .map(|&[low, high]| {
            let low_sentiment = if low == 0.0 {
                0.0
            } else {
                low * baseline_strength
            };
            let high_sentiment = if high == f64::INFINITY {
                1.0
            } else {
                high * baseline_strength
            };
            [low_sentiment, high_sentiment]
        })
        .collect();

    // Find most likely class
    let most_likely_class = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(2); // Default to neutral

    let confidence = probabilities[most_likely_class];

    // Calculate expected bullish ratio (weighted average)
    let expected_ratio = probabilities
        .iter()
        .zip(class_ratio_midpoints.iter())
        .map(|(prob, ratio)| prob * ratio)
        .sum::<f64>();

    // Convert expected ratio back to sentiment score for compatibility
    let expected_sentiment = expected_ratio * baseline_strength;

    // Generate interpretation
    let class_names = get_sentiment_class_names();
    let sentiment_interpretation = format!(
        "{} (confidence: {:.1}%, ratio: {:.3})",
        class_names[most_likely_class],
        confidence * 100.0,
        class_ratio_midpoints[most_likely_class]
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
