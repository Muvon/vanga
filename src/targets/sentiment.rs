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

use crate::data::structures::MarketDataRow;
use crate::targets::TargetResult;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
/// Generate sentiment targets with calibrated parameters - returns both class and strength
/// target generation between training and prediction. When None, uses base config.
pub fn generate_sentiment_targets_with_calibrated_params(
    df: &DataFrame,
    horizons: &[String],
    sequence_indices: &[usize],
    sequence_length: usize,
    calibrated_params: &crate::targets::calibration::SentimentParams, // Now mandatory
) -> Result<TargetResult> {
    // Use pre-calibrated parameters (always available)
    log::info!(
        "🎯 Using calibrated sentiment parameters: body_sensitivity={:.6}, volume_weight={:.3}, consistency_factor={:.3}",
        calibrated_params.body_sensitivity,
        calibrated_params.volume_weight,
        calibrated_params.consistency_factor
    );

    let ohlcv_data = extract_ohlcv_data(df)?;
    let mut targets = HashMap::new();
    let mut strengths = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_steps(horizon)?;
        let mut horizon_targets = Vec::new();
        let mut horizon_strengths = Vec::new();

        for &seq_start in sequence_indices {
            let seq_end = seq_start + sequence_length;
            let horizon_end = seq_end + horizon_steps;

            if horizon_end > ohlcv_data.len() {
                continue;
            }

            let sequence_data = &ohlcv_data[seq_start..seq_end];
            let horizon_data = &ohlcv_data[seq_end..horizon_end];

            match classify_sentiment_with_calibrated_params(
                sequence_data,
                horizon_data,
                calibrated_params,
            ) {
                Ok((class, strength)) => {
                    horizon_targets.push(class);
                    horizon_strengths.push(strength);
                }
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
            strengths.insert(horizon.clone(), horizon_strengths);
        }
    }

    Ok((targets, strengths))
}

/// Classify sentiment using simple price-volume momentum for strong ML signals
///
/// SIMPLE APPROACH: Direct price momentum with volume conviction - research-based and ML-friendly
/// Research: "Momentum is the difference between current price and price N periods ago"
///
/// ## Algorithm
/// 1. Calculate average close price for sequence (baseline) and horizon (target)
/// 2. Calculate price momentum = (horizon_price - sequence_price) / sequence_price
/// 3. Calculate volume conviction = horizon_volume / sequence_volume
/// 4. Combine momentum with volume conviction for final sentiment score
/// 5. Classify using adaptive thresholds on combined score
///
/// ## Signal Strength
/// - **Strong Signal**: Direct momentum values with clear positive/negative meaning
/// - **Volume Conviction**: Higher volume amplifies momentum signals
/// - **ML Friendly**: Unbounded numerical range perfect for LSTM learning
pub fn classify_sentiment_with_calibrated_params(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::SentimentParams,
) -> Result<(i32, f64)> {
    if sequence_ohlcv.is_empty() || horizon_ohlcv.is_empty() {
        return Err(VangaError::DataError(
            "Empty sequence or horizon OHLCV data for sentiment analysis".to_string(),
        ));
    }

    // Calculate simple price momentum (research-based approach)
    let sequence_price = calculate_bullish_strength(sequence_ohlcv); // Now returns avg close
    let horizon_price = calculate_bullish_strength(horizon_ohlcv); // Now returns avg close

    // Price momentum: percentage change from sequence to horizon
    let price_momentum = if sequence_price > 0.0 {
        (horizon_price - sequence_price) / sequence_price
    } else {
        0.0 // Avoid division by zero
    };

    // Calculate volume conviction factor
    let sequence_volume = calculate_volume_conviction(sequence_ohlcv);
    let horizon_volume = calculate_volume_conviction(horizon_ohlcv);

    // Volume conviction: ratio of horizon to sequence volume
    let volume_conviction = if sequence_volume > 0.0 {
        (horizon_volume / sequence_volume).ln().clamp(-2.0, 2.0) // Log scale, clamped
    } else {
        0.0 // Neutral if no volume data
    };

    // Combine momentum with volume conviction (momentum is primary signal)
    let sentiment_score = price_momentum + (volume_conviction * calibrated_params.volume_weight);

    // Use adaptive thresholds for classification (same structure as before)
    let moderate_threshold = calibrated_params.body_sensitivity; // Now represents momentum threshold
    let extreme_threshold =
        calibrated_params.body_sensitivity * calibrated_params.extreme_multiplier;

    // Classify based on combined sentiment score
    let class = if sentiment_score <= -extreme_threshold {
        0 // Strong Panic: Large negative momentum
    } else if sentiment_score <= -moderate_threshold {
        1 // Moderate Panic: Moderate negative momentum
    } else if sentiment_score < moderate_threshold {
        2 // Neutral: Small momentum in either direction
    } else if sentiment_score < extreme_threshold {
        3 // Moderate Greed: Moderate positive momentum
    } else {
        4 // Strong Greed: Large positive momentum
    };

    // Calculate classification strength based on distance from boundaries
    let strength = calculate_sentiment_strength(
        sentiment_score,
        moderate_threshold,
        extreme_threshold,
        class,
    );

    log::debug!(
        "🎯 Simple Momentum: seq_price={:.4}, hor_price={:.4}, momentum={:.4}, vol_conviction={:.4}, score={:.4}, thresholds=[{:.4}, {:.4}, {:.4}, {:.4}] → class={} ({}) strength={:.3}",
        sequence_price, horizon_price, price_momentum, volume_conviction, sentiment_score,
        -extreme_threshold, -moderate_threshold, moderate_threshold, extreme_threshold,
        class, ["STRONG_PANIC", "MODERATE_PANIC", "NEUTRAL", "MODERATE_GREED", "STRONG_GREED"][class as usize], strength
    );

    Ok((class, strength))
}

/// Calculate classification strength for sentiment based on distance from boundaries
///
/// Strength represents how confident/strong the classification is:
/// - 1.0 = Very strong (deep in the center of the class range)
/// - 0.5 = Moderate (near class boundaries)
/// - 0.0 = Very weak (just barely in the class)
fn calculate_sentiment_strength(
    sentiment_score: f64,
    moderate_threshold: f64,
    extreme_threshold: f64,
    class: i32,
) -> f64 {
    match class {
        0 => {
            // Strong Panic: sentiment_score <= -extreme_threshold
            // The more negative beyond extreme, the stronger
            let distance_beyond = (-sentiment_score - extreme_threshold).max(0.0);
            let max_distance = extreme_threshold; // Reasonable max distance
            (distance_beyond / max_distance).clamp(0.1, 1.0) // At least 0.1 strength
        }
        1 => {
            // Moderate Panic: -extreme_threshold < sentiment_score <= -moderate_threshold
            let range_center = -(extreme_threshold + moderate_threshold) / 2.0;
            let range_half_width = (extreme_threshold - moderate_threshold) / 2.0;
            let distance_from_center = (sentiment_score - range_center).abs();
            let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
            strength.max(0.1) // At least 0.1 strength
        }
        2 => {
            // Neutral: -moderate_threshold < sentiment_score < moderate_threshold
            // Closer to zero = stronger neutral signal
            let distance_from_zero = sentiment_score.abs();
            let strength = 1.0 - (distance_from_zero / moderate_threshold).min(1.0);
            strength.max(0.1) // At least 0.1 strength
        }
        3 => {
            // Moderate Greed: moderate_threshold <= sentiment_score < extreme_threshold
            let range_center = (moderate_threshold + extreme_threshold) / 2.0;
            let range_half_width = (extreme_threshold - moderate_threshold) / 2.0;
            let distance_from_center = (sentiment_score - range_center).abs();
            let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
            strength.max(0.1) // At least 0.1 strength
        }
        4 => {
            // Strong Greed: sentiment_score >= extreme_threshold
            // The more positive beyond extreme, the stronger
            let distance_beyond = (sentiment_score - extreme_threshold).max(0.0);
            let max_distance = extreme_threshold; // Reasonable max distance
            (distance_beyond / max_distance).clamp(0.1, 1.0) // At least 0.1 strength
        }
        _ => 0.5, // Default neutral strength
    }
}

/// Calculate average close price for momentum calculation
///
/// SIMPLE APPROACH: Returns average close price for momentum calculation
/// Used in classify_sentiment_with_calibrated_params for sequence vs horizon comparison
pub fn calculate_bullish_strength(candles: &[MarketDataRow]) -> f64 {
    if candles.is_empty() {
        return 0.0; // Neutral if no data
    }

    // Calculate average close price (simple and effective)
    candles.iter().map(|c| c.close).sum::<f64>() / candles.len() as f64
}

/// Calculate average volume for conviction factor
///
/// SIMPLE APPROACH: Returns average volume for volume conviction calculation
/// Higher volume = higher conviction in price moves
pub fn calculate_volume_conviction(candles: &[MarketDataRow]) -> f64 {
    if candles.is_empty() {
        return 1.0; // Neutral conviction if no data
    }

    // Calculate average volume
    candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64
}

/// Legacy function for backward compatibility - now uses simple momentum approach
pub fn calculate_sequence_sentiment_score(candles: &[MarketDataRow]) -> f64 {
    // For legacy compatibility, return simple price momentum relative to first candle
    if candles.len() < 2 {
        return 0.0; // Neutral if insufficient data
    }

    let first_price = candles[0].close;
    let last_price = candles[candles.len() - 1].close;

    if first_price > 0.0 {
        // Return momentum as percentage change (compatible with [-1, 1] range expected by legacy code)
        let momentum = (last_price - first_price) / first_price;
        momentum.clamp(-1.0, 1.0) // Clamp to expected range
    } else {
        0.0 // Neutral if invalid price
    }
}

/// Legacy function for backward compatibility - now uses simple momentum approach
/// Calculate sentiment score with optional horizon decay weighting
///
/// Note: The horizon_decay_factor parameter is kept for API compatibility but not used
/// in the simplified momentum-based approach. Will be removed in future version.
pub fn calculate_sequence_sentiment_score_with_weighting(
    candles: &[MarketDataRow],
    _horizon_decay_factor: f64, // Kept for API compatibility, not used in momentum-based approach
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

/// Calibrate sentiment sensitivity for balanced class distribution using momentum approach
///
/// OPTIMAL APPROACH: Iteratively tests different parameter combinations to find the best
/// balance for momentum-based sentiment classification. Now calibrates ALL parameters.
///
/// ## Algorithm
/// 1. Calculate all momentum and volume data from historical sequences
/// 2. Test multiple combinations of sensitivity, volume_weight, and extreme_multiplier
/// 3. For each combination, simulate classification and measure balance quality
/// 4. Return parameters that achieve closest to 20% per class distribution
///
/// ## Parameters
/// - `ohlcv_data`: Historical OHLCV data for momentum analysis
/// - `sequence_length`: Length of input sequences
/// - `horizon_steps`: Number of steps in prediction horizon
/// - `target_balance`: Target percentage per class (0.2 for 20% per class)
///
/// ## Returns
/// Calibrated body_sensitivity parameter (other optimal params available via get_optimal_* functions)
pub fn calibrate_sentiment_sensitivity(
    ohlcv_data: &[MarketDataRow],
    sequence_length: usize,
    horizon_steps: usize,
    target_balance: f64,
) -> Result<f64> {
    let (sensitivity, volume_weight, extreme_multiplier) = calibrate_all_sentiment_parameters(
        ohlcv_data,
        sequence_length,
        horizon_steps,
        target_balance,
    )?;

    // Store optimal parameters in thread-local storage for retrieval
    OPTIMAL_PARAMS.with(|params| {
        let mut p = params.borrow_mut();
        p.volume_weight = volume_weight;
        p.extreme_multiplier = extreme_multiplier;
    });

    Ok(sensitivity)
}

/// Internal structure for optimal parameters
#[derive(Debug, Clone)]
struct OptimalParams {
    volume_weight: f64,
    extreme_multiplier: f64,
}

impl Default for OptimalParams {
    fn default() -> Self {
        Self {
            volume_weight: 0.1,
            extreme_multiplier: 2.0,
        }
    }
}

// Thread-local storage for optimal parameters (safer than global unsafe)
use std::cell::RefCell;
thread_local! {
    static OPTIMAL_PARAMS: RefCell<OptimalParams> = RefCell::new(OptimalParams::default());
}

/// Get the optimal volume weight found during calibration
pub fn get_optimal_volume_weight() -> f64 {
    OPTIMAL_PARAMS.with(|params| params.borrow().volume_weight)
}

/// Get the optimal extreme multiplier found during calibration
pub fn get_optimal_extreme_multiplier() -> f64 {
    OPTIMAL_PARAMS.with(|params| params.borrow().extreme_multiplier)
}

/// Internal function that calibrates all parameters and returns them as tuple
fn calibrate_all_sentiment_parameters(
    ohlcv_data: &[MarketDataRow],
    sequence_length: usize,
    horizon_steps: usize,
    target_balance: f64,
) -> Result<(f64, f64, f64)> {
    if ohlcv_data.len() < sequence_length + horizon_steps + 50 {
        return Ok((0.02, 0.1, 2.0)); // Default parameters
    }

    // Collect all momentum and volume data for testing
    let mut test_data = Vec::new();

    for i in 0..(ohlcv_data.len() - sequence_length - horizon_steps) {
        let sequence_ohlcv = &ohlcv_data[i..i + sequence_length];
        let horizon_ohlcv = &ohlcv_data[i + sequence_length..i + sequence_length + horizon_steps];

        if sequence_ohlcv.len() >= 3 && horizon_ohlcv.len() >= 3 {
            let sequence_price = calculate_bullish_strength(sequence_ohlcv);
            let horizon_price = calculate_bullish_strength(horizon_ohlcv);
            let sequence_volume = calculate_volume_conviction(sequence_ohlcv);
            let horizon_volume = calculate_volume_conviction(horizon_ohlcv);

            if sequence_price > 0.0 && sequence_volume > 0.0 {
                let momentum = (horizon_price - sequence_price) / sequence_price;
                let volume_conviction = (horizon_volume / sequence_volume).ln().clamp(-2.0, 2.0);

                if momentum.is_finite() && volume_conviction.is_finite() {
                    test_data.push((momentum, volume_conviction));
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
        "🔍 Testing {} parameter combinations for optimal sentiment calibration...",
        sensitivity_candidates.len()
            * volume_weight_candidates.len()
            * extreme_multiplier_candidates.len()
    );

    for &sensitivity in &sensitivity_candidates {
        for &volume_weight in &volume_weight_candidates {
            for &extreme_multiplier in &extreme_multiplier_candidates {
                // Test this parameter combination
                let balance_score = test_parameter_combination_full(
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
        "🎯 Optimal sentiment parameters: sensitivity={:.4}, volume_weight={:.3}, extreme_multiplier={:.2}, balance_score={:.4}",
        best_sensitivity, best_volume_weight, best_extreme_multiplier, best_balance_score
    );

    Ok((
        best_sensitivity,
        best_volume_weight,
        best_extreme_multiplier,
    ))
}

/// Test a specific parameter combination and return balance quality score
fn test_parameter_combination_full(
    test_data: &[(f64, f64)], // (momentum, volume_conviction) pairs
    sensitivity: f64,
    volume_weight: f64,
    extreme_multiplier: f64,
    target_balance: f64,
) -> f64 {
    let mut class_counts = [0usize; 5];
    let extreme_threshold = sensitivity * extreme_multiplier; // Now calibrated!

    // Classify all test samples with these parameters
    for &(momentum, volume_conviction) in test_data {
        let sentiment_score = momentum + (volume_conviction * volume_weight);

        let class = if sentiment_score <= -extreme_threshold {
            0 // Strong Panic
        } else if sentiment_score <= -sensitivity {
            1 // Moderate Panic
        } else if sentiment_score < sensitivity {
            2 // Neutral
        } else if sentiment_score < extreme_threshold {
            3 // Moderate Greed
        } else {
            4 // Strong Greed
        };

        class_counts[class] += 1;
    }

    // Calculate balance quality score (lower is better)
    let total_samples = test_data.len() as f64;

    let mut balance_score = 0.0f64;

    for count in class_counts {
        let actual_percentage = (count as f64) / total_samples;
        let deviation = (actual_percentage - target_balance).abs();
        balance_score += deviation * deviation; // Squared deviation penalty
    }

    // Log best candidates for debugging
    if balance_score < 0.05 {
        // Only log very good candidates
        let percentages: Vec<String> = class_counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                format!(
                    "{}:{:.1}%",
                    ["SP", "MP", "N", "MG", "SG"][i],
                    (count as f64 / total_samples) * 100.0
                )
            })
            .collect();

        log::debug!(
            "🔍 Good candidate: sens={:.4}, vol_wt={:.3}, ext_mult={:.2}, score={:.4}, dist=[{}]",
            sensitivity,
            volume_weight,
            extreme_multiplier,
            balance_score,
            percentages.join(", ")
        );
    }

    balance_score
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

/// Reconstruct sentiment from model probabilities using momentum-based approach
///
/// This method reverses the new momentum-based classification logic to convert
/// raw model probabilities back to meaningful price momentum and sentiment metrics.
///
/// # Arguments
/// * `probabilities` - 5-element array of class probabilities [Strong Panic, Moderate Panic, Neutral, Moderate Greed, Strong Greed]
/// * `sequence_ohlcv` - OHLCV data for the input sequence (same as used in training)
/// * `adaptive_params` - Adaptive parameters used during training (for threshold calculation)
///
/// # Returns
/// * `SentimentReconstruction` - Complete reconstruction with momentum values and sentiment metrics
pub fn reconstruct_sentiment(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::SentimentParams,
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

    // Calculate sequence baseline price (same as training)
    let sequence_price = calculate_bullish_strength(sequence_ohlcv); // Now returns avg close

    // Use calibrated parameters for threshold calculation (same as training)
    let moderate_threshold = calibrated_params.body_sensitivity;
    let extreme_threshold =
        calibrated_params.body_sensitivity * calibrated_params.extreme_multiplier;

    // Define momentum ranges for each class (reverse of classification logic)
    let momentum_ranges = [
        [-f64::INFINITY, -extreme_threshold], // Strong Panic: Large negative momentum
        [-extreme_threshold, -moderate_threshold], // Moderate Panic: Moderate negative momentum
        [-moderate_threshold, moderate_threshold], // Neutral: Small momentum
        [moderate_threshold, extreme_threshold], // Moderate Greed: Moderate positive momentum
        [extreme_threshold, f64::INFINITY],   // Strong Greed: Large positive momentum
    ];

    // Calculate representative momentum values for each class (midpoints)
    let class_momentum_midpoints = [
        -extreme_threshold * 1.5,                        // Strong Panic
        -(extreme_threshold + moderate_threshold) / 2.0, // Moderate Panic
        0.0,                                             // Neutral
        (moderate_threshold + extreme_threshold) / 2.0,  // Moderate Greed
        extreme_threshold * 1.5,                         // Strong Greed
    ];

    // Convert momentum ranges to sentiment ranges (for compatibility)
    let sentiment_ranges: Vec<[f64; 2]> = momentum_ranges
        .iter()
        .map(|&[low, high]| {
            // Convert momentum to expected price ranges
            let low_price = if low == -f64::INFINITY {
                sequence_price * 0.5 // Assume 50% drop for extreme panic
            } else {
                sequence_price * (1.0 + low)
            };
            let high_price = if high == f64::INFINITY {
                sequence_price * 2.0 // Assume 100% gain for extreme greed
            } else {
                sequence_price * (1.0 + high)
            };
            [low_price.max(0.0), high_price.max(0.0)]
        })
        .collect();

    // Find most likely class
    let most_likely_class = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(2); // Default to neutral

    // UNIFIED CONFIDENCE CALCULATION
    // All targets now use the same research-based calibration function
    // This eliminates target-specific confidence inflation and ensures consistency
    let max_prob = probabilities.iter().fold(0.0_f64, |a, &b| a.max(b));
    let confidence = crate::output::confidence_calculator::calibrate_5_class_confidence(max_prob);

    // Calculate expected momentum (weighted average)
    let expected_momentum = probabilities
        .iter()
        .zip(class_momentum_midpoints.iter())
        .map(|(prob, momentum)| prob * momentum)
        .sum::<f64>();

    // Convert expected momentum back to sentiment score for compatibility
    let expected_sentiment = expected_momentum; // Direct momentum value

    // Generate interpretation
    let class_names = get_sentiment_class_names();
    let sentiment_interpretation = format!(
        "{} (confidence: {:.1}%, momentum: {:.3}%)",
        class_names[most_likely_class],
        confidence * 100.0,
        class_momentum_midpoints[most_likely_class] * 100.0
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
