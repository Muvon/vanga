//! Real Sentiment Analysis using Candle Psychology
//!
//! # 🎯 TARGET PURPOSE: "WHAT IS THE MARKET SENTIMENT?"
//!
//! This module implements **real candle psychology analysis** for market sentiment detection.
//! It answers: "Is the market showing fear, greed, or neutral sentiment based on candle patterns?"
//!
//! ## 📊 MATHEMATICAL FOUNDATION
//!
//! ### **Core Logic: Candle Body Psychology Analysis**
//! ```
//! 1. Body Conviction: (close - open) / (high - low) - directional strength
//! 2. Body Size: abs(close - open) / typical_price - magnitude
//! 3. Wick Imbalance: (upper_wick - lower_wick) / (high - low) - pressure
//! 4. Volume Conviction: ln(volume_ratio) * body_conviction - participation
//! 5. Combine into sentiment score with calibrated weights
//! ```
//!
//! ### **5-Class Sentiment Classification:**
//! - **0: STRONG_FEAR** - Large red bodies, long lower wicks, high volume
//! - **1: MODERATE_FEAR** - Medium red bodies, mixed wicks, moderate bearish
//! - **2: NEUTRAL** - Small bodies (doji-like), balanced wicks, indecision
//! - **3: MODERATE_GREED** - Medium green bodies, mixed wicks, moderate bullish
//! - **4: STRONG_GREED** - Large green bodies, long upper wicks, high volume

use crate::data::structures::MarketDataRow;
use crate::targets::TargetResult;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use std::collections::HashMap;

/// Sentiment features extracted from candle analysis
#[derive(Debug, Clone)]
pub struct SentimentFeatures {
    /// Body conviction: (close - open) / (high - low)
    /// Range: [-1, 1] where -1 = full red body, +1 = full green body
    pub body_conviction: f64,

    /// Body size: abs(close - open) / typical_price
    /// Range: [0, ~0.05] normalized by price
    pub body_size: f64,

    /// Wick imbalance: (upper_wick - lower_wick) / (high - low)
    /// Range: [-1, 1] where -1 = all lower wick, +1 = all upper wick
    pub wick_imbalance: f64,

    /// Volume conviction: ln(volume_ratio) * body_conviction
    /// Range: [-2, 2] combines volume strength with direction
    pub volume_conviction: f64,
}

/// Calculate sentiment features for a single candle
pub fn calculate_candle_sentiment_features(
    candle: &MarketDataRow,
    _prev_candle: Option<&MarketDataRow>,
    avg_volume: f64,
) -> SentimentFeatures {
    let open = candle.open;
    let high = candle.high;
    let low = candle.low;
    let close = candle.close;
    let volume = candle.volume;

    // Typical price for normalization
    let typical_price = (high + low + close) / 3.0;
    let range = high - low;

    // Avoid division by zero
    let safe_range = range.max(typical_price * 0.001); // Min 0.1% range
    let safe_volume = avg_volume.max(1.0);

    // 1. Body conviction (directional strength)
    let body_conviction = (close - open) / safe_range;

    // 2. Body size (magnitude of conviction)
    let body_size = if typical_price > 0.0 {
        (close - open).abs() / typical_price
    } else {
        0.0
    };

    // 3. Wick imbalance (buying vs selling pressure)
    let upper_wick = high - close.max(open);
    let lower_wick = open.min(close) - low;
    let wick_imbalance = (upper_wick - lower_wick) / safe_range;

    // 4. Volume conviction (volume-weighted direction)
    let volume_ratio = volume / safe_volume;
    let volume_conviction = volume_ratio.ln().clamp(-2.0, 2.0) * body_conviction;

    SentimentFeatures {
        body_conviction,
        body_size,
        wick_imbalance,
        volume_conviction,
    }
}

/// Aggregate sentiment features across multiple candles with exponential weighting
pub fn aggregate_sentiment_features(candles: &[MarketDataRow]) -> Result<SentimentFeatures> {
    if candles.is_empty() {
        return Err(VangaError::DataError(
            "Cannot aggregate sentiment from empty candles".to_string(),
        ));
    }

    // Calculate average volume for normalization
    let avg_volume = candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64;

    // Calculate features for each candle
    let mut all_features = Vec::new();
    for i in 0..candles.len() {
        let prev = if i > 0 { Some(&candles[i - 1]) } else { None };
        let features = calculate_candle_sentiment_features(&candles[i], prev, avg_volume);
        all_features.push(features);
    }

    // Aggregate with exponential weighting (recent candles more important)
    let decay_factor: f64 = 0.9;
    let mut total_weight = 0.0;
    let mut weighted_features = SentimentFeatures {
        body_conviction: 0.0,
        body_size: 0.0,
        wick_imbalance: 0.0,
        volume_conviction: 0.0,
    };

    for (i, features) in all_features.iter().enumerate() {
        let weight = decay_factor.powi((all_features.len() - i - 1) as i32);
        total_weight += weight;

        weighted_features.body_conviction += features.body_conviction * weight;
        weighted_features.body_size += features.body_size * weight;
        weighted_features.wick_imbalance += features.wick_imbalance * weight;
        weighted_features.volume_conviction += features.volume_conviction * weight;
    }

    // Normalize by total weight
    if total_weight > 0.0 {
        weighted_features.body_conviction /= total_weight;
        weighted_features.body_size /= total_weight;
        weighted_features.wick_imbalance /= total_weight;
        weighted_features.volume_conviction /= total_weight;
    }

    Ok(weighted_features)
}

/// Calculate composite sentiment score from features
pub fn calculate_sentiment_score(
    features: &SentimentFeatures,
    calibrated_params: &crate::targets::calibration::SentimentParams,
) -> f64 {
    // Weighted combination of all features
    features.body_conviction * calibrated_params.body_weight
        + features.body_size * calibrated_params.size_weight * features.body_conviction.signum()
        + features.wick_imbalance * calibrated_params.wick_weight
        + features.volume_conviction * calibrated_params.volume_weight
}

/// Generate sentiment targets with calibrated parameters - returns both class and strength
pub fn generate_sentiment_targets_with_calibrated_params(
    df: &DataFrame,
    horizons: &[String],
    sequence_indices: &[usize],
    sequence_length: usize,
    calibrated_params: &std::collections::HashMap<
        String,
        crate::targets::calibration::SentimentParams,
    >,
) -> Result<TargetResult> {
    log::info!("🎯 Generating sentiment targets with per-horizon calibrated parameters");
    let ohlcv_data = extract_ohlcv_data(df)?;
    let mut targets = HashMap::new();
    let mut strengths = HashMap::new();

    for horizon in horizons {
        let params = calibrated_params.get(horizon).ok_or_else(|| {
            crate::utils::error::VangaError::ConfigError(format!(
                "No calibrated sentiment parameters found for horizon: {}",
                horizon
            ))
        })?;

        log::debug!(
            "  Horizon {}: body_weight={:.2}, sensitivity={:.4}",
            horizon,
            params.body_weight,
            params.sensitivity
        );

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

            match classify_sentiment_with_calibrated_params(sequence_data, horizon_data, params) {
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

/// Classify sentiment using real candle psychology
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

    // Calculate sentiment features for both periods
    let sequence_features = aggregate_sentiment_features(sequence_ohlcv)?;
    let horizon_features = aggregate_sentiment_features(horizon_ohlcv)?;

    // Calculate sentiment scores
    let sequence_score = calculate_sentiment_score(&sequence_features, calibrated_params);
    let horizon_score = calculate_sentiment_score(&horizon_features, calibrated_params);

    // Sentiment change (how sentiment is shifting)
    let sentiment_change = horizon_score - sequence_score;

    // Use adaptive thresholds for classification
    let moderate_threshold = calibrated_params.sensitivity;
    let extreme_threshold = moderate_threshold * calibrated_params.extreme_multiplier;

    // Classify based on sentiment change
    let class = if sentiment_change <= -extreme_threshold {
        0 // STRONG FEAR: Major sentiment deterioration
    } else if sentiment_change <= -moderate_threshold {
        1 // MODERATE FEAR: Sentiment weakening
    } else if sentiment_change < moderate_threshold {
        2 // NEUTRAL: Sentiment stable
    } else if sentiment_change < extreme_threshold {
        3 // MODERATE GREED: Sentiment improving
    } else {
        4 // STRONG GREED: Major sentiment improvement
    };

    // Calculate classification strength
    let strength = calculate_sentiment_strength(
        sentiment_change,
        moderate_threshold,
        extreme_threshold,
        class,
    );

    log::debug!(
        "🎭 Real Sentiment: seq_score={:.4}, hor_score={:.4}, change={:.4}, thresholds=[{:.4}, {:.4}] → class={} ({}) strength={:.3}",
        sequence_score, horizon_score, sentiment_change,
        moderate_threshold, extreme_threshold,
        class, ["STRONG_FEAR", "MODERATE_FEAR", "NEUTRAL", "MODERATE_GREED", "STRONG_GREED"][class as usize],
        strength
    );

    Ok((class, strength))
}

/// Calculate classification strength based on distance from boundaries
fn calculate_sentiment_strength(
    sentiment_change: f64,
    moderate_threshold: f64,
    extreme_threshold: f64,
    class: i32,
) -> f64 {
    match class {
        0 => {
            // STRONG FEAR: sentiment_change <= -extreme_threshold
            let distance_beyond = (-sentiment_change - extreme_threshold).max(0.0);
            let max_distance = extreme_threshold;
            (distance_beyond / max_distance).clamp(0.1, 1.0)
        }
        1 => {
            // MODERATE FEAR: -extreme_threshold < sentiment_change <= -moderate_threshold
            let range_center = -(extreme_threshold + moderate_threshold) / 2.0;
            let range_half_width = (extreme_threshold - moderate_threshold) / 2.0;
            let distance_from_center = (sentiment_change - range_center).abs();
            let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
            strength.max(0.1)
        }
        2 => {
            // NEUTRAL: -moderate_threshold < sentiment_change < moderate_threshold
            let distance_from_zero = sentiment_change.abs();
            let strength = 1.0 - (distance_from_zero / moderate_threshold).min(1.0);
            strength.max(0.1)
        }
        3 => {
            // MODERATE GREED: moderate_threshold <= sentiment_change < extreme_threshold
            let range_center = (moderate_threshold + extreme_threshold) / 2.0;
            let range_half_width = (extreme_threshold - moderate_threshold) / 2.0;
            let distance_from_center = (sentiment_change - range_center).abs();
            let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
            strength.max(0.1)
        }
        4 => {
            // STRONG GREED: sentiment_change >= extreme_threshold
            let distance_beyond = (sentiment_change - extreme_threshold).max(0.0);
            let max_distance = extreme_threshold;
            (distance_beyond / max_distance).clamp(0.1, 1.0)
        }
        _ => 0.5,
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
                timestamp: i as i64,
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
        "STRONG_FEAR",
        "MODERATE_FEAR",
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
}

/// Get sentiment class names in order
pub fn get_sentiment_class_names() -> Vec<&'static str> {
    vec![
        "STRONG_FEAR",
        "MODERATE_FEAR",
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
    /// Expected sentiment change (weighted average)
    pub expected_sentiment_change: f64,
    /// Sentiment interpretation
    pub sentiment_interpretation: String,
}

/// Reconstruct sentiment from model probabilities
pub fn reconstruct_sentiment(
    probabilities: &[f64],
    _sequence_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::SentimentParams,
) -> Result<SentimentReconstruction> {
    if probabilities.len() != 5 {
        return Err(VangaError::DataError(
            "Expected 5 sentiment probabilities".to_string(),
        ));
    }

    // Use calibrated parameters for threshold calculation
    let moderate_threshold = calibrated_params.sensitivity;
    let extreme_threshold = moderate_threshold * calibrated_params.extreme_multiplier;

    // Define sentiment change ranges for each class
    let sentiment_ranges = [
        [-f64::INFINITY, -extreme_threshold],
        [-extreme_threshold, -moderate_threshold],
        [-moderate_threshold, moderate_threshold],
        [moderate_threshold, extreme_threshold],
        [extreme_threshold, f64::INFINITY],
    ];

    // Calculate representative sentiment changes for each class (midpoints)
    let class_sentiment_midpoints = [
        -extreme_threshold - (extreme_threshold - moderate_threshold) / 2.0,
        -(extreme_threshold + moderate_threshold) / 2.0,
        0.0,
        (moderate_threshold + extreme_threshold) / 2.0,
        extreme_threshold + (extreme_threshold - moderate_threshold) / 2.0,
    ];

    // Find most likely class
    let most_likely_class = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(2);

    // UNIFIED CONFIDENCE CALCULATION
    let max_prob = probabilities.iter().fold(0.0_f64, |a, &b| a.max(b));
    let confidence = crate::output::confidence_calculator::calibrate_5_class_confidence(max_prob);

    // Calculate expected sentiment change (weighted average)
    let expected_sentiment_change = probabilities
        .iter()
        .zip(class_sentiment_midpoints.iter())
        .map(|(prob, sentiment)| prob * sentiment)
        .sum::<f64>();

    // Generate interpretation
    let class_names = get_sentiment_class_names();
    let sentiment_interpretation = format!(
        "{} (confidence: {:.1}%, change: {:.3})",
        class_names[most_likely_class],
        confidence * 100.0,
        class_sentiment_midpoints[most_likely_class]
    );

    Ok(SentimentReconstruction {
        sentiment_ranges: sentiment_ranges.to_vec(),
        probabilities: probabilities.to_vec(),
        most_likely_class,
        confidence,
        expected_sentiment_change,
        sentiment_interpretation,
    })
}
