//! Sentiment Analysis using Volume-Price Divergence
//!
//! # 🎯 TARGET PURPOSE: "WHAT IS THE MARKET CONVICTION?"
//!
//! This module implements **volume-price divergence analysis** for sentiment detection.
//! It answers: "Is there accumulation (buying pressure) or distribution (selling pressure)?"
//!
//! ## 📊 MATHEMATICAL FOUNDATION
//!
//! ### **Core Logic: Volume-Price Divergence Score**
//! ```
//! 1. Calculate normalized price change: (horizon_price - sequence_price) / sequence_price
//! 2. Calculate normalized volume change: ln(horizon_volume / sequence_volume)
//! 3. Compute divergence: volume_change - price_change
//! 4. Positive divergence = Accumulation (volume > price movement)
//! 5. Negative divergence = Distribution (volume < price movement)
//! 6. Classify using adaptive thresholds (sensitivity, extreme_multiplier)
//! ```
//!
//! ### **Why This Works (Orthogonal to Direction Target):**
//! - **Independent Signal**: Direction measures price momentum, sentiment measures volume-price relationship
//! - **Accumulation/Distribution**: Classic market theory - volume leads price
//! - **Learnable Pattern**: Divergence predicts reversals and continuations
//! - **Market-Grounded**: Based on actual smart money behavior
//! - **Minimal Parameters**: Only 2 (sensitivity, extreme_multiplier) like volatility
//!
//! ### **5-Class Sentiment Classification:**
//! - **0: STRONG_DISTRIBUTION** - High volume, price falling (bearish conviction)
//! - **1: MODERATE_DISTRIBUTION** - Volume exceeds price drop (mild distribution)
//! - **2: NEUTRAL** - Volume matches price movement (balanced)
//! - **3: MODERATE_ACCUMULATION** - Volume exceeds price rise (mild accumulation)
//! - **4: STRONG_ACCUMULATION** - High volume, price rising (bullish conviction)

use crate::data::structures::MarketDataRow;
use crate::targets::TargetResult;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use std::collections::HashMap;

/// Volume-price divergence metrics
#[derive(Debug, Clone)]
pub struct VolumePriceDivergence {
    /// Normalized price change (percentage)
    pub price_change: f64,

    /// Normalized volume change (log-ratio)
    pub volume_change: f64,

    /// Divergence score (volume_change - price_change)
    pub divergence_score: f64,

    /// Average price for reference
    pub avg_price: f64,

    /// Average volume for reference
    pub avg_volume: f64,
}

/// Calculate volume-price divergence for a period
/// Returns normalized price change, volume change, and divergence score
pub fn calculate_period_metrics(candles: &[MarketDataRow]) -> Result<VolumePriceDivergence> {
    if candles.is_empty() {
        return Err(VangaError::DataError(
            "Cannot calculate metrics from empty candles".to_string(),
        ));
    }

    // Calculate average price (VWAP for better representation)
    let mut total_volume = 0.0;
    let mut vwap_sum = 0.0;

    for candle in candles {
        let typical_price = (candle.high + candle.low + candle.close) / 3.0;
        vwap_sum += typical_price * candle.volume;
        total_volume += candle.volume;
    }

    let safe_volume = total_volume.max(1.0);
    let avg_price = vwap_sum / safe_volume;
    let avg_volume = total_volume / candles.len() as f64;

    // Calculate price change (percentage)
    let first_price = (candles[0].open + candles[0].close) / 2.0;
    let last_price = (candles[candles.len() - 1].open + candles[candles.len() - 1].close) / 2.0;
    let price_change = if first_price > 0.0 {
        (last_price - first_price) / first_price
    } else {
        0.0
    };

    Ok(VolumePriceDivergence {
        price_change,
        volume_change: 0.0, // Will be calculated in divergence function
        divergence_score: 0.0,
        avg_price,
        avg_volume,
    })
}

/// Calculate volume-price divergence between two periods
/// Returns divergence score: positive = accumulation, negative = distribution
pub fn calculate_divergence_score(
    sequence_metrics: &VolumePriceDivergence,
    horizon_metrics: &VolumePriceDivergence,
) -> f64 {
    // Calculate volume ratio (log-space for symmetry)
    let volume_ratio = horizon_metrics.avg_volume / sequence_metrics.avg_volume.max(1.0);
    let volume_change = volume_ratio.ln();

    // Price change is already normalized (percentage)
    let price_change = horizon_metrics.price_change;

    // Divergence score: when volume increases more than price, it's accumulation
    // When volume decreases or price moves without volume, it's distribution
    volume_change - price_change
}

/// Calculate sentiment classification thresholds (like volatility)
fn calculate_sentiment_thresholds(
    calibrated_params: &crate::targets::calibration::SentimentParams,
) -> (f64, f64) {
    let moderate_threshold = calibrated_params.sensitivity;
    let extreme_threshold = moderate_threshold * calibrated_params.extreme_multiplier;
    (moderate_threshold, extreme_threshold)
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
    let timeframe_minutes = crate::utils::parser::detect_timeframe_minutes(df)?;
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
            "  Horizon {}: sensitivity={:.4}",
            horizon,
            params.sensitivity
        );

        let horizon_steps =
            crate::utils::parser::parse_horizon_to_steps(horizon, timeframe_minutes)?;
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

/// Classify sentiment using volume-price divergence
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

    // Calculate metrics for both periods
    let sequence_metrics = calculate_period_metrics(sequence_ohlcv)?;
    let horizon_metrics = calculate_period_metrics(horizon_ohlcv)?;

    // Calculate divergence score
    let divergence_score = calculate_divergence_score(&sequence_metrics, &horizon_metrics);

    // Use adaptive thresholds for classification
    let (moderate_threshold, extreme_threshold) = calculate_sentiment_thresholds(calibrated_params);

    // Classify based on divergence score
    // Negative = Distribution (selling pressure), Positive = Accumulation (buying pressure)
    let class = if divergence_score <= -extreme_threshold {
        0 // STRONG DISTRIBUTION: High volume, price falling
    } else if divergence_score <= -moderate_threshold {
        1 // MODERATE DISTRIBUTION: Volume exceeds price drop
    } else if divergence_score < moderate_threshold {
        2 // NEUTRAL: Volume matches price movement
    } else if divergence_score < extreme_threshold {
        3 // MODERATE ACCUMULATION: Volume exceeds price rise
    } else {
        4 // STRONG ACCUMULATION: High volume, price rising
    };

    // Calculate classification strength (distance from boundaries)
    let strength = calculate_sentiment_strength(
        divergence_score,
        moderate_threshold,
        extreme_threshold,
        class,
    );

    log::debug!(
        "🎭 Sentiment (Divergence): seq_price={:.4}%, hor_price={:.4}%, seq_vol={:.0}, hor_vol={:.0}, divergence={:.4}, thresholds=[{:.4}, {:.4}] → class={} ({}) strength={:.3}",
        sequence_metrics.price_change * 100.0,
        horizon_metrics.price_change * 100.0,
        sequence_metrics.avg_volume,
        horizon_metrics.avg_volume,
        divergence_score,
        moderate_threshold,
        extreme_threshold,
        class,
        ["VERY_BEARISH", "BEARISH", "NEUTRAL", "BULLISH", "VERY_BULLISH"][class as usize],
        strength
    );

    Ok((class, strength))
}

/// Calculate classification strength based on distance from boundaries
fn calculate_sentiment_strength(
    divergence_score: f64,
    moderate_threshold: f64,
    extreme_threshold: f64,
    class: i32,
) -> f64 {
    match class {
        0 => {
            // STRONG DISTRIBUTION: divergence_score <= -extreme_threshold
            let distance_beyond = (-divergence_score - extreme_threshold).max(0.0);
            let max_distance = extreme_threshold;
            (distance_beyond / max_distance).clamp(0.1, 1.0)
        }
        1 => {
            // MODERATE DISTRIBUTION: -extreme_threshold < divergence_score <= -moderate_threshold
            let range_center = -(extreme_threshold + moderate_threshold) / 2.0;
            let range_half_width = (extreme_threshold - moderate_threshold) / 2.0;
            let distance_from_center = (divergence_score - range_center).abs();
            let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
            strength.max(0.1)
        }
        2 => {
            // NEUTRAL: -moderate_threshold < divergence_score < moderate_threshold
            let distance_from_zero = divergence_score.abs();
            let strength = 1.0 - (distance_from_zero / moderate_threshold).min(1.0);
            strength.max(0.1)
        }
        3 => {
            // MODERATE ACCUMULATION: moderate_threshold <= divergence_score < extreme_threshold
            let range_center = (moderate_threshold + extreme_threshold) / 2.0;
            let range_half_width = (extreme_threshold - moderate_threshold) / 2.0;
            let distance_from_center = (divergence_score - range_center).abs();
            let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
            strength.max(0.1)
        }
        4 => {
            // STRONG ACCUMULATION: divergence_score >= extreme_threshold
            let distance_beyond = (divergence_score - extreme_threshold).max(0.0);
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

/// Log sentiment class distribution
fn log_sentiment_distribution(targets: &[i32], horizon: &str) {
    let class_names = [
        "VERY_BEARISH",
        "BEARISH",
        "NEUTRAL",
        "BULLISH",
        "VERY_BULLISH",
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
        "VERY_BEARISH",
        "BEARISH",
        "NEUTRAL",
        "BULLISH",
        "VERY_BULLISH",
    ]
}

// ============================================================================
// PREDICTION RECONSTRUCTION METHODS
// ============================================================================

/// Reconstruction result for sentiment predictions
#[derive(Debug, Clone)]
pub struct SentimentReconstruction {
    /// Divergence score ranges for each class [lower_bound, upper_bound]
    pub divergence_ranges: Vec<[f64; 2]>,
    /// Class probabilities from model
    pub probabilities: Vec<f64>,
    /// Most likely class index
    pub most_likely_class: usize,
    /// Confidence (probability of most likely class)
    pub confidence: f64,
    /// Expected divergence score (weighted average)
    pub expected_divergence_score: f64,
    /// Sentiment interpretation
    pub sentiment_interpretation: String,
}

/// Reconstruct sentiment from model probabilities
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

    // Use calibrated parameters for threshold calculation (same as classification)
    let (moderate_threshold, extreme_threshold) = calculate_sentiment_thresholds(calibrated_params);

    // Calculate actual sequence metrics for real divergence score
    let sequence_metrics = calculate_period_metrics(sequence_ohlcv)?;

    // Define divergence score ranges for each class (symmetric)
    let divergence_ranges = [
        [-f64::INFINITY, -extreme_threshold],
        [-extreme_threshold, -moderate_threshold],
        [-moderate_threshold, moderate_threshold],
        [moderate_threshold, extreme_threshold],
        [extreme_threshold, f64::INFINITY],
    ];

    // Calculate representative divergence scores for each class (midpoints)
    let class_divergence_midpoints = [
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

    // Calculate expected divergence score (weighted average)
    let expected_divergence_score = probabilities
        .iter()
        .zip(class_divergence_midpoints.iter())
        .map(|(prob, divergence)| prob * divergence)
        .sum::<f64>();

    // Generate interpretation with actual sequence context
    let class_names = get_sentiment_class_names();
    let sentiment_interpretation = format!(
        "{} (confidence: {:.1}%, divergence: {:.3}, seq_vol: {:.0}, seq_price: {:.4}%)",
        class_names[most_likely_class],
        confidence * 100.0,
        class_divergence_midpoints[most_likely_class],
        sequence_metrics.avg_volume,
        sequence_metrics.price_change * 100.0
    );

    Ok(SentimentReconstruction {
        divergence_ranges: divergence_ranges.to_vec(),
        probabilities: probabilities.to_vec(),
        most_likely_class,
        confidence,
        expected_divergence_score,
        sentiment_interpretation,
    })
}
