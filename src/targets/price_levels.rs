//! Price level target generation for cryptocurrency forecasting
//!
//! # 🎯 TARGET PURPOSE: "WHERE WILL PRICE BE?"
//!
//! This module implements **exponentially-weighted range analysis** for support/resistance breakout detection.
//! It answers: "Will the future price break above/below the recent trading range?"
//!
//! ## 📊 MATHEMATICAL FOUNDATION
//!
//! ### **Core Logic: Range Boundary Analysis**
//! ```
//! 1. Calculate exponentially-weighted close prices for input sequence (recent-focused)
//! 2. Find sequence_min and sequence_max from close prices
//! 3. Calculate target exponentially-weighted close price from horizon period
//! 4. Apply bandwidth expansion for breakout sensitivity
//! 5. Classify target price relative to expanded range boundaries
//! ```
//!
//! ### **5-Class Classification System:**
//! - **0: Strong Down** - `target < sequence_min - bandwidth` (Support breakdown)
//! - **1: Moderate Down** - `sequence_min - bandwidth ≤ target < sequence_min` (Below range)
//! - **2: Neutral** - `sequence_min ≤ target < sequence_max` (Within range)
//! - **3: Moderate Up** - `sequence_max ≤ target < sequence_max + bandwidth` (Above range)
//! - **4: Strong Up** - `target ≥ sequence_max + bandwidth` (Resistance breakout)
//!
//! ## 🔧 KEY FEATURES
//!
//! ### **Exponential Weighting Integration**
//! - Uses exponentially-weighted close prices instead of volume-weighted prices
//! - Provides more reliable price representation by emphasizing recent price action
//! - No dependency on potentially manipulated volume data
//!
//! ### **Bandwidth Sensitivity Control**
//! - `bandwidth_size`: Controls breakout sensitivity (default: 1.0)
//! - Smaller values (0.5): More sensitive to small breakouts
//! - Larger values (1.5): Requires stronger moves for breakout classification
//!
//! ### **Symbol-Agnostic Design**
//! - Works with any price range (BTC, ETH, altcoins)
//! - Bandwidth calculated as percentage of sequence range
//! - No hardcoded price thresholds
//!
//! ## 🎯 COMPLEMENTARY ROLE
//!
//! **Price Levels** work with other targets:
//! - **+ Direction**: Range breakout + trend acceleration = strong signal
//! - **+ Volatility**: Range breakout + high volatility = significant move
//! - **Training**: Provides "where" while others provide "how" and "risk"

use crate::data::structures::MarketDataRow;
use crate::targets::sequence_reconstruction::{SequenceAnalyzer, SequenceReconstructionConfig};
use crate::utils::error::Result;
use crate::utils::market_data::extract_ohlcv_data;
use crate::utils::parser::parse_horizon_to_steps;
use polars::prelude::*;
use std::collections::HashMap;

/// Configuration for price level target generation
#[derive(Debug, Clone)]
pub struct PriceLevelConfig {
    /// Bandwidth multiplier for breakout sensitivity (default: 1.0)
    /// - 0.5: More sensitive (smaller breakout thresholds)
    /// - 1.0: Standard behavior
    /// - 1.5: Less sensitive (larger breakout thresholds)
    pub bandwidth_size: f64,

    /// Neutral band factor for symmetric neutral zone (default: 0.4)
    /// - 0.2: Small neutral zone (20% of percentile range)
    /// - 0.4: Balanced neutral zone (40% of percentile range)
    /// - 0.6: Large neutral zone (60% of percentile range)
    pub neutral_band_factor: f64,
}

impl Default for PriceLevelConfig {
    fn default() -> Self {
        Self {
            bandwidth_size: 1.0,
            neutral_band_factor: 0.4, // 40% of percentile range becomes neutral zone
        }
    }
}

impl PriceLevelConfig {
    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.bandwidth_size <= 0.0 {
            return Err(crate::utils::error::VangaError::ConfigError(format!(
                "bandwidth_size must be positive, got: {}",
                self.bandwidth_size
            )));
        }

        if !self.bandwidth_size.is_finite() {
            return Err(crate::utils::error::VangaError::ConfigError(
                "bandwidth_size must be a finite number".to_string(),
            ));
        }

        if self.neutral_band_factor <= 0.0 || self.neutral_band_factor >= 1.0 {
            return Err(crate::utils::error::VangaError::ConfigError(format!(
                "neutral_band_factor must be between 0.0 and 1.0, got: {}",
                self.neutral_band_factor
            )));
        }

        if !self.neutral_band_factor.is_finite() {
            return Err(crate::utils::error::VangaError::ConfigError(
                "neutral_band_factor must be a finite number".to_string(),
            ));
        }

        Ok(())
    }
}

/// Generate price level targets using PriceLevelConfig
pub fn generate_price_level_targets_with_calibrated_params(
    df: &DataFrame,
    horizons: &[String],
    sequence_indices: &[usize],
    sequence_length: usize,
    calibrated_params: &crate::targets::calibration::PriceLevelParams,
) -> Result<HashMap<String, Vec<i32>>> {
    log::info!(
        "🎯 Using calibrated price levels parameters: bandwidth={:.6}, percentiles=[{:.2}, {:.2}], neutral_band_factor={:.2}, fallback=[{:.2}, {:.2}]",
        calibrated_params.bandwidth,
        calibrated_params.percentiles[0],
        calibrated_params.percentiles[1],
        calibrated_params.neutral_band_factor,
        calibrated_params.fallback_percentiles[0],
        calibrated_params.fallback_percentiles[1]
    );
    let ohlcv_data = extract_ohlcv_data(df)?;
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let mut horizon_targets = vec![-1; sequence_indices.len()];

        for (seq_position, &seq_idx) in sequence_indices.iter().enumerate() {
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            if target_end_idx <= ohlcv_data.len() && sequence_end_idx <= ohlcv_data.len() {
                // Sequence-to-horizon data flow (same pattern as direction/volatility)
                let sequence_ohlcv = &ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_ohlcv = &ohlcv_data[sequence_end_idx..target_end_idx];

                // Use enhanced classification with calibrated parameters
                let target_class = classify_price_level_with_calibrated_params(
                    sequence_ohlcv,
                    horizon_ohlcv,
                    calibrated_params,
                )?;
                horizon_targets[seq_position] = target_class;
            }
        }

        // Analyze and log class distribution (5 classes) - exponentially-weighted approach
        let valid_targets: Vec<i32> = horizon_targets
            .iter()
            .filter(|&&x| x != -1)
            .cloned()
            .collect();
        if !valid_targets.is_empty() {
            analyze_class_distribution(&valid_targets, horizon, 5)?;
        }

        targets.insert(horizon.clone(), horizon_targets);
    }

    Ok(targets)
}

/// Get exponentially-weighted close price from OHLCV data (replaces VWAP)
///
/// # 🎯 EXPONENTIAL WEIGHTING LOGIC
///
/// **Recent-Focused Weighting**: Recent prices get exponentially more weight than older prices
///
/// **Formula**: `EWP = Σ(close_price × weight) / Σ(weight)`
/// Where: `weight = (position / length)^2.0` (exponential focus on recent data)
///
/// **Benefits**:
/// - No volume dependency (volume can be faked)
/// - Recent prices matter more (reflects current market sentiment)
/// - Simple and reliable calculation
/// - More responsive to recent price action
///
/// **Weighting Example** (5 candles):
/// - Candle 1 (oldest): weight = (1/5)^2 = 0.04
/// - Candle 2: weight = (2/5)^2 = 0.16
/// - Candle 3: weight = (3/5)^2 = 0.36
/// - Candle 4: weight = (4/5)^2 = 0.64
/// - Candle 5 (newest): weight = (5/5)^2 = 1.00
///
/// **Fallback Logic**:
/// - If insufficient data (< 2 candles): Returns 0.0
/// - If single candle: Returns that candle's close price
pub fn get_sequence_exponential_weighted_close(sequence_ohlcv: &[MarketDataRow]) -> Result<f64> {
    if sequence_ohlcv.is_empty() {
        return Ok(0.0);
    }

    if sequence_ohlcv.len() == 1 {
        return Ok(sequence_ohlcv[0].close);
    }

    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    let len = sequence_ohlcv.len() as f64;

    for (i, candle) in sequence_ohlcv.iter().enumerate() {
        // Exponential weighting: recent data gets much more weight
        let weight = ((i as f64 + 1.0) / len).powf(2.0);
        weighted_sum += candle.close * weight;
        total_weight += weight;
    }

    Ok(weighted_sum / total_weight)
}

/// Get horizon exponentially-weighted close price (same calculation as sequence)
pub fn get_horizon_exponential_weighted_close(horizon_ohlcv: &[MarketDataRow]) -> Result<f64> {
    // Same calculation as sequence (exponentially-weighted close price)
    get_sequence_exponential_weighted_close(horizon_ohlcv)
}

/// Calculate exponentially-weighted center price from sequence data
///
/// This is the same as get_sequence_exponential_weighted_close but with a more descriptive name
/// for use in adaptive percentile calculations. The exponentially-weighted center represents
/// the recent-focused "fair value" price during the sequence period, emphasizing recent price action.
pub fn calculate_exponential_weighted_center(sequence_ohlcv: &[MarketDataRow]) -> Result<f64> {
    get_sequence_exponential_weighted_close(sequence_ohlcv)
}

/// Extract individual OHLC4 prices from sequence data
///
/// Returns a vector of OHLC4 prices (not volume-weighted) for distribution analysis.
/// Each price represents the average of open, high, low, close for that candle.
pub fn extract_sequence_prices(sequence_ohlcv: &[MarketDataRow]) -> Vec<f64> {
    sequence_ohlcv
        .iter()
        .map(|candle| (candle.open + candle.high + candle.low + candle.close) / 4.0)
        .collect()
}

/// Calculate adaptive percentiles based on sequence price distribution around volume-weighted center
///
/// # 🎯 ADAPTIVE PERCENTILE ALGORITHM
///
/// **Core Logic**: Analyze how prices are distributed around the volume-weighted center
/// to determine optimal percentile boundaries for classification.
///
/// **Steps**:
/// 1. Calculate volume-weighted center (fair value price)
/// 2. Partition prices into below/above center groups
/// 3. Determine adaptive percentiles based on distribution balance
/// 4. Apply volatility-based adjustments
///
/// **Adaptive Logic**:
/// - **Balanced distribution** (40-60% below center): Standard percentiles [0.1, 0.9]
/// - **Bottom-heavy** (>60% below center): Tighter lower, wider upper [0.15, 0.95]
/// - **Top-heavy** (<40% below center): Wider lower, tighter upper [0.05, 0.85]
/// - **High volatility**: Expand both boundaries for noise tolerance
/// - **Low volatility**: Contract boundaries for sensitivity
///
/// **Benefits**:
/// - Automatic adaptation to each sequence's characteristics
/// - Volume-aware (uses actual trading activity)
/// - Volatility-responsive (tight for low vol, wide for high vol)
/// - No hardcoded magic numbers
///
/// # Arguments
/// * `sequence_ohlcv` - OHLCV data for the sequence
/// * `fallback_percentiles` - Calibrated fallback percentiles (replaces hardcoded [0.1, 0.9])
///
/// # Returns
/// * `Result<[f64; 2]>` - Adaptive percentiles [lower, upper] optimized for the sequence
pub fn calculate_adaptive_percentiles_from_sequence(
    sequence_ohlcv: &[MarketDataRow],
    fallback_percentiles: Option<[f64; 2]>,
) -> Result<[f64; 2]> {
    if sequence_ohlcv.len() < 5 {
        // Use calibrated fallback percentiles instead of hardcoded [0.1, 0.9]
        return Ok(fallback_percentiles.unwrap_or([0.1, 0.9]));
    }

    // 1. Calculate exponentially-weighted center (recent-focused fair value)
    let exponential_weighted_center = calculate_exponential_weighted_center(sequence_ohlcv)?;

    // 2. Extract individual OHLC4 prices for distribution analysis
    let sequence_prices = extract_sequence_prices(sequence_ohlcv);

    // 3. Partition prices around the volume-weighted center
    let below_center_count = sequence_prices
        .iter()
        .filter(|&&price| price < exponential_weighted_center)
        .count();

    let below_center_ratio = below_center_count as f64 / sequence_prices.len() as f64;

    // 4. Calculate coefficient of variation for volatility adjustment
    let price_mean = sequence_prices.iter().sum::<f64>() / sequence_prices.len() as f64;
    let price_variance = sequence_prices
        .iter()
        .map(|&p| (p - price_mean).powi(2))
        .sum::<f64>()
        / sequence_prices.len() as f64;
    let coefficient_of_variation = if price_mean > 1e-8 {
        price_variance.sqrt() / price_mean
    } else {
        0.02 // Default 2% volatility
    };

    // 5. Determine base percentiles based on distribution balance
    let (base_lower, base_upper): (f64, f64) = match below_center_ratio {
        ratio if ratio < 0.4 => (0.05, 0.85), // Top-heavy: wider lower, tighter upper
        ratio if ratio > 0.6 => (0.15, 0.95), // Bottom-heavy: tighter lower, wider upper
        _ => (0.1, 0.9),                      // Balanced: standard percentiles
    };

    // 6. Apply volatility-based adjustments
    let volatility_adjustment: f64 = match coefficient_of_variation {
        cv if cv < 0.01 => 0.05, // Low volatility: contract boundaries (more sensitive)
        cv if cv > 0.04 => -0.05, // High volatility: expand boundaries (less sensitive)
        _ => 0.0,                // Medium volatility: no adjustment
    };

    let adaptive_lower = (base_lower + volatility_adjustment).clamp(0.02, 0.25);
    let adaptive_upper = (base_upper - volatility_adjustment).clamp(0.75, 0.98);

    // 7. Debug logging for transparency
    log::debug!(
        "🎯 Adaptive Percentiles: center={:.6}, below_ratio={:.2}, cv={:.3}, base=[{:.2}, {:.2}], adaptive=[{:.2}, {:.2}]",
        exponential_weighted_center,
        below_center_ratio,
        coefficient_of_variation,
        base_lower,
        base_upper,
        adaptive_lower,
        adaptive_upper
    );

    Ok([adaptive_lower, adaptive_upper])
}

/// **ENHANCED**: Calculate adaptive bandwidth based on sequence volatility characteristics
///
/// # 🎯 ADAPTIVE BANDWIDTH CALCULATION
///
/// **Mathematical Foundation**: Adjusts bandwidth based on sequence price volatility to ensure
/// consistent classification difficulty across different market regimes and volatility periods.
///
/// **Algorithm**:
/// 1. Calculate sequence price volatility (coefficient of variation)
/// 2. Compare to baseline volatility (2% for crypto markets)
/// 3. Scale bandwidth by volatility ratio with bounds
/// 4. Apply momentum weighting if specified
///
/// **Volatility Scaling Logic**:
/// - High volatility periods: Larger bandwidth (less sensitive to noise)
/// - Low volatility periods: Smaller bandwidth (more sensitive to small moves)
/// - Extreme volatility: Capped to prevent over-adjustment
///
/// **Benefits**:
/// - Consistent classification across market regimes
/// - Automatic adaptation to volatility clustering
/// - Maintains ~20% class distribution target
/// - Prevents over-fitting to specific volatility periods
pub fn calculate_adaptive_bandwidth(
    sequence_ohlcv: &[MarketDataRow],
    base_bandwidth: f64,
    momentum_factor: Option<f64>,
) -> Result<f64> {
    if sequence_ohlcv.len() < 3 {
        return Ok(base_bandwidth); // Fallback for insufficient data
    }

    // 1. Extract sequence prices for volatility calculation
    let prices: Vec<f64> = sequence_ohlcv
        .iter()
        .map(|c| (c.open + c.high + c.low + c.close) / 4.0)
        .collect();

    // 2. Calculate price volatility (coefficient of variation)
    let price_mean = prices.iter().sum::<f64>() / prices.len() as f64;
    let price_variance = prices
        .iter()
        .map(|&p| (p - price_mean).powi(2))
        .sum::<f64>()
        / prices.len() as f64;
    let price_std = price_variance.sqrt();
    let coefficient_of_variation = if price_mean > 1e-8 {
        price_std / price_mean
    } else {
        0.02 // Default 2% volatility for edge cases
    };

    // 3. Calculate volatility multiplier with bounds
    let baseline_volatility = 0.02; // 2% baseline for crypto markets
    let volatility_ratio = coefficient_of_variation / baseline_volatility;
    let volatility_multiplier = volatility_ratio.clamp(0.3, 3.0); // Prevent extreme adjustments

    // 4. Apply momentum weighting if specified
    let final_multiplier = if let Some(momentum) = momentum_factor {
        if momentum > 1.0 {
            // Higher momentum factor = more recent data weight = potentially higher volatility
            let momentum_adjustment = 1.0 + (momentum - 1.0) * 0.2; // 20% max adjustment
            volatility_multiplier * momentum_adjustment
        } else {
            volatility_multiplier
        }
    } else {
        volatility_multiplier
    };

    // 5. Calculate adaptive bandwidth
    let adaptive_bandwidth = base_bandwidth * final_multiplier;

    log::debug!(
        "🎯 Adaptive Bandwidth: base={:.3}, volatility_ratio={:.3}, multiplier={:.3}, adaptive={:.3}",
        base_bandwidth,
        volatility_ratio,
        final_multiplier,
        adaptive_bandwidth
    );

    Ok(adaptive_bandwidth)
}

pub fn classify_price_level_with_calibrated_params(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::PriceLevelParams,
) -> Result<i32> {
    if sequence_ohlcv.is_empty() {
        return Ok(2); // Default to neutral class
    }

    // 1. Calculate sequence exponentially-weighted close
    let seq_exponential_weighted = get_sequence_exponential_weighted_close(sequence_ohlcv)?;

    // 2. Calculate horizon exponentially-weighted close
    let hor_exponential_weighted = get_horizon_exponential_weighted_close(horizon_ohlcv)?;

    // 3. Use adaptive percentiles from calibrated params
    let percentiles = calculate_adaptive_percentiles_from_sequence(
        sequence_ohlcv,
        Some(calibrated_params.fallback_percentiles),
    )?;
    let base_bandwidth_size = 1.0;

    // 4. Calculate adaptive bandwidth with momentum factor from calibrated params
    let momentum_factor = Some(calibrated_params.momentum_factor);
    let final_bandwidth_size =
        calculate_adaptive_bandwidth(sequence_ohlcv, base_bandwidth_size, momentum_factor)?;

    // 5. Use enhanced sequence reconstruction with calibrated parameters
    let reconstruction_config = SequenceReconstructionConfig {
        percentiles,
        bandwidth_size: final_bandwidth_size,
        neutral_band_factor: calibrated_params.neutral_band_factor, // Use calibrated parameter
    };
    let analyzer = SequenceAnalyzer::new(reconstruction_config);

    // 6. Calculate boundaries using centralized logic
    let boundaries = analyzer.calculate_boundaries(sequence_ohlcv)?;

    // 7. Handle edge case: flat sequence
    if boundaries.bandwidth == 0.0 {
        return Ok(if hor_exponential_weighted >= boundaries.sequence_min {
            3
        } else {
            2
        });
    }

    // 8. Enhanced debug logging
    log::debug!(
        "🚀 Price Level with Exponential Weighting: seq_exponential={:.6}, hor_exponential={:.6}, final_bandwidth={:.3}",
        seq_exponential_weighted,
        hor_exponential_weighted,
        final_bandwidth_size
    );

    // 9. Use centralized classification logic
    let class = boundaries.classify_price(hor_exponential_weighted);

    log::debug!(
        "🎯 Price Level Result: target={:.6} → class={} (adaptive_range: [{:.6}, {:.6}], adaptive_bandwidth: {:.6})",
        hor_exponential_weighted, class, boundaries.sequence_min, boundaries.sequence_max, boundaries.bandwidth
    );

    Ok(class)
}

/// Analyze class distribution and log insights for debugging
fn analyze_class_distribution(targets: &[i32], horizon: &str, bins: u32) -> Result<()> {
    let mut class_counts = vec![0usize; bins as usize];
    let mut valid_targets = 0;

    // Count class occurrences
    for &target in targets {
        if target >= 0 && target < bins as i32 {
            class_counts[target as usize] += 1;
            valid_targets += 1;
        }
    }

    if valid_targets == 0 {
        log::warn!(
            "📊 Price Level Analysis [{}]: No valid targets found",
            horizon
        );
        return Ok(());
    }

    // Calculate class distribution statistics
    let total_samples = valid_targets as f64;
    let mut class_percentages = Vec::new();
    let mut min_class_size = usize::MAX;
    let mut max_class_size = 0;

    for &count in class_counts.iter() {
        let percentage = (count as f64 / total_samples) * 100.0;
        class_percentages.push(percentage);

        if count > 0 {
            min_class_size = min_class_size.min(count);
            max_class_size = max_class_size.max(count);
        }
    }

    // Calculate imbalance ratio
    let imbalance_ratio = if min_class_size != usize::MAX && min_class_size > 0 {
        max_class_size as f64 / min_class_size as f64
    } else {
        f64::INFINITY
    };

    // Log compact class distribution analysis
    log::info!(
        "📊 Price Level Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}] (BEFORE balanced selection)",
        horizon,
        valid_targets,
        imbalance_ratio,
        class_percentages
            .iter()
            .enumerate()
            .map(|(i, p)| format!("{}:{:.1}%", i, p))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Check for problematic imbalance
    if imbalance_ratio > 100.0 {
        log::warn!(
            "⚠️  Severe class imbalance detected for {} ({:.0}x ratio) - consider class weighting",
            horizon,
            imbalance_ratio
        );
    }

    // Identify empty classes
    let empty_classes: Vec<usize> = class_counts
        .iter()
        .enumerate()
        .filter(|(_, &count)| count == 0)
        .map(|(idx, _)| idx)
        .collect();

    if !empty_classes.is_empty() {
        log::warn!(
            "⚠️  Empty classes detected for {}: {:?} - may cause training instability",
            horizon,
            empty_classes
        );
    }

    Ok(())
}

// ============================================================================
// PREDICTION RECONSTRUCTION METHODS
// ============================================================================

/// Reconstruction result for price level predictions
#[derive(Debug, Clone)]
pub struct PriceLevelReconstruction {
    /// Percentage ranges for each class [lower_bound, upper_bound]
    pub percentage_ranges: Vec<[f64; 2]>,
    /// Absolute price ranges for each class [lower_price, upper_price]
    pub price_ranges: Vec<[f64; 2]>,
    /// Exponentially-weighted close relative percentage ranges for each class [lower_bound, upper_bound]
    pub exponential_weighted_percentage_ranges: Vec<[f64; 2]>,
    /// Class probabilities from model
    pub probabilities: Vec<f64>,
    /// Most likely class index
    pub most_likely_class: usize,
    /// Confidence (probability of most likely class)
    pub confidence: f64,
    /// Expected price change percentage (weighted average)
    pub expected_change_percent: f64,
    /// Sequence boundaries used for calculation
    pub sequence_min: f64,
    pub sequence_max: f64,
    pub bandwidth: f64,
}

/// Reconstruct price level predictions from model probabilities
///
/// This method reverses the training classification logic to convert
/// raw model probabilities back to meaningful price ranges and percentages.
///
/// # Arguments
/// * `probabilities` - 5-element array of class probabilities from model
/// * `sequence_ohlcv` - OHLCV data for the input sequence (same as used in training)
/// * `current_price` - Current price for percentage calculations
/// * `config` - Optional configuration (uses defaults if None)
///
/// # Returns
/// * `PriceLevelReconstruction` - Complete reconstruction with ranges and metrics
pub fn reconstruct_price_levels(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    current_price: f64,
    calibrated_params: &crate::targets::calibration::PriceLevelParams,
) -> Result<PriceLevelReconstruction> {
    // Validate inputs
    if probabilities.len() != 5 {
        return Err(crate::utils::error::VangaError::DataError(
            "Price level reconstruction requires exactly 5 class probabilities".to_string(),
        ));
    }

    if sequence_ohlcv.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Sequence OHLCV data is required for price level reconstruction".to_string(),
        ));
    }

    if current_price <= 0.0 {
        return Err(crate::utils::error::VangaError::DataError(
            "Current price must be positive for percentage calculations".to_string(),
        ));
    }

    // Use calibrated parameters
    let percentiles = calibrated_params.percentiles;
    let bandwidth_size = calibrated_params.bandwidth;

    // Calculate adaptive bandwidth (same logic as training)
    let final_bandwidth_size = calculate_adaptive_bandwidth(
        sequence_ohlcv,
        bandwidth_size,
        None, // No momentum factor needed with exponential weighting
    )?;

    // Use centralized sequence reconstruction logic (same as training)
    let reconstruction_config = SequenceReconstructionConfig {
        percentiles,
        bandwidth_size: final_bandwidth_size,
        neutral_band_factor: calibrated_params.neutral_band_factor, // Use calibrated parameter
    };
    let analyzer = SequenceAnalyzer::new(reconstruction_config);
    let boundaries = analyzer.calculate_boundaries(sequence_ohlcv)?;

    // Calculate sequence exponentially-weighted close (this is our reference center)
    let sequence_exponential_weighted = get_sequence_exponential_weighted_close(sequence_ohlcv)?;

    // Calculate percentage ranges relative to sequence center
    // This ensures neutral zone is properly centered around 0%
    let percentage_ranges_from_center =
        boundaries.get_price_level_ranges(sequence_exponential_weighted);

    // Convert percentage ranges to be relative to current price for display
    // This shows where the levels are relative to current market price
    let percentage_ranges: Vec<[f64; 2]> = percentage_ranges_from_center
        .iter()
        .map(|[lower_pct, upper_pct]| {
            // Convert from sequence-center-relative to current-price-relative
            let lower_price_abs = sequence_exponential_weighted * (1.0 + lower_pct / 100.0);
            let upper_price_abs = sequence_exponential_weighted * (1.0 + upper_pct / 100.0);
            let lower_pct_from_current =
                ((lower_price_abs - current_price) / current_price) * 100.0;
            let upper_pct_from_current =
                ((upper_price_abs - current_price) / current_price) * 100.0;
            [lower_pct_from_current, upper_pct_from_current]
        })
        .collect();

    // Calculate absolute price ranges
    // These are the actual price levels based on the sequence analysis
    let price_ranges: Vec<[f64; 2]> = percentage_ranges_from_center
        .iter()
        .map(|[lower_pct, upper_pct]| {
            // Convert from percentages to absolute prices using sequence center
            let lower_price = sequence_exponential_weighted * (1.0 + lower_pct / 100.0);
            let upper_price = sequence_exponential_weighted * (1.0 + upper_pct / 100.0);
            [lower_price, upper_price]
        })
        .collect();

    // The exponentially-weighted percentage ranges are the same as from center
    let exponential_weighted_percentage_ranges: Vec<[f64; 2]> = percentage_ranges_from_center;

    // Find most likely class and confidence
    let (most_likely_class, confidence) = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &prob)| (idx, prob))
        .unwrap_or((2, 0.2)); // Default to neutral class

    // Calculate expected price change (weighted average of class midpoints)
    let class_midpoints: Vec<f64> = percentage_ranges
        .iter()
        .map(|[lower, upper]| (lower + upper) / 2.0)
        .collect();

    let expected_change_percent: f64 = probabilities
        .iter()
        .zip(class_midpoints.iter())
        .map(|(&prob, &midpoint)| prob * midpoint)
        .sum();

    Ok(PriceLevelReconstruction {
        percentage_ranges,
        price_ranges,
        exponential_weighted_percentage_ranges,
        probabilities: probabilities.to_vec(),
        most_likely_class,
        confidence,
        expected_change_percent,
        sequence_min: boundaries.sequence_min,
        sequence_max: boundaries.sequence_max,
        bandwidth: boundaries.bandwidth,
    })
}
