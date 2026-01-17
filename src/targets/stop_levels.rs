//! Stop level target generation for cryptocurrency forecasting
//!
//! # 🎯 TARGET PURPOSE: "WHAT'S THE MAXIMUM ADVERSE EXCURSION (MAE)?"
//!
//! This module implements **Maximum Adverse Excursion (MAE) analysis** for risk management.
//! It answers: "What's the worst drawdown I should expect during the price movement?"
//!
//! ## 📊 MATHEMATICAL FOUNDATION
//!
//! ### **Core Logic: MAE-Based Risk Classification**
//! ```
//! 1. Calculate sequence reference price (exponentially-weighted close)
//! 2. Find Maximum Adverse Excursion (MAE) during horizon:
//!    - MAE = maximum drawdown from reference price
//!    - Always measures downside risk (reference - lowest_price)
//! 3. Normalize MAE by sequence volatility range
//! 4. Classify MAE severity into 5 risk levels
//! ```
//!
//! ### **5-Class Risk Classification System:**
//! - **0: Extreme Risk** - MAE > 200% of sequence range (massive drawdown)
//! - **1: High Risk** - MAE 100-200% of sequence range (large drawdown)
//! - **2: Moderate Risk** - MAE 50-100% of sequence range (normal drawdown)
//! - **3: Low Risk** - MAE 25-50% of sequence range (small drawdown)
//! - **4: Minimal Risk** - MAE < 25% of sequence range (very safe)
//!
//! ## 🔧 KEY FEATURES
//!
//! ### **MAE (Maximum Adverse Excursion)**
//! - Industry-standard metric for stop-loss placement
//! - Measures worst intrabar drawdown from entry point
//! - Direction-independent (always measures downside risk)
//! - Exponentially weighted for recent-focused analysis
//!
//! ### **Symbol-Agnostic Design**
//! - MAE normalized by sequence volatility range
//! - Works with any price range (BTC, ETH, altcoins)
//! - No hardcoded price thresholds
//!
//! ### **Trading-Oriented Classification**
//! - Class 4 (Minimal): Safe trades, tight stops work
//! - Class 3 (Low): Normal trades, standard stops
//! - Class 2 (Moderate): Wider stops needed
//! - Class 1 (High): Very wide stops or avoid
//! - Class 0 (Extreme): Stop-loss will trigger, high risk
//!
//! ## 🎯 COMPLEMENTARY ROLE
//!
//! **Stop Levels** work with **Price Levels**:
//! - **Price Levels**: "Where will price END UP?" (destination)
//! - **Stop Levels**: "What's the WORST DRAWDOWN along the way?" (risk)
//! - **Combined**: Complete risk-reward picture for trading decisions

use crate::data::structures::MarketDataRow;
use crate::targets::TargetResult;
use crate::utils::error::Result;
use crate::utils::market_data::extract_ohlcv_data;
use crate::utils::parser::parse_horizon_to_steps;
use polars::prelude::*;
use std::collections::HashMap;

/// Configuration for stop level target generation
#[derive(Debug, Clone)]
pub struct StopLevelConfig {
    /// Bandwidth multiplier for extreme sensitivity (default: 1.0)
    pub bandwidth_size: f64,
    /// Neutral band factor for symmetric neutral zone (default: 0.4)
    pub neutral_band_factor: f64,
}

impl Default for StopLevelConfig {
    fn default() -> Self {
        Self {
            bandwidth_size: 1.0,
            neutral_band_factor: 0.4,
        }
    }
}

impl StopLevelConfig {
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

/// Generate stop level targets using calibrated parameters - returns both class and strength
pub fn generate_stop_level_targets_with_calibrated_params(
    df: &DataFrame,
    horizons: &[String],
    sequence_indices: &[usize],
    sequence_length: usize,
    calibrated_params: &HashMap<String, crate::targets::calibration::StopLevelParams>,
) -> Result<TargetResult> {
    log::info!("🛑 Generating stop level targets with per-horizon calibrated parameters");
    let timeframe_minutes = crate::utils::parser::detect_timeframe_minutes(df)?;
    let ohlcv_data = extract_ohlcv_data(df)?;
    let mut targets = HashMap::new();
    let mut strengths = HashMap::new();

    for horizon in horizons {
        let params = calibrated_params.get(horizon).ok_or_else(|| {
            crate::utils::error::VangaError::ConfigError(format!(
                "No calibrated stop level parameters found for horizon: {}",
                horizon
            ))
        })?;

        log::debug!(
            "  Horizon {}: bandwidth={:.2}, percentiles=[{:.2}, {:.2}]",
            horizon,
            params.bandwidth,
            params.percentiles[0],
            params.percentiles[1]
        );

        let horizon_steps = parse_horizon_to_steps(horizon, timeframe_minutes)?;
        let mut horizon_targets = vec![-1; sequence_indices.len()];
        let mut horizon_strengths = vec![0.5; sequence_indices.len()];

        for (seq_position, &seq_idx) in sequence_indices.iter().enumerate() {
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            if target_end_idx <= ohlcv_data.len() && sequence_end_idx <= ohlcv_data.len() {
                let sequence_ohlcv = &ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_ohlcv = &ohlcv_data[sequence_end_idx..target_end_idx];

                let (target_class, strength) = classify_stop_level_with_calibrated_params(
                    sequence_ohlcv,
                    horizon_ohlcv,
                    params,
                )?;
                horizon_targets[seq_position] = target_class;
                horizon_strengths[seq_position] = strength;
            }
        }

        let valid_targets: Vec<i32> = horizon_targets
            .iter()
            .filter(|&&x| x != -1)
            .cloned()
            .collect();
        if !valid_targets.is_empty() {
            analyze_class_distribution(&valid_targets, horizon, 5)?;
        }

        targets.insert(horizon.clone(), horizon_targets);
        strengths.insert(horizon.clone(), horizon_strengths);
    }

    Ok((targets, strengths))
}

/// Calculate sequence extreme range from LOW/HIGH using calibrated percentiles
/// This ensures we're comparing extremes against extreme-based boundaries
fn calculate_sequence_extreme_range(
    sequence_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::StopLevelParams,
) -> Result<(f64, f64)> {
    if sequence_ohlcv.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Empty sequence for extreme range calculation".to_string(),
        ));
    }

    // Collect all lows and highs with exponential weighting
    let len = sequence_ohlcv.len() as f64;
    let mut weighted_lows = Vec::new();
    let mut weighted_highs = Vec::new();

    for (i, candle) in sequence_ohlcv.iter().enumerate() {
        let weight = ((i as f64 + 1.0) / len).powf(2.0);
        // Weight lows: recent lows more significant (less penalty)
        let weighted_low = candle.low / (1.0 + (1.0 - weight) * 0.05);
        // Weight highs: recent highs more significant (bonus)
        let weighted_high = candle.high * (1.0 + (1.0 - weight) * 0.05);
        weighted_lows.push(weighted_low);
        weighted_highs.push(weighted_high);
    }

    // Sort to find percentiles
    weighted_lows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    weighted_highs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = weighted_lows.len();
    let lower_idx = ((n as f64 * calibrated_params.percentiles[0]) as usize).min(n - 1);
    let upper_idx = ((n as f64 * calibrated_params.percentiles[1]) as usize).min(n - 1);

    let sequence_min = weighted_lows[lower_idx];
    let sequence_max = weighted_highs[upper_idx];

    Ok((sequence_min, sequence_max))
}

/// Classify stop level with calibrated parameters using MAE (Maximum Adverse Excursion)
pub fn classify_stop_level_with_calibrated_params(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::StopLevelParams,
) -> Result<(i32, f64)> {
    if sequence_ohlcv.is_empty() || horizon_ohlcv.is_empty() {
        return Ok((2, 0.5)); // Default to neutral class with neutral strength
    }

    // 1. Calculate sequence reference price (exponentially-weighted close)
    let reference_price =
        crate::targets::price_levels::get_sequence_exponential_weighted_close(sequence_ohlcv)?;

    // 2. Calculate Maximum Adverse Excursion (MAE) - worst drawdown from reference
    let mae = calculate_maximum_adverse_excursion(horizon_ohlcv, reference_price)?;

    // 3. Calculate sequence volatility range for context
    let (sequence_min, sequence_max) =
        calculate_sequence_extreme_range(sequence_ohlcv, calibrated_params)?;
    let sequence_range = sequence_max - sequence_min;

    // Handle edge case: flat sequence
    if sequence_range == 0.0 {
        return Ok((2, 0.3)); // Neutral with low strength
    }

    // 4. Calculate MAE as percentage of sequence range (normalized risk metric)
    let mae_ratio = mae / sequence_range;

    // 5. Classify based on MAE severity using calibrated thresholds
    // bandwidth parameter controls the threshold scaling
    let threshold_multiplier = calibrated_params.bandwidth;

    // Thresholds based on MAE ratio (how many times sequence range)
    let class = if mae_ratio < 0.25 * threshold_multiplier {
        4 // Minimal Risk: MAE < 25% of sequence range
    } else if mae_ratio < 0.5 * threshold_multiplier {
        3 // Low Risk: MAE 25-50% of sequence range
    } else if mae_ratio < 1.0 * threshold_multiplier {
        2 // Moderate Risk: MAE 50-100% of sequence range
    } else if mae_ratio < 2.0 * threshold_multiplier {
        1 // High Risk: MAE 100-200% of sequence range
    } else {
        0 // Extreme Risk: MAE > 200% of sequence range
    };

    // 6. Calculate classification strength based on distance from boundaries
    let strength = calculate_mae_strength(mae_ratio, threshold_multiplier, class);

    log::debug!(
        "🛑 Stop Level MAE: ref={:.6}, mae={:.6} ({:.1}% of range), ratio={:.3} → class={} strength={:.3}",
        reference_price,
        mae,
        (mae / sequence_range) * 100.0,
        mae_ratio,
        class,
        strength
    );

    Ok((class, strength))
}

/// Calculate Maximum Adverse Excursion (MAE) from reference price
/// MAE = maximum drawdown during horizon period (exponentially weighted)
fn calculate_maximum_adverse_excursion(
    horizon_ohlcv: &[MarketDataRow],
    reference_price: f64,
) -> Result<f64> {
    if horizon_ohlcv.is_empty() {
        return Ok(0.0);
    }

    let len = horizon_ohlcv.len() as f64;
    let mut max_drawdown = 0.0;

    for (i, candle) in horizon_ohlcv.iter().enumerate() {
        // Exponential weighting: recent drawdowns more significant
        let weight = ((i as f64 + 1.0) / len).powf(2.0);

        // Calculate drawdown from reference (always positive for adverse movement)
        let drawdown = if reference_price > candle.low {
            reference_price - candle.low
        } else {
            0.0_f64
        };

        // Apply exponential weighting (50-100% weight based on recency)
        let weighted_drawdown = drawdown * (0.5 + 0.5 * weight);

        if weighted_drawdown > max_drawdown {
            max_drawdown = weighted_drawdown;
        }
    }

    Ok(max_drawdown)
}

/// Calculate classification strength for MAE-based classification
fn calculate_mae_strength(mae_ratio: f64, threshold_multiplier: f64, class: i32) -> f64 {
    let thresholds = [
        0.25 * threshold_multiplier, // Class 4|3 boundary
        0.5 * threshold_multiplier,  // Class 3|2 boundary
        1.0 * threshold_multiplier,  // Class 2|1 boundary
        2.0 * threshold_multiplier,  // Class 1|0 boundary
    ];

    match class {
        4 => {
            // Minimal Risk: closer to 0 = stronger
            let distance_from_zero = mae_ratio;
            let max_distance = thresholds[0];
            if max_distance > 0.0 {
                (1.0 - (distance_from_zero / max_distance).min(1.0)).max(0.1)
            } else {
                0.5
            }
        }
        3 => {
            // Low Risk: closer to center of range = stronger
            let range_center = (thresholds[0] + thresholds[1]) / 2.0;
            let range_half_width = (thresholds[1] - thresholds[0]) / 2.0;
            if range_half_width > 0.0 {
                let distance_from_center = (mae_ratio - range_center).abs();
                (1.0 - (distance_from_center / range_half_width).min(1.0)).max(0.1)
            } else {
                0.5
            }
        }
        2 => {
            // Moderate Risk: closer to center = stronger
            let range_center = (thresholds[1] + thresholds[2]) / 2.0;
            let range_half_width = (thresholds[2] - thresholds[1]) / 2.0;
            if range_half_width > 0.0 {
                let distance_from_center = (mae_ratio - range_center).abs();
                (1.0 - (distance_from_center / range_half_width).min(1.0)).max(0.1)
            } else {
                0.5
            }
        }
        1 => {
            // High Risk: closer to center = stronger
            let range_center = (thresholds[2] + thresholds[3]) / 2.0;
            let range_half_width = (thresholds[3] - thresholds[2]) / 2.0;
            if range_half_width > 0.0 {
                let distance_from_center = (mae_ratio - range_center).abs();
                (1.0 - (distance_from_center / range_half_width).min(1.0)).max(0.1)
            } else {
                0.5
            }
        }
        0 => {
            // Extreme Risk: further from threshold = stronger
            let distance_above = (mae_ratio - thresholds[3]).max(0.0);
            let max_distance = thresholds[3];
            if max_distance > 0.0 {
                (distance_above / max_distance).clamp(0.1, 1.0)
            } else {
                0.5
            }
        }
        _ => 0.5,
    }
}

/// Analyze class distribution and log insights
fn analyze_class_distribution(targets: &[i32], horizon: &str, bins: u32) -> Result<()> {
    let mut class_counts = vec![0usize; bins as usize];
    let mut valid_targets = 0;

    for &target in targets {
        if target >= 0 && target < bins as i32 {
            class_counts[target as usize] += 1;
            valid_targets += 1;
        }
    }

    if valid_targets == 0 {
        log::warn!(
            "📊 Stop Level Analysis [{}]: No valid targets found",
            horizon
        );
        return Ok(());
    }

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

    let imbalance_ratio = if min_class_size != usize::MAX && min_class_size > 0 {
        max_class_size as f64 / min_class_size as f64
    } else {
        f64::INFINITY
    };

    log::info!(
        "📊 Stop Level Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}] (BEFORE balanced selection)",
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

    if imbalance_ratio > 100.0 {
        log::warn!(
            "⚠️  Severe class imbalance detected for {} ({:.0}x ratio)",
            horizon,
            imbalance_ratio
        );
    }

    let empty_classes: Vec<usize> = class_counts
        .iter()
        .enumerate()
        .filter(|(_, &count)| count == 0)
        .map(|(idx, _)| idx)
        .collect();

    if !empty_classes.is_empty() {
        log::warn!(
            "⚠️  Empty classes detected for {}: {:?}",
            horizon,
            empty_classes
        );
    }

    Ok(())
}

/// Reconstruction result for stop level predictions
#[derive(Debug, Clone)]
pub struct StopLevelReconstruction {
    /// MAE ranges for each class as ratio of sequence range
    pub mae_ratio_ranges: Vec<[f64; 2]>,
    /// MAE ranges as absolute price values
    pub mae_price_ranges: Vec<[f64; 2]>,
    /// Class probabilities from model
    pub probabilities: Vec<f64>,
    /// Most likely class index
    pub most_likely_class: usize,
    /// Confidence (calibrated from max probability)
    pub confidence: f64,
    /// Expected MAE (weighted average)
    pub expected_mae: f64,
    /// Sequence range used for normalization
    pub sequence_range: f64,
    /// Reference price
    pub reference_price: f64,
}

/// Reconstruct stop level predictions from model probabilities
pub fn reconstruct_stop_levels(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    current_price: f64,
    calibrated_params: &crate::targets::calibration::StopLevelParams,
) -> Result<StopLevelReconstruction> {
    // Validate inputs
    if probabilities.len() != 5 {
        return Err(crate::utils::error::VangaError::DataError(
            "Stop level reconstruction requires exactly 5 class probabilities".to_string(),
        ));
    }

    if sequence_ohlcv.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Sequence OHLCV data is required for stop level reconstruction".to_string(),
        ));
    }

    if current_price <= 0.0 {
        return Err(crate::utils::error::VangaError::DataError(
            "Current price must be positive for percentage calculations".to_string(),
        ));
    }

    // Calculate reference price (same as training)
    let reference_price =
        crate::targets::price_levels::get_sequence_exponential_weighted_close(sequence_ohlcv)?;

    // Calculate sequence range for normalization
    let (sequence_min, sequence_max) =
        calculate_sequence_extreme_range(sequence_ohlcv, calibrated_params)?;
    let sequence_range = sequence_max - sequence_min;

    if sequence_range == 0.0 {
        return Err(crate::utils::error::VangaError::DataError(
            "Sequence has zero range - cannot reconstruct stop levels".to_string(),
        ));
    }

    // Define MAE ratio ranges for each class (based on classification thresholds)
    let threshold_multiplier = calibrated_params.bandwidth;
    let mae_ratio_ranges = vec![
        // Class 0: Extreme Risk (MAE > 200% of sequence range)
        [2.0 * threshold_multiplier, f64::INFINITY],
        // Class 1: High Risk (MAE 100-200% of sequence range)
        [1.0 * threshold_multiplier, 2.0 * threshold_multiplier],
        // Class 2: Moderate Risk (MAE 50-100% of sequence range)
        [0.5 * threshold_multiplier, 1.0 * threshold_multiplier],
        // Class 3: Low Risk (MAE 25-50% of sequence range)
        [0.25 * threshold_multiplier, 0.5 * threshold_multiplier],
        // Class 4: Minimal Risk (MAE < 25% of sequence range)
        [0.0, 0.25 * threshold_multiplier],
    ];

    // Convert MAE ratios to absolute price values
    let mae_price_ranges: Vec<[f64; 2]> = mae_ratio_ranges
        .iter()
        .map(|[lower_ratio, upper_ratio]| {
            let lower_mae = lower_ratio * sequence_range;
            let upper_mae = if upper_ratio.is_infinite() {
                sequence_range * 3.0 // Cap at 3x for display
            } else {
                upper_ratio * sequence_range
            };
            [lower_mae, upper_mae]
        })
        .collect();

    // Find most likely class
    let (most_likely_class, max_prob) = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &prob)| (idx, prob))
        .unwrap_or((2, 0.2));

    // Calculate confidence using unified calibration
    let confidence = crate::output::confidence_calculator::calibrate_5_class_confidence(max_prob);

    // Calculate expected MAE (weighted average of class midpoints)
    let class_midpoints: Vec<f64> = mae_ratio_ranges
        .iter()
        .map(|[lower, upper]| {
            if upper.is_infinite() {
                lower + 0.5 // For infinite upper bound, use lower + 0.5
            } else {
                (lower + upper) / 2.0
            }
        })
        .collect();

    let expected_mae_ratio: f64 = probabilities
        .iter()
        .zip(class_midpoints.iter())
        .map(|(&prob, &midpoint)| prob * midpoint)
        .sum();

    let expected_mae = expected_mae_ratio * sequence_range;

    Ok(StopLevelReconstruction {
        mae_ratio_ranges,
        mae_price_ranges,
        probabilities: probabilities.to_vec(),
        most_likely_class,
        confidence,
        expected_mae,
        sequence_range,
        reference_price,
    })
}
