//! Stop level target generation for cryptocurrency forecasting
//!
//! # 🎯 TARGET PURPOSE: "WHAT'S THE ADVERSE PRICE MOVEMENT RISK?"
//!
//! This module implements **Adverse Price Level Analysis** for risk management.
//! It answers: "How far will price move AGAINST the expected direction?"
//!
//! ## 📊 MATHEMATICAL FOUNDATION
//!
//! ### **Core Logic: Direction-Aware Adverse Analysis**
//! ```text
//! 1. Determine expected direction (target vs reference price)
//! 2. Collect ADVERSE prices from sequence (lows for bullish, highs for bearish)
//! 3. Build boundaries from SAME adverse price distribution
//! 4. Find worst adverse price during horizon (weighted by recency)
//! 5. Classify adverse severity against sequence-derived boundaries
//! ```
//!
//! ### **KEY INSIGHT: Distribution Matching**
//!
//! Unlike price_levels (which classifies closes against closes), stop_levels must:
//! - For bullish: Build boundaries from sequence LOWS, classify horizon LOWS
//! - For bearish: Build boundaries from sequence HIGHS, classify horizon HIGHS
//!
//! This ensures the boundary distribution matches what we're classifying.
//!
//! ### **5-Class Risk Classification System:**
//! - **0: Extreme Risk** - Adverse beyond sequence adverse range (deep dip/high bounce)
//! - **1: High Risk** - Adverse at edge of sequence adverse range
//! - **2: Moderate Risk** - Adverse in neutral zone (center of range)
//! - **3: Low Risk** - Adverse within normal sequence adverse range
//! - **4: Minimal Risk** - Adverse very contained (shallow dip/small bounce)
//!
//! ## 🔧 KEY FEATURES
//!
//! ### **Direction-Aware Analysis**
//! - Bullish: Measures downside risk using LOWS
//! - Bearish: Measures upside risk using HIGHS
//! - Boundaries built from same price type as classification target
//!
//! ### **Calibrated Boundaries**
//! - Uses Bayesian-optimized percentiles on adverse prices
//! - Bandwidth parameter controls extreme zone size
//! - Neutral band factor controls center zone size
//!
//! ### **Symbol-Agnostic Design**
//! - Normalized by adverse price range from sequence
//! - Works with any price range (BTC, ETH, altcoins)
//! - Percentage-based classification
//!
//! ## 🎯 COMPLEMENTARY ROLE
//!
//! **Stop Levels** work with **Price Levels**:
//! - **Price Levels**: "Where will price END UP?" (destination via closes)
//! - **Stop Levels**: "What's the WORST ADVERSE MOVEMENT?" (risk via wicks)
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

/// Adverse price boundaries calculated from sequence data
/// Similar to SequenceBoundaries but for adverse (dip/bounce) analysis
#[derive(Debug, Clone)]
pub struct AdverseBoundaries {
    /// Lower percentile boundary of adverse prices
    pub adverse_min: f64,
    /// Upper percentile boundary of adverse prices  
    pub adverse_max: f64,
    /// Bandwidth for extreme classification
    pub bandwidth: f64,
    /// Reference price (exponentially-weighted close)
    pub reference_price: f64,
    /// Classification boundaries [b0, b1, b2, b3]
    ///
    /// For bullish (measuring dips - lower is worse):
    /// - Class 0 (Extreme): adverse < b0 (very deep dip)
    /// - Class 1 (High): b0 <= adverse < b1
    /// - Class 2 (Moderate): b1 <= adverse < b2
    /// - Class 3 (Low): b2 <= adverse < b3
    /// - Class 4 (Minimal): adverse >= b3 (shallow dip)
    ///
    /// For bearish (measuring bounces - higher is worse):
    /// - Class 0 (Extreme): adverse > b3 (very high bounce)
    /// - Class 1 (High): b2 < adverse <= b3
    /// - Class 2 (Moderate): b1 < adverse <= b2
    /// - Class 3 (Low): b0 < adverse <= b1
    /// - Class 4 (Minimal): adverse <= b0 (small bounce)
    pub boundaries: [f64; 4],
    /// Whether this is for bullish (true) or bearish (false) direction
    pub is_bullish: bool,
}

impl AdverseBoundaries {
    /// Classify an adverse price into one of 5 risk classes
    pub fn classify_adverse(&self, adverse_price: f64) -> i32 {
        if self.is_bullish {
            // Bullish: lower adverse = worse (deeper dip)
            // boundaries are ordered: b0 < b1 < b2 < b3
            if adverse_price < self.boundaries[0] {
                0 // Extreme Risk: very deep dip
            } else if adverse_price < self.boundaries[1] {
                1 // High Risk
            } else if adverse_price < self.boundaries[2] {
                2 // Moderate Risk
            } else if adverse_price < self.boundaries[3] {
                3 // Low Risk
            } else {
                4 // Minimal Risk: shallow dip
            }
        } else {
            // Bearish: higher adverse = worse (higher bounce)
            // boundaries are ordered: b0 < b1 < b2 < b3
            if adverse_price > self.boundaries[3] {
                0 // Extreme Risk: very high bounce
            } else if adverse_price > self.boundaries[2] {
                1 // High Risk
            } else if adverse_price > self.boundaries[1] {
                2 // Moderate Risk
            } else if adverse_price > self.boundaries[0] {
                3 // Low Risk
            } else {
                4 // Minimal Risk: small bounce
            }
        }
    }

    /// Calculate strength based on distance from boundary
    pub fn calculate_strength(&self, adverse_price: f64, class: i32) -> f64 {
        let (lower, upper) = match class {
            0 => {
                if self.is_bullish {
                    (self.adverse_min - self.bandwidth, self.boundaries[0])
                } else {
                    (self.boundaries[3], self.adverse_max + self.bandwidth)
                }
            }
            1 => {
                if self.is_bullish {
                    (self.boundaries[0], self.boundaries[1])
                } else {
                    (self.boundaries[2], self.boundaries[3])
                }
            }
            2 => (self.boundaries[1], self.boundaries[2]),
            3 => {
                if self.is_bullish {
                    (self.boundaries[2], self.boundaries[3])
                } else {
                    (self.boundaries[0], self.boundaries[1])
                }
            }
            4 => {
                if self.is_bullish {
                    (self.boundaries[3], self.adverse_max + self.bandwidth)
                } else {
                    (self.adverse_min - self.bandwidth, self.boundaries[0])
                }
            }
            _ => return 0.5,
        };

        let range = (upper - lower).abs();
        if range == 0.0 {
            return 0.5;
        }

        let center = (lower + upper) / 2.0;
        let distance = (adverse_price - center).abs();
        let half_range = range / 2.0;

        (1.0 - (distance / half_range).min(1.0)).max(0.1)
    }
}

/// Calculate adverse boundaries from sequence data
///
/// **KEY FIX**: Builds boundaries from the SAME type of prices we classify:
/// - For bullish: boundaries from sequence LOWS (since we classify horizon lows)
/// - For bearish: boundaries from sequence HIGHS (since we classify horizon highs)
///
/// This ensures the boundary distribution matches the classification target distribution,
/// enabling proper calibration for balanced 5-class output.
pub fn calculate_adverse_boundaries(
    sequence_ohlcv: &[MarketDataRow],
    is_bullish: bool,
    calibrated_params: &crate::targets::calibration::StopLevelParams,
) -> Result<AdverseBoundaries> {
    if sequence_ohlcv.len() < 2 {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient sequence data for adverse boundary calculation".to_string(),
        ));
    }

    // Get reference price (exponentially-weighted close)
    let reference_price =
        crate::targets::price_levels::get_sequence_exponential_weighted_close(sequence_ohlcv)?;

    // **CRITICAL**: Collect ADVERSE prices (lows for bullish, highs for bearish)
    // This matches the distribution of what we're classifying in the horizon
    let mut adverse_prices: Vec<f64> = if is_bullish {
        sequence_ohlcv.iter().map(|c| c.low).collect()
    } else {
        sequence_ohlcv.iter().map(|c| c.high).collect()
    };
    adverse_prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = adverse_prices.len();
    let lower_idx = ((n as f64 * calibrated_params.percentiles[0]) as usize).min(n - 1);
    let upper_idx = ((n as f64 * calibrated_params.percentiles[1]) as usize).min(n - 1);

    let sequence_min = adverse_prices[lower_idx];
    let sequence_max = adverse_prices[upper_idx];

    // Calculate effective range from adverse prices.
    //
    // `bandwidth` is computed below as `effective_range * calibrated_params.bandwidth`,
    // so each branch here returns a pure "range" value (no extra bandwidth multiplication).
    // The last-resort path previously multiplied by `calibrated_params.bandwidth` here AND
    // again below, squaring it. The fixed version uses a small absolute fraction of the
    // price level as a fallback range, so the final bandwidth is a single multiplication.
    let base_range = (sequence_max - sequence_min).abs();
    let effective_range = if base_range > 0.0 {
        base_range
    } else {
        // Flat sequence - use spread between adverse and reference as proxy
        let price_spread = (reference_price - sequence_min).abs();
        if price_spread > 0.0 {
            price_spread
        } else {
            // Last resort: fully flat sequence (reference == adverse == constant).
            // Use a small fraction of the price level so downstream classification
            // still has a non-zero bandwidth without compounding `calibrated_params.bandwidth`.
            sequence_min.abs() * 0.01
        }
    };

    let bandwidth = effective_range * calibrated_params.bandwidth;
    let range_center = (sequence_min + sequence_max) / 2.0;
    let neutral_half = effective_range * calibrated_params.neutral_band_factor / 2.0;

    // Build boundaries for 5-class classification
    // Boundaries are based on ADVERSE prices (lows for bullish, highs for bearish)
    // For bullish: we track LOWS (dips) - lower adverse = worse risk
    //   Class 0: adverse < b0 (extreme dip below sequence low range)
    //   Class 4: adverse >= b3 (minimal dip, stays within sequence low range)
    // For bearish: we track HIGHS (bounces) - higher adverse = worse risk
    //   Class 0: adverse > b3 (extreme bounce above sequence high range)
    //   Class 4: adverse <= b0 (minimal bounce, stays within sequence high range)
    let boundaries = [
        sequence_min - bandwidth,
        range_center - neutral_half,
        range_center + neutral_half,
        sequence_max + bandwidth,
    ];

    Ok(AdverseBoundaries {
        adverse_min: sequence_min,
        adverse_max: sequence_max,
        bandwidth,
        reference_price,
        boundaries,
        is_bullish,
    })
}

/// Classify stop level with calibrated parameters using boundary-based approach
///
/// Uses the same boundary-based classification as price_levels but for adverse prices.
/// Boundaries are built from sequence adverse prices, then horizon worst adverse is classified.
pub fn classify_stop_level_with_calibrated_params(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::StopLevelParams,
) -> Result<(i32, f64)> {
    if sequence_ohlcv.is_empty() || horizon_ohlcv.is_empty() {
        return Ok((2, 0.5));
    }

    // 1. Determine direction from horizon
    let reference_price =
        crate::targets::price_levels::get_sequence_exponential_weighted_close(sequence_ohlcv)?;
    let target_price =
        crate::targets::price_levels::get_horizon_exponential_weighted_close(horizon_ohlcv)?;
    let is_bullish = target_price >= reference_price;

    // 2. Calculate adverse boundaries from sequence
    let boundaries = calculate_adverse_boundaries(sequence_ohlcv, is_bullish, calibrated_params)?;

    // 3. Find worst adverse price in horizon (with exponential weighting for recency)
    let worst_adverse = find_weighted_worst_adverse(horizon_ohlcv, is_bullish, reference_price);

    // 4. Classify using boundaries
    let class = boundaries.classify_adverse(worst_adverse);
    let strength = boundaries.calculate_strength(worst_adverse, class);

    log::debug!(
        "🛑 Stop Level: dir={}, ref={:.6}, worst={:.6}, bounds=[{:.6}, {:.6}, {:.6}, {:.6}] → class={} strength={:.3}",
        if is_bullish { "BULL" } else { "BEAR" },
        reference_price,
        worst_adverse,
        boundaries.boundaries[0],
        boundaries.boundaries[1],
        boundaries.boundaries[2],
        boundaries.boundaries[3],
        class,
        strength
    );

    Ok((class, strength))
}

/// Find the worst adverse price in horizon with exponential weighting
///
/// Recent adverse prices are weighted more heavily because:
/// - Recent drawdowns are more relevant for stop-loss placement
/// - Early drawdowns that recover are less concerning
fn find_weighted_worst_adverse(
    horizon_ohlcv: &[MarketDataRow],
    is_bullish: bool,
    reference_price: f64,
) -> f64 {
    if horizon_ohlcv.is_empty() {
        return reference_price;
    }

    let len = horizon_ohlcv.len() as f64;
    let mut worst_adverse = reference_price;
    let mut max_weighted_severity = 0.0;

    for (i, candle) in horizon_ohlcv.iter().enumerate() {
        // Exponential weight: recent candles weighted more (0.5 to 1.0 range)
        let position_ratio = (i as f64 + 1.0) / len;
        let weight = 0.5 + 0.5 * position_ratio.powi(2);

        let adverse_price = if is_bullish { candle.low } else { candle.high };

        // Calculate severity (how far from reference in adverse direction)
        let raw_severity = if is_bullish {
            (reference_price - adverse_price).max(0.0)
        } else {
            (adverse_price - reference_price).max(0.0)
        };

        let weighted_severity = raw_severity * weight;

        if weighted_severity > max_weighted_severity {
            max_weighted_severity = weighted_severity;
            worst_adverse = adverse_price;
        }
    }

    worst_adverse
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
    /// Adverse price boundaries for each class
    pub adverse_price_ranges: Vec<[f64; 2]>,
    /// Class probabilities from model
    pub probabilities: Vec<f64>,
    /// Most likely class index
    pub most_likely_class: usize,
    /// Confidence (calibrated from max probability)
    pub confidence: f64,
    /// Expected adverse price (weighted average)
    pub expected_adverse: f64,
    /// Adverse boundaries used for classification
    pub boundaries: AdverseBoundaries,
    /// Reference price
    pub reference_price: f64,
    /// Direction used for reconstruction (true = bullish/dip-risk, false = bearish/bounce-risk)
    pub is_bullish: bool,
}

/// Reconstruct stop level predictions from model probabilities.
///
/// Stop level training is direction-aware: for bullish samples it builds boundaries
/// from sequence LOWS (deepest dip = class 0), for bearish samples it builds boundaries
/// from sequence HIGHS (highest bounce = class 0). At prediction time, we don't know
/// the realized direction, so we derive it from the model's direction prediction.
///
/// # Arguments
/// * `direction_probabilities` - Optional [DUMP, DOWN, SIDEWAYS, UP, PUMP] probabilities
///   from the direction head. If `Some`, `is_bullish` is set to `(UP + PUMP) >= (DUMP + DOWN)`,
///   matching the training rule `target_price >= reference_price`. SIDEWAYS does not
///   tip either way. If `None`, falls back to bullish (legacy behavior).
pub fn reconstruct_stop_levels(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    current_price: f64,
    calibrated_params: &crate::targets::calibration::StopLevelParams,
    direction_probabilities: Option<&[f64]>,
) -> Result<StopLevelReconstruction> {
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

    // Determine direction from the model's direction head to mirror the training-time
    // rule (`target_price >= reference_price`). Bullish samples were labeled against
    // sequence LOWS; bearish against sequence HIGHS. Without this, the reconstructed
    // price ranges are systematically wrong for bearish predictions.
    let is_bullish = match direction_probabilities {
        Some(probs) if probs.len() == 5 => {
            let bull_score = probs[3] + probs[4]; // UP + PUMP
            let bear_score = probs[0] + probs[1]; // DUMP + DOWN
            // Tie (including the all-SIDEWAYS case) defaults to bullish so behavior
            // matches training's `>=` rule.
            bull_score >= bear_score
        }
        Some(_) => {
            log::warn!(
                "Stop level reconstruction received direction_probabilities with unexpected length ({}); defaulting to bullish",
                direction_probabilities.map(|p| p.len()).unwrap_or(0)
            );
            true
        }
        None => {
            log::debug!("Stop level reconstruction: no direction prediction available, defaulting to bullish");
            true
        }
    };

    // Calculate adverse boundaries using same logic as classification
    let boundaries = calculate_adverse_boundaries(sequence_ohlcv, is_bullish, calibrated_params)?;

    // Build price ranges from boundaries.
    // `boundaries.boundaries` is always sorted b0 < b1 < b2 < b3 where
    //   b0 = adverse_min - bandwidth,  b3 = adverse_max + bandwidth.
    //
    // The two open-ended tails (class 0 and class 4) each need a finite extension
    // for display/expectation calculations. We extend each tail by `bandwidth`
    // beyond the boundary it sits against, giving every class a non-degenerate
    // range of width ~bandwidth on the ends.
    //
    // For BULLISH we track dips (lows), so lower adverse = worse: Class 0 sits below b0.
    // For BEARISH we track bounces (highs), so higher adverse = worse: Class 0 sits above b3.
    let bw = boundaries.bandwidth;
    let adverse_price_ranges = if is_bullish {
        vec![
            [boundaries.boundaries[0] - bw, boundaries.boundaries[0]], // Class 0: Extreme dip (deepest)
            [boundaries.boundaries[0], boundaries.boundaries[1]],      // Class 1: High dip
            [boundaries.boundaries[1], boundaries.boundaries[2]],      // Class 2: Moderate dip
            [boundaries.boundaries[2], boundaries.boundaries[3]],      // Class 3: Low dip
            [boundaries.boundaries[3], boundaries.boundaries[3] + bw], // Class 4: Minimal dip (shallowest)
        ]
    } else {
        vec![
            [boundaries.boundaries[3], boundaries.boundaries[3] + bw], // Class 0: Extreme bounce (highest)
            [boundaries.boundaries[2], boundaries.boundaries[3]],      // Class 1: High bounce
            [boundaries.boundaries[1], boundaries.boundaries[2]],      // Class 2: Moderate bounce
            [boundaries.boundaries[0], boundaries.boundaries[1]],      // Class 3: Low bounce
            [boundaries.boundaries[0] - bw, boundaries.boundaries[0]], // Class 4: Minimal bounce (smallest)
        ]
    };

    let (most_likely_class, max_prob) = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &prob)| (idx, prob))
        .unwrap_or((2, 0.2));

    let confidence = crate::output::confidence_calculator::calibrate_5_class_confidence(max_prob);

    // Calculate expected adverse as weighted average of class midpoints
    let class_midpoints: Vec<f64> = adverse_price_ranges
        .iter()
        .map(|[lower, upper]| (lower + upper) / 2.0)
        .collect();

    let expected_adverse: f64 = probabilities
        .iter()
        .zip(class_midpoints.iter())
        .map(|(&prob, &midpoint)| prob * midpoint)
        .sum();

    let reference_price = boundaries.reference_price;

    Ok(StopLevelReconstruction {
        adverse_price_ranges,
        probabilities: probabilities.to_vec(),
        most_likely_class,
        confidence,
        expected_adverse,
        boundaries,
        reference_price,
        is_bullish,
    })
}
