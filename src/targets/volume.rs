//! Volume target generation for cryptocurrency volume regime classification
//!
//! # 🎯 TARGET PURPOSE: "WHAT IS THE VOLUME REGIME?"
//!
//! This module implements **logarithmic volume ratio analysis** for volume regime detection.
//! It answers: "Will the future volume be significantly higher, lower, or similar to recent volume?"
//!
//! ## 📊 MATHEMATICAL FOUNDATION
//!
//! ### **Core Logic: Logarithmic Volume Ratio Classification**
//! ```
//! 1. Calculate sequence median volume (baseline)
//! 2. Calculate horizon median volume (target)
//! 3. Compute volume ratio: horizon_volume / sequence_volume
//! 4. Apply logarithmic transformation: ln(volume_ratio)
//! 5. Use percentile-based ADAPTIVE THRESHOLDS in log space
//! 6. Classify using symmetric boundaries around ln(1.0) = 0
//! ```
//!
//! ### **5-Class Volume Classification:**
//! - **0: Very Low** - Major volume decrease (log_ratio < -extreme_threshold)
//! - **1: Low** - Moderate volume decrease (-extreme < log_ratio < -base)
//! - **2: Medium** - Similar volume (-base < log_ratio < +base)
//! - **3: High** - Moderate volume increase (+base < log_ratio < +extreme)
//! - **4: Very High** - Major volume surge (log_ratio > +extreme_threshold)
//!
//! ## 🔧 KEY FEATURES
//!
//! ### **Logarithmic Symmetry**
//! Volume ratios are naturally multiplicative and asymmetric. A 2x increase (ratio=2.0)
//! should be treated equally to a 0.5x decrease (ratio=0.5), but in linear space:
//! - 2.0 - 1.0 = +1.0 (increase)
//! - 0.5 - 1.0 = -0.5 (decrease) ← Asymmetric!
//!
//! In logarithmic space, ratios become symmetric:
//! - ln(2.0) = +0.693 (increase)
//! - ln(0.5) = -0.693 (decrease) ← Perfectly symmetric!
//!
//! ### **Adaptive Thresholds via Percentiles**
//! - Percentiles (p5-p95) determine typical volume variation WITHIN each sequence
//! - typical_log_variation = ln(p95_volume / p5_volume)
//! - base_threshold = typical_log_variation × bandwidth
//! - extreme_threshold = base_threshold × extreme_multiplier
//! - This makes classification adaptive to each sequence's volatility characteristics
//!
//! ### **Calibrated Parameters**
//! - All parameters automatically calibrated for balanced 20% per class distribution
//! - bandwidth: Controls sensitivity (0.1-3.0)
//! - extreme_multiplier: Controls extreme class boundaries (1.2-6.0)
//! - percentile_low/high: Adaptive range calculation (0.01-0.30 / 0.70-0.99)
//! - smoothing_periods: Noise reduction (1-30)

use crate::data::structures::MarketDataRow;
use crate::targets::TargetResult;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Volume analysis thresholds in logarithmic space
#[derive(Debug, Clone)]
pub struct LogVolumeThresholds {
    /// Maximum log ratio for Very Low class
    pub very_low_max: f64,
    /// Maximum log ratio for Low class
    pub low_max: f64,
    /// Maximum log ratio for Medium class
    pub medium_max: f64,
    /// Maximum log ratio for High class
    pub high_max: f64,
}

/// Volume configuration for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeConfig {
    /// Controls the width of adaptive thresholds for volume regime classification.
    /// A larger value increases the separation between classes, making classification stricter.
    /// Typical range: 0.2–0.6. Default is 0.4. Adjust based on volume volatility.
    pub bandwidth_size: f64,
    pub extreme_multiplier: f64,
    pub smoothing_periods: usize,
}

impl Default for VolumeConfig {
    fn default() -> Self {
        Self {
            bandwidth_size: 0.4,
            extreme_multiplier: 2.0,
            smoothing_periods: 3,
        }
    }
}

/// Volume distribution statistics
#[derive(Debug, Clone)]
pub struct VolumeDistributionStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

/// Generate volume targets with optional adaptive parameters
///
/// When adaptive_params is provided, uses the pre-calibrated parameters for consistent
/// Generate volume targets with calibrated parameters - returns both class and strength
/// target generation between training and prediction. When None, uses base config.
pub fn generate_volume_targets_with_calibrated_params(
    df: &DataFrame,
    horizons: &[String],
    sequence_indices: &[usize],
    sequence_length: usize,
    calibrated_params: &std::collections::HashMap<
        String,
        crate::targets::calibration::VolumeParams,
    >,
) -> Result<TargetResult> {
    let volume_data = extract_volume_data(df)?;

    log::info!("🎯 Generating volume targets with per-horizon calibrated parameters");

    let mut targets = HashMap::new();
    let mut strengths = HashMap::new();

    for horizon in horizons {
        // Get parameters for this specific horizon
        let params = calibrated_params.get(horizon).ok_or_else(|| {
            crate::utils::error::VangaError::ConfigError(format!(
                "No calibrated volume parameters found for horizon: {}",
                horizon
            ))
        })?;

        log::debug!(
            "  Horizon {}: bandwidth={:.4}, extreme_mult={:.2}, smoothing={}",
            horizon,
            params.bandwidth,
            params.extreme_multiplier,
            params.smoothing_periods
        );

        let config = VolumeConfig {
            bandwidth_size: params.bandwidth,
            extreme_multiplier: params.extreme_multiplier,
            smoothing_periods: params.smoothing_periods,
        };

        // Calculate logarithmic volume thresholds for this horizon
        let thresholds = calculate_log_volume_thresholds(&config)?;

        let horizon_steps = parse_horizon_steps(horizon)?;
        let mut horizon_targets = Vec::new();
        let mut horizon_strengths = Vec::new();

        for &seq_start in sequence_indices {
            let seq_end = seq_start + sequence_length;
            let horizon_end = seq_end + horizon_steps;

            if horizon_end > volume_data.len() {
                continue;
            }

            let sequence_volumes = &volume_data[seq_start..seq_end];
            let horizon_volumes = &volume_data[seq_end..horizon_end];

            // Use the strength-returning version of classify_volume_regime
            match classify_volume_regime_with_strength(
                sequence_volumes,
                horizon_volumes,
                &thresholds,
                &config,
                params.percentile_low,
                params.percentile_high,
            ) {
                Ok((class, strength)) => {
                    horizon_targets.push(class);
                    horizon_strengths.push(strength);
                }
                Err(e) => {
                    log::warn!(
                        "Failed to classify volume for sequence {}: {}",
                        seq_start,
                        e
                    );
                    continue;
                }
            }
        }

        if !horizon_targets.is_empty() {
            log_volume_distribution(&horizon_targets, horizon);
            targets.insert(horizon.clone(), horizon_targets);
            strengths.insert(horizon.clone(), horizon_strengths);
        }
    }

    Ok((targets, strengths))
}

/// Classify volume regime using calibrated parameters (consistent API)
pub fn classify_volume_with_calibrated_params(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    calibrated_params: &crate::targets::calibration::VolumeParams,
) -> Result<(i32, f64)> {
    // Extract volume data from OHLCV
    let sequence_volumes: Vec<f64> = sequence_ohlcv.iter().map(|row| row.volume).collect();
    let horizon_volumes: Vec<f64> = horizon_ohlcv.iter().map(|row| row.volume).collect();

    // Create thresholds from calibrated params with minimum thresholds applied
    let base_threshold = calibrated_params
        .bandwidth
        .max(calibrated_params.min_base_threshold);
    let extreme_threshold = (calibrated_params.extreme_multiplier * calibrated_params.bandwidth)
        .max(calibrated_params.min_extreme_threshold);

    let thresholds = LogVolumeThresholds {
        very_low_max: (1.0_f64 - extreme_threshold).ln(),
        low_max: (1.0_f64 - base_threshold).ln(),
        medium_max: (1.0_f64 + base_threshold).ln(),
        high_max: (1.0_f64 + extreme_threshold).ln(),
    };

    // Create config from calibrated params
    let config = VolumeConfig {
        bandwidth_size: calibrated_params.bandwidth,
        extreme_multiplier: calibrated_params.extreme_multiplier,
        smoothing_periods: calibrated_params.smoothing_periods,
    };

    classify_volume_regime_with_strength(
        &sequence_volumes,
        &horizon_volumes,
        &thresholds,
        &config,
        calibrated_params.percentile_low,
        calibrated_params.percentile_high,
    )
}

/// Classify volume regime using LOGARITHMIC RATIO analysis with adaptive thresholds
///
/// **APPROACH**: Logarithmic ratio classification with percentile-based adaptive thresholds
/// for symmetric, balanced classification that handles multiplicative volume changes correctly.
///
/// **Logic**:
/// 1. Calculate sequence median and horizon median volumes
/// 2. Compute log_ratio = ln(horizon_median / sequence_median)
/// 3. Calculate adaptive thresholds using CALIBRATED percentile_low/percentile_high
///    - Percentiles determine typical volume variation within sequence
///    - This makes thresholds adaptive to each sequence's volatility
/// 4. Classify log_ratio into 5 symmetric classes around ln(1.0) = 0
/// 5. This creates symmetric boundaries: 2x increase = 0.5x decrease in log space
pub fn classify_volume_regime_with_strength(
    sequence_volumes: &[f64],
    horizon_volumes: &[f64],
    _thresholds: &LogVolumeThresholds,
    config: &VolumeConfig,
    percentile_low: f64,
    percentile_high: f64,
) -> Result<(i32, f64)> {
    if sequence_volumes.is_empty() || horizon_volumes.is_empty() {
        return Err(VangaError::DataError(
            "Empty volume data for analysis".to_string(),
        ));
    }

    // LOGARITHMIC RATIO APPROACH: Symmetric classification around 1.0 ratio

    // 1. Calculate sequence median volume (baseline)
    let mut sorted_seq_volumes = sequence_volumes.to_vec();
    sorted_seq_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sequence_median = if sorted_seq_volumes.len() % 2 == 0 {
        let mid = sorted_seq_volumes.len() / 2;
        (sorted_seq_volumes[mid - 1] + sorted_seq_volumes[mid]) / 2.0
    } else {
        sorted_seq_volumes[sorted_seq_volumes.len() / 2]
    };

    // 2. Calculate horizon median volume (target)
    let mut sorted_hor_volumes = horizon_volumes.to_vec();
    sorted_hor_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let horizon_median = if sorted_hor_volumes.len() % 2 == 0 {
        let mid = sorted_hor_volumes.len() / 2;
        (sorted_hor_volumes[mid - 1] + sorted_hor_volumes[mid]) / 2.0
    } else {
        sorted_hor_volumes[sorted_hor_volumes.len() / 2]
    };

    // Handle edge case: zero volume
    if sequence_median < 1e-10 {
        return Ok((2, 0.5)); // Default to medium with neutral strength
    }

    // 3. Calculate volume ratio and apply logarithmic transformation
    let volume_ratio = horizon_median / sequence_median;
    let log_ratio = volume_ratio.ln();

    // 4. Calculate adaptive thresholds using percentile-based range
    // Use percentiles to determine typical volume variation
    let plow_idx = (sorted_seq_volumes.len() as f64 * percentile_low).floor() as usize;
    let phigh_idx = ((sorted_seq_volumes.len() as f64 * percentile_high).ceil() as usize)
        .min(sorted_seq_volumes.len() - 1);

    let seq_vol_low = sorted_seq_volumes[plow_idx];
    let seq_vol_high = sorted_seq_volumes[phigh_idx];

    // Calculate typical log variation within sequence
    let typical_log_variation = if seq_vol_low > 1e-10 {
        (seq_vol_high / seq_vol_low).ln()
    } else {
        0.693 // Default to ln(2) = ~69% variation
    };

    // 5. Define symmetric thresholds in log space
    let base_threshold = typical_log_variation * config.bandwidth_size;
    let extreme_threshold = base_threshold * config.extreme_multiplier;

    // Symmetric boundaries around 0 (log(1.0) = 0)
    let boundary_0 = -extreme_threshold; // Very Low: Major decrease
    let boundary_1 = -base_threshold; // Low: Moderate decrease
    let boundary_2 = base_threshold; // Medium: Similar volume
    let boundary_3 = extreme_threshold; // High: Moderate increase
                                        // Above boundary_3 = Very High: Major surge

    // 6. Classify based on log ratio
    let class = if log_ratio < boundary_0 {
        0 // Very Low: Major volume decrease
    } else if log_ratio < boundary_1 {
        1 // Low: Moderate volume decrease
    } else if log_ratio < boundary_2 {
        2 // Medium: Similar volume
    } else if log_ratio < boundary_3 {
        3 // High: Moderate volume increase
    } else {
        4 // Very High: Major volume surge
    };

    // 7. Calculate classification strength based on position within range
    let strength = calculate_volume_strength_log_ratio(
        log_ratio, boundary_0, boundary_1, boundary_2, boundary_3, class,
    );

    log::debug!(
        "🎯 Volume Log Ratio Analysis: seq_median={:.2}, hor_median={:.2}, ratio={:.3}, log_ratio={:.3}, thresholds=[{:.3}, {:.3}, {:.3}, {:.3}] → class={} ({}) strength={:.3}",
        sequence_median, horizon_median, volume_ratio, log_ratio,
        boundary_0, boundary_1, boundary_2, boundary_3,
        class, ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"][class as usize], strength
    );

    Ok((class, strength))
}

/// Calculate volume strength for percentile-based classification
/// Calculate volume classification strength based on log ratio position
fn calculate_volume_strength_log_ratio(
    log_ratio: f64,
    boundary_0: f64,
    boundary_1: f64,
    boundary_2: f64,
    boundary_3: f64,
    class: i32,
) -> f64 {
    match class {
        0 => {
            // Very Low: log_ratio < boundary_0
            let distance_below = (boundary_0 - log_ratio).max(0.0);
            let max_distance = (boundary_1 - boundary_0).abs();
            if max_distance > 0.0 {
                (distance_below / max_distance).clamp(0.1, 1.0)
            } else {
                0.5
            }
        }
        1 => {
            // Low: boundary_0 <= log_ratio < boundary_1
            let range_center = (boundary_0 + boundary_1) / 2.0;
            let range_half_width = (boundary_1 - boundary_0) / 2.0;
            if range_half_width > 0.0 {
                let distance_from_center = (log_ratio - range_center).abs();
                let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
                strength.max(0.1)
            } else {
                0.5
            }
        }
        2 => {
            // Medium: boundary_1 <= log_ratio < boundary_2
            let range_center = (boundary_1 + boundary_2) / 2.0;
            let range_half_width = (boundary_2 - boundary_1) / 2.0;
            if range_half_width > 0.0 {
                let distance_from_center = (log_ratio - range_center).abs();
                let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
                strength.max(0.1)
            } else {
                0.5
            }
        }
        3 => {
            // High: boundary_2 <= log_ratio < boundary_3
            let range_center = (boundary_2 + boundary_3) / 2.0;
            let range_half_width = (boundary_3 - boundary_2) / 2.0;
            if range_half_width > 0.0 {
                let distance_from_center = (log_ratio - range_center).abs();
                let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
                strength.max(0.1)
            } else {
                0.5
            }
        }
        4 => {
            // Very High: log_ratio >= boundary_3
            let distance_beyond = (log_ratio - boundary_3).max(0.0);
            let max_distance = (boundary_3 - boundary_2).abs();
            if max_distance > 0.0 {
                (distance_beyond / max_distance).clamp(0.1, 1.0)
            } else {
                0.5
            }
        }
        _ => 0.5,
    }
}

/// Calculate volume classification strength based on percentile position (OLD - DEPRECATED)
#[allow(dead_code)]
fn calculate_volume_strength_percentile(
    target_volume: f64,
    boundary_0: f64,
    boundary_1: f64,
    boundary_2: f64,
    boundary_3: f64,
    class: i32,
) -> f64 {
    match class {
        0 => {
            // Very Low: target < boundary_0
            let distance_below = (boundary_0 - target_volume).max(0.0);
            let max_distance = (boundary_1 - boundary_0).abs();
            if max_distance > 0.0 {
                (distance_below / max_distance).clamp(0.1, 1.0)
            } else {
                0.5
            }
        }
        1 => {
            // Low: boundary_0 <= target < boundary_1
            let range_center = (boundary_0 + boundary_1) / 2.0;
            let range_half_width = (boundary_1 - boundary_0) / 2.0;
            if range_half_width > 0.0 {
                let distance_from_center = (target_volume - range_center).abs();
                let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
                strength.max(0.1)
            } else {
                0.5
            }
        }
        2 => {
            // Medium: boundary_1 <= target < boundary_2
            let range_center = (boundary_1 + boundary_2) / 2.0;
            let range_half_width = (boundary_2 - boundary_1) / 2.0;
            if range_half_width > 0.0 {
                let distance_from_center = (target_volume - range_center).abs();
                let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
                strength.max(0.1)
            } else {
                0.5
            }
        }
        3 => {
            // High: boundary_2 <= target < boundary_3
            let range_center = (boundary_2 + boundary_3) / 2.0;
            let range_half_width = (boundary_3 - boundary_2) / 2.0;
            if range_half_width > 0.0 {
                let distance_from_center = (target_volume - range_center).abs();
                let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
                strength.max(0.1)
            } else {
                0.5
            }
        }
        4 => {
            // Very High: target >= boundary_3
            let distance_beyond = (target_volume - boundary_3).max(0.0);
            let max_distance = (boundary_3 - boundary_2).abs();
            if max_distance > 0.0 {
                (distance_beyond / max_distance).clamp(0.1, 1.0)
            } else {
                0.5
            }
        }
        _ => 0.5,
    }
}

/// Classify volume regime using logarithmic ratio analysis (legacy function for compatibility)
/// NOTE: This now requires percentile parameters - use classify_volume_with_calibrated_params instead
pub fn classify_volume_regime(
    sequence_volumes: &[f64],
    horizon_volumes: &[f64],
    thresholds: &LogVolumeThresholds,
    config: &VolumeConfig,
    percentile_low: f64,
    percentile_high: f64,
) -> Result<i32> {
    let (class, _strength) = classify_volume_regime_with_strength(
        sequence_volumes,
        horizon_volumes,
        thresholds,
        config,
        percentile_low,
        percentile_high,
    )?;
    Ok(class)
}

/// OLD FUNCTION - Kept for backward compatibility with legacy code
/// Calculate classification strength for volume based on distance from boundaries (LOG-BASED)
///
/// NOTE: This function is no longer used by the main classification logic,
/// which now uses percentile-based approach. Kept for any legacy code that might reference it.
#[allow(dead_code)]
fn calculate_volume_strength(log_ratio: f64, thresholds: &LogVolumeThresholds, class: i32) -> f64 {
    match class {
        0 => {
            // Very Low: log_ratio <= very_low_max
            // The more negative beyond very_low_max, the stronger
            let distance_beyond = (thresholds.very_low_max - log_ratio).max(0.0);
            let max_distance = (thresholds.very_low_max - thresholds.low_max).abs(); // Use range as reference
            (distance_beyond / max_distance).clamp(0.1, 1.0) // At least 0.1 strength
        }
        1 => {
            // Low: very_low_max < log_ratio <= low_max
            let range_center = (thresholds.very_low_max + thresholds.low_max) / 2.0;
            let range_half_width = (thresholds.low_max - thresholds.very_low_max) / 2.0;
            let distance_from_center = (log_ratio - range_center).abs();
            let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
            strength.max(0.1) // At least 0.1 strength
        }
        2 => {
            // Medium: low_max < log_ratio <= medium_max
            // Closer to zero (ln(1.0) = 0) = stronger medium signal
            let distance_from_zero = log_ratio.abs();
            let max_distance = thresholds.medium_max.abs(); // Distance to boundary
            let strength = 1.0 - (distance_from_zero / max_distance).min(1.0);
            strength.max(0.1) // At least 0.1 strength
        }
        3 => {
            // High: medium_max < log_ratio <= high_max
            let range_center = (thresholds.medium_max + thresholds.high_max) / 2.0;
            let range_half_width = (thresholds.high_max - thresholds.medium_max) / 2.0;
            let distance_from_center = (log_ratio - range_center).abs();
            let strength = 1.0 - (distance_from_center / range_half_width).min(1.0);
            strength.max(0.1) // At least 0.1 strength
        }
        4 => {
            // Very High: log_ratio > high_max
            // The more positive beyond high_max, the stronger
            let distance_beyond = (log_ratio - thresholds.high_max).max(0.0);
            let max_distance = (thresholds.high_max - thresholds.medium_max).abs(); // Use range as reference
            (distance_beyond / max_distance).clamp(0.1, 1.0) // At least 0.1 strength
        }
        _ => 0.5, // Default neutral strength
    }
}

/// Calculate logarithmic volume thresholds for regime classification
///
/// Uses the same mathematical approach as volatility target for consistency:
/// - half_bandwidth: Boundary between Medium and Low/High classes
/// - extreme_bandwidth: Boundary between Low/High and VeryLow/VeryHigh classes
///
/// ## Threshold Structure
/// ```
/// VeryLow:  log_ratio <= -extreme_bandwidth
/// Low:      -extreme_bandwidth < log_ratio <= -half_bandwidth
/// Medium:   -half_bandwidth < log_ratio <= +half_bandwidth
/// High:     +half_bandwidth < log_ratio <= +extreme_bandwidth
/// VeryHigh: log_ratio > +extreme_bandwidth
/// ```
pub fn calculate_log_volume_thresholds(config: &VolumeConfig) -> Result<LogVolumeThresholds> {
    let half_bandwidth = config.bandwidth_size / 2.0;
    let extreme_bandwidth = config.bandwidth_size * config.extreme_multiplier;

    let thresholds = LogVolumeThresholds {
        very_low_max: -extreme_bandwidth, // Most negative in log space
        low_max: -half_bandwidth,         // Negative side of medium
        medium_max: half_bandwidth,       // Positive side of medium
        high_max: extreme_bandwidth,      // Most positive before very high
    };

    // Convert log thresholds back to ratio ranges for logging
    let very_low_ratio = (-extreme_bandwidth).exp();
    let low_ratio = (-half_bandwidth).exp();
    let medium_high_ratio = half_bandwidth.exp();
    let high_ratio = extreme_bandwidth.exp();

    log::debug!(
        "🎯 Log Volume Thresholds: bandwidth={:.3}, extreme_multiplier={:.1}, log_thresholds=[{:.4}, {:.4}, {:.4}, {:.4}], ratio_ranges=[{:.3}, {:.3}, {:.3}, {:.3}]",
        config.bandwidth_size, config.extreme_multiplier,
        thresholds.very_low_max, thresholds.low_max, thresholds.medium_max, thresholds.high_max,
        very_low_ratio, low_ratio, medium_high_ratio, high_ratio
    );

    Ok(thresholds)
}

/// Classify volume using logarithmic ratio approach
pub fn classify_volume_log_ratio(log_ratio: f64, thresholds: &LogVolumeThresholds) -> i32 {
    // Classify using log space thresholds
    if log_ratio <= thresholds.very_low_max {
        0 // Very Low
    } else if log_ratio <= thresholds.low_max {
        1 // Low
    } else if log_ratio <= thresholds.medium_max {
        2 // Medium (balanced around ln(1.0) = 0)
    } else if log_ratio <= thresholds.high_max {
        3 // High
    } else {
        4 // Very High
    }
}

/// Calculate smoothed volume using moving average
pub fn calculate_smoothed_volume(volumes: &[f64], smoothing_periods: usize) -> Result<f64> {
    if volumes.is_empty() {
        return Err(VangaError::DataError("Empty volume data".to_string()));
    }

    if smoothing_periods <= 1 || volumes.len() < smoothing_periods {
        // Simple average if insufficient data for smoothing
        let sum: f64 = volumes.iter().filter(|&&v| v > 0.0).sum();
        let count = volumes.iter().filter(|&&v| v > 0.0).count();
        return if count > 0 {
            Ok(sum / count as f64)
        } else {
            Ok(1.0) // Default volume for edge cases
        };
    }

    // Calculate moving average for the last smoothing_periods
    let start_idx = volumes.len().saturating_sub(smoothing_periods);
    let recent_volumes = &volumes[start_idx..];

    let sum: f64 = recent_volumes.iter().filter(|&&v| v > 0.0).sum();
    let count = recent_volumes.iter().filter(|&&v| v > 0.0).count();

    if count > 0 {
        Ok(sum / count as f64)
    } else {
        Ok(1.0) // Default volume for edge cases
    }
}

/// Calculate volume distribution statistics
pub fn calculate_volume_distribution_stats(volumes: &[f64]) -> VolumeDistributionStats {
    if volumes.is_empty() {
        return VolumeDistributionStats {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }

    let valid_volumes: Vec<f64> = volumes.iter().filter(|&&v| v > 0.0).copied().collect();

    if valid_volumes.is_empty() {
        return VolumeDistributionStats {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }

    let mean = valid_volumes.iter().sum::<f64>() / valid_volumes.len() as f64;
    let variance = valid_volumes
        .iter()
        .map(|&v| (v - mean).powi(2))
        .sum::<f64>()
        / valid_volumes.len() as f64;
    let std_dev = variance.sqrt();
    let min = valid_volumes.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = valid_volumes
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    VolumeDistributionStats {
        mean,
        std_dev,
        min,
        max,
    }
}

/// Extract volume data from DataFrame
fn extract_volume_data(df: &DataFrame) -> Result<Vec<f64>> {
    let volume_series = df.column("volume")?;
    let volume_data = volume_series.f64()?;

    let volumes: Vec<f64> = volume_data.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    Ok(volumes)
}

/// Parse horizon string to steps
fn parse_horizon_steps(horizon: &str) -> Result<usize> {
    let horizon_clean = horizon.trim_end_matches('h');
    horizon_clean
        .parse::<usize>()
        .map_err(|_| VangaError::DataError(format!("Invalid horizon format: {}", horizon)))
}

/// Log volume class distribution with logarithmic ratio analysis
fn log_volume_distribution(targets: &[i32], horizon: &str) {
    let class_names = ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"];
    let mut class_counts = [0usize; 5];
    let mut valid_targets = 0;

    for &target in targets {
        if (0..5).contains(&target) {
            class_counts[target as usize] += 1;
            valid_targets += 1;
        }
    }

    if valid_targets == 0 {
        log::warn!("📊 Volume Analysis [{}]: No valid targets found", horizon);
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
        "📊 Volume Distribution [{}]: {} samples, {} | Imbalance: {:.2}x",
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
        "📊 Volume Balance Quality [{}]: {} (target: ~20% per class)",
        horizon,
        balance_quality
    );
}

/// Calibrate volume sensitivity for balanced class distribution
///
/// This function analyzes historical volume data to find the optimal bandwidth_size
/// parameter that achieves the target class balance (e.g., 15% in extreme classes).
///
/// ## Algorithm
/// 1. Sample volume ratios from historical data using the same logic as target generation
/// 2. Convert to logarithmic space for symmetric analysis
/// 3. Find the percentile threshold that corresponds to target_balance for extreme classes
/// 4. Calculate bandwidth_size to achieve that threshold with extreme_multiplier
/// 5. Apply reasonable bounds and return calibrated parameter
///
/// ## Parameters
/// - `volume_data`: Historical volume data for calibration
/// - `sequence_length`: Length of input sequences
/// - `horizon_steps`: Number of steps in prediction horizon
/// - `target_balance`: Target percentage for extreme classes (e.g., 0.15 for 15%)
///
/// ## Returns
/// Calibrated bandwidth_size parameter for balanced volume classification
pub fn calibrate_volume_sensitivity(
    volume_data: &[f64],
    sequence_length: usize,
    horizon_steps: usize,
    target_balance: f64,
) -> Result<f64> {
    if volume_data.len() < sequence_length + horizon_steps + 10 {
        return Ok(0.4); // Default fallback for insufficient data
    }

    let mut log_volume_ratios = Vec::new();

    // Sample volume ratios from the data using same logic as target generation
    for i in 0..(volume_data.len() - sequence_length - horizon_steps) {
        let sequence_volumes = &volume_data[i..i + sequence_length];
        let horizon_volumes =
            &volume_data[i + sequence_length..i + sequence_length + horizon_steps];

        if sequence_volumes.len() >= 3 && horizon_volumes.len() >= 3 {
            // Use same smoothing logic as target generation
            let config = VolumeConfig::default();

            match (
                calculate_smoothed_volume(sequence_volumes, config.smoothing_periods),
                calculate_smoothed_volume(horizon_volumes, config.smoothing_periods),
            ) {
                (Ok(seq_vol), Ok(hor_vol)) if seq_vol > 0.0 && hor_vol > 0.0 => {
                    let volume_ratio = hor_vol / seq_vol;
                    let log_volume_ratio = volume_ratio.ln();

                    if log_volume_ratio.is_finite() {
                        log_volume_ratios.push(log_volume_ratio.abs()); // Use absolute values for threshold calculation
                    }
                }
                _ => continue,
            }
        }
    }

    if log_volume_ratios.is_empty() {
        return Ok(0.4); // Default fallback
    }

    // Sort log ratios to find percentiles
    log_volume_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = log_volume_ratios.len();

    // Find the percentile that corresponds to target_balance for extreme classes
    // We want target_balance% in each extreme class, so (1.0 - 2*target_balance) in middle classes
    let extreme_percentile = 1.0 - target_balance;
    let extreme_idx = ((n as f64) * extreme_percentile) as usize;
    let extreme_threshold = log_volume_ratios[extreme_idx.min(n - 1)];

    // The bandwidth_size should be set so that extreme_threshold becomes the extreme boundary
    // With extreme_multiplier = 2.0: extreme_boundary = bandwidth_size * 2.0
    // So: bandwidth_size = extreme_threshold / 2.0
    let calibrated_bandwidth = extreme_threshold / 2.0;

    // Ensure reasonable bounds for volume analysis
    let final_bandwidth = calibrated_bandwidth.clamp(0.1, 2.0);

    log::info!(
        "🎯 Calibrated volume bandwidth: {:.6} (from {} samples, extreme_threshold: {:.6})",
        final_bandwidth,
        n,
        extreme_threshold
    );

    Ok(final_bandwidth)
}

/// Get volume class names in order
pub fn get_volume_class_names() -> Vec<&'static str> {
    vec!["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
}

// ============================================================================
// PREDICTION RECONSTRUCTION METHODS
// ============================================================================

/// Reconstruction result for volume predictions
#[derive(Debug, Clone)]
pub struct VolumeReconstruction {
    /// Volume ratio ranges for each class [lower_bound, upper_bound]
    pub volume_ratio_ranges: Vec<[f64; 2]>,
    /// Absolute volume ranges for each class [lower_volume, upper_volume]
    pub volume_ranges: Vec<[f64; 2]>,
    /// Class probabilities from model
    pub probabilities: Vec<f64>,
    /// Most likely class index
    pub most_likely_class: usize,
    /// Confidence (probability of most likely class)
    pub confidence: f64,
    /// Expected volume ratio (weighted average)
    pub expected_volume_ratio: f64,
    /// Baseline sequence volume
    pub sequence_volume: f64,
    /// Volume regime interpretation
    pub volume_interpretation: String,
}

/// Reconstruct volume from model probabilities
///
/// **CRITICAL**: Uses LOG-RATIO reconstruction to match training classification logic
/// This ensures perfect mathematical consistency between training and prediction.
pub fn reconstruct_volume(
    probabilities: &[f64],
    sequence_volumes: &[f64],
    calibrated_params: &crate::targets::calibration::VolumeParams,
) -> Result<VolumeReconstruction> {
    if probabilities.len() != 5 {
        return Err(VangaError::DataError(
            "Expected 5 volume probabilities".to_string(),
        ));
    }

    if sequence_volumes.is_empty() {
        return Err(VangaError::DataError(
            "Sequence volumes required for log-ratio reconstruction".to_string(),
        ));
    }

    // RECONSTRUCTION MUST MATCH LOG-RATIO TRAINING LOGIC
    // Training uses: logarithmic ratio classification with adaptive thresholds

    // 1. Calculate sequence median volume (baseline - same as training)
    let mut sorted_seq_volumes = sequence_volumes.to_vec();
    sorted_seq_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let sequence_median = if sorted_seq_volumes.len() % 2 == 0 {
        let mid = sorted_seq_volumes.len() / 2;
        (sorted_seq_volumes[mid - 1] + sorted_seq_volumes[mid]) / 2.0
    } else {
        sorted_seq_volumes[sorted_seq_volumes.len() / 2]
    };

    // Handle edge case: zero volume
    if sequence_median < 1e-10 {
        return Ok(VolumeReconstruction {
            volume_ratio_ranges: vec![[0.0, 1.0]; 5],
            volume_ranges: vec![[0.0, 1.0]; 5],
            probabilities: probabilities.to_vec(),
            most_likely_class: 2,
            confidence: 0.2,
            expected_volume_ratio: 1.0,
            sequence_volume: 1.0,
            volume_interpretation: "Medium volume regime (20% confidence)".to_string(),
        });
    }

    // 2. Calculate adaptive thresholds using percentile-based range (SAME AS TRAINING)
    let plow_idx =
        (sorted_seq_volumes.len() as f64 * calibrated_params.percentile_low).floor() as usize;
    let phigh_idx = ((sorted_seq_volumes.len() as f64 * calibrated_params.percentile_high).ceil()
        as usize)
        .min(sorted_seq_volumes.len() - 1);

    let seq_vol_low = sorted_seq_volumes[plow_idx];
    let seq_vol_high = sorted_seq_volumes[phigh_idx];

    // Calculate typical log variation within sequence (SAME AS TRAINING)
    let typical_log_variation = if seq_vol_low > 1e-10 {
        (seq_vol_high / seq_vol_low).ln()
    } else {
        0.693 // Default to ln(2) = ~69% variation
    };

    // 3. Define symmetric thresholds in log space (SAME AS TRAINING)
    let base_threshold = typical_log_variation * calibrated_params.bandwidth;
    let extreme_threshold = base_threshold * calibrated_params.extreme_multiplier;

    // Symmetric boundaries around 0 (log(1.0) = 0)
    let log_boundary_0 = -extreme_threshold; // Very Low: Major decrease
    let log_boundary_1 = -base_threshold; // Low: Moderate decrease
    let log_boundary_2 = base_threshold; // Medium: Similar volume
    let log_boundary_3 = extreme_threshold; // High: Moderate increase

    // 4. Convert log boundaries to volume ratios
    // Add margin for extreme classes (beyond boundaries)
    let margin = base_threshold * 0.5; // 50% of base threshold for margin

    let ratio_boundary_0 = (log_boundary_0).exp(); // e.g., 0.5 for -0.693
    let ratio_boundary_1 = (log_boundary_1).exp(); // e.g., 0.7 for -0.357
    let ratio_boundary_2 = (log_boundary_2).exp(); // e.g., 1.4 for +0.357
    let ratio_boundary_3 = (log_boundary_3).exp(); // e.g., 2.0 for +0.693

    // 5. Define volume ratio ranges for each class
    let volume_ratio_ranges = vec![
        // Very Low: ratio < boundary_0 (e.g., < 0.5)
        [(log_boundary_0 - margin).exp(), ratio_boundary_0],
        // Low: boundary_0 <= ratio < boundary_1 (e.g., 0.5 - 0.7)
        [ratio_boundary_0, ratio_boundary_1],
        // Medium: boundary_1 <= ratio < boundary_2 (e.g., 0.7 - 1.4)
        [ratio_boundary_1, ratio_boundary_2],
        // High: boundary_2 <= ratio < boundary_3 (e.g., 1.4 - 2.0)
        [ratio_boundary_2, ratio_boundary_3],
        // Very High: ratio >= boundary_3 (e.g., > 2.0)
        [ratio_boundary_3, (log_boundary_3 + margin).exp()],
    ];

    // 6. Calculate representative volume ratios for each class (midpoints in log space)
    let class_representative_log_ratios = [
        log_boundary_0 - margin / 2.0,                    // Very Low
        (log_boundary_0 + log_boundary_1) / 2.0,          // Low
        (log_boundary_1 + log_boundary_2) / 2.0,          // Medium (around 0)
        (log_boundary_2 + log_boundary_3) / 2.0,          // High
        log_boundary_3 + margin / 2.0,                    // Very High
    ];

    // 7. Convert volume ratios to absolute volumes
    let volume_ranges: Vec<[f64; 2]> = volume_ratio_ranges
        .iter()
        .map(|[lower_ratio, upper_ratio]| {
            [
                lower_ratio * sequence_median,
                upper_ratio * sequence_median,
            ]
        })
        .collect();

    // 8. Find most likely class and confidence
    let (most_likely_class, confidence) = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &prob)| (idx, prob))
        .unwrap_or((2, 0.2)); // Default to MEDIUM

    // 9. Calculate expected volume ratio (weighted average in log space, then convert)
    let expected_log_ratio: f64 = probabilities
        .iter()
        .zip(class_representative_log_ratios.iter())
        .map(|(&prob, &log_ratio)| prob * log_ratio)
        .sum();

    let expected_volume_ratio = expected_log_ratio.exp();

    // 10. Generate interpretation
    let class_names = ["Very Low", "Low", "Medium", "High", "Very High"];
    let volume_interpretation = format!(
        "{} volume regime ({}% confidence)",
        class_names[most_likely_class],
        (confidence * 100.0) as i32
    );

    log::debug!(
        "🔄 Volume Reconstruction: seq_median={:.2}, log_boundaries=[{:.3}, {:.3}, {:.3}, {:.3}], ratio_boundaries=[{:.3}, {:.3}, {:.3}, {:.3}], expected_ratio={:.3}",
        sequence_median,
        log_boundary_0, log_boundary_1, log_boundary_2, log_boundary_3,
        ratio_boundary_0, ratio_boundary_1, ratio_boundary_2, ratio_boundary_3,
        expected_volume_ratio
    );

    Ok(VolumeReconstruction {
        volume_ratio_ranges,
        volume_ranges,
        probabilities: probabilities.to_vec(),
        most_likely_class,
        confidence,
        expected_volume_ratio,
        sequence_volume: sequence_median,
        volume_interpretation,
    })
}
