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
//! 1. Calculate sequence average volume (baseline)
//! 2. Calculate horizon average volume (target)
//! 3. Compute volume ratio: horizon_volume / sequence_volume
//! 4. Apply logarithmic transformation: ln(volume_ratio)
//! 5. Classify using symmetric thresholds in log space
//! ```
//!
//! ### **5-Class Volume Classification:**
//! - **0: Very Low** - Major volume decrease (>50% drop)
//! - **1: Low** - Moderate volume decrease (20-50% drop)
//! - **2: Medium** - Similar volume (±20% change)
//! - **3: High** - Moderate volume increase (20-100% increase)
//! - **4: Very High** - Major volume surge (>100% increase)
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
//! ### **Adaptive Thresholds**
//! - Automatically calibrated for balanced 20% per class distribution
//! - Adjusts to volume volatility and market conditions
//! - Uses same pattern as volatility target for consistency

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
/// target generation between training and prediction. When None, uses base config.
pub fn generate_volume_targets_with_calibrated_params(
    df: &DataFrame,
    horizons: &[String],
    sequence_indices: &[usize],
    sequence_length: usize,
    calibrated_params: &crate::targets::calibration::VolumeParams, // Now mandatory
) -> Result<HashMap<String, Vec<i32>>> {
    let volume_data = extract_volume_data(df)?;

    // Use pre-calibrated parameters (always available)
    log::info!(
        "🎯 Using calibrated volume parameters: bandwidth={:.4}, extreme_multiplier={:.2}, smoothing={}",
        calibrated_params.bandwidth,
        calibrated_params.extreme_multiplier,
        calibrated_params.smoothing_periods
    );

    let config = VolumeConfig {
        bandwidth_size: calibrated_params.bandwidth,
        extreme_multiplier: calibrated_params.extreme_multiplier,
        smoothing_periods: calibrated_params.smoothing_periods,
    };

    log::info!(
        "🎯 Volume targets using calibrated bandwidth: {:.6}",
        calibrated_params.bandwidth
    );

    let mut targets = HashMap::new();

    // Calculate logarithmic volume thresholds
    let thresholds = calculate_log_volume_thresholds(&config)?;

    for horizon in horizons {
        let horizon_steps = parse_horizon_steps(horizon)?;
        let mut horizon_targets = Vec::new();

        for &seq_start in sequence_indices {
            let seq_end = seq_start + sequence_length;
            let horizon_end = seq_end + horizon_steps;

            if horizon_end > volume_data.len() {
                continue;
            }

            let sequence_volumes = &volume_data[seq_start..seq_end];
            let horizon_volumes = &volume_data[seq_end..horizon_end];

            match classify_volume_regime(sequence_volumes, horizon_volumes, &thresholds, &config) {
                Ok(class) => horizon_targets.push(class),
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
        }
    }

    Ok(targets)
}

/// Classify volume regime using logarithmic ratio analysis
pub fn classify_volume_regime(
    sequence_volumes: &[f64],
    horizon_volumes: &[f64],
    thresholds: &LogVolumeThresholds,
    config: &VolumeConfig,
) -> Result<i32> {
    if sequence_volumes.is_empty() || horizon_volumes.is_empty() {
        return Err(VangaError::DataError(
            "Empty volume data for analysis".to_string(),
        ));
    }

    // Calculate smoothed average volumes
    let sequence_avg_volume =
        calculate_smoothed_volume(sequence_volumes, config.smoothing_periods)?;
    let horizon_avg_volume = calculate_smoothed_volume(horizon_volumes, config.smoothing_periods)?;

    // Handle edge cases
    if sequence_avg_volume <= 0.0 || horizon_avg_volume <= 0.0 {
        return Ok(2); // Default to medium for invalid volume values
    }

    // Calculate volume ratio and log transformation
    let volume_ratio = horizon_avg_volume / sequence_avg_volume;
    let log_volume_ratio = volume_ratio.ln();

    // Classify using logarithmic thresholds
    let class = classify_volume_log_ratio(log_volume_ratio, thresholds);

    log::debug!(
        "🎯 Volume Analysis: seq_vol={:.2}, hor_vol={:.2}, ratio={:.3}, log_ratio={:.4} → class={} ({})",
        sequence_avg_volume, horizon_avg_volume, volume_ratio, log_volume_ratio, class,
        ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"][class as usize]
    );

    Ok(class)
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
pub fn reconstruct_volume(
    probabilities: &[f64],
    sequence_volume: f64,
    calibrated_params: &crate::targets::calibration::VolumeParams,
) -> Result<VolumeReconstruction> {
    if probabilities.len() != 5 {
        return Err(VangaError::DataError(
            "Expected 5 volume probabilities".to_string(),
        ));
    }

    // Convert calibrated parameters to log thresholds
    let bandwidth = calibrated_params.bandwidth;
    let extreme_multiplier = calibrated_params.extreme_multiplier;

    let half_bandwidth = bandwidth / 2.0;
    let extreme_bandwidth = bandwidth * extreme_multiplier;

    // Convert log thresholds to ratio boundaries
    let ratio_boundaries = [
        (-extreme_bandwidth).exp(), // Very Low upper bound
        (-half_bandwidth).exp(),    // Low upper bound
        half_bandwidth.exp(),       // Medium upper bound
        extreme_bandwidth.exp(),    // High upper bound
    ];

    // Define volume ratio ranges for each class
    let volume_ratio_ranges = vec![
        [0.0, ratio_boundaries[0]],                 // Very Low
        [ratio_boundaries[0], ratio_boundaries[1]], // Low
        [ratio_boundaries[1], ratio_boundaries[2]], // Medium
        [ratio_boundaries[2], ratio_boundaries[3]], // High
        [ratio_boundaries[3], f64::INFINITY],       // Very High
    ];

    // Convert to absolute volume ranges
    let volume_ranges: Vec<[f64; 2]> = volume_ratio_ranges
        .iter()
        .map(|[lower_ratio, upper_ratio]| {
            [
                sequence_volume * lower_ratio,
                if upper_ratio.is_infinite() {
                    f64::INFINITY
                } else {
                    sequence_volume * upper_ratio
                },
            ]
        })
        .collect();

    // Find most likely class
    let most_likely_class = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(2); // Default to medium

    let confidence = probabilities[most_likely_class];

    // Calculate expected volume ratio (weighted average)
    let class_midpoints = [
        ratio_boundaries[0] * 0.5,                         // Very Low
        (ratio_boundaries[0] + ratio_boundaries[1]) / 2.0, // Low
        (ratio_boundaries[1] + ratio_boundaries[2]) / 2.0, // Medium
        (ratio_boundaries[2] + ratio_boundaries[3]) / 2.0, // High
        ratio_boundaries[3] * 1.5,                         // Very High
    ];

    let expected_volume_ratio = probabilities
        .iter()
        .zip(class_midpoints.iter())
        .map(|(prob, midpoint)| prob * midpoint)
        .sum::<f64>();

    // Generate interpretation
    let class_names = get_volume_class_names();
    let volume_interpretation = format!(
        "{} (confidence: {:.1}%, expected ratio: {:.2}x)",
        class_names[most_likely_class],
        confidence * 100.0,
        expected_volume_ratio
    );

    Ok(VolumeReconstruction {
        volume_ratio_ranges,
        volume_ranges,
        probabilities: probabilities.to_vec(),
        most_likely_class,
        confidence,
        expected_volume_ratio,
        sequence_volume,
        volume_interpretation,
    })
}
