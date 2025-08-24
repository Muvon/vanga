//! Volume Calibration Module
//!
//! Contains volume-specific calibration logic including regime classification,
//! smoothing period optimization, and bandwidth calculations.

use super::core::ParameterCalibrator;
use super::types::*;
use crate::utils::error::Result;

/// Calibrate volume parameters
pub async fn calibrate_volume(
    calibrator: &ParameterCalibrator,
    context: &EvaluationContext<'_>,
) -> Result<VolumeParams> {
    log::info!(
        "🔬 Starting state-of-the-art volume calibration - optimizing each parameter independently"
    );

    // Start with reasonable defaults
    let mut current_bandwidth = 0.4;
    let mut current_multiplier = 2.5;
    let mut current_smoothing = 5;

    let mut total_tested = 0;
    let mut total_improvements = 0;

    // Parameter ranges
    let bandwidths = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let multipliers = vec![1.5, 2.0, 2.5, 3.0, 4.0, 5.0];
    let smoothing_values = vec![1, 3, 5, 7, 9, 11, 15];

    let utils = calibrator.get_utils();

    // Step 1: Optimize bandwidth
    log::info!("📊 Step 1/3: Optimizing bandwidth parameter...");
    let mut best_bandwidth_score = f64::INFINITY;
    for &bandwidth in &bandwidths {
        total_tested += 1;
        let balance = evaluate_volume_params(
            &utils,
            context,
            &VolumeEvalParams {
                bandwidth,
                multiplier: current_multiplier,
                smoothing: current_smoothing,
            },
        )?;

        if balance.composite_quality_score < best_bandwidth_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_bandwidth_score = balance.composite_quality_score;
            current_bandwidth = bandwidth;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better bandwidth found: {:.2} (score: {:.4})",
                bandwidth,
                best_bandwidth_score
            );
        }
    }
    log::info!(
        "  → Best bandwidth: {:.2} (tested {} values)",
        current_bandwidth,
        bandwidths.len()
    );

    // Step 2: Optimize extreme multiplier
    log::info!("📊 Step 2/3: Optimizing extreme multiplier...");
    let mut best_multiplier_score = f64::INFINITY;
    for &multiplier in &multipliers {
        total_tested += 1;
        let balance = evaluate_volume_params(
            &utils,
            context,
            &VolumeEvalParams {
                bandwidth: current_bandwidth,
                multiplier,
                smoothing: current_smoothing,
            },
        )?;

        if balance.composite_quality_score < best_multiplier_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_multiplier_score = balance.composite_quality_score;
            current_multiplier = multiplier;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better multiplier found: {:.1} (score: {:.4})",
                multiplier,
                best_multiplier_score
            );
        }
    }
    log::info!(
        "  → Best extreme multiplier: {:.1} (tested {} values)",
        current_multiplier,
        multipliers.len()
    );

    // Step 3: Optimize smoothing periods
    log::info!("📊 Step 3/3: Optimizing smoothing periods...");
    let mut best_smoothing_score = f64::INFINITY;
    let mut final_balance = ClassBalance::default();
    for &smoothing in &smoothing_values {
        total_tested += 1;
        let balance = evaluate_volume_params(
            &utils,
            context,
            &VolumeEvalParams {
                bandwidth: current_bandwidth,
                multiplier: current_multiplier,
                smoothing,
            },
        )?;

        if balance.composite_quality_score < best_smoothing_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_smoothing_score = balance.composite_quality_score;
            current_smoothing = smoothing;
            final_balance = balance;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better smoothing found: {} (score: {:.4})",
                smoothing,
                best_smoothing_score
            );
        }
    }
    log::info!(
        "  → Best smoothing periods: {} (tested {} values)",
        current_smoothing,
        smoothing_values.len()
    );

    // Calculate derived thresholds
    let min_base_threshold = current_bandwidth * 0.1; // 10% of bandwidth as minimum
    let min_extreme_threshold = current_bandwidth * current_multiplier * 0.1;

    let best_params = VolumeParams {
        bandwidth: current_bandwidth,
        extreme_multiplier: current_multiplier,
        smoothing_periods: current_smoothing,
        min_base_threshold,
        min_extreme_threshold,
        balance: final_balance,
    };

    log::info!(
        "🎯 Volume Calibration Complete!\n  Tested: {} combinations\n  Improvements: {}\n  Final Parameters:\n    - Bandwidth: {:.2}\n    - Extreme Multiplier: {:.1}\n    - Smoothing Periods: {}\n    - Min Base Threshold: {:.4}\n    - Min Extreme Threshold: {:.4}\n  Final Score: {:.4}",
        total_tested,
        total_improvements,
        best_params.bandwidth,
        best_params.extreme_multiplier,
        best_params.smoothing_periods,
        best_params.min_base_threshold,
        best_params.min_extreme_threshold,
        best_smoothing_score
    );

    Ok(best_params)
}

/// Evaluate volume parameters using proper volume regime classification
fn evaluate_volume_params(
    utils: &super::utils::CalibrationUtils,
    context: &EvaluationContext,
    params: &VolumeEvalParams,
) -> Result<ClassBalance> {
    use crate::targets::volume::{classify_volume_regime, LogVolumeThresholds, VolumeConfig};

    let mut class_counts = [0usize; 5];
    let sample_limit = context.sample_indices.len().min(500); // Limit for performance

    let config = VolumeConfig {
        bandwidth_size: params.bandwidth,
        extreme_multiplier: params.multiplier,
        smoothing_periods: params.smoothing,
    };

    // Create thresholds using same logic as volume.rs
    let half_bandwidth = params.bandwidth / 2.0;
    let extreme_bandwidth = params.bandwidth * params.multiplier;

    let thresholds = LogVolumeThresholds {
        very_low_max: -extreme_bandwidth,
        low_max: -half_bandwidth,
        medium_max: half_bandwidth,
        high_max: extreme_bandwidth,
    };

    for &seq_idx in context.sample_indices.iter().take(sample_limit) {
        let sequence_end_idx = seq_idx + context.sequence_length;
        let target_end_idx = sequence_end_idx + context.horizon_steps;

        if target_end_idx <= context.ohlcv_data.len() {
            let sequence_data = &context.ohlcv_data[seq_idx..sequence_end_idx];
            let horizon_data = &context.ohlcv_data[sequence_end_idx..target_end_idx];

            if sequence_data.len() >= 2 && horizon_data.len() >= 2 {
                // Extract volume data from OHLCV
                let sequence_volumes: Vec<f64> =
                    sequence_data.iter().map(|row| row.volume).collect();
                let horizon_volumes: Vec<f64> = horizon_data.iter().map(|row| row.volume).collect();

                match classify_volume_regime(
                    &sequence_volumes,
                    &horizon_volumes,
                    &thresholds,
                    &config,
                ) {
                    Ok(class) => {
                        if (0..5).contains(&class) {
                            class_counts[class as usize] += 1;
                        }
                    }
                    Err(_) => continue, // Skip invalid classifications
                }
            }
        }
    }

    let total = class_counts.iter().sum::<usize>();
    utils.calculate_balance(class_counts.as_ref(), total)
}
