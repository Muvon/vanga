//! Volume Calibration Module
//!
//! Contains volume-specific calibration logic including regime classification,
//! smoothing period optimization, and bandwidth calculations.

use super::core::ParameterCalibrator;
use super::types::*;
use crate::utils::error::Result;

/// Calibrate volume parameters using Bayesian optimization
pub async fn calibrate_volume(
    calibrator: &ParameterCalibrator,
    context: &EvaluationContext<'_>,
) -> Result<VolumeParams> {
    use super::bayesian::BayesianConfig;

    log::info!("🔬 Starting Bayesian Optimization for Volume calibration");

    let utils = calibrator.get_utils();

    // Define 3D parameter space with WIDE, ADAPTIVE bounds for all market conditions
    let param_bounds = vec![
        (0.1, 3.0),  // bandwidth: 0.1-3.0 (narrow to very wide volume ranges)
        (1.2, 6.0),  // extreme_multiplier: 1.2-6.0 (narrow to very wide extremes)
        (1.0, 30.0), // smoothing_periods: 1-30 (no smoothing to heavy smoothing)
    ];

    let param_names = vec![
        "bandwidth".to_string(),
        "extreme_multiplier".to_string(),
        "smoothing_periods".to_string(),
    ];

    // Objective function: minimize composite_quality_score
    let objective_fn = |params: &[f64]| -> Result<f64> {
        let test_params = VolumeEvalParams {
            bandwidth: params[0],
            multiplier: params[1],
            smoothing: params[2].round() as usize, // Round to integer
        };

        let balance = evaluate_volume_params(&utils, context, &test_params)?;

        // Return score only if diversity is acceptable
        if balance.diversity_score >= 0.3 {
            Ok(balance.composite_quality_score)
        } else {
            // Penalize low diversity
            Ok(balance.composite_quality_score + 10.0)
        }
    };

    // Bayesian optimization configuration
    // Use quality-first Bayesian configuration (default for 3D space)
    let bayesian_config = BayesianConfig::default();

    // Run Bayesian optimization
    let best_params = calibrator
        .calibrate_with_bayesian(param_bounds, param_names, objective_fn, bayesian_config)
        .await?;

    // Evaluate final parameters to get balance
    let final_eval_params = VolumeEvalParams {
        bandwidth: best_params[0],
        multiplier: best_params[1],
        smoothing: best_params[2].round() as usize,
    };

    let final_balance = evaluate_volume_params(&utils, context, &final_eval_params)?;

    // Calculate derived thresholds
    let min_base_threshold = best_params[0] * 0.1;
    let min_extreme_threshold = best_params[0] * best_params[1] * 0.1;

    let result = VolumeParams {
        bandwidth: best_params[0],
        extreme_multiplier: best_params[1],
        smoothing_periods: best_params[2].round() as usize,
        min_base_threshold,
        min_extreme_threshold,
        balance: final_balance,
    };

    log::info!(
        "🎯 Volume Calibration Complete!\n  Final Parameters:\n    - Bandwidth: {:.2}\n    - Extreme Multiplier: {:.1}\n    - Smoothing Periods: {}\n    - Min Base Threshold: {:.4}\n    - Min Extreme Threshold: {:.4}\n  Final Score: {:.4}",
        result.bandwidth,
        result.extreme_multiplier,
        result.smoothing_periods,
        result.min_base_threshold,
        result.min_extreme_threshold,
        result.balance.composite_quality_score
    );

    Ok(result)
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
    // Use diversity-aware balance calculation
    utils.calculate_balance_with_diversity(
        class_counts.as_ref(),
        total,
        context.ohlcv_data,
        context.sample_indices,
    )
}
