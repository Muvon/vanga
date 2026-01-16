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
    prefix: &str,
) -> Result<VolumeParams> {
    use super::bayesian::BayesianConfig;

    log::info!(
        "{} 🔬 Starting Bayesian Optimization for Volume calibration",
        prefix
    );

    let utils = calibrator.get_utils();

    // Define 5D parameter space with WIDE, ADAPTIVE bounds for all market conditions
    // CRITICAL: Percentile bounds MUST be wide enough for longer horizons (64h+)
    let param_bounds = vec![
        (0.1, 3.0),   // bandwidth: 0.1-3.0 (narrow to very wide volume ranges)
        (1.2, 6.0),   // extreme_multiplier: 1.2-6.0 (narrow to very wide extremes)
        (1.0, 30.0),  // smoothing_periods: 1-30 (no smoothing to heavy smoothing)
        (0.01, 0.30), // percentile_low: 0.01-0.30 (p1 to p30) - WIDER for longer horizons
        (0.70, 0.99), // percentile_high: 0.70-0.99 (p70 to p99) - WIDER for longer horizons
    ];

    let param_names = vec![
        "bandwidth".to_string(),
        "extreme_multiplier".to_string(),
        "smoothing_periods".to_string(),
        "percentile_low".to_string(),
        "percentile_high".to_string(),
    ];

    // Objective function: minimize composite_quality_score
    let objective_fn = |params: &[f64]| -> Result<f64> {
        let test_params = VolumeEvalParams {
            bandwidth: params[0],
            multiplier: params[1],
            smoothing: params[2].round() as usize, // Round to integer
            percentile_low: params[3],
            percentile_high: params[4],
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
    // Use quality-first Bayesian configuration (default for 5D space)
    let bayesian_config = BayesianConfig::default();

    // Run Bayesian optimization
    let best_params = calibrator
        .calibrate_with_bayesian(
            param_bounds,
            param_names,
            objective_fn,
            bayesian_config,
            prefix,
        )
        .await?;

    // Evaluate final parameters to get balance
    let final_eval_params = VolumeEvalParams {
        bandwidth: best_params[0],
        multiplier: best_params[1],
        smoothing: best_params[2].round() as usize,
        percentile_low: best_params[3],
        percentile_high: best_params[4],
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
        percentile_low: best_params[3],
        percentile_high: best_params[4],
        balance: final_balance,
    };

    log::info!(
        "🎯 Volume Calibration Complete!\n  Final Parameters:\n    - Bandwidth: {:.2}\n    - Extreme Multiplier: {:.1}\n    - Smoothing Periods: {}\n    - Percentile Range: p{:.0}-p{:.0}\n    - Min Base Threshold: {:.4}\n    - Min Extreme Threshold: {:.4}\n  Final Score: {:.4}",
        result.bandwidth,
        result.extreme_multiplier,
        result.smoothing_periods,
        result.percentile_low * 100.0,
        result.percentile_high * 100.0,
        result.min_base_threshold,
        result.min_extreme_threshold,
        result.balance.composite_quality_score
    );

    Ok(result)
}

/// Evaluate volume parameters using PERCENTILE-BASED classification (MATCHES TARGET GENERATION)
///
/// CRITICAL: This MUST use the same percentile-based logic as classify_volume_regime_with_strength()
/// in volume.rs to ensure calibration optimizes the ACTUAL classification algorithm being used.
fn evaluate_volume_params(
    utils: &super::utils::CalibrationUtils,
    context: &EvaluationContext,
    params: &VolumeEvalParams,
) -> Result<ClassBalance> {
    let mut class_counts = [0usize; 5];
    let sample_limit = context.sample_indices.len().min(500); // Limit for performance

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

                // CRITICAL: Use PERCENTILE-BASED classification with smoothing (same as target generation)
                match classify_volume_percentile_based(
                    &sequence_volumes,
                    &horizon_volumes,
                    params.bandwidth,
                    params.multiplier,
                    params.smoothing,
                    params.percentile_low,
                    params.percentile_high,
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
    // NOTE: Missing classes are handled gracefully by calculate_balance_with_diversity()
    // which sets imbalance_ratio = f64::INFINITY and poor balance score automatically
    utils.calculate_balance_with_diversity(
        class_counts.as_ref(),
        total,
        context.ohlcv_data,
        context.sample_indices,
        context.sequence_length,
    )
}

/// Logarithmic ratio volume classification (MATCHES classify_volume_regime_with_strength)
///
/// This function MUST match classify_volume_regime_with_strength() in volume.rs
/// to ensure calibration optimizes the correct classification algorithm.
///
/// **CRITICAL**: Uses smoothed volume with logarithmic ratio approach for symmetric classification.
fn classify_volume_percentile_based(
    sequence_volumes: &[f64],
    horizon_volumes: &[f64],
    bandwidth_size: f64,
    extreme_multiplier: f64,
    smoothing: usize,
    percentile_low: f64,
    percentile_high: f64,
) -> Result<i32> {
    use crate::targets::volume::calculate_smoothed_volume;
    use crate::utils::error::VangaError;

    if sequence_volumes.is_empty() || horizon_volumes.is_empty() {
        return Err(VangaError::DataError(
            "Empty volume data for analysis".to_string(),
        ));
    }

    // 1. Calculate smoothed sequence volume (baseline - matches training)
    let sequence_smooth =
        calculate_smoothed_volume(sequence_volumes, smoothing).unwrap_or_else(|_| {
            let sum: f64 = sequence_volumes.iter().filter(|&&v| v > 0.0).sum();
            let count = sequence_volumes.iter().filter(|&&v| v > 0.0).count();
            if count > 0 {
                sum / count as f64
            } else {
                1.0
            }
        });

    // 2. Calculate smoothed horizon volume (target - matches training)
    let horizon_smooth =
        calculate_smoothed_volume(horizon_volumes, smoothing).unwrap_or_else(|_| {
            let sum: f64 = horizon_volumes.iter().filter(|&&v| v > 0.0).sum();
            let count = horizon_volumes.iter().filter(|&&v| v > 0.0).count();
            if count > 0 {
                sum / count as f64
            } else {
                1.0
            }
        });

    // Handle edge case: zero volume
    if sequence_smooth < 1e-10 {
        return Ok(2); // Default to medium
    }

    // 3. Calculate volume ratio and apply logarithmic transformation
    let volume_ratio = horizon_smooth / sequence_smooth;
    let log_ratio = volume_ratio.ln();

    // 4. Calculate adaptive thresholds using percentile-based range
    let mut sorted_seq_volumes = sequence_volumes.to_vec();
    sorted_seq_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
    let base_threshold = typical_log_variation * bandwidth_size;
    let extreme_threshold = base_threshold * extreme_multiplier;

    // Symmetric boundaries around 0 (log(1.0) = 0)
    let boundary_0 = -extreme_threshold; // Very Low: Major decrease
    let boundary_1 = -base_threshold; // Low: Moderate decrease
    let boundary_2 = base_threshold; // Medium: Similar volume
    let boundary_3 = extreme_threshold; // High: Moderate increase

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

    Ok(class)
}
