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

                // CRITICAL: Use PERCENTILE-BASED classification (same as target generation)
                match classify_volume_percentile_based(
                    &sequence_volumes,
                    &horizon_volumes,
                    params.bandwidth,
                    params.multiplier,
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
    
    // CRITICAL: Ensure ALL 5 classes are present before calculating balance
    // Missing classes during Bayesian optimization are NORMAL - they get penalized automatically
    let missing_classes: Vec<usize> = (0..5)
        .filter(|&i| class_counts[i] == 0)
        .collect();
    
    if !missing_classes.is_empty() {
        // Return poor score to guide optimization away from these parameters
        // NO LOGGING - this is expected during exploration and just creates noise
        return Ok(ClassBalance {
            class_percentages: [0.0; 5],
            balance_score: 10.0, // Very poor balance
            imbalance_ratio: f64::INFINITY,
            total_samples: total,
            target_balance: 0.2,
            diversity_score: 0.0,
            temporal_spread: 0.0,
            feature_diversity: 0.0,
            market_condition_diversity: 0.0,
            composite_quality_score: 10.0,
        });
    }
    
    // Use diversity-aware balance calculation
    utils.calculate_balance_with_diversity(
        class_counts.as_ref(),
        total,
        context.ohlcv_data,
        context.sample_indices,
    )
}

/// Percentile-based volume classification (EXACT COPY of logic from volume.rs)
///
/// This function MUST match classify_volume_regime_with_strength() in volume.rs
/// to ensure calibration optimizes the correct classification algorithm.
fn classify_volume_percentile_based(
    sequence_volumes: &[f64],
    horizon_volumes: &[f64],
    bandwidth_size: f64,
    _extreme_multiplier: f64, // Not used in percentile-based classification
) -> Result<i32> {
    use crate::utils::error::VangaError;

    if sequence_volumes.is_empty() || horizon_volumes.is_empty() {
        return Err(VangaError::DataError(
            "Empty volume data for analysis".to_string(),
        ));
    }

    // 1. Calculate sequence volume percentiles to establish range
    let mut sorted_seq_volumes = sequence_volumes.to_vec();
    sorted_seq_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p10_idx = (sorted_seq_volumes.len() as f64 * 0.15).floor() as usize;
    let p90_idx = ((sorted_seq_volumes.len() as f64 * 0.85).ceil() as usize)
        .min(sorted_seq_volumes.len() - 1);

    let sequence_volume_min = sorted_seq_volumes[p10_idx];
    let sequence_volume_max = sorted_seq_volumes[p90_idx];
    let sequence_volume_range = sequence_volume_max - sequence_volume_min;

    // 2. Calculate horizon median volume (more robust than mean)
    let mut sorted_hor_volumes = horizon_volumes.to_vec();
    sorted_hor_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let horizon_median_volume = if sorted_hor_volumes.len() % 2 == 0 {
        let mid = sorted_hor_volumes.len() / 2;
        (sorted_hor_volumes[mid - 1] + sorted_hor_volumes[mid]) / 2.0
    } else {
        sorted_hor_volumes[sorted_hor_volumes.len() / 2]
    };

    // 3. Calculate bandwidth for breakout detection (similar to price levels)
    let bandwidth = sequence_volume_range * bandwidth_size;

    // Handle edge case: flat volume
    if sequence_volume_range < 1e-10 || bandwidth < 1e-10 {
        return Ok(2); // Default to medium
    }

    // 4. Define classification boundaries (5-class system)
    let boundary_0 = sequence_volume_min - bandwidth; // Strong Down boundary
    let boundary_1 = sequence_volume_min; // Moderate Down boundary
    let boundary_2 = sequence_volume_max; // Neutral/Moderate Up boundary
    let boundary_3 = sequence_volume_max + bandwidth; // Strong Up boundary

    // 5. Classify based on where horizon median falls
    let class = if horizon_median_volume < boundary_0 {
        0 // Very Low: Major volume decrease
    } else if horizon_median_volume < boundary_1 {
        1 // Low: Moderate volume decrease
    } else if horizon_median_volume < boundary_2 {
        2 // Medium: Within sequence range
    } else if horizon_median_volume < boundary_3 {
        3 // High: Moderate volume increase
    } else {
        4 // Very High: Major volume surge
    };

    Ok(class)
}
