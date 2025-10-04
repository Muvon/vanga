//! Volatility Calibration Module
//!
//! Contains volatility-specific calibration logic including ATR analysis,
//! volume weighting, and horizon decay calculations.

use super::core::ParameterCalibrator;
use super::types::*;
use crate::utils::error::Result;

/// Calibrate volatility parameters using Bayesian optimization
pub async fn calibrate_volatility(
    calibrator: &ParameterCalibrator,
    context: &EvaluationContext<'_>,
) -> Result<VolatilityParams> {
    use super::bayesian::{AcquisitionFunction, BayesianConfig};

    log::info!("🔬 Starting Bayesian Optimization for Volatility calibration");

    let utils = calibrator.get_utils();

    // Define 5D parameter space
    let param_bounds = vec![
        (0.1, 1.0),    // bandwidth
        (1.5, 3.0),    // extreme_multiplier
        (0.85, 1.0),   // horizon_decay
        (0.05, 0.3),   // volume_weight
        (0.001, 0.01), // min_volatility_baseline
    ];

    let param_names = vec![
        "bandwidth".to_string(),
        "extreme_multiplier".to_string(),
        "horizon_decay".to_string(),
        "volume_weight".to_string(),
        "min_volatility_baseline".to_string(),
    ];

    // Objective function: minimize composite_quality_score
    let objective_fn = |params: &[f64]| -> Result<f64> {
        let test_params = VolatilityEvalParams {
            bandwidth: params[0],
            multiplier: params[1],
            decay: params[2],
            volume_weight: params[3],
            min_baseline: params[4],
        };

        let balance = evaluate_volatility_params(&utils, context, &test_params)?;

        // Return score only if diversity is acceptable
        if balance.diversity_score >= 0.3 {
            Ok(balance.composite_quality_score)
        } else {
            // Penalize low diversity
            Ok(balance.composite_quality_score + 10.0)
        }
    };

    // Bayesian optimization configuration
    let bayesian_config = BayesianConfig {
        n_initial: 15,
        max_iterations: 50,
        tolerance: 1e-4,
        acquisition: AcquisitionFunction::ExpectedImprovement,
        gp_length_scale: 0.5,
        gp_noise: 1e-6,
    };

    // Run Bayesian optimization
    let best_params = calibrator
        .calibrate_with_bayesian(param_bounds, param_names, objective_fn, bayesian_config)
        .await?;

    // Evaluate final parameters to get balance
    let final_eval_params = VolatilityEvalParams {
        bandwidth: best_params[0],
        multiplier: best_params[1],
        decay: best_params[2],
        volume_weight: best_params[3],
        min_baseline: best_params[4],
    };

    let final_balance = evaluate_volatility_params(&utils, context, &final_eval_params)?;

    let result = VolatilityParams {
        bandwidth: best_params[0],
        extreme_multiplier: best_params[1],
        volume_weight: best_params[3],
        horizon_decay: best_params[2],
        min_volatility_baseline: best_params[4],
        balance: final_balance,
    };

    log::info!(
        "🎯 Volatility Calibration Complete!\n  Final Parameters:\n    - Bandwidth: {:.2}\n    - Extreme Multiplier: {:.1}\n    - Volume Weight: {:.2}\n    - Horizon Decay: {:.2}\n    - Min Baseline: {:.3}\n  Final Score: {:.4}",
        result.bandwidth,
        result.extreme_multiplier,
        result.volume_weight,
        result.horizon_decay,
        result.min_volatility_baseline,
        result.balance.composite_quality_score
    );

    Ok(result)
}

/// Evaluate volatility parameters using simplified ATR momentum classification
fn evaluate_volatility_params(
    utils: &super::utils::CalibrationUtils,
    context: &EvaluationContext,
    params: &VolatilityEvalParams,
) -> Result<ClassBalance> {
    use crate::targets::volatility::classify_volatility_with_calibrated_params;

    let mut class_counts = [0usize; 5];

    // Create calibrated parameters for the new simplified approach
    let calibrated_params = VolatilityParams {
        bandwidth: params.bandwidth,
        extreme_multiplier: params.multiplier,
        volume_weight: params.volume_weight, // Use the calibrated volume weight
        horizon_decay: params.decay,         // Use the passed decay parameter
        min_volatility_baseline: params.min_baseline, // Use the calibrated min baseline
        balance: Default::default(),
    };

    // Process each sample using the new simplified classification
    for &seq_idx in context.sample_indices {
        let sequence_end_idx = seq_idx + context.sequence_length;
        let target_end_idx = sequence_end_idx + context.horizon_steps;

        if target_end_idx <= context.ohlcv_data.len() {
            let sequence_candles = &context.ohlcv_data[seq_idx..sequence_end_idx];
            let horizon_candles = &context.ohlcv_data[sequence_end_idx..target_end_idx];

            if sequence_candles.len() >= 2 && horizon_candles.len() >= 2 {
                // Use the new simplified classification approach
                if let Ok((class, _strength)) = classify_volatility_with_calibrated_params(
                    sequence_candles,
                    horizon_candles,
                    &calibrated_params,
                ) {
                    if (0..5).contains(&class) {
                        class_counts[class as usize] += 1;
                    }
                }
            }
        }
    }

    let total = class_counts.iter().sum::<usize>();
    utils.calculate_balance(class_counts.as_ref(), total)
}
