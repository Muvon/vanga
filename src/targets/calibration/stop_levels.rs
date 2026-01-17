//! Stop level calibration using Bayesian optimization
//!
//! This module calibrates stop level parameters to achieve balanced 5-class distribution
//! using the same Bayesian optimization approach as price_levels.

use super::core::ParameterCalibrator;
use super::types::*;
use crate::utils::error::Result;

/// Calibrate stop level parameters using Bayesian optimization
pub async fn calibrate_stop_levels(
    calibrator: &ParameterCalibrator,
    context: &EvaluationContext<'_>,
    prefix: &str,
) -> Result<StopLevelParams> {
    use super::bayesian::BayesianConfig;

    log::info!(
        "{} 🔬 Starting Bayesian Optimization for Stop Levels calibration",
        prefix
    );

    let utils = calibrator.get_utils();

    // Define 3D parameter space (only what classification uses)
    let param_bounds = vec![
        (0.01, 0.75), // bandwidth: 5%-90% of range = neutral zone width
        (0.01, 0.45), // percentile_low: 1%-45% = min from weighted lows
        (0.55, 0.99), // percentile_high: 55%-99% = max from weighted highs
    ];

    let param_names = vec![
        "bandwidth".to_string(),
        "percentile_low".to_string(),
        "percentile_high".to_string(),
    ];

    // Objective function: minimize balance_score
    let objective_fn = |params: &[f64]| -> Result<f64> {
        let test_params = StopLevelEvalParams {
            bandwidth: params[0],
            percentiles: [params[1], params[2]],
            neutral_band: 0.4,    // Fixed default - not calibrated
            momentum_factor: 1.0, // Fixed default - not used
        };

        let balance = evaluate_stop_level_params(&utils, context, &test_params)?;

        // Use balance_score (same as price_levels)
        Ok(balance.balance_score)
    };

    // Bayesian optimization configuration
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
    let final_eval_params = StopLevelEvalParams {
        bandwidth: best_params[0],
        percentiles: [best_params[1], best_params[2]],
        neutral_band: 0.4,
        momentum_factor: 1.0,
    };

    let final_balance = evaluate_stop_level_params(&utils, context, &final_eval_params)?;

    let result = StopLevelParams {
        bandwidth: best_params[0],
        percentiles: [best_params[1], best_params[2]],
        neutral_band_factor: 0.4, // Fixed default
        momentum_factor: 1.0,     // Fixed default
        balance: final_balance,
    };

    log::info!(
        "🛑 Stop Level Calibration Complete!
  Final Parameters:
    - Bandwidth (neutral zone width): {:.2}
    - Percentiles: [{:.2}, {:.2}]
  Final Score: {:.4}",
        result.bandwidth,
        result.percentiles[0],
        result.percentiles[1],
        result.balance.balance_score
    );

    Ok(result)
}

/// Evaluate stop level parameters and return balance metrics
fn evaluate_stop_level_params(
    utils: &super::utils::CalibrationUtils,
    context: &EvaluationContext<'_>,
    params: &StopLevelEvalParams,
) -> Result<ClassBalance> {
    use crate::targets::stop_levels::classify_stop_level_with_calibrated_params;

    // Create full StopLevelParams for classification
    let stop_params = StopLevelParams {
        bandwidth: params.bandwidth,
        percentiles: params.percentiles,
        neutral_band_factor: params.neutral_band,
        momentum_factor: params.momentum_factor,
        balance: ClassBalance::default(),
    };

    // Generate targets with these parameters
    let mut targets = Vec::new();
    for &idx in context.sample_indices {
        let sequence_end = idx + context.sequence_length;
        let target_end = sequence_end + context.horizon_steps;

        if target_end <= context.ohlcv_data.len() {
            let sequence_ohlcv = &context.ohlcv_data[idx..sequence_end];
            let horizon_ohlcv = &context.ohlcv_data[sequence_end..target_end];

            if let Ok((class, _)) = classify_stop_level_with_calibrated_params(
                sequence_ohlcv,
                horizon_ohlcv,
                &stop_params,
            ) {
                targets.push(class);
            }
        }
    }

    if targets.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "No valid stop level targets generated".to_string(),
        ));
    }

    // Count class frequencies
    let mut class_counts = [0usize; 5];
    for &class in &targets {
        if (0..5).contains(&class) {
            class_counts[class as usize] += 1;
        }
    }

    // Calculate balance using CalibrationUtils
    let balance = utils.calculate_balance(&class_counts, targets.len())?;

    Ok(balance)
}
