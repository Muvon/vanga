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

    if calibrator.stop_level_definition.uses_anchor_volatility()? {
        return calibrate_anchor_stop_levels(calibrator, context, prefix).await;
    }

    log::info!(
        "{} 🔬 Starting Bayesian Optimization for Stop Levels calibration",
        prefix
    );

    let utils = calibrator.get_utils();

    // Define 4D parameter space (all parameters that affect boundary calculation)
    let param_bounds = vec![
        (0.1, 3.0),   // bandwidth: 0.1-3.0 (controls extreme zone size)
        (0.01, 0.45), // percentile_low: 1%-45% = min from weighted adverse
        (0.55, 0.99), // percentile_high: 55%-99% = max from weighted adverse
        (0.1, 0.8),   // neutral_band_factor: 10%-80% (controls neutral zone size)
    ];

    let param_names = vec![
        "bandwidth".to_string(),
        "percentile_low".to_string(),
        "percentile_high".to_string(),
        "neutral_band_factor".to_string(),
    ];

    // Objective function: minimize composite_quality_score
    let objective_fn = |params: &[f64]| -> Result<f64> {
        let test_params = StopLevelEvalParams {
            bandwidth: params[0],
            percentiles: [params[1], params[2]],
            neutral_band: params[3],
            momentum_factor: 1.0, // Fixed - not used in boundary calculation
        };

        let balance = evaluate_stop_level_params(&utils, context, &test_params)?;

        // Same as price_levels: balance AND diversity, so adverse boundaries
        // are not placed purely to equalise class counts
        Ok(balance.composite_quality_score)
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
        neutral_band: best_params[3],
        momentum_factor: 1.0,
    };

    let final_balance = evaluate_stop_level_params(&utils, context, &final_eval_params)?;

    let result = StopLevelParams {
        bandwidth: best_params[0],
        percentiles: [best_params[1], best_params[2]],
        neutral_band_factor: best_params[3],
        momentum_factor: 1.0, // Fixed default
        balance: final_balance,
        ..Default::default()
    };

    log::info!(
        "🛑 Stop Level Calibration Complete!
  Final Parameters:
    - Bandwidth: {:.2}
    - Percentiles: [{:.2}, {:.2}]
    - Neutral Band Factor: {:.2}
  Final Score: {:.4}",
        result.bandwidth,
        result.percentiles[0],
        result.percentiles[1],
        result.neutral_band_factor,
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
        ..Default::default()
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

/// Empirical gaps seed the same Bayesian threshold search used by price levels.
async fn calibrate_anchor_stop_levels(
    calibrator: &ParameterCalibrator,
    context: &EvaluationContext<'_>,
    prefix: &str,
) -> Result<StopLevelParams> {
    use crate::targets::stop_levels::{
        anchor_stop_level_statistic, anchor_stop_level_statistic_with_history,
        anchor_stop_level_thresholds, classify_stop_level_with_history,
    };
    use crate::utils::error::VangaError;

    let mut statistics = Vec::new();
    let mut valid_indices = Vec::new();
    let regime_window_bars = if calibrator.stop_level_definition.uses_regime_volatility() {
        let window_bars = calibrator.regime_window_bars.ok_or_else(|| {
            VangaError::config(
                "Regime-volatility stop calibration requires features.stats_window_bars",
            )
        })?;
        if window_bars < 2 {
            return Err(VangaError::config(
                "Regime-volatility stop calibration requires features.stats_window_bars >= 2",
            ));
        }
        Some(window_bars)
    } else {
        None
    };
    let base_params = StopLevelParams {
        definition: calibrator.stop_level_definition,
        regime_window_bars,
        ..Default::default()
    };
    let mut insufficient_history = 0usize;
    for &start in context.sample_indices {
        let end = start
            .checked_add(context.sequence_length)
            .ok_or_else(|| VangaError::data("Stop-level sequence index overflow"))?;
        let future_end = end
            .checked_add(context.horizon_steps)
            .ok_or_else(|| VangaError::data("Stop-level horizon index overflow"))?;
        if future_end <= context.ohlcv_data.len() {
            if regime_window_bars.is_some_and(|window_bars| end < window_bars) {
                insufficient_history += 1;
                continue;
            }
            let sequence = &context.ohlcv_data[start..end];
            let history = if regime_window_bars.is_some() {
                &context.ohlcv_data[..end]
            } else {
                sequence
            };
            let future = &context.ohlcv_data[end..future_end];
            statistics.push(if regime_window_bars.is_some() {
                anchor_stop_level_statistic_with_history(sequence, history, future, &base_params)?
            } else {
                anchor_stop_level_statistic(
                    sequence,
                    future,
                    calibrator.stop_level_definition.side,
                )?
            });
            valid_indices.push(start);
        }
    }
    if insufficient_history > 0 {
        log::warn!(
            "{} Skipped {} stop-level calibration anchors because regime history was incomplete",
            prefix,
            insufficient_history
        );
    }
    if statistics.len() < 5 {
        return Err(VangaError::data(
            "Stop-level calibration requires at least five valid samples",
        ));
    }
    statistics.sort_by(f64::total_cmp);
    let seed = super::price_levels::distinct_gap_seed(&statistics).ok_or_else(|| {
        VangaError::data("Stop-level calibration requires at least five distinct excursions")
    })?;
    let mut result = StopLevelParams {
        definition: calibrator.stop_level_definition,
        regime_window_bars,
        ..Default::default()
    };
    let utils = calibrator.get_utils();
    let evaluate = |thresholds: [f64; 4]| -> Result<ClassBalance> {
        let params = StopLevelParams {
            thresholds: Some(thresholds),
            ..result.clone()
        };
        let mut counts = [0usize; 5];
        // Evaluate in the same price space as labels and served stops, avoiding
        // a second floating-point convention at exact threshold prices.
        for &start in &valid_indices {
            let end = start + context.sequence_length;
            let sequence = &context.ohlcv_data[start..end];
            let history = if regime_window_bars.is_some() {
                &context.ohlcv_data[..end]
            } else {
                sequence
            };
            let (class, _) = classify_stop_level_with_history(
                sequence,
                history,
                &context.ohlcv_data[end..end + context.horizon_steps],
                &params,
            )?;
            counts[class as usize] += 1;
        }
        utils.calculate_balance_with_diversity(
            &counts,
            statistics.len(),
            context.ohlcv_data,
            &valid_indices,
            context.sequence_length,
        )
    };
    let sorted_thresholds = |values: &[f64]| {
        let mut thresholds = [values[0], values[1], values[2], values[3]];
        thresholds.sort_by(f64::total_cmp);
        thresholds
    };
    let seed_balance = evaluate(seed)?;
    let best = calibrator
        .calibrate_with_bayesian(
            vec![(statistics[0], statistics[statistics.len() - 1]); 4],
            (0..4)
                .map(|index| format!("stop_level_threshold_{index}"))
                .collect(),
            |values: &[f64]| {
                let thresholds = sorted_thresholds(values);
                if thresholds.windows(2).any(|pair| pair[0] >= pair[1]) {
                    return Ok(f64::INFINITY);
                }
                Ok(evaluate(thresholds)?.composite_quality_score)
            },
            super::bayesian::BayesianConfig {
                initial_candidate: Some(seed.to_vec()),
                objective_floor: (seed_balance.balance_score == 0.0)
                    .then_some(seed_balance.composite_quality_score),
                ..Default::default()
            },
            prefix,
        )
        .await?;
    let thresholds = sorted_thresholds(&best);
    let balance = evaluate(thresholds)?;
    if balance.class_percentages.contains(&0.0) {
        return Err(VangaError::data(
            "Final stop-level thresholds must produce five non-empty calibration classes",
        ));
    }
    result.thresholds = Some(thresholds);
    anchor_stop_level_thresholds(&result)?;
    result.balance = balance;
    Ok(result)
}
