//! Direction Calibration Module
//!
//! Contains direction-specific calibration logic including parameter optimization,
//! evaluation functions, and classification helpers.

use super::core::ParameterCalibrator;
use super::types::*;
use crate::data::structures::MarketDataRow;
use crate::utils::error::Result;

/// Calibrate direction parameters using Bayesian optimization
pub async fn calibrate_direction(
    calibrator: &ParameterCalibrator,
    ohlcv_data: &[MarketDataRow],
    sequence_length: usize,
    horizon_steps: usize,
    sample_indices: &[usize],
) -> Result<DirectionParams> {
    use super::bayesian::BayesianConfig;

    log::info!("🔬 Starting Bayesian Optimization for Direction calibration");

    let close_prices: Vec<f64> = ohlcv_data.iter().map(|row| row.close).collect();
    let utils = calibrator.get_utils();

    // Define 5D parameter space with WIDE, ADAPTIVE bounds for all market conditions
    let param_bounds = vec![
        (0.005, 0.8),   // sensitivity: 0.005-0.8 (very sensitive to very conservative)
        (1.2, 6.0),     // extreme_multiplier: 1.2-6.0 (narrow to very wide extremes)
        (0.0001, 0.02), // min_base_threshold: 0.01%-2% (minimum movement detection)
        (0.001, 0.05),  // min_extreme_threshold: 0.1%-5% (minimum extreme movement)
        (1.0, 50.0),    // base_multiplier: 1-50 (adaptive scaling for different volatilities)
    ];

    let param_names = vec![
        "sensitivity".to_string(),
        "extreme_multiplier".to_string(),
        "min_base_threshold".to_string(),
        "min_extreme_threshold".to_string(),
        "base_multiplier".to_string(),
    ];

    // Objective function: minimize composite_quality_score
    let objective_fn = |params: &[f64]| -> Result<f64> {
        let test_params = DirectionEvalParams {
            sensitivity: params[0],
            extreme_multiplier: params[1],
            min_base_threshold: params[2],
            min_extreme_threshold: params[3],
            base_multiplier: params[4],
        };

        let balance = evaluate_direction_params(
            &utils,
            &close_prices,
            ohlcv_data,
            sample_indices,
            sequence_length,
            horizon_steps,
            &test_params,
        )?;

        // Return score only if diversity is acceptable
        if balance.diversity_score >= 0.3 {
            Ok(balance.composite_quality_score)
        } else {
            // Penalize low diversity
            Ok(balance.composite_quality_score + 10.0)
        }
    };

    // Use quality-first Bayesian configuration (default for 5D space)
    let bayesian_config = BayesianConfig::default();

    // Run Bayesian optimization
    let best_params = calibrator
        .calibrate_with_bayesian(param_bounds, param_names, objective_fn, bayesian_config)
        .await?;

    // Evaluate final parameters to get balance
    let final_eval_params = DirectionEvalParams {
        sensitivity: best_params[0],
        extreme_multiplier: best_params[1],
        min_base_threshold: best_params[2],
        min_extreme_threshold: best_params[3],
        base_multiplier: best_params[4],
    };

    let final_balance = evaluate_direction_params(
        &utils,
        &close_prices,
        ohlcv_data,
        sample_indices,
        sequence_length,
        horizon_steps,
        &final_eval_params,
    )?;

    let result = DirectionParams {
        sensitivity: best_params[0],
        extreme_multiplier: best_params[1],
        min_base_threshold: best_params[2],
        min_extreme_threshold: best_params[3],
        base_multiplier: best_params[4],
        balance: final_balance,
    };

    log::info!(
        "🎯 Direction Calibration Complete!\n  Final Parameters:\n    - Sensitivity: {:.4}\n    - Extreme Multiplier: {:.1}\n    - Min Base Threshold: {:.3}\n    - Min Extreme Threshold: {:.3}\n    - Base Multiplier: {:.1}\n  Final Score: {:.4}",
        result.sensitivity,
        result.extreme_multiplier,
        result.min_base_threshold,
        result.min_extreme_threshold,
        result.base_multiplier,
        result.balance.composite_quality_score
    );

    Ok(result)
}
/// Evaluate direction parameters with REAL diversity metrics
fn evaluate_direction_params(
    utils: &super::utils::CalibrationUtils,
    close_prices: &[f64],
    ohlcv_data: &[MarketDataRow],
    sample_indices: &[usize],
    sequence_length: usize,
    horizon_steps: usize,
    params: &DirectionEvalParams,
) -> Result<ClassBalance> {
    let mut class_counts = vec![0; 5];
    let mut total = 0;

    for &idx in sample_indices {
        let seq_end = idx + sequence_length;
        let target_end = seq_end + horizon_steps;

        if target_end <= close_prices.len() {
            let sequence_prices = &close_prices[idx..seq_end];
            let horizon_prices = &close_prices[seq_end..target_end];

            // Use the same logic as the actual direction classification but with calibrated parameters
            let class = classify_direction_with_params(sequence_prices, horizon_prices, params)?;

            if (0..5).contains(&class) {
                class_counts[class as usize] += 1;
                total += 1;
            }
        }
    }

    // Use diversity-aware balance calculation
    utils.calculate_balance_with_diversity(&class_counts, total, ohlcv_data, sample_indices)
}

/// Classify direction using calibrated parameters (mirrors actual classification logic)
fn classify_direction_with_params(
    sequence_prices: &[f64],
    horizon_prices: &[f64],
    params: &DirectionEvalParams,
) -> Result<i32> {
    if sequence_prices.len() < 2 || horizon_prices.len() < 2 {
        return Ok(2); // Default to SIDEWAYS for insufficient data
    }

    // Calculate momentum change (same as actual implementation)
    let (_, _, momentum_change) =
        calculate_directional_momentum_change(sequence_prices, horizon_prices)?;

    // Calculate sequence trend consistency (same as actual implementation)
    let trend_consistency = calculate_sequence_trend_consistency(sequence_prices)?;

    // Use calibrated parameters instead of hardcoded values
    let base_threshold_calc = trend_consistency * params.sensitivity * params.base_multiplier;
    let extreme_threshold_calc = base_threshold_calc * params.extreme_multiplier;

    // Apply calibrated minimum thresholds instead of hardcoded 0.01 and 0.03
    let final_base_threshold = base_threshold_calc.max(params.min_base_threshold);
    let final_extreme_threshold = extreme_threshold_calc.max(params.min_extreme_threshold);

    // Same classification logic as actual implementation
    let class = if momentum_change <= -final_extreme_threshold {
        0 // DUMP
    } else if momentum_change <= -final_base_threshold {
        1 // DOWN
    } else if momentum_change.abs() <= final_base_threshold {
        2 // SIDEWAYS
    } else if momentum_change <= final_extreme_threshold {
        3 // UP
    } else {
        4 // PUMP
    };

    Ok(class)
}

/// Calculate directional momentum change (helper for calibration)
fn calculate_directional_momentum_change(
    sequence_prices: &[f64],
    horizon_prices: &[f64],
) -> Result<(f64, f64, f64)> {
    // Same logic as in direction.rs - USE PERCENTAGE CHANGE, NOT RAW SLOPE
    if sequence_prices.len() < 2 || horizon_prices.len() < 2 {
        return Ok((0.0, 0.0, 0.0));
    }

    let seq_start = sequence_prices[0];
    let seq_end = sequence_prices[sequence_prices.len() - 1];

    // Avoid division by zero - use epsilon check
    let sequence_momentum = if seq_start.abs() < 1e-10 {
        0.0
    } else {
        (seq_end - seq_start) / seq_start
    };

    let hor_start = horizon_prices[0];
    let hor_end = horizon_prices[horizon_prices.len() - 1];

    // Avoid division by zero - use epsilon check
    let horizon_momentum = if hor_start.abs() < 1e-10 {
        0.0
    } else {
        (hor_end - hor_start) / hor_start
    };

    let momentum_change = horizon_momentum - sequence_momentum;
    Ok((sequence_momentum, horizon_momentum, momentum_change))
}

/// Calculate sequence trend consistency (helper for calibration)
fn calculate_sequence_trend_consistency(sequence_prices: &[f64]) -> Result<f64> {
    if sequence_prices.len() < 3 {
        return Ok(0.01); // Minimum consistency for short sequences
    }

    let mut momentum_changes = Vec::new();

    // Calculate momentum between consecutive segments - MUST MATCH direction.rs EXACTLY
    let segment_size = (sequence_prices.len() / 3).max(2);
    for i in 0..(sequence_prices.len() - segment_size * 2) {
        let seg1_start = sequence_prices[i];
        let seg1_end = sequence_prices[i + segment_size];
        let seg2_start = seg1_end;
        let seg2_end = sequence_prices[i + segment_size * 2];

        if seg1_start != 0.0 && seg2_start != 0.0 {
            let seg1_momentum = (seg1_end - seg1_start) / seg1_start;
            let seg2_momentum = (seg2_end - seg2_start) / seg2_start;
            let momentum_change = seg2_momentum - seg1_momentum;

            if momentum_change.is_finite() {
                momentum_changes.push(momentum_change);
            }
        }
    }

    if momentum_changes.is_empty() {
        return Ok(0.01);
    }

    // Calculate standard deviation of momentum changes (trend consistency)
    let mean = momentum_changes.iter().sum::<f64>() / momentum_changes.len() as f64;
    let variance = momentum_changes
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / momentum_changes.len() as f64;
    let std_dev = variance.sqrt();

    Ok(std_dev.max(0.005)) // Minimum consistency threshold
}
