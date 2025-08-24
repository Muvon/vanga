//! Direction Calibration Module
//!
//! Contains direction-specific calibration logic including parameter optimization,
//! evaluation functions, and classification helpers.

use super::core::ParameterCalibrator;
use super::types::*;
use crate::data::structures::MarketDataRow;
use crate::utils::error::Result;

/// Calibrate direction parameters
pub async fn calibrate_direction(
    calibrator: &ParameterCalibrator,
    ohlcv_data: &[MarketDataRow],
    sequence_length: usize,
    horizon_steps: usize,
    sample_indices: &[usize],
) -> Result<DirectionParams> {
    log::info!("🔬 Starting state-of-the-art direction calibration - optimizing each parameter independently");

    let close_prices: Vec<f64> = ohlcv_data.iter().map(|row| row.close).collect();

    // Start with reasonable defaults
    let mut current_sensitivity = 0.05;
    let mut current_multiplier = 2.5;
    let mut current_min_base = 0.005;
    let mut current_min_extreme = 0.01;
    let mut current_base_mult = 10.0;

    let mut total_tested = 0;
    let mut total_improvements = 0;

    // Parameter ranges
    let sensitivities = vec![0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5];
    let multipliers = vec![1.5, 2.0, 2.5, 3.0, 4.0, 5.0];
    let min_base_thresholds = vec![0.001, 0.003, 0.005, 0.01, 0.015];
    let min_extreme_thresholds = vec![0.005, 0.01, 0.015, 0.02, 0.03];
    let base_multipliers = vec![2.0, 5.0, 10.0, 15.0, 20.0, 30.0];

    let utils = calibrator.get_utils();

    // Step 1: Optimize sensitivity
    log::info!("📊 Step 1/5: Optimizing sensitivity parameter...");
    let mut best_sensitivity_score = f64::INFINITY;
    for &sensitivity in &sensitivities {
        total_tested += 1;
        let params = DirectionEvalParams {
            sensitivity,
            extreme_multiplier: current_multiplier,
            min_base_threshold: current_min_base,
            min_extreme_threshold: current_min_extreme,
            base_multiplier: current_base_mult,
        };

        let balance = evaluate_direction_params_extended(
            &utils,
            &close_prices,
            sample_indices,
            sequence_length,
            horizon_steps,
            &params,
        )?;

        if balance.composite_quality_score < best_sensitivity_score
            && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_sensitivity_score = balance.composite_quality_score;
            current_sensitivity = sensitivity;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better sensitivity found: {:.4} (score: {:.4})",
                sensitivity,
                best_sensitivity_score
            );
        }
    }
    log::info!(
        "  → Best sensitivity: {:.4} (tested {} values)",
        current_sensitivity,
        sensitivities.len()
    );

    // Step 2: Optimize extreme multiplier
    log::info!("📊 Step 2/5: Optimizing extreme multiplier...");
    let mut best_multiplier_score = f64::INFINITY;
    for &multiplier in &multipliers {
        total_tested += 1;
        let params = DirectionEvalParams {
            sensitivity: current_sensitivity,
            extreme_multiplier: multiplier,
            min_base_threshold: current_min_base,
            min_extreme_threshold: current_min_extreme,
            base_multiplier: current_base_mult,
        };

        let balance = evaluate_direction_params_extended(
            &utils,
            &close_prices,
            sample_indices,
            sequence_length,
            horizon_steps,
            &params,
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

    // Step 3: Optimize min base threshold
    log::info!("📊 Step 3/5: Optimizing minimum base threshold...");
    let mut best_base_score = f64::INFINITY;
    for &min_base in &min_base_thresholds {
        total_tested += 1;
        let params = DirectionEvalParams {
            sensitivity: current_sensitivity,
            extreme_multiplier: current_multiplier,
            min_base_threshold: min_base,
            min_extreme_threshold: current_min_extreme,
            base_multiplier: current_base_mult,
        };

        let balance = evaluate_direction_params_extended(
            &utils,
            &close_prices,
            sample_indices,
            sequence_length,
            horizon_steps,
            &params,
        )?;

        if balance.composite_quality_score < best_base_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_base_score = balance.composite_quality_score;
            current_min_base = min_base;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better min base found: {:.3} (score: {:.4})",
                min_base,
                best_base_score
            );
        }
    }
    log::info!(
        "  → Best min base threshold: {:.3} (tested {} values)",
        current_min_base,
        min_base_thresholds.len()
    );

    // Step 4: Optimize min extreme threshold
    log::info!("📊 Step 4/5: Optimizing minimum extreme threshold...");
    let mut best_extreme_score = f64::INFINITY;
    for &min_extreme in &min_extreme_thresholds {
        total_tested += 1;
        let params = DirectionEvalParams {
            sensitivity: current_sensitivity,
            extreme_multiplier: current_multiplier,
            min_base_threshold: current_min_base,
            min_extreme_threshold: min_extreme,
            base_multiplier: current_base_mult,
        };

        let balance = evaluate_direction_params_extended(
            &utils,
            &close_prices,
            sample_indices,
            sequence_length,
            horizon_steps,
            &params,
        )?;

        if balance.composite_quality_score < best_extreme_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_extreme_score = balance.composite_quality_score;
            current_min_extreme = min_extreme;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better min extreme found: {:.3} (score: {:.4})",
                min_extreme,
                best_extreme_score
            );
        }
    }
    log::info!(
        "  → Best min extreme threshold: {:.3} (tested {} values)",
        current_min_extreme,
        min_extreme_thresholds.len()
    );

    // Step 5: Optimize base multiplier
    log::info!("📊 Step 5/5: Optimizing base multiplier...");
    let mut best_base_mult_score = f64::INFINITY;
    let mut final_balance = ClassBalance::default();
    for &base_mult in &base_multipliers {
        total_tested += 1;
        let params = DirectionEvalParams {
            sensitivity: current_sensitivity,
            extreme_multiplier: current_multiplier,
            min_base_threshold: current_min_base,
            min_extreme_threshold: current_min_extreme,
            base_multiplier: base_mult,
        };

        let balance = evaluate_direction_params_extended(
            &utils,
            &close_prices,
            sample_indices,
            sequence_length,
            horizon_steps,
            &params,
        )?;

        if balance.composite_quality_score < best_base_mult_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_base_mult_score = balance.composite_quality_score;
            current_base_mult = base_mult;
            final_balance = balance;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better base multiplier found: {:.1} (score: {:.4})",
                base_mult,
                best_base_mult_score
            );
        }
    }
    log::info!(
        "  → Best base multiplier: {:.1} (tested {} values)",
        current_base_mult,
        base_multipliers.len()
    );

    let best_params = DirectionParams {
        sensitivity: current_sensitivity,
        extreme_multiplier: current_multiplier,
        min_base_threshold: current_min_base,
        min_extreme_threshold: current_min_extreme,
        base_multiplier: current_base_mult,
        balance: final_balance,
    };

    log::info!(
        "🎯 Direction Calibration Complete!\n  Tested: {} combinations\n  Improvements: {}\n  Final Parameters:\n    - Sensitivity: {:.4}\n    - Extreme Multiplier: {:.1}\n    - Min Base Threshold: {:.3}\n    - Min Extreme Threshold: {:.3}\n    - Base Multiplier: {:.1}\n  Final Score: {:.4}",
        total_tested,
        total_improvements,
        best_params.sensitivity,
        best_params.extreme_multiplier,
        best_params.min_base_threshold,
        best_params.min_extreme_threshold,
        best_params.base_multiplier,
        best_base_mult_score
    );

    Ok(best_params)
}

/// Evaluate direction parameters with extended calibration including previously hardcoded values
fn evaluate_direction_params_extended(
    utils: &super::utils::CalibrationUtils,
    close_prices: &[f64],
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

    utils.calculate_balance(&class_counts, total)
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
