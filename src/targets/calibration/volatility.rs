//! Volatility Calibration Module
//!
//! Contains volatility-specific calibration logic including ATR analysis,
//! volume weighting, and horizon decay calculations.

use super::core::ParameterCalibrator;
use super::types::*;
use crate::utils::error::Result;

/// Calibrate volatility parameters using proper ATR analysis with extended grid search
pub async fn calibrate_volatility(
    calibrator: &ParameterCalibrator,
    context: &EvaluationContext<'_>,
) -> Result<VolatilityParams> {
    log::info!("🔬 Starting state-of-the-art volatility calibration - optimizing each parameter independently");

    // Start with reasonable defaults
    let mut current_bandwidth = 0.3;
    let mut current_multiplier = 2.0;
    let mut current_decay = 0.95;
    let mut current_volume_weight = 0.15;
    let mut current_min_baseline = 0.005;

    let mut total_tested = 0;
    let mut total_improvements = 0;

    // Parameter ranges
    let bandwidths = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0];
    let multipliers = vec![1.5, 2.0, 2.5, 3.0];
    let decay_factors = vec![0.85, 0.90, 0.95, 1.0];
    let volume_weights = vec![0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
    let min_volatility_baselines = vec![0.001, 0.003, 0.005, 0.007, 0.01];

    let utils = calibrator.get_utils();

    // Step 1: Optimize bandwidth
    log::info!("📊 Step 1/5: Optimizing bandwidth parameter...");
    let mut best_bandwidth_score = f64::INFINITY;
    for &bandwidth in &bandwidths {
        total_tested += 1;
        let balance = evaluate_volatility_params(
            &utils,
            context,
            &VolatilityEvalParams {
                bandwidth,
                multiplier: current_multiplier,
                decay: current_decay,
                volume_weight: current_volume_weight,
                min_baseline: current_min_baseline,
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
    log::info!("📊 Step 2/5: Optimizing extreme multiplier...");
    let mut best_multiplier_score = f64::INFINITY;
    for &multiplier in &multipliers {
        total_tested += 1;
        let balance = evaluate_volatility_params(
            &utils,
            context,
            &VolatilityEvalParams {
                bandwidth: current_bandwidth,
                multiplier,
                decay: current_decay,
                volume_weight: current_volume_weight,
                min_baseline: current_min_baseline,
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

    // Step 3: Optimize decay factor
    log::info!("📊 Step 3/5: Optimizing horizon decay factor...");
    let mut best_decay_score = f64::INFINITY;
    for &decay in &decay_factors {
        total_tested += 1;
        let balance = evaluate_volatility_params(
            &utils,
            context,
            &VolatilityEvalParams {
                bandwidth: current_bandwidth,
                multiplier: current_multiplier,
                decay,
                volume_weight: current_volume_weight,
                min_baseline: current_min_baseline,
            },
        )?;

        if balance.composite_quality_score < best_decay_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_decay_score = balance.composite_quality_score;
            current_decay = decay;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better decay found: {:.2} (score: {:.4})",
                decay,
                best_decay_score
            );
        }
    }
    log::info!(
        "  → Best horizon decay: {:.2} (tested {} values)",
        current_decay,
        decay_factors.len()
    );

    // Step 4: Optimize volume weight
    log::info!("📊 Step 4/5: Optimizing volume weight...");
    let mut best_volume_score = f64::INFINITY;
    for &volume_weight in &volume_weights {
        total_tested += 1;
        let balance = evaluate_volatility_params(
            &utils,
            context,
            &VolatilityEvalParams {
                bandwidth: current_bandwidth,
                multiplier: current_multiplier,
                decay: current_decay,
                volume_weight,
                min_baseline: current_min_baseline,
            },
        )?;

        if balance.composite_quality_score < best_volume_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_volume_score = balance.composite_quality_score;
            current_volume_weight = volume_weight;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better volume weight found: {:.2} (score: {:.4})",
                volume_weight,
                best_volume_score
            );
        }
    }
    log::info!(
        "  → Best volume weight: {:.2} (tested {} values)",
        current_volume_weight,
        volume_weights.len()
    );

    // Step 5: Optimize min volatility baseline
    log::info!("📊 Step 5/5: Optimizing minimum volatility baseline...");
    let mut best_baseline_score = f64::INFINITY;
    let mut final_balance = ClassBalance::default();
    for &min_baseline in &min_volatility_baselines {
        total_tested += 1;
        let balance = evaluate_volatility_params(
            &utils,
            context,
            &VolatilityEvalParams {
                bandwidth: current_bandwidth,
                multiplier: current_multiplier,
                decay: current_decay,
                volume_weight: current_volume_weight,
                min_baseline,
            },
        )?;

        if balance.composite_quality_score < best_baseline_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_baseline_score = balance.composite_quality_score;
            current_min_baseline = min_baseline;
            final_balance = balance;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better min baseline found: {:.3} (score: {:.4})",
                min_baseline,
                best_baseline_score
            );
        }
    }
    log::info!(
        "  → Best min baseline: {:.3} (tested {} values)",
        current_min_baseline,
        min_volatility_baselines.len()
    );

    let best_params = VolatilityParams {
        bandwidth: current_bandwidth,
        extreme_multiplier: current_multiplier,
        volume_weight: current_volume_weight,
        horizon_decay: current_decay,
        min_volatility_baseline: current_min_baseline,
        balance: final_balance,
    };

    log::info!(
        "🎯 Volatility Calibration Complete!\n  Tested: {} combinations\n  Improvements: {}\n  Final Parameters:\n    - Bandwidth: {:.2}\n    - Extreme Multiplier: {:.1}\n    - Volume Weight: {:.2}\n    - Horizon Decay: {:.2}\n    - Min Baseline: {:.3}\n  Final Score: {:.4}",
        total_tested,
        total_improvements,
        best_params.bandwidth,
        best_params.extreme_multiplier,
        best_params.volume_weight,
        best_params.horizon_decay,
        best_params.min_volatility_baseline,
        best_baseline_score
    );

    Ok(best_params)
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
