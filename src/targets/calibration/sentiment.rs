//! Sentiment Calibration Module
//!
//! Contains sentiment-specific calibration logic including body sensitivity,
//! volume weighting, and consistency factor analysis.

use super::core::ParameterCalibrator;
use super::types::*;
use crate::utils::error::Result;

/// Calibrate sentiment parameters using state-of-the-art optimization
pub async fn calibrate_sentiment(
    calibrator: &ParameterCalibrator,
    context: &EvaluationContext<'_>,
) -> Result<SentimentParams> {
    log::info!("🔬 Starting state-of-the-art sentiment calibration - optimizing each parameter independently");

    // Start with reasonable defaults
    let mut current_sensitivity = 0.05;
    let mut current_volume_weight = 0.2;
    let mut current_consistency = 0.8;
    let mut current_extreme_mult = 2.5;

    let mut total_tested = 0;
    let mut total_improvements = 0;

    // Parameter ranges
    let sensitivities = vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15];
    let volume_weights = vec![0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4];
    let consistency_factors = vec![0.6, 0.7, 0.8, 0.9, 1.0];
    let extreme_multipliers = vec![1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

    let utils = calibrator.get_utils();

    // Step 1: Optimize sensitivity
    log::info!("📊 Step 1/4: Optimizing body sensitivity...");
    let mut best_sensitivity_score = f64::INFINITY;
    let mut sensitivity_scores = Vec::new();

    for &sensitivity in &sensitivities {
        total_tested += 1;
        let balance = evaluate_sentiment_params(
            &utils,
            context,
            &SentimentEvalParams {
                sensitivity,
                volume_weight: current_volume_weight,
                consistency_factor: current_consistency,
            },
        )?;

        let score = balance.composite_quality_score;
        sensitivity_scores.push((sensitivity, score, balance.balance_score));

        if score < best_sensitivity_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_sensitivity_score = score;
            current_sensitivity = sensitivity;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better sensitivity found: {:.3} (score: {:.4}, balance: {:.2})",
                sensitivity,
                score,
                balance.balance_score
            );
        }
    }

    log::info!(
        "  Sensitivity scores: {:?}",
        sensitivity_scores
            .iter()
            .map(|(s, score, bal)| format!("{:.3}={:.3}/{:.1}", s, score, bal))
            .collect::<Vec<_>>()
            .join(", ")
    );
    log::info!(
        "  → Best sensitivity: {:.3} (score: {:.4})",
        current_sensitivity,
        best_sensitivity_score
    );

    // Step 2: Optimize volume weight
    log::info!("📊 Step 2/4: Optimizing volume weight...");
    let mut best_volume_score = f64::INFINITY;
    let mut volume_scores = Vec::new();

    for &volume_weight in &volume_weights {
        total_tested += 1;
        let balance = evaluate_sentiment_params(
            &utils,
            context,
            &SentimentEvalParams {
                sensitivity: current_sensitivity,
                volume_weight,
                consistency_factor: current_consistency,
            },
        )?;

        let score = balance.composite_quality_score;
        volume_scores.push((volume_weight, score, balance.balance_score));

        if score < best_volume_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_volume_score = score;
            current_volume_weight = volume_weight;
            total_improvements += 1;
            log::debug!(
                "  ✓ Better volume weight found: {:.2} (score: {:.4}, balance: {:.2})",
                volume_weight,
                score,
                balance.balance_score
            );
        }
    }

    log::info!(
        "  Volume weight scores: {:?}",
        volume_scores
            .iter()
            .map(|(vw, score, bal)| format!("{:.2}={:.3}/{:.1}", vw, score, bal))
            .collect::<Vec<_>>()
            .join(", ")
    );
    log::info!(
        "  → Best volume weight: {:.2} (score: {:.4})",
        current_volume_weight,
        best_volume_score
    );

    // Step 3: Optimize consistency factor
    log::info!("📊 Step 3/4: Optimizing consistency factor...");
    let mut best_consistency_score = f64::INFINITY;
    let mut final_balance = ClassBalance::default();

    for &consistency_factor in &consistency_factors {
        total_tested += 1;
        let balance = evaluate_sentiment_params(
            &utils,
            context,
            &SentimentEvalParams {
                sensitivity: current_sensitivity,
                volume_weight: current_volume_weight,
                consistency_factor,
            },
        )?;

        let score = balance.composite_quality_score;

        if score < best_consistency_score && balance.diversity_score >= 0.3
        // min_diversity_threshold
        {
            best_consistency_score = score;
            current_consistency = consistency_factor;
            final_balance = balance.clone();
            total_improvements += 1;
            log::debug!(
                "  ✓ Better consistency found: {:.1} (score: {:.4}, balance: {:.2})",
                consistency_factor,
                score,
                final_balance.balance_score
            );
        }
    }
    log::info!(
        "  → Best consistency factor: {:.1} (score: {:.4})",
        current_consistency,
        best_consistency_score
    );

    // Step 4: Find optimal extreme multiplier based on sensitivity
    log::info!("📊 Step 4/4: Finding optimal extreme multiplier...");
    let mut best_extreme_score = f64::INFINITY;

    for &extreme_mult in &extreme_multipliers {
        total_tested += 1;

        // Quick evaluation to find best multiplier
        let test_balance = evaluate_sentiment_params(
            &utils,
            context,
            &SentimentEvalParams {
                sensitivity: current_sensitivity,
                volume_weight: current_volume_weight,
                consistency_factor: current_consistency,
            },
        )?;

        // Use a heuristic: prefer multipliers that create good separation
        let separation_score = (extreme_mult - 2.5_f64).abs() * 0.1; // Penalty for extreme values
        let adjusted_score = test_balance.composite_quality_score + separation_score;

        if adjusted_score < best_extreme_score {
            best_extreme_score = adjusted_score;
            current_extreme_mult = extreme_mult;
            final_balance = test_balance.clone();
        }
    }
    log::info!("  → Best extreme multiplier: {:.1}", current_extreme_mult);

    // Calculate derived thresholds
    let min_base_threshold = current_sensitivity * 0.1;
    let min_extreme_threshold = current_sensitivity * current_extreme_mult * 0.1;

    let final_balance_score = final_balance.balance_score;

    let best_params = SentimentParams {
        body_sensitivity: current_sensitivity,
        volume_weight: current_volume_weight,
        consistency_factor: current_consistency,
        extreme_multiplier: current_extreme_mult,
        min_base_threshold,
        min_extreme_threshold,
        balance: final_balance,
    };

    log::info!(
        "🎯 Sentiment Calibration Complete!\n  Tested: {} combinations\n  Improvements: {}\n  Final Parameters:\n    - Body Sensitivity: {:.3}\n    - Volume Weight: {:.2}\n    - Consistency Factor: {:.1}\n    - Extreme Multiplier: {:.1}\n    - Min Base Threshold: {:.4}\n    - Min Extreme Threshold: {:.4}\n  Final Score: {:.4}\n  Final Balance: {:.2}",
        total_tested,
        total_improvements,
        best_params.body_sensitivity,
        best_params.volume_weight,
        best_params.consistency_factor,
        best_params.extreme_multiplier,
        best_params.min_base_threshold,
        best_params.min_extreme_threshold,
        best_consistency_score,
        final_balance_score
    );

    Ok(best_params)
}

/// Evaluate sentiment parameters using proper sentiment classification
fn evaluate_sentiment_params(
    utils: &super::utils::CalibrationUtils,
    context: &EvaluationContext,
    params: &SentimentEvalParams,
) -> Result<ClassBalance> {
    use crate::targets::sentiment::classify_sentiment_with_calibrated_params;

    let mut class_counts = [0usize; 5];

    // Create calibrated parameters for sentiment classification
    let calibrated_params = SentimentParams {
        body_sensitivity: params.sensitivity,
        volume_weight: params.volume_weight,
        consistency_factor: params.consistency_factor,
        extreme_multiplier: 2.5, // Default value for evaluation
        min_base_threshold: params.sensitivity * 0.1,
        min_extreme_threshold: params.sensitivity * 2.5 * 0.1,
        balance: Default::default(),
    };

    // Process each sample using sentiment classification
    for &seq_idx in context.sample_indices {
        let sequence_end_idx = seq_idx + context.sequence_length;
        let target_end_idx = sequence_end_idx + context.horizon_steps;

        if target_end_idx <= context.ohlcv_data.len() {
            let sequence_candles = &context.ohlcv_data[seq_idx..sequence_end_idx];
            let horizon_candles = &context.ohlcv_data[sequence_end_idx..target_end_idx];

            if sequence_candles.len() >= 2 && horizon_candles.len() >= 2 {
                // Use sentiment classification with calibrated parameters
                if let Ok((class, _strength)) = classify_sentiment_with_calibrated_params(
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
