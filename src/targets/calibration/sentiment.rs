//! Sentiment Calibration Module (Volume-Price Divergence)
//!
//! Contains sentiment-specific calibration logic using Bayesian Optimization
//! for volume-price divergence analysis.

use super::bayesian::BayesianConfig;
use super::core::ParameterCalibrator;
use super::types::*;
use crate::utils::error::Result;

/// Calibrate sentiment parameters using Bayesian Optimization
pub async fn calibrate_sentiment(
    calibrator: &ParameterCalibrator,
    context: &EvaluationContext<'_>,
    prefix: &str,
) -> Result<SentimentParams> {
    log::info!(
        "{} 🔬 Starting Bayesian Optimization for Sentiment (Volume-Price Divergence)",
        prefix
    );

    // Define 5D parameter space with WIDE, ADAPTIVE bounds (matching volume/direction)
    // Extended from 2D to 5D for better balance with fewer samples
    let param_bounds = vec![
        (0.01, 0.5),  // sensitivity: 0.01-0.5 (very sensitive to conservative)
        (1.2, 6.0), // extreme_multiplier: 1.2-6.0 (narrow to very wide extremes) - EXPANDED from 1.5-4.0
        (0.01, 0.45), // percentile_low: 1%-45% (adaptive lower boundary) - NEW
        (0.55, 0.99), // percentile_high: 55%-99% (adaptive upper boundary) - NEW
        (1.0, 30.0), // smoothing_periods: 1-30 (no smoothing to heavy smoothing) - NEW
    ];

    let param_names = vec![
        "sensitivity".to_string(),
        "extreme_multiplier".to_string(),
        "percentile_low".to_string(),
        "percentile_high".to_string(),
        "smoothing_periods".to_string(),
    ];

    let utils = calibrator.get_utils();
    let objective_fn = |params: &[f64]| -> Result<f64> {
        let test_params = SentimentEvalParams {
            sensitivity: params[0],
            extreme_multiplier: params[1],
            percentile_low: params[2],
            percentile_high: params[3],
            smoothing: params[4].round() as usize,
        };

        let balance = evaluate_sentiment_params(&utils, context, &test_params)?;

        // Return score only if diversity is acceptable
        if balance.diversity_score >= 0.3 {
            Ok(balance.composite_quality_score)
        } else {
            // Penalize low diversity
            Ok(balance.composite_quality_score + 10.0)
        }
    };

    // Use quality-first Bayesian configuration (5D space)
    let bayesian_config = BayesianConfig::default();

    let best_params = calibrator
        .calibrate_with_bayesian(
            param_bounds,
            param_names,
            objective_fn,
            bayesian_config,
            prefix,
        )
        .await?;

    // Evaluate final balance
    let final_params = SentimentEvalParams {
        sensitivity: best_params[0],
        extreme_multiplier: best_params[1],
        percentile_low: best_params[2],
        percentile_high: best_params[3],
        smoothing: best_params[4].round() as usize,
    };

    let final_balance = evaluate_sentiment_params(&utils, context, &final_params)?;

    // Calculate derived thresholds for storage
    let min_base_threshold = best_params[0] * 0.1;
    let min_extreme_threshold = best_params[0] * best_params[1] * 0.1;

    let result = SentimentParams {
        sensitivity: best_params[0],
        extreme_multiplier: best_params[1],
        percentile_low: best_params[2],
        percentile_high: best_params[3],
        smoothing_periods: best_params[4].round() as usize,
        balance: final_balance,
    };

    log::info!(
        "🎯 Final Sentiment Parameters (Divergence):\n  Sensitivity: {:.4}\n  Extreme Multiplier: {:.2}\n  Percentiles: [{:.2}, {:.2}]\n  Smoothing: {}\n  Min Base Threshold: {:.4}\n  Min Extreme Threshold: {:.4}",
        result.sensitivity,
        result.extreme_multiplier,
        result.percentile_low,
        result.percentile_high,
        result.smoothing_periods,
        min_base_threshold,
        min_extreme_threshold
    );

    Ok(result)
}

/// Evaluate sentiment parameters using volume-price divergence with REAL diversity metrics
fn evaluate_sentiment_params(
    utils: &super::utils::CalibrationUtils,
    context: &EvaluationContext,
    params: &SentimentEvalParams,
) -> Result<ClassBalance> {
    use crate::targets::sentiment::classify_sentiment_with_evaluation_params;

    let mut class_counts = [0usize; 5];
    let sample_limit = context.sample_indices.len().min(500); // Limit for performance

    // Process each sample
    for &seq_idx in context.sample_indices.iter().take(sample_limit) {
        let sequence_end_idx = seq_idx + context.sequence_length;
        let target_end_idx = sequence_end_idx + context.horizon_steps;

        if target_end_idx <= context.ohlcv_data.len() {
            let sequence_candles = &context.ohlcv_data[seq_idx..sequence_end_idx];
            let horizon_candles = &context.ohlcv_data[sequence_end_idx..target_end_idx];

            if sequence_candles.len() >= 2 && horizon_candles.len() >= 2 {
                if let Ok((class, _strength)) = classify_sentiment_with_evaluation_params(
                    sequence_candles,
                    horizon_candles,
                    params,
                ) {
                    if (0..5).contains(&class) {
                        class_counts[class as usize] += 1;
                    }
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
        context.sequence_length,
    )
}
