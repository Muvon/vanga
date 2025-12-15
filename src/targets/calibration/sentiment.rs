//! Sentiment Calibration Module (Volume-Price Divergence)
//!
//! Contains sentiment-specific calibration logic using Bayesian Optimization
//! for volume-price divergence analysis.

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

    // Define 2D parameter space (SIMPLIFIED - like volatility target)
    // Only 2 parameters needed for momentum ratio classification
    let param_bounds = vec![
        (0.01, 0.5), // sensitivity: 0.01-0.5 (moderate range for log-ratio thresholds)
        (1.5, 4.0),  // extreme_multiplier: 1.5-4.0 (narrow to wide extreme zones)
    ];

    let param_names = vec!["sensitivity".to_string(), "extreme_multiplier".to_string()];

    // Define objective function
    let utils = calibrator.get_utils();
    let objective_fn = |params: &[f64]| -> Result<f64> {
        let test_params = SentimentParams {
            sensitivity: params[0],
            extreme_multiplier: params[1],
            balance: Default::default(),
        };

        let balance = evaluate_sentiment_params(&utils, context, &test_params)?;
        Ok(balance.composite_quality_score)
    };

    // Run Bayesian optimization with low-dimensional config (2D space)
    let bayesian_config = super::bayesian::BayesianConfig::default();

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
    let final_params = SentimentParams {
        sensitivity: best_params[0],
        extreme_multiplier: best_params[1],
        balance: Default::default(),
    };

    let final_balance = evaluate_sentiment_params(&utils, context, &final_params)?;

    log::info!(
        "🎯 Final Sentiment Parameters (Divergence):\n  Sensitivity: {:.4}\n  Extreme Multiplier: {:.2}",
        final_params.sensitivity,
        final_params.extreme_multiplier
    );

    Ok(SentimentParams {
        balance: final_balance,
        ..final_params
    })
}

/// Evaluate sentiment parameters using volume-price divergence with REAL diversity metrics
fn evaluate_sentiment_params(
    utils: &super::utils::CalibrationUtils,
    context: &EvaluationContext,
    params: &SentimentParams,
) -> Result<ClassBalance> {
    use crate::targets::sentiment::classify_sentiment_with_calibrated_params;

    let mut class_counts = [0usize; 5];

    // Process each sample
    for &seq_idx in context.sample_indices {
        let sequence_end_idx = seq_idx + context.sequence_length;
        let target_end_idx = sequence_end_idx + context.horizon_steps;

        if target_end_idx <= context.ohlcv_data.len() {
            let sequence_candles = &context.ohlcv_data[seq_idx..sequence_end_idx];
            let horizon_candles = &context.ohlcv_data[sequence_end_idx..target_end_idx];

            if sequence_candles.len() >= 2 && horizon_candles.len() >= 2 {
                if let Ok((class, _strength)) = classify_sentiment_with_calibrated_params(
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
