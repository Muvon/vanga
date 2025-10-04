//! Sentiment Calibration Module (Real Candle Psychology)
//!
//! Contains sentiment-specific calibration logic using Bayesian Optimization
//! for real candle psychology features.

use super::core::ParameterCalibrator;
use super::types::*;
use crate::utils::error::Result;

/// Calibrate sentiment parameters using Bayesian Optimization
pub async fn calibrate_sentiment(
    calibrator: &ParameterCalibrator,
    context: &EvaluationContext<'_>,
) -> Result<SentimentParams> {
    log::info!("🔬 Starting Bayesian Optimization for Real Sentiment Analysis");

    // Define parameter bounds (4 feature weights + 2 thresholds)
    let param_bounds = vec![
        (0.5, 2.0),   // body_weight
        (0.1, 1.0),   // size_weight
        (0.1, 0.8),   // wick_weight
        (0.1, 0.8),   // volume_weight
        (0.01, 0.15), // sensitivity
        (1.5, 4.0),   // extreme_multiplier
    ];

    let param_names = vec![
        "body_weight".to_string(),
        "size_weight".to_string(),
        "wick_weight".to_string(),
        "volume_weight".to_string(),
        "sensitivity".to_string(),
        "extreme_multiplier".to_string(),
    ];

    // Define objective function
    let utils = calibrator.get_utils();
    let objective_fn = |params: &[f64]| -> Result<f64> {
        let test_params = SentimentParams {
            body_weight: params[0],
            size_weight: params[1],
            wick_weight: params[2],
            volume_weight: params[3],
            sensitivity: params[4],
            extreme_multiplier: params[5],
            min_base_threshold: params[4] * 0.1,
            min_extreme_threshold: params[4] * params[5] * 0.1,
            balance: Default::default(),
        };

        let balance = evaluate_sentiment_params(&utils, context, &test_params)?;
        Ok(balance.composite_quality_score)
    };

    // Run Bayesian optimization with more samples (6D space)
    let bayesian_config = super::bayesian::BayesianConfig {
        n_initial: 20,      // More initial samples for 6D space
        max_iterations: 60, // More iterations for complex space
        tolerance: 1e-4,
        acquisition: super::bayesian::AcquisitionFunction::ExpectedImprovement,
        gp_length_scale: 0.5,
        gp_noise: 1e-6,
    };

    let best_params = calibrator
        .calibrate_with_bayesian(param_bounds, param_names, objective_fn, bayesian_config)
        .await?;

    // Evaluate final balance
    let final_params = SentimentParams {
        body_weight: best_params[0],
        size_weight: best_params[1],
        wick_weight: best_params[2],
        volume_weight: best_params[3],
        sensitivity: best_params[4],
        extreme_multiplier: best_params[5],
        min_base_threshold: best_params[4] * 0.1,
        min_extreme_threshold: best_params[4] * best_params[5] * 0.1,
        balance: Default::default(),
    };

    let final_balance = evaluate_sentiment_params(&utils, context, &final_params)?;

    log::info!(
        "🎯 Final Sentiment Parameters:\\n  Body Weight: {:.3}\\n  Size Weight: {:.3}\\n  Wick Weight: {:.3}\\n  Volume Weight: {:.3}\\n  Sensitivity: {:.4}\\n  Extreme Multiplier: {:.2}",
        final_params.body_weight,
        final_params.size_weight,
        final_params.wick_weight,
        final_params.volume_weight,
        final_params.sensitivity,
        final_params.extreme_multiplier
    );

    Ok(SentimentParams {
        balance: final_balance,
        ..final_params
    })
}

/// Evaluate sentiment parameters using real candle psychology
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
    utils.calculate_balance(class_counts.as_ref(), total)
}
