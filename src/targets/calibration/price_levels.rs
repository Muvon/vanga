//! Price Levels Calibration Module
//!
//! Contains price level-specific calibration logic including parameter optimization,
//! evaluation functions, and VWAP-weighted analysis.

use super::core::ParameterCalibrator;
use super::types::*;
use crate::utils::error::Result;

/// Calibrate price level parameters using Bayesian optimization
pub async fn calibrate_price_levels(
    calibrator: &ParameterCalibrator,
    context: &EvaluationContext<'_>,
) -> Result<PriceLevelParams> {
    use super::bayesian::BayesianConfig;

    log::info!("🔬 Starting Bayesian Optimization for Price Levels calibration");

    let utils = calibrator.get_utils();

    // Define 5D parameter space with WIDE, ADAPTIVE bounds for all market conditions
    let param_bounds = vec![
        (0.1, 3.0),   // bandwidth: 0.1-3.0 (narrow to very wide price ranges)
        (0.01, 0.45), // percentile_low: 1%-45% (adaptive lower boundary)
        (0.55, 0.99), // percentile_high: 55%-99% (adaptive upper boundary)
        (0.05, 0.9),  // neutral_band_factor: 5%-90% (narrow to wide neutral zone)
        (0.8, 6.0),   // momentum_factor: 0.8-6.0 (conservative to aggressive momentum)
    ];

    let param_names = vec![
        "bandwidth".to_string(),
        "percentile_low".to_string(),
        "percentile_high".to_string(),
        "neutral_band_factor".to_string(),
        "momentum_factor".to_string(),
    ];

    // Objective function: minimize balance_score (not composite_quality_score)
    let objective_fn = |params: &[f64]| -> Result<f64> {
        let test_params = PriceLevelEvalParams {
            bandwidth: params[0],
            percentiles: [params[1], params[2]], // Combine low/high
            neutral_band: params[3],
            momentum_factor: params[4],
        };

        let balance = evaluate_price_level_params(&utils, context, &test_params)?;

        // Price levels use balance_score, not composite_quality_score
        Ok(balance.balance_score)
    };

    // Bayesian optimization configuration
    // Use quality-first Bayesian configuration (default for 4D space)
    let bayesian_config = BayesianConfig::default();

    // Run Bayesian optimization
    let best_params = calibrator
        .calibrate_with_bayesian(param_bounds, param_names, objective_fn, bayesian_config)
        .await?;

    // Evaluate final parameters to get balance
    let final_eval_params = PriceLevelEvalParams {
        bandwidth: best_params[0],
        percentiles: [best_params[1], best_params[2]],
        neutral_band: best_params[3],
        momentum_factor: best_params[4],
    };

    let final_balance = evaluate_price_level_params(&utils, context, &final_eval_params)?;

    let result = PriceLevelParams {
        bandwidth: best_params[0],
        percentiles: [best_params[1], best_params[2]],
        neutral_band_factor: best_params[3],
        momentum_factor: best_params[4],
        balance: final_balance,
    };

    log::info!(
        "🎯 Price Level Calibration Complete!\n  Final Parameters:\n    - Bandwidth: {:.2}\n    - Percentiles: [{:.2}, {:.2}]\n    - Neutral Band: {:.2}\n    - Momentum Factor: {:.2}\n  Final Score: {:.4}",
        result.bandwidth,
        result.percentiles[0],
        result.percentiles[1],
        result.neutral_band_factor,
        result.momentum_factor,
        result.balance.balance_score
    );

    Ok(result)
}

/// Evaluate price level parameters using EXACT calibrated parameters (NO adaptive overrides)
fn evaluate_price_level_params(
    utils: &super::utils::CalibrationUtils,
    context: &EvaluationContext,
    params: &PriceLevelEvalParams,
) -> Result<ClassBalance> {
    use crate::targets::get_horizon_exponential_weighted_close;
    use crate::targets::sequence_reconstruction::{SequenceAnalyzer, SequenceReconstructionConfig};

    let mut class_counts = [0usize; 5];

    let samples_to_test = context.sample_indices.len();
    log::debug!("Testing price level params on {} samples (bandwidth={:.2}, percentiles=[{:.2},{:.2}], neutral={:.2}, momentum={:.2})",
        samples_to_test, params.bandwidth, params.percentiles[0], params.percentiles[1],
        params.neutral_band, params.momentum_factor);

    let mut samples_processed = 0;

    for &seq_idx in context.sample_indices.iter() {
        let sequence_end_idx = seq_idx + context.sequence_length;
        let target_end_idx = sequence_end_idx + context.horizon_steps;

        if target_end_idx <= context.ohlcv_data.len() {
            let sequence_ohlcv = &context.ohlcv_data[seq_idx..sequence_end_idx];
            let horizon_ohlcv = &context.ohlcv_data[sequence_end_idx..target_end_idx];

            if sequence_ohlcv.len() >= 2 && horizon_ohlcv.len() >= 2 {
                samples_processed += 1;

                // Calculate target exponentially-weighted close
                let target_weighted_price = get_horizon_exponential_weighted_close(horizon_ohlcv)?;

                // FIXED: Use EXACT calibrated parameters - NO adaptive overrides!
                let exact_percentiles = params.percentiles; // Use calibrated percentiles exactly
                let exact_bandwidth = params.bandwidth; // Use calibrated bandwidth exactly

                // Use sequence reconstruction with EXACT calibrated parameters
                let reconstruction_config = SequenceReconstructionConfig {
                    percentiles: exact_percentiles,
                    bandwidth_size: exact_bandwidth,
                    neutral_band_factor: params.neutral_band,
                };
                let analyzer = SequenceAnalyzer::new(reconstruction_config);
                let boundaries = analyzer.calculate_boundaries(sequence_ohlcv)?;

                // Handle edge case: flat sequence
                if boundaries.bandwidth == 0.0 {
                    let class = if target_weighted_price >= boundaries.sequence_min {
                        3
                    } else {
                        2
                    };
                    class_counts[class] += 1;
                    continue;
                }

                // Classify using centralized logic
                let class = boundaries.classify_price(target_weighted_price);
                if (0..5).contains(&class) {
                    class_counts[class as usize] += 1;
                }
            }
        }
    }

    let total = class_counts.iter().sum::<usize>();

    log::debug!(
        "  Processed {}/{} samples, distribution: {:?}",
        samples_processed,
        samples_to_test,
        class_counts
    );

    if total == 0 {
        log::warn!("  WARNING: No valid samples processed!");
    }

    // Use diversity-aware balance calculation
    utils.calculate_balance_with_diversity(class_counts.as_ref(), total, context.ohlcv_data, context.sample_indices)
}
