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
	prefix: &str,
) -> Result<PriceLevelParams> {
	use super::bayesian::BayesianConfig;

	if calibrator.price_level_definition.uses_anchor_volatility()? {
		return calibrate_anchor_price_levels(calibrator, context, prefix).await;
	}

	log::info!(
		"{} 🔬 Starting Bayesian Optimization for Price Levels calibration",
		prefix
	);

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

	// Objective function: minimize composite_quality_score
	let objective_fn = |params: &[f64]| -> Result<f64> {
		let test_params = PriceLevelEvalParams {
			bandwidth: params[0],
			percentiles: [params[1], params[2]], // Combine low/high
			neutral_band: params[3],
			momentum_factor: params[4],
		};

		let balance = evaluate_price_level_params(&utils, context, &test_params)?;

		// Score thresholds on balance AND sample diversity (temporal spread,
		// feature spread, market-regime coverage) — balance alone lets a
		// boundary land wherever it needs to to equalise counts
		Ok(balance.composite_quality_score)
	};

	// Bayesian optimization configuration
	// Use quality-first Bayesian configuration (default for 4D space)
	let bayesian_config = BayesianConfig::default();

	// Run Bayesian optimization
	let best_params = calibrator
		.calibrate_with_bayesian(param_bounds, param_names, objective_fn, bayesian_config, prefix)
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
		..PriceLevelParams::default()
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

	for &seq_idx in context.sample_indices.iter() {
		let sequence_end_idx = seq_idx + context.sequence_length;
		let target_end_idx = sequence_end_idx + context.horizon_steps;

		if target_end_idx <= context.ohlcv_data.len() {
			let sequence_ohlcv = &context.ohlcv_data[seq_idx..sequence_end_idx];
			let horizon_ohlcv = &context.ohlcv_data[sequence_end_idx..target_end_idx];

			if sequence_ohlcv.len() >= 2 && horizon_ohlcv.len() >= 2 {
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
				if (0..=4).contains(&class) {
					class_counts[class as usize] += 1;
				}
			}
		}
	}

	let total = class_counts.iter().sum::<usize>();

	if total == 0 {
		log::warn!("  WARNING: No valid samples processed!");
	}

	// Use diversity-aware balance calculation
	utils.calculate_balance_with_diversity(
		class_counts.as_ref(),
		total,
		context.ohlcv_data,
		context.sample_indices,
		context.sequence_length,
	)
}

/// Seed the existing Bayesian search with empirical quintile thresholds. The
/// search retains exact balance when distinct data admit it; tied statistics
/// cannot be arbitrarily split across labels.
async fn calibrate_anchor_price_levels(
	calibrator: &ParameterCalibrator,
	context: &EvaluationContext<'_>,
	prefix: &str,
) -> Result<PriceLevelParams> {
	use crate::targets::price_levels::anchor_price_level_statistic_with_history;
	use crate::utils::error::VangaError;

	let regime_window_bars = if calibrator.price_level_definition.uses_regime_volatility() {
		let window_bars = calibrator.regime_window_bars.ok_or_else(|| {
			VangaError::config("Regime-volatility price-level calibration requires features.stats_window_bars")
		})?;
		if window_bars < 2 {
			return Err(VangaError::config(
				"Regime-volatility price-level calibration requires features.stats_window_bars >= 2",
			));
		}
		Some(window_bars)
	} else {
		None
	};
	let mut result = PriceLevelParams {
		definition: calibrator.price_level_definition,
		regime_window_bars,
		..PriceLevelParams::default()
	};
	let mut statistics = Vec::new();
	let mut valid_indices = Vec::new();
	let mut insufficient_history = 0usize;
	for &start in context.sample_indices {
		let end = start
			.checked_add(context.sequence_length)
			.ok_or_else(|| VangaError::data("Price-level sequence index overflow"))?;
		let future_end = end
			.checked_add(context.horizon_steps)
			.ok_or_else(|| VangaError::data("Price-level horizon index overflow"))?;
		if future_end <= context.ohlcv_data.len() {
			if regime_window_bars.is_some_and(|window_bars| end < window_bars) {
				insufficient_history += 1;
				continue;
			}
			statistics.push(anchor_price_level_statistic_with_history(
				&context.ohlcv_data[start..end],
				Some(&context.ohlcv_data[..end]),
				&context.ohlcv_data[end..future_end],
				&result,
			)?);
			valid_indices.push(start);
		}
	}
	if insufficient_history > 0 {
		log::warn!(
			"{} Skipped {} price-level calibration anchors because regime history was incomplete",
			prefix,
			insufficient_history
		);
	}
	if statistics.len() < 5 {
		return Err(VangaError::data(
			"Price-level calibration requires at least five valid samples",
		));
	}
	let mut sorted = statistics.clone();
	sorted.sort_by(f64::total_cmp);
	let seed = distinct_gap_seed(&sorted);
	let utils = calibrator.get_utils();
	let class_counts = |thresholds: [f64; 4]| {
		let mut counts = [0usize; 5];
		for &statistic in &statistics {
			let class = thresholds.partition_point(|&threshold| statistic >= threshold);
			counts[class] += 1;
		}
		counts
	};
	let evaluate = |thresholds: [f64; 4]| -> Result<ClassBalance> {
		let counts = class_counts(thresholds);
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
	let objective =
		|values: &[f64]| -> Result<f64> { Ok(evaluate(sorted_thresholds(values))?.composite_quality_score) };
	// Diversity depends only on the fixed sample set in this objective. The most
	// even split is therefore a proven global minimum, also when the sample count
	// is not a multiple of five.
	let objective_floor = seed
		.map(evaluate)
		.transpose()?
		.filter(ClassBalance::is_most_even_split)
		.map(|balance| balance.composite_quality_score);
	let best = calibrator
		.calibrate_with_bayesian(
			vec![(sorted[0], sorted[sorted.len() - 1]); 4],
			(0..4).map(|i| format!("price_level_threshold_{i}")).collect(),
			objective,
			super::bayesian::BayesianConfig {
				initial_candidate: seed.map(Vec::from),
				objective_floor,
				..Default::default()
			},
			prefix,
		)
		.await?;
	let best_thresholds = sorted_thresholds(&best);
	result.thresholds = Some(best_thresholds);
	validate_anchor_price_level_thresholds(&statistics, &result)?;
	result.balance = evaluate(best_thresholds)?;
	Ok(result)
}

pub(crate) fn distinct_gap_seed(sorted: &[f64]) -> Option<[f64; 4]> {
	let gaps: Vec<(usize, f64)> = sorted
		.windows(2)
		.enumerate()
		.filter_map(|(index, pair)| {
			(pair[0] < pair[1]).then(|| {
				let midpoint = pair[0] * 0.5 + pair[1] * 0.5;
				let threshold = if midpoint > pair[0] && midpoint < pair[1] {
					midpoint
				} else {
					pair[1]
				};
				(index + 1, threshold)
			})
		})
		.collect();
	if gaps.len() < 4 {
		return None;
	}

	let mut seed = [0.0; 4];
	let mut first_gap = 0;
	for (cut, threshold) in seed.iter_mut().enumerate() {
		let last_gap = gaps.len() - (4 - cut);
		let target_count = sorted.len() as f64 * (cut + 1) as f64 / 5.0;
		let best_gap = (first_gap..=last_gap).min_by(|&left, &right| {
			let left_distance = (gaps[left].0 as f64 - target_count).abs();
			let right_distance = (gaps[right].0 as f64 - target_count).abs();
			left_distance.total_cmp(&right_distance)
		})?;
		*threshold = gaps[best_gap].1;
		first_gap = best_gap + 1;
	}
	Some(seed)
}

pub(crate) fn validate_anchor_price_level_thresholds(
	statistics: &[f64],
	params: &PriceLevelParams,
) -> Result<[usize; 5]> {
	let thresholds = crate::targets::price_levels::anchor_price_level_thresholds(params)?;
	let mut counts = [0usize; 5];
	for &statistic in statistics {
		let class = thresholds.partition_point(|&threshold| statistic >= threshold);
		counts[class] += 1;
	}
	if counts.contains(&0) {
		return Err(crate::utils::error::VangaError::data(
			"Final price-level thresholds must produce five non-empty calibration classes",
		));
	}
	Ok(counts)
}
