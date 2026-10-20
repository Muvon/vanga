//! Comprehensive tests for price level classification to ensure mathematical correctness
//!
//! These tests validate:
//! 1. Percentile-based boundary calculation accuracy
//! 2. VWAP price calculation correctness
//! 3. Bandwidth scaling appropriateness
//! 4. Classification balance across different scenarios
//! 5. Edge case handling

#[cfg(test)]
mod tests {
	use super::super::price_levels::*;
	use crate::data::structures::MarketDataRow;

	/// Helper function to create test market data
	fn create_test_candles(ohlcv_data: Vec<(f64, f64, f64, f64, f64)>) -> Vec<MarketDataRow> {
		ohlcv_data
			.into_iter()
			.map(|(open, high, low, close, volume)| MarketDataRow {
				timestamp: 0,
				open,
				high,
				low,
				close,
				volume,
			})
			.collect()
	}

	/// **ENHANCED**: Test adaptive bandwidth calculation
	#[test]
	fn test_adaptive_bandwidth() {
		// Test case 1: Low volatility sequence should have smaller bandwidth
		let low_vol_sequence = create_test_candles(vec![
			(100.0, 100.1, 99.9, 100.0, 1000.0), // Very tight range
			(100.0, 100.1, 99.9, 100.05, 1000.0),
			(100.05, 100.15, 99.95, 100.1, 1000.0),
			(100.1, 100.2, 100.0, 100.15, 1000.0),
		]);

		// Test case 2: High volatility sequence should have larger bandwidth
		let high_vol_sequence = create_test_candles(vec![
			(100.0, 105.0, 95.0, 102.0, 1000.0), // Wide range
			(102.0, 108.0, 98.0, 104.0, 1000.0),
			(104.0, 110.0, 100.0, 106.0, 1000.0),
			(106.0, 112.0, 102.0, 108.0, 1000.0),
		]);

		let base_bandwidth = 1.0;

		let low_vol_bandwidth = calculate_adaptive_bandwidth(&low_vol_sequence, base_bandwidth, None).unwrap();
		let high_vol_bandwidth = calculate_adaptive_bandwidth(&high_vol_sequence, base_bandwidth, None).unwrap();

		// High volatility should result in larger bandwidth
		assert!(high_vol_bandwidth > low_vol_bandwidth);

		// Both should be reasonable multiples of base bandwidth
		assert!(low_vol_bandwidth >= 0.3 * base_bandwidth); // Minimum bound
		assert!(high_vol_bandwidth <= 3.0 * base_bandwidth); // Maximum bound

		println!("Low volatility adaptive bandwidth: {:.3}", low_vol_bandwidth);
		println!("High volatility adaptive bandwidth: {:.3}", high_vol_bandwidth);
	}
}

use crate::config::training::{
	PriceLevelDefinition, PriceLevelReference, PriceLevelScale, TargetsConfig, TrainingConfig,
};
use crate::data::structures::MarketDataRow;
use crate::targets::calibration::{CalibratedParameters, EvaluationContext, ParameterCalibrator, PriceLevelParams};
use crate::targets::price_levels::*;

fn price_level_candles(closes: &[f64]) -> Vec<MarketDataRow> {
	closes
		.iter()
		.enumerate()
		.map(|(i, &close)| MarketDataRow {
			timestamp: i as i64 * 60_000,
			open: close,
			high: close,
			low: close,
			close,
			volume: 1000.0,
		})
		.collect()
}

fn anchor_definition() -> PriceLevelDefinition {
	PriceLevelDefinition {
		reference: PriceLevelReference::AnchorClose,
		scale: PriceLevelScale::Volatility,
	}
}

#[test]
fn price_level_default_preserves_legacy_labels_strengths_and_meta() {
	let params = PriceLevelParams {
		bandwidth: 1.0,
		percentiles: [0.2, 0.8],
		neutral_band_factor: 0.4,
		momentum_factor: 1.2,
		balance: crate::targets::calibration::ClassBalance {
			composite_quality_score: 0.0,
			..Default::default()
		},
		..Default::default()
	};
	// This tuple is the exact original Bincode struct layout, including an
	// enclosing suffix to catch accidental reads into the next model field.
	let legacy_binary = bincode::serialize(&(
		params.bandwidth,
		params.percentiles,
		params.neutral_band_factor,
		params.momentum_factor,
		&params.balance,
	))
	.unwrap();
	assert_eq!(bincode::serialize(&params).unwrap(), legacy_binary);
	let loaded_binary: PriceLevelParams = bincode::deserialize(&legacy_binary).unwrap();
	assert_eq!(loaded_binary.definition, PriceLevelDefinition::default());
	let mut with_suffix = legacy_binary.clone();
	with_suffix.extend(bincode::serialize(&123456u64).unwrap());
	let (nested, suffix): (PriceLevelParams, u64) = bincode::deserialize(&with_suffix).unwrap();
	assert_eq!(suffix, 123456);
	assert_eq!(bincode::serialize(&nested).unwrap(), legacy_binary);
	let mut legacy_meta = serde_json::to_value(&params).unwrap();
	assert!(legacy_meta.get("reference").is_none());
	assert!(legacy_meta.get("scale").is_none());
	assert!(legacy_meta.get("thresholds").is_none());
	// Older serde records could carry additional fields; retain that tolerance.
	let mut extended_legacy = legacy_meta.clone();
	extended_legacy["obsolete_parameter"] = 1.0.into();
	let extended: PriceLevelParams = serde_json::from_value(extended_legacy).unwrap();
	assert_eq!(serde_json::to_value(&extended).unwrap(), legacy_meta);
	let loaded: PriceLevelParams = serde_json::from_value(legacy_meta.clone()).unwrap();
	assert_eq!(serde_json::to_value(&loaded).unwrap(), legacy_meta);
	legacy_meta["reference"] = "window_weighted".into();
	legacy_meta["scale"] = "window_range".into();
	let explicit: PriceLevelParams = serde_json::from_value(legacy_meta).unwrap();
	let sequence = price_level_candles(&[90.0, 95.0, 100.0, 105.0, 110.0]);
	// Legacy ladder: [80, 99.5, 105.5, 125], bandwidth 15.
	// Unequal future closes exercise the original j^2 weighting.
	for (future, class, strength) in [
		([65.0, 71.25], 0, 10.0 / 15.0),
		([85.0, 91.25], 1, 1.0 - 0.25 / 9.75),
		([97.5, 103.75], 2, 1.0),
		([110.0, 116.25], 3, 1.0 - 0.25 / 9.75),
		([125.0, 131.25], 4, 5.0 / 15.0),
	] {
		let horizon = price_level_candles(&future);
		let actual = classify_price_level_with_calibrated_params(&sequence, &horizon, &loaded).unwrap();
		assert_eq!(actual.0, class);
		assert!((actual.1 - strength).abs() < 1e-13);
		let configured = classify_price_level_with_calibrated_params(&sequence, &horizon, &explicit).unwrap();
		assert_eq!(actual.0, configured.0);
		assert_eq!(actual.1.to_bits(), configured.1.to_bits());
	}
	let flat = price_level_candles(&[100.0, 100.0]);
	assert_eq!(
		classify_price_level_with_calibrated_params(&flat, &flat, &loaded).unwrap(),
		(3, 0.3)
	);
	assert_eq!(
		classify_price_level_with_calibrated_params(&[], &flat, &loaded).unwrap(),
		(2, 0.5)
	);
	let probabilities = [0.1, 0.2, 0.4, 0.2, 0.1];
	let old = reconstruct_price_levels(&probabilities, &sequence, 110.0, &loaded).unwrap();
	let explicit = reconstruct_price_levels(&probabilities, &sequence, 110.0, &explicit).unwrap();
	assert_eq!(old.price_ranges, explicit.price_ranges);
	assert_eq!(old.percentage_ranges, explicit.percentage_ranges);
	for (band, expected) in old.price_ranges.iter().zip([
		[65.0, 80.0],
		[80.0, 99.5],
		[99.5, 105.5],
		[105.5, 125.0],
		[125.0, 140.0],
	]) {
		assert!((band[0] - expected[0]).abs() < 1e-12);
		// Legacy display subtracts 0.0001% of the window-weighted close.
		let upper = if expected[1] < 140.0 {
			expected[1] - (5800.0 / 55.0) * 1e-6
		} else {
			expected[1]
		};
		assert!((band[1] - upper).abs() < 1e-12);
	}
}

#[test]
fn price_level_config_loads_old_and_new_toml_and_roundtrips() {
	let old = "price_level = true\nstop_level = false\ndirection = false\nvolatility = false\nsentiment = false\nvolume = false\n";
	let old_targets: TargetsConfig = toml::from_str(old).unwrap();
	assert_eq!(old_targets.price_level_definition, PriceLevelDefinition::default());
	assert_eq!(toml::to_string(&old_targets).unwrap(), old);
	let experiment = old.replace("price_level = true\n", "")
		+ "\n[price_level]\nreference = \"anchor_close\"\nscale = \"volatility\"\n";
	let targets: TargetsConfig = toml::from_str(&experiment).unwrap();
	assert!(targets.price_level);
	assert_eq!(targets.price_level_definition, anchor_definition());
	let mut training = TrainingConfig::default();
	let mut model = toml::Value::try_from(&training.model).unwrap();
	model
		.as_table_mut()
		.unwrap()
		.insert("targets".into(), toml::from_str(&experiment).unwrap());
	let mut config = toml::map::Map::new();
	config.insert("model".into(), model);
	let file = tempfile::NamedTempFile::new().unwrap();
	std::fs::write(file.path(), toml::to_string(&config).unwrap()).unwrap();
	training = training.with_config_from_file(file.path()).unwrap();
	assert_eq!(training.targets.price_level_definition, anchor_definition());
	let saved = serde_json::to_value(&training).unwrap();
	assert_eq!(saved["targets"]["price_level"]["reference"], "anchor_close");
	let restored: TrainingConfig = serde_json::from_value(saved).unwrap();
	assert_eq!(restored.targets.price_level_definition, anchor_definition());
	let mut disabled = targets.clone();
	disabled.price_level = false;
	let restored: TargetsConfig = serde_json::from_str(&serde_json::to_string(&disabled).unwrap()).unwrap();
	assert!(!restored.price_level);
	assert_eq!(restored.price_level_definition, anchor_definition());
	for invalid in [
		experiment.replace("volatility\"", "window_range\""),
		experiment.replace("anchor_close", "typo"),
	] {
		assert!(toml::from_str::<TargetsConfig>(&invalid).is_err());
	}
	let regime = experiment.replace("volatility\"", "regime_volatility\"");
	let error = toml::from_str::<TargetsConfig>(&regime).unwrap_err().to_string();
	assert!(
		error.contains("price_level does not support scale = regime_volatility"),
		"{error}"
	);
}

#[test]
fn price_level_anchor_volatility_is_causal_and_fail_fast() {
	let sequence = price_level_candles(&[100.0, 100.0 * 0.02_f64.exp(), 100.0]);
	let (anchor, vol) = price_level_anchor_volatility(&sequence).unwrap();
	assert_eq!(anchor, 100.0);
	assert!((vol - 0.02).abs() < 1e-15);
	let future = price_level_candles(&[101.0, 103.0]);
	let statistic = anchor_price_level_statistic(&sequence, &future).unwrap();
	assert!((statistic - ((101.0 + 4.0 * 103.0) / 5.0 / 100.0 - 1.0) / vol).abs() < 1e-13);
	assert!(price_level_anchor_volatility(&price_level_candles(&[100.0, 100.0])).is_err());
	let tiny = price_level_candles(&[100.0, 100.0 * 1e-7_f64.exp(), 100.0]);
	let tiny_volatility = price_level_anchor_volatility(&tiny).unwrap().1;
	assert!((tiny_volatility - 1e-7).abs() < 1e-15);
	assert!(tiny_volatility < 1e-6);
	for closes in [
		vec![],
		vec![100.0],
		vec![0.0, 100.0],
		vec![100.0, f64::NAN],
		vec![f64::MIN_POSITIVE, f64::MAX],
	] {
		assert!(price_level_anchor_volatility(&price_level_candles(&closes)).is_err());
	}
	let mut params = PriceLevelParams {
		definition: anchor_definition(),
		..Default::default()
	};
	assert!(classify_price_level_with_calibrated_params(&sequence, &future, &params).is_err());
	assert!(reconstruct_price_levels(&[0.2; 5], &sequence, anchor, &params).is_err());
	for thresholds in [[-1.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 1.0, f64::INFINITY]] {
		params.thresholds = Some(thresholds);
		assert!(classify_price_level_with_calibrated_params(&sequence, &future, &params).is_err());
	}
}

#[test]
fn anchor_price_endpoints_are_lower_inclusive_for_full_classifier() {
	let sequence = price_level_candles(&[100.0, 101.0, 99.0, 102.0, 98.0, 100.0]);
	let params = PriceLevelParams {
		definition: anchor_definition(),
		thresholds: Some([-2.0, -1.0, 1.0, 2.0]),
		..Default::default()
	};
	let boundaries = anchor_price_level_boundaries(&sequence, &params).unwrap();
	assert_eq!(boundaries.boundaries[3], 105.21445018549522);

	for (index, boundary) in boundaries.boundaries.into_iter().enumerate() {
		let below = f64::from_bits(boundary.to_bits() - 1);
		let above = f64::from_bits(boundary.to_bits() + 1);
		for (price, expected_class) in [
			(below, index as i32),
			(boundary, index as i32 + 1),
			(above, index as i32 + 1),
		] {
			let future = price_level_candles(&[price]);
			let (class, _) = classify_price_level_with_calibrated_params(&sequence, &future, &params).unwrap();
			assert_eq!(class, expected_class, "boundary {index} at {price:?}");
		}
	}
}

#[tokio::test]
async fn price_level_anchor_balances_and_reconstructs_bands() {
	use crate::output::formatter::{infer_stop_is_bullish, OutputFormatter};
	use crate::targets::{MultiTargetConfig, TargetGenerator};
	use polars::prelude::*;

	let input_closes = [100.0, 101.0, 99.0, 102.0, 98.0, 100.0];
	let input = price_level_candles(&input_closes);
	let (anchor, volatility) = price_level_anchor_volatility(&input).unwrap();
	let mut data = Vec::new();
	let mut indices = Vec::new();
	for i in 0..100 {
		indices.push(data.len());
		data.extend(input.clone());
		let statistic = (i as f64 - 49.5) / 10.0;
		let price = anchor * (1.0 + statistic * volatility);
		data.extend(price_level_candles(&[price * 0.96, price * 1.01]));
	}
	for (i, bar) in data.iter_mut().enumerate() {
		bar.timestamp = i as i64 * 60_000;
	}
	let targets = TargetsConfig {
		price_level_definition: anchor_definition(),
		stop_level: false,
		direction: false,
		volatility: false,
		sentiment: false,
		volume: false,
		..Default::default()
	};
	let calibrator = ParameterCalibrator::from_config_with_seed(&targets, Some(42));
	let params = calibrator
		.calibrate_price_levels(
			&EvaluationContext {
				ohlcv_data: &data,
				sample_indices: &indices,
				sequence_length: input.len(),
				horizon_steps: 2,
			},
			"price_level_test",
		)
		.await
		.unwrap();
	assert_eq!(params.balance.class_percentages, [20.0; 5]);
	let mut meta = CalibratedParameters::default();
	meta.metadata.overall_balance_score = params.balance.composite_quality_score;
	meta.price_levels.insert("2m".into(), params);
	let serialized = serde_json::to_value(&meta).unwrap();
	assert_eq!(serialized["price_levels"]["2m"]["reference"], "anchor_close");
	assert_eq!(
		serialized["price_levels"]["2m"]["thresholds"].as_array().unwrap().len(),
		4
	);
	let meta: CalibratedParameters = serde_json::from_value(serialized).unwrap();
	let binary = bincode::serialize(&(meta.clone(), 123456u64)).unwrap();
	let (binary_meta, suffix): (CalibratedParameters, u64) = bincode::deserialize(&binary).unwrap();
	assert_eq!(suffix, 123456);
	assert_eq!(
		serde_json::to_value(&binary_meta).unwrap(),
		serde_json::to_value(&meta).unwrap()
	);
	let params = &meta.price_levels["2m"];
	let df = df!(
		"timestamp" => data.iter().map(|bar| bar.timestamp).collect::<Vec<_>>(),
		"open" => data.iter().map(|bar| bar.open).collect::<Vec<_>>(),
		"high" => data.iter().map(|bar| bar.high).collect::<Vec<_>>(),
		"low" => data.iter().map(|bar| bar.low).collect::<Vec<_>>(),
		"close" => data.iter().map(|bar| bar.close).collect::<Vec<_>>(),
		"volume" => data.iter().map(|bar| bar.volume).collect::<Vec<_>>()
	)
	.unwrap();
	let training = TrainingConfig {
		targets,
		horizons: vec!["2m".into()],
		..Default::default()
	};
	let generator = TargetGenerator::new(MultiTargetConfig::from_training_config(&training));
	let generated = generator
		.generate_all_targets_with_calibrated_params(&df, &indices, input.len(), Some(&meta))
		.await
		.unwrap();
	let formatter = OutputFormatter::new(Default::default())
		.with_sequence_ohlcv(input.clone())
		.with_calibrated_parameters(meta.clone());
	let mut counts = [0; 5];
	for (i, &start) in indices.iter().enumerate() {
		let future = &data[start + input.len()..start + input.len() + 2];
		let (class, _) = classify_price_level_with_calibrated_params(&input, future, params).unwrap();
		assert_eq!(generated.price_levels["2m"][i], class);
		counts[class as usize] += 1;
		let mut probabilities = [0.0; 5];
		probabilities[class as usize] = 1.0;
		let prediction = formatter
			.create_price_level_prediction(&probabilities, anchor, "2m", None, None)
			.unwrap();
		let name = ["strong_down", "moderate_down", "neutral", "moderate_up", "strong_up"][class as usize];
		let band = prediction.bins[name].price;
		let realised = get_horizon_exponential_weighted_close(future).unwrap();
		assert!(
			band[0] <= realised && realised < band[1],
			"{class}: {band:?} excludes {realised}"
		);
		let thresholds = params.thresholds.unwrap();
		if class > 0 {
			assert_eq!(band[0], anchor * (1.0 + thresholds[class as usize - 1] * volatility));
		}
		if class < 4 {
			assert_eq!(band[1], anchor * (1.0 + thresholds[class as usize] * volatility));
		}
		if class == 0 {
			assert_eq!(infer_stop_is_bullish(&prediction, anchor), Some(false));
		}
		if class == 4 {
			assert_eq!(infer_stop_is_bullish(&prediction, anchor), Some(true));
		}
	}
	assert_eq!(counts, [20; 5]);
}

#[test]
fn price_level_anchor_calibration_handles_tied_seed_quintiles() {
	use crate::targets::calibration::price_levels::{distinct_gap_seed, validate_anchor_price_level_thresholds};

	let mut statistics: Vec<f64> = [-2.0, -1.0]
		.into_iter()
		.chain(std::iter::repeat_n(0.0, 16))
		.chain([1.0, 2.0])
		.collect();
	statistics.sort_by(f64::total_cmp);
	let thresholds = distinct_gap_seed(&statistics).unwrap();
	assert_eq!(thresholds, [-1.5, -0.5, 0.5, 1.5]);
	let params = PriceLevelParams {
		definition: anchor_definition(),
		thresholds: Some(thresholds),
		..Default::default()
	};
	assert!(thresholds.iter().all(|threshold| threshold.is_finite()));
	assert!(thresholds.windows(2).all(|pair| pair[0] < pair[1]));
	assert_eq!(
		validate_anchor_price_level_thresholds(&statistics, &params).unwrap(),
		[1, 1, 16, 1, 1]
	);

	let duplicate_params = PriceLevelParams {
		thresholds: Some([0.0; 4]),
		..params
	};
	assert!(validate_anchor_price_level_thresholds(&statistics, &duplicate_params).is_err());
}

#[test]
fn price_level_experimental_binary_preserves_thresholds_and_record_alignment() {
	let params = PriceLevelParams {
		definition: anchor_definition(),
		thresholds: Some([-2.0, -1.0, 1.0, 2.0]),
		..Default::default()
	};
	let binary = bincode::serialize(&(params, 123456u64)).unwrap();
	let (loaded, suffix): (PriceLevelParams, u64) = bincode::deserialize(&binary).unwrap();
	assert_eq!(suffix, 123456);
	assert_eq!(loaded.definition, anchor_definition());
	assert_eq!(loaded.thresholds, Some([-2.0, -1.0, 1.0, 2.0]));
	assert_eq!(bincode::serialize(&(loaded.clone(), suffix)).unwrap(), binary);
	let sequence = price_level_candles(&[100.0, 101.0, 99.0, 102.0, 98.0, 100.0]);
	let boundaries = crate::targets::price_levels::anchor_price_level_boundaries(&sequence, &loaded).unwrap();
	assert!(boundaries.boundaries.windows(2).all(|pair| pair[0] < pair[1]));
	let mut unsupported = binary;
	unsupported[8] = 255; // Version immediately follows the f64 sentinel.
	assert!(bincode::deserialize::<(PriceLevelParams, u64)>(&unsupported).is_err());
}
