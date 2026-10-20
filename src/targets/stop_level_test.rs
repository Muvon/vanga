use crate::data::structures::MarketDataRow;
use crate::targets::calibration::StopLevelParams;
use crate::targets::stop_levels::{classify_stop_level_with_calibrated_params, reconstruct_stop_levels};

#[test]
fn test_adverse_classification_minimal_risk() {
	// Sequence: 100-110 with lows around 99-109
	let sequence = create_test_sequence(100.0, 110.0, 10);

	// Horizon: bullish move (105 → 108), minimal adverse (lows stay within sequence range)
	// The lows (104.9, 105.8, 106.9) are all within the sequence's low range
	let horizon = vec![
		create_candle(105.0, 104.9, 106.0, 106.0),
		create_candle(106.0, 105.8, 107.0, 107.0),
		create_candle(107.0, 106.9, 108.0, 108.0),
	];

	let params = StopLevelParams {
		bandwidth: 1.0,
		percentiles: [0.1, 0.9],
		neutral_band_factor: 0.4,
		momentum_factor: 1.0,
		balance: Default::default(),
		..Default::default()
	};

	let (class, strength) =
		classify_stop_level_with_calibrated_params(&sequence, &horizon, &params).expect("Classification failed");

	// With boundary-based classification, the adverse (lowest low = 104.9)
	// is compared against sequence adverse boundaries
	// Class 2 (moderate) or higher is acceptable for this scenario
	assert!(
		class >= 2,
		"Expected moderate or lower risk classification, got class {}",
		class
	);
	assert!(strength > 0.1, "Strength should be positive");
}

#[test]
fn test_adverse_classification_high_risk() {
	// Sequence: 100-110
	let sequence = create_test_sequence(100.0, 110.0, 10);

	// Horizon: bullish move (105 → 110), but large adverse (drops to 90)
	let horizon = vec![
		create_candle(105.0, 90.0, 106.0, 100.0),
		create_candle(100.0, 95.0, 105.0, 105.0),
		create_candle(105.0, 100.0, 110.0, 110.0),
	];

	let params = StopLevelParams {
		bandwidth: 1.0,
		percentiles: [0.1, 0.9],
		neutral_band_factor: 0.4,
		momentum_factor: 1.0,
		balance: Default::default(),
		..Default::default()
	};

	let (class, _strength) =
		classify_stop_level_with_calibrated_params(&sequence, &horizon, &params).expect("Classification failed");

	// Large adverse movement (reference ~105, worst low 90)
	assert!(
		class <= 1,
		"Expected high or extreme risk classification, got class {}",
		class
	);
}

#[test]
fn test_adverse_direction_awareness() {
	// Sequence: 100-110
	let sequence = create_test_sequence(100.0, 110.0, 10);

	// Bullish horizon: adverse = lows
	let horizon_bullish = vec![
		create_candle(105.0, 95.0, 110.0, 110.0), // Low 95 is adverse
		create_candle(110.0, 105.0, 115.0, 115.0),
	];

	// Bearish horizon: adverse = highs
	let horizon_bearish = vec![
		create_candle(105.0, 95.0, 115.0, 95.0), // High 115 is adverse
		create_candle(95.0, 90.0, 100.0, 90.0),
	];

	let params = StopLevelParams {
		bandwidth: 1.0,
		percentiles: [0.1, 0.9],
		neutral_band_factor: 0.4,
		momentum_factor: 1.0,
		balance: Default::default(),
		..Default::default()
	};

	let (class_bull, _) = classify_stop_level_with_calibrated_params(&sequence, &horizon_bullish, &params)
		.expect("Classification failed");
	let (class_bear, _) = classify_stop_level_with_calibrated_params(&sequence, &horizon_bearish, &params)
		.expect("Classification failed");

	// Both should detect high risk (different adverse directions)
	assert!(
		class_bull <= 2,
		"Bullish adverse should be high risk, got class {}",
		class_bull
	);
	assert!(
		class_bear <= 2,
		"Bearish adverse should be high risk, got class {}",
		class_bear
	);
}

#[test]
fn test_adverse_entry_proximity_weighting() {
	// Test that early adverse prices (closest to entry) are weighted more heavily
	let sequence = create_test_sequence(100.0, 110.0, 10);

	// Reference (exp-weighted sequence close) is ~106.86, boundaries [92.0, 102.4, 105.6, 116.0].
	// Early dip to 104 (weight ~0.94) vs deeper late dip to 102 (weight 0.5):
	// entry-proximity weighting picks the early 104 (class 2); recency weighting
	// would pick the late 102 (class 1).
	let horizon_early_and_late = vec![
		create_candle(105.0, 104.0, 108.0, 108.0), // Early adverse (weighted more)
		create_candle(108.0, 106.0, 109.0, 109.0),
		create_candle(109.0, 102.0, 110.0, 110.0), // Deeper late adverse (weighted less)
	];

	// Only the late dip to 102: always classified from 102 (class 1)
	let horizon_late_only = vec![
		create_candle(107.0, 106.0, 108.0, 108.0),
		create_candle(108.0, 106.0, 109.0, 109.0),
		create_candle(109.0, 102.0, 110.0, 110.0),
	];

	let params = StopLevelParams {
		bandwidth: 1.0,
		percentiles: [0.1, 0.9],
		neutral_band_factor: 0.4,
		momentum_factor: 1.0,
		balance: Default::default(),
		..Default::default()
	};

	let (class_early, _) = classify_stop_level_with_calibrated_params(&sequence, &horizon_early_and_late, &params)
		.expect("Classification failed");
	let (class_late, _) = classify_stop_level_with_calibrated_params(&sequence, &horizon_late_only, &params)
		.expect("Classification failed");

	// The moderate early dip must outweigh the deeper late dip
	assert!(
		class_early > class_late,
		"Early adverse should outweigh deeper late adverse (got {} vs {})",
		class_early,
		class_late
	);
}

#[test]
fn test_reconstruction_structure() {
	let sequence = create_test_sequence(100.0, 110.0, 10);
	let probabilities = vec![0.1, 0.15, 0.3, 0.25, 0.2];

	let params = StopLevelParams {
		bandwidth: 1.0,
		percentiles: [0.1, 0.9],
		neutral_band_factor: 0.4,
		momentum_factor: 1.0,
		balance: Default::default(),
		..Default::default()
	};

	let reconstruction =
		reconstruct_stop_levels(&probabilities, &sequence, 105.0, &params, None).expect("Reconstruction failed");

	// Verify structure
	assert_eq!(reconstruction.probabilities.len(), 5);
	assert_eq!(reconstruction.adverse_price_ranges.len(), 5);
	assert!(reconstruction.confidence > 0.0 && reconstruction.confidence <= 1.0);
	assert!(reconstruction.reference_price > 0.0);
	// No direction probs → defaults to bullish (legacy behavior).
	assert!(reconstruction.is_bullish);
}

#[test]
fn test_reconstruction_class_ordering() {
	let sequence = create_test_sequence(100.0, 110.0, 10);
	let probabilities = vec![0.05, 0.1, 0.2, 0.3, 0.35]; // Class 4 most likely

	let params = StopLevelParams {
		bandwidth: 1.0,
		percentiles: [0.1, 0.9],
		neutral_band_factor: 0.4,
		momentum_factor: 1.0,
		balance: Default::default(),
		..Default::default()
	};

	let reconstruction =
		reconstruct_stop_levels(&probabilities, &sequence, 105.0, &params, None).expect("Reconstruction failed");

	// Most likely class should be 4 (highest probability)
	assert_eq!(reconstruction.most_likely_class, 4);

	// For bullish: Class 0 has lowest prices (deepest dip), Class 4 has highest (shallowest)
	// So adverse_price_ranges[0] should have lower values than adverse_price_ranges[4]
	assert!(
		reconstruction.adverse_price_ranges[0][1] < reconstruction.adverse_price_ranges[4][0],
		"Class 0 (extreme) should have lower price range than Class 4 (minimal)"
	);
}

#[test]
fn test_reconstruction_direction_awareness() {
	let sequence = create_test_sequence(100.0, 110.0, 10);
	let probabilities = vec![0.2, 0.2, 0.2, 0.2, 0.2];

	let params = StopLevelParams {
		bandwidth: 1.0,
		percentiles: [0.1, 0.9],
		neutral_band_factor: 0.4,
		momentum_factor: 1.0,
		balance: Default::default(),
		..Default::default()
	};

	let bull = reconstruct_stop_levels(&probabilities, &sequence, 105.0, &params, Some(true))
		.expect("Bullish reconstruction failed");
	assert!(bull.is_bullish);

	let bear = reconstruct_stop_levels(&probabilities, &sequence, 105.0, &params, Some(false))
		.expect("Bearish reconstruction failed");
	assert!(!bear.is_bullish);

	// Bullish class 0 is the deepest dip — should be BELOW the bearish class 0 (a high bounce).
	let bull_class0_mid = (bull.adverse_price_ranges[0][0] + bull.adverse_price_ranges[0][1]) / 2.0;
	let bear_class0_mid = (bear.adverse_price_ranges[0][0] + bear.adverse_price_ranges[0][1]) / 2.0;
	assert!(
		bull_class0_mid < bear_class0_mid,
		"Bullish extreme (deep dip) midpoint {} should be below bearish extreme (high bounce) midpoint {}",
		bull_class0_mid,
		bear_class0_mid
	);

	// Every class range must be non-degenerate (width > 0).
	for (i, [lo, hi]) in bull.adverse_price_ranges.iter().enumerate() {
		assert!(hi > lo, "Bullish class {} range [{}, {}] is non-positive", i, lo, hi);
	}
	for (i, [lo, hi]) in bear.adverse_price_ranges.iter().enumerate() {
		assert!(hi > lo, "Bearish class {} range [{}, {}] is non-positive", i, lo, hi);
	}
}

// Helper functions
fn create_test_sequence(min: f64, max: f64, count: usize) -> Vec<MarketDataRow> {
	let step = (max - min) / (count as f64);
	(0..count)
		.map(|i| {
			let price = min + (i as f64) * step;
			create_candle(price, price - 1.0, price + 1.0, price)
		})
		.collect()
}

fn create_candle(open: f64, low: f64, high: f64, close: f64) -> MarketDataRow {
	MarketDataRow {
		timestamp: 0,
		open,
		high,
		low,
		close,
		volume: 1000.0,
	}
}

#[test]
fn stop_level_legacy_fixture_bytes() {
	let params = StopLevelParams {
		bandwidth: 1.0,
		percentiles: [0.1, 0.9],
		neutral_band_factor: 0.4,
		momentum_factor: 1.0,
		balance: crate::targets::calibration::ClassBalance {
			composite_quality_score: 0.0,
			..Default::default()
		},
		..Default::default()
	};
	let sequence = create_test_sequence(100.0, 110.0, 10);
	let mut labels = Vec::new();
	let mut ranges = Vec::new();
	for bullish in [true, false] {
		for price in [80.0, 95.0, 102.0, 104.0, 108.0, 115.0, 130.0] {
			let horizon = if bullish {
				vec![create_candle(109.0, price, 140.0, 135.0)]
			} else {
				vec![create_candle(109.0, 70.0, price, 75.0)]
			};
			let (class, strength) = classify_stop_level_with_calibrated_params(&sequence, &horizon, &params).unwrap();
			labels.push((class, strength.to_bits()));
		}
		let result =
			reconstruct_stop_levels(&[0.1, 0.15, 0.3, 0.25, 0.2], &sequence, 109.0, &params, Some(bullish)).unwrap();
		ranges.extend(result.adverse_price_ranges.into_iter().flatten().map(f64::to_bits));
		ranges.push(result.expected_adverse.to_bits());
		ranges.push(result.reference_price.to_bits());
	}
	// Captured from the unmodified legacy implementation.
	assert_eq!(
		labels,
		vec![
			(0, 4602678819172646912),
			(1, 4603371680653780832),
			(1, 4591870180066957722),
			(2, 4607182418800017408),
			(3, 4597878278910504796),
			(3, 4597878278910504796),
			(3, 4597878278910504796),
			(2, 4602035447797308190),
			(2, 4602035447797308190),
			(2, 4602035447797308190),
			(2, 4602035447797308190),
			(1, 4591870180066957722),
			(1, 4603371680653780832),
			(0, 4602678819172646912)
		]
	);
	assert_eq!(
		ranges,
		vec![
			4635611391447793664,
			4636174341401214976,
			4636174341401214976,
			4636906176340662682,
			4636906176340662682,
			4637131356322031206,
			4637131356322031206,
			4637863191261478912,
			4637863191261478912,
			4638426141214900224,
			4637179207068072018,
			4637219819886140270,
			4638003928749834240,
			4638566878703255552,
			4637272093810386534,
			4638003928749834240,
			4637046913829018010,
			4637272093810386534,
			4636315078889570304,
			4637046913829018010,
			4635752128936148992,
			4636315078889570304,
			4636999063082977198,
			4637219819886140270
		]
	);
	let legacy_binary = bincode::serialize(&(
		params.bandwidth,
		params.percentiles,
		params.neutral_band_factor,
		params.momentum_factor,
		&params.balance,
		123456u64,
	))
	.unwrap();
	assert_eq!(bincode::serialize(&(params.clone(), 123456u64)).unwrap(), legacy_binary);
	let (loaded, suffix): (StopLevelParams, u64) = bincode::deserialize(&legacy_binary).unwrap();
	assert_eq!(suffix, 123456);
	assert_eq!(bincode::serialize(&(loaded, suffix)).unwrap(), legacy_binary);
	let legacy_json = serde_json::json!({
		"bandwidth": params.bandwidth, "percentiles": params.percentiles,
		"neutral_band_factor": params.neutral_band_factor,
		"momentum_factor": params.momentum_factor, "balance": params.balance,
	});
	assert_eq!(serde_json::to_value(&params).unwrap(), legacy_json);
	let loaded: StopLevelParams = serde_json::from_value(legacy_json).unwrap();
	assert_eq!(loaded.definition, Default::default());
	assert_eq!(loaded.thresholds, None);
}

fn anchor_stop_definition(
	side: crate::config::training::StopLevelSide,
) -> crate::config::training::StopLevelDefinition {
	crate::config::training::StopLevelDefinition {
		reference: crate::config::training::PriceLevelReference::AnchorClose,
		scale: crate::config::training::PriceLevelScale::Volatility,
		side,
	}
}

fn regime_stop_definition(
	side: crate::config::training::StopLevelSide,
) -> crate::config::training::StopLevelDefinition {
	crate::config::training::StopLevelDefinition {
		reference: crate::config::training::PriceLevelReference::AnchorClose,
		scale: crate::config::training::PriceLevelScale::RegimeVolatility,
		side,
	}
}

#[test]
fn stop_level_window_volatility_binary_layout_is_unchanged() {
	let params = StopLevelParams {
		definition: anchor_stop_definition(crate::config::training::StopLevelSide::Either),
		thresholds: Some([1.0, 2.0, 3.0, 4.0]),
		bandwidth: 1.0,
		percentiles: [0.1, 0.9],
		neutral_band_factor: 0.4,
		momentum_factor: 1.0,
		balance: crate::targets::calibration::ClassBalance {
			composite_quality_score: 0.0,
			..Default::default()
		},
		..Default::default()
	};
	let expected = bincode::serialize(&(
		f64::NEG_INFINITY,
		1u8,
		(
			params.definition.reference,
			params.definition.scale,
			params.definition.side,
		),
		params.thresholds.unwrap(),
		params.bandwidth,
		params.percentiles,
		params.neutral_band_factor,
		params.momentum_factor,
		&params.balance,
	))
	.unwrap();
	assert_eq!(bincode::serialize(&params).unwrap(), expected);
}

fn anchor_stop_sequence() -> Vec<MarketDataRow> {
	[100.0, 101.0, 99.0, 102.0, 98.0, 100.0]
		.into_iter()
		.map(|price| create_candle(price, price - 1.0, price + 1.0, price))
		.collect()
}

#[test]
fn stop_level_config_defaults_and_roundtrips() {
	use crate::config::training::{StopLevelSide, TargetsConfig, TrainingConfig};
	let old = "price_level = true\nstop_level = true\ndirection = false\nvolatility = false\nsentiment = false\nvolume = false\n";
	let loaded: TargetsConfig = toml::from_str(old).unwrap();
	assert_eq!(loaded.stop_level_definition, Default::default());
	assert_eq!(toml::to_string(&loaded).unwrap(), old);
	for side in [StopLevelSide::Realised, StopLevelSide::Either] {
		let experiment = old.replace("stop_level = true\n", "")
			+ "\n[stop_level]\nreference = \"anchor_close\"\nscale = \"volatility\"\n"
			+ if side == StopLevelSide::Either {
				"side = \"either\"\n"
			} else {
				""
			};
		let targets: TargetsConfig = toml::from_str(&experiment).unwrap();
		assert!(targets.stop_level);
		assert_eq!(targets.stop_level_definition, anchor_stop_definition(side));
		targets.validate().unwrap();
		let mut model = toml::Value::try_from(TrainingConfig::default().model).unwrap();
		model
			.as_table_mut()
			.unwrap()
			.insert("targets".into(), toml::from_str(&experiment).unwrap());
		let mut config = toml::map::Map::new();
		config.insert("model".into(), model);
		let file = tempfile::NamedTempFile::new().unwrap();
		std::fs::write(file.path(), toml::to_string(&config).unwrap()).unwrap();
		let training = TrainingConfig::default().with_config_from_file(file.path()).unwrap();
		assert_eq!(training.targets.stop_level_definition, targets.stop_level_definition);
		let json = serde_json::to_value(&training).unwrap();
		assert_eq!(json["targets"]["stop_level"]["reference"], "anchor_close");
		let restored: TrainingConfig = serde_json::from_value(json).unwrap();
		assert_eq!(restored.targets.stop_level_definition, targets.stop_level_definition);
		let mut disabled = targets.clone();
		disabled.stop_level = false;
		let restored: TargetsConfig = toml::from_str(&toml::to_string(&disabled).unwrap()).unwrap();
		assert!(!restored.stop_level);
		assert_eq!(restored.stop_level_definition, targets.stop_level_definition);
		for invalid in [
			experiment.replace("volatility\"", "window_range\""),
			experiment.replace("anchor_close", "typo"),
		] {
			assert!(toml::from_str::<TargetsConfig>(&invalid).is_err());
		}
	}
	let invalid = old.replace("stop_level = true\n", "") + "\n[stop_level]\nside = \"either\"\n";
	assert!(toml::from_str::<TargetsConfig>(&invalid).is_err());

	let regime = old.replace("stop_level = true\n", "")
		+ "\n[stop_level]\nreference = \"anchor_close\"\nscale = \"regime_volatility\"\nside = \"either\"\n";
	let targets: TargetsConfig = toml::from_str(&regime).unwrap();
	assert_eq!(
		targets.stop_level_definition,
		regime_stop_definition(StopLevelSide::Either)
	);
	let mut training = TrainingConfig {
		targets,
		..Default::default()
	};
	training.features.stats_window_bars = 1;
	assert!(training.validate().is_err());
}

#[test]
fn stop_level_anchor_either_dominates_realised_and_uses_unweighted_extremes() {
	use crate::config::training::StopLevelSide;
	use crate::targets::stop_levels::anchor_stop_level_statistic;
	let input = anchor_stop_sequence();
	let (anchor, vol) = crate::targets::price_levels::price_level_anchor_volatility(&input).unwrap();
	for close in [95.0, 100.0, 105.0] {
		for (low, high) in [(90.0, 120.0), (80.0, 110.0)] {
			let future = vec![create_candle(anchor, low, high, close)];
			let either = anchor_stop_level_statistic(&input, &future, StopLevelSide::Either).unwrap();
			let realised = anchor_stop_level_statistic(&input, &future, StopLevelSide::Realised).unwrap();
			assert!(either >= realised);
			assert_eq!(either, (anchor - low).max(high - anchor) / anchor / vol);
			assert_eq!(
				realised,
				(if close >= anchor { anchor - low } else { high - anchor }) / anchor / vol
			);
		}
	}
	let future = vec![
		create_candle(anchor, 94.0, 101.0, 101.0),
		create_candle(anchor, 90.0, 102.0, 102.0),
	];
	assert_eq!(
		anchor_stop_level_statistic(&input, &future, StopLevelSide::Realised).unwrap(),
		10.0 / anchor / vol
	);
	// No clamping: preserve the supplied signed formula for a gap above anchor.
	let future = vec![create_candle(103.0, 102.0, 105.0, 104.0)];
	assert_eq!(
		anchor_stop_level_statistic(&input, &future, StopLevelSide::Realised).unwrap(),
		-2.0 / anchor / vol
	);
}

#[test]
fn stop_level_anchor_formatter_threshold_prices_share_classification_boundaries() {
	use crate::config::training::StopLevelSide;
	use crate::output::formatter::OutputFormatter;
	use crate::targets::calibration::CalibratedParameters;
	let input = anchor_stop_sequence();
	let (anchor, vol) = crate::targets::price_levels::price_level_anchor_volatility(&input).unwrap();
	for side in [StopLevelSide::Realised, StopLevelSide::Either] {
		let params = StopLevelParams {
			definition: anchor_stop_definition(side),
			thresholds: Some([1.0, 2.0, 3.0, 4.0]),
			..Default::default()
		};
		let mut meta = CalibratedParameters::default();
		meta.stop_levels.insert("2m".into(), params.clone());
		let formatter = OutputFormatter::new(Default::default())
			.with_sequence_ohlcv(input.clone())
			.with_calibrated_parameters(meta);
		for bullish in [true, false] {
			let reconstruction = reconstruct_stop_levels(&[0.2; 5], &input, anchor, &params, Some(bullish)).unwrap();
			let prediction = formatter
				.create_stop_level_prediction(&[0.2; 5], anchor, "2m", None, None, Some(bullish))
				.unwrap();
			assert_eq!(reconstruction.reference_price, anchor);
			for (class, name) in [
				"extreme_adverse",
				"high_adverse",
				"moderate_adverse",
				"low_adverse",
				"minimal_adverse",
			]
			.iter()
			.enumerate()
			{
				assert_eq!(prediction.bins[*name].price, reconstruction.adverse_price_ranges[class]);
				if class == 0 {
					continue;
				}
				let threshold = params.thresholds.unwrap()[4 - class];
				let price = anchor
					* (if bullish {
						1.0 - threshold * vol
					} else {
						1.0 + threshold * vol
					});
				assert_eq!(prediction.bins[*name].price[usize::from(!bullish)], price);
				let future = if bullish {
					vec![create_candle(anchor, price, anchor, anchor)]
				} else {
					vec![create_candle(anchor, anchor, price, anchor - vol)]
				};
				assert_eq!(
					classify_stop_level_with_calibrated_params(&input, &future, &params)
						.unwrap()
						.0,
					class as i32
				);
				// One representable price beyond the stop enters the deeper bin.
				let mut beyond = future;
				if bullish {
					beyond[0].low = f64::from_bits(price.to_bits() - 1);
				} else {
					beyond[0].high = f64::from_bits(price.to_bits() + 1);
				}
				assert_eq!(
					classify_stop_level_with_calibrated_params(&input, &beyond, &params)
						.unwrap()
						.0,
					class as i32 - 1
				);
			}
			let b = reconstruction.boundaries.boundaries;
			let width = b[3] - b[0];
			assert_eq!(
				reconstruction.adverse_price_ranges[0],
				if bullish {
					[b[0] - width, b[0]]
				} else {
					[b[3], b[3] + width]
				}
			);
		}
	}
}

#[test]
fn stop_level_either_shallow_band_stops_at_anchor() {
	use crate::config::training::StopLevelSide;
	let input = anchor_stop_sequence();
	let (anchor, _) = crate::targets::price_levels::price_level_anchor_volatility(&input).unwrap();
	let params = StopLevelParams {
		definition: anchor_stop_definition(StopLevelSide::Either),
		thresholds: Some([1.0, 2.0, 3.0, 4.0]),
		..Default::default()
	};

	for bullish in [true, false] {
		let reconstruction =
			reconstruct_stop_levels(&[0.0, 0.0, 0.0, 0.0, 1.0], &input, anchor, &params, Some(bullish)).unwrap();
		let shallow = reconstruction.adverse_price_ranges[4];
		assert_eq!(shallow[usize::from(bullish)], anchor);
		assert_eq!(reconstruction.expected_adverse, (shallow[0] + shallow[1]) / 2.0);
		if bullish {
			assert!(shallow[0] < shallow[1]);
			assert!(shallow[1] <= anchor);
		} else {
			assert!(shallow[0] >= anchor);
			assert!(shallow[0] < shallow[1]);
		}
	}
}

#[tokio::test]
async fn stop_level_anchor_calibration_balanced_and_saved_generator_roundtrip() {
	use crate::config::training::{StopLevelSide, TargetsConfig, TrainingConfig};
	use crate::targets::calibration::{CalibratedParameters, EvaluationContext, ParameterCalibrator};
	use crate::targets::{MultiTargetConfig, TargetGenerator};
	use polars::prelude::*;
	let input = anchor_stop_sequence();
	let (anchor, vol) = crate::targets::price_levels::price_level_anchor_volatility(&input).unwrap();
	for side in [StopLevelSide::Realised, StopLevelSide::Either] {
		let mut data = Vec::new();
		let mut indices = Vec::new();
		for i in 0..100 {
			indices.push(data.len());
			data.extend(input.clone());
			let excursion = (i as f64 + 1.0) / 10.0;
			let low = anchor * (1.0 - excursion * vol);
			let high = anchor * (1.0 + excursion * vol);
			let close = if i % 2 == 0 { anchor + vol } else { anchor - vol };
			data.extend(vec![create_candle(anchor, low, high, close); 2]);
		}
		for (i, bar) in data.iter_mut().enumerate() {
			bar.timestamp = i as i64 * 60_000;
		}
		let targets = TargetsConfig {
			price_level: false,
			stop_level_definition: anchor_stop_definition(side),
			direction: false,
			volatility: false,
			sentiment: false,
			volume: false,
			..Default::default()
		};
		let calibrator = ParameterCalibrator::from_config_with_seed(&targets, Some(42));
		let params = calibrator
			.calibrate_stop_levels(
				&EvaluationContext {
					ohlcv_data: &data,
					sample_indices: &indices,
					sequence_length: input.len(),
					horizon_steps: 2,
				},
				"stop_level_test",
			)
			.await
			.unwrap();
		assert_eq!(params.balance.class_percentages, [20.0; 5]);
		let mut meta = CalibratedParameters::default();
		meta.metadata.overall_balance_score = params.balance.composite_quality_score;
		meta.stop_levels.insert("2m".into(), params);
		let json = serde_json::to_value(&meta).unwrap();
		assert_eq!(json["stop_levels"]["2m"]["reference"], "anchor_close");
		assert_eq!(json["stop_levels"]["2m"]["thresholds"].as_array().unwrap().len(), 4);
		let meta: CalibratedParameters = serde_json::from_value(json).unwrap();
		let binary = bincode::serialize(&(meta.clone(), 123456u64)).unwrap();
		let (meta, suffix): (CalibratedParameters, u64) = bincode::deserialize(&binary).unwrap();
		assert_eq!(suffix, 123456);
		assert_eq!(bincode::serialize(&(meta.clone(), suffix)).unwrap(), binary);
		assert_eq!(meta.stop_levels["2m"].definition, anchor_stop_definition(side));
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
		let mut counts = [0; 5];
		for (i, start) in indices.into_iter().enumerate() {
			let class = classify_stop_level_with_calibrated_params(
				&input,
				&data[start + input.len()..start + input.len() + 2],
				&meta.stop_levels["2m"],
			)
			.unwrap()
			.0;
			assert_eq!(generated.stop_levels["2m"][i], class);
			counts[class as usize] += 1;
		}
		assert_eq!(counts, [20; 5]);
	}
}

#[test]
fn stop_level_anchor_rejects_invalid_data_and_missing_thresholds() {
	use crate::config::training::StopLevelSide;
	let input = anchor_stop_sequence();
	let future = vec![create_candle(100.0, 95.0, 105.0, 100.0)];
	let mut params = StopLevelParams {
		definition: anchor_stop_definition(StopLevelSide::Realised),
		..Default::default()
	};
	assert!(classify_stop_level_with_calibrated_params(&input, &future, &params).is_err());
	assert!(reconstruct_stop_levels(&[0.2; 5], &input, 100.0, &params, None).is_err());
	for thresholds in [[1.0; 4], [1.0, 2.0, 4.0, 3.0], [1.0, 2.0, 3.0, f64::NAN]] {
		params.thresholds = Some(thresholds);
		assert!(classify_stop_level_with_calibrated_params(&input, &future, &params).is_err());
	}
	params.thresholds = Some([1.0, 2.0, 3.0, 4.0]);
	assert!(classify_stop_level_with_calibrated_params(&[], &future, &params).is_err());
	assert!(classify_stop_level_with_calibrated_params(&input, &[], &params).is_err());
	for price in [0.0, f64::NAN, f64::INFINITY] {
		let invalid = vec![create_candle(100.0, price, 105.0, 100.0)];
		assert!(classify_stop_level_with_calibrated_params(&input, &invalid, &params).is_err());
	}
	let flat = vec![create_candle(100.0, 100.0, 100.0, 100.0); 6];
	assert!(classify_stop_level_with_calibrated_params(&flat, &future, &params).is_err());
	let mut binary = bincode::serialize(&(params, 123456u64)).unwrap();
	binary[8] = 255;
	assert!(bincode::deserialize::<(StopLevelParams, u64)>(&binary).is_err());
}

#[test]
fn regime_stop_level_label_and_serving_boundaries_match_for_both_sides() {
	use crate::config::training::StopLevelSide;
	use crate::targets::stop_levels::{classify_stop_level_with_history, reconstruct_stop_levels_with_history};

	let history = anchor_stop_sequence();
	let sequence = &history[history.len() - 3..];
	let (anchor, regime_volatility) = crate::targets::price_levels::price_level_anchor_volatility(&history).unwrap();
	for side in [StopLevelSide::Realised, StopLevelSide::Either] {
		let params = StopLevelParams {
			definition: regime_stop_definition(side),
			thresholds: Some([1.0, 2.0, 3.0, 4.0]),
			regime_window_bars: Some(history.len()),
			..Default::default()
		};
		for is_bullish in [true, false] {
			let distance = anchor * regime_volatility * 2.5;
			let direction_close = anchor
				* if is_bullish {
					1.0 + regime_volatility
				} else {
					1.0 - regime_volatility
				};
			let future = if is_bullish {
				vec![create_candle(
					anchor,
					anchor - distance,
					direction_close,
					direction_close,
				)]
			} else {
				vec![create_candle(
					anchor,
					direction_close,
					anchor + distance,
					direction_close,
				)]
			};
			let (class, _) = classify_stop_level_with_history(sequence, &history, &future, &params).unwrap();
			assert_eq!(class, 2);

			let reconstruction = reconstruct_stop_levels_with_history(
				&[0.0, 0.0, 1.0, 0.0, 0.0],
				sequence,
				&history,
				anchor,
				&params,
				Some(is_bullish),
			)
			.unwrap();
			let adverse = if is_bullish { future[0].low } else { future[0].high };
			assert!(reconstruction.adverse_price_ranges[2][0] <= adverse);
			assert!(adverse < reconstruction.adverse_price_ranges[2][1]);
			let thresholds = params.thresholds.unwrap();
			for (index, threshold) in thresholds.into_iter().enumerate() {
				let expected = anchor
					* if is_bullish {
						1.0 - threshold * regime_volatility
					} else {
						1.0 + threshold * regime_volatility
					};
				let boundary_index = if is_bullish { 3 - index } else { index };
				assert_eq!(reconstruction.boundaries.boundaries[boundary_index], expected);
			}
		}
	}
}

#[test]
fn regime_stop_level_metadata_roundtrips_and_serving_validates_history() {
	use crate::config::training::StopLevelSide;
	use crate::targets::stop_levels::{reconstruct_stop_levels_with_history, validate_stop_level_regime_window};

	let history = anchor_stop_sequence();
	let sequence = &history[history.len() - 3..];
	let params = StopLevelParams {
		definition: regime_stop_definition(StopLevelSide::Either),
		thresholds: Some([1.0, 2.0, 3.0, 4.0]),
		regime_window_bars: Some(history.len()),
		balance: crate::targets::calibration::ClassBalance {
			composite_quality_score: 0.0,
			..Default::default()
		},
		..Default::default()
	};
	let mut meta = crate::targets::calibration::CalibratedParameters::default();
	meta.metadata.overall_balance_score = 0.0;
	meta.stop_levels.insert("1h".to_string(), params.clone());
	let json = serde_json::to_value(&meta).unwrap();
	assert_eq!(json["stop_levels"]["1h"]["scale"], "regime_volatility");
	assert_eq!(json["stop_levels"]["1h"]["regime_window_bars"], history.len());
	let json_loaded: crate::targets::calibration::CalibratedParameters = serde_json::from_value(json).unwrap();
	assert_eq!(json_loaded.stop_levels["1h"].regime_window_bars, Some(history.len()));
	let binary = bincode::serialize(&(meta, 123456u64)).unwrap();
	let (binary_loaded, suffix): (crate::targets::calibration::CalibratedParameters, u64) =
		bincode::deserialize(&binary).unwrap();
	assert_eq!(suffix, 123456);
	assert_eq!(binary_loaded.stop_levels["1h"].definition, params.definition);
	assert_eq!(binary_loaded.stop_levels["1h"].thresholds, params.thresholds);
	assert_eq!(
		binary_loaded.stop_levels["1h"].regime_window_bars,
		params.regime_window_bars
	);
	assert_eq!(bincode::serialize(&(binary_loaded, suffix)).unwrap(), binary);

	validate_stop_level_regime_window(&params, history.len()).unwrap();
	assert!(validate_stop_level_regime_window(&params, history.len() + 1).is_err());
	assert!(reconstruct_stop_levels_with_history(
		&[0.2; 5],
		sequence,
		&history[1..],
		sequence.last().unwrap().close,
		&params,
		Some(true),
	)
	.is_err());
}

#[tokio::test]
async fn regime_stop_level_calibration_and_label_generation_skip_short_history() {
	use crate::config::training::{StopLevelSide, TrainingConfig};
	use crate::targets::calibration::{EvaluationContext, ParameterCalibrator};
	use polars::prelude::*;
	use std::collections::HashMap;

	let regime_window_bars = 6;
	let sequence_length = 3;
	let sample_indices = vec![0, 10, 20, 30, 40, 50];
	let pattern = [100.0, 101.0, 99.0, 102.0, 98.0, 100.0];
	let mut data: Vec<MarketDataRow> = (0..60)
		.map(|index| {
			let close = pattern[index % pattern.len()];
			let mut row = create_candle(close, close, close, close);
			row.timestamp = index as i64 * 60_000;
			row
		})
		.collect();
	for (position, start) in sample_indices.iter().copied().skip(1).enumerate() {
		let end = start + sequence_length;
		let history = &data[..end];
		let (anchor, volatility) =
			crate::targets::price_levels::price_level_anchor_volatility(&history[history.len() - regime_window_bars..])
				.unwrap();
		let excursion = position as f64 + 1.0;
		let low = anchor * (1.0 - excursion * volatility);
		let close = anchor * (1.0 + volatility);
		data[end] = create_candle(anchor, low, close, close);
		data[end].timestamp = end as i64 * 60_000;
	}

	let mut training = TrainingConfig::default();
	training.targets.price_level = false;
	training.targets.stop_level_definition = regime_stop_definition(StopLevelSide::Realised);
	training.features.stats_window_bars = regime_window_bars;
	let calibrator = ParameterCalibrator::from_training_config_with_seed(&training, Some(42));
	let params = calibrator
		.calibrate_stop_levels(
			&EvaluationContext {
				ohlcv_data: &data,
				sample_indices: &sample_indices,
				sequence_length,
				horizon_steps: 1,
			},
			"regime_stop_level_test",
		)
		.await
		.unwrap();
	assert_eq!(params.regime_window_bars, Some(regime_window_bars));
	assert_eq!(params.balance.total_samples, sample_indices.len() - 1);
	assert_eq!(params.balance.class_percentages, [20.0; 5]);

	let df = df!(
		"timestamp" => data.iter().map(|row| row.timestamp).collect::<Vec<_>>(),
		"open" => data.iter().map(|row| row.open).collect::<Vec<_>>(),
		"high" => data.iter().map(|row| row.high).collect::<Vec<_>>(),
		"low" => data.iter().map(|row| row.low).collect::<Vec<_>>(),
		"close" => data.iter().map(|row| row.close).collect::<Vec<_>>(),
		"volume" => data.iter().map(|row| row.volume).collect::<Vec<_>>()
	)
	.unwrap();
	let mut by_horizon = HashMap::new();
	by_horizon.insert("1m".to_string(), params);
	let (labels, _) = crate::targets::stop_levels::generate_stop_level_targets_with_calibrated_params(
		&df,
		&["1m".to_string()],
		&sample_indices,
		sequence_length,
		&by_horizon,
	)
	.unwrap();
	assert_eq!(labels["1m"][0], -1);
	assert!(labels["1m"][1..].iter().all(|label| *label >= 0));
}

#[tokio::test]
async fn regime_incomputable_rows_are_removed_before_balancing_with_all_heads_aligned() {
	use crate::data::balance::{create_sequences_with_targets, BalanceConfig, SequenceBalancer};
	use crate::data::{retain_computable_target_rows, DataMetadata, PreparedData};
	use crate::targets::{PreparedTargets, TargetType};
	use ndarray::Array3;

	let sequence_length = 1;
	let regime_window_bars = 6;
	let row_count = 30;
	assert!(regime_window_bars > sequence_length);
	let first_computable = regime_window_bars - sequence_length;
	let mut sequences = Array3::zeros((row_count, sequence_length, 1));
	for index in 0..row_count {
		sequences[[index, 0, 0]] = index as f64;
	}
	let mut targets = PreparedTargets::new(row_count);
	let stop_labels: Vec<i32> = (0..row_count)
		.map(|index| {
			if index < first_computable {
				-1
			} else {
				((index - first_computable) % 5) as i32
			}
		})
		.collect();
	let aligned_labels: Vec<i32> = (0..row_count)
		.map(|index| ((index.saturating_sub(first_computable)) % 5) as i32)
		.collect();
	targets.stop_levels.insert("1m".to_string(), stop_labels);
	targets.direction.insert("1m".to_string(), aligned_labels);
	targets
		.stop_levels_strength
		.insert("1m".to_string(), (0..row_count).map(|index| index as f64).collect());
	targets
		.direction_strength
		.insert("1m".to_string(), (0..row_count).map(|index| index as f64).collect());
	targets.target_names = vec!["stop_level_1m".to_string(), "direction_1m".to_string()];
	targets.valid_indices = targets.computable_stop_indices();
	let now = chrono::Utc::now();
	let filtered = retain_computable_target_rows(PreparedData {
		sequences,
		targets,
		feature_names: vec!["source_index".to_string()],
		metadata: DataMetadata {
			symbol: "TEST".to_string(),
			start_time: now,
			end_time: now,
			total_records: row_count,
			feature_count: 1,
			sequence_length,
			horizons: vec!["1m".to_string()],
		},
		sequence_indices: (0..row_count).map(|index| (index, index + 1)).collect(),
	})
	.unwrap();

	assert_eq!(filtered.sequences.shape()[0], row_count - first_computable);
	assert_eq!(filtered.targets.data_length, row_count - first_computable);
	assert_eq!(filtered.sequence_indices.len(), row_count - first_computable);
	for filtered_index in 0..filtered.targets.data_length {
		let source_index = filtered_index + first_computable;
		assert_eq!(filtered.sequences[[filtered_index, 0, 0]], source_index as f64);
		assert_eq!(
			filtered.sequence_indices[filtered_index],
			(source_index, source_index + 1)
		);
		for target_type in [TargetType::StopLevel, TargetType::Direction] {
			assert_eq!(
				filtered.targets.get_targets("1m", target_type).unwrap()[filtered_index],
				(filtered_index % 5) as i32
			);
			assert_eq!(
				filtered.targets.get_strengths("1m", target_type).unwrap()[filtered_index],
				source_index as f64
			);
		}
	}
	assert!(filtered.targets.stop_levels["1m"].iter().all(|label| *label >= 0));

	let joined = create_sequences_with_targets(
		filtered.sequences.clone(),
		&filtered.targets,
		filtered.sequence_indices.clone(),
	)
	.await
	.unwrap();
	let balancer = SequenceBalancer::new(BalanceConfig {
		max_overlap: 0.0,
		prefer_non_overlapping: false,
		min_sequences_per_class: 1,
	});
	for target_type in [TargetType::StopLevel, TargetType::Direction] {
		let selection = balancer
			.balance_sequences_for_window(&joined, target_type, "1m", &[], None)
			.unwrap();
		assert_eq!(selection.selected_indices.len(), row_count - first_computable);
	}
}
