use super::formatter::{infer_stop_is_bullish, stop_level_reference_price};
use super::prediction_types::{PriceBin, PriceLevelPrediction};
use std::collections::HashMap;

fn price_level_prediction(price: [f64; 2]) -> PriceLevelPrediction {
	let mut bins = HashMap::new();
	bins.insert(
		"predicted".to_string(),
		PriceBin {
			range: [0.0, 0.0],
			vwap_range: [0.0, 0.0],
			price,
			probability: 1.0,
		},
	);
	PriceLevelPrediction {
		bins,
		most_likely_range: [0.0, 0.0],
		confidence: 1.0,
	}
}

#[test]
fn stop_direction_uses_predicted_future_price_level() {
	let bullish = price_level_prediction([104.0, 106.0]);
	let bearish = price_level_prediction([94.0, 96.0]);

	assert_eq!(infer_stop_is_bullish(&bullish, 100.0), Some(true));
	assert_eq!(infer_stop_is_bullish(&bearish, 100.0), Some(false));
}

#[test]
fn stop_direction_reference_follows_stop_definition() {
	use crate::config::training::{PriceLevelReference, PriceLevelScale, StopLevelDefinition, StopLevelSide};
	use crate::data::structures::MarketDataRow;
	use crate::targets::calibration::StopLevelParams;

	let sequence: Vec<MarketDataRow> = [90.0, 95.0, 100.0, 105.0, 110.0]
		.into_iter()
		.map(|close| MarketDataRow {
			timestamp: 0,
			open: close,
			high: close,
			low: close,
			close,
			volume: 1.0,
		})
		.collect();
	let forecast = price_level_prediction([109.54, 109.54]);
	let legacy = StopLevelParams::default();
	let anchor = StopLevelParams {
		definition: StopLevelDefinition {
			reference: PriceLevelReference::AnchorClose,
			scale: PriceLevelScale::Volatility,
			side: StopLevelSide::Realised,
		},
		..Default::default()
	};

	let legacy_reference = stop_level_reference_price(&sequence, &sequence, &legacy).unwrap();
	let anchor_reference = stop_level_reference_price(&sequence, &sequence, &anchor).unwrap();
	assert!(legacy_reference < 109.54);
	assert_eq!(anchor_reference, 110.0);
	assert_eq!(infer_stop_is_bullish(&forecast, legacy_reference), Some(true));
	assert_eq!(infer_stop_is_bullish(&forecast, anchor_reference), Some(false));
}

#[test]
fn regime_stop_serving_rejects_config_window_mismatch() {
	use crate::config::training::{PriceLevelReference, PriceLevelScale, StopLevelDefinition, StopLevelSide};
	use crate::data::structures::MarketDataRow;
	use crate::output::formatter::OutputFormatter;
	use crate::targets::calibration::{CalibratedParameters, StopLevelParams};

	let history: Vec<MarketDataRow> = [90.0, 95.0, 100.0, 105.0, 110.0]
		.into_iter()
		.map(|close| MarketDataRow {
			timestamp: 0,
			open: close,
			high: close,
			low: close,
			close,
			volume: 1.0,
		})
		.collect();
	let mut calibrated = CalibratedParameters::default();
	calibrated.stop_levels.insert(
		"1h".to_string(),
		StopLevelParams {
			definition: StopLevelDefinition {
				reference: PriceLevelReference::AnchorClose,
				scale: PriceLevelScale::RegimeVolatility,
				side: StopLevelSide::Either,
			},
			thresholds: Some([1.0, 2.0, 3.0, 4.0]),
			regime_window_bars: Some(history.len()),
			..Default::default()
		},
	);
	let formatter = OutputFormatter::new(Default::default())
		.with_sequence_ohlcv(history[history.len() - 3..].to_vec())
		.with_stop_level_history(history.clone())
		.with_stats_window_bars(history.len() + 1)
		.with_calibrated_parameters(calibrated);
	let error = formatter
		.create_stop_level_prediction(&[0.2; 5], history.last().unwrap().close, "1h", None, None, Some(true))
		.unwrap_err()
		.to_string();
	assert!(error.contains("regime window mismatch"), "{error}");
}

#[test]
fn anchor_price_ranges_use_current_and_vwap_references() {
	use crate::config::training::{PriceLevelDefinition, PriceLevelReference, PriceLevelScale};
	use crate::data::structures::MarketDataRow;
	use crate::output::formatter::OutputFormatter;
	use crate::targets::calibration::{CalibratedParameters, PriceLevelParams};
	use crate::targets::get_sequence_exponential_weighted_close;

	let sequence: Vec<MarketDataRow> = [90.0, 95.0, 100.0, 105.0, 110.0]
		.into_iter()
		.enumerate()
		.map(|(index, close)| MarketDataRow {
			timestamp: index as i64 * 60_000,
			open: close,
			high: close,
			low: close,
			close,
			volume: 1000.0,
		})
		.collect();
	let anchor = sequence.last().unwrap().close;
	let vwap = get_sequence_exponential_weighted_close(&sequence).unwrap();
	assert_ne!(anchor, vwap);
	let mut calibrated = CalibratedParameters::default();
	calibrated.price_levels.insert(
		"1h".to_string(),
		PriceLevelParams {
			definition: PriceLevelDefinition {
				reference: PriceLevelReference::AnchorClose,
				scale: PriceLevelScale::Volatility,
			},
			thresholds: Some([-2.0, -1.0, 1.0, 2.0]),
			..Default::default()
		},
	);
	let prediction = OutputFormatter::new(Default::default())
		.with_sequence_ohlcv(sequence)
		.with_calibrated_parameters(calibrated)
		.create_price_level_prediction(&[0.2; 5], anchor, "1h", None, None)
		.unwrap();

	for name in ["strong_down", "moderate_down", "neutral", "moderate_up", "strong_up"] {
		let bin = &prediction.bins[name];
		for endpoint in 0..2 {
			assert_eq!(bin.range[endpoint], (bin.price[endpoint] / anchor - 1.0) * 100.0);
			assert_eq!(bin.vwap_range[endpoint], (bin.price[endpoint] / vwap - 1.0) * 100.0);
		}
	}
}

#[test]
fn probabilities_reject_nonfinite_and_negative_for_every_head() {
	use crate::output::multi_target_parser::MultiTargetParser;
	for target in [
		"price_level",
		"stop_level",
		"direction",
		"volatility",
		"sentiment",
		"volume",
	] {
		let parser = MultiTargetParser::with_target_names(&[format!("{target}_8h")]);
		for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01] {
			let values = ndarray::array![0.2, 0.2, value, 0.2, 0.2];
			assert!(parser.parse_output(values.view()).is_err(), "{target} accepted {value}");
		}
		assert!(parser
			.parse_output(ndarray::array![0.1, 0.2, 0.3, 0.2, 0.2].view())
			.is_ok());
	}
}

#[test]
fn stop_side_ties_are_long_and_missing_price_signal_has_no_side() {
	let flat = price_level_prediction([99.0, 101.0]);
	assert_eq!(infer_stop_is_bullish(&flat, 100.0), Some(true));
	assert_eq!(infer_stop_is_bullish(&flat, f64::NAN), None);
	let mut missing = flat;
	missing.bins.clear();
	assert_eq!(infer_stop_is_bullish(&missing, 100.0), None);
}

#[test]
fn serving_skips_stop_only_predictions_without_aborting_horizon() {
	let names = vec!["stop_level_8h".to_string()];
	let formatter = crate::output::OutputFormatter::new(Default::default())
		.with_target_names(&names)
		.with_min_confidence(0.0);
	let predictions = ndarray::Array2::from_elem((1, 5), 0.2);
	let results = formatter
		.format_predictions_with_targets(&predictions, "T", "8h", 100.0, None, Some(&names))
		.unwrap();
	assert_eq!(results.len(), 1);
	assert!(results[0].price_levels.is_none());
	assert!(results[0].stop_levels.is_none());
}
