use crate::config::training::{PriceLevelDefinition, PriceLevelReference, PriceLevelScale};
use crate::data::structures::MarketDataRow;
use crate::features::boundary::{add_boundary_features, BoundaryInputs};
use crate::features::context::is_context_column;
use crate::targets::calibration::{CalibratedParameters, DirectionParams, PriceLevelParams, StopLevelParams};
use crate::targets::direction::classify_direction_with_calibrated_params;
use crate::targets::price_levels::{
	classify_price_level_with_calibrated_params, extract_sequence_prices, get_horizon_exponential_weighted_close,
	get_sequence_exponential_weighted_close,
};
use crate::targets::stop_levels::calculate_adverse_boundaries;
use polars::prelude::*;
use std::collections::BTreeSet;

const HOUR: i64 = 3600;
/// 2026-01-05 00:00:00 UTC.
const START: i64 = 1_767_571_200;
const ROWS: usize = 240;
const SEQUENCE_BARS: usize = 12;
const HORIZON: &str = "4h";
const HORIZON_BARS: usize = 4;
const COLUMNS_PER_HORIZON: usize = 17;
const TOLERANCE: f64 = 1e-9;

/// Deterministic values in (-1, 1), so tests need no random source.
fn pseudo_shocks(count: usize) -> Vec<f64> {
	let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
	(0..count)
		.map(|_| {
			state = state
				.wrapping_mul(6_364_136_223_846_793_005)
				.wrapping_add(1_442_695_040_888_963_407);
			((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
		})
		.collect()
}

/// Hourly bars: a random walk of closes with a wick on each side, every price times `price_scale`.
fn bars(price_scale: f64) -> Vec<MarketDataRow> {
	let shocks = pseudo_shocks(ROWS * 3);
	let mut close = 100.0 * price_scale;
	(0..ROWS)
		.map(|row| {
			let open = close;
			close *= (0.01 * shocks[3 * row]).exp();
			let high = open.max(close) * (1.0 + 0.003 * shocks[3 * row + 1].abs());
			let low = open.min(close) * (1.0 - 0.003 * shocks[3 * row + 2].abs());
			MarketDataRow::new(START + row as i64 * HOUR, open, high, low, close, 10.0)
		})
		.collect()
}

fn frame(bars: &[MarketDataRow]) -> DataFrame {
	df!(
		"timestamp" => bars.iter().map(|bar| bar.timestamp).collect::<Vec<i64>>(),
		"open" => bars.iter().map(|bar| bar.open).collect::<Vec<f64>>(),
		"high" => bars.iter().map(|bar| bar.high).collect::<Vec<f64>>(),
		"low" => bars.iter().map(|bar| bar.low).collect::<Vec<f64>>(),
		"close" => bars.iter().map(|bar| bar.close).collect::<Vec<f64>>(),
		"volume" => bars.iter().map(|bar| bar.volume).collect::<Vec<f64>>(),
	)
	.unwrap()
}

fn calibrated() -> CalibratedParameters {
	let mut calibrated = CalibratedParameters::default();
	calibrated.price_levels.insert(
		HORIZON.to_string(),
		PriceLevelParams {
			bandwidth: 0.66,
			percentiles: [0.26, 0.78],
			neutral_band_factor: 0.64,
			..Default::default()
		},
	);
	calibrated.stop_levels.insert(
		HORIZON.to_string(),
		StopLevelParams {
			bandwidth: 0.45,
			percentiles: [0.39, 0.71],
			neutral_band_factor: 0.68,
			..Default::default()
		},
	);
	calibrated.direction.insert(
		HORIZON.to_string(),
		DirectionParams {
			sensitivity: 0.4,
			extreme_multiplier: 2.5,
			min_base_threshold: 0.002,
			min_extreme_threshold: 0.006,
			base_multiplier: 2.5,
			..Default::default()
		},
	);
	calibrated
}

fn boundary_columns(
	bars: &[MarketDataRow],
	calibrated: &CalibratedParameters,
) -> crate::utils::error::Result<DataFrame> {
	let horizons = [HORIZON.to_string()];
	let inputs = BoundaryInputs {
		calibrated,
		sequence_length: SEQUENCE_BARS,
		horizons: &horizons,
	};
	add_boundary_features(frame(bars), &inputs)
}

fn column(df: &DataFrame, name: &str) -> Vec<f64> {
	df.column(name)
		.unwrap()
		.f64()
		.unwrap()
		.into_iter()
		.map(|value| value.unwrap())
		.collect()
}

/// The four boundary columns of one label, lowest boundary first.
fn numbered(df: &DataFrame, label: &str) -> [Vec<f64>; 4] {
	[1, 2, 3, 4].map(|boundary| column(df, &format!("ctx_bound_{label}_{HORIZON}_{boundary}")))
}

fn context_names(df: &DataFrame) -> Vec<String> {
	df.get_column_names()
		.iter()
		.map(|name| name.to_string())
		.filter(|name| is_context_column(name))
		.collect()
}

/// The unit of every column: the standard deviation of the bar returns inside the input sequence, times
/// the square root of the horizon length.
fn normal_move(sequence: &[MarketDataRow]) -> f64 {
	let returns: Vec<f64> = sequence
		.windows(2)
		.map(|pair| (pair[1].close / pair[0].close).ln())
		.collect();
	let mean = returns.iter().sum::<f64>() / returns.len() as f64;
	let variance = returns.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / returns.len() as f64;
	variance.sqrt() * (HORIZON_BARS as f64).sqrt()
}

/// Every row with a complete input sequence and a complete horizon: the row, its sequence, its horizon.
fn calls(bars: &[MarketDataRow]) -> impl Iterator<Item = (usize, &[MarketDataRow], &[MarketDataRow])> {
	(SEQUENCE_BARS - 1..bars.len() - HORIZON_BARS).map(move |end| {
		(
			end,
			&bars[end + 1 - SEQUENCE_BARS..=end],
			&bars[end + 1..=end + HORIZON_BARS],
		)
	})
}

#[test]
fn adds_seventeen_columns_and_fills_them_from_the_first_complete_sequence() {
	let df = boundary_columns(&bars(1.0), &calibrated()).unwrap();
	let names = context_names(&df);
	assert_eq!(names.len(), COLUMNS_PER_HORIZON);
	for name in &names {
		let values = column(&df, name);
		assert!(
			values[..SEQUENCE_BARS - 1].iter().all(|value| value.is_nan()),
			"{name} warm-up"
		);
		assert!(
			values[SEQUENCE_BARS - 1..].iter().all(|value| value.is_finite()),
			"{name} after warm-up"
		);
	}
}

#[test]
fn price_label_is_the_number_of_price_columns_at_or_below_the_future_price() {
	let (bars, calibrated) = (bars(1.0), calibrated());
	let columns = numbered(&boundary_columns(&bars, &calibrated).unwrap(), "price");
	let params = calibrated.get_price_levels(HORIZON).unwrap();
	let mut classes_seen = BTreeSet::new();
	for (end, sequence, horizon) in calls(&bars) {
		let (label, _) = classify_price_level_with_calibrated_params(sequence, horizon, params).unwrap();
		let future_price = get_horizon_exponential_weighted_close(horizon).unwrap();
		let position = (future_price / bars[end].close - 1.0) / normal_move(sequence);
		let from_columns = columns.iter().filter(|column| position >= column[end]).count() as i32;
		assert_eq!(from_columns, label, "row {end}");
		classes_seen.insert(label);
	}
	assert!(classes_seen.len() >= 3, "classes seen: {classes_seen:?}");
}

#[test]
fn direction_label_is_the_number_of_direction_columns_below_the_future_momentum() {
	let (bars, calibrated) = (bars(1.0), calibrated());
	let columns = numbered(&boundary_columns(&bars, &calibrated).unwrap(), "direction");
	let params = calibrated.get_direction(HORIZON).unwrap();
	let mut classes_seen = BTreeSet::new();
	for (end, sequence, horizon) in calls(&bars) {
		let (label, _) = classify_direction_with_calibrated_params(sequence, horizon, params).unwrap();
		let prices = extract_sequence_prices(horizon);
		let momentum = (prices[prices.len() - 1] - prices[0]) / prices[0];
		let position = momentum / normal_move(sequence);
		let from_columns = columns.iter().filter(|column| position > column[end]).count() as i32;
		assert_eq!(from_columns, label, "row {end}");
		classes_seen.insert(label);
	}
	assert!(classes_seen.len() >= 3, "classes seen: {classes_seen:?}");
}

#[test]
fn stop_columns_are_the_label_boundaries_of_the_side_the_horizon_ends_on() {
	let (bars, calibrated) = (bars(1.0), calibrated());
	let df = boundary_columns(&bars, &calibrated).unwrap();
	let (dips, bounces) = (numbered(&df, "stop_dip"), numbered(&df, "stop_bounce"));
	let side = column(&df, &format!("ctx_bound_stop_side_{HORIZON}"));
	let params = calibrated.get_stop_levels(HORIZON).unwrap();
	let mut sides_seen = BTreeSet::new();
	for (end, sequence, horizon) in calls(&bars) {
		let (last_close, unit) = (bars[end].close, normal_move(sequence));
		let to_price = |distance: f64| last_close * (1.0 + distance * unit);
		let reference = get_sequence_exponential_weighted_close(sequence).unwrap();
		assert!((to_price(side[end]) / reference - 1.0).abs() < TOLERANCE, "row {end}");
		// The label reads dips when the horizon ends at or above the reference price, bounces otherwise.
		let future_price = get_horizon_exponential_weighted_close(horizon).unwrap();
		let ends_above = (future_price / last_close - 1.0) / unit >= side[end];
		assert_eq!(ends_above, future_price >= reference, "row {end}");
		let label_boundaries = calculate_adverse_boundaries(sequence, ends_above, params)
			.unwrap()
			.boundaries;
		for (column, boundary) in (if ends_above { &dips } else { &bounces }).iter().zip(label_boundaries) {
			assert!((to_price(column[end]) / boundary - 1.0).abs() < TOLERANCE, "row {end}");
		}
		sides_seen.insert(ends_above);
	}
	assert_eq!(sides_seen.len(), 2);
}

#[test]
fn columns_do_not_depend_on_the_price_unit() {
	let calibrated = calibrated();
	let base = boundary_columns(&bars(1.0), &calibrated).unwrap();
	let scaled = boundary_columns(&bars(0.0003), &calibrated).unwrap();
	for name in context_names(&base) {
		let (left, right) = (column(&base, &name), column(&scaled, &name));
		for (a, b) in left.iter().zip(&right).skip(SEQUENCE_BARS - 1) {
			assert!((a - b).abs() < 1e-6, "{name}: {a} vs {b}");
		}
	}
}

#[test]
fn only_calibrated_labels_get_columns() {
	let mut calibrated = calibrated();
	calibrated.price_levels.clear();
	calibrated.stop_levels.clear();
	let df = boundary_columns(&bars(1.0), &calibrated).unwrap();
	assert_eq!(context_names(&df).len(), 4);
	// Nothing to add is a configuration mistake, not an empty result.
	calibrated.direction.clear();
	assert!(boundary_columns(&bars(1.0), &calibrated).is_err());
}

#[test]
fn a_regime_volatility_label_starts_when_its_regime_window_is_full() {
	let regime_bars = 3 * SEQUENCE_BARS;
	let mut calibrated = calibrated();
	calibrated.stop_levels.clear();
	calibrated.direction.clear();
	calibrated.price_levels.insert(
		HORIZON.to_string(),
		PriceLevelParams {
			definition: PriceLevelDefinition {
				reference: PriceLevelReference::AnchorClose,
				scale: PriceLevelScale::RegimeVolatility,
				..Default::default()
			},
			thresholds: Some([-2.0, -1.0, 1.0, 2.0]),
			regime_window_bars: Some(regime_bars),
			..Default::default()
		},
	);
	let df = boundary_columns(&bars(1.0), &calibrated).unwrap();
	for values in numbered(&df, "price") {
		assert!(values[..regime_bars - 1].iter().all(|value| value.is_nan()));
		assert!(values[regime_bars - 1..].iter().all(|value| value.is_finite()));
	}
}

#[test]
fn rejects_an_input_sequence_that_never_moved() {
	let mut bars = bars(1.0);
	for bar in &mut bars[40..40 + SEQUENCE_BARS] {
		bar.close = 100.0;
	}
	assert!(boundary_columns(&bars, &calibrated()).is_err());
}
