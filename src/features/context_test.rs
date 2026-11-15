use crate::features::context::{add_context_features, is_context_column, lookback_ladder, CONTEXT_PREFIX};
use polars::prelude::*;

const HOUR: i64 = 3600;
/// 2026-01-05 00:00:00 UTC, a Monday.
const MONDAY_MIDNIGHT: i64 = 1_767_571_200;
const REGIME_BARS: usize = 27;
const TOLERANCE: f64 = 1e-9;

/// Deterministic returns in (-1, 1), so tests need no random source.
fn pseudo_returns(count: usize) -> Vec<f64> {
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

/// Hourly bars whose per-bar return size is given by `step_sizes`.
fn frame(step_sizes: &[f64], price_scale: f64, volume_scale: f64) -> DataFrame {
	let noise = pseudo_returns(step_sizes.len());
	let mut close = 100.0 * price_scale;
	let mut closes = Vec::with_capacity(step_sizes.len());
	for (size, shock) in step_sizes.iter().zip(&noise) {
		close *= (size * shock).exp();
		closes.push(close);
	}
	let timestamps: Vec<i64> = (0..step_sizes.len() as i64)
		.map(|i| MONDAY_MIDNIGHT + i * HOUR)
		.collect();
	let volumes: Vec<f64> = noise.iter().map(|shock| (10.0 + 5.0 * shock) * volume_scale).collect();
	df!(
		"timestamp" => timestamps,
		"open" => closes.clone(),
		"high" => closes.clone(),
		"low" => closes.clone(),
		"close" => closes,
		"volume" => volumes,
	)
	.unwrap()
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

fn context_names(df: &DataFrame) -> Vec<String> {
	df.get_column_names()
		.iter()
		.map(|name| name.to_string())
		.filter(|name| is_context_column(name))
		.collect()
}

#[test]
fn ladder_is_derived_from_the_regime_window() {
	assert_eq!(lookback_ladder(720).unwrap(), [9, 80, 720]);
	assert_eq!(lookback_ladder(REGIME_BARS).unwrap(), [3, 9, 27]);
	// Cube root of 3 rounds to 1, a window too short to hold a return.
	assert!(lookback_ladder(3).is_err());
}

#[test]
fn adds_eleven_columns_and_fills_them_once_the_regime_window_is_full() {
	let df = add_context_features(frame(&[0.01; 80], 1.0, 1.0), REGIME_BARS).unwrap();
	let names = context_names(&df);
	assert_eq!(names.len(), 11);
	assert!(names.iter().all(|name| name.starts_with(CONTEXT_PREFIX)));
	for name in names
		.iter()
		.filter(|name| !name.contains("hour") && !name.contains("week"))
	{
		let values = column(&df, name);
		assert!(
			values[..REGIME_BARS - 1].iter().all(|value| value.is_nan()),
			"{name} warm-up"
		);
		assert!(
			values[REGIME_BARS - 1..].iter().all(|value| value.is_finite()),
			"{name} after warm-up"
		);
	}
}

#[test]
fn clock_columns_encode_hour_of_day_and_hour_of_week() {
	let df = add_context_features(frame(&[0.01; 80], 1.0, 1.0), REGIME_BARS).unwrap();
	let (hour_sin, hour_cos) = (column(&df, "ctx_hour_sin"), column(&df, "ctx_hour_cos"));
	let (week_sin, week_cos) = (column(&df, "ctx_week_sin"), column(&df, "ctx_week_cos"));
	// Row 0 is Monday 00:00: the start of both cycles.
	assert!(hour_sin[0].abs() < TOLERANCE && (hour_cos[0] - 1.0).abs() < TOLERANCE);
	assert!(week_sin[0].abs() < TOLERANCE && (week_cos[0] - 1.0).abs() < TOLERANCE);
	// Six hours later is a quarter of the day; 42 hours later is a quarter of the week.
	assert!((hour_sin[6] - 1.0).abs() < TOLERANCE);
	assert!((week_sin[42] - 1.0).abs() < TOLERANCE);
	// The day cycle repeats after 24 hours, the week cycle does not.
	assert!((hour_sin[5] - hour_sin[29]).abs() < TOLERANCE);
	assert!((week_sin[5] - week_sin[29]).abs() > TOLERANCE);
}

#[test]
fn volatility_column_reports_the_level_of_the_recent_window() {
	let mut step_sizes = vec![0.002; 60];
	step_sizes.extend(vec![0.02; 9]);
	let df = add_context_features(frame(&step_sizes, 1.0, 1.0), REGIME_BARS).unwrap();
	let recent = column(&df, "ctx_vol_2");
	// Row 59 ends a calm stretch; row 68 ends nine wild bars inside a mostly calm regime window.
	assert!(recent[68] > recent[59] * 1.5, "calm {} wild {}", recent[59], recent[68]);
	// Equal volatility in both windows gives ln(1 + 1).
	assert!((recent[59] - std::f64::consts::LN_2).abs() < 0.35);
}

#[test]
fn columns_do_not_depend_on_price_or_volume_units() {
	let step_sizes = vec![0.01; 80];
	let base = add_context_features(frame(&step_sizes, 1.0, 1.0), REGIME_BARS).unwrap();
	let scaled = add_context_features(frame(&step_sizes, 0.0003, 50_000.0), REGIME_BARS).unwrap();
	for name in context_names(&base) {
		let (left, right) = (column(&base, &name), column(&scaled, &name));
		for (a, b) in left.iter().zip(&right).skip(REGIME_BARS - 1) {
			assert!((a - b).abs() < 1e-6, "{name}: {a} vs {b}");
		}
	}
}

#[test]
fn rejects_a_market_that_never_moved() {
	let flat = frame(&[0.0; 80], 1.0, 1.0);
	assert!(add_context_features(flat, REGIME_BARS).is_err());
}
