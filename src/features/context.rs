//! Context inputs: what a self-normalised sequence window cannot show.
//!
//! Every other input column is z-scored inside its own sequence window, which removes its level:
//! a calm week and a wild week look the same to the model. These columns carry the level instead
//! and reach the model unscaled: the clock, and volatility, volume and trend measured against the
//! regime window (`features.stats_window_bars`). All of them are ratios or cyclic encodings, so
//! they mean the same on every symbol.

use crate::utils::error::{Result, VangaError};
use polars::prelude::*;

/// Columns with this prefix skip per-window normalisation and outlier capping.
pub const CONTEXT_PREFIX: &str = "ctx_";

const SECONDS_PER_HOUR: i64 = 3600;
const HOURS_PER_DAY: i64 = 24;
const HOURS_PER_WEEK: i64 = 168;
/// 1970-01-01 was a Thursday; this shift makes hour 0 of the week Monday 00:00 UTC.
const EPOCH_TO_MONDAY_HOURS: i64 = 72;
/// The shortest lookback must hold at least one return.
const MIN_LOOKBACK_BARS: usize = 2;

pub fn is_context_column(name: &str) -> bool {
	name.starts_with(CONTEXT_PREFIX)
}

/// Lookbacks in bars: the cube root of the regime window, that squared, and the regime window.
///
/// The steps are equal on a log scale, so no length is chosen by hand.
pub fn lookback_ladder(regime_bars: usize) -> Result<[usize; 3]> {
	let short = (regime_bars as f64).cbrt().round() as usize;
	let middle = (regime_bars as f64).powf(2.0 / 3.0).round() as usize;
	if short < MIN_LOOKBACK_BARS || short >= middle || middle >= regime_bars {
		return Err(VangaError::ConfigError(format!(
			"features.stats_window_bars = {regime_bars} is too short for context features: \
			 lookbacks {short}, {middle}, {regime_bars} must be increasing and start at {MIN_LOOKBACK_BARS} or more"
		)));
	}
	Ok([short, middle, regime_bars])
}

/// Running sums that give the sum over any window of bars in constant time.
struct PrefixSums(Vec<f64>);

impl PrefixSums {
	fn new(values: impl Iterator<Item = f64>) -> Self {
		let mut sums = vec![0.0];
		let mut total = 0.0;
		for value in values {
			total += value;
			sums.push(total);
		}
		Self(sums)
	}

	/// Sum of values `first..=last`.
	fn sum(&self, first: usize, last: usize) -> f64 {
		self.0[last + 1] - self.0[first]
	}
}

/// Population standard deviation of the returns inside the `bars`-long window ending at `end`.
fn window_volatility(returns: &PrefixSums, squared: &PrefixSums, end: usize, bars: usize) -> f64 {
	// A window of `bars` closes holds `bars - 1` returns; return `i` is the move into bar `i`.
	let first = end + 2 - bars;
	let count = (bars - 1) as f64;
	let mean = returns.sum(first, end) / count;
	(squared.sum(first, end) / count - mean * mean).max(0.0).sqrt()
}

/// Add the context columns. Rows before the regime window is full hold NaN.
pub fn add_context_features(mut df: DataFrame, regime_bars: usize) -> Result<DataFrame> {
	let ladder = lookback_ladder(regime_bars)?;
	let rows = crate::utils::market_data::extract_ohlcv_data(&df)?;
	if let Some(index) = rows.iter().position(|row| row.close <= 0.0) {
		return Err(VangaError::DataError(format!(
			"Context features need positive close prices, row {index} has {}",
			rows[index].close
		)));
	}

	let log_closes: Vec<f64> = rows.iter().map(|row| row.close.ln()).collect();
	// Index 0 has no previous bar; its zero is never read because every window starts at 1 or later.
	let log_returns: Vec<f64> = std::iter::once(0.0)
		.chain(log_closes.windows(2).map(|pair| pair[1] - pair[0]))
		.collect();
	let returns = PrefixSums::new(log_returns.iter().copied());
	let squared = PrefixSums::new(log_returns.iter().map(|value| value * value));
	let volumes = PrefixSums::new(rows.iter().map(|row| row.volume));

	let total = rows.len();
	let mut volatility = [vec![f64::NAN; total], vec![f64::NAN; total]];
	let mut volume = [vec![f64::NAN; total], vec![f64::NAN; total]];
	let mut trend = [vec![f64::NAN; total], vec![f64::NAN; total], vec![f64::NAN; total]];

	for end in (regime_bars - 1)..total {
		let regime_volatility = window_volatility(&returns, &squared, end, regime_bars);
		let regime_volume = volumes.sum(end + 1 - regime_bars, end) / regime_bars as f64;
		if regime_volatility <= 0.0 || regime_volume <= 0.0 {
			return Err(VangaError::DataError(format!(
				"Context features need a moving, traded market: row {end} has volatility {regime_volatility} \
				 and mean volume {regime_volume} over the last {regime_bars} bars"
			)));
		}
		for (slot, &bars) in ladder[..2].iter().enumerate() {
			// ln(1 + ratio) stays finite when the short window is flat or untraded.
			volatility[slot][end] = (window_volatility(&returns, &squared, end, bars) / regime_volatility).ln_1p();
			volume[slot][end] = (volumes.sum(end + 1 - bars, end) / bars as f64 / regime_volume).ln_1p();
		}
		for (slot, &bars) in ladder.iter().enumerate() {
			let moved = log_closes[end] - log_closes[end + 1 - bars];
			trend[slot][end] = moved / (regime_volatility * ((bars - 1) as f64).sqrt());
		}
	}

	let mut columns: Vec<(String, Vec<f64>)> = Vec::new();
	for (slot, values) in volatility.into_iter().enumerate() {
		columns.push((format!("{CONTEXT_PREFIX}vol_{}", slot + 1), values));
	}
	for (slot, values) in volume.into_iter().enumerate() {
		columns.push((format!("{CONTEXT_PREFIX}volume_{}", slot + 1), values));
	}
	for (slot, values) in trend.into_iter().enumerate() {
		columns.push((format!("{CONTEXT_PREFIX}trend_{}", slot + 1), values));
	}
	for (name, period) in [("hour", HOURS_PER_DAY), ("week", HOURS_PER_WEEK)] {
		let angles: Vec<f64> = rows
			.iter()
			.map(|row| {
				let hour_of_week =
					(row.timestamp.div_euclid(SECONDS_PER_HOUR) + EPOCH_TO_MONDAY_HOURS).rem_euclid(HOURS_PER_WEEK);
				std::f64::consts::TAU * hour_of_week.rem_euclid(period) as f64 / period as f64
			})
			.collect();
		columns.push((
			format!("{CONTEXT_PREFIX}{name}_sin"),
			angles.iter().map(|angle| angle.sin()).collect(),
		));
		columns.push((
			format!("{CONTEXT_PREFIX}{name}_cos"),
			angles.iter().map(|angle| angle.cos()).collect(),
		));
	}

	for (name, values) in columns {
		df.with_column(Series::new(name.as_str().into(), values))?;
	}
	Ok(df)
}
