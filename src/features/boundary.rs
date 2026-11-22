//! Label-boundary inputs: how far each label's class boundaries lie from the last close.
//!
//! Every label cuts a future price, or a future momentum, by boundaries that are fixed at call time from
//! the input sequence and the calibrated label parameters. Per-window normalisation removes the scale of
//! the sequence, so a model cannot see whether a boundary is half a normal move away or five of them.
//! These columns state it: each boundary as its distance from the last close of the input sequence, in
//! units of the sequence's own normal move over the horizon (the standard deviation of its bar returns
//! times the square root of the horizon length). The boundaries come from the functions that build the
//! labels. Only the input sequence is read (a regime-volatility label also reads its regime window), and
//! every value is a ratio of two moves, so the columns hold no price and mean the same on every symbol.

use crate::data::structures::MarketDataRow;
use crate::features::context::CONTEXT_PREFIX;
use crate::targets::calibration::{CalibratedParameters, DirectionParams, PriceLevelParams, StopLevelParams};
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;

/// A label with five classes has four boundaries.
const BOUNDARIES_PER_LABEL: usize = crate::config::model::NUM_CLASSES - 1;
/// The shortest input sequence that holds a bar return.
const MIN_SEQUENCE_BARS: usize = 2;

/// What the boundary inputs of a model are computed from.
pub struct BoundaryInputs<'a> {
	/// Calibrated label parameters. A label without parameters for a horizon gets no columns.
	pub calibrated: &'a CalibratedParameters,
	/// Bars in the model's input sequence.
	pub sequence_length: usize,
	pub horizons: &'a [String],
}

/// The calibrated labels of one horizon.
struct HorizonLabels<'a> {
	price: Option<&'a PriceLevelParams>,
	stop: Option<&'a StopLevelParams>,
	direction: Option<&'a DirectionParams>,
}

impl HorizonLabels<'_> {
	/// Column names, in the order `distances` returns the values.
	fn column_names(&self, horizon: &str) -> Vec<String> {
		let numbered = |label: &str| -> Vec<String> {
			(1..=BOUNDARIES_PER_LABEL)
				.map(|boundary| format!("{CONTEXT_PREFIX}bound_{label}_{horizon}_{boundary}"))
				.collect()
		};
		let mut names = Vec::new();
		if self.price.is_some() {
			names.extend(numbered("price"));
		}
		if self.stop.is_some() {
			names.extend(numbered("stop_dip"));
			names.extend(numbered("stop_bounce"));
			names.push(format!("{CONTEXT_PREFIX}bound_stop_side_{horizon}"));
		}
		if self.direction.is_some() {
			names.extend(numbered("direction"));
		}
		names
	}

	/// The first row whose input sequence is complete, and whose regime window is where a label has one.
	fn first_row(&self, sequence_length: usize) -> Result<usize> {
		let price_window = match self.price {
			Some(params) => crate::targets::price_levels::price_level_regime_window_bars(params)?,
			None => None,
		};
		let stop_window = match self.stop {
			Some(params) => crate::targets::stop_levels::stop_level_regime_window_bars(params)?,
			None => None,
		};
		Ok(price_window
			.into_iter()
			.chain(stop_window)
			.fold(sequence_length, usize::max)
			- 1)
	}

	/// Distances of the class boundaries from the last close of `sequence`, in normal moves of that
	/// sequence over the horizon. `history` ends at the same bar and starts at the first bar of the data.
	fn distances(
		&self,
		sequence: &[MarketDataRow],
		history: &[MarketDataRow],
		horizon_steps: usize,
	) -> Result<Vec<f64>> {
		let (last_close, bar_volatility) = crate::targets::price_levels::price_level_anchor_volatility(sequence)?;
		let normal_move = bar_volatility * (horizon_steps as f64).sqrt();
		let price_distance = |price: f64| (price / last_close - 1.0) / normal_move;
		let mut distances = Vec::new();
		if let Some(params) = self.price {
			let boundaries =
				crate::targets::price_levels::price_level_boundaries_with_history(sequence, Some(history), params)?;
			distances.extend(boundaries.boundaries.map(price_distance));
		}
		if let Some(params) = self.stop {
			// The label reads dips when the horizon ends at or above the reference price and bounces when it
			// ends below. Which of the two happens is not known at call time, so both sets are given, with
			// the reference price that decides between them.
			let dips = crate::targets::stop_levels::calculate_adverse_boundaries_with_history(
				sequence, history, true, params,
			)?;
			let bounces = crate::targets::stop_levels::calculate_adverse_boundaries_with_history(
				sequence, history, false, params,
			)?;
			distances.extend(dips.boundaries.map(price_distance));
			distances.extend(bounces.boundaries.map(price_distance));
			distances.push(price_distance(dips.reference_price));
		}
		if let Some(params) = self.direction {
			let boundaries = crate::targets::direction::direction_momentum_boundaries(sequence, params)?;
			distances.extend(boundaries.map(|momentum| momentum / normal_move));
		}
		Ok(distances)
	}
}

/// Add the boundary columns of every calibrated price-level, stop-level and direction label. Rows before
/// the first complete input sequence hold NaN.
pub fn add_boundary_features(mut df: DataFrame, inputs: &BoundaryInputs) -> Result<DataFrame> {
	if inputs.sequence_length < MIN_SEQUENCE_BARS {
		return Err(VangaError::ConfigError(format!(
			"Boundary inputs need an input sequence of at least {MIN_SEQUENCE_BARS} bars, got {}",
			inputs.sequence_length
		)));
	}
	let rows = crate::utils::market_data::extract_ohlcv_data(&df)?;
	let timeframe_minutes = crate::utils::parser::detect_timeframe_minutes(&df)?;
	for horizon in inputs.horizons {
		let horizon_steps = crate::utils::parser::parse_horizon_to_steps(horizon, timeframe_minutes)?;
		let labels = HorizonLabels {
			price: inputs.calibrated.get_price_levels(horizon),
			stop: inputs.calibrated.get_stop_levels(horizon),
			direction: inputs.calibrated.get_direction(horizon),
		};
		let names = labels.column_names(horizon);
		if names.is_empty() {
			return Err(VangaError::ConfigError(format!(
				"features.context.boundaries is on, but horizon {horizon} has no calibrated price-level, \
				 stop-level or direction label"
			)));
		}
		let mut columns = vec![vec![f64::NAN; rows.len()]; names.len()];
		for end in labels.first_row(inputs.sequence_length)?..rows.len() {
			let sequence = &rows[end + 1 - inputs.sequence_length..=end];
			let distances = labels
				.distances(sequence, &rows[..=end], horizon_steps)
				.map_err(|error| {
					VangaError::DataError(format!(
						"Boundary inputs of the input sequence ending at row {end}: {error}"
					))
				})?;
			for (column, distance) in columns.iter_mut().zip(distances) {
				column[end] = distance;
			}
		}
		for (name, values) in names.into_iter().zip(columns) {
			df.with_column(Series::new(name.as_str().into(), values))?;
		}
	}
	Ok(df)
}
