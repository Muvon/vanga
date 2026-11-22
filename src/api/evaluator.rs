//! Saved-model holdout scoring. No training, calibration, sampling, or RNG.
//!
//! Row indices refer to the original, chronologically ordered CSV (zero based).
//! An anchor is the last INPUT candle. Labels use the processed training frame;
//! replay OHLC outcomes use original market prices, never outlier replacements.
//! Confidence is the formatter's per-head confidence, not overall trade confidence.
//! Production vetoes remain in the scored population: no_signal=true, confidence=0
//! (the formatter discards its computed confidence together with the signal).
//! Confidence deciles are equal-population rank groups (ascending confidence,
//! chronological ties); ECE bins are equal-width max-probability intervals.
use crate::config::TrainingConfig;
use crate::data::{CryptoDataSchema, DataLoader, DataPreprocessor, SequenceGenerator};
use crate::model::multi_target::MultiTargetLSTMModel;
use crate::output::multi_target_parser::{MultiTargetParser, ParsedOutput};
use crate::output::prediction_types::{PriceBin, PriceLevelPrediction, StopLevelPrediction};
use crate::output::{OutputFormatter, PredictionResult};
use crate::targets::{MultiTargetConfig, TargetGenerator, TargetType};
use crate::utils::error::{Result, VangaError};
use crate::utils::market_data::extract_ohlcv_data;
use crate::utils::parser::{detect_timeframe_minutes, parse_horizon_to_steps};
use chrono::{DateTime, NaiveDate, NaiveDateTime, SecondsFormat, Utc};
use ndarray::{s, Array3};
use polars::prelude::*;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

const CLASSES: usize = crate::config::model::NUM_CLASSES;

pub struct EvaluationOptions {
	pub symbol: String,
	pub test_data: PathBuf,
	pub holdout_from: String,
	pub config: PathBuf,
	pub models_dir: PathBuf,
	pub horizons: Option<Vec<String>>,
	pub run_id: String,
	pub change: String,
	pub results_csv: Option<PathBuf>,
	pub eval_json: PathBuf,
}

#[derive(Serialize)]
pub(crate) struct Evaluation {
	pub meta: EvaluationMeta,
	pub heads: BTreeMap<String, HeadMetrics>,
}

#[derive(Serialize)]
pub(crate) struct EvaluationMeta {
	pub symbol: String,
	pub run_id: String,
	pub change: String,
	pub config_sha1: String,
	pub seed: u64,
	pub split_row: usize,
	pub split_ts: String,
	pub holdout_from: String,
	pub horizons: Vec<String>,
	pub sequence_length: usize,
	pub input_size: usize,
	pub n_rows: usize,
}

#[derive(Clone, Serialize)]
pub(crate) struct ScoredRow {
	pub ts: String,
	pub true_bin: usize,
	pub pred_bin: usize,
	pub probs: [f64; CLASSES],
	pub confidence: f64,
	pub no_signal: bool,
	pub close: f64,
	pub fut_max_high_rel: f64,
	pub fut_min_low_rel: f64,
	pub fut_close_rel: f64,
	pub band_lo: Option<f64>,
	pub band_hi: Option<f64>,
	pub ref_price: Option<f64>,
	pub stop_price: Option<f64>,
	pub stop_is_bullish: Option<bool>,
	/// [low, high] prices of all five classes of the head, in class order: the price head's zones or the stop
	/// head's adverse zones. A trade rule needs the whole distribution, not only the most likely class.
	pub class_bands: Option<[[f64; 2]; CLASSES]>,
	/// The stop head's adverse zones for the trade side opposite to `stop_is_bullish`, in class order.
	/// `stop_is_bullish` compares the expected price with the stop reference price, as the labels do; a trade
	/// entered at the current price can be on the other side and then needs that side's zones.
	pub opposite_side_class_bands: Option<[[f64; 2]; CLASSES]>,
}

struct ReconstructedActionPrices {
	price_levels: Option<PriceLevelPrediction>,
	stop_levels: Option<StopLevelPrediction>,
	opposite_side_stop_levels: Option<StopLevelPrediction>,
	ref_price: Option<f64>,
	stop_is_bullish: Option<bool>,
}

#[derive(Default)]
struct HeadActionPrices {
	band_lo: Option<f64>,
	band_hi: Option<f64>,
	ref_price: Option<f64>,
	stop_price: Option<f64>,
	stop_is_bullish: Option<bool>,
	class_bands: Option<[[f64; 2]; CLASSES]>,
	opposite_side_class_bands: Option<[[f64; 2]; CLASSES]>,
}

fn class_bands(bins: &HashMap<String, PriceBin>, names: &[&str; CLASSES]) -> Result<[[f64; 2]; CLASSES]> {
	let mut bands = [[0.0; 2]; CLASSES];
	for (band, name) in bands.iter_mut().zip(names) {
		*band = bins
			.get(*name)
			.ok_or_else(|| VangaError::model(format!("Formatter omitted the {name} bin")))?
			.price;
	}
	Ok(bands)
}

impl ReconstructedActionPrices {
	fn for_head(&self, kind: TargetType, pred_bin: usize) -> Result<HeadActionPrices> {
		const PRICE_BIN_NAMES: [&str; CLASSES] =
			["strong_down", "moderate_down", "neutral", "moderate_up", "strong_up"];
		const STOP_BIN_NAMES: [&str; CLASSES] = [
			"extreme_adverse",
			"high_adverse",
			"moderate_adverse",
			"low_adverse",
			"minimal_adverse",
		];

		match kind {
			TargetType::PriceLevel => {
				let Some(prediction) = self.price_levels.as_ref() else {
					return Ok(HeadActionPrices::default());
				};
				let bands = class_bands(&prediction.bins, &PRICE_BIN_NAMES)?;
				let band = bands
					.get(pred_bin)
					.ok_or_else(|| VangaError::model("Invalid predicted price bin"))?;
				Ok(HeadActionPrices {
					band_lo: Some(band[0]),
					band_hi: Some(band[1]),
					ref_price: self.ref_price,
					class_bands: Some(bands),
					..Default::default()
				})
			}
			TargetType::StopLevel => {
				let (Some(prediction), Some(is_bullish)) = (self.stop_levels.as_ref(), self.stop_is_bullish) else {
					return Ok(HeadActionPrices::default());
				};
				let bands = class_bands(&prediction.bins, &STOP_BIN_NAMES)?;
				let band = bands
					.get(pred_bin)
					.ok_or_else(|| VangaError::model("Invalid predicted stop bin"))?;
				Ok(HeadActionPrices {
					stop_price: Some(if is_bullish { band[0] } else { band[1] }),
					stop_is_bullish: Some(is_bullish),
					class_bands: Some(bands),
					opposite_side_class_bands: self
						.opposite_side_stop_levels
						.as_ref()
						.map(|opposite| class_bands(&opposite.bins, &STOP_BIN_NAMES))
						.transpose()?,
					..Default::default()
				})
			}
			_ => Ok(HeadActionPrices::default()),
		}
	}
}

#[derive(Serialize)]
pub(crate) struct ConfidenceDecile {
	pub decile: usize,
	pub n: usize,
	pub acc: Option<f64>,
	pub mean_confidence: Option<f64>,
}

#[derive(Serialize)]
pub(crate) struct HeadMetrics {
	pub n: usize,
	pub n_no_signal: usize,
	pub confusion: [[usize; CLASSES]; CLASSES],
	pub accuracy: f64,
	pub majority_class_baseline_accuracy: f64,
	pub macro_f1: f64,
	pub ordinal_mae: f64,
	pub side_accuracy: Option<f64>,
	pub side_count: usize,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub wide_enough_rate: Option<f64>,
	pub confidence_deciles: Vec<ConfidenceDecile>,
	pub top_decile_acc: f64,
	pub ece: f64,
	pub rows: Vec<ScoredRow>,
}

/// Fraction cuts use floor(fraction * original CSV rows). ISO cuts select the
/// first row at or after the requested instant. Naive ISO times mean UTC.
pub(crate) fn split_row(timestamps: &[DateTime<Utc>], cut: &str) -> Result<usize> {
	if let Ok(fraction) = cut.parse::<f64>() {
		if !fraction.is_finite() || !(0.0..1.0).contains(&fraction) {
			return Err(VangaError::config("--holdout-from fraction must be in [0, 1)"));
		}
		Ok((fraction * timestamps.len() as f64).floor() as usize)
	} else {
		let instant = parse_iso(cut)?;
		let split = timestamps.partition_point(|ts| *ts < instant);
		if split == timestamps.len() {
			return Err(VangaError::data("Holdout timestamp is after the last CSV row"));
		}
		Ok(split)
	}
}

fn parse_iso(value: &str) -> Result<DateTime<Utc>> {
	if let Ok(ts) = DateTime::parse_from_rfc3339(value) {
		return Ok(ts.with_timezone(&Utc));
	}
	for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
		if let Ok(ts) = NaiveDateTime::parse_from_str(value, format) {
			return Ok(ts.and_utc());
		}
	}
	if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
		if let Some(ts) = date.and_hms_opt(0, 0, 0) {
			return Ok(ts.and_utc());
		}
	}
	Err(VangaError::config(format!("Invalid ISO timestamp: {value}")))
}

fn timestamps(df: &DataFrame) -> Result<Vec<DateTime<Utc>>> {
	let column = df.column("timestamp")?;
	(0..df.height())
		.map(|i| {
			let (epoch, units) = match column.get(i)? {
				AnyValue::String(value) => return parse_iso(value),
				AnyValue::StringOwned(value) => return parse_iso(&value),
				AnyValue::Datetime(value, unit, _) => (
					value,
					match unit {
						TimeUnit::Milliseconds => 1_000,
						TimeUnit::Microseconds => 1_000_000,
						TimeUnit::Nanoseconds => 1_000_000_000,
					},
				),
				AnyValue::Int64(value) => {
					let magnitude = value.unsigned_abs();
					let units = if magnitude >= 100_000_000_000_000_000 {
						1_000_000_000
					} else if magnitude >= 100_000_000_000_000 {
						1_000_000
					} else if magnitude >= 100_000_000_000 {
						1_000
					} else {
						1
					};
					(value, units)
				}
				value => return Err(VangaError::data(format!("Unsupported timestamp at row {i}: {value:?}"))),
			};
			DateTime::from_timestamp(
				epoch.div_euclid(units),
				(epoch.rem_euclid(units) * (1_000_000_000 / units)) as u32,
			)
			.ok_or_else(|| VangaError::data(format!("Invalid timestamp at row {i}")))
		})
		.collect()
}

/// Map cleaned rows back to ORIGINAL row indices; never subtract a guessed
/// warmup length. Reject deletion of internal candles rather than alter horizons.
fn original_indices(original: &[DateTime<Utc>], processed: &[DateTime<Utc>]) -> Result<Vec<usize>> {
	let indices: Vec<usize> = processed
		.iter()
		.map(|ts| {
			original
				.binary_search(ts)
				.map_err(|_| VangaError::data("Processed timestamp is absent from original CSV"))
		})
		.collect::<Result<_>>()?;
	if indices.windows(2).any(|pair| pair[1] != pair[0] + 1) {
		return Err(VangaError::data(
			"Preprocessing removed internal candles; cannot preserve horizon alignment",
		));
	}
	Ok(indices)
}

/// Each returned index is a processed INPUT START. The embargo is measured in
/// original rows and includes BOTH sequence length and the largest scored horizon.
pub(crate) fn holdout_starts(
	original_rows: &[usize],
	split: usize,
	seq_len: usize,
	max_h: usize,
	head_h: usize,
	gaps: &[usize],
) -> Result<Vec<usize>> {
	let first_allowed = split
		.checked_add(seq_len)
		.and_then(|n| n.checked_add(max_h))
		.ok_or_else(|| VangaError::data("Holdout embargo overflow"))?;
	let window = seq_len
		.checked_add(head_h)
		.ok_or_else(|| VangaError::data("Window overflow"))?;
	if seq_len == 0 || head_h == 0 || head_h > max_h {
		return Err(VangaError::config("Invalid sequence length or horizon"));
	}
	Ok(original_rows
		.iter()
		.enumerate()
		.filter_map(|(start, &original)| {
			// A gap index g sits between raw rows g and g+1; the window's rows
			// original..original+window-1 are consecutive only if none is inside.
			let spans_gap = gaps.iter().any(|&g| g >= original && g + 1 < original + window);
			(original >= first_allowed && window <= original_rows.len() - start && !spans_gap).then_some(start)
		})
		.collect())
}

fn head_type(name: &str) -> Result<(TargetType, &str)> {
	let (target, horizon) = name
		.rsplit_once('_')
		.ok_or_else(|| VangaError::model(format!("Invalid saved target name: {name}")))?;
	let kind = match target {
		"price_level" | "price_levels" => TargetType::PriceLevel,
		"stop_level" | "stop_levels" => TargetType::StopLevel,
		"direction" => TargetType::Direction,
		"volatility" => TargetType::Volatility,
		"sentiment" => TargetType::Sentiment,
		"volume" => TargetType::Volume,
		_ => return Err(VangaError::model(format!("Unknown saved target: {name}"))),
	};
	Ok((kind, horizon))
}

/// Parse probabilities through the same per-horizon parser as production.
fn head_probabilities(parsed: &ParsedOutput, kind: TargetType) -> Result<[f64; CLASSES]> {
	let probs = match kind {
		TargetType::PriceLevel => parsed.price_levels.clone(),
		TargetType::StopLevel => parsed.stop_levels.clone(),
		TargetType::Direction => parsed.direction.as_ref().map(|p| {
			vec![
				p.dump_probability,
				p.down_probability,
				p.sideways_probability,
				p.up_probability,
				p.pump_probability,
			]
		}),
		TargetType::Volatility => parsed.volatility.as_ref().map(|p| {
			vec![
				p.very_low_probability,
				p.low_probability,
				p.medium_probability,
				p.high_probability,
				p.very_high_probability,
			]
		}),
		TargetType::Sentiment => parsed.sentiment.as_ref().map(|p| {
			vec![
				p.very_bearish_probability,
				p.bearish_probability,
				p.neutral_probability,
				p.bullish_probability,
				p.very_bullish_probability,
			]
		}),
		TargetType::Volume => parsed.volume.as_ref().map(|p| {
			vec![
				p.very_low_probability,
				p.low_probability,
				p.medium_probability,
				p.high_probability,
				p.very_high_probability,
			]
		}),
	}
	.ok_or_else(|| VangaError::model("Saved head is missing from parsed prediction"))?;
	probs
		.try_into()
		.map_err(|_| VangaError::model("Expected five probabilities"))
}

pub(crate) fn head_confidence(result: Option<&PredictionResult>, kind: TargetType) -> Result<f64> {
	// An empty formatter result is a production veto, not a scoring failure.
	// Its internal confidence is unavailable through the production API.
	let Some(result) = result else {
		return Ok(0.0);
	};
	// Production cannot serve a stop without the price head that picks its side;
	// that is a veto of the stop head, not a scoring failure.
	if kind == TargetType::StopLevel && result.price_levels.is_none() {
		return Ok(0.0);
	}
	let confidence = match kind {
		TargetType::PriceLevel => result.price_levels.as_ref().map(|p| p.confidence),
		TargetType::StopLevel => result.stop_levels.as_ref().map(|p| p.confidence),
		TargetType::Direction => result.direction.as_ref().map(|p| p.confidence),
		TargetType::Volatility => result.volatility.as_ref().map(|p| p.confidence),
		TargetType::Sentiment => result.sentiment.as_ref().map(|p| p.confidence),
		TargetType::Volume => result.volume.as_ref().map(|p| p.confidence),
	}
	.ok_or_else(|| VangaError::model("Formatter omitted a saved head"))?;
	if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
		return Err(VangaError::model("Formatter returned invalid confidence"));
	}
	Ok(confidence)
}

fn reconstruct_action_prices_without_veto(
	formatter: &OutputFormatter,
	parsed: &ParsedOutput,
	current_price: f64,
	horizon: &str,
	calibrated: &crate::targets::calibration::CalibratedParameters,
	sequence_ohlcv: &[crate::data::structures::MarketDataRow],
	history_ohlcv: &[crate::data::structures::MarketDataRow],
) -> Result<ReconstructedActionPrices> {
	let price_params = calibrated.get_price_levels(horizon);
	let price_levels = parsed
		.price_levels
		.as_ref()
		.map(|probabilities| {
			formatter.create_price_level_prediction(
				probabilities,
				current_price,
				horizon,
				price_params.map(|params| params.bandwidth),
				price_params.map(|params| params.percentiles),
			)
		})
		.transpose()?;
	let ref_price = price_levels.as_ref().map(|_| current_price);

	let stop_params = calibrated.get_stop_levels(horizon);
	let formatter_reference_price = stop_params
		.map(|params| crate::output::formatter::stop_level_reference_price(sequence_ohlcv, history_ohlcv, params))
		.transpose()?;
	let stop_is_bullish = price_levels.as_ref().and_then(|prediction| {
		formatter_reference_price
			.and_then(|reference_price| crate::output::formatter::infer_stop_is_bullish(prediction, reference_price))
	});
	let stop_levels_for = |is_bullish: bool| {
		parsed
			.stop_levels
			.as_ref()
			.map(|probabilities| {
				formatter.create_stop_level_prediction(
					probabilities,
					current_price,
					horizon,
					stop_params
						.map(|params| params.bandwidth)
						.or_else(|| price_params.map(|params| params.bandwidth)),
					stop_params
						.map(|params| params.percentiles)
						.or_else(|| price_params.map(|params| params.percentiles)),
					Some(is_bullish),
				)
			})
			.transpose()
	};
	let (stop_levels, opposite_side_stop_levels) = match stop_is_bullish {
		Some(is_bullish) => (stop_levels_for(is_bullish)?, stop_levels_for(!is_bullish)?),
		None => (None, None),
	};

	Ok(ReconstructedActionPrices {
		price_levels,
		stop_levels,
		opposite_side_stop_levels,
		ref_price,
		stop_is_bullish,
	})
}

pub(crate) fn validate_evaluator_stop_history(
	calibrated: &crate::targets::calibration::CalibratedParameters,
	horizon: &str,
	available_bars: usize,
) -> Result<()> {
	if let Some(window_bars) = calibrated
		.get_stop_levels(horizon)
		.map(crate::targets::stop_levels::stop_level_regime_window_bars)
		.transpose()?
		.flatten()
	{
		if available_bars < window_bars {
			return Err(VangaError::data(format!(
				"Insufficient stop-level regime history for evaluator anchor: {} bars available, {} required",
				available_bars, window_bars
			)));
		}
	}
	Ok(())
}

pub(crate) fn validate_evaluator_price_history(
	calibrated: &crate::targets::calibration::CalibratedParameters,
	horizon: &str,
	available_bars: usize,
) -> Result<()> {
	if let Some(window_bars) = calibrated
		.get_price_levels(horizon)
		.map(crate::targets::price_levels::price_level_regime_window_bars)
		.transpose()?
		.flatten()
	{
		if available_bars < window_bars {
			return Err(VangaError::data(format!(
				"Insufficient price-level regime history for evaluator anchor: {} bars available, {} required",
				available_bars, window_bars
			)));
		}
	}
	Ok(())
}

/// Compare the complete serialized feature/data sections, including parameters
/// that preserve the column count but change the feature values or ordering.
pub(crate) fn validate_preprocessing_config(saved: &TrainingConfig, requested: &TrainingConfig) -> Result<()> {
	let saved = serde_json::to_value(saved).map_err(|e| VangaError::SerializationError(e.to_string()))?;
	let requested = serde_json::to_value(requested).map_err(|e| VangaError::SerializationError(e.to_string()))?;
	let mut differences = Vec::new();
	differing_config_keys(&saved["features"], &requested["features"], "features", &mut differences);
	// Only the [data] keys that shape the feature frame; the trainer rewrites
	// sequence_overlap at runtime and the rest are sampling/split knobs.
	for key in ["outlier_handling", "normalization", "feature_selection"] {
		differing_config_keys(
			&saved["data"][key],
			&requested["data"][key],
			&format!("data.{key}"),
			&mut differences,
		);
	}
	if !differences.is_empty() {
		let message = format!(
			"--config differs from saved preprocessing configuration:\n{}",
			differences.join("\n")
		);
		log::error!("❌ {}", message);
		return Err(VangaError::config(message));
	}
	Ok(())
}

fn differing_config_keys(
	saved: &serde_json::Value,
	requested: &serde_json::Value,
	path: &str,
	differences: &mut Vec<String>,
) {
	if let (Some(saved), Some(requested)) = (saved.as_object(), requested.as_object()) {
		let keys: BTreeSet<_> = saved.keys().chain(requested.keys()).collect();
		for key in keys {
			differing_config_keys(
				saved.get(key).unwrap_or(&serde_json::Value::Null),
				requested.get(key).unwrap_or(&serde_json::Value::Null),
				&format!("{path}.{key}"),
				differences,
			);
		}
	} else if saved != requested {
		differences.push(format!("{path}: saved={saved}, --config={requested}"));
	}
}

/// Labels detect cadence independently on the processed frame. Replay must use
/// the same cadence. Whole missing candles (exchange outages) are returned as
/// gap indices so their windows can be skipped; anything else is an error.
pub(crate) fn validate_cadence(
	raw_timestamps: &[DateTime<Utc>],
	raw_minutes: usize,
	processed_minutes: usize,
) -> Result<Vec<usize>> {
	if raw_minutes != processed_minutes {
		return Err(VangaError::data(format!(
			"Timeframe mismatch: raw={raw_minutes} minutes, processed={processed_minutes} minutes"
		)));
	}
	let expected = i64::try_from(raw_minutes)
		.ok()
		.and_then(chrono::Duration::try_minutes)
		.filter(|duration| *duration > chrono::Duration::zero())
		.ok_or_else(|| VangaError::data("Invalid detected timeframe"))?;
	let mut gaps = Vec::new();
	for (index, pair) in raw_timestamps.windows(2).enumerate() {
		let gap = pair[1] - pair[0];
		if gap == expected {
			continue;
		}
		let whole_candles = gap > expected && (gap.num_milliseconds() % expected.num_milliseconds()) == 0;
		if !whole_candles {
			return Err(VangaError::data(format!(
				"Irregular CSV at rows {}..{} ({} -> {}): gap={}, detected timeframe={} minutes",
				index,
				index + 1,
				iso(&pair[0]),
				iso(&pair[1]),
				gap,
				raw_minutes
			)));
		}
		log::warn!(
			"⚠️ Missing candles at rows {}..{} ({} -> {}): windows spanning it are skipped",
			index,
			index + 1,
			iso(&pair[0]),
			iso(&pair[1])
		);
		gaps.push(index);
	}
	Ok(gaps)
}

pub async fn evaluate(options: EvaluationOptions) -> Result<()> {
	let config_bytes = std::fs::read(&options.config)?;
	let config = TrainingConfig::default().with_config_from_file(&options.config)?;
	let base = options.models_dir.join(&options.symbol);
	// Require the modern saved contract instead of taking legacy loader defaults.
	let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(base.with_extension("meta"))?)
		.map_err(|e| VangaError::SerializationError(e.to_string()))?;
	for field in [
		"target_names",
		"input_size",
		"trained_horizons",
		"training_config",
		"calibrated_parameters",
	] {
		if metadata.get(field).is_none_or(serde_json::Value::is_null) {
			return Err(VangaError::model(format!("Saved metadata missing {field}")));
		}
	}
	let model = MultiTargetLSTMModel::load(&base)?;
	let saved = model
		.get_training_config()
		.ok_or_else(|| VangaError::model("Missing training config"))?;
	validate_preprocessing_config(saved, &config)?;
	let calibrated = model
		.get_calibrated_parameters()
		.ok_or_else(|| VangaError::model("Missing saved calibration"))?;
	for params in calibrated.stop_levels.values() {
		crate::targets::stop_levels::validate_stop_level_regime_window(params, saved.features.stats_window_bars)?;
	}
	for params in calibrated.price_levels.values() {
		crate::targets::price_levels::validate_price_level_regime_window(params, saved.features.stats_window_bars)?;
	}
	if model.models.is_empty() {
		return Err(VangaError::model("Saved model has no heads"));
	}
	// The production loader permits RF fallback. Evaluation must fail on missing
	// hybrid artifacts instead of quietly scoring a different model.
	for (i, head) in model.models.iter().enumerate() {
		let rf_meta = PathBuf::from(format!("{}_{}.smartcore.meta", base.display(), i));
		if (saved.model.xgboost.enabled || rf_meta.exists())
			&& (!rf_meta.is_file()
				|| !PathBuf::from(format!("{}_{}.smartcore.model", base.display(), i)).is_file()
				|| head.xgboost_model.as_ref().is_none_or(|rf| !rf.is_trained()))
		{
			return Err(VangaError::model(format!(
				"Missing or invalid saved hybrid model for {}_{}",
				base.display(),
				i
			)));
		}
	}
	let sequence_length = model.models[0].config.sequence_length;
	if model
		.models
		.iter()
		.any(|head| head.config.sequence_length != sequence_length)
	{
		return Err(VangaError::model("Saved heads have inconsistent sequence lengths"));
	}
	let horizons = options
		.horizons
		.clone()
		.unwrap_or_else(|| model.get_trained_horizons().to_vec());
	let mut unique = BTreeSet::new();
	for horizon in &horizons {
		if !model.get_trained_horizons().contains(horizon) || !unique.insert(horizon) {
			return Err(VangaError::config(format!(
				"Untrained or duplicate horizon {horizon}; saved: {:?}",
				model.get_trained_horizons()
			)));
		}
	}
	if horizons.is_empty() || sequence_length == 0 {
		return Err(VangaError::model("Saved horizons and sequence length must be nonempty"));
	}
	let raw = DataLoader::new().load_csv(&options.test_data).await?;
	CryptoDataSchema::validate(&raw)?;
	let original_ts = timestamps(&raw)?;
	if original_ts.windows(2).any(|pair| pair[0] >= pair[1]) {
		return Err(VangaError::data("CSV timestamps must be unique and chronological"));
	}
	// Ensure loader cleanup did not change what a CSV ROW fraction means.
	let mut csv = csv::Reader::from_path(&options.test_data).map_err(csv_error)?;
	let timestamp_column = csv
		.headers()
		.map_err(csv_error)?
		.iter()
		.position(|name| name.eq_ignore_ascii_case("timestamp"))
		.ok_or_else(|| VangaError::data("CSV is missing timestamp column"))?;
	let mut csv_rows = 0;
	for record in csv.records() {
		let record = record.map_err(csv_error)?;
		let value = record
			.get(timestamp_column)
			.ok_or_else(|| VangaError::data("CSV row is missing timestamp"))?;
		let same_row = if let Ok(epoch) = value.parse::<i64>() {
			raw.column("timestamp")?.i64()?.get(csv_rows) == Some(epoch)
		} else {
			original_ts.get(csv_rows) == Some(&parse_iso(value)?)
		};
		if !same_row {
			return Err(VangaError::data(format!(
				"CSV row {csv_rows} changed during loading; input must be chronological with valid timestamps"
			)));
		}
		csv_rows += 1;
	}
	if csv_rows != raw.height() {
		return Err(VangaError::data(format!(
			"CSV has {csv_rows} rows but loader retained {}; cannot preserve row-based split",
			raw.height()
		)));
	}
	let split = split_row(&original_ts, &options.holdout_from)?;
	let timeframe = detect_timeframe_minutes(&raw)?;
	let steps: Vec<usize> = horizons
		.iter()
		.map(|h| parse_horizon_to_steps(h, timeframe))
		.collect::<Result<_>>()?;
	let trained_steps: Vec<usize> = model
		.get_trained_horizons()
		.iter()
		.map(|h| parse_horizon_to_steps(h, timeframe))
		.collect::<Result<_>>()?;
	let max_h = *trained_steps
		.iter()
		.max()
		.ok_or_else(|| VangaError::model("No trained horizons"))?;
	let raw_ohlcv = extract_ohlcv_data(&raw)?;
	let processed = DataPreprocessor::new()
		.process_features_only(
			raw,
			&saved.data,
			Some(&saved.features),
			crate::data::boundary_inputs(saved, Some(calibrated)).as_ref(),
		)
		.await?;
	let processed_timeframe = detect_timeframe_minutes(&processed)?;
	let gaps = validate_cadence(&original_ts, timeframe, processed_timeframe)?;
	let processed_ts = timestamps(&processed)?;
	let row_indices = original_indices(&original_ts, &processed_ts)?;
	let feature_names: Vec<String> = processed
		.get_column_names()
		.iter()
		.filter(|name| name.as_str() != "timestamp")
		.map(|name| name.to_string())
		.collect();
	if feature_names.len() != model.get_input_size() {
		return Err(VangaError::model(format!(
			"Feature count mismatch: saved input_size={}, computed feature count={}",
			model.get_input_size(),
			feature_names.len()
		)));
	}
	for name in &feature_names {
		if processed
			.column(name)?
			.f64()?
			.into_iter()
			.any(|value| value.is_none_or(|n| !n.is_finite()))
		{
			return Err(VangaError::data(format!(
				"Non-finite or missing processed feature: {name}"
			)));
		}
	}
	let mut processed_ohlcv = extract_ohlcv_data(&processed)?;
	for (row, ts) in processed_ohlcv.iter_mut().zip(&processed_ts) {
		row.timestamp = ts.timestamp();
	}
	let mut report = Evaluation {
		meta: EvaluationMeta {
			symbol: options.symbol.clone(),
			run_id: options.run_id,
			change: options.change,
			config_sha1: sha1_hex(&config_bytes),
			seed: config.training.seed,
			split_row: split,
			split_ts: iso(&original_ts[split]),
			holdout_from: options.holdout_from,
			horizons: horizons.clone(),
			sequence_length,
			input_size: model.get_input_size(),
			n_rows: original_ts.len(),
		},
		heads: BTreeMap::new(),
	};
	println!(
		"📊 Split row: {} | split timestamp: {} | embargo input start >= {}",
		split,
		report.meta.split_ts,
		split + sequence_length + max_h
	);
	let generator = SequenceGenerator::new();
	for (horizon, head_h) in horizons.iter().zip(steps) {
		let starts = holdout_starts(&row_indices, split, sequence_length, max_h, head_h, &gaps)?;
		let gap_skipped =
			holdout_starts(&row_indices, split, sequence_length, max_h, head_h, &[])?.len() - starts.len();
		if starts.is_empty() {
			return Err(VangaError::data(format!(
				"No holdout anchors for {horizon} after embargo"
			)));
		}
		validate_evaluator_stop_history(calibrated, horizon, starts[0] + sequence_length)?;
		validate_evaluator_price_history(calibrated, horizon, starts[0] + sequence_length)?;
		let head_names: Vec<String> = model
			.get_target_names()
			.iter()
			.filter(|name| name.ends_with(&format!("_{horizon}")))
			.cloned()
			.collect();
		if head_names.is_empty() {
			return Err(VangaError::model(format!("No saved heads for {horizon}")));
		}
		let kinds: Vec<TargetType> = head_names
			.iter()
			.map(|name| head_type(name).map(|(kind, _)| kind))
			.collect::<Result<_>>()?;
		if kinds.iter().collect::<BTreeSet<_>>().len() != kinds.len() {
			return Err(VangaError::model(format!("Duplicate target types for {horizon}")));
		}
		let mut label_config = MultiTargetConfig::from_training_config(saved);
		label_config.horizons = vec![horizon.clone()];
		label_config.price_level.enabled = kinds.contains(&TargetType::PriceLevel);
		label_config.stop_level.enabled = kinds.contains(&TargetType::StopLevel);
		label_config.direction.enabled = kinds.contains(&TargetType::Direction);
		label_config.volatility.enabled = kinds.contains(&TargetType::Volatility);
		label_config.sentiment.enabled = kinds.contains(&TargetType::Sentiment);
		label_config.volume.enabled = kinds.contains(&TargetType::Volume);
		let labels = TargetGenerator::new(label_config)
			.generate_all_targets_with_calibrated_params(&processed, &starts, sequence_length, Some(calibrated))
			.await?;
		let parser = MultiTargetParser::with_target_names(&head_names);
		let mut head_rows: Vec<Vec<ScoredRow>> = head_names.iter().map(|_| Vec::with_capacity(starts.len())).collect();
		println!(
			"⏱️ {horizon}: first scored anchor {} | {} anchors per head | {} skipped for missing candles",
			iso(&processed_ts[starts[0] + sequence_length - 1]),
			starts.len(),
			gap_skipped
		);
		for (anchor_index, &start) in starts.iter().enumerate() {
			// Production serves exactly one sequence. MoH routing normalizes
			// across the batch, so co-batching anchors would change predictions.
			let mut sequences = Array3::zeros((1, sequence_length, feature_names.len()));
			let window = processed.slice(start as i64, sequence_length);
			let normalized = generator.normalize_sequence_window(&window)?;
			let matrix = generator.extract_feature_matrix(&normalized, &feature_names)?;
			sequences.slice_mut(s![0, .., ..]).assign(&matrix);
			// Uses LSTMModel::predict -> forward(false), RF templates, production blend.
			let predictions = model.predict(&sequences).await?;
			if predictions.iter().any(|p| !p.is_finite() || *p < 0.0) {
				return Err(VangaError::model("Invalid model probabilities"));
			}
			let end = start + sequence_length;
			let anchor = row_indices[end - 1];
			let close = raw_ohlcv[anchor].close;
			let future = &raw_ohlcv[anchor + 1..=anchor + head_h];
			if !close.is_finite()
				|| close <= 0.0
				|| future
					.iter()
					.any(|row| !row.high.is_finite() || !row.low.is_finite() || !row.close.is_finite())
			{
				return Err(VangaError::data(format!("Invalid replay OHLC at anchor {anchor}")));
			}
			let formatter = OutputFormatter::new(Default::default())
				.with_target_names(model.get_target_names())
				.with_sequence_ohlcv(processed_ohlcv[start..end].to_vec())
				.with_regime_history(processed_ohlcv[..end].to_vec())
				.with_stats_window_bars(saved.features.stats_window_bars)
				.with_metadata(feature_names.len(), sequence_length)
				.with_min_confidence(0.0)
				.with_calibrated_parameters(calibrated.clone());
			let formatted = formatter.format_predictions_with_targets(
				&predictions,
				&options.symbol,
				horizon,
				processed_ohlcv[end - 1].close,
				None,
				Some(model.get_target_names()),
			)?;
			let formatted = formatted.first();
			let mut local = Vec::with_capacity(head_names.len() * CLASSES);
			for (index, name) in model.get_target_names().iter().enumerate() {
				if head_names.contains(name) {
					local.extend(
						predictions
							.slice(s![0, index * CLASSES..(index + 1) * CLASSES])
							.iter()
							.copied(),
					);
				}
			}
			let parsed = parser.parse_output(ndarray::ArrayView1::from(&local))?;
			let action_prices = reconstruct_action_prices_without_veto(
				&formatter,
				&parsed,
				processed_ohlcv[end - 1].close,
				horizon,
				calibrated,
				&processed_ohlcv[start..end],
				&processed_ohlcv[..end],
			)?;
			for (head_index, kind) in kinds.iter().enumerate() {
				let probs = head_probabilities(&parsed, *kind)?;
				// Production reconstruction resolves argmax ties to the last class.
				let pred_bin = probs
					.iter()
					.enumerate()
					.max_by(|a, b| a.1.total_cmp(b.1))
					.map(|(index, _)| index)
					.ok_or_else(|| VangaError::model("Empty probabilities"))?;
				let action = action_prices.for_head(*kind, pred_bin)?;
				let true_bin = *labels
					.get_targets(horizon, *kind)
					.and_then(|values| values.get(anchor_index))
					.ok_or_else(|| VangaError::data("Missing aligned label"))?;
				if !(0..CLASSES as i32).contains(&true_bin) {
					return Err(VangaError::data(format!(
						"Invalid true bin {true_bin} for {} at anchor {anchor}",
						head_names[head_index]
					)));
				}
				head_rows[head_index].push(ScoredRow {
					ts: iso(&original_ts[anchor]),
					true_bin: true_bin as usize,
					pred_bin,
					probs,
					confidence: head_confidence(formatted, *kind)?,
					no_signal: formatted.is_none(),
					close,
					fut_max_high_rel: future.iter().map(|r| r.high).fold(f64::NEG_INFINITY, f64::max) / close - 1.0,
					fut_min_low_rel: future.iter().map(|r| r.low).fold(f64::INFINITY, f64::min) / close - 1.0,
					fut_close_rel: raw_ohlcv[anchor + head_h].close / close - 1.0,
					band_lo: action.band_lo,
					band_hi: action.band_hi,
					ref_price: action.ref_price,
					stop_price: action.stop_price,
					stop_is_bullish: action.stop_is_bullish,
					class_bands: action.class_bands,
					opposite_side_class_bands: action.opposite_side_class_bands,
				});
			}
		}
		for ((name, kind), rows) in head_names.into_iter().zip(kinds).zip(head_rows) {
			let metrics = compute_metrics(rows, kind == TargetType::StopLevel)?;
			report.heads.insert(name, metrics);
		}
	}
	print_report(&report);
	let json = serde_json::to_vec_pretty(&report).map_err(|e| VangaError::SerializationError(e.to_string()))?;
	std::fs::write(&options.eval_json, json)?;
	if let Some(path) = options.results_csv {
		append_results(&path, &report)?;
	}
	log::info!("✅ Saved holdout evaluation to {}", options.eval_json.display());
	Ok(())
}

fn iso(ts: &DateTime<Utc>) -> String {
	ts.to_rfc3339_opts(SecondsFormat::AutoSi, true)
}

pub(crate) fn compute_metrics(rows: Vec<ScoredRow>, stop: bool) -> Result<HeadMetrics> {
	let n = rows.len();
	if n == 0 {
		return Err(VangaError::data("Cannot score an empty holdout"));
	}
	let mut confusion = [[0usize; CLASSES]; CLASSES];
	let mut ordinal = 0usize;
	let mut side_count = 0usize;
	let mut side_correct = 0usize;
	let mut wide = 0usize;
	let mut bins = [(0usize, 0usize, 0.0f64); 10];
	for row in &rows {
		if row.true_bin >= CLASSES
			|| row.pred_bin >= CLASSES
			|| !row.confidence.is_finite()
			|| !(0.0..=1.0).contains(&row.confidence)
			|| row.probs.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
			|| (row.probs.iter().sum::<f64>() - 1.0).abs() > 1e-6
		{
			return Err(VangaError::data("Invalid classification row"));
		}
		confusion[row.true_bin][row.pred_bin] += 1;
		ordinal += row.true_bin.abs_diff(row.pred_bin);
		if row.true_bin != 2 {
			side_count += 1;
			side_correct += usize::from(row.true_bin.cmp(&2) == row.pred_bin.cmp(&2));
		}
		// Stop class 0 is the deepest adverse move: the stop holds when the predicted class is at least as deep.
		wide += usize::from(row.pred_bin <= row.true_bin);
		let max_prob = row.probs.iter().copied().fold(0.0, f64::max);
		let bin = &mut bins[((max_prob * 10.0).floor() as usize).min(9)];
		bin.0 += 1;
		bin.1 += usize::from(row.true_bin == row.pred_bin);
		bin.2 += max_prob;
	}
	let correct: usize = (0..CLASSES).map(|i| confusion[i][i]).sum();
	let majority = confusion.iter().map(|row| row.iter().sum::<usize>()).max().unwrap_or(0);
	// All five classes participate; an absent class has F1=0.
	let macro_f1 = (0..CLASSES)
		.map(|i| {
			let actual = confusion[i].iter().sum::<usize>();
			let predicted: usize = confusion.iter().map(|row| row[i]).sum();
			if actual + predicted == 0 {
				0.0
			} else {
				2.0 * confusion[i][i] as f64 / (actual + predicted) as f64
			}
		})
		.sum::<f64>()
		/ CLASSES as f64;
	let ece = bins
		.iter()
		.filter(|bin| bin.0 > 0)
		.map(|bin| (bin.1 as f64 - bin.2).abs() / n as f64)
		.sum();
	let mut ranked: Vec<usize> = (0..n).collect();
	ranked.sort_by(|a, b| rows[*a].confidence.total_cmp(&rows[*b].confidence).then(a.cmp(b)));
	let confidence_deciles: Vec<ConfidenceDecile> = (0..10)
		.map(|d| {
			let group = &ranked[d * n / 10..(d + 1) * n / 10];
			let count = group.len();
			ConfidenceDecile {
				decile: d + 1,
				n: count,
				acc: (count > 0).then(|| {
					group.iter().filter(|&&i| rows[i].true_bin == rows[i].pred_bin).count() as f64 / count as f64
				}),
				mean_confidence: (count > 0)
					.then(|| group.iter().map(|&i| rows[i].confidence).sum::<f64>() / count as f64),
			}
		})
		.collect();
	let top_decile_acc = confidence_deciles[9]
		.acc
		.ok_or_else(|| VangaError::data("Empty top decile"))?;
	Ok(HeadMetrics {
		n,
		n_no_signal: rows.iter().filter(|row| row.no_signal).count(),
		confusion,
		accuracy: correct as f64 / n as f64,
		majority_class_baseline_accuracy: majority as f64 / n as f64,
		macro_f1,
		ordinal_mae: ordinal as f64 / n as f64,
		side_accuracy: (side_count > 0).then(|| side_correct as f64 / side_count as f64),
		side_count,
		wide_enough_rate: stop.then_some(wide as f64 / n as f64),
		confidence_deciles,
		top_decile_acc,
		ece,
		rows,
	})
}

fn print_report(report: &Evaluation) {
	println!("\nhead                       n no-signal      acc majority macro-F1 ord-MAE side-acc side-n wide-rate top-decile ECE");
	for (name, m) in &report.heads {
		println!(
			"{name:25} {:6} {:9} {:8.4} {:8.4} {:8.4} {:7.4} {:>8} {:6} {:>9} {:10.4} {:.4}",
			m.n,
			m.n_no_signal,
			m.accuracy,
			m.majority_class_baseline_accuracy,
			m.macro_f1,
			m.ordinal_mae,
			optional_number(m.side_accuracy),
			m.side_count,
			optional_number(m.wide_enough_rate),
			m.top_decile_acc,
			m.ece
		);
	}
	for (name, m) in &report.heads {
		println!("\n{name}: confusion (rows=true, cols=pred)");
		for row in m.confusion {
			println!("{:8} {:8} {:8} {:8} {:8}", row[0], row[1], row[2], row[3], row[4]);
		}
		println!("confidence decile       n accuracy mean-confidence");
		for d in &m.confidence_deciles {
			println!(
				"{:17} {:7} {:>8} {:>15}",
				d.decile,
				d.n,
				optional_number(d.acc),
				optional_number(d.mean_confidence)
			);
		}
	}
}

fn optional_number(value: Option<f64>) -> String {
	value.map(|n| format!("{n:.4}")).unwrap_or_default()
}

fn csv_error(error: csv::Error) -> VangaError {
	VangaError::data(error.to_string())
}

const RESULTS_HEADER: [&str; 19] = [
	"run_id",
	"date",
	"symbol",
	"target",
	"horizon",
	"seed",
	"config_sha",
	"change",
	"n_holdout",
	"acc",
	"majority_acc",
	"macro_f1",
	"ordinal_mae",
	"side_acc",
	"stop_wide_rate",
	"top_decile_acc",
	"ece",
	"train_val_acc",
	"wall_clock_s",
];

pub(crate) fn append_results(path: &Path, report: &Evaluation) -> Result<()> {
	let has_header = path.exists() && std::fs::metadata(path)?.len() > 0;
	if has_header {
		let mut reader = csv::Reader::from_path(path).map_err(csv_error)?;
		if reader.headers().map_err(csv_error)?.iter().ne(RESULTS_HEADER) {
			return Err(VangaError::data(
				"Existing results.csv header does not match evaluation schema",
			));
		}
	}
	let file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
	let mut writer = csv::Writer::from_writer(file);
	if !has_header {
		writer.write_record(RESULTS_HEADER).map_err(csv_error)?;
	}
	let date = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
	let meta = &report.meta;
	for (name, m) in &report.heads {
		let (target, horizon) = name
			.rsplit_once('_')
			.ok_or_else(|| VangaError::model("Invalid head name"))?;
		writer
			.write_record([
				meta.run_id.clone(),
				date.clone(),
				meta.symbol.clone(),
				target.to_string(),
				horizon.to_string(),
				meta.seed.to_string(),
				meta.config_sha1.clone(),
				meta.change.clone(),
				m.n.to_string(),
				m.accuracy.to_string(),
				m.majority_class_baseline_accuracy.to_string(),
				m.macro_f1.to_string(),
				m.ordinal_mae.to_string(),
				m.side_accuracy.map(|n| n.to_string()).unwrap_or_default(),
				m.wide_enough_rate.map(|n| n.to_string()).unwrap_or_default(),
				m.top_decile_acc.to_string(),
				m.ece.to_string(),
				String::new(),
				String::new(),
			])
			.map_err(csv_error)?;
	}
	writer.flush()?;
	Ok(())
}

/// SHA-1 of the exact configuration bytes, for experiment identity only.
/// Implements the standard 512-bit block / 80-round digest without dependencies.
/// This is not used for authentication or any cryptographic security decision.
pub(crate) fn sha1_hex(bytes: &[u8]) -> String {
	let mut padded = bytes.to_vec();
	padded.push(0x80);
	while padded.len() % 64 != 56 {
		padded.push(0);
	}
	padded.extend_from_slice(&(bytes.len() as u64 * 8).to_be_bytes());
	let mut state = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0];
	for block in padded.chunks_exact(64) {
		let mut words = [0u32; 80];
		for (word, chunk) in words.iter_mut().zip(block.chunks_exact(4)) {
			*word = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
		}
		for i in 16..80 {
			words[i] = (words[i - 3] ^ words[i - 8] ^ words[i - 14] ^ words[i - 16]).rotate_left(1);
		}
		let [mut a, mut b, mut c, mut d, mut e] = state;
		for (i, word) in words.iter().enumerate() {
			let (f, k) = match i {
				0..=19 => ((b & c) | (!b & d), 0x5a827999),
				20..=39 => (b ^ c ^ d, 0x6ed9eba1),
				40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdc),
				_ => (b ^ c ^ d, 0xca62c1d6),
			};
			let temp = a
				.rotate_left(5)
				.wrapping_add(f)
				.wrapping_add(e)
				.wrapping_add(k)
				.wrapping_add(*word);
			e = d;
			d = c;
			c = b.rotate_left(30);
			b = a;
			a = temp;
		}
		for (value, update) in state.iter_mut().zip([a, b, c, d, e]) {
			*value = value.wrapping_add(update);
		}
	}
	state.iter().map(|word| format!("{word:08x}")).collect()
}
