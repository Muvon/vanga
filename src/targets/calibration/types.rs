//! Calibration Parameter Types
//!
//! This module contains all parameter structures used in the calibration system.
//! Each target type has its own parameter struct with calibrated values that
//! ensure balanced 5-class classification across all targets.
//!
//! ## PER-HORIZON CALIBRATION
//!
//! Parameters are stored per-horizon because different time horizons require
//! different classification thresholds:
//! - 1h: High noise, needs higher sensitivity
//! - 4h: Moderate noise, moderate sensitivity
//! - 24h: Low noise, lower sensitivity

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Calibrated parameters for all targets with PER-HORIZON storage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CalibratedParameters {
	/// Direction parameters per horizon (e.g., "1h" -> DirectionParams)
	pub direction: HashMap<String, DirectionParams>,
	/// Price level parameters per horizon
	pub price_levels: HashMap<String, PriceLevelParams>,
	/// Stop level parameters per horizon
	pub stop_levels: HashMap<String, StopLevelParams>,
	/// Volatility parameters per horizon
	pub volatility: HashMap<String, VolatilityParams>,
	/// Sentiment parameters per horizon
	pub sentiment: HashMap<String, SentimentParams>,
	/// Volume parameters per horizon
	pub volume: HashMap<String, VolumeParams>,
	/// Calibration metadata
	pub metadata: CalibrationMetadata,
}

impl CalibratedParameters {
	/// Get direction parameters for specific horizon
	pub fn get_direction(&self, horizon: &str) -> Option<&DirectionParams> {
		self.direction.get(horizon)
	}

	/// Get price level parameters for specific horizon
	pub fn get_price_levels(&self, horizon: &str) -> Option<&PriceLevelParams> {
		self.price_levels.get(horizon)
	}

	/// Get volatility parameters for specific horizon
	pub fn get_volatility(&self, horizon: &str) -> Option<&VolatilityParams> {
		self.volatility.get(horizon)
	}

	/// Get sentiment parameters for specific horizon
	pub fn get_sentiment(&self, horizon: &str) -> Option<&SentimentParams> {
		self.sentiment.get(horizon)
	}

	/// Get volume parameters for specific horizon
	pub fn get_volume(&self, horizon: &str) -> Option<&VolumeParams> {
		self.volume.get(horizon)
	}

	/// Get stop level parameters for specific horizon
	pub fn get_stop_levels(&self, horizon: &str) -> Option<&StopLevelParams> {
		self.stop_levels.get(horizon)
	}

	/// Get all calibrated horizons
	pub fn get_horizons(&self) -> Vec<String> {
		if !self.metadata.horizons.is_empty() {
			let mut horizons = Vec::new();
			for horizon in &self.metadata.horizons {
				if !horizons.contains(horizon) {
					horizons.push(horizon.clone());
				}
			}
			return horizons;
		}

		// Backward-compatible fallback for older serialized models without
		// metadata horizons. Any target may have been the only enabled one.
		let mut horizons: Vec<String> = self
			.direction
			.keys()
			.chain(self.price_levels.keys())
			.chain(self.stop_levels.keys())
			.chain(self.volatility.keys())
			.chain(self.sentiment.keys())
			.chain(self.volume.keys())
			.cloned()
			.collect();
		horizons.sort();
		horizons.dedup();
		horizons
	}

	/// Validate that all targets have parameters for all horizons
	pub fn validate(&self) -> Result<(), String> {
		let horizons = self.get_horizons();

		fn validate_target_map<T>(
			target_name: &str,
			target_map: &HashMap<String, T>,
			horizons: &[String],
		) -> Result<(), String> {
			if target_map.is_empty() {
				return Ok(());
			}
			for horizon in horizons {
				if !target_map.contains_key(horizon) {
					return Err(format!("Missing {} params for horizon: {}", target_name, horizon));
				}
			}
			Ok(())
		}

		validate_target_map("direction", &self.direction, &horizons)?;
		validate_target_map("price_levels", &self.price_levels, &horizons)?;
		validate_target_map("stop_levels", &self.stop_levels, &horizons)?;
		validate_target_map("volatility", &self.volatility, &horizons)?;
		validate_target_map("sentiment", &self.sentiment, &horizons)?;
		validate_target_map("volume", &self.volume, &horizons)?;

		Ok(())
	}
}

/// Direction target parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DirectionParams {
	pub sensitivity: f64,
	pub extreme_multiplier: f64,
	pub min_base_threshold: f64,    // NEW: Replaces hardcoded 0.01
	pub min_extreme_threshold: f64, // NEW: Replaces hardcoded 0.03
	pub base_multiplier: f64,       // NEW: Replaces hardcoded 20.0
	pub balance: ClassBalance,
}

/// Price level target parameters. Legacy binary records retain their exact
/// five-field layout; experimental records use a reserved bandwidth sentinel.
#[derive(Debug, Clone, Default)]
pub struct PriceLevelParams {
	pub definition: crate::config::training::PriceLevelDefinition,
	pub thresholds: Option<[f64; 4]>,
	pub regime_window_bars: Option<usize>,
	pub bandwidth: f64,
	pub percentiles: [f64; 2],
	pub neutral_band_factor: f64,
	pub momentum_factor: f64,
	pub balance: ClassBalance,
}

#[derive(Serialize, Deserialize)]
struct PriceLevelParamsJson {
	#[serde(flatten)]
	definition: crate::config::training::PriceLevelDefinition,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	thresholds: Option<[f64; 4]>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	regime_window_bars: Option<usize>,
	bandwidth: f64,
	percentiles: [f64; 2],
	neutral_band_factor: f64,
	momentum_factor: f64,
	balance: ClassBalance,
}

impl Serialize for PriceLevelParams {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		if serializer.is_human_readable() {
			return PriceLevelParamsJson {
				definition: self.definition,
				thresholds: self.thresholds,
				regime_window_bars: self.regime_window_bars,
				bandwidth: self.bandwidth,
				percentiles: self.percentiles,
				neutral_band_factor: self.neutral_band_factor,
				momentum_factor: self.momentum_factor,
				balance: self.balance.clone(),
			}
			.serialize(serializer);
		}
		if self
			.definition
			.uses_anchor_volatility()
			.map_err(serde::ser::Error::custom)?
		{
			let thresholds = self
				.thresholds
				.ok_or_else(|| serde::ser::Error::custom("Anchor price levels require calibrated thresholds"))?;
			// Negative infinity cannot be a valid legacy bandwidth. A version
			// byte follows it so new fields never shift an old record's layout.
			// Version 1 stays the exact record older models wrote; version 2 adds
			// the statistic and the regime window.
			if self.definition.uses_regime_volatility()
				|| self.definition.statistic != crate::config::training::PriceLevelStatistic::WeightedClose
			{
				let regime_window_bars = if self.definition.uses_regime_volatility() {
					Some(self.regime_window_bars.filter(|window| *window >= 2).ok_or_else(|| {
						serde::ser::Error::custom("Regime-volatility price levels require regime_window_bars >= 2")
					})?)
				} else {
					None
				};
				(
					f64::NEG_INFINITY,
					2u8,
					(
						self.definition.reference,
						self.definition.scale,
						self.definition.statistic,
					),
					thresholds,
					regime_window_bars,
					self.bandwidth,
					self.percentiles,
					self.neutral_band_factor,
					self.momentum_factor,
					&self.balance,
				)
					.serialize(serializer)
			} else {
				(
					f64::NEG_INFINITY,
					1u8,
					(self.definition.reference, self.definition.scale),
					thresholds,
					self.bandwidth,
					self.percentiles,
					self.neutral_band_factor,
					self.momentum_factor,
					&self.balance,
				)
					.serialize(serializer)
			}
		} else {
			(
				self.bandwidth,
				self.percentiles,
				self.neutral_band_factor,
				self.momentum_factor,
				&self.balance,
			)
				.serialize(serializer)
		}
	}
}

impl<'de> Deserialize<'de> for PriceLevelParams {
	fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
		if deserializer.is_human_readable() {
			let value = PriceLevelParamsJson::deserialize(deserializer)?;
			if value.definition.uses_regime_volatility() && value.regime_window_bars.is_none_or(|window| window < 2) {
				return Err(serde::de::Error::custom(
					"Regime-volatility price levels require regime_window_bars >= 2",
				));
			}
			return Ok(Self {
				definition: value.definition,
				thresholds: value.thresholds,
				regime_window_bars: value.regime_window_bars,
				bandwidth: value.bandwidth,
				percentiles: value.percentiles,
				neutral_band_factor: value.neutral_band_factor,
				momentum_factor: value.momentum_factor,
				balance: value.balance,
			});
		}
		struct BinaryPriceLevelVisitor;
		impl<'de> serde::de::Visitor<'de> for BinaryPriceLevelVisitor {
			type Value = PriceLevelParams;

			fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
				formatter.write_str("legacy or versioned price-level parameters")
			}

			fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
				fn next<'de, T: Deserialize<'de>, A: serde::de::SeqAccess<'de>>(
					sequence: &mut A,
				) -> Result<T, A::Error> {
					sequence
						.next_element()?
						.ok_or_else(|| serde::de::Error::custom("Truncated price-level parameters"))
				}
				let first: f64 = next(&mut sequence)?;
				let mut value = PriceLevelParams::default();
				if first == f64::NEG_INFINITY {
					let version: u8 = next(&mut sequence)?;
					if version != 1 && version != 2 {
						return Err(serde::de::Error::custom("Unsupported price-level parameter version"));
					}
					if version == 1 {
						let (reference, scale) = next(&mut sequence)?;
						value.definition = crate::config::training::PriceLevelDefinition {
							reference,
							scale,
							..Default::default()
						};
					} else {
						let (reference, scale, statistic) = next(&mut sequence)?;
						value.definition = crate::config::training::PriceLevelDefinition {
							reference,
							scale,
							statistic,
						};
					}
					value
						.definition
						.uses_anchor_volatility()
						.map_err(serde::de::Error::custom)?;
					value.thresholds = Some(next(&mut sequence)?);
					if version == 2 {
						let window_bars: Option<usize> = next(&mut sequence)?;
						if value.definition.uses_regime_volatility() != window_bars.is_some() {
							return Err(serde::de::Error::custom(
								"Price-level parameter version 2 carries a regime window exactly for regime_volatility",
							));
						}
						if window_bars.is_some_and(|window| window < 2) {
							return Err(serde::de::Error::custom(
								"Regime-volatility price levels require regime_window_bars >= 2",
							));
						}
						value.regime_window_bars = window_bars;
					} else if value.definition.uses_regime_volatility() {
						return Err(serde::de::Error::custom(
							"Regime-volatility price levels require parameter version 2",
						));
					}
					value.bandwidth = next(&mut sequence)?;
				} else {
					value.bandwidth = first;
				}
				value.percentiles = next(&mut sequence)?;
				value.neutral_band_factor = next(&mut sequence)?;
				value.momentum_factor = next(&mut sequence)?;
				value.balance = next(&mut sequence)?;
				Ok(value)
			}
		}
		// Bincode reads only the fields consumed by the visitor; legacy records
		// consume five, version 1 records nine and version 2 records ten,
		// including in enclosing models.
		deserializer.deserialize_tuple(10, BinaryPriceLevelVisitor)
	}
}

/// Stop level target parameters. Legacy binary records retain their exact
/// five-field layout; experimental records use a reserved bandwidth sentinel.
#[derive(Debug, Clone, Default)]
pub struct StopLevelParams {
	pub definition: crate::config::training::StopLevelDefinition,
	pub thresholds: Option<[f64; 4]>,
	pub regime_window_bars: Option<usize>,
	pub bandwidth: f64,
	pub percentiles: [f64; 2],
	pub neutral_band_factor: f64,
	pub momentum_factor: f64,
	pub balance: ClassBalance,
}

#[derive(Serialize, Deserialize)]
struct StopLevelParamsJson {
	#[serde(flatten)]
	definition: crate::config::training::StopLevelDefinition,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	thresholds: Option<[f64; 4]>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	regime_window_bars: Option<usize>,
	bandwidth: f64,
	percentiles: [f64; 2],
	neutral_band_factor: f64,
	momentum_factor: f64,
	balance: ClassBalance,
}

impl Serialize for StopLevelParams {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		if serializer.is_human_readable() {
			return StopLevelParamsJson {
				definition: self.definition,
				thresholds: self.thresholds,
				regime_window_bars: self.regime_window_bars,
				bandwidth: self.bandwidth,
				percentiles: self.percentiles,
				neutral_band_factor: self.neutral_band_factor,
				momentum_factor: self.momentum_factor,
				balance: self.balance.clone(),
			}
			.serialize(serializer);
		}
		if self
			.definition
			.uses_anchor_volatility()
			.map_err(serde::ser::Error::custom)?
		{
			let thresholds = self
				.thresholds
				.ok_or_else(|| serde::ser::Error::custom("Anchor stop levels require calibrated thresholds"))?;
			// Negative infinity cannot be a valid legacy bandwidth. A version
			// byte follows it so new fields never shift an old record's layout.
			if self.definition.uses_regime_volatility() {
				let regime_window_bars = self.regime_window_bars.filter(|window| *window >= 2).ok_or_else(|| {
					serde::ser::Error::custom("Regime-volatility stop levels require regime_window_bars >= 2")
				})?;
				(
					f64::NEG_INFINITY,
					2u8,
					(self.definition.reference, self.definition.scale, self.definition.side),
					thresholds,
					regime_window_bars,
					self.bandwidth,
					self.percentiles,
					self.neutral_band_factor,
					self.momentum_factor,
					&self.balance,
				)
					.serialize(serializer)
			} else {
				(
					f64::NEG_INFINITY,
					1u8,
					(self.definition.reference, self.definition.scale, self.definition.side),
					thresholds,
					self.bandwidth,
					self.percentiles,
					self.neutral_band_factor,
					self.momentum_factor,
					&self.balance,
				)
					.serialize(serializer)
			}
		} else {
			(
				self.bandwidth,
				self.percentiles,
				self.neutral_band_factor,
				self.momentum_factor,
				&self.balance,
			)
				.serialize(serializer)
		}
	}
}

impl<'de> Deserialize<'de> for StopLevelParams {
	fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
		if deserializer.is_human_readable() {
			let value = StopLevelParamsJson::deserialize(deserializer)?;
			if value.definition.uses_regime_volatility() && value.regime_window_bars.is_none_or(|window| window < 2) {
				return Err(serde::de::Error::custom(
					"Regime-volatility stop levels require regime_window_bars >= 2",
				));
			}
			return Ok(Self {
				definition: value.definition,
				thresholds: value.thresholds,
				regime_window_bars: value.regime_window_bars,
				bandwidth: value.bandwidth,
				percentiles: value.percentiles,
				neutral_band_factor: value.neutral_band_factor,
				momentum_factor: value.momentum_factor,
				balance: value.balance,
			});
		}
		struct BinaryStopLevelVisitor;
		impl<'de> serde::de::Visitor<'de> for BinaryStopLevelVisitor {
			type Value = StopLevelParams;

			fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
				formatter.write_str("legacy or versioned stop-level parameters")
			}

			fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
				fn next<'de, T: Deserialize<'de>, A: serde::de::SeqAccess<'de>>(
					sequence: &mut A,
				) -> Result<T, A::Error> {
					sequence
						.next_element()?
						.ok_or_else(|| serde::de::Error::custom("Truncated stop-level parameters"))
				}
				let first: f64 = next(&mut sequence)?;
				let mut value = StopLevelParams::default();
				if first == f64::NEG_INFINITY {
					let version: u8 = next(&mut sequence)?;
					if version != 1 && version != 2 {
						return Err(serde::de::Error::custom("Unsupported stop-level parameter version"));
					}
					let (reference, scale, side) = next(&mut sequence)?;
					value.definition = crate::config::training::StopLevelDefinition { reference, scale, side };
					value
						.definition
						.uses_anchor_volatility()
						.map_err(serde::de::Error::custom)?;
					value.thresholds = Some(next(&mut sequence)?);
					if version == 2 {
						if !value.definition.uses_regime_volatility() {
							return Err(serde::de::Error::custom(
								"Stop-level parameter version 2 requires regime_volatility",
							));
						}
						let window_bars = next(&mut sequence)?;
						if window_bars < 2 {
							return Err(serde::de::Error::custom(
								"Regime-volatility stop levels require regime_window_bars >= 2",
							));
						}
						value.regime_window_bars = Some(window_bars);
					} else if value.definition.uses_regime_volatility() {
						return Err(serde::de::Error::custom(
							"Regime-volatility stop levels require parameter version 2",
						));
					}
					value.bandwidth = next(&mut sequence)?;
				} else {
					value.bandwidth = first;
				}
				value.percentiles = next(&mut sequence)?;
				value.neutral_band_factor = next(&mut sequence)?;
				value.momentum_factor = next(&mut sequence)?;
				value.balance = next(&mut sequence)?;
				Ok(value)
			}
		}
		// Bincode reads only the fields consumed by the visitor; legacy records
		// consume five, and versioned records consume only their encoded fields.
		deserializer.deserialize_tuple(10, BinaryStopLevelVisitor)
	}
}

/// Volatility target parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VolatilityParams {
	pub bandwidth: f64,
	pub extreme_multiplier: f64,
	pub volume_weight: f64, // NEW: Volume weight for volatility score calculation
	pub horizon_decay: f64,
	pub min_volatility_baseline: f64, // NEW: Replaces hardcoded 0.005
	pub balance: ClassBalance,
}

/// Sentiment target parameters (Extended with percentile and smoothing for better balance)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SentimentParams {
	// Classification thresholds (Base parameters)
	pub sensitivity: f64,        // Base threshold for classification
	pub extreme_multiplier: f64, // Multiplier for extreme classes

	// Percentile-based adaptive thresholds (NEW - for better balance with fewer samples)
	pub percentile_low: f64,  // Lower percentile for sequence range (e.g., 0.05 for p5)
	pub percentile_high: f64, // Upper percentile for sequence range (e.g., 0.95 for p95)

	// Smoothing for noise reduction (NEW - reduces noise in volume-price divergence)
	pub smoothing_periods: usize, // Moving average periods for volume smoothing (1-30)

	// Balance metrics
	pub balance: ClassBalance,
}

/// Volume target parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VolumeParams {
	pub bandwidth: f64,
	pub extreme_multiplier: f64,
	pub smoothing_periods: usize,
	pub min_base_threshold: f64,    // NEW: Minimum base threshold for consistency
	pub min_extreme_threshold: f64, // NEW: Minimum extreme threshold for consistency
	pub percentile_low: f64,        // NEW: Lower percentile for sequence range (e.g., 0.05 for p5)
	pub percentile_high: f64,       // NEW: Upper percentile for sequence range (e.g., 0.95 for p95)
	pub balance: ClassBalance,
}

/// Enhanced class distribution balance metrics with diversity scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassBalance {
	pub class_percentages: [f64; 5], // Fixed array for 5 classes
	pub balance_score: f64,
	pub imbalance_ratio: f64,
	pub total_samples: usize,
	pub target_balance: f64, // Added for compatibility

	// NEW: Diversity-focused metrics
	pub diversity_score: f64,            // Overall diversity score (0.0 to 1.0, higher is better)
	pub temporal_spread: f64,            // Temporal distribution diversity (0.0 to 1.0)
	pub feature_diversity: f64,          // Feature space diversity (0.0 to 1.0)
	pub market_condition_diversity: f64, // Market condition diversity (0.0 to 1.0)
	pub composite_quality_score: f64,    // Combined balance + diversity score (lower is better)
}

impl Default for ClassBalance {
	fn default() -> Self {
		Self {
			class_percentages: [20.0, 20.0, 20.0, 20.0, 20.0],
			balance_score: 0.0,
			imbalance_ratio: 1.0,
			total_samples: 0,
			target_balance: 0.2,

			// NEW: Default diversity metrics
			diversity_score: 0.0,
			temporal_spread: 0.0,
			feature_diversity: 0.0,
			market_condition_diversity: 0.0,
			composite_quality_score: f64::INFINITY, // Start with worst possible score
		}
	}
}

impl ClassBalance {
	/// True when the five class sizes differ by at most one sample. No other split of the same samples is more
	/// even, so a threshold search that only scores balance cannot improve on it.
	pub fn is_most_even_split(&self) -> bool {
		let counts = self
			.class_percentages
			.map(|percentage| (percentage / 100.0 * self.total_samples as f64).round() as usize);
		let smallest = counts.iter().copied().min().unwrap_or(0);
		let largest = counts.iter().copied().max().unwrap_or(0);
		smallest > 0 && counts.iter().sum::<usize>() == self.total_samples && largest - smallest <= 1
	}
}

// Legacy compatibility
pub type ClassDistributionBalance = ClassBalance;

/// Calibration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetadata {
	pub data_length: usize,
	pub sequence_length: usize,
	pub horizons: Vec<String>, // NEW: Track all calibrated horizons
	pub calibration_samples: usize,
	pub calibration_iterations: usize,
	pub optimization_time_ms: u64,
	pub target_balance: f64,
	pub overall_balance_score: f64,
	pub calibration_success: bool,
}

impl Default for CalibrationMetadata {
	fn default() -> Self {
		Self {
			data_length: 0,
			sequence_length: 96,
			horizons: vec![], // Empty by default
			calibration_samples: 0,
			calibration_iterations: 100,
			optimization_time_ms: 0,
			target_balance: 0.2,
			overall_balance_score: f64::INFINITY,
			calibration_success: false,
		}
	}
}

/// Context for evaluation functions
#[derive(Clone, Copy)]
pub struct EvaluationContext<'a> {
	pub ohlcv_data: &'a [crate::data::structures::MarketDataRow],
	pub sample_indices: &'a [usize],
	pub sequence_length: usize,
	pub horizon_steps: usize,
}

/// Parameters for direction evaluation
#[derive(Debug, Clone)]
pub struct DirectionEvalParams {
	pub sensitivity: f64,
	pub extreme_multiplier: f64,
	pub min_base_threshold: f64,
	pub min_extreme_threshold: f64,
	pub base_multiplier: f64,
}

/// Parameters for price level evaluation
#[derive(Debug, Clone)]
pub struct PriceLevelEvalParams {
	pub bandwidth: f64,
	pub percentiles: [f64; 2], // Base percentiles for adaptive calculation
	pub neutral_band: f64,
	pub momentum_factor: f64, // Momentum factor for bandwidth adjustment
}

/// Parameters for stop level evaluation
#[derive(Debug, Clone)]
pub struct StopLevelEvalParams {
	pub bandwidth: f64,
	pub percentiles: [f64; 2],
	pub neutral_band: f64,
	pub momentum_factor: f64,
}

/// Parameters for volatility evaluation
#[derive(Debug, Clone)]
pub struct VolatilityEvalParams {
	pub bandwidth: f64,
	pub multiplier: f64,
	pub decay: f64,
	pub volume_weight: f64,
	pub min_baseline: f64,
}

/// Parameters for volume evaluation
#[derive(Debug, Clone)]
pub struct VolumeEvalParams {
	pub bandwidth: f64,
	pub multiplier: f64,
	pub smoothing: usize,
	pub percentile_low: f64,  // Lower percentile for sequence range
	pub percentile_high: f64, // Upper percentile for sequence range
}

/// Parameters for sentiment evaluation (extended with percentiles and smoothing)
#[derive(Debug, Clone)]
pub struct SentimentEvalParams {
	pub sensitivity: f64,
	pub extreme_multiplier: f64,
	pub percentile_low: f64,
	pub percentile_high: f64,
	pub smoothing: usize,
}
