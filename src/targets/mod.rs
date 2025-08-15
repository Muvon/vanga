//! Multi-target prediction system for cryptocurrency forecasting
//!
//! This module orchestrates the generation of multiple prediction targets:
//! - Price levels: Quantile-based price classification
//! - Direction: Up/down/sideways movement classification
//! - Volatility: Low/medium/high volatility regime classification
//!
//! ## Adaptive Parameter System
//!
//! The system now includes automatic parameter optimization that finds the optimal
//! "sweet spot" parameters for balanced class distribution across all target types:
//!
//! - **AdaptiveTargetParameters**: Stores calibrated parameters for all targets
//! - **UnifiedTargetCalibrator**: Orchestrates system-wide parameter optimization
//! - **Model Integration**: Parameters are saved/loaded with the model for consistency

pub mod calibration;
pub mod direction;
pub mod interface;
pub mod price_levels;
pub mod sentiment;
pub mod sequence_reconstruction;
pub mod volatility;
pub mod volume;

#[cfg(test)]
mod math_consistency_test;
#[cfg(test)]
mod price_level_test;

#[cfg(test)]
mod direction_test;
#[cfg(test)]
mod volatility_test;
#[cfg(test)]
mod volume_test;

use crate::targets::calibration::CalibratedParameters;
use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

// Re-export configurations and calibrated parameters
pub use calibration::{
    DirectionParams, PriceLevelParams, SentimentParams, VolatilityParams, VolumeParams,
};
use direction::generate_direction_targets_with_calibrated_params;
pub use price_levels::{
    generate_price_level_targets_with_calibrated_params, get_horizon_exponential_weighted_close,
    get_sequence_exponential_weighted_close, reconstruct_price_levels, PriceLevelConfig,
};
pub use sentiment::{
    generate_sentiment_targets_with_calibrated_params, get_sentiment_class_names, SentimentConfig,
};
pub use sequence_reconstruction::{
    SequenceAnalyzer, SequenceBoundaries, SequenceReconstructionConfig, SequenceReconstructor,
};
use volatility::generate_volatility_targets_with_calibrated_params;
pub use volume::{
    generate_volume_targets_with_calibrated_params, get_volume_class_names, VolumeConfig,
};

/// Comprehensive target configuration
#[derive(Debug, Clone)]
pub struct MultiTargetConfig {
    pub price_level_config: PriceLevelConfig,
    pub horizons: Vec<String>,
    pub price_level: TargetTypeConfig,
    pub direction: TargetTypeConfig,
    pub volatility: TargetTypeConfig,
    pub sentiment: TargetTypeConfig,
    pub volume: TargetTypeConfig,
}

#[derive(Debug, Clone)]
pub struct TargetTypeConfig {
    pub enabled: bool,
}

impl Default for MultiTargetConfig {
    fn default() -> Self {
        Self {
            price_level_config: PriceLevelConfig::default(),
            horizons: vec![
                "1h".to_string(), // FIXED: Default to single horizon - should be overridden by training config
            ],
            price_level: TargetTypeConfig { enabled: true },
            direction: TargetTypeConfig { enabled: true },
            volatility: TargetTypeConfig { enabled: true },
            sentiment: TargetTypeConfig { enabled: true },
            volume: TargetTypeConfig { enabled: true },
        }
    }
}

impl MultiTargetConfig {
    /// Create MultiTargetConfig from TrainingConfig with proper target enablement
    pub fn from_training_config(training_config: &crate::config::training::TrainingConfig) -> Self {
        Self {
            // Price level config not used with calibration system - all parameters come from calibration
            price_level_config: PriceLevelConfig::default(),
            horizons: training_config.horizons.clone(),
            price_level: TargetTypeConfig {
                enabled: training_config.targets.price_level,
            },
            direction: TargetTypeConfig {
                enabled: training_config.targets.direction,
            },
            volatility: TargetTypeConfig {
                enabled: training_config.targets.volatility,
            },
            sentiment: TargetTypeConfig {
                enabled: training_config.targets.sentiment,
            },
            volume: TargetTypeConfig {
                enabled: training_config.targets.volume,
            },
        }
    }
}

/// Container for all prepared targets
#[derive(Debug, Clone)]
pub struct PreparedTargets {
    pub price_levels: HashMap<String, Vec<i32>>,
    pub direction: HashMap<String, Vec<i32>>,
    pub volatility: HashMap<String, Vec<i32>>,
    pub sentiment: HashMap<String, Vec<i32>>,
    pub volume: HashMap<String, Vec<i32>>,
    pub target_names: Vec<String>, // ADDED: Avoid redundant TargetGenerator creation
    pub data_length: usize,
    pub valid_indices: Vec<usize>,
}

impl PreparedTargets {
    /// Create new empty PreparedTargets
    pub fn new(data_length: usize) -> Self {
        Self {
            price_levels: HashMap::new(),
            direction: HashMap::new(),
            volatility: HashMap::new(),
            sentiment: HashMap::new(),
            volume: HashMap::new(),
            target_names: Vec::new(), // Initialize empty target names
            data_length,
            valid_indices: Vec::new(),
        }
    }

    /// Get targets for a specific horizon and target type
    pub fn get_targets(&self, horizon: &str, target_type: TargetType) -> Option<&Vec<i32>> {
        match target_type {
            TargetType::PriceLevel => self.price_levels.get(horizon),
            TargetType::Direction => self.direction.get(horizon),
            TargetType::Volatility => self.volatility.get(horizon),
            TargetType::Sentiment => self.sentiment.get(horizon),
            TargetType::Volume => self.volume.get(horizon),
        }
    }

    /// Get all horizons available from ANY target type
    pub fn get_horizons(&self) -> Vec<String> {
        let mut horizons: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Collect horizons from all target types
        horizons.extend(self.price_levels.keys().cloned());
        horizons.extend(self.direction.keys().cloned());
        horizons.extend(self.volatility.keys().cloned());
        horizons.extend(self.sentiment.keys().cloned());
        horizons.extend(self.volume.keys().cloned());

        let mut horizon_vec: Vec<String> = horizons.into_iter().collect();
        horizon_vec.sort();
        horizon_vec
    }

    /// Validate targets consistency across all types
    pub fn validate(&self) -> Result<()> {
        let horizons = self.get_horizons();

        for horizon in &horizons {
            let price_len = self.price_levels.get(horizon).map(|v| v.len());
            let direction_len = self.direction.get(horizon).map(|v| v.len());
            let volatility_len = self.volatility.get(horizon).map(|v| v.len());
            let sentiment_len = self.sentiment.get(horizon).map(|v| v.len());
            let volume_len = self.volume.get(horizon).map(|v| v.len());

            if price_len != direction_len
                || direction_len != volatility_len
                || volatility_len != sentiment_len
                || sentiment_len != volume_len
            {
                return Err(crate::utils::error::VangaError::DataError(format!(
                    "Target length mismatch for horizon {}: price={:?}, direction={:?}, volatility={:?}, sentiment={:?}, volume={:?}",
                    horizon, price_len, direction_len, volatility_len, sentiment_len, volume_len
                )));
            }

            if let Some(len) = price_len {
                if len != self.data_length {
                    return Err(crate::utils::error::VangaError::DataError(format!(
                        "Target length {} does not match data length {} for horizon {}",
                        len, self.data_length, horizon
                    )));
                }
            }
        }

        Ok(())
    }

    /// Calculate target statistics
    pub fn calculate_statistics(&self) -> TargetStatistics {
        let mut stats = TargetStatistics::new();

        for (horizon, targets) in &self.price_levels {
            let valid_targets: Vec<i32> = targets.iter().filter(|&&t| t >= 0).copied().collect();
            stats.price_level_stats.insert(
                horizon.clone(),
                calculate_class_distribution(&valid_targets),
            );
        }

        for (horizon, targets) in &self.direction {
            let valid_targets: Vec<i32> = targets.iter().filter(|&&t| t >= 0).copied().collect();
            stats.direction_stats.insert(
                horizon.clone(),
                calculate_class_distribution(&valid_targets),
            );
        }

        for (horizon, targets) in &self.volatility {
            let valid_targets: Vec<i32> = targets.iter().filter(|&&t| t >= 0).copied().collect();
            stats.volatility_stats.insert(
                horizon.clone(),
                calculate_class_distribution(&valid_targets),
            );
        }

        stats
    }
}

/// Target type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TargetType {
    PriceLevel,
    Direction,
    Volatility,
    Sentiment,
    Volume,
}

/// Target statistics container
#[derive(Debug, Clone)]
pub struct TargetStatistics {
    pub price_level_stats: HashMap<String, ClassDistribution>,
    pub direction_stats: HashMap<String, ClassDistribution>,
    pub volatility_stats: HashMap<String, ClassDistribution>,
    pub sentiment_stats: HashMap<String, ClassDistribution>,
    pub volume_stats: HashMap<String, ClassDistribution>,
}

impl TargetStatistics {
    fn new() -> Self {
        Self {
            price_level_stats: HashMap::new(),
            direction_stats: HashMap::new(),
            volatility_stats: HashMap::new(),
            sentiment_stats: HashMap::new(),
            volume_stats: HashMap::new(),
        }
    }
}

/// Class distribution statistics
#[derive(Debug, Clone)]
pub struct ClassDistribution {
    pub class_counts: HashMap<i32, usize>,
    pub total_samples: usize,
    pub class_percentages: HashMap<i32, f64>,
}

/// Target generation orchestrator
pub struct TargetGenerator {
    config: MultiTargetConfig,
}

impl TargetGenerator {
    /// Create new target generator with configuration
    pub fn new(config: MultiTargetConfig) -> Self {
        Self { config }
    }

    /// Create target generator with default configuration
    pub fn with_defaults() -> Self {
        Self::new(MultiTargetConfig::default())
    }

    /// Get descriptive names for enabled targets only
    pub fn get_target_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        // Only add enabled targets
        if self.config.price_level.enabled {
            for horizon in &self.config.horizons {
                names.push(format!("price_level_{}", horizon));
            }
        }

        if self.config.direction.enabled {
            for horizon in &self.config.horizons {
                names.push(format!("direction_{}", horizon));
            }
        }

        if self.config.volatility.enabled {
            for horizon in &self.config.horizons {
                names.push(format!("volatility_{}", horizon));
            }
        }

        if self.config.sentiment.enabled {
            for horizon in &self.config.horizons {
                names.push(format!("sentiment_{}", horizon));
            }
        }

        if self.config.volume.enabled {
            for horizon in &self.config.horizons {
                names.push(format!("volume_{}", horizon));
            }
        }

        names
    }

    /// Get the total number of targets that will be generated (enabled targets only)
    pub fn get_num_targets(&self) -> usize {
        let mut count = 0;
        if self.config.price_level.enabled {
            count += self.config.horizons.len();
        }
        if self.config.direction.enabled {
            count += self.config.horizons.len();
        }
        if self.config.volatility.enabled {
            count += self.config.horizons.len();
        }
        if self.config.sentiment.enabled {
            count += self.config.horizons.len();
        }
        if self.config.volume.enabled {
            count += self.config.horizons.len();
        }
        count
    }

    /// Generate all targets aligned with specific sequence indices (FIXED: for proper synchronization)
    pub async fn generate_all_targets(
        &self,
        df: &DataFrame,
        sequence_indices: &[usize],
        sequence_length: usize,
    ) -> Result<PreparedTargets> {
        self.generate_all_targets_with_calibrated_params(
            df,
            sequence_indices,
            sequence_length,
            None, // No adaptive parameters - use calibration/base config
        )
        .await
    }
    /// Generate all targets with optional adaptive parameters (conditional generation)
    ///
    /// When adaptive_params is provided, uses the pre-calibrated parameters for consistent
    /// target generation between training and prediction. When None, uses calibration/base config.
    /// Only generates enabled targets for performance optimization.
    /// CRITICAL: Requires calibrated adaptive parameters - no defaults allowed!
    pub async fn generate_all_targets_with_calibrated_params(
        &self,
        df: &DataFrame,
        sequence_indices: &[usize],
        sequence_length: usize,
        calibrated_params: Option<&CalibratedParameters>,
    ) -> Result<PreparedTargets> {
        // FIXED: Data length should be the number of sequences, not original data length
        let data_length = sequence_indices.len();
        let mut prepared_targets = PreparedTargets::new(data_length);

        log::info!(
            "🎯 Generating aligned targets for {} sequences at specific indices (enabled targets only)",
            sequence_indices.len()
        );

        // CRITICAL FIX: Require calibrated parameters - no defaults allowed during training!
        let calibrated_params = calibrated_params.ok_or_else(|| {
            crate::utils::VangaError::ConfigError(
                "FATAL: Calibrated parameters are REQUIRED for target generation. \
                 Calibration must be performed before generating targets. \
                 This ensures consistent classification between training and prediction."
                    .to_string(),
            )
        })?;

        // Extract calibrated parameters - no defaults!
        let direction_params = &calibrated_params.direction;
        let price_level_params = &calibrated_params.price_levels;
        let volatility_params = &calibrated_params.volatility;
        let sentiment_params = &calibrated_params.sentiment;
        let volume_params = &calibrated_params.volume;

        log::debug!(
            "✅ Using calibrated parameters: direction_sensitivity={:.4}, \
             price_bandwidth={:.4}, volatility_bandwidth={:.4}",
            direction_params.sensitivity,
            price_level_params.bandwidth,
            volatility_params.bandwidth
        );

        // CONDITIONAL GENERATION: Only generate enabled targets
        if self.config.price_level.enabled {
            log::debug!("🏷️ Generating price level targets");
            let price_targets = generate_price_level_targets_with_calibrated_params(
                df,
                &self.config.horizons,
                sequence_indices,
                sequence_length,
                price_level_params,
            )?;
            prepared_targets.price_levels = price_targets;
        }

        if self.config.direction.enabled {
            log::debug!("🧭 Generating direction targets");
            let direction_targets = generate_direction_targets_with_calibrated_params(
                df,
                &self.config.horizons,
                sequence_indices,
                sequence_length,
                direction_params,
            )?;
            prepared_targets.direction = direction_targets;
        }

        if self.config.volatility.enabled {
            log::debug!("📊 Generating volatility targets");
            let volatility_targets = generate_volatility_targets_with_calibrated_params(
                df,
                &self.config.horizons,
                sequence_indices,
                sequence_length,
                volatility_params,
            )?;
            prepared_targets.volatility = volatility_targets;
        }

        if self.config.sentiment.enabled {
            log::debug!("💭 Generating sentiment targets");
            let sentiment_targets = generate_sentiment_targets_with_calibrated_params(
                df,
                &self.config.horizons,
                sequence_indices,
                sequence_length,
                sentiment_params,
            )?;
            prepared_targets.sentiment = sentiment_targets;
        }

        if self.config.volume.enabled {
            log::debug!("📈 Generating volume targets");
            let volume_targets = generate_volume_targets_with_calibrated_params(
                df,
                &self.config.horizons,
                sequence_indices,
                sequence_length,
                volume_params,
            )?;
            prepared_targets.volume = volume_targets;
        }

        // Set target names for enabled targets only
        prepared_targets.target_names = self.get_target_names();

        // Validate that we have targets for all sequences
        prepared_targets.valid_indices = (0..data_length).collect();

        log::info!(
            "✅ Generated {} target types across {} horizons for {} sequences",
            prepared_targets.target_names.len() / self.config.horizons.len().max(1),
            self.config.horizons.len(),
            data_length
        );

        Ok(prepared_targets)
    }
}

/// Calculate class distribution for target analysis
fn calculate_class_distribution(targets: &[i32]) -> ClassDistribution {
    let mut class_counts = HashMap::new();

    for &target in targets {
        *class_counts.entry(target).or_insert(0) += 1;
    }

    let total_samples = targets.len();
    let mut class_percentages = HashMap::new();

    for (&class, &count) in &class_counts {
        class_percentages.insert(class, count as f64 / total_samples as f64 * 100.0);
    }

    ClassDistribution {
        class_counts,
        total_samples,
        class_percentages,
    }
}
