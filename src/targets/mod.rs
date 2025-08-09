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

pub mod adaptive_parameters;
pub mod direction;
#[cfg(test)]
mod direction_test;

#[cfg(test)]
mod math_consistency_test;
#[cfg(test)]
mod price_level_test;
pub mod price_levels;
pub mod sentiment;
#[cfg(test)]
mod sentiment_test;
pub mod sequence_reconstruction;
pub mod unified_calibrator;
pub mod volatility;
#[cfg(test)]
mod volatility_test;
pub mod volume;
#[cfg(test)]
mod volume_test;

use crate::config::model::TargetsConfig;
use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

// Re-export configurations and adaptive parameters
pub use adaptive_parameters::{
    calculate_class_distribution_balance, AdaptiveParameterCalibrator, AdaptiveTargetParameters,
    CalibrationMetadata, ClassDistributionBalance, DirectionAdaptiveParams,
    PriceLevelAdaptiveParams, SentimentAdaptiveParams, VolatilityAdaptiveParams,
    VolumeAdaptiveParams,
};
pub use direction::{
    generate_direction_targets, generate_direction_targets_with_adaptive_params, Direction,
};
pub use price_levels::{
    generate_price_level_targets, generate_price_level_targets_from_model_config,
    generate_price_level_targets_with_adaptive_params,
    generate_price_level_targets_with_targets_config, PriceLevelConfig,
};
pub use sentiment::{
    generate_sentiment_targets, generate_sentiment_targets_with_adaptive_params,
    get_sentiment_class_names, SentimentConfig,
};
pub use sequence_reconstruction::{
    SequenceAnalyzer, SequenceBoundaries, SequenceReconstructionConfig, SequenceReconstructor,
};
pub use unified_calibrator::{
    calibrate_adaptive_parameters, CrossTargetCorrelation, SystemBalanceMetrics,
    UnifiedTargetCalibrator, UnifiedValidationResult,
};
pub use volatility::{
    generate_volatility_targets, generate_volatility_targets_with_adaptive_params,
};
pub use volume::{
    generate_volume_targets, generate_volume_targets_with_adaptive_params, get_volume_class_names,
    VolumeConfig,
};

/// Comprehensive target configuration
#[derive(Debug, Clone)]
pub struct MultiTargetConfig {
    pub price_level_config: PriceLevelConfig,
    pub horizons: Vec<String>,
}

impl Default for MultiTargetConfig {
    fn default() -> Self {
        Self {
            price_level_config: PriceLevelConfig::default(),
            horizons: vec![
                "1h".to_string(), // FIXED: Default to single horizon - should be overridden by training config
            ],
        }
    }
}

impl MultiTargetConfig {
    /// DEPRECATED: Create MultiTargetConfig from ModelConfig (use TargetsConfig directly instead)
    pub fn from_model_config(
        model_config: &crate::config::model::ModelConfig,
        horizons: Vec<String>,
    ) -> Self {
        // Use the new TargetsConfig approach
        Self {
            price_level_config: PriceLevelConfig {
                bandwidth_size: model_config.targets.base_sensitivity,
                neutral_band_factor: 0.4, // Default neutral band factor (40% of range)
            },
            horizons,
        }
    }
}

/// Container for all prepared targets
#[derive(Debug, Clone)]
pub struct PreparedTargets {
    pub price_levels: HashMap<String, Vec<i32>>,
    pub directions: HashMap<String, Vec<i32>>,
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
            directions: HashMap::new(),
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
            TargetType::Direction => self.directions.get(horizon),
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
        horizons.extend(self.directions.keys().cloned());
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
            let direction_len = self.directions.get(horizon).map(|v| v.len());
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

        for (horizon, targets) in &self.directions {
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

    /// Get descriptive names for all generated targets
    pub fn get_target_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        // Price level targets
        for horizon in &self.config.horizons {
            names.push(format!("price_level_{}", horizon));
        }

        // Direction targets
        for horizon in &self.config.horizons {
            names.push(format!("direction_{}", horizon));
        }

        // Volatility targets
        for horizon in &self.config.horizons {
            names.push(format!("volatility_{}", horizon));
        }

        // Sentiment targets
        for horizon in &self.config.horizons {
            names.push(format!("sentiment_{}", horizon));
        }

        // Volume targets
        for horizon in &self.config.horizons {
            names.push(format!("volume_{}", horizon));
        }

        names
    }

    /// Get the total number of targets that will be generated
    pub fn get_num_targets(&self) -> usize {
        // Each horizon generates: 1 price level + 1 direction + 1 volatility + 1 sentiment + 1 volume = 5 targets per horizon
        self.config.horizons.len() * 5
    }

    /// Generate all targets aligned with specific sequence indices (FIXED: for proper synchronization)
    pub async fn generate_all_targets(
        &self,
        df: &DataFrame,
        model_config: Option<&crate::config::model::ModelConfig>,
        sequence_indices: &[usize],
        sequence_length: usize,
    ) -> Result<PreparedTargets> {
        self.generate_all_targets_with_adaptive_params(
            df,
            model_config,
            sequence_indices,
            sequence_length,
            None, // No adaptive parameters - use calibration/base config
        )
        .await
    }

    /// Generate all targets with optional adaptive parameters
    ///
    /// When adaptive_params is provided, uses the pre-calibrated parameters for consistent
    /// target generation between training and prediction. When None, uses calibration/base config.
    pub async fn generate_all_targets_with_adaptive_params(
        &self,
        df: &DataFrame,
        model_config: Option<&crate::config::model::ModelConfig>,
        sequence_indices: &[usize],
        sequence_length: usize,
        adaptive_params: Option<&AdaptiveTargetParameters>,
    ) -> Result<PreparedTargets> {
        // FIXED: Data length should be the number of sequences, not original data length
        let data_length = sequence_indices.len();
        let mut prepared_targets = PreparedTargets::new(data_length);

        log::info!(
            "🎯 Generating aligned targets for {} sequences at specific indices",
            sequence_indices.len()
        );

        // PARALLELIZED: Generate all target types concurrently
        let default_config = TargetsConfig::default();
        let targets_config = model_config
            .map(|cfg| &cfg.targets)
            .unwrap_or(&default_config);

        let direction_adaptive_params = adaptive_params.map(|p| &p.direction);
        let price_level_adaptive_params = adaptive_params.map(|p| &p.price_levels);
        let volatility_adaptive_params = adaptive_params.map(|p| &p.volatility);
        let sentiment_adaptive_params = adaptive_params.map(|p| &p.sentiment);
        let volume_adaptive_params = adaptive_params.map(|p| &p.volume);

        let (
            price_targets,
            (direction_targets, (volatility_targets, (sentiment_targets, volume_targets))),
        ) = rayon::join(
            || {
                log::debug!("Generating price level targets in parallel");
                generate_price_level_targets_with_adaptive_params(
                    df,
                    &self.config.horizons,
                    targets_config,
                    sequence_indices,
                    sequence_length,
                    price_level_adaptive_params,
                )
            },
            || {
                rayon::join(
                    || {
                        log::debug!("Generating direction targets in parallel");
                        generate_direction_targets_with_adaptive_params(
                            df,
                            &self.config.horizons,
                            targets_config,
                            sequence_indices,
                            sequence_length,
                            direction_adaptive_params,
                        )
                    },
                    || {
                        rayon::join(
                            || {
                                log::debug!("Generating volatility targets in parallel");
                                generate_volatility_targets_with_adaptive_params(
                                    df,
                                    &self.config.horizons,
                                    targets_config,
                                    sequence_indices,
                                    sequence_length,
                                    volatility_adaptive_params,
                                )
                            },
                            || {
                                rayon::join(
                                    || {
                                        log::debug!("Generating sentiment targets in parallel");
                                        generate_sentiment_targets_with_adaptive_params(
                                            df,
                                            &self.config.horizons,
                                            targets_config,
                                            sequence_indices,
                                            sequence_length,
                                            sentiment_adaptive_params,
                                        )
                                    },
                                    || {
                                        log::debug!("Generating volume targets in parallel");
                                        generate_volume_targets_with_adaptive_params(
                                            df,
                                            &self.config.horizons,
                                            targets_config,
                                            sequence_indices,
                                            sequence_length,
                                            volume_adaptive_params,
                                        )
                                    },
                                )
                            },
                        )
                    },
                )
            },
        );

        // Assign results
        prepared_targets.price_levels = price_targets?;
        prepared_targets.directions = direction_targets?;
        prepared_targets.volatility = volatility_targets?;
        prepared_targets.sentiment = sentiment_targets?;
        prepared_targets.volume = volume_targets?;

        // FIXED: Set target names to avoid redundant TargetGenerator creation
        prepared_targets.target_names = self.get_target_names();

        // FIXED: Calculate valid indices based on sequence alignment
        // Valid indices should be 0, 1, 2, ... sequence_count-1 (not original data indices)
        prepared_targets.valid_indices = (0..sequence_indices.len()).collect();

        // FIXED: Validate target-sequence alignment
        crate::utils::sequence_utils::validate_target_sequence_alignment(
            sequence_indices.len(),
            &prepared_targets.valid_indices,
            &(0..sequence_indices.len()).collect::<Vec<_>>(), // Use sequence positions, not original indices
            sequence_indices.len(),                           // Data length is now sequence count
        )?;

        // Validate targets
        prepared_targets.validate()?;

        log::info!(
            "✅ Successfully generated aligned targets with {} valid samples",
            prepared_targets.valid_indices.len()
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

/// Legacy methods for backward compatibility - DEPRECATED
/// These methods are kept for API compatibility but should not be used
/// Use the new generate_all_targets() method instead
impl TargetGenerator {
    /// Generate price level targets using model configuration
    pub async fn generate_price_level_targets_with_model_config(
        &self,
        df: &DataFrame,
        model_config: &crate::config::model::ModelConfig,
    ) -> Result<HashMap<String, Vec<i32>>> {
        log::info!(
            "Generating price level targets for {} horizons using model config",
            self.config.horizons.len()
        );

        // Calculate sequence parameters for legacy method
        let sequence_length = match &model_config.sequence_length {
            crate::config::model::SequenceLengthConfig::Fixed(len) => *len as usize,
            crate::config::model::SequenceLengthConfig::Auto { min_length, .. } => {
                *min_length as usize
            }
            crate::config::model::SequenceLengthConfig::Adaptive => 60,
        };

        // Calculate sequence indices for the data
        let data_length = df.height();
        let max_horizon_steps = 24; // Default horizon for "1h" with hourly data
        let step_size = 1; // Default step size

        let sequence_indices = crate::utils::sequence_utils::calculate_sequence_indices(
            data_length,
            sequence_length,
            step_size,
            max_horizon_steps,
        )?;

        generate_price_level_targets_with_targets_config(
            df,
            &self.config.horizons,
            &model_config.targets,
            &sequence_indices,
            sequence_length,
        )
    }
}
