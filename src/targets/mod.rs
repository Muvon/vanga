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
pub mod calibration; // New clean calibration module
pub mod direction;
pub mod generators;
pub mod interface;
pub mod price_levels;
pub mod registry;
pub mod sentiment;
pub mod sequence_reconstruction;
pub mod volatility;
pub mod volume;

#[cfg(test)]
mod direction_test;
#[cfg(test)]
mod math_consistency_test;
#[cfg(test)]
mod price_level_test;
#[cfg(test)]
mod volatility_test;
#[cfg(test)]
mod volume_test;

use crate::targets::adaptive_parameters::AdaptiveTargetParameters;
use crate::targets::interface::AdaptiveParameters;
use crate::targets::registry::TargetRegistry;
use crate::utils::error::Result;
use polars::prelude::*;
use rayon::prelude::*; // RESTORED: Parallel execution support
use std::collections::HashMap;

// Re-export configurations and adaptive parameters
pub use adaptive_parameters::{
    calculate_class_distribution_balance, AdaptiveParameterCalibrator, CalibrationMetadata,
    ClassDistributionBalance, DirectionAdaptiveParams, PriceLevelAdaptiveParams,
    SentimentAdaptiveParams, VolatilityAdaptiveParams, VolumeAdaptiveParams,
};
pub use direction::{generate_direction_targets_with_adaptive_params, Direction};
pub use price_levels::{
    generate_price_level_targets, generate_price_level_targets_with_adaptive_params,
    get_horizon_exponential_weighted_close, get_sequence_exponential_weighted_close,
    reconstruct_price_levels, PriceLevelConfig,
};
pub use sentiment::{
    generate_sentiment_targets_with_adaptive_params, get_sentiment_class_names, SentimentConfig,
};
pub use sequence_reconstruction::{
    SequenceAnalyzer, SequenceBoundaries, SequenceReconstructionConfig, SequenceReconstructor,
};
pub use volatility::generate_volatility_targets_with_adaptive_params;
pub use volume::{
    generate_volume_targets_with_adaptive_params, get_volume_class_names, VolumeConfig,
};

/// Comprehensive target configuration
#[derive(Debug, Clone)]
pub struct MultiTargetConfig {
    pub price_level_config: PriceLevelConfig,
    pub horizons: Vec<String>,
    pub price_levels: TargetTypeConfig,
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
            price_levels: TargetTypeConfig { enabled: true },
            direction: TargetTypeConfig { enabled: true },
            volatility: TargetTypeConfig { enabled: true },
            sentiment: TargetTypeConfig { enabled: true },
            volume: TargetTypeConfig { enabled: true },
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
            price_levels: TargetTypeConfig { enabled: true },
            direction: TargetTypeConfig { enabled: true },
            volatility: TargetTypeConfig { enabled: true },
            sentiment: TargetTypeConfig { enabled: true },
            volume: TargetTypeConfig { enabled: true },
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
    registry: TargetRegistry, // NEW: Add registry for trait-based approach
}

impl TargetGenerator {
    /// Create new target generator with configuration
    pub fn new(config: MultiTargetConfig) -> Self {
        Self {
            config,
            registry: TargetRegistry::new(), // NEW: Initialize registry
        }
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
        sequence_indices: &[usize],
        sequence_length: usize,
    ) -> Result<PreparedTargets> {
        self.generate_all_targets_with_adaptive_params(
            df,
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
        // Create default parameters for cases where adaptive_params is None
        let default_direction =
            crate::targets::adaptive_parameters::DirectionAdaptiveParams::default();
        let default_price_level =
            crate::targets::adaptive_parameters::PriceLevelAdaptiveParams::default();
        let default_volatility =
            crate::targets::adaptive_parameters::VolatilityAdaptiveParams::default();
        let default_sentiment =
            crate::targets::adaptive_parameters::SentimentAdaptiveParams::default();
        let default_volume = crate::targets::adaptive_parameters::VolumeAdaptiveParams::default();

        let direction_adaptive_params = adaptive_params
            .map(|p| &p.direction)
            .unwrap_or(&default_direction);
        let price_level_adaptive_params = adaptive_params
            .map(|p| &p.price_levels)
            .unwrap_or(&default_price_level);
        let volatility_adaptive_params = adaptive_params
            .map(|p| &p.volatility)
            .unwrap_or(&default_volatility);
        let sentiment_adaptive_params = adaptive_params
            .map(|p| &p.sentiment)
            .unwrap_or(&default_sentiment);
        let volume_adaptive_params = adaptive_params
            .map(|p| &p.volume)
            .unwrap_or(&default_volume);

        let (
            price_targets,
            (direction_targets, (volatility_targets, (sentiment_targets, volume_targets))),
        ) = rayon::join(
            || {
                log::debug!("Generating price level targets in parallel");
                generate_price_level_targets_with_adaptive_params(
                    df,
                    &self.config.horizons,
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
        prepared_targets.direction = direction_targets?;
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

    /// Generate all targets using trait-based approach (NEW METHOD)
    ///
    /// This method uses the new trait-based interface for target generation,
    /// providing better extensibility and cleaner code while maintaining
    /// full backward compatibility with existing logic.
    pub async fn generate_all_targets_trait_based(
        &self,
        df: &DataFrame,
        sequence_indices: &[usize],
        sequence_length: usize,
        adaptive_params: Option<&AdaptiveTargetParameters>,
    ) -> Result<PreparedTargets> {
        let data_length = sequence_indices.len();
        let mut prepared_targets = PreparedTargets::new(data_length);

        // Get enabled target generators from registry
        let enabled_generators = self.registry.get_enabled_generators(&self.config);

        log::info!(
            "🎯 Generating targets using {} enabled generators: [{}]",
            enabled_generators.len(),
            enabled_generators
                .iter()
                .map(|g| g.target_name())
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Generate all targets in parallel using trait interface
        // RESTORED: Parallel execution using rayon to match original performance
        let target_results: Result<Vec<_>> = enabled_generators
            .par_iter()
            .map(|generator| {
                let adaptive_param =
                    self.get_adaptive_param_for_target(generator.target_type(), adaptive_params);

                let result = generator.generate_targets(
                    df,
                    &self.config.horizons,
                    sequence_indices,
                    sequence_length,
                    adaptive_param,
                )?;

                Ok((generator.target_type(), result))
            })
            .collect();

        let target_results = target_results?;

        // Process results and merge into prepared_targets
        for (target_type, result) in target_results {
            self.merge_target_results(&mut prepared_targets, target_type, &result)?;
        }

        // Validate final results
        prepared_targets.validate()?;

        log::info!(
            "✅ Trait-based target generation completed: {} sequences across {} horizons",
            data_length,
            self.config.horizons.len()
        );

        Ok(prepared_targets)
    }

    /// Get adaptive parameter for specific target type
    fn get_adaptive_param_for_target<'a>(
        &self,
        target_type: &str,
        adaptive_params: Option<&'a AdaptiveTargetParameters>,
    ) -> Option<&'a dyn AdaptiveParameters> {
        adaptive_params.and_then(|params| match target_type {
            "price_levels" => Some(&params.price_levels as &dyn AdaptiveParameters),
            "direction" => Some(&params.direction as &dyn AdaptiveParameters),
            "volatility" => Some(&params.volatility as &dyn AdaptiveParameters),
            "sentiment" => Some(&params.sentiment as &dyn AdaptiveParameters),
            "volume" => Some(&params.volume as &dyn AdaptiveParameters),
            _ => None,
        })
    }

    /// Merge target results into prepared targets structure
    fn merge_target_results(
        &self,
        prepared_targets: &mut PreparedTargets,
        target_type: &str,
        target_map: &HashMap<String, Vec<i32>>,
    ) -> Result<()> {
        for (horizon, targets) in target_map {
            match target_type {
                "price_levels" => {
                    prepared_targets
                        .price_levels
                        .insert(horizon.clone(), targets.clone());
                }
                "direction" => {
                    prepared_targets
                        .direction
                        .insert(horizon.clone(), targets.clone());
                }
                "volatility" => {
                    prepared_targets
                        .volatility
                        .insert(horizon.clone(), targets.clone());
                }
                "sentiment" => {
                    prepared_targets
                        .sentiment
                        .insert(horizon.clone(), targets.clone());
                }
                "volume" => {
                    prepared_targets
                        .volume
                        .insert(horizon.clone(), targets.clone());
                }
                _ => {
                    log::warn!("Unknown target type: {}", target_type);
                }
            }
        }
        Ok(())
    }

    /// Get target registry (for testing and extensibility)
    pub fn get_registry(&self) -> &TargetRegistry {
        &self.registry
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
