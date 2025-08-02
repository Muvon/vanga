//! Multi-target prediction system for cryptocurrency forecasting
//!
//! This module orchestrates the generation of multiple prediction targets:
//! - Price levels: Quantile-based price classification
//! - Direction: Up/down/sideways movement classification
//! - Volatility: Low/medium/high volatility regime classification

pub mod direction;
#[cfg(test)]
mod direction_test;
pub mod imbalance_mitigation;
#[cfg(test)]
mod math_consistency_test;
#[cfg(test)]
mod price_level_test;
pub mod price_levels;
pub mod sequence_reconstruction;
pub mod volatility;
#[cfg(test)]
mod volatility_test;

use crate::config::model::TargetsConfig;
use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

// Re-export configurations
pub use direction::{generate_direction_targets, Direction};
pub use price_levels::{
    generate_price_level_targets, generate_price_level_targets_from_model_config,
    generate_price_level_targets_with_targets_config, PriceLevelConfig,
};
pub use sequence_reconstruction::{
    SequenceAnalyzer, SequenceBoundaries, SequenceReconstructionConfig, SequenceReconstructor,
};
pub use volatility::generate_volatility_targets;

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
    pub target_names: Vec<String>, // ADDED: Avoid redundant TargetGenerator creation
    pub data_length: usize,
    pub valid_indices: Vec<usize>,
}

impl PreparedTargets {
    /// Create new empty PreparedTargets
    /// Create new empty PreparedTargets
    pub fn new(data_length: usize) -> Self {
        Self {
            price_levels: HashMap::new(),
            directions: HashMap::new(),
            volatility: HashMap::new(),
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
        }
    }

    /// Get all horizons available
    pub fn get_horizons(&self) -> Vec<String> {
        let mut horizons: Vec<String> = self.price_levels.keys().cloned().collect();
        horizons.sort();
        horizons
    }

    /// Validate targets consistency across all types
    pub fn validate(&self) -> Result<()> {
        let horizons = self.get_horizons();

        for horizon in &horizons {
            let price_len = self.price_levels.get(horizon).map(|v| v.len());
            let direction_len = self.directions.get(horizon).map(|v| v.len());
            let volatility_len = self.volatility.get(horizon).map(|v| v.len());

            if price_len != direction_len || direction_len != volatility_len {
                return Err(crate::utils::error::VangaError::DataError(format!(
                    "Target length mismatch for horizon {}: price={:?}, direction={:?}, volatility={:?}",
                    horizon, price_len, direction_len, volatility_len
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetType {
    PriceLevel,
    Direction,
    Volatility,
}

/// Target statistics container
#[derive(Debug, Clone)]
pub struct TargetStatistics {
    pub price_level_stats: HashMap<String, ClassDistribution>,
    pub direction_stats: HashMap<String, ClassDistribution>,
    pub volatility_stats: HashMap<String, ClassDistribution>,
}

impl TargetStatistics {
    fn new() -> Self {
        Self {
            price_level_stats: HashMap::new(),
            direction_stats: HashMap::new(),
            volatility_stats: HashMap::new(),
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

        names
    }

    /// Get the total number of targets that will be generated
    pub fn get_num_targets(&self) -> usize {
        // Each horizon generates: 1 price level + 1 direction + 1 volatility = 3 targets per horizon
        self.config.horizons.len() * 3
    }

    /// Generate all targets aligned with specific sequence indices (FIXED: for proper synchronization)
    pub async fn generate_all_targets(
        &self,
        df: &DataFrame,
        model_config: Option<&crate::config::model::ModelConfig>,
        sequence_indices: &[usize],
        sequence_length: usize,
    ) -> Result<PreparedTargets> {
        // FIXED: Data length should be the number of sequences, not original data length
        let data_length = sequence_indices.len();
        let mut prepared_targets = PreparedTargets::new(data_length);

        log::info!(
            "🎯 Generating aligned targets for {} sequences at specific indices",
            sequence_indices.len()
        );

        // PARALLELIZED: Generate all target types concurrently
        let (price_targets, (direction_targets, volatility_targets)) = rayon::join(
            || {
                log::debug!("Generating price level targets in parallel");
                generate_price_level_targets_with_targets_config(
                    df,
                    &self.config.horizons,
                    model_config
                        .map(|cfg| &cfg.targets)
                        .unwrap_or(&TargetsConfig::default()),
                    sequence_indices,
                    sequence_length,
                )
            },
            || {
                rayon::join(
                    || {
                        log::debug!("Generating direction targets in parallel");
                        generate_direction_targets(
                            df,
                            &self.config.horizons,
                            model_config
                                .map(|cfg| &cfg.targets)
                                .unwrap_or(&TargetsConfig::default()),
                            sequence_indices,
                            sequence_length,
                        )
                    },
                    || {
                        log::debug!("Generating volatility targets in parallel");
                        generate_volatility_targets(
                            df,
                            &self.config.horizons,
                            model_config
                                .map(|cfg| &cfg.targets)
                                .unwrap_or(&TargetsConfig::default()),
                            sequence_indices,
                            sequence_length,
                        )
                    },
                )
            },
        );

        // Assign results
        prepared_targets.price_levels = price_targets?;
        prepared_targets.directions = direction_targets?;
        prepared_targets.volatility = volatility_targets?;

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
