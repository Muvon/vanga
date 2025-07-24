//! Multi-target prediction system for cryptocurrency forecasting
//!
//! This module orchestrates the generation of multiple prediction targets:
//! - Price levels: Quantile-based price classification
//! - Direction: Up/down/sideways movement classification
//! - Volatility: Low/medium/high volatility regime classification

pub mod direction;
pub mod imbalance_mitigation;
pub mod price_levels;
pub mod volatility;

use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

// Re-export configurations
pub use direction::{generate_direction_targets, Direction, DirectionConfig};
pub use price_levels::{
    generate_price_level_targets, generate_price_level_targets_from_model_config, PriceLevelConfig,
};
pub use volatility::{generate_volatility_targets, VolatilityConfig, VolatilityRegime};

/// Comprehensive target configuration
#[derive(Debug, Clone)]
pub struct MultiTargetConfig {
    pub price_level_config: PriceLevelConfig,
    pub direction_config: DirectionConfig,
    pub volatility_config: VolatilityConfig,
    pub horizons: Vec<String>,
}

impl Default for MultiTargetConfig {
    fn default() -> Self {
        Self {
            price_level_config: PriceLevelConfig::default(),
            direction_config: DirectionConfig::default(),
            volatility_config: VolatilityConfig::default(),
            horizons: vec![
                "1h".to_string(),
                "4h".to_string(),
                "1d".to_string(),
                "7d".to_string(),
            ],
        }
    }
}

/// Container for all prepared targets
#[derive(Debug, Clone)]
pub struct PreparedTargets {
    pub price_levels: HashMap<String, Vec<i32>>,
    pub directions: HashMap<String, Vec<i32>>,
    pub volatility: HashMap<String, Vec<i32>>,
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

    /// Generate all targets for the given DataFrame
    pub async fn generate_all_targets(&self, df: &DataFrame) -> Result<PreparedTargets> {
        let data_length = df.height();
        let mut prepared_targets = PreparedTargets::new(data_length);

        log::info!(
            "Generating all targets in parallel for {} horizons",
            self.config.horizons.len()
        );

        // PARALLELIZED: Generate all target types concurrently using rayon::join
        let (price_targets, (direction_targets, volatility_targets)) = rayon::join(
            || {
                log::debug!("Generating price level targets in parallel");
                generate_price_level_targets(
                    df,
                    &self.config.horizons,
                    &self.config.price_level_config,
                )
            },
            || {
                rayon::join(
                    || {
                        log::debug!("Generating direction targets in parallel");
                        generate_direction_targets(
                            df,
                            &self.config.horizons,
                            &self.config.direction_config,
                        )
                    },
                    || {
                        log::debug!("Generating volatility targets in parallel");
                        generate_volatility_targets(
                            df,
                            &self.config.horizons,
                            &self.config.volatility_config,
                        )
                    },
                )
            },
        );

        // Assign results
        prepared_targets.price_levels = price_targets?;
        prepared_targets.directions = direction_targets?;
        prepared_targets.volatility = volatility_targets?;

        // Calculate valid indices (where all targets are available)
        prepared_targets.valid_indices = self.calculate_valid_indices(&prepared_targets)?;

        // Validate targets
        prepared_targets.validate()?;

        log::info!(
            "Successfully generated targets with {} valid samples",
            prepared_targets.valid_indices.len()
        );

        Ok(prepared_targets)
    }

    /// Calculate indices where all targets are valid (not -1)
    fn calculate_valid_indices(&self, targets: &PreparedTargets) -> Result<Vec<usize>> {
        let mut valid_indices = Vec::new();

        if targets.data_length == 0 {
            return Ok(valid_indices);
        }

        // Use first horizon to validate configuration consistency
        let first_horizon = self.config.horizons.first().ok_or_else(|| {
            crate::utils::error::VangaError::DataError("No horizons configured".to_string())
        })?;

        // Validate that the first horizon exists in all target types
        if !targets.price_levels.contains_key(first_horizon) {
            return Err(crate::utils::error::VangaError::DataError(format!(
                "First horizon '{}' missing from price level targets",
                first_horizon
            )));
        }
        if !targets.directions.contains_key(first_horizon) {
            return Err(crate::utils::error::VangaError::DataError(format!(
                "First horizon '{}' missing from direction targets",
                first_horizon
            )));
        }
        if !targets.volatility.contains_key(first_horizon) {
            return Err(crate::utils::error::VangaError::DataError(format!(
                "First horizon '{}' missing from volatility targets",
                first_horizon
            )));
        }

        for i in 0..targets.data_length {
            let mut all_valid = true;

            // Check if all target types have valid values for this index
            for horizon in &self.config.horizons {
                if let Some(price_targets) = targets.price_levels.get(horizon) {
                    if i >= price_targets.len() || price_targets[i] < 0 {
                        all_valid = false;
                        break;
                    }
                }

                if let Some(direction_targets) = targets.directions.get(horizon) {
                    if i >= direction_targets.len() || direction_targets[i] < 0 {
                        all_valid = false;
                        break;
                    }
                }

                if let Some(volatility_targets) = targets.volatility.get(horizon) {
                    if i >= volatility_targets.len() || volatility_targets[i] < 0 {
                        all_valid = false;
                        break;
                    }
                }
            }

            if all_valid {
                valid_indices.push(i);
            }
        }

        Ok(valid_indices)
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
    /// Generate price level targets (DEPRECATED - use generate_all_targets)
    #[deprecated(note = "Use generate_all_targets() instead")]
    pub fn generate_price_level_targets_legacy(
        &self,
        prices: &[f64],
        _bins: u32,          // Deprecated parameter, ignored
        _range_percent: f64, // Acknowledge unused parameter
    ) -> Result<Vec<Vec<f64>>> {
        log::warn!(
            "DEPRECATED: Use generate_all_targets() instead of legacy price level generation"
        );

        // For backward compatibility, create a temporary DataFrame and delegate to the working implementation
        if prices.is_empty() {
            return Err(crate::utils::error::VangaError::DataError(
                "Empty price data provided to deprecated method".to_string(),
            ));
        }

        // Create a minimal DataFrame for compatibility
        let df =
            polars::prelude::DataFrame::new(vec![polars::prelude::Series::new("close", prices)])
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to create DataFrame for backward compatibility: {}",
                        e
                    ))
                })?;

        // Use the working implementation with adapted config
        let config = PriceLevelConfig {
            bandwidth_size: 1.0, // Default bandwidth size for backward compatibility
        };

        let targets = generate_price_level_targets(&df, &self.config.horizons, &config)?;

        // Convert HashMap<String, Vec<i32>> to Vec<Vec<f64>> for backward compatibility
        let mut result = Vec::new();
        for horizon in &self.config.horizons {
            if let Some(horizon_targets) = targets.get(horizon) {
                let float_targets: Vec<f64> = horizon_targets.iter().map(|&x| x as f64).collect();
                result.push(float_targets);
            }
        }

        Ok(result)
    }

    /// Generate direction targets (DEPRECATED - use generate_all_targets)
    #[deprecated(note = "Use generate_all_targets() instead")]
    pub fn generate_direction_targets(&self, prices: &[f64], threshold: f64) -> Result<Vec<f64>> {
        log::warn!("DEPRECATED: Use generate_all_targets() instead of legacy direction generation");

        // For backward compatibility, create a temporary DataFrame and delegate to the working implementation
        if prices.is_empty() {
            return Err(crate::utils::error::VangaError::DataError(
                "Empty price data provided to deprecated method".to_string(),
            ));
        }

        // Create a minimal DataFrame for compatibility
        let df =
            polars::prelude::DataFrame::new(vec![polars::prelude::Series::new("close", prices)])
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to create DataFrame for backward compatibility: {}",
                        e
                    ))
                })?;

        // Use the working implementation with adapted config
        let config = crate::targets::direction::DirectionConfig {
            up_threshold: threshold,
            down_threshold: -threshold,
            ..Default::default()
        };

        let targets = crate::targets::direction::generate_direction_targets(
            &df,
            &self.config.horizons,
            &config,
        )?;

        // Convert HashMap<String, Vec<i32>> to Vec<f64> for backward compatibility
        // Take the first horizon's targets or return empty if none
        if let Some(first_horizon) = self.config.horizons.first() {
            if let Some(horizon_targets) = targets.get(first_horizon) {
                let float_targets: Vec<f64> = horizon_targets.iter().map(|&x| x as f64).collect();
                return Ok(float_targets);
            }
        }

        Ok(vec![])
    }

    /// Generate volatility targets (DEPRECATED - use generate_all_targets)
    #[deprecated(note = "Use generate_all_targets() instead")]
    pub fn generate_volatility_targets(
        &self,
        prices: &[f64],
        horizons: &[String],
    ) -> Result<Vec<Vec<f64>>> {
        log::warn!(
            "DEPRECATED: Use generate_all_targets() instead of legacy volatility generation"
        );

        // For backward compatibility, create a temporary DataFrame and delegate to the working implementation
        if prices.is_empty() {
            return Err(crate::utils::error::VangaError::DataError(
                "Empty price data provided to deprecated method".to_string(),
            ));
        }

        // Create a minimal DataFrame for compatibility
        let df =
            polars::prelude::DataFrame::new(vec![polars::prelude::Series::new("close", prices)])
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to create DataFrame for backward compatibility: {}",
                        e
                    ))
                })?;

        // Use the working implementation with default config
        let config = crate::targets::volatility::VolatilityConfig::default();

        let targets =
            crate::targets::volatility::generate_volatility_targets(&df, horizons, &config)?;

        // Convert HashMap<String, Vec<i32>> to Vec<Vec<f64>> for backward compatibility
        let mut result = Vec::new();
        for horizon in horizons {
            if let Some(horizon_targets) = targets.get(horizon) {
                let float_targets: Vec<f64> = horizon_targets.iter().map(|&x| x as f64).collect();
                result.push(float_targets);
            }
        }

        Ok(result)
    }

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

        generate_price_level_targets_from_model_config(df, &self.config.horizons, model_config)
    }
}
