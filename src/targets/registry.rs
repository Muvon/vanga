//! Target registry for dynamic target management
//!
//! This module provides a registry system for managing target generators,
//! enabling runtime discovery, configuration-driven filtering, and extensibility.

use crate::targets::interface::TargetGenerator;
use crate::targets::MultiTargetConfig;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry for managing target generators
///
/// The registry maintains a collection of target generators and provides
/// methods for registration, discovery, and configuration-based filtering.
pub struct TargetRegistry {
    generators: HashMap<String, Arc<dyn TargetGenerator>>,
}

impl TargetRegistry {
    /// Create new registry with default target generators
    pub fn new() -> Self {
        let mut registry = Self {
            generators: HashMap::new(),
        };

        // Register all default target generators
        registry.register_default_targets();
        registry
    }

    /// Register all default target generators
    fn register_default_targets(&mut self) {
        use crate::targets::generators::{
            DirectionTargetGenerator, PriceLevelTargetGenerator, SentimentTargetGenerator,
            VolatilityTargetGenerator, VolumeTargetGenerator,
        };

        self.register("price_levels", Arc::new(PriceLevelTargetGenerator));
        self.register("direction", Arc::new(DirectionTargetGenerator));
        self.register("volatility", Arc::new(VolatilityTargetGenerator));
        self.register("sentiment", Arc::new(SentimentTargetGenerator));
        self.register("volume", Arc::new(VolumeTargetGenerator));
    }

    /// Register a target generator
    pub fn register(&mut self, target_type: &str, generator: Arc<dyn TargetGenerator>) {
        self.generators.insert(target_type.to_string(), generator);
    }

    /// Get a target generator by type
    pub fn get(&self, target_type: &str) -> Option<Arc<dyn TargetGenerator>> {
        self.generators.get(target_type).cloned()
    }

    /// Get all registered target types
    pub fn get_all_types(&self) -> Vec<String> {
        self.generators.keys().cloned().collect()
    }

    /// Get enabled target generators based on configuration
    ///
    /// This method filters the registered generators based on the
    /// enabled flags in the MultiTargetConfig.
    pub fn get_enabled_generators(
        &self,
        config: &MultiTargetConfig,
    ) -> Vec<Arc<dyn TargetGenerator>> {
        let mut enabled = Vec::new();

        // Check which targets are enabled in config
        if config.price_levels.enabled {
            if let Some(gen) = self.get("price_levels") {
                enabled.push(gen);
            }
        }
        if config.direction.enabled {
            if let Some(gen) = self.get("direction") {
                enabled.push(gen);
            }
        }
        if config.volatility.enabled {
            if let Some(gen) = self.get("volatility") {
                enabled.push(gen);
            }
        }
        if config.sentiment.enabled {
            if let Some(gen) = self.get("sentiment") {
                enabled.push(gen);
            }
        }
        if config.volume.enabled {
            if let Some(gen) = self.get("volume") {
                enabled.push(gen);
            }
        }

        enabled
    }

    /// Get target generator names for enabled targets
    pub fn get_enabled_target_names(&self, config: &MultiTargetConfig) -> Vec<String> {
        self.get_enabled_generators(config)
            .iter()
            .map(|gen| gen.target_name().to_string())
            .collect()
    }

    /// Get total number of registered generators
    pub fn len(&self) -> usize {
        self.generators.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }
}

impl Default for TargetRegistry {
    fn default() -> Self {
        Self::new()
    }
}
