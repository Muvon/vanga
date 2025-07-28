//! Convert prepared targets to training arrays
//!
//! This module handles the conversion from PreparedTargets (HashMap<String, Vec<i32>>)
//! to the Array2<f64> format expected by the LSTM model for training.

use crate::config::model::{OutputHeadsConfig, NUM_CLASSES};
use crate::targets::PreparedTargets;
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;
use std::collections::HashMap;

/// Converts prepared targets to training arrays for LSTM model
pub struct TargetConverter {
    output_heads: OutputHeadsConfig,
    total_output_size: usize,
}

impl TargetConverter {
    /// Create new target converter with output configuration
    pub fn new(output_heads: OutputHeadsConfig) -> Self {
        let total_output_size = output_heads.calculate_total_output_size();

        Self {
            output_heads,
            total_output_size,
        }
    }

    /// Convert prepared targets to training array format
    pub fn convert_to_training_array(
        &self,
        targets: &PreparedTargets,
        valid_indices: &[usize],
        horizon: &str,
    ) -> Result<Array2<f64>> {
        let num_samples = valid_indices.len();

        if num_samples == 0 {
            return Err(VangaError::DataError(
                "No valid indices provided for target conversion".to_string(),
            ));
        }

        // Initialize output array
        let mut training_array = Array2::<f64>::zeros((num_samples, self.total_output_size));

        // Process each enabled prediction head
        for (sample_idx, &data_idx) in valid_indices.iter().enumerate() {
            let mut output_idx = 0;

            // 1. Price Level Head (Classification - One-hot encoding)
            if self.output_heads.price_levels.enabled {
                let price_target = self.extract_target_value(
                    &targets.price_levels,
                    horizon,
                    data_idx,
                    "price_levels",
                )?;

                // Convert to one-hot encoding (sequence-aware classification uses NUM_CLASSES)
                let num_bins = NUM_CLASSES;
                if price_target < num_bins {
                    training_array[[sample_idx, output_idx + price_target]] = 1.0;
                }
                output_idx += num_bins;
            }

            // 2. Direction Head (Classification - One-hot encoding)
            if self.output_heads.direction.enabled {
                let direction_target = self.extract_target_value(
                    &targets.directions,
                    horizon,
                    data_idx,
                    "directions",
                )?;

                // Convert to one-hot encoding (5-class system)
                if direction_target < NUM_CLASSES {
                    training_array[[sample_idx, output_idx + direction_target]] = 1.0;
                }
                output_idx += NUM_CLASSES;
            }

            // 3. Volatility Head (Classification - One-hot encoding)
            if self.output_heads.volatility.enabled {
                let volatility_target = self.extract_target_value(
                    &targets.volatility, // CORRECTED: Was using targets.directions
                    horizon,
                    data_idx,
                    "volatility",
                )?;

                // Convert to one-hot encoding (5-class system)
                if volatility_target < NUM_CLASSES {
                    training_array[[sample_idx, output_idx + volatility_target]] = 1.0;
                }
            }
        }

        log::debug!(
            "Converted targets to training array: {} samples x {} outputs",
            num_samples,
            self.total_output_size
        );

        Ok(training_array)
    }

    /// Extract target value from HashMap for specific horizon and data index
    fn extract_target_value(
        &self,
        target_map: &HashMap<String, Vec<i32>>,
        horizon: &str,
        data_idx: usize,
        target_type: &str,
    ) -> Result<usize> {
        let target_vec = target_map.get(horizon).ok_or_else(|| {
            VangaError::DataError(format!(
                "Missing {} targets for horizon: {}",
                target_type, horizon
            ))
        })?;

        if data_idx >= target_vec.len() {
            return Err(VangaError::DataError(format!(
                "Data index {} out of bounds for {} targets (length: {})",
                data_idx,
                target_type,
                target_vec.len()
            )));
        }

        let target_value = target_vec[data_idx];
        if target_value < 0 {
            return Err(VangaError::DataError(format!(
                "Invalid negative target value {} for {} at index {}",
                target_value, target_type, data_idx
            )));
        }

        Ok(target_value as usize)
    }

    /// Validate that prepared targets are compatible with output configuration
    pub fn validate_targets(&self, targets: &PreparedTargets, horizon: &str) -> Result<()> {
        // Check price levels if enabled
        if self.output_heads.price_levels.enabled && !targets.price_levels.contains_key(horizon) {
            return Err(VangaError::ConfigError(format!(
                "Missing price level targets for horizon: {}",
                horizon
            )));
        }

        // Check directions if enabled
        if self.output_heads.direction.enabled && !targets.directions.contains_key(horizon) {
            return Err(VangaError::ConfigError(format!(
                "Missing direction targets for horizon: {}",
                horizon
            )));
        }

        // Check volatility if enabled
        if self.output_heads.volatility.enabled && !targets.volatility.contains_key(horizon) {
            return Err(VangaError::ConfigError(format!(
                "Missing volatility targets for horizon: {}", // CORRECTED: Was 'direction'
                horizon
            )));
        }

        log::debug!("Target validation passed for all enabled heads");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{DirectionHead, PriceLevelHead, VolatilityHead};

    fn create_test_output_heads() -> OutputHeadsConfig {
        OutputHeadsConfig {
            price_levels: PriceLevelHead {
                enabled: true,
                bandwidth_size: Some(1.0), // Default bandwidth size for testing
                distribution_type: crate::config::model::DistributionType::Categorical,
            },
            direction: DirectionHead {
                enabled: true,
                bandwidth_size: Some(0.8),
                base_threshold_factor: 0.5,
                extreme_multiplier: 2.5,
            },
            volatility: VolatilityHead {
                enabled: true,
                bandwidth_size: Some(1.2),
            },
        }
    }

    fn create_test_targets() -> PreparedTargets {
        let mut targets = PreparedTargets::new(100);
        targets
            .price_levels
            .insert("1h".to_string(), vec![0, 1, 2, 3, 4]);
        targets
            .directions
            .insert("1h".to_string(), vec![0, 1, 2, 3, 4]); // Use full 5-class range
        targets
            .volatility
            .insert("1h".to_string(), vec![4, 3, 2, 1, 0]); // Use full 5-class range
        targets.valid_indices = vec![0, 1, 2, 3, 4];
        targets
    }

    #[test]
    fn test_target_conversion() {
        let output_heads = create_test_output_heads();
        let converter = TargetConverter::new(output_heads);
        let targets = create_test_targets();

        let result = converter.convert_to_training_array(&targets, &targets.valid_indices, "1h");
        assert!(result.is_ok());

        let training_array = result.unwrap();
        assert_eq!(training_array.shape()[0], 5); // 5 samples
        assert_eq!(training_array.shape()[1], 15); // 5 (price) + 5 (direction) + 5 (volatility)
    }

    #[test]
    fn test_target_validation() {
        let output_heads = create_test_output_heads();
        let converter = TargetConverter::new(output_heads);
        let targets = create_test_targets();

        let result = converter.validate_targets(&targets, "1h");
        assert!(result.is_ok());
    }
}
