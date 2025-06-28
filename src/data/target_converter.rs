//! Convert prepared targets to training arrays
//!
//! This module handles the conversion from PreparedTargets (HashMap<String, Vec<i32>>)
//! to the Array2<f64> format expected by the LSTM model for training.

use crate::config::model::{OutputHeadsConfig, OutputSegments};
use crate::targets::PreparedTargets;
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;
use std::collections::HashMap;

/// Converts prepared targets to training arrays for LSTM model
pub struct TargetConverter {
    output_heads: OutputHeadsConfig,
    #[allow(dead_code)] // Used for future output parsing
    segments: OutputSegments,
    total_output_size: usize,
}

impl TargetConverter {
    /// Create new target converter with output configuration
    pub fn new(output_heads: OutputHeadsConfig) -> Self {
        let segments = output_heads.get_output_segments();
        let total_output_size = output_heads.calculate_total_output_size();

        Self {
            output_heads,
            segments,
            total_output_size,
        }
    }

    /// Convert prepared targets to training array format
    ///
    /// # Arguments
    /// * `targets` - Prepared targets from target generation
    /// * `valid_indices` - Indices of valid samples for training
    ///
    /// # Returns
    /// Array2<f64> with shape [samples, total_output_size] for LSTM training
    pub fn convert_to_training_array(
        &self,
        targets: &PreparedTargets,
        valid_indices: &[usize],
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
                    "1h", // Use default horizon for price levels
                    data_idx,
                    "price_levels",
                )?;

                // Convert to one-hot encoding
                let num_bins = self.output_heads.price_levels.bins as usize;
                if price_target < num_bins {
                    training_array[[sample_idx, output_idx + price_target]] = 1.0;
                }
                output_idx += num_bins;
            }

            // 2. Direction Head (Classification - One-hot encoding)
            if self.output_heads.direction.enabled {
                let direction_target = self.extract_target_value(
                    &targets.directions,
                    "1h", // Use default horizon for directions
                    data_idx,
                    "directions",
                )?;

                // Convert to one-hot encoding (0=DOWN, 1=SIDEWAYS, 2=UP)
                if direction_target < 3 {
                    training_array[[sample_idx, output_idx + direction_target]] = 1.0;
                }
                output_idx += 3;
            }

            // 3. Volatility Head (Regression - Direct values)
            if self.output_heads.volatility.enabled {
                for horizon in &self.output_heads.volatility.horizons {
                    let volatility_target = self.extract_target_value(
                        &targets.volatility,
                        horizon,
                        data_idx,
                        "volatility",
                    )?;

                    // Convert to normalized volatility value (0-1 range)
                    // Assuming volatility classes 0-4 map to 0.0-1.0
                    let normalized_volatility = volatility_target as f64 / 4.0;
                    training_array[[sample_idx, output_idx]] =
                        normalized_volatility.clamp(0.0, 1.0);
                    output_idx += 1;
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
    pub fn validate_targets(&self, targets: &PreparedTargets) -> Result<()> {
        // Check price levels if enabled
        if self.output_heads.price_levels.enabled && !targets.price_levels.contains_key("1h") {
            return Err(VangaError::ConfigError(
                "Missing price level targets for default horizon: 1h".to_string(),
            ));
        }

        // Check directions if enabled
        if self.output_heads.direction.enabled && !targets.directions.contains_key("1h") {
            return Err(VangaError::ConfigError(
                "Missing direction targets for default horizon: 1h".to_string(),
            ));
        }

        // Check volatility if enabled
        if self.output_heads.volatility.enabled {
            for horizon in &self.output_heads.volatility.horizons {
                if !targets.volatility.contains_key(horizon) {
                    return Err(VangaError::ConfigError(format!(
                        "Missing volatility targets for horizon: {}",
                        horizon
                    )));
                }
            }
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
                bins: 5,
                range_percent: 0.1,
                distribution_type: crate::config::model::DistributionType::Categorical,
            },
            direction: DirectionHead {
                enabled: true,
                threshold: 0.02,
                confidence_calibration: false,
            },
            volatility: VolatilityHead {
                enabled: true,
                method: crate::config::model::VolatilityPredictionMethod::Direct,
                horizons: vec!["1h".to_string()],
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
            .insert("1h".to_string(), vec![0, 1, 2, 0, 1]);
        targets
            .volatility
            .insert("1h".to_string(), vec![0, 1, 2, 3, 4]);
        targets.valid_indices = vec![0, 1, 2, 3, 4];
        targets
    }

    #[test]
    fn test_target_conversion() {
        let output_heads = create_test_output_heads();
        let converter = TargetConverter::new(output_heads);
        let targets = create_test_targets();

        let result = converter.convert_to_training_array(&targets, &targets.valid_indices);
        assert!(result.is_ok());

        let training_array = result.unwrap();
        assert_eq!(training_array.shape()[0], 5); // 5 samples
        assert_eq!(training_array.shape()[1], 9); // 5 + 3 + 1 outputs
    }

    #[test]
    fn test_target_validation() {
        let output_heads = create_test_output_heads();
        let converter = TargetConverter::new(output_heads);
        let targets = create_test_targets();

        let result = converter.validate_targets(&targets);
        assert!(result.is_ok());
    }
}
