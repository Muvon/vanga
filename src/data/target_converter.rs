//! Convert prepared targets to training arrays
//!
//! This module handles the conversion from PreparedTargets (HashMap<String, Vec<i32>>)
//! to the Array2<f64> format expected by the LSTM model for training.

use crate::config::model::NUM_CLASSES;
use crate::targets::PreparedTargets;
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;
use std::collections::HashMap;

/// Converts prepared targets to training arrays for LSTM model
pub struct TargetConverter {
    total_output_size: usize,
}

impl Default for TargetConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl TargetConverter {
    /// Create new target converter with output configuration
    pub fn new() -> Self {
        let total_output_size = NUM_CLASSES * 3;

        Self { total_output_size }
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

            // 2. Direction Head (Classification - One-hot encoding)
            let direction_target =
                self.extract_target_value(&targets.direction, horizon, data_idx, "directions")?;

            // Convert to one-hot encoding (5-class system)
            if direction_target < NUM_CLASSES {
                training_array[[sample_idx, output_idx + direction_target]] = 1.0;
            }
            output_idx += NUM_CLASSES;

            // 3. Volatility Head (Classification - One-hot encoding)
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
        if !targets.price_levels.contains_key(horizon) {
            return Err(VangaError::ConfigError(format!(
                "Missing price level targets for horizon: {}",
                horizon
            )));
        }

        // Check directions if enabled
        if !targets.direction.contains_key(horizon) {
            return Err(VangaError::ConfigError(format!(
                "Missing direction targets for horizon: {}",
                horizon
            )));
        }

        // Check volatility if enabled
        if !targets.volatility.contains_key(horizon) {
            return Err(VangaError::ConfigError(format!(
                "Missing volatility targets for horizon: {}", // CORRECTED: Was 'direction'
                horizon
            )));
        }

        log::debug!("Target validation passed for all enabled heads");
        Ok(())
    }
}
