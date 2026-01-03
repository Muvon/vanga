//! Adaptive per-class temperature scaling
//!
//! Optimizes temperature per class to minimize ECE (Expected Calibration Error).
//! Temperature scaling: softmax(logits / T) where T is learned per class.

use super::ece::{calculate_ece, calculate_per_class_ece};
use crate::utils::error::{Result, VangaError};
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

/// Adaptive temperature scaling with per-class optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveTemperatureScaling {
    /// Per-class temperatures (learned from validation data)
    pub temperatures: [f64; 5],
    /// ECE history for convergence tracking
    pub ece_history: Vec<f64>,
    /// Per-class ECE values
    pub per_class_ece: [f64; 5],
    /// Whether temperatures have been optimized
    pub is_optimized: bool,
}

impl Default for AdaptiveTemperatureScaling {
    fn default() -> Self {
        Self {
            temperatures: [1.0; 5], // Start with no scaling
            ece_history: Vec::new(),
            per_class_ece: [0.0; 5],
            is_optimized: false,
        }
    }
}

impl AdaptiveTemperatureScaling {
    pub fn new() -> Self {
        Self::default()
    }

    /// Optimize temperatures to minimize ECE using gradient descent
    ///
    /// Uses binary search per class to find optimal temperature that minimizes ECE
    pub fn optimize_temperatures(
        &mut self,
        logits: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<()> {
        let num_samples = logits.nrows();
        if num_samples == 0 {
            return Ok(());
        }

        if logits.ncols() != 5 || targets.ncols() != 5 {
            return Err(VangaError::DataError(format!(
                "Expected 5 classes, got logits: {}, targets: {}",
                logits.ncols(),
                targets.ncols()
            )));
        }

        log::info!("🌡️ Optimizing per-class temperatures via ECE minimization...");

        // Calculate initial ECE
        let predictions_initial = self.apply_temperature_to_logits(logits, &[1.0; 5])?;
        let initial_ece = calculate_ece(&predictions_initial, targets)?;
        log::info!("   Initial ECE (T=1.0 for all): {:.6}", initial_ece);

        // Optimize temperature for each class independently using binary search
        for class_idx in 0..5 {
            let optimal_temp =
                self.find_optimal_temperature_binary_search(logits, targets, class_idx)?;
            self.temperatures[class_idx] = optimal_temp;
        }

        // Calculate final ECE with optimized temperatures
        let predictions_final = self.apply_temperature_to_logits(logits, &self.temperatures)?;
        let final_ece = calculate_ece(&predictions_final, targets)?;
        self.ece_history.push(final_ece);

        // Calculate per-class ECE
        self.per_class_ece = calculate_per_class_ece(&predictions_final, targets)?;

        self.is_optimized = true;

        log::info!("   Final ECE (optimized): {:.6}", final_ece);
        log::info!(
            "   ECE improvement: {:.6} → {:.6} ({:.2}% reduction)",
            initial_ece,
            final_ece,
            ((initial_ece - final_ece) / initial_ece * 100.0).max(0.0)
        );
        log::info!("   Optimized temperatures: {:?}", self.temperatures);
        log::info!("   Per-class ECE: {:?}", self.per_class_ece);

        Ok(())
    }

    /// Find optimal temperature for a class using binary search
    fn find_optimal_temperature_binary_search(
        &self,
        logits: &Array2<f64>,
        targets: &Array2<f64>,
        class_idx: usize,
    ) -> Result<f64> {
        let mut low = 0.1; // Minimum temperature (sharper predictions)
        let mut high = 5.0; // Maximum temperature (softer predictions)
        let tolerance = 0.01;
        let max_iterations = 50;

        let mut best_temp = 1.0;
        let mut best_ece = f64::MAX;

        for _ in 0..max_iterations {
            if (high - low) < tolerance {
                break;
            }

            // Try three points: low, mid, high
            let mid = (low + high) / 2.0;
            let temps = [low, mid, high];

            for &temp in &temps {
                let mut test_temps = self.temperatures;
                test_temps[class_idx] = temp;

                let predictions = self.apply_temperature_to_logits(logits, &test_temps)?;
                let ece = calculate_ece(&predictions, targets)?;

                if ece < best_ece {
                    best_ece = ece;
                    best_temp = temp;
                }
            }

            // Narrow search range around best temperature
            if (best_temp - low).abs() < tolerance {
                high = mid;
            } else if (best_temp - high).abs() < tolerance {
                low = mid;
            } else {
                // Best is in middle, narrow both sides
                let range = (high - low) / 4.0;
                low = (best_temp - range).max(0.1);
                high = (best_temp + range).min(5.0);
            }
        }

        Ok(best_temp)
    }

    /// Apply temperature scaling to logits and return probabilities
    fn apply_temperature_to_logits(
        &self,
        logits: &Array2<f64>,
        temperatures: &[f64; 5],
    ) -> Result<Array2<f64>> {
        let num_samples = logits.nrows();
        let mut predictions = Array2::zeros((num_samples, 5));

        for (i, logit_row) in logits.axis_iter(Axis(0)).enumerate() {
            // Apply per-class temperature scaling
            let scaled_logits: Vec<f64> = logit_row
                .iter()
                .enumerate()
                .map(|(j, &x)| x / temperatures[j])
                .collect();

            // Softmax with numerical stability
            let max_logit = scaled_logits
                .iter()
                .fold(f64::NEG_INFINITY, |max, &val| max.max(val));
            let exp_sum: f64 = scaled_logits.iter().map(|&x| (x - max_logit).exp()).sum();

            for j in 0..5 {
                predictions[[i, j]] = ((scaled_logits[j] - max_logit).exp()) / exp_sum;
            }
        }

        Ok(predictions)
    }

    /// Apply temperature scaling to logits (for inference)
    pub fn apply_to_logits(&self, logits: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.is_optimized {
            // No optimization yet, return original logits
            return Ok(logits.clone());
        }

        self.apply_temperature_to_logits(logits, &self.temperatures)
    }

    /// Get current temperatures
    pub fn get_temperatures(&self) -> [f64; 5] {
        self.temperatures
    }

    /// Get per-class ECE values
    pub fn get_per_class_ece(&self) -> [f64; 5] {
        self.per_class_ece
    }

    /// Get latest overall ECE
    pub fn get_latest_ece(&self) -> Option<f64> {
        self.ece_history.last().copied()
    }

    /// Reset optimization state
    pub fn reset(&mut self) {
        self.temperatures = [1.0; 5];
        self.ece_history.clear();
        self.per_class_ece = [0.0; 5];
        self.is_optimized = false;
    }
}
