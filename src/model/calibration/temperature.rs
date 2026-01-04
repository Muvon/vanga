//! Adaptive per-class temperature scaling
//!
//! Optimizes temperature per class to minimize NLL (Negative Log-Likelihood).
//! Temperature scaling: softmax(logits / T) where T is learned per class.
//!
//! **CRITICAL**: Based on Guo et al. 2017 "On Calibration of Modern Neural Networks",
//! temperature scaling should minimize NLL on validation set, NOT ECE.

use super::ece::{calculate_ece, calculate_per_class_ece};
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};
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

    /// Optimize temperatures to minimize NLL (Negative Log-Likelihood)
    ///
    /// **CRITICAL**: Temperature scaling should minimize NLL, NOT ECE.
    /// Research papers (Guo et al. 2017) show NLL minimization is the correct approach.
    /// Uses binary search per class to find optimal temperature that minimizes NLL.
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

        log::info!("🌡️ Optimizing temperatures (NLL minimization)...");

        // Calculate initial NLL and ECE
        let predictions_initial = self.apply_temperature_to_logits(logits, &[1.0; 5])?;
        let initial_nll = self.calculate_nll(&predictions_initial, targets)?;

        // Optimize temperature for each class independently using binary search on NLL
        for class_idx in 0..5 {
            let optimal_temp =
                self.find_optimal_temperature_binary_search_nll(logits, targets, class_idx)?;
            self.temperatures[class_idx] = optimal_temp;
        }

        // Calculate final NLL and ECE with optimized temperatures
        let predictions_final = self.apply_temperature_to_logits(logits, &self.temperatures)?;
        let final_nll = self.calculate_nll(&predictions_final, targets)?;
        let final_ece = calculate_ece(&predictions_final, targets)?;
        self.ece_history.push(final_ece);

        // Calculate per-class ECE
        self.per_class_ece = calculate_per_class_ece(&predictions_final, targets)?;

        self.is_optimized = true;

        let nll_reduction = ((initial_nll - final_nll) / initial_nll * 100.0).max(0.0);
        log::info!(
            "   NLL: {:.4} → {:.4} ({:.1}% reduction)",
            initial_nll,
            final_nll,
            nll_reduction
        );

        Ok(())
    }

    /// Find optimal temperature for a class using binary search on NLL
    fn find_optimal_temperature_binary_search_nll(
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
        let mut best_nll = f64::MAX;

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
                let nll = self.calculate_nll(&predictions, targets)?;

                if nll < best_nll {
                    best_nll = nll;
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

    /// Calculate Negative Log-Likelihood (NLL) loss
    ///
    /// This is the correct loss function for temperature scaling optimization.
    /// NLL = -Σ log(p_correct) where p_correct is the predicted probability of the true class.
    fn calculate_nll(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        let num_samples = predictions.nrows();
        if num_samples == 0 {
            return Ok(0.0);
        }

        let mut total_nll = 0.0;
        let epsilon = 1e-10; // Prevent log(0)

        for (pred_row, target_row) in predictions
            .axis_iter(ndarray::Axis(0))
            .zip(targets.axis_iter(ndarray::Axis(0)))
        {
            // Find true class
            let true_class = target_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap();

            // Get predicted probability for true class
            let pred_prob = pred_row[true_class].max(epsilon);

            // Add negative log-likelihood
            total_nll -= pred_prob.ln();
        }

        // Return average NLL
        Ok(total_nll / num_samples as f64)
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

    /// Apply temperature scaling to tensor logits (preserves gradients)
    ///
    /// This applies per-class temperature scaling directly to tensors,
    /// preserving gradient flow for training.
    pub fn apply_to_tensor(&self, logits: &Tensor, device: &Device) -> Result<Tensor> {
        if !self.is_optimized {
            return Ok(logits.clone());
        }

        let shape = logits.shape();
        if shape.dims().len() != 2 || shape.dims()[1] != 5 {
            return Err(VangaError::ModelError(format!(
                "Expected logits shape [batch, 5], got {:?}",
                shape.dims()
            )));
        }

        // Create temperature tensor [1, 5] for broadcasting
        let temps_f32: Vec<f32> = self.temperatures.iter().map(|&t| t as f32).collect();
        let temp_tensor = Tensor::from_vec(temps_f32, (1, 5), device)?;

        // Broadcast temperature to match logits shape and ensure contiguous
        let temp_broadcast = temp_tensor.broadcast_as(shape.dims())?.contiguous()?;
        let logits_contiguous = logits.contiguous()?;

        // Apply temperature scaling: logits / temperature
        let scaled_logits = logits_contiguous
            .broadcast_div(&temp_broadcast)?
            .contiguous()?;

        Ok(scaled_logits)
    }
}
