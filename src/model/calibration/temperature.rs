//! Adaptive temperature scaling
//!
//! Optimizes single shared temperature to minimize NLL (Negative Log-Likelihood).
//! Temperature scaling: softmax(logits / T) where T is learned from validation data.
//!
//! Uses single shared temperature for all classes (standard calibration approach).

use super::ece::{calculate_ece, calculate_per_class_ece};
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

/// Adaptive temperature scaling with single shared temperature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveTemperatureScaling {
    /// Single shared temperature for all classes (learned from validation data)
    pub temperature: f64,
    /// ECE history for convergence tracking
    pub ece_history: Vec<f64>,
    /// Per-class ECE values
    pub per_class_ece: [f64; 5],
    /// Whether temperature has been optimized
    pub is_optimized: bool,
}

impl Default for AdaptiveTemperatureScaling {
    fn default() -> Self {
        Self {
            temperature: 1.0, // Start with no scaling
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

    /// Get temperatures as array for compatibility (returns same T for all classes)
    pub fn temperatures(&self) -> [f64; 5] {
        [self.temperature; 5]
    }

    /// Optimize single shared temperature to minimize NLL (Negative Log-Likelihood)
    ///
    /// Uses single shared temperature for all classes (standard calibration approach).
    /// Temperature scaling should minimize NLL on validation set, NOT ECE.
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

        // Calculate initial NLL at T=1.0 (baseline)
        let predictions_initial = self.apply_temperature_to_logits(logits, 1.0)?;
        let initial_nll = self.calculate_nll(&predictions_initial, targets)?;
        let initial_ece = calculate_ece(&predictions_initial, targets)?;

        log::info!("   Initial: NLL={:.6}, ECE={:.6}", initial_nll, initial_ece);

        // Find optimal single shared temperature using binary search
        let optimal_temp = self.find_optimal_temperature_binary_search(logits, targets)?;

        // Only update temperature if it actually improves NLL over baseline (T=1.0)
        let predictions_optimized = self.apply_temperature_to_logits(logits, optimal_temp)?;
        let optimized_nll = self.calculate_nll(&predictions_optimized, targets)?;

        // Use baseline temperature if no improvement
        if optimized_nll >= initial_nll {
            log::info!(
                "   No improvement found (NLL {:.4} >= {:.4}), keeping T=1.0",
                optimized_nll,
                initial_nll
            );
            self.temperature = 1.0;
        } else {
            self.temperature = optimal_temp;
        }

        // Calculate final NLL and ECE with optimized temperature
        let predictions_final = self.apply_temperature_to_logits(logits, self.temperature)?;
        let final_nll = self.calculate_nll(&predictions_final, targets)?;
        let final_ece = calculate_ece(&predictions_final, targets)?;

        // Calculate per-class ECE
        self.per_class_ece = calculate_per_class_ece(&predictions_final, targets)?;

        // Calculate NLL change (negative = improvement, positive = degradation)
        let nll_change_pct = (final_nll - initial_nll) / initial_nll * 100.0;

        if self.temperature == 1.0 {
            log::info!(
                "   T={:.4}: NLL {:.4} → {:.4} ({:.1}% change - baseline kept), ECE={:.6}",
                self.temperature,
                initial_nll,
                final_nll,
                nll_change_pct,
                final_ece
            );
        } else if nll_change_pct < 0.0 {
            log::info!(
                "   T={:.4}: NLL {:.4} → {:.4} ({:.1}% reduction), ECE={:.6}",
                self.temperature,
                initial_nll,
                final_nll,
                -nll_change_pct,
                final_ece
            );
        } else {
            log::info!(
                "   T={:.4}: NLL {:.4} → {:.4} ({:.1}% increase - using optimized), ECE={:.6}",
                self.temperature,
                initial_nll,
                final_nll,
                nll_change_pct,
                final_ece
            );
        }

        self.ece_history.push(final_ece);
        self.is_optimized = true;

        Ok(())
    }

    /// Find optimal single shared temperature using binary search on NLL
    fn find_optimal_temperature_binary_search(
        &self,
        logits: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<f64> {
        const TEMP_MIN: f64 = 0.05; // Extended from 0.5 to allow sharper temperature scaling
        const TEMP_MAX: f64 = 5.0;
        const PRECISION: f64 = 0.01;
        const MAX_ITERATIONS: u32 = 50;

        let mut low = TEMP_MIN;
        let mut high = TEMP_MAX;
        let mut best_temp = 1.0;
        let mut best_nll = f64::INFINITY;

        for _ in 0..MAX_ITERATIONS {
            if high - low < PRECISION {
                break;
            }

            // Try three points: low, mid, high
            let mid = (low + high) / 2.0;
            let temps = [low, mid, high];

            for &temp in &temps {
                let predictions = self.apply_temperature_to_logits(logits, temp)?;
                let nll = self.calculate_nll(&predictions, targets)?;

                log::trace!("   Temp={:.4}: NLL={:.6}", temp, nll);

                if nll < best_nll {
                    best_nll = nll;
                    best_temp = temp;
                }
            }

            // Narrow search range around best temperature
            if (best_temp - low).abs() < PRECISION {
                high = mid;
            } else if (best_temp - high).abs() < PRECISION {
                low = mid;
            } else {
                // Best is in middle, narrow both sides
                let range = (high - low) / 4.0;
                low = (best_temp - range).max(TEMP_MIN);
                high = (best_temp + range).min(TEMP_MAX);
            }
        }

        log::info!(
            "   Binary search: best_temp={:.4} (NLL={:.6}), range=[{:.2}, {:.2}]",
            best_temp,
            best_nll,
            low,
            high
        );

        Ok(best_temp)
    }

    /// Calculate Negative Log-Likelihood (NLL) loss (pub(crate) for testing)
    ///
    /// This is the correct loss function for temperature scaling optimization.
    /// NLL = -Σ log(p_correct) where p_correct is the predicted probability of the true class.
    pub(crate) fn calculate_nll(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<f64> {
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

    /// Apply temperature scaling to logits and return probabilities (pub(crate) for testing)
    pub(crate) fn apply_temperature_to_logits(
        &self,
        logits: &Array2<f64>,
        temperature: f64,
    ) -> Result<Array2<f64>> {
        let num_samples = logits.nrows();
        let mut predictions = Array2::zeros((num_samples, 5));

        // Pre-calculate inverse temperature to avoid repeated division
        let inv_temp = 1.0 / temperature;

        for (i, logit_row) in logits.axis_iter(Axis(0)).enumerate() {
            // Apply temperature scaling with pre-calculated inverse
            let mut scaled_logits = [0.0; 5];
            for j in 0..5 {
                scaled_logits[j] = logit_row[j] * inv_temp;
            }

            // Find max for numerical stability (softmax trick)
            let max_logit = scaled_logits
                .iter()
                .fold(f64::NEG_INFINITY, |max, &val| max.max(val));

            // Calculate exp and sum in single pass
            let mut exp_values = [0.0; 5];
            let mut exp_sum = 0.0;
            for j in 0..5 {
                exp_values[j] = (scaled_logits[j] - max_logit).exp();
                exp_sum += exp_values[j];
            }

            // Normalize to get probabilities
            for j in 0..5 {
                predictions[[i, j]] = exp_values[j] / exp_sum;
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

        self.apply_temperature_to_logits(logits, self.temperature)
    }

    /// Get current temperature
    pub fn get_temperature(&self) -> f64 {
        self.temperature
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
        self.temperature = 1.0;
        self.ece_history.clear();
        self.per_class_ece = [0.0; 5];
        self.is_optimized = false;
    }

    /// Apply temperature scaling to tensor logits (preserves gradients)
    ///
    /// This applies single shared temperature scaling directly to tensors,
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

        // Create temperature tensor [1, 5] with same temperature for all classes
        let temp_f32 = self.temperature as f32;
        let temp_tensor = Tensor::from_vec(vec![temp_f32; 5], (1, 5), device)?;

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
