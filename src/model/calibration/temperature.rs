//! Adaptive temperature scaling
//!
//! Optimizes single shared temperature to minimize NLL (Negative Log-Likelihood).
//! Temperature scaling: softmax(logits / T) where T is learned from validation data.
//!
//! Uses single shared temperature for all classes (standard calibration approach).
//!
//! ## Entropy-Based Adaptive Temperature (2024 Research)
//!
//! Temperature can be dynamically adjusted based on prediction uncertainty:
//! - **High entropy** (uncertain predictions): warmer temperature → softer probabilities
//! - **Low entropy** (confident predictions): sharper temperature → more concentrated
//!
//! Formula: `T_adaptive = T_base * (0.75 + 0.5 * confidence)` where `confidence = 1 - entropy`
//!
//! This prevents overconfidence on uncertain predictions while sharpening confident ones.

use super::ece::{calculate_ece, calculate_per_class_ece};
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

/// Adaptive temperature scaling with single shared temperature
///
/// Supports both static temperature scaling (learned from validation) and
/// entropy-based adaptive temperature (2024 research) for dynamic adjustment
/// based on prediction uncertainty.
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
        // CRITICAL: Also reject truly pathological temperatures (T > 5.0 or T < 0.5)
        // Research shows T can legitimately be in range [0.5, 5.0] per Guo et al. 2017
        let predictions_optimized = self.apply_temperature_to_logits(logits, optimal_temp)?;
        let optimized_nll = self.calculate_nll(&predictions_optimized, targets)?;

        // Check if optimal temperature is truly pathological
        // T > 5.0 makes probabilities nearly uniform (0.2 each) - indicates model issues
        // T < 0.5 is also unusual and indicates calibration problems
        let is_truly_pathological = optimal_temp > 5.0 || optimal_temp < 0.5;

        // Use baseline temperature if no improvement OR if temperature is truly pathological
        if optimized_nll >= initial_nll || is_truly_pathological {
            if is_truly_pathological {
                log::info!(
                    "   ⚠️ Pathological temperature {:.4} detected, keeping T=1.0",
                    optimal_temp
                );
            } else {
                log::info!(
                    "   No improvement found (NLL {:.4} >= {:.4}), keeping T=1.0",
                    optimized_nll,
                    initial_nll
                );
            }
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

    /// Find optimal temperature using unconstrained optimization with softplus constraint
    ///
    /// Based on WACV 2024 (Krumpl et al.) and Neural Computing 2024 (Balanya et al.)
    /// Modern approach: optimize raw parameter, apply softplus to ensure T > 0
    /// This avoids arbitrary bounds and naturally handles the positivity constraint
    fn find_optimal_temperature_binary_search(
        &self,
        logits: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<f64> {
        // Softplus-based temperature: T = softplus(raw) = ln(1 + exp(raw))
        // This ensures T > 0 without arbitrary bounds
        // - raw = 0 → T ≈ 0.693 (slightly softening)
        // - raw = 2 → T ≈ 1.31 (moderate softening)
        // - raw = 5 → T ≈ 148 (extreme - indicates model issues)
        // - raw negative → T < 1 (sharpening)
        const PRECISION: f64 = 0.01;
        const MAX_ITERATIONS: u32 = 50;

        // Search range for raw parameter (unconstrained)
        // This maps to reasonable temperature range after softplus
        const RAW_MIN: f64 = -5.0; // T ≈ 0.006 (very sharp)
        const RAW_MAX: f64 = 5.0; // T ≈ 148 (very soft)

        let mut low = RAW_MIN;
        let mut high = RAW_MAX;
        let mut best_raw = 0.0; // Start with T ≈ 0.693
        let mut best_nll = f64::INFINITY;

        for _ in 0..MAX_ITERATIONS {
            if high - low < PRECISION {
                break;
            }

            // Try three points: low, mid, high
            let mid = (low + high) / 2.0;
            let temps = [low, mid, high];

            for &raw in &temps {
                // Apply softplus: T = ln(1 + exp(raw))
                let temp = (raw.exp() + 1.0).ln();
                let predictions = self.apply_temperature_to_logits(logits, temp)?;
                let nll = self.calculate_nll(&predictions, targets)?;

                log::trace!("   Raw={:.4}, Temp={:.4}: NLL={:.6}", raw, temp, nll);

                if nll < best_nll {
                    best_nll = nll;
                    best_raw = raw;
                }
            }

            // Narrow search range around best raw value
            if (best_raw - low).abs() < PRECISION {
                high = mid;
            } else if (best_raw - high).abs() < PRECISION {
                low = mid;
            } else {
                // Best is in middle, narrow both sides
                let range = (high - low) / 4.0;
                low = (best_raw - range).max(RAW_MIN);
                high = (best_raw + range).min(RAW_MAX);
            }
        }

        // Convert raw to temperature using softplus
        let optimal_temp = (best_raw.exp() + 1.0).ln();

        // Validate: warn if temperature is pathological
        // T > 5.0 or T < 0.5 indicates model calibration issues
        if optimal_temp > 5.0 {
            log::warn!(
                "   ⚠️ Very high temperature {:.4} detected - model severely overconfident",
                optimal_temp
            );
        } else if optimal_temp < 0.5 {
            log::warn!(
                "   ⚠️ Very low temperature {:.4} detected - model severely underconfident",
                optimal_temp
            );
        }

        log::info!(
            "   Binary search: raw={:.4}, temperature={:.4} (NLL={:.6})",
            best_raw,
            optimal_temp,
            best_nll
        );

        Ok(optimal_temp)
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

    /// Calculate prediction entropy for a batch of predictions
    ///
    /// Entropy measures uncertainty: high entropy = uncertain predictions,
    /// low entropy = confident predictions.
    /// Range: [0, ln(5)] ≈ [0, 1.609]
    pub fn calculate_entropy(&self, predictions: &Array2<f64>) -> f64 {
        let num_samples = predictions.nrows();
        if num_samples == 0 {
            return 0.0;
        }

        let epsilon = 1e-10;
        let log_base = (5.0_f64).ln(); // Natural log of number of classes

        let mut total_entropy = 0.0;

        for pred_row in predictions.axis_iter(ndarray::Axis(0)) {
            let mut row_entropy = 0.0;
            for &p in pred_row.iter() {
                // Clamp probability to avoid log(0)
                let prob = p.max(epsilon).min(1.0 - epsilon);
                row_entropy -= prob * prob.ln();
            }
            // Normalize by log(5) to get value in [0, 1]
            total_entropy += row_entropy / log_base;
        }

        total_entropy / num_samples as f64
    }

    /// Calculate batch-wise entropies for per-sample adaptive temperature
    ///
    /// Returns entropies for each sample in the batch.
    /// High entropy = uncertain prediction, low entropy = confident.
    pub fn calculate_batch_entropies(&self, predictions: &Array2<f64>) -> Vec<f64> {
        let num_samples = predictions.nrows();
        if num_samples == 0 {
            return Vec::new();
        }

        let epsilon = 1e-10;
        let log_base = (5.0_f64).ln();

        let mut entropies = Vec::with_capacity(num_samples);

        for pred_row in predictions.axis_iter(ndarray::Axis(0)) {
            let mut row_entropy = 0.0;
            for &p in pred_row.iter() {
                let prob = p.max(epsilon).min(1.0 - epsilon);
                row_entropy -= prob * prob.ln();
            }
            // Normalize to [0, 1]
            entropies.push(row_entropy / log_base);
        }

        entropies
    }

    /// Apply entropy-based adaptive temperature scaling
    ///
    /// This 2024 research approach adapts temperature based on prediction entropy:
    /// - High entropy (uncertain): warmer temperature → softer probabilities
    /// - Low entropy (confident): sharper temperature → more concentrated
    ///
    /// Formula: T_adaptive = T_base * (0.75 + 0.5 * confidence)
    /// Where confidence = 1 - normalized_entropy
    /// Resulting range: [0.75 * T_base, 1.25 * T_base]
    ///
    /// Args:
    ///     logits: Raw model logits [batch, 5]
    ///     base_temperature: Optimized temperature from validation calibration
    ///     ramp_factor: Gradual ramp-up factor [0, 1] from training loop
    ///
    /// Returns:
    ///     Tuple of (adapted_predictions, average_entropy) for logging
    pub fn apply_entropy_adaptive_temperature(
        &self,
        logits: &Array2<f64>,
        base_temperature: f64,
        ramp_factor: f64,
    ) -> Result<(Array2<f64>, f64)> {
        // First apply base temperature to get probabilities
        let predictions = self.apply_temperature_to_logits(logits, base_temperature)?;

        // Calculate entropy for each sample
        let entropies = self.calculate_batch_entropies(&predictions);
        let avg_entropy = entropies.iter().sum::<f64>() / entropies.len() as f64;

        // Confidence is inverse of entropy (normalized to [0, 1])
        // entropy=1 (max) → confidence=0 (uncertain)
        // entropy=0 (min) → confidence=1 (confident)
        let confidence = 1.0 - avg_entropy;

        // Adaptive temperature: higher for uncertain, lower for confident
        // Range: 0.75×T_base to 1.25×T_base based on confidence
        let adaptive_factor = 0.75 + 0.5 * confidence;
        let adaptive_temperature = base_temperature * adaptive_factor;

        // If ramp factor < 1, interpolate between T=1 (no scaling) and adaptive T
        let ramped_temperature = if ramp_factor < 1.0 {
            1.0 + (adaptive_temperature - 1.0) * ramp_factor
        } else {
            adaptive_temperature
        };

        // Re-apply with adaptive temperature
        let adapted_predictions = self.apply_temperature_to_logits(logits, ramped_temperature)?;

        Ok((adapted_predictions, avg_entropy))
    }

    /// Get temperature adjustment info for logging
    ///
    /// Returns tuple of (adaptive_factor, confidence, entropy) for monitoring
    pub fn get_temperature_adjustment_info(&self, predictions: &Array2<f64>) -> (f64, f64, f64) {
        let entropy = self.calculate_entropy(predictions);
        let confidence = 1.0 - entropy;
        let adaptive_factor = 0.75 + 0.5 * confidence;
        (adaptive_factor, confidence, entropy)
    }
}
