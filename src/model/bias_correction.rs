use crate::utils::error::Result;
use candle_core::Tensor;
use ndarray::{Array2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for bias correction system
/// Configuration for bias correction system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasCorrection {
    /// Enable/disable bias correction
    pub enabled: bool,
    /// Smoothing factor for bias factor calculation (0.0 = no smoothing, 1.0 = maximum smoothing)
    pub smoothing_factor: f64,
    /// Minimum and maximum allowed bias correction factors
    pub correction_bounds: [f64; 2],
    /// Minimum samples required for reliable bias correction
    pub min_samples: usize,
    /// Confidence adjustment factor
    pub confidence_adjustment: f64,

    /// Print detailed bias correction info (false = concise single-line summary)
    #[serde(default = "default_print_info")]
    pub print_info: bool,

    /// Ramp-up epochs for gradual integration into training
    #[serde(default = "default_ramp_up_epochs")]
    pub ramp_up_epochs: usize,

    /// Recalibration frequency (epochs between recalibration, 0 = no recalibration)
    #[serde(default = "default_recalibration_frequency")]
    pub recalibration_frequency: usize,
}

fn default_print_info() -> bool {
    false
}

fn default_ramp_up_epochs() -> usize {
    10
}

fn default_recalibration_frequency() -> usize {
    5
}

impl Default for BiasCorrection {
    fn default() -> Self {
        Self {
            enabled: true,
            smoothing_factor: 0.1,
            correction_bounds: [0.5, 2.0], // Prevent extreme corrections
            min_samples: 100,
            confidence_adjustment: 1.0,
            print_info: false, // Default to concise logging
            ramp_up_epochs: default_ramp_up_epochs(),
            recalibration_frequency: default_recalibration_frequency(),
        }
    }
}

/// Statistics for validation period used in bias correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    pub total_samples: usize,
    pub class_frequencies_predicted: [f64; 5],
    pub class_frequencies_actual: [f64; 5],
    pub overall_accuracy: f64,
    pub confidence_distribution: [f64; 5], // Confidence quartiles
}

/// Linear bias correction system adapted for 5-class classification
///
/// Based on the paper "Seeing Beyond Noise: Improving Cryptocurrency Forecasting with Linear Bias Correction"
/// but adapted from regression to classification by correcting class probability distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearBiasCorrector {
    /// Configuration for bias correction
    pub config: BiasCorrection,
    /// Bias correction factors for each class (0-4)
    pub class_bias_factors: [f64; 5],
    /// Overall confidence scaling factor
    pub confidence_scaling: f64,
    /// Validation statistics used for correction calculation
    pub validation_stats: Option<ValidationStats>,
    /// Whether the corrector has been calibrated
    pub is_calibrated: bool,

    // OPTIMIZATION: Cache frequently computed values
    /// Cached combined correction factors (bias * confidence)
    #[serde(skip)]
    cached_combined_factors: Option<[f64; 5]>,
    /// Cached adaptive strength for current calibration
    #[serde(skip)]
    cached_adaptive_strength: Option<f64>,
}

impl Default for LinearBiasCorrector {
    fn default() -> Self {
        Self {
            config: BiasCorrection::default(),
            class_bias_factors: [1.0; 5], // No correction initially
            confidence_scaling: 1.0,
            validation_stats: None,
            is_calibrated: false,
            cached_combined_factors: None,
            cached_adaptive_strength: None,
        }
    }
}

impl LinearBiasCorrector {
    /// Create new bias corrector with configuration
    pub fn new(config: BiasCorrection) -> Self {
        Self {
            config,
            class_bias_factors: [1.0; 5],
            confidence_scaling: 1.0,
            validation_stats: None,
            is_calibrated: false,
            cached_combined_factors: None,
            cached_adaptive_strength: None,
        }
    }

    /// Calculate bias correction factors from validation data
    ///
    /// This implements the core logic from the paper, adapted for classification:
    /// Original: V_corrected = V_predicted * (M_actual / M_predicted)
    /// Our adaptation: P_corrected[class] = P_predicted[class] * (freq_actual[class] / freq_predicted[class])
    pub fn calibrate_from_validation(
        &mut self,
        validation_predictions: &Array2<f64>, // [samples, 5_classes]
        validation_targets: &Array2<f64>,     // [samples, 5_classes] (one-hot or soft labels)
    ) -> Result<()> {
        if !self.config.enabled {
            log::info!("🔧 Bias correction disabled, skipping calibration");
            return Ok(());
        }

        let num_samples = validation_predictions.nrows();
        if num_samples < self.config.min_samples {
            log::warn!(
                "⚠️ Insufficient validation samples for bias correction: {} < {}",
                num_samples,
                self.config.min_samples
            );
            return Ok(());
        }

        // Only log if print_info is true
        if self.config.print_info {
            log::info!(
                "🎯 Calibrating bias correction from {} validation samples",
                num_samples
            );
        }

        // Validate input dimensions
        let num_classes = validation_predictions.ncols();
        let target_classes = validation_targets.ncols();

        if num_classes != 5 {
            log::error!("❌ Expected 5 classes in predictions, got {}", num_classes);
            return Err(crate::utils::error::VangaError::ModelError(format!(
                "Bias correction requires 5-class predictions, got {}",
                num_classes
            )));
        }

        if target_classes != 5 {
            log::error!("❌ Expected 5 classes in targets, got {}", target_classes);
            return Err(crate::utils::error::VangaError::ModelError(format!(
                "Bias correction requires 5-class targets, got {}",
                target_classes
            )));
        }

        if validation_predictions.nrows() != validation_targets.nrows() {
            log::error!(
                "❌ Predictions and targets have different number of samples: {} vs {}",
                validation_predictions.nrows(),
                validation_targets.nrows()
            );
            return Err(crate::utils::error::VangaError::ModelError(
                "Predictions and targets must have same number of samples".to_string(),
            ));
        }

        // VALIDATION: Check if predictions are proper probabilities (sum to ~1.0)
        // This helps catch issues where raw logits are passed instead of probabilities
        let mut sample_check_count = 0;
        let max_samples_to_check = std::cmp::min(10, validation_predictions.nrows());
        for row_idx in 0..max_samples_to_check {
            let row_sum: f64 = (0..5)
                .map(|col| validation_predictions[[row_idx, col]])
                .sum();

            if (row_sum - 1.0).abs() > 0.1 {
                sample_check_count += 1;
            }
        }

        if sample_check_count > max_samples_to_check / 2 {
            log::warn!(
                "⚠️ Predictions don't appear to be probabilities! Sample sums: {:?}",
                (0..std::cmp::min(3, validation_predictions.nrows()))
                    .map(|i| (0..5).map(|j| validation_predictions[[i, j]]).sum::<f64>())
                    .collect::<Vec<f64>>()
            );
            log::warn!("⚠️ Ensure softmax is applied to convert logits to probabilities before bias correction");
        }

        // Calculate predicted and actual class frequencies in parallel
        let frequencies: Vec<(f64, f64)> = (0..5)
            .into_par_iter()
            .map(|class_idx| {
                // Predicted frequency = average predicted probability for this class
                let pred_freq = validation_predictions
                    .column(class_idx)
                    .mean()
                    .unwrap_or(0.0);

                // Actual frequency = average actual probability/label for this class
                let actual_freq = validation_targets.column(class_idx).mean().unwrap_or(0.0);

                (pred_freq, actual_freq)
            })
            .collect();

        let mut predicted_frequencies = [0.0; 5];
        let mut actual_frequencies = [0.0; 5];
        for (idx, (pred, actual)) in frequencies.iter().enumerate() {
            predicted_frequencies[idx] = *pred;
            actual_frequencies[idx] = *actual;
        }

        // Debug: Check if frequencies sum to ~1.0 (they should for proper probabilities)
        let predicted_sum: f64 = predicted_frequencies.iter().sum();
        let actual_sum: f64 = actual_frequencies.iter().sum();

        if self.config.print_info {
            log::debug!(
                "📊 Frequency sums - Predicted: {:.4}, Actual: {:.4} (should be ~1.0)",
                predicted_sum,
                actual_sum
            );
        }

        // Warn if predicted frequencies don't sum to ~1.0
        if (predicted_sum - 1.0).abs() > 0.1 {
            log::warn!(
                "⚠️ Predicted frequencies sum to {:.4} instead of ~1.0. Likely receiving logits instead of probabilities!",
                predicted_sum
            );
        }

        // Calculate bias correction factors with bounds checking
        for class_idx in 0..5 {
            let predicted_freq = predicted_frequencies[class_idx];
            let actual_freq = actual_frequencies[class_idx];

            let raw_factor = if predicted_freq > 1e-6 {
                actual_freq / predicted_freq
            } else {
                1.0 // No correction if no predictions for this class
            };

            // Apply smoothing and bounds
            let smoothed_factor = if self.is_calibrated {
                // Smooth with previous factor
                self.class_bias_factors[class_idx] * (1.0 - self.config.smoothing_factor)
                    + raw_factor * self.config.smoothing_factor
            } else {
                raw_factor
            };

            // Apply bounds to prevent extreme corrections
            self.class_bias_factors[class_idx] = smoothed_factor
                .max(self.config.correction_bounds[0])
                .min(self.config.correction_bounds[1]);
        }

        // Calculate overall confidence scaling
        let predicted_confidence = self.calculate_average_confidence(validation_predictions)?;
        let actual_confidence = self.calculate_average_confidence(validation_targets)?;

        let raw_confidence_scaling = if predicted_confidence > 1e-6 {
            (actual_confidence / predicted_confidence) * self.config.confidence_adjustment
        } else {
            self.config.confidence_adjustment
        };

        // CRITICAL FIX: Bound confidence scaling to prevent gradient explosion
        // When used as temperature (1/confidence_scaling), extreme values cause instability
        // Bound to [0.5, 2.0] to keep temperature in reasonable range [0.5, 2.0]
        self.confidence_scaling = raw_confidence_scaling.clamp(0.5, 2.0);

        if (raw_confidence_scaling - self.confidence_scaling).abs() > 0.1 {
            log::warn!(
                "⚠️ Confidence scaling bounded: {:.3} → {:.3} to prevent gradient instability",
                raw_confidence_scaling,
                self.confidence_scaling
            );
        }

        // Store validation statistics
        let overall_accuracy =
            self.calculate_accuracy(validation_predictions, validation_targets)?;
        let confidence_distribution =
            self.calculate_confidence_distribution(validation_predictions)?;

        self.validation_stats = Some(ValidationStats {
            total_samples: num_samples,
            class_frequencies_predicted: predicted_frequencies,
            class_frequencies_actual: actual_frequencies,
            overall_accuracy,
            confidence_distribution,
        });

        self.is_calibrated = true;

        // Invalidate caches after calibration
        self.cached_combined_factors = None;
        self.cached_adaptive_strength = None;

        // Log calibration results based on print_info setting
        if self.config.print_info {
            // Detailed logging when print_info is true
            log::info!("✅ Bias correction calibrated successfully");
            log::info!("📊 Class bias factors: {:?}", self.class_bias_factors);
            log::info!("🎯 Confidence scaling: {:.4}", self.confidence_scaling);
            log::info!("📈 Validation accuracy: {:.4}", overall_accuracy);
            log::info!(
                "🔧 Configuration: enabled={}, smoothing={:.3}",
                self.config.enabled,
                self.config.smoothing_factor
            );
            log::info!(
                "📏 Correction bounds: [{:.2}, {:.2}]",
                self.config.correction_bounds[0],
                self.config.correction_bounds[1]
            );

            for (class_idx, (pred_freq, actual_freq)) in predicted_frequencies
                .iter()
                .zip(actual_frequencies.iter())
                .enumerate()
            {
                log::info!(
                    "   📊 Class {}: predicted={:.4}, actual={:.4}, factor={:.4}",
                    class_idx,
                    pred_freq,
                    actual_freq,
                    self.class_bias_factors[class_idx]
                );
            }
        }

        Ok(())
    }

    /// Apply bias correction to raw model predictions
    ///
    /// This is the main correction method that applies the calibrated bias factors
    /// to improve prediction accuracy and class balance
    pub fn apply_correction(&self, raw_predictions: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.config.enabled || !self.is_calibrated {
            return Ok(raw_predictions.clone());
        }

        // Validate input dimensions
        let num_classes = raw_predictions.ncols();
        if num_classes != 5 {
            log::error!(
                "❌ Expected 5 classes in predictions for bias correction, got {}",
                num_classes
            );
            return Err(crate::utils::error::VangaError::ModelError(format!(
                "Bias correction requires 5-class predictions, got {}",
                num_classes
            )));
        }

        let num_samples = raw_predictions.nrows();

        // OPTIMIZATION: Use cached combined factors if available
        let combined_factors = self.get_combined_factors();

        // Use in-place operations to avoid unnecessary allocations
        let mut corrected = raw_predictions.clone();

        // Apply combined factors in single pass
        for mut row in corrected.axis_iter_mut(Axis(0)) {
            // Apply factors in single pass
            for (idx, &factor) in combined_factors.iter().enumerate() {
                row[idx] *= factor;
            }
            // Renormalize in-place
            let sum: f64 = row.sum();
            if sum > 1e-10 {
                let inv_sum = 1.0 / sum; // Avoid repeated division
                row.mapv_inplace(|x| x * inv_sum);
            } else {
                row.fill(0.2); // Uniform distribution for 5 classes
            }
        }

        log::debug!(
            "🔧 Applied bias correction to {} samples with factors: {:?}",
            num_samples,
            self.class_bias_factors
        );

        Ok(corrected)
    }

    /// Get combined correction factors (cached)
    fn get_combined_factors(&self) -> [f64; 5] {
        // Return cached value if available, otherwise compute and cache
        if let Some(cached) = self.cached_combined_factors {
            cached
        } else {
            let mut combined = [0.0; 5];
            for (idx, &factor) in self.class_bias_factors.iter().enumerate() {
                combined[idx] = factor * self.confidence_scaling;
            }
            combined
        }
    }

    /// Renormalize probability distributions to sum to 1.0
    pub fn renormalize_probabilities(&self, predictions: &mut Array2<f64>) -> Result<()> {
        for mut row in predictions.axis_iter_mut(Axis(0)) {
            let sum: f64 = row.sum();
            if sum > 1e-10 {
                let inv_sum = 1.0 / sum; // Avoid repeated division
                row.mapv_inplace(|x| x * inv_sum);
            } else {
                // If all probabilities are near zero, set to uniform distribution
                row.fill(0.2); // 1/5 for each class
            }
        }
        Ok(())
    }

    /// Calculate average confidence (maximum probability per sample)
    fn calculate_average_confidence(&self, predictions: &Array2<f64>) -> Result<f64> {
        let mut sum = 0.0;
        let count = predictions.nrows();

        for row in predictions.axis_iter(Axis(0)) {
            // Use fold for efficient max finding
            let max_conf = row.iter().fold(0.0_f64, |max, &val| max.max(val));
            sum += max_conf;
        }

        Ok(sum / count as f64)
    }

    /// Calculate accuracy between predictions and targets
    fn calculate_accuracy(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        let mut correct = 0;
        let total = predictions.nrows();

        for (pred_row, target_row) in predictions
            .axis_iter(Axis(0))
            .zip(targets.axis_iter(Axis(0)))
        {
            // Use fold for efficient argmax
            let pred_class = pred_row
                .iter()
                .enumerate()
                .fold((0, 0.0), |(max_idx, max_val), (idx, &val)| {
                    if val > max_val {
                        (idx, val)
                    } else {
                        (max_idx, max_val)
                    }
                })
                .0;

            let target_class = target_row
                .iter()
                .enumerate()
                .fold((0, 0.0), |(max_idx, max_val), (idx, &val)| {
                    if val > max_val {
                        (idx, val)
                    } else {
                        (max_idx, max_val)
                    }
                })
                .0;

            if pred_class == target_class {
                correct += 1;
            }
        }

        Ok(correct as f64 / total as f64)
    }

    /// Calculate confidence distribution (quartiles)
    fn calculate_confidence_distribution(&self, predictions: &Array2<f64>) -> Result<[f64; 5]> {
        // Collect confidences more efficiently
        let mut confidences: Vec<f64> = Vec::with_capacity(predictions.nrows());

        for row in predictions.axis_iter(Axis(0)) {
            let max_conf = row.iter().fold(0.0_f64, |max, &val| max.max(val));
            confidences.push(max_conf);
        }

        // Use unstable sort for better performance
        confidences.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = confidences.len();
        if len == 0 {
            return Ok([0.0; 5]);
        }

        // Pre-calculate indices to avoid repeated division
        let q1_idx = len >> 2; // len / 4
        let median_idx = len >> 1; // len / 2
        let q3_idx = (3 * len) >> 2; // 3 * len / 4
        let max_idx = len - 1;

        Ok([
            confidences[0],          // Min
            confidences[q1_idx],     // Q1
            confidences[median_idx], // Median
            confidences[q3_idx],     // Q3
            confidences[max_idx],    // Max
        ])
    }

    /// Get correction effectiveness metrics
    pub fn get_correction_metrics(&self) -> Option<CorrectionMetrics> {
        if !self.is_calibrated {
            return None;
        }

        let stats = self.validation_stats.as_ref()?;

        // Calculate class balance improvement
        let pred_imbalance = self.calculate_imbalance(&stats.class_frequencies_predicted);
        let actual_imbalance = self.calculate_imbalance(&stats.class_frequencies_actual);
        let balance_improvement = pred_imbalance - actual_imbalance;

        // Calculate correction strength (average deviation from 1.0)
        let correction_strength = self
            .class_bias_factors
            .iter()
            .map(|&f| (f - 1.0).abs())
            .sum::<f64>()
            / 5.0;

        Some(CorrectionMetrics {
            is_calibrated: self.is_calibrated,
            total_samples: stats.total_samples,
            validation_accuracy: stats.overall_accuracy,
            class_bias_factors: self.class_bias_factors,
            confidence_scaling: self.confidence_scaling,
            balance_improvement,
            correction_strength,
            predicted_imbalance: pred_imbalance,
            actual_imbalance,
        })
    }

    /// Calculate class imbalance (standard deviation of frequencies)
    fn calculate_imbalance(&self, frequencies: &[f64; 5]) -> f64 {
        const INV_5: f64 = 0.2; // 1.0 / 5.0 pre-calculated
        let mean = frequencies.iter().sum::<f64>() * INV_5;
        let variance = frequencies
            .iter()
            .map(|&f| {
                let diff = f - mean;
                diff * diff // More efficient than powi(2)
            })
            .sum::<f64>()
            * INV_5;
        variance.sqrt()
    }

    /// Reset calibration (useful for retraining)
    pub fn reset_calibration(&mut self) {
        self.class_bias_factors = [1.0; 5];
        self.confidence_scaling = 1.0;
        self.validation_stats = None;
        self.is_calibrated = false;
        // Clear caches
        self.cached_combined_factors = None;
        self.cached_adaptive_strength = None;
        log::info!("🔄 Bias correction calibration reset");
    }

    /// Check if bias correction is enabled and calibrated
    pub fn is_active(&self) -> bool {
        self.config.enabled && self.is_calibrated
    }

    /// Calculate adaptive strength based on actual class imbalance (cached)
    /// Returns a value between 0.0 and 1.0 based on how imbalanced the classes are
    fn calculate_adaptive_strength_for_ordinal(&self) -> f64 {
        // Return cached value if available
        if let Some(cached) = self.cached_adaptive_strength {
            return cached;
        }

        if let Some(ref stats) = self.validation_stats {
            // Calculate the variance from perfect balance (0.2 for each of 5 classes)
            let perfect_freq = 0.2;

            // Measure how far we are from perfect balance
            let pred_imbalance: f64 = stats
                .class_frequencies_predicted
                .iter()
                .map(|&f| (f - perfect_freq).abs())
                .sum::<f64>()
                / 5.0;

            let actual_imbalance: f64 = stats
                .class_frequencies_actual
                .iter()
                .map(|&f| (f - perfect_freq).abs())
                .sum::<f64>()
                / 5.0;

            // The strength should be proportional to the imbalance
            // But capped to prevent over-correction
            // Max imbalance would be 0.16 (if one class has 100% and others 0%)
            // We scale this to a 0-1 range with a conservative multiplier
            let imbalance_factor = ((pred_imbalance + actual_imbalance) / 2.0) / 0.16;

            // For ordinal loss, we need to be more conservative
            // because aggressive corrections can break monotonic relationships
            let ordinal_dampening = 0.4; // Maximum 40% correction strength

            // Also consider the validation accuracy
            // If accuracy is already good, reduce correction strength
            let accuracy_factor = if stats.overall_accuracy > 0.6 {
                0.5 // Reduce strength if accuracy is already decent
            } else if stats.overall_accuracy < 0.3 {
                1.2 // Increase strength if accuracy is poor
            } else {
                1.0
            };

            (imbalance_factor * ordinal_dampening * accuracy_factor).min(0.5)
        } else {
            // No stats available, use minimal correction
            0.1
        }
    }

    /// Calculate ordinal-aware adjustments that preserve monotonic relationships
    /// FIXED: Proper bounded logit adjustments that don't cause gradient jumping
    fn calculate_ordinal_aware_adjustments(&self, strength: f64) -> Vec<f32> {
        // For ordinal classes (0=VeryDown, 1=Down, 2=Neutral, 3=Up, 4=VeryUp)
        // We want to preserve the ordering while correcting bias

        // FIXED: Instead of taking ln() which can be negative and unbounded,
        // we calculate relative adjustments from the mean bias factor
        let mean_factor = self.class_bias_factors.iter().sum::<f64>() / 5.0;

        // Calculate deviations from mean, bounded to prevent extreme adjustments
        let raw_adjustments: Vec<f64> = self
            .class_bias_factors
            .iter()
            .map(|&factor| {
                // Calculate relative deviation from mean
                let deviation = (factor - mean_factor) / mean_factor;
                // Bound the deviation to prevent extreme adjustments
                deviation.clamp(-0.5, 0.5) // Max ±50% adjustment
            })
            .collect();

        // Apply smoothing to preserve ordinal relationships
        // Use a weighted average with neighbors
        let mut smoothed_adjustments = [0.0; 5];
        for i in 0..5 {
            let mut sum = raw_adjustments[i] * 0.6; // Current class weight (increased)
            let mut weight = 0.6;

            // Add contribution from left neighbor
            if i > 0 {
                sum += raw_adjustments[i - 1] * 0.2;
                weight += 0.2;
            }

            // Add contribution from right neighbor
            if i < 4 {
                sum += raw_adjustments[i + 1] * 0.2;
                weight += 0.2;
            }

            smoothed_adjustments[i] = sum / weight;
        }

        // Apply strength with additional bounds to prevent gradient jumping
        smoothed_adjustments
            .iter()
            .map(|&adj| {
                let final_adj = adj * strength;
                // CRITICAL: Bound final adjustments to prevent gradient instability
                final_adj.clamp(-0.2, 0.2) as f32 // Max ±0.2 logit adjustment
            })
            .collect()
    }
    /// Apply bias correction to LOGITS (before softmax) for training compatibility
    /// This preserves gradient flow and works with ordinal loss
    /// FULLY ADAPTIVE: Strength automatically determined by class imbalance
    pub fn apply_correction_to_logits(
        &self,
        logits: &Tensor,
        current_epoch: usize,
    ) -> Result<Tensor> {
        if !self.is_calibrated || !self.config.enabled {
            return Ok(logits.clone());
        }

        // ADAPTIVE STRENGTH CALCULATION based on actual class imbalance
        let adaptive_strength = self.calculate_adaptive_strength_for_ordinal();

        // Gradual ramp-up to prevent training instability
        let ramp_factor = if current_epoch < self.config.ramp_up_epochs {
            current_epoch as f64 / self.config.ramp_up_epochs as f64
        } else {
            1.0
        };

        let strength = adaptive_strength * ramp_factor;

        // If strength is too small, skip correction
        if strength < 0.01 {
            return Ok(logits.clone());
        }

        // Get device from input tensor
        let device = logits.device();

        // ORDINAL-AWARE LOGIT ADJUSTMENTS
        let logit_adjustments = self.calculate_ordinal_aware_adjustments(strength);

        // OPTIMIZATION: Create adjustment tensor once
        let adjustment_tensor =
            Tensor::from_slice(&logit_adjustments, (1, 5), device).map_err(|e| {
                crate::utils::error::VangaError::ModelError(format!(
                    "Failed to create logit adjustment tensor: {}",
                    e
                ))
            })?;

        // Add adjustments to logits (broadcasting across batch)
        let adjusted_logits = logits.broadcast_add(&adjustment_tensor).map_err(|e| {
            crate::utils::error::VangaError::ModelError(format!(
                "Failed to apply logit adjustments: {}",
                e
            ))
        })?;

        // Log correction impact if verbose (reduced frequency)
        if self.config.print_info {
            let max_adjustment = logit_adjustments
                .iter()
                .map(|&x| x.abs())
                .fold(0.0f32, f32::max);

            log::debug!(
                "🔧 Logit bias correction: strength={:.3}, max_adj={:.3}, epoch={}",
                strength,
                max_adjustment,
                current_epoch
            );
        }

        Ok(adjusted_logits)
    }

    /// Apply bias correction to PROBABILITIES (after softmax) for inference
    /// This includes temperature scaling for probability calibration
    pub fn apply_correction_tensor(&self, probabilities: &Tensor) -> Result<Tensor> {
        if !self.is_calibrated || !self.config.enabled {
            return Ok(probabilities.clone());
        }

        // Get device from input tensor
        let device = probabilities.device();

        // OPTIMIZATION: Combine bias factors and confidence scaling into single tensor
        let combined_factors: Vec<f32> = self
            .class_bias_factors
            .iter()
            .map(|&x| (x * self.confidence_scaling) as f32)
            .collect();

        let correction_tensor =
            Tensor::from_slice(&combined_factors, (1, 5), device).map_err(|e| {
                crate::utils::error::VangaError::ModelError(format!(
                    "Failed to create correction tensor: {}",
                    e
                ))
            })?;

        // Apply combined correction in single operation
        let corrected_probs = probabilities
            .broadcast_mul(&correction_tensor)
            .map_err(|e| {
                crate::utils::error::VangaError::ModelError(format!(
                    "Failed to apply corrections: {}",
                    e
                ))
            })?;

        // Renormalize to ensure probabilities sum to 1.0
        let row_sums = corrected_probs.sum_keepdim(1).map_err(|e| {
            crate::utils::error::VangaError::ModelError(format!(
                "Failed to compute row sums: {}",
                e
            ))
        })?;

        // Add small epsilon to prevent division by zero
        let epsilon = 1e-10_f32;
        let eps_tensor = Tensor::new(&[epsilon], device)?;
        let safe_sums = row_sums.broadcast_add(&eps_tensor)?;

        let normalized_probs = corrected_probs.broadcast_div(&safe_sums).map_err(|e| {
            crate::utils::error::VangaError::ModelError(format!(
                "Failed to normalize probabilities: {}",
                e
            ))
        })?;

        Ok(normalized_probs)
    }

    /// Calculate KL divergence between original and corrected predictions for monitoring
    pub fn calculate_correction_impact(
        &self,
        original: &Tensor,
        corrected: &Tensor,
    ) -> Result<f64> {
        // Add small epsilon to prevent log(0)
        let epsilon = 1e-10_f32;
        let device = original.device();
        let eps_tensor = Tensor::new(&[epsilon], device).map_err(|e| {
            crate::utils::error::VangaError::ModelError(format!(
                "Failed to create epsilon tensor: {}",
                e
            ))
        })?;

        // P * log(P/Q) where P is original, Q is corrected
        let original_safe = original.broadcast_add(&eps_tensor).map_err(|e| {
            crate::utils::error::VangaError::ModelError(format!(
                "Failed to add epsilon to original: {}",
                e
            ))
        })?;
        let corrected_safe = corrected.broadcast_add(&eps_tensor).map_err(|e| {
            crate::utils::error::VangaError::ModelError(format!(
                "Failed to add epsilon to corrected: {}",
                e
            ))
        })?;

        let ratio = original_safe.div(&corrected_safe).map_err(|e| {
            crate::utils::error::VangaError::ModelError(format!("Failed to compute ratio: {}", e))
        })?;
        let log_ratio = ratio.log().map_err(|e| {
            crate::utils::error::VangaError::ModelError(format!(
                "Failed to compute log ratio: {}",
                e
            ))
        })?;
        let kl_terms = original.mul(&log_ratio).map_err(|e| {
            crate::utils::error::VangaError::ModelError(format!(
                "Failed to compute KL terms: {}",
                e
            ))
        })?;

        // Sum all terms and get mean
        let kl_sum = kl_terms.sum_all().map_err(|e| {
            crate::utils::error::VangaError::ModelError(format!(
                "Failed to sum KL divergence: {}",
                e
            ))
        })?;

        let kl_value = kl_sum.to_scalar::<f32>().map_err(|e| {
            crate::utils::error::VangaError::ModelError(format!(
                "Failed to extract KL divergence scalar: {}",
                e
            ))
        })? as f64;

        Ok(kl_value)
    }
}

/// Metrics for evaluating bias correction effectiveness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionMetrics {
    pub is_calibrated: bool,
    pub total_samples: usize,
    pub validation_accuracy: f64,
    pub class_bias_factors: [f64; 5],
    pub confidence_scaling: f64,
    pub balance_improvement: f64,
    pub correction_strength: f64,
    pub predicted_imbalance: f64,
    pub actual_imbalance: f64,
}

impl CorrectionMetrics {
    /// Get a summary string of correction effectiveness
    pub fn summary(&self) -> String {
        format!(
            "Bias Correction: calibrated={}, samples={}, accuracy={:.4}, balance_improvement={:.4}, strength={:.4}",
            self.is_calibrated,
            self.total_samples,
            self.validation_accuracy,
            self.balance_improvement,
            self.correction_strength
        )
    }
}
