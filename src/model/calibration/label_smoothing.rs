//! Adaptive label smoothing based on class overconfidence
//!
//! Calculates smoothing factor per class based on confidence-accuracy gap.
//! Prevents overconfidence by softening hard labels.

use crate::utils::error::{Result, VangaError};
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

/// Adaptive label smoothing with per-class epsilon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLabelSmoothing {
    /// Per-class smoothing factors (0.0 = no smoothing, higher = more smoothing)
    pub epsilons: [f64; 5],
    /// Per-class average confidence
    pub avg_confidences: [f64; 5],
    /// Per-class average accuracy
    pub avg_accuracies: [f64; 5],
    /// Whether smoothing factors have been calculated
    pub is_calibrated: bool,
}

impl Default for AdaptiveLabelSmoothing {
    fn default() -> Self {
        Self {
            epsilons: [0.0; 5],
            avg_confidences: [0.0; 5],
            avg_accuracies: [0.0; 5],
            is_calibrated: false,
        }
    }
}

impl AdaptiveLabelSmoothing {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate adaptive smoothing factors from validation data
    ///
    /// Formula: epsilon = max(0.0, (avg_confidence - avg_accuracy) * scaling_factor)
    /// Higher epsilon when class is overconfident (confidence > accuracy)
    pub fn calibrate_from_validation(
        &mut self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<()> {
        let num_samples = predictions.nrows();
        if num_samples == 0 {
            return Ok(());
        }

        if predictions.ncols() != 5 || targets.ncols() != 5 {
            return Err(VangaError::DataError(format!(
                "Expected 5 classes, got predictions: {}, targets: {}",
                predictions.ncols(),
                targets.ncols()
            )));
        }

        log::info!("🎯 Calculating adaptive label smoothing factors...");

        // Calculate per-class statistics
        for class_idx in 0..5 {
            let mut confidence_sum = 0.0;
            let mut accuracy_sum = 0.0;
            let mut class_count = 0;

            for (pred_row, target_row) in predictions
                .axis_iter(Axis(0))
                .zip(targets.axis_iter(Axis(0)))
            {
                let confidence = pred_row[class_idx];
                let is_true_class = target_row[class_idx] > 0.5;

                confidence_sum += confidence;
                if is_true_class {
                    // For true class samples, check if prediction was correct
                    let pred_class = pred_row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap();

                    if pred_class == class_idx {
                        accuracy_sum += 1.0;
                    }
                    class_count += 1;
                }
            }

            // Calculate averages
            self.avg_confidences[class_idx] = if num_samples > 0 {
                confidence_sum / num_samples as f64
            } else {
                0.0
            };

            self.avg_accuracies[class_idx] = if class_count > 0 {
                accuracy_sum / class_count as f64
            } else {
                0.0
            };

            // Calculate epsilon: higher when overconfident
            let confidence_gap = self.avg_confidences[class_idx] - self.avg_accuracies[class_idx];

            // Scaling factor: adaptive based on gap magnitude
            // Larger gaps need more smoothing
            let scaling_factor = if confidence_gap > 0.0 {
                // Overconfident: apply smoothing proportional to gap
                1.0 + confidence_gap
            } else {
                // Underconfident or well-calibrated: minimal smoothing
                0.5
            };

            self.epsilons[class_idx] = (confidence_gap * scaling_factor).max(0.0).min(0.3);
        }

        self.is_calibrated = true;

        log::info!("   Per-class smoothing factors: {:?}", self.epsilons);
        log::info!("   Avg confidences: {:?}", self.avg_confidences);
        log::info!("   Avg accuracies: {:?}", self.avg_accuracies);

        // Log overconfident classes
        for class_idx in 0..5 {
            if self.epsilons[class_idx] > 0.05 {
                log::info!(
                    "   Class {}: overconfident (conf={:.3}, acc={:.3}) → smoothing={:.3}",
                    class_idx,
                    self.avg_confidences[class_idx],
                    self.avg_accuracies[class_idx],
                    self.epsilons[class_idx]
                );
            }
        }

        Ok(())
    }

    /// Apply label smoothing to targets
    ///
    /// smoothed = (1 - epsilon) * one_hot + epsilon / num_classes
    pub fn apply_smoothing(&self, targets: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.is_calibrated {
            return Ok(targets.clone());
        }

        let mut smoothed = targets.clone();

        for (i, mut target_row) in smoothed.axis_iter_mut(Axis(0)).enumerate() {
            // Find true class
            let true_class = targets
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap();

            let epsilon = self.epsilons[true_class];

            if epsilon > 0.0 {
                // Apply smoothing: (1 - epsilon) * one_hot + epsilon / 5
                for j in 0..5 {
                    if j == true_class {
                        target_row[j] = (1.0 - epsilon) + epsilon / 5.0;
                    } else {
                        target_row[j] = epsilon / 5.0;
                    }
                }
            }
        }

        Ok(smoothed)
    }

    /// Get smoothing factors
    pub fn get_epsilons(&self) -> [f64; 5] {
        self.epsilons
    }

    /// Check if any class needs significant smoothing
    pub fn has_significant_smoothing(&self) -> bool {
        self.epsilons.iter().any(|&eps| eps > 0.05)
    }

    /// Reset calibration
    pub fn reset(&mut self) {
        self.epsilons = [0.0; 5];
        self.avg_confidences = [0.0; 5];
        self.avg_accuracies = [0.0; 5];
        self.is_calibrated = false;
    }
}
