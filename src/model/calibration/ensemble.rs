//! Ensemble calibrator orchestrating all calibration methods
//!
//! **CRITICAL: Temperature Scaling is POST-HOC ONLY**
//!
//! This module implements ensemble calibration combining:
//! - **Temperature Scaling**: POST-HOC calibration applied AFTER training
//! - **Label Smoothing**: Training-time regularization
//! - **Mixup**: Training-time data augmentation
//!
//! ## Research Foundation
//!
//! Temperature scaling is a **post-processing method** that should ONLY be applied
//! after training completes, never during training:
//!
//! - **Guo et al. 2017**: "On Calibration of Modern Neural Networks"
//!   > "Temperature scaling is a post-processing method that fixes [overconfidence]"
//!
//! - **ICLR 2025**: "GETS: Ensemble Temperature Scaling for Calibration"
//!   > "Existing post-hoc methods, such as temperature scaling..."
//!
//! - **AWS Prescriptive Guidance 2024**:
//!   > "After a model training is completed, extract the temperature value T
//!   > by using the validation dataset"
//!
//! ## Correct Usage
//!
//! ```rust
//! // TRAINING: Use label smoothing and mixup (training-time methods)
//! let smoothed_targets = ensemble_cal.apply_label_smoothing(&targets)?;
//! let (mixed_seq, mixed_tgt) = ensemble_cal.apply_mixup(&sequences, &targets, &mut rng)?;
//!
//! // AFTER TRAINING: Calibrate temperature on validation set (POST-HOC)
//! ensemble_cal.calibrate_from_validation(&val_predictions, &val_targets)?;
//!
//! // INFERENCE: Apply temperature scaling to logits
//! let calibrated_logits = ensemble_cal.apply_to_logits(&logits)?;
//! ```
//!
//! ## Why Temperature Scaling is Post-Hoc
//!
//! 1. **Validation Data Leakage**: Optimizing temperature on validation during
//!    training causes the model to indirectly learn validation patterns
//!
//! 2. **Gradient Instability**: Changing temperature during training causes
//!    gradients to be scaled differently, preventing convergence
//!
//! 3. **Moving Target**: Model parameters adapt to current temperature, then
//!    temperature changes, creating a never-ending chase
//!
//! 4. **Conflicting Objectives**: Training loss (cross-entropy) vs calibration
//!    loss (NLL on validation) pull in different directions

use super::ece::{
    calculate_ece, calculate_per_class_ece, generate_reliability_diagram, ReliabilityDiagram,
};
use super::label_smoothing::AdaptiveLabelSmoothing;
use super::mixup::AdaptiveMixup;
use super::temperature::AdaptiveTemperatureScaling;
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

/// Ensemble calibrator combining all calibration methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleCalibrator {
    /// Temperature scaling component
    pub temperature_scaling: AdaptiveTemperatureScaling,
    /// Label smoothing component
    pub label_smoothing: AdaptiveLabelSmoothing,
    /// Mixup augmentation component
    pub mixup: AdaptiveMixup,
    /// Overall ECE history
    pub ece_history: Vec<f64>,
    /// Per-class ECE history
    pub per_class_ece_history: Vec<[f64; 5]>,
    /// Whether ensemble has been calibrated
    pub is_calibrated: bool,
    /// Reliability diagram for visualization
    pub reliability_diagram: Option<ReliabilityDiagram>,
}

impl Default for EnsembleCalibrator {
    fn default() -> Self {
        Self {
            temperature_scaling: AdaptiveTemperatureScaling::new(),
            label_smoothing: AdaptiveLabelSmoothing::new(),
            mixup: AdaptiveMixup::new(),
            ece_history: Vec::new(),
            per_class_ece_history: Vec::new(),
            is_calibrated: false,
            reliability_diagram: None,
        }
    }
}

impl EnsembleCalibrator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calibrate all components from validation data
    ///
    /// 1. Calculate initial ECE
    /// 2. Optimize temperatures via gradient descent
    /// 3. Calculate adaptive label smoothing factors
    /// 4. Tune mixup alpha from ECE
    ///
    /// **CRITICAL**: Accepts PROBABILITIES (after softmax), not logits!
    /// Converts probabilities to logits internally for temperature optimization.
    pub fn calibrate_from_validation(
        &mut self,
        probabilities: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<()> {
        let num_samples = probabilities.nrows();
        if num_samples == 0 {
            log::warn!("⚠️ No validation samples for calibration");
            return Ok(());
        }

        if probabilities.ncols() != 5 || targets.ncols() != 5 {
            return Err(VangaError::DataError(format!(
                "Expected 5 classes, got probabilities: {}, targets: {}",
                probabilities.ncols(),
                targets.ncols()
            )));
        }

        log::info!(
            "🎯 Ensemble calibration starting ({} samples)...",
            num_samples
        );

        // Convert probabilities back to logits for temperature optimization
        // logit = log(p / (1 - p)) but for multi-class we use log(p) - log(sum(p))
        let mut logits = Array2::zeros((num_samples, 5));
        for i in 0..num_samples {
            let mut row_logits = [0.0; 5];
            for j in 0..5 {
                let p = probabilities[[i, j]].clamp(1e-10, 1.0 - 1e-10); // Clamp to avoid log(0)
                row_logits[j] = p.ln();
            }
            // Normalize logits (subtract mean for numerical stability)
            let mean_logit = row_logits.iter().sum::<f64>() / 5.0;
            for j in 0..5 {
                logits[[i, j]] = row_logits[j] - mean_logit;
            }
        }

        // Step 1: Optimize temperature scaling
        self.temperature_scaling
            .optimize_temperatures(&logits, targets)?;

        // Get calibrated predictions for subsequent steps
        let calibrated_logits = self.temperature_scaling.apply_to_logits(&logits)?;

        // Step 2: Calculate label smoothing factors
        self.label_smoothing
            .calibrate_from_validation(&calibrated_logits, targets)?;

        // Step 3: Calculate overall and per-class ECE
        let overall_ece = calculate_ece(&calibrated_logits, targets)?;
        let per_class_ece = calculate_per_class_ece(&calibrated_logits, targets)?;

        self.ece_history.push(overall_ece);
        self.per_class_ece_history.push(per_class_ece);

        // Step 4: Tune mixup from ECE
        self.mixup.calibrate_from_ece(overall_ece)?;

        // Step 5: Generate reliability diagram
        self.reliability_diagram = Some(generate_reliability_diagram(&calibrated_logits, targets)?);

        self.is_calibrated = true;

        // Log comprehensive summary
        self.log_calibration_summary(overall_ece);

        Ok(())
    }

    /// Apply temperature scaling to logits (for inference)
    pub fn apply_to_logits(&self, logits: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.is_calibrated {
            return Ok(logits.clone());
        }

        self.temperature_scaling.apply_to_logits(logits)
    }

    /// Apply label smoothing to targets (for training)
    pub fn apply_label_smoothing(&self, targets: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.is_calibrated {
            return Ok(targets.clone());
        }

        self.label_smoothing.apply_smoothing(targets)
    }

    /// Apply mixup to training batch (for training)
    pub fn apply_mixup(
        &self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        rng_state: &mut u64,
    ) -> Result<(Array3<f64>, Array2<f64>)> {
        if !self.is_calibrated || !self.mixup.is_enabled() {
            return Ok((sequences.clone(), targets.clone()));
        }

        self.mixup.mixup_batch(sequences, targets, rng_state)
    }

    /// Get calibration metrics for logging
    pub fn get_calibration_metrics(&self) -> CalibrationMetrics {
        CalibrationMetrics {
            is_calibrated: self.is_calibrated,
            overall_ece: self.ece_history.last().copied().unwrap_or(0.0),
            per_class_ece: self
                .per_class_ece_history
                .last()
                .copied()
                .unwrap_or([0.0; 5]),
            temperatures: self.temperature_scaling.temperatures(),
            label_smoothing_epsilons: self.label_smoothing.get_epsilons(),
            mixup_alpha: self.mixup.get_alpha(),
            mixup_enabled_classes: self.mixup.enabled_for_classes,
        }
    }

    /// Log comprehensive calibration summary
    fn log_calibration_summary(&self, overall_ece: f64) {
        let calibration_quality = if overall_ece < 0.05 {
            "Excellent"
        } else if overall_ece < 0.10 {
            "Good"
        } else if overall_ece < 0.15 {
            "Fair"
        } else {
            "Needs improvement"
        };

        log::info!(
            "✅ Calibration complete: ECE={:.4} ({})",
            overall_ece,
            calibration_quality
        );

        // Temperature scaling summary
        log::info!(
            "   🌡️  Temperature: {:.4}",
            self.temperature_scaling.temperature
        );

        // Label smoothing summary (only if significant)
        if self.label_smoothing.has_significant_smoothing() {
            let overconfident_classes: Vec<usize> = self
                .label_smoothing
                .get_epsilons()
                .iter()
                .enumerate()
                .filter(|(_, &eps)| eps > 0.05)
                .map(|(idx, _)| idx)
                .collect();
            log::info!(
                "   🎯 Label smoothing: classes {:?} overconfident",
                overconfident_classes
            );
        }

        // Mixup summary (only if enabled)
        if self.mixup.is_enabled() {
            let enabled_classes: Vec<usize> = self
                .mixup
                .enabled_for_classes
                .iter()
                .enumerate()
                .filter(|(_, &enabled)| enabled)
                .map(|(idx, _)| idx)
                .collect();
            log::info!(
                "   🔀 Mixup α={:.2}: enabled for classes {:?}",
                self.mixup.get_alpha(),
                enabled_classes
            );
        }
    }

    /// Reset all calibration state
    pub fn reset(&mut self) {
        self.temperature_scaling.reset();
        self.label_smoothing.reset();
        self.mixup.reset();
        self.ece_history.clear();
        self.per_class_ece_history.clear();
        self.is_calibrated = false;
        self.reliability_diagram = None;
    }

    /// Check if calibration is active
    pub fn is_active(&self) -> bool {
        self.is_calibrated
    }

    /// Apply temperature scaling to tensor logits (preserves gradients for training)
    ///
    /// This method applies temperature scaling directly to tensors without
    /// converting to ndarray, preserving gradient flow for backpropagation.
    pub fn apply_to_tensor(&self, logits: &Tensor, device: &Device) -> Result<Tensor> {
        if !self.is_calibrated {
            return Ok(logits.clone());
        }

        self.temperature_scaling.apply_to_tensor(logits, device)
    }
}

/// Calibration metrics for logging and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    pub is_calibrated: bool,
    pub overall_ece: f64,
    pub per_class_ece: [f64; 5],
    pub temperatures: [f64; 5],
    pub label_smoothing_epsilons: [f64; 5],
    pub mixup_alpha: f64,
    pub mixup_enabled_classes: [bool; 5],
}

impl CalibrationMetrics {
    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Calibration: ECE={:.4}, Temps={:?}, Smoothing={:?}, Mixup α={:.3}",
            self.overall_ece,
            self.temperatures
                .iter()
                .map(|&t| format!("{:.2}", t))
                .collect::<Vec<_>>(),
            self.label_smoothing_epsilons
                .iter()
                .map(|&e| format!("{:.2}", e))
                .collect::<Vec<_>>(),
            self.mixup_alpha
        )
    }
}
