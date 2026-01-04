//! Ensemble calibrator orchestrating all calibration methods
//!
//! Combines temperature scaling, label smoothing, and mixup for optimal calibration.

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
    pub fn calibrate_from_validation(
        &mut self,
        logits: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<()> {
        let num_samples = logits.nrows();
        if num_samples == 0 {
            log::warn!("⚠️ No validation samples for calibration");
            return Ok(());
        }

        if logits.ncols() != 5 || targets.ncols() != 5 {
            return Err(VangaError::DataError(format!(
                "Expected 5 classes, got logits: {}, targets: {}",
                logits.ncols(),
                targets.ncols()
            )));
        }

        log::info!("🎯 ═══════════════════════════════════════════════════════════");
        log::info!("🎯 ENSEMBLE CALIBRATION: Optimizing all components");
        log::info!("🎯 ═══════════════════════════════════════════════════════════");
        log::info!("📊 Validation samples: {}", num_samples);

        // Step 1: Optimize temperature scaling
        self.temperature_scaling
            .optimize_temperatures(logits, targets)?;

        // Get calibrated predictions for subsequent steps
        let calibrated_logits = self.temperature_scaling.apply_to_logits(logits)?;

        // Step 2: Calculate label smoothing factors
        self.label_smoothing
            .calibrate_from_validation(&calibrated_logits, targets)?;

        // Step 3: Calculate overall and per-class ECE
        let overall_ece = calculate_ece(&calibrated_logits, targets)?;
        let per_class_ece = calculate_per_class_ece(&calibrated_logits, targets)?;

        self.ece_history.push(overall_ece);
        self.per_class_ece_history.push(per_class_ece);

        // Step 4: Tune mixup from ECE
        self.mixup.calibrate_from_ece(overall_ece, &per_class_ece)?;

        // Step 5: Generate reliability diagram
        self.reliability_diagram = Some(generate_reliability_diagram(&calibrated_logits, targets)?);

        self.is_calibrated = true;

        // Log comprehensive summary
        self.log_calibration_summary(overall_ece, &per_class_ece);

        log::info!("🎯 ═══════════════════════════════════════════════════════════");
        log::info!("✅ Ensemble calibration completed successfully");
        log::info!("🎯 ═══════════════════════════════════════════════════════════");

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
            temperatures: self.temperature_scaling.get_temperatures(),
            label_smoothing_epsilons: self.label_smoothing.get_epsilons(),
            mixup_alpha: self.mixup.get_alpha(),
            mixup_enabled_classes: self.mixup.enabled_for_classes,
        }
    }

    /// Log comprehensive calibration summary
    fn log_calibration_summary(&self, overall_ece: f64, per_class_ece: &[f64; 5]) {
        log::info!("📊 ═══════════════════════════════════════════════════════════");
        log::info!("📊 CALIBRATION SUMMARY");
        log::info!("📊 ═══════════════════════════════════════════════════════════");

        log::info!("🌡️ Temperature Scaling:");
        log::info!(
            "   Temperatures: {:?}",
            self.temperature_scaling.get_temperatures()
        );
        for (class_idx, &temp) in self
            .temperature_scaling
            .get_temperatures()
            .iter()
            .enumerate()
        {
            let status = if temp < 0.9 {
                "sharpening (more confident)"
            } else if temp > 1.1 {
                "softening (less confident)"
            } else {
                "well-calibrated"
            };
            log::info!("   Class {}: T={:.3} ({})", class_idx, temp, status);
        }

        log::info!("🎯 Label Smoothing:");
        log::info!("   Epsilons: {:?}", self.label_smoothing.get_epsilons());
        if self.label_smoothing.has_significant_smoothing() {
            for (class_idx, &eps) in self.label_smoothing.get_epsilons().iter().enumerate() {
                if eps > 0.05 {
                    log::info!("   Class {}: ε={:.3} (overconfident)", class_idx, eps);
                }
            }
        } else {
            log::info!("   No significant overconfidence detected");
        }

        log::info!("🔀 Mixup Augmentation:");
        log::info!("   Alpha: {:.3}", self.mixup.get_alpha());
        log::info!(
            "   Enabled for classes: {:?}",
            self.mixup
                .enabled_for_classes
                .iter()
                .enumerate()
                .filter(|(_, &enabled)| enabled)
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>()
        );

        log::info!("📈 Calibration Quality:");
        log::info!("   Overall ECE: {:.6}", overall_ece);
        log::info!("   Per-class ECE: {:?}", per_class_ece);

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
            "   Quality: {} (ECE < 0.05 = excellent)",
            calibration_quality
        );

        log::info!("📊 ═══════════════════════════════════════════════════════════");
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
