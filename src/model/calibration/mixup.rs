//! Adaptive mixup augmentation with ECE-based tuning
//!
//! Auto-tunes mixup alpha based on calibration error.
//! Mixup: mix two samples with lambda ~ Beta(alpha, alpha)

use crate::utils::error::Result;
use ndarray::{Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

/// Adaptive mixup augmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveMixup {
    /// Mixup alpha parameter (learned from ECE)
    pub alpha: f64,
    /// Per-class enable/disable based on ECE
    pub enabled_for_classes: [bool; 5],
    /// Overall ECE that triggered current alpha
    pub current_ece: f64,
    /// Whether mixup has been calibrated
    pub is_calibrated: bool,
}

impl Default for AdaptiveMixup {
    fn default() -> Self {
        Self {
            alpha: 0.2, // Conservative starting point
            enabled_for_classes: [true; 5], // Enable for all initially
            current_ece: 0.0,
            is_calibrated: false,
        }
    }
}

impl AdaptiveMixup {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calibrate mixup parameters from ECE
    ///
    /// Higher ECE → more aggressive mixup (higher alpha)
    /// Per-class enable based on per-class ECE
    pub fn calibrate_from_ece(&mut self, overall_ece: f64, per_class_ece: &[f64; 5]) -> Result<()> {
        log::info!("🔀 Calibrating adaptive mixup from ECE...");

        self.current_ece = overall_ece;

        // Auto-tune alpha based on overall ECE
        // Higher ECE = more calibration needed = more aggressive mixup
        let base_alpha = 0.2;
        let ece_factor = (overall_ece * 2.0).min(1.0); // Cap at 1.0
        self.alpha = base_alpha * (1.0 + ece_factor);

        // Clamp alpha to reasonable range
        self.alpha = self.alpha.clamp(0.1, 0.5);

        // Calculate median ECE for threshold
        let mut sorted_ece = *per_class_ece;
        sorted_ece.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_ece = sorted_ece[2]; // Middle value of 5

        // Enable mixup only for classes with ECE above median
        for (class_idx, &class_ece) in per_class_ece.iter().enumerate() {
            self.enabled_for_classes[class_idx] = class_ece > median_ece;
        }

        self.is_calibrated = true;

        log::info!("   Mixup alpha: {:.3} (ECE: {:.4})", self.alpha, overall_ece);
        log::info!(
            "   Enabled for classes: {:?}",
            self.enabled_for_classes
                .iter()
                .enumerate()
                .filter(|(_, &enabled)| enabled)
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>()
        );
        log::info!("   Per-class ECE: {:?}", per_class_ece);
        log::info!("   Median ECE threshold: {:.4}", median_ece);

        Ok(())
    }

    /// Apply mixup to a batch of sequences and targets
    ///
    /// Returns mixed sequences and soft targets
    pub fn mixup_batch(
        &self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        rng_state: &mut u64,
    ) -> Result<(Array3<f64>, Array2<f64>)> {
        if !self.is_calibrated {
            return Ok((sequences.clone(), targets.clone()));
        }

        let batch_size = sequences.shape()[0];
        if batch_size < 2 {
            return Ok((sequences.clone(), targets.clone()));
        }

        // Check if any samples should be mixed based on their class
        let mut should_mix = vec![false; batch_size];
        for (i, target_row) in targets.axis_iter(Axis(0)).enumerate() {
            let true_class = target_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap();

            should_mix[i] = self.enabled_for_classes[true_class];
        }

        // If no samples should be mixed, return original
        if !should_mix.iter().any(|&x| x) {
            return Ok((sequences.clone(), targets.clone()));
        }

        let mut mixed_sequences = sequences.clone();
        let mut mixed_targets = targets.clone();

        // Generate shuffled indices for pairing
        let mut shuffled_indices: Vec<usize> = (0..batch_size).collect();
        self.shuffle_indices(&mut shuffled_indices, rng_state);

        // Apply mixup to samples that should be mixed
        for i in 0..batch_size {
            if !should_mix[i] {
                continue;
            }

            let j = shuffled_indices[i];
            if i == j {
                continue; // Skip if paired with itself
            }

            // Sample lambda from Beta(alpha, alpha)
            let lambda = self.sample_beta(self.alpha, self.alpha, rng_state);

            // Mix sequences: lambda * seq_i + (1 - lambda) * seq_j
            for seq_idx in 0..sequences.shape()[1] {
                for feat_idx in 0..sequences.shape()[2] {
                    mixed_sequences[[i, seq_idx, feat_idx]] = lambda * sequences[[i, seq_idx, feat_idx]]
                        + (1.0 - lambda) * sequences[[j, seq_idx, feat_idx]];
                }
            }

            // Mix targets: lambda * target_i + (1 - lambda) * target_j
            for class_idx in 0..5 {
                mixed_targets[[i, class_idx]] =
                    lambda * targets[[i, class_idx]] + (1.0 - lambda) * targets[[j, class_idx]];
            }
        }

        Ok((mixed_sequences, mixed_targets))
    }

    /// Sample from Beta(alpha, alpha) distribution using rejection sampling
    fn sample_beta(&self, alpha: f64, beta: f64, rng_state: &mut u64) -> f64 {
        // For Beta(alpha, alpha), use simple approximation
        // When alpha = beta, distribution is symmetric around 0.5

        if (alpha - beta).abs() < 1e-6 {
            // Symmetric case: use uniform sampling with transformation
            let u1 = self.random_uniform(rng_state);
            let u2 = self.random_uniform(rng_state);

            // Box-Muller-like transformation for Beta
            let x = u1.powf(1.0 / alpha);
            let y = u2.powf(1.0 / alpha);
            x / (x + y)
        } else {
            // General case: use rejection sampling
            let mut attempts = 0;
            loop {
                let u = self.random_uniform(rng_state);
                let v = self.random_uniform(rng_state);

                let x = u.powf(1.0 / alpha);
                let y = v.powf(1.0 / beta);
                let sum = x + y;

                if sum <= 1.0 {
                    return x / sum;
                }

                attempts += 1;
                if attempts > 100 {
                    // Fallback to 0.5 if rejection sampling fails
                    return 0.5;
                }
            }
        }
    }

    /// Generate uniform random number in [0, 1]
    fn random_uniform(&self, rng_state: &mut u64) -> f64 {
        // Linear congruential generator
        *rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*rng_state as f64) / (u64::MAX as f64)
    }

    /// Shuffle indices using Fisher-Yates algorithm
    fn shuffle_indices(&self, indices: &mut [usize], rng_state: &mut u64) {
        for i in (1..indices.len()).rev() {
            *rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let j = (*rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }
    }

    /// Get current alpha
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }

    /// Check if mixup is enabled for any class
    pub fn is_enabled(&self) -> bool {
        self.is_calibrated && self.enabled_for_classes.iter().any(|&x| x)
    }

    /// Reset calibration
    pub fn reset(&mut self) {
        self.alpha = 0.2;
        self.enabled_for_classes = [true; 5];
        self.current_ece = 0.0;
        self.is_calibrated = false;
    }
}
