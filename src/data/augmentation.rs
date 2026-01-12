//! Time series augmentation techniques for overlapping sequences
//!
//! Based on latest research (2024-2025):
//! - Magnitude Warping: MDPI 2024 (11.5-22.5% improvement)
//! - Jittering with Gaussian noise: IJASEIT 2024 (best for imbalance)
//! - Time Warping: Standard technique (uchidalab)
//! - Scaling: Symbol-agnostic augmentation
//!
//! Only augments sequences that actually overlap to add diversity.

use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Augmentation configuration for time series
///
/// Based on research (2024-2025):
/// - Magnitude Warping: MDPI 2024 (11.5-22.5% improvement)
/// - Jittering with Gaussian noise: IJASEIT 2024 (best for imbalance)
/// - Time Warping: Standard technique (uchidalab)
/// - Scaling: Symbol-agnostic augmentation
///
/// CONSERVATIVE parameters for financial data (crypto is volatile):
/// - Smaller sigma values to prevent target class changes
/// - Lower probability for aggressive techniques
pub struct AugmentationConfig {
    /// Magnitude warping sigma (default: 0.1 for financial data)
    /// ±10% variation preserves target classification better
    pub magnitude_sigma: f64,
    /// Jittering sigma for Gaussian noise (default: 0.02 for crypto)
    /// Small noise prevents memorization without affecting targets
    pub jitter_sigma: f64,
    /// Time warping sigma (default: 0.1 for financial data)
    /// Conservative to preserve temporal patterns
    pub time_warp_sigma: f64,
    /// Scaling sigma (default: 0.05 for financial data)
    /// Small scaling preserves price relationships
    pub scaling_sigma: f64,
    /// Probability of applying jitter (default: 0.3)
    pub jitter_probability: f64,
    /// Probability of applying scaling (default: 0.3)
    pub scaling_probability: f64,
    /// Probability of applying time warping (default: 0.2)
    /// Lower because it's more aggressive
    pub time_warp_probability: f64,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            magnitude_sigma: 0.1,    // ±10% instead of ±20%
            jitter_sigma: 0.02,      // Small noise
            time_warp_sigma: 0.1,    // Conservative
            scaling_sigma: 0.05,     // Small scaling
            jitter_probability: 0.3, // Lower probability
            scaling_probability: 0.3,
            time_warp_probability: 0.2,
        }
    }
}

impl AugmentationConfig {
    /// Create default augmentation config
    /// The overlap parameter is kept for API compatibility but augmentation
    /// is always applied when this config is used (caller controls when to augment)
    pub fn from_overlap(_overlap: f64) -> Self {
        Self::default()
    }
}

/// Augment a single sequence using research-backed techniques
///
/// CRITICAL: Skips price and volume columns that determine targets!
/// Only augments technical indicators (RSI, MACD, SMA, EMA, etc.)
///
/// Research-based strategy (2024-2025 state-of-art):
/// - Always apply magnitude warping (highest impact - MDPI 2024: 11.5-22.5% improvement)
/// - 50% probability: Gaussian jittering (prevents memorization)
/// - 50% probability: scaling (symbol-agnostic)
/// - 30% probability: time warping (temporal diversity, more aggressive)
pub fn augment_sequence(
    sequence: &Array2<f64>,
    config: &AugmentationConfig,
    rng: &mut impl Rng,
    price_volume_cols: &[usize],
) -> Array2<f64> {
    let mut augmented = sequence.clone();

    // 1. ALWAYS: Magnitude warping (highest impact - MDPI 2024)
    augmented = magnitude_warp(&augmented, config.magnitude_sigma, rng, price_volume_cols);

    // 2. Conservative probability: Gaussian jittering (prevents memorization)
    if rng.random_bool(config.jitter_probability) {
        augmented = jitter(&augmented, config.jitter_sigma, rng, price_volume_cols);
    }

    // 3. Conservative probability: Scaling (symbol-agnostic)
    if rng.random_bool(config.scaling_probability) {
        augmented = scaling(&augmented, config.scaling_sigma, rng, price_volume_cols);
    }

    // 4. Lower probability: Time warping (temporal diversity)
    // Only applies to indicators, not price/volume
    if rng.random_bool(config.time_warp_probability) {
        augmented = time_warp(&augmented, config.time_warp_sigma, rng, price_volume_cols);
    }

    augmented
}
/// Effect: Changes amplitude while preserving temporal patterns
/// CRITICAL: Skips price/volume columns that determine targets
pub fn magnitude_warp(
    sequence: &Array2<f64>,
    sigma: f64,
    rng: &mut impl Rng,
    price_volume_cols: &[usize],
) -> Array2<f64> {
    let timesteps = sequence.shape()[0];
    let features = sequence.shape()[1];

    // Generate smooth warping curve using cubic interpolation
    let knots = 4;
    let mut warp_points = Vec::with_capacity(knots + 2);
    for _ in 0..knots + 2 {
        warp_points.push(rng.random_range((1.0 - sigma)..(1.0 + sigma)));
    }

    // Create smooth curve through knot points
    let warp_curve = cubic_interpolate(&warp_points, timesteps);

    // Apply warping to each feature independently, skipping price/volume columns
    let mut warped = sequence.clone();
    let price_volume_set: std::collections::HashSet<usize> =
        price_volume_cols.iter().cloned().collect();
    for t in 0..timesteps {
        for f in 0..features {
            if !price_volume_set.contains(&f) {
                warped[[t, f]] *= warp_curve[t];
            }
        }
    }

    warped
}

/// Jittering: Add small Gaussian noise
///
/// Research: "Best performance for data imbalance" (IJASEIT 2024)
/// Uses proper Gaussian distribution as recommended by all major surveys.
/// Effect: Prevents memorization of exact values
/// CRITICAL: Skips price/volume columns that determine targets
pub fn jitter(
    sequence: &Array2<f64>,
    sigma: f64,
    rng: &mut impl Rng,
    price_volume_cols: &[usize],
) -> Array2<f64> {
    let shape = sequence.shape();
    let mut jittered = sequence.clone();

    // Use Gaussian noise as per research recommendations
    let normal = Normal::new(0.0, sigma).unwrap_or_else(|_| Normal::new(0.0, 0.03).unwrap());
    let price_volume_set: std::collections::HashSet<usize> =
        price_volume_cols.iter().cloned().collect();

    for t in 0..shape[0] {
        for f in 0..shape[1] {
            if !price_volume_set.contains(&f) {
                let noise = normal.sample(rng);
                jittered[[t, f]] += noise;
            }
        }
    }

    jittered
}

/// Scaling: Multiply by random constant
///
/// Effect: Simulates different price ranges (symbol-agnostic)
/// CRITICAL: Skips price/volume columns that determine targets
pub fn scaling(
    sequence: &Array2<f64>,
    sigma: f64,
    rng: &mut impl Rng,
    price_volume_cols: &[usize],
) -> Array2<f64> {
    let factor = rng.random_range((1.0 - sigma)..(1.0 + sigma));
    let mut scaled = sequence.clone();
    let price_volume_set: std::collections::HashSet<usize> =
        price_volume_cols.iter().cloned().collect();

    // Only scale non-price/volume columns
    let shape = sequence.shape();
    for t in 0..shape[0] {
        for f in 0..shape[1] {
            if !price_volume_set.contains(&f) {
                scaled[[t, f]] *= factor;
            }
        }
    }

    scaled
}

/// Time Warping: Non-linear time axis distortion
///
/// Research: Standard in time series augmentation (uchidalab)
/// Effect: Simulates different market speeds
/// CRITICAL: Now skips price/volume columns to preserve target consistency.
/// Only applies to technical indicators (RSI, MACD, etc.) where temporal
/// distortion doesn't affect target classification.
pub fn time_warp(
    sequence: &Array2<f64>,
    sigma: f64,
    rng: &mut impl Rng,
    price_volume_cols: &[usize],
) -> Array2<f64> {
    let timesteps = sequence.shape()[0];
    let features = sequence.shape()[1];

    // Generate random time warping
    let mut warp_steps = Vec::with_capacity(timesteps);
    let mut cumsum = 0.0;
    for _ in 0..timesteps {
        let step = rng.random_range((1.0 - sigma)..(1.0 + sigma));
        cumsum += step;
        warp_steps.push(cumsum);
    }

    // Normalize to [0, timesteps-1]
    let min_warp = warp_steps[0];
    let max_warp = warp_steps[timesteps - 1];
    let range = max_warp - min_warp;

    for step in &mut warp_steps {
        *step = (*step - min_warp) / range * (timesteps - 1) as f64;
    }

    // Create price/volume lookup set
    let price_volume_set: std::collections::HashSet<usize> =
        price_volume_cols.iter().cloned().collect();

    // Interpolate ONLY non-price/volume columns
    // Price/volume columns are copied as-is to preserve temporal relationships
    let mut warped = Array2::zeros((timesteps, features));

    // First, copy price/volume columns unchanged
    for &col in price_volume_cols {
        if col < features {
            for t in 0..timesteps {
                warped[[t, col]] = sequence[[t, col]];
            }
        }
    }

    // Then interpolate indicator columns
    for f in 0..features {
        if !price_volume_set.contains(&f) {
            let feature_values: Vec<f64> = (0..timesteps).map(|t| sequence[[t, f]]).collect();
            let warped_values = linear_interpolate(&feature_values, &warp_steps);

            for t in 0..timesteps {
                warped[[t, f]] = warped_values[t];
            }
        }
    }

    warped
}

/// Cubic interpolation for smooth curves
pub fn cubic_interpolate(points: &[f64], target_length: usize) -> Vec<f64> {
    let n_points = points.len();
    if n_points < 2 {
        return vec![1.0; target_length];
    }

    let mut result = Vec::with_capacity(target_length);
    let step = (n_points - 1) as f64 / (target_length - 1) as f64;

    for i in 0..target_length {
        let pos = i as f64 * step;
        let idx = pos.floor() as usize;

        if idx >= n_points - 1 {
            result.push(points[n_points - 1]);
        } else {
            // Linear interpolation between points
            let t = pos - idx as f64;
            let val = points[idx] * (1.0 - t) + points[idx + 1] * t;
            result.push(val);
        }
    }

    result
}

/// Linear interpolation for time warping
pub fn linear_interpolate(values: &[f64], new_indices: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut result = Vec::with_capacity(new_indices.len());

    for &idx in new_indices {
        let idx_clamped = idx.clamp(0.0, (n - 1) as f64);
        let i = idx_clamped.floor() as usize;

        if i >= n - 1 {
            result.push(values[n - 1]);
        } else {
            let t = idx_clamped - i as f64;
            let val = values[i] * (1.0 - t) + values[i + 1] * t;
            result.push(val);
        }
    }

    result
}

/// Check if two sequences overlap based on their indices
pub fn sequences_overlap(start1: usize, end1: usize, start2: usize, end2: usize) -> bool {
    start1 < end2 && start2 < end1
}

/// Calculate overlap ratio between two sequences
pub fn calculate_overlap_ratio(start1: usize, end1: usize, start2: usize, end2: usize) -> f64 {
    if !sequences_overlap(start1, end1, start2, end2) {
        return 0.0;
    }

    let overlap_start = start1.max(start2);
    let overlap_end = end1.min(end2);
    let overlap_size = overlap_end - overlap_start;
    let seq1_size = end1 - start1;

    overlap_size as f64 / seq1_size as f64
}
