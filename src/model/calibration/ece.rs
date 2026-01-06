//! Expected Calibration Error (ECE) calculation
//!
//! ECE measures how well predicted confidences match actual accuracy.
//! Perfect calibration: ECE = 0.0 (70% confidence predictions are correct 70% of the time)
//! Poor calibration: ECE > 0.1 (confidence doesn't match accuracy)

use crate::utils::error::{Result, VangaError};
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

/// Number of bins for ECE calculation (research standard)
const NUM_BINS: usize = 15;

/// Reliability diagram data for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityDiagram {
    /// Bin boundaries (confidence ranges)
    pub bin_boundaries: Vec<f64>,
    /// Average confidence per bin
    pub bin_confidences: Vec<f64>,
    /// Average accuracy per bin
    pub bin_accuracies: Vec<f64>,
    /// Sample count per bin
    pub bin_counts: Vec<usize>,
    /// Overall ECE value
    pub ece: f64,
}

/// Calculate Expected Calibration Error (ECE)
///
/// ECE = Σ (n_b / n) * |confidence_b - accuracy_b|
/// where b is bin index, n_b is samples in bin, n is total samples
pub fn calculate_ece(predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
    let num_samples = predictions.nrows();
    if num_samples == 0 {
        return Ok(0.0);
    }

    if predictions.nrows() != targets.nrows() {
        return Err(VangaError::DataError(format!(
            "Predictions and targets must have same number of samples: {} vs {}",
            predictions.nrows(),
            targets.nrows()
        )));
    }

    // Initialize bins
    let mut bins: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); NUM_BINS];

    // OPTIMIZATION: Pre-calculate NUM_BINS as f64 to avoid repeated casting
    const NUM_BINS_F64: f64 = NUM_BINS as f64;

    // Assign samples to bins
    for (pred_row, target_row) in predictions
        .axis_iter(Axis(0))
        .zip(targets.axis_iter(Axis(0)))
    {
        // OPTIMIZATION: Use fold for efficient argmax (faster than max_by)
        let (pred_class, max_conf) =
            pred_row
                .iter()
                .enumerate()
                .fold((0, 0.0), |(max_idx, max_val), (idx, &val)| {
                    if val > max_val {
                        (idx, val)
                    } else {
                        (max_idx, max_val)
                    }
                });

        // OPTIMIZATION: Use fold for true class argmax
        let true_class = target_row
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

        // OPTIMIZATION: Use pre-calculated constant and avoid min() with direct clamp
        let bin_idx = ((max_conf * NUM_BINS_F64) as usize).min(NUM_BINS - 1);

        // Update bin statistics in single pass
        bins[bin_idx].0 += max_conf;
        bins[bin_idx].1 += (pred_class == true_class) as u8 as f64;
        bins[bin_idx].2 += 1;
    }

    // OPTIMIZATION: Calculate ECE with pre-calculated inverse
    let inv_num_samples = 1.0 / num_samples as f64;
    let mut ece = 0.0;

    for (conf_sum, acc_sum, count) in bins {
        if count > 0 {
            let inv_count = 1.0 / count as f64;
            let avg_conf = conf_sum * inv_count;
            let avg_acc = acc_sum * inv_count;
            let weight = count as f64 * inv_num_samples;
            ece += weight * (avg_conf - avg_acc).abs();
        }
    }

    Ok(ece)
}

/// Calculate per-class ECE to identify which classes need calibration
pub fn calculate_per_class_ece(
    predictions: &Array2<f64>,
    targets: &Array2<f64>,
) -> Result<[f64; 5]> {
    let num_samples = predictions.nrows();
    if num_samples == 0 {
        return Ok([0.0; 5]);
    }

    if predictions.ncols() != 5 || targets.ncols() != 5 {
        return Err(VangaError::DataError(format!(
            "Expected 5 classes, got predictions: {}, targets: {}",
            predictions.ncols(),
            targets.ncols()
        )));
    }

    let mut per_class_ece = [0.0; 5];

    // OPTIMIZATION: Pre-calculate NUM_BINS as f64
    const NUM_BINS_F64: f64 = NUM_BINS as f64;

    for class_idx in 0..5 {
        // Initialize bins for this class
        let mut bins: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); NUM_BINS];

        // Assign samples to bins (only for this class)
        for (pred_row, target_row) in predictions
            .axis_iter(Axis(0))
            .zip(targets.axis_iter(Axis(0)))
        {
            let confidence = pred_row[class_idx];
            let is_true_class = target_row[class_idx] > 0.5;

            // OPTIMIZATION: Use pre-calculated constant
            let bin_idx = ((confidence * NUM_BINS_F64) as usize).min(NUM_BINS - 1);

            // Update bin statistics
            bins[bin_idx].0 += confidence;
            bins[bin_idx].1 += is_true_class as u8 as f64;
            bins[bin_idx].2 += 1;
        }

        // Calculate ECE for this class
        let mut class_ece = 0.0;
        let total_count: usize = bins.iter().map(|(_, _, count)| count).sum();

        if total_count > 0 {
            let inv_total = 1.0 / total_count as f64;
            for (conf_sum, acc_sum, count) in bins {
                if count > 0 {
                    let inv_count = 1.0 / count as f64;
                    let avg_conf = conf_sum * inv_count;
                    let avg_acc = acc_sum * inv_count;
                    let weight = count as f64 * inv_total;
                    class_ece += weight * (avg_conf - avg_acc).abs();
                }
            }
        }

        per_class_ece[class_idx] = class_ece;
    }

    Ok(per_class_ece)
}

/// Generate reliability diagram data for visualization
pub fn generate_reliability_diagram(
    predictions: &Array2<f64>,
    targets: &Array2<f64>,
) -> Result<ReliabilityDiagram> {
    let num_samples = predictions.nrows();
    if num_samples == 0 {
        return Ok(ReliabilityDiagram {
            bin_boundaries: vec![],
            bin_confidences: vec![],
            bin_accuracies: vec![],
            bin_counts: vec![],
            ece: 0.0,
        });
    }

    // Initialize bins
    let mut bins: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); NUM_BINS];

    // Assign samples to bins
    for (pred_row, target_row) in predictions
        .axis_iter(Axis(0))
        .zip(targets.axis_iter(Axis(0)))
    {
        let (pred_class, max_conf) = pred_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let true_class = target_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap();

        let bin_idx = ((max_conf * NUM_BINS as f64) as usize).min(NUM_BINS - 1);

        bins[bin_idx].0 += max_conf;
        bins[bin_idx].1 += if pred_class == true_class { 1.0 } else { 0.0 };
        bins[bin_idx].2 += 1;
    }

    // Calculate ECE and extract bin data
    let mut ece = 0.0;
    let mut bin_boundaries = Vec::with_capacity(NUM_BINS + 1);
    let mut bin_confidences = Vec::with_capacity(NUM_BINS);
    let mut bin_accuracies = Vec::with_capacity(NUM_BINS);
    let mut bin_counts = Vec::with_capacity(NUM_BINS);

    for i in 0..=NUM_BINS {
        bin_boundaries.push(i as f64 / NUM_BINS as f64);
    }

    for (conf_sum, acc_sum, count) in bins {
        if count > 0 {
            let avg_conf = conf_sum / count as f64;
            let avg_acc = acc_sum / count as f64;
            let weight = count as f64 / num_samples as f64;
            ece += weight * (avg_conf - avg_acc).abs();

            bin_confidences.push(avg_conf);
            bin_accuracies.push(avg_acc);
        } else {
            bin_confidences.push(0.0);
            bin_accuracies.push(0.0);
        }
        bin_counts.push(count);
    }

    Ok(ReliabilityDiagram {
        bin_boundaries,
        bin_confidences,
        bin_accuracies,
        bin_counts,
        ece,
    })
}

/// Calculate ECE gradient for temperature optimization
///
/// Gradient of ECE with respect to temperature T for a given class
pub fn calculate_ece_gradient_for_temperature(
    logits: &Array2<f64>,
    targets: &Array2<f64>,
    temperature: f64,
    _class_idx: usize,
) -> Result<f64> {
    let num_samples = logits.nrows();
    if num_samples == 0 {
        return Ok(0.0);
    }

    // Calculate predictions with current temperature
    let mut predictions = Array2::zeros((num_samples, 5));
    for (i, logit_row) in logits.axis_iter(Axis(0)).enumerate() {
        let scaled_logits: Vec<f64> = logit_row.iter().map(|&x| x / temperature).collect();
        let max_logit = scaled_logits
            .iter()
            .fold(f64::NEG_INFINITY, |max, &val| max.max(val));
        let exp_sum: f64 = scaled_logits.iter().map(|&x| (x - max_logit).exp()).sum();

        for j in 0..5 {
            predictions[[i, j]] = ((scaled_logits[j] - max_logit).exp()) / exp_sum;
        }
    }

    // Calculate ECE at current temperature
    let ece_current = calculate_ece(&predictions, targets)?;

    // Calculate ECE at slightly higher temperature (numerical gradient)
    let delta_t = 0.01;
    let temp_plus = temperature + delta_t;

    let mut predictions_plus = Array2::zeros((num_samples, 5));
    for (i, logit_row) in logits.axis_iter(Axis(0)).enumerate() {
        let scaled_logits: Vec<f64> = logit_row.iter().map(|&x| x / temp_plus).collect();
        let max_logit = scaled_logits
            .iter()
            .fold(f64::NEG_INFINITY, |max, &val| max.max(val));
        let exp_sum: f64 = scaled_logits.iter().map(|&x| (x - max_logit).exp()).sum();

        for j in 0..5 {
            predictions_plus[[i, j]] = ((scaled_logits[j] - max_logit).exp()) / exp_sum;
        }
    }

    let ece_plus = calculate_ece(&predictions_plus, targets)?;

    // Numerical gradient
    let gradient = (ece_plus - ece_current) / delta_t;

    Ok(gradient)
}
