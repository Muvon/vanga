//! Overlap Optimizer Module
//!
//! Dynamically adjusts sequence overlap to achieve target sample counts
//! while maintaining perfect class balance (20% per class).
//!
//! # Algorithm
//!
//! 1. **Mathematical Estimation**: Calculate initial overlap using closed-form formula
//! 2. **Binary Search**: Fine-tune overlap with actual calibration results
//! 3. **Intelligent Truncation**: Preserve temporal diversity while maintaining balance
//!
//! # Formula
//!
//! Given:
//! - N = total data points
//! - L = sequence length
//! - O = overlap (what we optimize)
//! - S = resulting sequences = (N - L) / (L - O) + 1
//!
//! To find optimal overlap for target samples T:
//! ```text
//! T = (N - L) / (L - O) + 1
//! T - 1 = (N - L) / (L - O)
//! L - O = (N - L) / (T - 1)
//! O = L - (N - L) / (T - 1)
//! ```

use crate::utils::error::{Result, VangaError};
use ndarray::{Array2, Array3, Axis};
use std::collections::HashMap;

/// Calculate optimal overlap to achieve target sample count
///
/// Uses closed-form mathematical formula to estimate the overlap needed.
/// This provides a good starting point for binary search optimization.
///
/// # Arguments
///
/// * `data_points` - Total number of data points in the dataset
/// * `sequence_length` - Length of each sequence
/// * `baseline_overlap` - Baseline overlap from configuration
/// * `target_samples` - Desired number of samples after calibration
///
/// # Returns
///
/// Estimated optimal overlap value (bounded by sequence_length - 1)
///
/// # Errors
///
/// Returns error if target is impossible to achieve even with maximum overlap
pub fn calculate_optimal_overlap(
    data_points: usize,
    sequence_length: usize,
    baseline_overlap: usize,
    target_samples: usize,
) -> Result<usize> {
    // Validate inputs
    if sequence_length >= data_points {
        return Err(VangaError::ConfigError(format!(
            "Sequence length ({}) must be less than data points ({})",
            sequence_length, data_points
        )));
    }

    if target_samples == 0 {
        return Ok(baseline_overlap);
    }

    // Calculate maximum possible sequences with max overlap
    let max_overlap = sequence_length - 1;
    let max_sequences = (data_points - sequence_length) / (sequence_length - max_overlap) + 1;

    if target_samples > max_sequences {
        return Err(VangaError::ConfigError(format!(
            "Target samples ({}) impossible to achieve. Maximum possible: {} (with overlap={})",
            target_samples, max_sequences, max_overlap
        )));
    }

    // Calculate minimum sequences with no overlap
    let min_sequences = (data_points - sequence_length) / sequence_length + 1;

    if target_samples <= min_sequences {
        // Target achievable with baseline or less overlap
        return Ok(baseline_overlap.min(sequence_length - 1));
    }

    // Use formula: O = L - (N - L) / (T - 1)
    let numerator = data_points - sequence_length;
    let denominator = target_samples - 1;

    if denominator == 0 {
        return Ok(baseline_overlap);
    }

    let step_size = numerator as f64 / denominator as f64;
    let optimal_overlap = sequence_length as f64 - step_size;

    // Bound the result
    let overlap = optimal_overlap.round() as usize;
    let overlap = overlap.max(baseline_overlap); // At least baseline
    let overlap = overlap.min(max_overlap); // At most max_overlap

    Ok(overlap)
}

/// Find optimal overlap using binary search with actual calibration
///
/// Iteratively tests different overlap values and runs calibration to find
/// the overlap that produces samples >= target_samples.
///
/// # Arguments
///
/// * `calibration_fn` - Function that takes overlap and returns balanced sample count
/// * `baseline_overlap` - Starting overlap from configuration
/// * `sequence_length` - Length of each sequence
/// * `target_samples` - Desired minimum number of samples
/// * `tolerance` - Acceptable deviation from target (e.g., 0.05 = ±5%)
///
/// # Returns
///
/// Tuple of (optimal_overlap, actual_sample_count)
///
/// # Algorithm
///
/// 1. Start with mathematical estimate as initial guess
/// 2. Binary search between baseline and max_overlap
/// 3. For each candidate: run calibration and count balanced samples
/// 4. Stop when samples >= target_samples * (1 - tolerance)
/// 5. Maximum 20 iterations to prevent infinite loops
pub fn find_optimal_overlap_with_calibration<F>(
    mut calibration_fn: F,
    data_points: usize,
    baseline_overlap: usize,
    sequence_length: usize,
    target_samples: usize,
    tolerance: f64,
) -> Result<(usize, usize)>
where
    F: FnMut(usize) -> Result<usize>,
{
    if target_samples == 0 {
        let samples = calibration_fn(baseline_overlap)?;
        return Ok((baseline_overlap, samples));
    }

    // Calculate initial estimate
    let initial_overlap = calculate_optimal_overlap(
        data_points,
        sequence_length,
        baseline_overlap,
        target_samples,
    )?;

    // Test initial estimate
    let initial_samples = calibration_fn(initial_overlap)?;
    let min_acceptable = (target_samples as f64 * (1.0 - tolerance)) as usize;

    if initial_samples >= min_acceptable {
        return Ok((initial_overlap, initial_samples));
    }

    // Binary search
    let mut lower_bound = baseline_overlap;
    let mut upper_bound = sequence_length - 1;
    let mut best_overlap = initial_overlap;
    let mut best_samples = initial_samples;
    let max_iterations = 20;

    for iteration in 0..max_iterations {
        if upper_bound <= lower_bound {
            break;
        }

        let mid_overlap = (lower_bound + upper_bound) / 2;

        // Avoid testing the same overlap twice
        if mid_overlap == best_overlap {
            break;
        }

        let samples = calibration_fn(mid_overlap)?;

        log::debug!(
            "🔍 Iteration {}: overlap={}, samples={}, target={}",
            iteration + 1,
            mid_overlap,
            samples,
            target_samples
        );

        if samples >= min_acceptable {
            // Found acceptable solution
            best_overlap = mid_overlap;
            best_samples = samples;

            // Try to find lower overlap that still works
            upper_bound = mid_overlap.saturating_sub(1);
        } else {
            // Need more overlap
            lower_bound = mid_overlap + 1;

            // Update best if this is closer to target
            if samples > best_samples {
                best_overlap = mid_overlap;
                best_samples = samples;
            }
        }
    }

    if best_samples < min_acceptable {
        log::warn!(
            "⚠️  Could not reach target samples {}. Best achieved: {} with overlap={}",
            target_samples,
            best_samples,
            best_overlap
        );
    }

    Ok((best_overlap, best_samples))
}

/// Truncate sequences while preserving temporal diversity and perfect balance
///
/// Selects evenly distributed sequences across time for each class to maintain
/// both temporal diversity and perfect 20% per class balance.
///
/// # Arguments
///
/// * `sequences` - Input sequences [batch_size, sequence_length, features]
/// * `targets` - Input targets [batch_size, num_classes]
/// * `target_count` - Desired total number of samples
///
/// # Returns
///
/// Tuple of (truncated_sequences, truncated_targets)
///
/// # Algorithm
///
/// 1. Validate input has perfect balance (20% per class)
/// 2. Calculate samples_per_class = target_count / 5
/// 3. For each class:
///    - Find all indices belonging to that class
///    - Select evenly distributed indices (stride = class_total / samples_per_class)
/// 4. Combine all selected indices maintaining chronological order
/// 5. Validate output is perfectly balanced
///
/// # Errors
///
/// Returns error if:
/// - Input is not perfectly balanced
/// - Target count is not divisible by 5
/// - Target count exceeds input size
pub fn truncate_balanced_sequences(
    sequences: &Array3<f64>,
    targets: &Array2<f64>,
    target_count: usize,
) -> Result<(Array3<f64>, Array2<f64>)> {
    let batch_size = sequences.shape()[0];
    let num_classes = targets.shape()[1];

    // Validate inputs
    if batch_size != targets.shape()[0] {
        return Err(VangaError::DataError(format!(
            "Sequence batch size ({}) doesn't match target batch size ({})",
            batch_size,
            targets.shape()[0]
        )));
    }

    if target_count > batch_size {
        return Err(VangaError::ConfigError(format!(
            "Target count ({}) exceeds available samples ({})",
            target_count, batch_size
        )));
    }

    if !target_count.is_multiple_of(num_classes) {
        return Err(VangaError::ConfigError(format!(
            "Target count ({}) must be divisible by number of classes ({})",
            target_count, num_classes
        )));
    }

    // Validate perfect balance in input
    let class_counts = count_classes(targets);
    let expected_per_class = batch_size / num_classes;
    for (class_idx, count) in class_counts.iter() {
        if *count != expected_per_class {
            return Err(VangaError::DataError(format!(
                "Input not perfectly balanced. Class {} has {} samples, expected {}",
                class_idx, count, expected_per_class
            )));
        }
    }

    let samples_per_class = target_count / num_classes;

    // Collect indices for each class
    let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..batch_size {
        let class_idx = targets.row(i).iter().position(|&x| x > 0.5).unwrap_or(0);
        class_indices.entry(class_idx).or_default().push(i);
    }

    // Select evenly distributed indices for each class
    let mut selected_indices = Vec::new();
    for class_idx in 0..num_classes {
        if let Some(indices) = class_indices.get(&class_idx) {
            let stride = indices.len() as f64 / samples_per_class as f64;
            for i in 0..samples_per_class {
                let idx = (i as f64 * stride).round() as usize;
                let idx = idx.min(indices.len() - 1);
                selected_indices.push(indices[idx]);
            }
        }
    }

    // Sort to maintain chronological order
    selected_indices.sort_unstable();

    // Extract selected sequences and targets
    let selected_sequences = sequences.select(Axis(0), &selected_indices);
    let selected_targets = targets.select(Axis(0), &selected_indices);

    // Validate output balance
    let output_class_counts = count_classes(&selected_targets);
    for (class_idx, count) in output_class_counts.iter() {
        if *count != samples_per_class {
            return Err(VangaError::DataError(format!(
                "Output not perfectly balanced. Class {} has {} samples, expected {}",
                class_idx, count, samples_per_class
            )));
        }
    }

    log::info!(
        "✂️  Truncated from {} to {} samples ({}% reduction, perfect balance maintained)",
        batch_size,
        target_count,
        ((batch_size - target_count) as f64 / batch_size as f64 * 100.0).round()
    );

    Ok((selected_sequences, selected_targets))
}

/// Count samples per class in one-hot encoded targets
pub(crate) fn count_classes(targets: &Array2<f64>) -> HashMap<usize, usize> {
    let mut counts = HashMap::new();
    for i in 0..targets.shape()[0] {
        let class_idx = targets.row(i).iter().position(|&x| x > 0.5).unwrap_or(0);
        *counts.entry(class_idx).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_optimal_overlap_basic() {
        let result = calculate_optimal_overlap(1000, 60, 30, 500);
        assert!(result.is_ok());
        let overlap = result.unwrap();
        assert!(overlap >= 30);
        assert!(overlap < 60);
    }

    #[test]
    fn test_calculate_optimal_overlap_impossible_target() {
        let result = calculate_optimal_overlap(1000, 60, 30, 100000);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_optimal_overlap_disabled() {
        let result = calculate_optimal_overlap(1000, 60, 30, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 30);
    }

    #[test]
    fn test_count_classes() {
        let targets = Array2::from_shape_vec(
            (10, 5),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, // Class 0
                1.0, 0.0, 0.0, 0.0, 0.0, // Class 0
                0.0, 1.0, 0.0, 0.0, 0.0, // Class 1
                0.0, 1.0, 0.0, 0.0, 0.0, // Class 1
                0.0, 0.0, 1.0, 0.0, 0.0, // Class 2
                0.0, 0.0, 1.0, 0.0, 0.0, // Class 2
                0.0, 0.0, 0.0, 1.0, 0.0, // Class 3
                0.0, 0.0, 0.0, 1.0, 0.0, // Class 3
                0.0, 0.0, 0.0, 0.0, 1.0, // Class 4
                0.0, 0.0, 0.0, 0.0, 1.0, // Class 4
            ],
        )
        .unwrap();

        let counts = count_classes(&targets);
        assert_eq!(counts.len(), 5);
        assert_eq!(counts[&0], 2);
        assert_eq!(counts[&1], 2);
        assert_eq!(counts[&2], 2);
        assert_eq!(counts[&3], 2);
        assert_eq!(counts[&4], 2);
    }
}
