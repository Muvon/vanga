//! Sequence generation utilities for consistent step size calculation and alignment
//!
//! This module provides unified functions for calculating step sizes and ensuring
//! proper alignment between sequences and targets across the entire codebase.

use crate::utils::error::{Result, VangaError};

/// Calculate consistent step size for sequence generation based on overlap configuration
///
/// This function ensures that both sequence generation and walk-forward windows
/// use the same step size calculation logic to maintain target-sequence alignment.
///
/// # Arguments
/// * `sequence_overlap` - Overlap ratio between sequences (0.0 = no overlap, 0.9 = 90% overlap)
/// * `sequence_length` - Length of each sequence window
///
/// # Returns
/// * Step size for advancing sequence windows
///
/// # Examples
/// ```
/// // No overlap - advance by full sequence length
/// assert_eq!(calculate_step_size(0.0, 60), 60);
///
/// // 80% overlap - advance by 20% of sequence length
/// assert_eq!(calculate_step_size(0.8, 60), 12);
///
/// // 90% overlap - advance by 10% of sequence length
/// assert_eq!(calculate_step_size(0.9, 60), 6);
/// ```
pub fn calculate_step_size(sequence_overlap: f64, sequence_length: usize) -> usize {
    if sequence_overlap <= 0.0 {
        // No overlap - advance by full sequence length
        sequence_length
    } else if sequence_overlap >= 1.0 {
        // Maximum overlap - advance by 1 (every timestep)
        1
    } else {
        // Calculate step size based on overlap percentage
        // overlap = 0.8 means 80% overlap, so advance by 20% of sequence length
        let advance_ratio = 1.0 - sequence_overlap;
        let step_size_f64 = advance_ratio * sequence_length as f64;

        // Handle floating point precision issues by rounding to nearest integer first
        let step_size = if (step_size_f64 - step_size_f64.round()).abs() < 1e-10 {
            step_size_f64.round() as usize
        } else {
            step_size_f64.ceil() as usize
        };

        // Ensure minimum step size of 1
        step_size.max(1)
    }
}

/// Calculate sequence indices for proper target alignment
///
/// This function generates the actual starting indices where sequences will be created,
/// taking into account the step size and data constraints.
///
/// # Arguments
/// * `total_data_length` - Total number of data points available
/// * `sequence_length` - Length of each sequence
/// * `step_size` - Step size for advancing sequences
/// * `max_horizon_steps` - Maximum prediction horizon
///
/// # Returns
/// * Vector of starting indices for sequence generation
pub fn calculate_sequence_indices(
    total_data_length: usize,
    sequence_length: usize,
    step_size: usize,
    max_horizon_steps: usize,
) -> Result<Vec<usize>> {
    if total_data_length < sequence_length + max_horizon_steps {
        return Err(VangaError::DataError(format!(
            "Insufficient data: need {} points (sequence {} + horizon {}), have {}",
            sequence_length + max_horizon_steps,
            sequence_length,
            max_horizon_steps,
            total_data_length
        )));
    }

    let mut indices = Vec::new();
    let mut current_idx = 0;

    // Generate indices while ensuring we have enough data for sequence + horizon
    while current_idx + sequence_length + max_horizon_steps <= total_data_length {
        indices.push(current_idx);
        current_idx += step_size;
    }

    if indices.is_empty() {
        return Err(VangaError::DataError(
            "No valid sequence indices could be generated with current parameters".to_string(),
        ));
    }

    Ok(indices)
}

/// Validate sequence overlap parameter
pub fn validate_sequence_overlap(overlap: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&overlap) {
        return Err(VangaError::ConfigError(format!(
            "sequence_overlap must be between 0.0 and 1.0 (inclusive), got: {}",
            overlap
        )));
    }
    Ok(())
}

/// Validate target-sequence synchronization
pub fn validate_target_sequence_alignment(
    sequence_count: usize,
    target_indices: &[usize],
    sequence_indices: &[usize],
    data_length: usize,
) -> Result<()> {
    // Check sequence count matches
    if sequence_count != sequence_indices.len() {
        return Err(VangaError::DataError(format!(
            "Sequence count mismatch: expected {}, got {}",
            sequence_count,
            sequence_indices.len()
        )));
    }

    // Check all indices are within bounds
    for &idx in sequence_indices {
        if idx >= data_length {
            return Err(VangaError::DataError(format!(
                "Sequence index {} exceeds data length {}",
                idx, data_length
            )));
        }
    }

    // Check target indices alignment
    for &idx in target_indices {
        if idx >= data_length {
            return Err(VangaError::DataError(format!(
                "Target index {} exceeds data length {}",
                idx, data_length
            )));
        }
    }

    log::info!(
        "✅ Target-sequence alignment validated: {} sequences, {} targets, data_length={}",
        sequence_count,
        target_indices.len(),
        data_length
    );

    Ok(())
}

/// Log detailed synchronization information for debugging
pub fn log_synchronization_details(
    sequence_indices: &[usize],
    step_size: usize,
    sequence_length: usize,
    max_horizon_steps: usize,
    overlap: f64,
) {
    log::info!("🔍 SYNCHRONIZATION DETAILS:");
    log::info!("   • Sequence overlap: {:.1}%", overlap * 100.0);
    log::info!("   • Step size: {}", step_size);
    log::info!("   • Sequence length: {}", sequence_length);
    log::info!("   • Max horizon steps: {}", max_horizon_steps);
    log::info!("   • Total sequences: {}", sequence_indices.len());

    if !sequence_indices.is_empty() {
        log::info!("   • First sequence index: {}", sequence_indices[0]);
        log::info!(
            "   • Last sequence index: {}",
            sequence_indices.last().unwrap()
        );

        // Show step pattern for first few sequences
        let sample_size = sequence_indices.len().min(5);
        let steps: Vec<String> = sequence_indices[..sample_size]
            .iter()
            .map(|&idx| idx.to_string())
            .collect();
        log::info!(
            "   • Index pattern (first {}): [{}]",
            sample_size,
            steps.join(", ")
        );

        // Verify step consistency
        if sequence_indices.len() > 1 {
            let actual_step = sequence_indices[1] - sequence_indices[0];
            if actual_step != step_size {
                log::warn!(
                    "⚠️  Step size inconsistency: expected {}, actual {}",
                    step_size,
                    actual_step
                );
            }
        }
    }
}

/// Calculate the number of sequences that will be generated
pub fn calculate_expected_sequence_count(
    total_data_length: usize,
    sequence_length: usize,
    step_size: usize,
    max_horizon_steps: usize,
) -> usize {
    if total_data_length < sequence_length + max_horizon_steps {
        return 0;
    }

    let available_length = total_data_length - sequence_length - max_horizon_steps + 1;
    available_length.div_ceil(step_size) // Ceiling division
}
