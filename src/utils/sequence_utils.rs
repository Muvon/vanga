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
    if !(0.0..1.0).contains(&overlap) {
        return Err(VangaError::ConfigError(format!(
            "sequence_overlap must be between 0.0 and 1.0 (exclusive), got: {}",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_step_size() {
        // No overlap
        assert_eq!(calculate_step_size(0.0, 60), 60);

        // 50% overlap
        assert_eq!(calculate_step_size(0.5, 60), 30);

        // 80% overlap
        assert_eq!(calculate_step_size(0.8, 60), 12);

        // 90% overlap
        assert_eq!(calculate_step_size(0.9, 60), 6);

        // Maximum overlap
        assert_eq!(calculate_step_size(1.0, 60), 1);

        // Edge case: very high overlap
        assert_eq!(calculate_step_size(0.99, 60), 1);
    }

    #[test]
    fn test_calculate_sequence_indices() {
        let indices = calculate_sequence_indices(1000, 60, 12, 24).unwrap();

        // Should start at 0
        assert_eq!(indices[0], 0);

        // Should advance by step_size
        assert_eq!(indices[1], 12);
        assert_eq!(indices[2], 24);

        // Last index should leave room for sequence + horizon
        let last_idx = *indices.last().unwrap();
        assert!(last_idx + 60 + 24 <= 1000);
    }

    #[test]
    fn test_validate_sequence_overlap() {
        assert!(validate_sequence_overlap(0.0).is_ok());
        assert!(validate_sequence_overlap(0.5).is_ok());
        assert!(validate_sequence_overlap(0.99).is_ok());

        assert!(validate_sequence_overlap(-0.1).is_err());
        assert!(validate_sequence_overlap(1.0).is_err());
        assert!(validate_sequence_overlap(1.1).is_err());
    }

    #[test]
    fn test_calculate_expected_sequence_count() {
        // With step_size=12, sequence_length=60, horizon=24
        // Available length = 1000 - 60 - 24 + 1 = 917
        // Count = ceil(917 / 12) = 77
        assert_eq!(calculate_expected_sequence_count(1000, 60, 12, 24), 77);

        // Insufficient data
        assert_eq!(calculate_expected_sequence_count(50, 60, 12, 24), 0);
    }

    #[test]
    fn test_target_sequence_synchronization() {
        // Test case: 1000 data points, 60 sequence length, 24 horizon, 80% overlap
        let total_data = 1000;
        let seq_len = 60;
        let horizon = 24;
        let overlap = 0.8;

        let step_size = calculate_step_size(overlap, seq_len);
        assert_eq!(step_size, 12); // 20% of 60 = 12

        let indices = calculate_sequence_indices(total_data, seq_len, step_size, horizon).unwrap();

        // Verify indices are properly spaced
        for i in 1..indices.len() {
            assert_eq!(indices[i] - indices[i - 1], step_size);
        }

        // Verify last sequence has room for sequence + horizon
        let last_idx = *indices.last().unwrap();
        assert!(last_idx + seq_len + horizon <= total_data);
    }

    #[test]
    fn test_different_overlap_configurations() {
        let seq_len = 60;
        let total_data = 1000;
        let horizon = 24;

        // Test various overlap configurations - calculate expected values dynamically
        let overlaps = vec![0.0, 0.5, 0.8, 0.9, 0.95, 0.99];

        for overlap in overlaps {
            let step_size = calculate_step_size(overlap, seq_len);

            // Verify step size is reasonable
            assert!(
                step_size >= 1,
                "Step size must be at least 1 for overlap {}",
                overlap
            );
            assert!(
                step_size <= seq_len,
                "Step size cannot exceed sequence length for overlap {}",
                overlap
            );

            // For no overlap, step size should equal sequence length
            if overlap == 0.0 {
                assert_eq!(step_size, seq_len);
            }

            // For high overlap, step size should be small
            if overlap >= 0.9 {
                assert!(
                    step_size <= seq_len / 10,
                    "High overlap should result in small step size"
                );
            }

            let indices =
                calculate_sequence_indices(total_data, seq_len, step_size, horizon).unwrap();
            assert!(
                !indices.is_empty(),
                "No indices generated for overlap {}",
                overlap
            );

            // Verify step consistency
            if indices.len() > 1 {
                assert_eq!(indices[1] - indices[0], step_size);
            }

            println!(
                "Overlap {}: step_size={}, sequences={}",
                overlap,
                step_size,
                indices.len()
            );
        }
    }

    #[test]
    fn test_edge_cases() {
        // Minimum sequence length - ceil(0.2 * 5) = ceil(1.0) = 1
        let result = calculate_step_size(0.8, 5);
        println!("calculate_step_size(0.8, 5) = {}", result);
        assert_eq!(result, 1);

        // Maximum overlap
        assert_eq!(calculate_step_size(0.99, 100), 1);

        // No overlap
        assert_eq!(calculate_step_size(0.0, 100), 100);

        // Insufficient data
        let result = calculate_sequence_indices(50, 60, 12, 24);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_functions() {
        // Valid overlap values
        assert!(validate_sequence_overlap(0.0).is_ok());
        assert!(validate_sequence_overlap(0.5).is_ok());
        assert!(validate_sequence_overlap(0.99).is_ok());

        // Invalid overlap values
        assert!(validate_sequence_overlap(-0.1).is_err());
        assert!(validate_sequence_overlap(1.0).is_err());
        assert!(validate_sequence_overlap(1.1).is_err());
    }
}
