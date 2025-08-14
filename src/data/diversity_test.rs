// use crate::data::balance::SequenceWithTargets; // Unused import
use crate::data::diversity::*;
use crate::utils::error::Result;
use ndarray::Array2;
// use std::collections::HashMap; // Unused import

/// Test cosine distance calculation for normalized sequences
#[test]
fn test_cosine_distance_normalized_sequences() -> Result<()> {
    let selector = DiversitySelector::new(DiversityConfig::default());

    // Create two normalized sequences with different patterns
    let seq1 = Array2::from_shape_vec(
        (5, 3),
        vec![
            -1.0, -0.5, -1.0, -0.5, 0.0, -0.5, 0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 1.0, 1.5, 1.0,
        ],
    )
    .unwrap();

    let seq2 = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 1.5, 1.0, 0.5, 1.0, 0.5, 0.0, 0.5, 0.0, -0.5, 0.0, -0.5, -1.0, -0.5, -1.0,
        ],
    )
    .unwrap();

    // Calculate distance
    let distance = selector.calculate_cosine_distance(&seq1, &seq2)?;

    // Distance should be in [0, 1] range
    assert!((0.0..=1.0).contains(&distance));

    println!("✅ Cosine distance test passed: {:.4}", distance);

    Ok(())
}

/// Test edge cases with zero vectors
#[test]
fn test_cosine_distance_edge_cases() -> Result<()> {
    let selector = DiversitySelector::new(DiversityConfig::default());

    let zero_seq = Array2::zeros((3, 2));
    let normal_seq = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let distance = selector.calculate_cosine_distance(&zero_seq, &normal_seq)?;
    assert_eq!(distance, 1.0);

    println!("✅ Edge cases test passed: {:.4}", distance);

    Ok(())
}
