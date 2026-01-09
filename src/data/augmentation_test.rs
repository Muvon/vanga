use crate::data::augmentation::*;
use ndarray::Array2;

#[test]
fn test_augmentation_config_creation() {
    // Config should be created with default values
    let config = AugmentationConfig::from_overlap(0.1);
    assert!((config.magnitude_sigma - 0.2).abs() < 1e-6);
    assert!((config.jitter_sigma - 0.03).abs() < 1e-6);

    let config = AugmentationConfig::from_overlap(0.9);
    assert!((config.scaling_sigma - 0.1).abs() < 1e-6);
}

#[test]
fn test_magnitude_warp_preserves_shape() {
    let mut rng = rand::rng();
    let sequence = Array2::ones((100, 10));
    let warped = magnitude_warp(&sequence, 0.2, &mut rng);

    assert_eq!(warped.shape(), sequence.shape());
    assert_ne!(warped, sequence); // Should be different
}

#[test]
fn test_jitter_adds_noise() {
    let mut rng = rand::rng();
    let sequence = Array2::ones((50, 5));
    let jittered = jitter(&sequence, 0.03, &mut rng);

    assert_eq!(jittered.shape(), sequence.shape());
    // Values should be close to 1.0 but not exactly 1.0
    let mean = jittered.mean().unwrap();
    assert!(mean > 0.95 && mean < 1.05);
}

#[test]
fn test_scaling_uniform() {
    let mut rng = rand::rng();
    let sequence = Array2::ones((50, 5));
    let scaled = scaling(&sequence, 0.1, &mut rng);

    assert_eq!(scaled.shape(), sequence.shape());
    // All values should be scaled by same factor
    let first_val = scaled[[0, 0]];
    for i in 0..50 {
        for j in 0..5 {
            assert!((scaled[[i, j]] - first_val).abs() < 1e-10);
        }
    }
}

#[test]
fn test_time_warp_preserves_shape() {
    let mut rng = rand::rng();
    let mut sequence = Array2::zeros((100, 5));
    // Create linear pattern
    for i in 0..100 {
        for j in 0..5 {
            sequence[[i, j]] = i as f64;
        }
    }

    let warped = time_warp(&sequence, 0.2, &mut rng);
    assert_eq!(warped.shape(), sequence.shape());
}

#[test]
fn test_augment_sequence_applies_multiple_techniques() {
    let mut rng = rand::rng();
    let sequence = Array2::ones((100, 10));
    let config = AugmentationConfig::from_overlap(0.9);

    let augmented = augment_sequence(&sequence, &config, &mut rng);

    assert_eq!(augmented.shape(), sequence.shape());
    assert_ne!(augmented, sequence); // Should be different due to magnitude warp
}

#[test]
fn test_sequences_overlap_detection() {
    // Overlapping sequences
    assert!(sequences_overlap(0, 100, 50, 150));
    assert!(sequences_overlap(50, 150, 0, 100));
    assert!(sequences_overlap(0, 100, 0, 100)); // Same sequence

    // Non-overlapping sequences
    assert!(!sequences_overlap(0, 100, 100, 200));
    assert!(!sequences_overlap(100, 200, 0, 100));
    assert!(!sequences_overlap(0, 50, 100, 150));
}

#[test]
fn test_calculate_overlap_ratio() {
    // 50% overlap
    let ratio = calculate_overlap_ratio(0, 100, 50, 150);
    assert!((ratio - 0.5).abs() < 1e-6);

    // 10% overlap
    let ratio = calculate_overlap_ratio(0, 100, 90, 110);
    assert!((ratio - 0.1).abs() < 1e-6);

    // No overlap
    let ratio = calculate_overlap_ratio(0, 100, 100, 200);
    assert_eq!(ratio, 0.0);

    // Full overlap
    let ratio = calculate_overlap_ratio(0, 100, 0, 100);
    assert!((ratio - 1.0).abs() < 1e-6);
}

#[test]
fn test_cubic_interpolate() {
    let points = vec![1.0, 2.0, 3.0, 4.0];
    let interpolated = cubic_interpolate(&points, 10);

    assert_eq!(interpolated.len(), 10);
    assert!((interpolated[0] - 1.0).abs() < 0.1);
    assert!((interpolated[9] - 4.0).abs() < 0.1);
}

#[test]
fn test_linear_interpolate() {
    let values = vec![0.0, 10.0, 20.0, 30.0];
    let indices = vec![0.0, 1.5, 2.5, 3.0];
    let interpolated = linear_interpolate(&values, &indices);

    assert_eq!(interpolated.len(), 4);
    assert!((interpolated[0] - 0.0).abs() < 1e-6);
    assert!((interpolated[1] - 15.0).abs() < 1e-6); // Midpoint between 10 and 20
    assert!((interpolated[3] - 30.0).abs() < 1e-6);
}

#[test]
fn test_augmentation_deterministic_with_seed() {
    use rand::SeedableRng;
    let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
    let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);

    let sequence = Array2::ones((50, 5));
    let config = AugmentationConfig::from_overlap(0.9);

    let aug1 = augment_sequence(&sequence, &config, &mut rng1);
    let aug2 = augment_sequence(&sequence, &config, &mut rng2);

    // Same seed should produce same result
    assert_eq!(aug1, aug2);
}

#[test]
fn test_augmentation_different_with_different_seed() {
    use rand::SeedableRng;
    let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
    let mut rng2 = rand::rngs::StdRng::seed_from_u64(43);

    let sequence = Array2::ones((50, 5));
    let config = AugmentationConfig::from_overlap(0.9);

    let aug1 = augment_sequence(&sequence, &config, &mut rng1);
    let aug2 = augment_sequence(&sequence, &config, &mut rng2);

    // Different seeds should produce different results
    assert_ne!(aug1, aug2);
}

#[test]
fn test_magnitude_warp_range() {
    let mut rng = rand::rng();
    let sequence = Array2::ones((100, 5));
    let sigma = 0.2;
    let warped = magnitude_warp(&sequence, sigma, &mut rng);

    // Values should be within reasonable range (1.0 ± sigma)
    for val in warped.iter() {
        assert!(*val > 1.0 - sigma - 0.1 && *val < 1.0 + sigma + 0.1);
    }
}

#[test]
fn test_jitter_range() {
    let mut rng = rand::rng();
    let sequence = Array2::ones((100, 5));
    let sigma = 0.03;
    let jittered = jitter(&sequence, sigma, &mut rng);

    // With Gaussian noise, values should be mostly within 3*sigma (99.7% of values)
    // but we check a wider range to avoid flaky tests
    for val in jittered.iter() {
        assert!(*val > 1.0 - 4.0 * sigma && *val < 1.0 + 4.0 * sigma);
    }
}

#[test]
fn test_scaling_range() {
    let mut rng = rand::rng();
    let sequence = Array2::ones((100, 5));
    let sigma = 0.1;
    let scaled = scaling(&sequence, sigma, &mut rng);

    let mean = scaled.mean().unwrap();
    // Mean should be within range (1.0 ± sigma)
    assert!(mean > 1.0 - sigma && mean < 1.0 + sigma);
}

#[test]
fn test_augmentation_preserves_non_nan() {
    let mut rng = rand::rng();
    let sequence = Array2::from_elem((100, 5), 1.0);
    let config = AugmentationConfig::from_overlap(0.9);

    let augmented = augment_sequence(&sequence, &config, &mut rng);

    // No NaN values should be introduced
    for val in augmented.iter() {
        assert!(!val.is_nan());
        assert!(!val.is_infinite());
    }
}

#[test]
fn test_single_feature_augmentation() {
    let mut rng = rand::rng();
    let sequence = Array2::ones((100, 1)); // Single feature
    let config = AugmentationConfig::from_overlap(0.9);

    let augmented = augment_sequence(&sequence, &config, &mut rng);

    assert_eq!(augmented.shape(), [100, 1]);
    assert_ne!(augmented, sequence);
}

#[test]
fn test_large_sequence_augmentation() {
    let mut rng = rand::rng();
    let sequence = Array2::ones((1000, 50)); // Large sequence
    let config = AugmentationConfig::from_overlap(0.9);

    let augmented = augment_sequence(&sequence, &config, &mut rng);

    assert_eq!(augmented.shape(), sequence.shape());
    assert_ne!(augmented, sequence);
}
