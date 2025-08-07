#!/usr/bin/env cargo +nightly -Zscript

//! Test to validate that the shuffling algorithm produces different results for different epochs
//! This ensures we're not accidentally using the same data order every epoch

use vanga::model::lstm::training::shuffle_indices_deterministic;

#[test]
fn test_shuffle_produces_different_orders() {
    let sample_count = 100;

    // Test epoch 0
    let mut indices_epoch_0: Vec<usize> = (0..sample_count).collect();
    let seed_components_0 = [42u64, 0u64, sample_count as u64]; // seed=42, epoch=0
    let seed_0 = shuffle_indices_deterministic(&mut indices_epoch_0, &seed_components_0);

    // Test epoch 1
    let mut indices_epoch_1: Vec<usize> = (0..sample_count).collect();
    let seed_components_1 = [42u64, 1u64, sample_count as u64]; // seed=42, epoch=1
    let seed_1 = shuffle_indices_deterministic(&mut indices_epoch_1, &seed_components_1);

    // Test epoch 2
    let mut indices_epoch_2: Vec<usize> = (0..sample_count).collect();
    let seed_components_2 = [42u64, 2u64, sample_count as u64]; // seed=42, epoch=2
    let seed_2 = shuffle_indices_deterministic(&mut indices_epoch_2, &seed_components_2);

    println!("🔍 Shuffle Test Results:");
    println!(
        "   Epoch 0 seed: {}, first 10: {:?}",
        seed_0,
        &indices_epoch_0[0..10]
    );
    println!(
        "   Epoch 1 seed: {}, first 10: {:?}",
        seed_1,
        &indices_epoch_1[0..10]
    );
    println!(
        "   Epoch 2 seed: {}, first 10: {:?}",
        seed_2,
        &indices_epoch_2[0..10]
    );

    // Verify seeds are different
    assert_ne!(seed_0, seed_1, "Epoch 0 and 1 should have different seeds");
    assert_ne!(seed_1, seed_2, "Epoch 1 and 2 should have different seeds");
    assert_ne!(seed_0, seed_2, "Epoch 0 and 2 should have different seeds");

    // Verify shuffled orders are different
    assert_ne!(
        indices_epoch_0, indices_epoch_1,
        "Epoch 0 and 1 should have different orders"
    );
    assert_ne!(
        indices_epoch_1, indices_epoch_2,
        "Epoch 1 and 2 should have different orders"
    );
    assert_ne!(
        indices_epoch_0, indices_epoch_2,
        "Epoch 0 and 2 should have different orders"
    );

    // Verify all indices are still present (no data loss)
    for epoch_indices in [&indices_epoch_0, &indices_epoch_1, &indices_epoch_2] {
        let mut sorted = epoch_indices.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..sample_count).collect();
        assert_eq!(
            sorted, expected,
            "All original indices must be present exactly once"
        );
    }

    println!("✅ Shuffle validation passed: Different orders per epoch, all data preserved");
}

#[test]
fn test_shuffle_reproducibility() {
    let sample_count = 50;

    // Same seed components should produce same result
    let seed_components = [123u64, 5u64, sample_count as u64];

    let mut indices_1: Vec<usize> = (0..sample_count).collect();
    let seed_1 = shuffle_indices_deterministic(&mut indices_1, &seed_components);

    let mut indices_2: Vec<usize> = (0..sample_count).collect();
    let seed_2 = shuffle_indices_deterministic(&mut indices_2, &seed_components);

    assert_eq!(
        seed_1, seed_2,
        "Same seed components should produce same seed"
    );
    assert_eq!(
        indices_1, indices_2,
        "Same seed components should produce same shuffle order"
    );

    println!("✅ Shuffle reproducibility passed: Same inputs produce same outputs");
}

#[test]
fn test_shuffle_edge_cases() {
    // Test with small arrays
    let mut small_indices = vec![0, 1];
    let seed = shuffle_indices_deterministic(&mut small_indices, &[1, 2, 3]);
    assert!(small_indices == vec![0, 1] || small_indices == vec![1, 0]);
    println!(
        "✅ Small array shuffle: {:?} (seed: {})",
        small_indices, seed
    );

    // Test with single element
    let mut single_indices = vec![0];
    let seed = shuffle_indices_deterministic(&mut single_indices, &[4, 5, 6]);
    assert_eq!(single_indices, vec![0]);
    println!(
        "✅ Single element shuffle: {:?} (seed: {})",
        single_indices, seed
    );

    // Test with empty array
    let mut empty_indices: Vec<usize> = vec![];
    let seed = shuffle_indices_deterministic(&mut empty_indices, &[7, 8, 9]);
    assert_eq!(empty_indices, Vec::<usize>::new());
    println!(
        "✅ Empty array shuffle: {:?} (seed: {})",
        empty_indices, seed
    );
}
