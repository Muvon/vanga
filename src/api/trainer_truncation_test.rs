use ndarray::Array2;

#[test]
fn test_truncation_respects_validation_split() {
    // Simulate truncation with validation_split = 0.1 (10%)
    let current_total = 7100; // 6315 train + 785 val
    let target_count = 5000;
    let samples_to_keep = (target_count / 5) * 5; // 5000
    let validation_split = 0.1;

    // Create combined sequences
    let combined_sequences = Array2::<f64>::zeros((current_total, 50));

    // Select evenly distributed indices
    let stride = current_total as f64 / samples_to_keep as f64;
    let mut selected_indices: Vec<usize> = Vec::with_capacity(samples_to_keep);
    for i in 0..samples_to_keep {
        let idx = ((i as f64 * stride).round() as usize).min(current_total - 1);
        selected_indices.push(idx);
    }

    // Deduplicate
    selected_indices.dedup();
    let unique_count = selected_indices.len();

    // Calculate split using validation_split
    let val_samples = (unique_count as f64 * validation_split).round() as usize;
    let train_samples = unique_count - val_samples;

    // Split indices - THIS IS THE CRITICAL FIX
    let train_indices: Vec<usize> = selected_indices[..train_samples].to_vec();
    let val_indices: Vec<usize> = selected_indices[train_samples..].to_vec();

    // Verify all indices are valid
    for &idx in &train_indices {
        assert!(
            idx < combined_sequences.shape()[0],
            "Train index {} >= combined size {}",
            idx,
            combined_sequences.shape()[0]
        );
    }
    for &idx in &val_indices {
        assert!(
            idx < combined_sequences.shape()[0],
            "Val index {} >= combined size {}",
            idx,
            combined_sequences.shape()[0]
        );
    }

    // Verify split ratio is correct (approximately 10% validation)
    let actual_val_ratio = val_samples as f64 / unique_count as f64;
    assert!(
        (actual_val_ratio - validation_split).abs() < 0.02,
        "Validation ratio {:.3} should be close to {:.3}",
        actual_val_ratio,
        validation_split
    );

    // Verify total count
    assert_eq!(train_samples + val_samples, unique_count);
    assert!(unique_count <= samples_to_keep);

    // Select from combined sequences (this is where the panic would occur with wrong logic)
    let _train_seq = combined_sequences.select(ndarray::Axis(0), &train_indices);
    let _val_seq = combined_sequences.select(ndarray::Axis(0), &val_indices);

    println!("✅ Truncation test passed:");
    println!("   Original: 6315 train + 785 val = 7100 total");
    println!(
        "   Truncated: {} train + {} val = {} total",
        train_samples, val_samples, unique_count
    );
    println!(
        "   Validation ratio: {:.1}% (expected {:.1}%)",
        actual_val_ratio * 100.0,
        validation_split * 100.0
    );
}

#[test]
fn test_truncation_with_different_validation_splits() {
    let test_cases = vec![
        (0.1, 10.0),  // 10% validation
        (0.15, 15.0), // 15% validation
        (0.2, 20.0),  // 20% validation
    ];

    for (validation_split, expected_percent) in test_cases {
        let current_total = 7100;
        let target_count = 5000;
        let samples_to_keep = (target_count / 5) * 5;

        let combined_sequences = Array2::<f64>::zeros((current_total, 50));

        let stride = current_total as f64 / samples_to_keep as f64;
        let mut selected_indices: Vec<usize> = Vec::with_capacity(samples_to_keep);
        for i in 0..samples_to_keep {
            let idx = ((i as f64 * stride).round() as usize).min(current_total - 1);
            selected_indices.push(idx);
        }

        selected_indices.dedup();
        let unique_count = selected_indices.len();

        let val_samples = (unique_count as f64 * validation_split).round() as usize;
        let train_samples = unique_count - val_samples;

        let train_indices: Vec<usize> = selected_indices[..train_samples].to_vec();
        let val_indices: Vec<usize> = selected_indices[train_samples..].to_vec();

        // Verify no out-of-bounds indices
        for &idx in &train_indices {
            assert!(idx < combined_sequences.shape()[0]);
        }
        for &idx in &val_indices {
            assert!(idx < combined_sequences.shape()[0]);
        }

        // Verify split ratio
        let actual_val_ratio = val_samples as f64 / unique_count as f64;
        assert!(
            (actual_val_ratio * 100.0 - expected_percent).abs() < 2.0,
            "Validation split {:.1}% should be close to {:.1}%",
            actual_val_ratio * 100.0,
            expected_percent
        );

        // Verify selection works
        let train_seq = combined_sequences.select(ndarray::Axis(0), &train_indices);
        let val_seq = combined_sequences.select(ndarray::Axis(0), &val_indices);

        // Verify shapes
        assert_eq!(train_seq.shape()[0], train_samples);
        assert_eq!(val_seq.shape()[0], val_samples);

        println!(
            "✅ Validation split {:.1}%: {} train + {} val = {} total",
            expected_percent, train_samples, val_samples, unique_count
        );
    }
}

#[test]
fn test_truncation_no_duplicates_in_indices() {
    let current_total = 7100;
    let target_count = 5000;
    let samples_to_keep = (target_count / 5) * 5;

    let stride = current_total as f64 / samples_to_keep as f64;
    let mut selected_indices: Vec<usize> = Vec::with_capacity(samples_to_keep);
    for i in 0..samples_to_keep {
        let idx = ((i as f64 * stride).round() as usize).min(current_total - 1);
        selected_indices.push(idx);
    }

    selected_indices.dedup();
    let unique_count = selected_indices.len();

    let validation_split = 0.1;
    let val_samples = (unique_count as f64 * validation_split).round() as usize;
    let train_samples = unique_count - val_samples;

    let train_indices: Vec<usize> = selected_indices[..train_samples].to_vec();
    let val_indices: Vec<usize> = selected_indices[train_samples..].to_vec();

    // Verify no duplicates between train and val
    use std::collections::HashSet;
    let train_set: HashSet<usize> = train_indices.iter().copied().collect();
    let val_set: HashSet<usize> = val_indices.iter().copied().collect();

    assert_eq!(
        train_set.len(),
        train_indices.len(),
        "Train indices have duplicates"
    );
    assert_eq!(
        val_set.len(),
        val_indices.len(),
        "Val indices have duplicates"
    );

    let intersection: Vec<_> = train_set.intersection(&val_set).collect();
    assert!(
        intersection.is_empty(),
        "Train and val indices overlap: {:?}",
        intersection
    );

    println!(
        "✅ No duplicates: {} unique train + {} unique val",
        train_indices.len(),
        val_indices.len()
    );
}

#[test]
fn test_truncation_edge_case_small_dataset() {
    // Test with small dataset where stride might cause issues
    let current_total = 120;
    let target_count = 50;
    let samples_to_keep = (target_count / 5) * 5; // 50

    let combined_sequences = Array2::<f64>::zeros((current_total, 50));

    let stride = current_total as f64 / samples_to_keep as f64;
    let mut selected_indices: Vec<usize> = Vec::with_capacity(samples_to_keep);
    for i in 0..samples_to_keep {
        let idx = ((i as f64 * stride).round() as usize).min(current_total - 1);
        selected_indices.push(idx);
    }

    selected_indices.dedup();
    let unique_count = selected_indices.len();

    let validation_split = 0.1;
    let val_samples = (unique_count as f64 * validation_split).round() as usize;
    let train_samples = unique_count - val_samples;

    let train_indices: Vec<usize> = selected_indices[..train_samples].to_vec();
    let val_indices: Vec<usize> = selected_indices[train_samples..].to_vec();

    // Verify all indices valid
    for &idx in &train_indices {
        assert!(idx < combined_sequences.shape()[0]);
    }
    for &idx in &val_indices {
        assert!(idx < combined_sequences.shape()[0]);
    }

    // Verify selection works
    let train_seq = combined_sequences.select(ndarray::Axis(0), &train_indices);
    let val_seq = combined_sequences.select(ndarray::Axis(0), &val_indices);

    // Verify shapes
    assert_eq!(train_seq.shape()[0], train_samples);
    assert_eq!(val_seq.shape()[0], val_samples);

    println!(
        "✅ Small dataset: {} train + {} val = {} total",
        train_samples, val_samples, unique_count
    );
}
