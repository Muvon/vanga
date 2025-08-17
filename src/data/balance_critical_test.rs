//! Critical tests for balance logic - MUST achieve perfect 20% per class distribution

use crate::data::balance::*;
use crate::targets::TargetType;
use ndarray::Array2;

#[test]
fn test_perfect_balance_mandatory() {
    // Create test sequences with severe class imbalance
    let mut sequences = Vec::new();

    // Create 100 sequences with severe imbalance:
    // Class 0: 50 sequences (50%)
    // Class 1: 30 sequences (30%)
    // Class 2: 15 sequences (15%)
    // Class 3: 4 sequences (4%)
    // Class 4: 1 sequence (1%)

    for i in 0..100 {
        let class = if i < 50 {
            0
        }
        // 50 sequences
        else if i < 80 {
            1
        }
        // 30 sequences
        else if i < 95 {
            2
        }
        // 15 sequences
        else if i < 99 {
            3
        }
        // 4 sequences
        else {
            4
        }; // 1 sequence

        let targets = vec![TargetData {
            target_type: TargetType::PriceLevel,
            horizon: "1h".to_string(),
            class,
            strength: 0.5,
        }];

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx: i * 10,
            end_idx: i * 10 + 30, // 30-length sequences with potential overlap
            sequence_data: Array2::zeros((30, 5)),
            targets,
        });
    }

    // Test with high overlap allowed (0.8 = 80%) to enable balance through overlap
    let config = BalanceConfig {
        max_overlap: 0.8,
        prefer_non_overlapping: false, // Allow overlap for balance
        min_sequences_per_class: 10,
    };

    let balancer = SequenceBalancer::new(config);

    // Try to select 50 sequences (10 per class for perfect 20% balance)
    let result = balancer.balance_sequences_for_window(
        &sequences,
        TargetType::PriceLevel,
        "1h",
        &[],             // No validation sequences to exclude
        Some((0, 1000)), // Wide window range
    );

    match result {
        Ok(selection) => {
            // VERIFY PERFECT BALANCE
            let total_selected = selection.selected_indices.len();
            println!("Total selected: {}", total_selected);
            println!("Class distribution: {:?}", selection.class_distribution);

            // Check each class has exactly 20%
            for class in 0..5 {
                let count = selection.class_distribution.get(&class).unwrap_or(&0);
                let percentage = (*count as f64 / total_selected as f64) * 100.0;
                println!("Class {}: {} sequences ({:.1}%)", class, count, percentage);

                // MUST be exactly 20% (or very close due to rounding)
                assert!(
                    (percentage - 20.0).abs() < 1.0,
                    "Class {} has {:.1}% instead of 20% - BALANCE FAILED!",
                    class,
                    percentage
                );
            }

            // Verify we selected exactly the expected number per class
            let expected_per_class = total_selected / 5;
            for class in 0..5 {
                let count = selection.class_distribution.get(&class).unwrap_or(&0);
                assert_eq!(
                    *count, expected_per_class,
                    "Class {} has {} sequences instead of {} - PERFECT BALANCE REQUIRED!",
                    class, count, expected_per_class
                );
            }

            println!("✅ PERFECT BALANCE ACHIEVED!");
        }
        Err(e) => {
            panic!(
                "Balance selection failed: {} - THIS SHOULD NEVER HAPPEN!",
                e
            );
        }
    }
}

#[test]
fn test_overlap_configuration_respected() {
    // Create sequences with potential for overlap
    let mut sequences = Vec::new();

    // Create 20 sequences with multiple classes but heavy overlap
    for i in 0..20 {
        let class = (i % 5) as i32; // Distribute across all 5 classes, cast to i32
        let targets = vec![TargetData {
            target_type: TargetType::PriceLevel,
            horizon: "1h".to_string(),
            class,
            strength: 0.5,
        }];

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx: i * 5, // Heavy overlap: seq0=[0,30], seq1=[5,35], seq2=[10,40], etc.
            end_idx: i * 5 + 30,
            sequence_data: Array2::zeros((30, 5)),
            targets,
        });
    }

    // Test with 40% overlap configuration
    let config = BalanceConfig {
        max_overlap: 0.4, // 40% overlap allowed
        prefer_non_overlapping: false,
        min_sequences_per_class: 5,
    };

    let balancer = SequenceBalancer::new(config);

    let result = balancer
        .balance_sequences_for_window(
            &sequences,
            TargetType::PriceLevel,
            "1h",
            &[],
            Some((0, 200)),
        )
        .expect("Should succeed");

    println!("Average overlap: {:.1}%", result.avg_overlap * 100.0);

    // The average overlap should be UNDER the configured maximum (40%)
    assert!(
        result.avg_overlap * 100.0 <= 40.0,
        "Average overlap {:.1}% exceeds configured maximum 40% - OVERLAP LIMIT VIOLATED!",
        result.avg_overlap * 100.0
    );

    // Also verify it's using reasonable overlap (not too low)
    assert!(
        result.avg_overlap * 100.0 >= 10.0,
        "Average overlap {:.1}% is too low - should use some overlap for efficiency",
        result.avg_overlap * 100.0
    );

    println!(
        "✅ OVERLAP CONFIGURATION RESPECTED: {:.1}% ≤ 40% maximum",
        result.avg_overlap * 100.0
    );
}

#[test]
fn test_insufficient_sequences_for_balance() {
    // Create scenario where perfect balance is impossible
    let mut sequences = Vec::new();

    // Only 3 sequences total, but need 5 classes
    for i in 0..3 {
        let targets = vec![TargetData {
            target_type: TargetType::PriceLevel,
            horizon: "1h".to_string(),
            class: i as i32,
            strength: 0.5,
        }];

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx: i * 10,
            end_idx: i * 10 + 30,
            sequence_data: Array2::zeros((30, 5)),
            targets,
        });
    }

    let config = BalanceConfig::default();
    let balancer = SequenceBalancer::new(config);

    // This should FAIL with a clear error
    let result = balancer.balance_sequences_for_window(
        &sequences,
        TargetType::PriceLevel,
        "1h",
        &[],
        Some((0, 100)),
    );

    assert!(
        result.is_err(),
        "Should fail when perfect balance is impossible - MUST BE FATAL ERROR!"
    );

    if let Err(e) = result {
        println!("Expected error: {}", e);
        // Error should mention balance impossibility
        assert!(
            e.to_string().contains("balance") || e.to_string().contains("class"),
            "Error should mention balance/class issues: {}",
            e
        );
    }
}

#[test]
fn test_overlap_enables_balance_for_rare_classes() {
    // Create scenario where rare classes need overlap to achieve balance
    let mut sequences = Vec::new();

    // Class distribution:
    // Class 0: 40 sequences (abundant)
    // Class 1: 40 sequences (abundant)
    // Class 2: 15 sequences (moderate)
    // Class 3: 4 sequences (rare - needs overlap)
    // Class 4: 1 sequence (very rare - needs heavy overlap)

    for i in 0..100 {
        let class = if i < 40 {
            0
        } else if i < 80 {
            1
        } else if i < 95 {
            2
        } else if i < 99 {
            3
        } else {
            4
        };

        let targets = vec![TargetData {
            target_type: TargetType::PriceLevel,
            horizon: "1h".to_string(),
            class,
            strength: 0.5,
        }];

        // Create overlapping sequences for rare classes
        let start_idx = if class >= 3 {
            // Rare classes: create overlapping sequences
            i * 2 // More overlap potential
        } else {
            // Common classes: normal spacing
            i * 10
        };

        sequences.push(SequenceWithTargets {
            sequence_idx: i,
            start_idx,
            end_idx: start_idx + 30,
            sequence_data: Array2::zeros((30, 5)),
            targets,
        });
    }

    // High overlap allowed to enable balance through overlap
    let config = BalanceConfig {
        max_overlap: 0.9, // 90% overlap allowed
        prefer_non_overlapping: false,
        min_sequences_per_class: 10, // Need 10 per class
    };

    let balancer = SequenceBalancer::new(config);

    // Try to select 50 sequences (10 per class)
    let result = balancer.balance_sequences_for_window(
        &sequences,
        TargetType::PriceLevel,
        "1h",
        &[],
        Some((0, 1000)),
    );

    match result {
        Ok(selection) => {
            println!("Selected {} sequences", selection.selected_indices.len());
            println!("Class distribution: {:?}", selection.class_distribution);

            // MUST achieve perfect balance limited by rarest class (1 sequence)
            // Since class 4 has only 1 sequence, all classes should have 1 sequence
            for class in 0..5 {
                let count = selection.class_distribution.get(&class).unwrap_or(&0);
                assert_eq!(
                    *count, 1,
                    "Class {} has {} sequences instead of 1 - should be limited by rarest class!",
                    class, count
                );
            }

            println!("✅ OVERLAP ENABLED PERFECT BALANCE for rare classes!");
        }
        Err(e) => {
            panic!(
                "Should achieve balance through overlap: {} - LOGIC BROKEN!",
                e
            );
        }
    }
}

#[test]
fn test_missing_classes_fatal_error() {
    // Test that missing classes cause FATAL errors (this is correct behavior)
    // Real data should NEVER have missing classes - this indicates target generation failure

    let mut sequences = Vec::new();

    // Create sequences for ONLY 3 classes (1, 2, 3) - missing 0 and 4
    let class_data = [(1, 22), (2, 260), (3, 57)];
    let mut seq_idx = 0;

    for (class, count) in class_data {
        for _ in 0..count {
            let targets = vec![TargetData {
                target_type: TargetType::PriceLevel,
                horizon: "16h".to_string(),
                class,
                strength: 0.5,
            }];

            sequences.push(SequenceWithTargets {
                sequence_idx: seq_idx,
                start_idx: seq_idx * 10,
                end_idx: seq_idx * 10 + 30,
                sequence_data: Array2::zeros((30, 5)),
                targets,
            });
            seq_idx += 1;
        }
    }

    let config = BalanceConfig {
        max_overlap: 0.9,
        prefer_non_overlapping: false,
        min_sequences_per_class: 10,
    };

    let balancer = SequenceBalancer::new(config);

    // This should FAIL with missing classes error
    let result = balancer.balance_sequences_for_window(
        &sequences,
        TargetType::PriceLevel,
        "16h",
        &[],
        Some((0, 100000)),
    );

    match result {
        Err(e) => {
            let error_msg = format!("{}", e);
            assert!(
                error_msg.contains("Missing classes"),
                "Should detect missing classes: {}",
                error_msg
            );
            assert!(
                error_msg.contains("[0, 4]"),
                "Should identify missing classes 0 and 4: {}",
                error_msg
            );
            println!("✅ CORRECTLY DETECTED MISSING CLASSES: {}", error_msg);
        }
        Ok(_) => {
            panic!(
                "Should FAIL when classes are missing - this indicates target generation failure!"
            );
        }
    }
}
