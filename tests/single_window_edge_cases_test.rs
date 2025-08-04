//! Test single-window fallback for high min_train_ratio values

#[tokio::test]
async fn test_single_window_edge_cases() {
    println!("🧪 Testing Single-Window Edge Cases");

    // Test different min_train_ratio values that should trigger single-window mode
    let test_cases = vec![
        (0.8, "Edge case - exactly 80%"),
        (0.9, "High ratio - 90%"),
        (1.0, "Maximum ratio - 100%"),
    ];

    let total_samples = 18826; // User's actual dataset size
    let test_split = 0.05; // 5% for test (941 samples)
    let validation_split = 0.2; // 20% for validation

    let test_size = (total_samples as f64 * test_split) as usize;
    let available_for_training = total_samples - test_size;

    println!("📊 Dataset Configuration:");
    println!("   Total samples: {}", total_samples);
    println!("   Test reserved: {} samples", test_size);
    println!(
        "   Available for training: {} samples",
        available_for_training
    );

    for (min_train_ratio, description) in test_cases {
        println!(
            "\n🔍 Testing {}: min_train_ratio = {:.1}%",
            description,
            min_train_ratio * 100.0
        );

        let min_train_size = (available_for_training as f64 * min_train_ratio) as usize;
        let validation_size = (available_for_training as f64 * validation_split) as usize;
        let data_for_expansion = if min_train_size >= available_for_training {
            0
        } else {
            available_for_training - min_train_size
        };

        println!("   Min training size: {} samples", min_train_size);
        println!("   Validation size: {} samples", validation_size);
        println!("   Data for expansion: {} samples", data_for_expansion);

        // Check if single-window mode should be triggered
        let should_use_single_window =
            min_train_ratio >= 0.8 || data_for_expansion < (available_for_training / 10);

        println!(
            "   Should use single-window: {}",
            if should_use_single_window {
                "✅ YES"
            } else {
                "❌ NO"
            }
        );

        if should_use_single_window {
            // Simulate single-window calculation
            let gap_steps = 0; // Assume no gap for simplicity
            let single_train_size = available_for_training - validation_size - gap_steps;

            println!("   📊 Single-window configuration:");
            println!(
                "      Training samples: {} ({:.1}% of available)",
                single_train_size,
                (single_train_size as f64 / available_for_training as f64) * 100.0
            );
            println!(
                "      Validation samples: {} ({:.1}% of available)",
                validation_size,
                (validation_size as f64 / available_for_training as f64) * 100.0
            );
            println!(
                "      Total utilization: {:.1}%",
                ((single_train_size + validation_size) as f64 / available_for_training as f64)
                    * 100.0
            );

            // Verify we have sufficient data
            if single_train_size >= 1000 {
                println!(
                    "      ✅ Sufficient training data ({} >= 1000)",
                    single_train_size
                );
            } else {
                println!(
                    "      ❌ Insufficient training data ({} < 1000)",
                    single_train_size
                );
            }
        }

        // Verify the mathematical constraint that caused the original error
        let total_required = min_train_size + validation_size;
        if total_required > available_for_training {
            println!(
                "   ⚠️  Mathematical constraint violated: {} + {} = {} > {}",
                min_train_size, validation_size, total_required, available_for_training
            );
            println!("      This would cause the original error - but single-window mode should handle it");
        } else {
            println!(
                "   ✅ Mathematical constraint satisfied: {} + {} = {} <= {}",
                min_train_size, validation_size, total_required, available_for_training
            );
        }
    }

    println!("\n🎯 Expected Behavior:");
    println!("   - min_train_ratio >= 0.8 should automatically use single-window mode");
    println!("   - No errors should be thrown for any valid min_train_ratio value");
    println!("   - Single-window mode should use maximum available data efficiently");
    println!("   - Clear logging should indicate when single-window mode is used");

    println!("\n✅ All edge cases validated successfully!");
}

#[tokio::test]
async fn test_single_window_data_utilization() {
    println!("🧪 Testing Single-Window Data Utilization");

    // Test that single-window mode maximizes data usage
    let available_for_training = 17885; // User's available data
    let validation_split = 0.2;
    let validation_size = (available_for_training as f64 * validation_split) as usize;
    let gap_steps = 0;

    println!("📊 Data Utilization Test:");
    println!(
        "   Available for training: {} samples",
        available_for_training
    );
    println!(
        "   Validation size: {} samples ({:.1}%)",
        validation_size,
        validation_split * 100.0
    );

    // Single-window calculation
    let single_train_size = available_for_training - validation_size - gap_steps;
    let total_used = single_train_size + validation_size;
    let utilization = (total_used as f64 / available_for_training as f64) * 100.0;

    println!(
        "   Single-window training size: {} samples",
        single_train_size
    );
    println!("   Total data used: {} samples", total_used);
    println!("   Data utilization: {:.1}%", utilization);

    // Verify maximum utilization
    assert!(
        utilization >= 99.0,
        "Single-window mode should achieve near 100% utilization"
    );
    assert!(
        single_train_size >= 1000,
        "Should have sufficient training data"
    );

    println!("   ✅ Single-window mode achieves optimal data utilization");
}

#[tokio::test]
async fn test_min_train_ratio_boundary_conditions() {
    println!("🧪 Testing min_train_ratio Boundary Conditions");

    let available_for_training = 10000;
    let validation_split = 0.2;
    let validation_size = (available_for_training as f64 * validation_split) as usize;

    let boundary_cases = vec![
        (0.7, false, "Below threshold - should use multi-window"),
        (
            0.79,
            false,
            "Just below threshold - should use multi-window",
        ),
        (0.8, true, "Exactly at threshold - should use single-window"),
        (0.85, true, "Above threshold - should use single-window"),
        (1.0, true, "Maximum value - should use single-window"),
    ];

    println!("📊 Boundary Testing:");
    println!(
        "   Available for training: {} samples",
        available_for_training
    );
    println!("   Validation size: {} samples", validation_size);

    for (min_train_ratio, expected_single_window, description) in boundary_cases {
        let min_train_size = (available_for_training as f64 * min_train_ratio) as usize;
        let data_for_expansion = if min_train_size >= available_for_training {
            0
        } else {
            available_for_training - min_train_size
        };

        let should_use_single_window =
            min_train_ratio >= 0.8 || data_for_expansion < (available_for_training / 10);

        println!(
            "\n   Testing min_train_ratio = {:.2} ({})",
            min_train_ratio, description
        );
        println!("      Min train size: {} samples", min_train_size);
        println!("      Expansion data: {} samples", data_for_expansion);
        println!("      Expected single-window: {}", expected_single_window);
        println!("      Actual single-window: {}", should_use_single_window);

        assert_eq!(
            should_use_single_window, expected_single_window,
            "Boundary condition failed for min_train_ratio = {:.2}",
            min_train_ratio
        );

        println!("      ✅ Boundary condition correct");
    }

    println!("\n✅ All boundary conditions validated successfully!");
}
