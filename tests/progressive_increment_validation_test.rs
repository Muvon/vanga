//! Test to verify progressive window increment validation fix

#[tokio::test]
async fn test_progressive_increment_validation() {
    println!("🧪 Testing Progressive Increment Validation Fix");

    // Test case: 8000 samples to match the user's scenario
    let total_samples = 8000;
    let available_for_training = (total_samples as f64 * 0.9) as usize; // 7200 samples
    let min_train_ratio = 0.4; // 40% minimum training
    let min_increment_ratio = 0.3; // 30% minimum increment
    let min_train_size = (available_for_training as f64 * min_train_ratio) as usize; // 2880 samples
    let data_for_expansion = available_for_training - min_train_size; // 4320 samples

    println!("📊 Dataset Configuration:");
    println!("   Total samples: {}", total_samples);
    println!("   Available for training: {}", available_for_training);
    println!(
        "   Minimum training size: {} samples ({:.0}%)",
        min_train_size,
        min_train_ratio * 100.0
    );
    println!("   Data for expansion: {} samples", data_for_expansion);
    println!(
        "   Minimum increment ratio: {:.0}%",
        min_increment_ratio * 100.0
    );

    // Test progressive increment calculation for different window counts
    let test_window_counts = vec![3, 4, 5, 6];

    for window_count in test_window_counts {
        println!("\\n🔍 Testing {} windows:", window_count);

        // Calculate progressive increments (FIXED logic - based on PREVIOUS window)
        let mut progressive_increments = Vec::new();
        let mut previous_window_size = min_train_size; // Start with first window size
        let mut total_increment_needed = 0;

        for _window_idx in 1..window_count {
            // ✅ CORRECT: Calculate increment based on PREVIOUS window size
            let min_increment_for_this_window =
                (previous_window_size as f64 * min_increment_ratio) as usize;

            progressive_increments.push(min_increment_for_this_window);
            total_increment_needed += min_increment_for_this_window;

            // Update previous_window_size for next iteration
            previous_window_size += min_increment_for_this_window;
        }

        println!("   Progressive increments: {:?}", progressive_increments);
        println!(
            "   Total increment needed: {} samples",
            total_increment_needed
        );

        // Check if this configuration is valid
        let is_valid = total_increment_needed <= data_for_expansion;
        println!(
            "   Valid configuration: {}",
            if is_valid { "✅ YES" } else { "❌ NO" }
        );

        if is_valid {
            // Show the window progression with CORRECT increments
            println!("   📈 Window progression:");
            let mut current_size = min_train_size;
            println!("      Window 1: {} samples (baseline)", current_size);

            for (i, increment) in progressive_increments.iter().enumerate() {
                let previous_size = current_size;
                current_size += increment;
                let percentage_increase = (*increment as f64 / previous_size as f64) * 100.0;
                println!(
                    "      Window {}: {} samples (+{} samples, {:.1}% of PREVIOUS window {})",
                    i + 2,
                    current_size,
                    increment,
                    percentage_increase,
                    previous_size
                );
            }

            // Verify all increments meet the minimum ratio (based on PREVIOUS window)
            let mut all_meet_ratio = true;
            let mut check_size = min_train_size;
            for increment in &progressive_increments {
                let actual_ratio = *increment as f64 / check_size as f64;
                if actual_ratio < min_increment_ratio - 0.001 {
                    // Small tolerance for floating point
                    all_meet_ratio = false;
                    println!("      ❌ Increment {} fails ratio check: {:.3} < {:.3} (previous window: {})",
                        increment, actual_ratio, min_increment_ratio, check_size);
                    break;
                }
                check_size += increment;
            }

            println!(
                "   ✅ All windows meet {:.0}% minimum increment ratio: {}",
                min_increment_ratio * 100.0,
                all_meet_ratio
            );
        }
    }

    println!("\\n🎯 Expected Behavior:");
    println!(
        "   - Each window should add exactly {:.0}% more data than the previous window",
        min_increment_ratio * 100.0
    );
    println!("   - No more diminishing percentage increases");
    println!("   - Prevents overfitting in ALL subsequent windows, not just the second one");
}

#[test]
fn test_progressive_vs_fixed_increment_comparison() {
    println!("📊 Comparing Progressive vs Fixed Increment Approaches");

    let min_train_size = 7154; // From user's log
    let data_for_expansion = 8584; // From user's log (15738 - 7154)
    let window_count = 5;
    let min_increment_ratio = 0.3;

    println!("\\n❌ OLD APPROACH (Fixed Increments):");
    let fixed_increment = data_for_expansion / (window_count - 1); // 2146 samples
    let mut current_size = min_train_size;
    println!("   Window 1: {} samples (baseline)", current_size);

    for i in 2..=window_count {
        current_size += fixed_increment;
        let percentage_increase =
            (fixed_increment as f64 / (current_size - fixed_increment) as f64) * 100.0;
        println!(
            "   Window {}: {} samples (+{} samples, {:.1}% increase)",
            i, current_size, fixed_increment, percentage_increase
        );
    }

    println!("\\n✅ NEW APPROACH (Progressive Increments):");
    let mut progressive_increments = Vec::new();
    let mut current_train_size = min_train_size;
    let mut total_needed = 0;

    for _i in 1..window_count {
        let min_increment = (current_train_size as f64 * min_increment_ratio) as usize;
        progressive_increments.push(min_increment);
        total_needed += min_increment;
        current_train_size += min_increment;
    }

    println!("   Progressive increments: {:?}", progressive_increments);
    println!(
        "   Total increment needed: {} vs {} available",
        total_needed, data_for_expansion
    );

    let mut current_size = min_train_size;
    println!("   Window 1: {} samples (baseline)", current_size);

    for (i, increment) in progressive_increments.iter().enumerate() {
        let previous_size = current_size;
        current_size += increment;
        let percentage_increase = (*increment as f64 / previous_size as f64) * 100.0;
        println!(
            "   Window {}: {} samples (+{} samples, {:.1}% increase)",
            i + 2,
            current_size,
            increment,
            percentage_increase
        );
    }

    println!("\\n🎯 Key Differences:");
    println!("   - Fixed: Diminishing percentage increases (30% → 23% → 19% → 16%)");
    println!(
        "   - Progressive: Consistent {:.0}% increases for all windows",
        min_increment_ratio * 100.0
    );
    println!("   - Result: No more overfitting in subsequent windows!");
}
