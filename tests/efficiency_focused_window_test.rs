//! Test to verify efficiency-focused window splitting algorithm

#[tokio::test]
async fn test_window_splitting_8000_samples() {
    println!("🚀 Testing Window Splitting with 8000 samples");

    let total_samples = 8000;
    let test_split = 0.1; // 10% for test
    let validation_split = 0.2; // 20% for validation

    // Efficiency-focused minimum training ratio (40%)
    let min_train_ratio = 0.4;

    // STEP 1: Reserve test set
    let test_size = (total_samples as f64 * test_split) as usize; // 800 samples
    let available_for_training = total_samples - test_size; // 7200 samples

    // STEP 2: Calculate validation size and minimum training size
    let validation_size = (available_for_training as f64 * validation_split) as usize; // 1440 samples
    let min_train_size = (available_for_training as f64 * min_train_ratio) as usize; // 2880 samples (40%)

    println!("📊 Sample Distribution:");
    println!(
        "   Test reserved: {} samples ({:.1}%)",
        test_size,
        test_split * 100.0
    );
    println!(
        "   Available for training: {} samples",
        available_for_training
    );
    println!(
        "   Validation size: {} samples ({:.1}%)",
        validation_size,
        validation_split * 100.0
    );
    println!(
        "   Minimum train size: {} samples ({:.1}%)",
        min_train_size,
        min_train_ratio * 100.0
    );

    let data_for_expansion = available_for_training - min_train_size;
    println!("   Data for expansion: {} samples", data_for_expansion);

    // Test algorithm
    let max_reasonable_windows = std::cmp::min(6, available_for_training / 1000);
    println!("\n🚀 Algorithm (max {} windows):", max_reasonable_windows);

    let mut best_score = 0.0;
    let mut best_config = None;

    for window_count in 2..=max_reasonable_windows {
        let avg_increment = data_for_expansion / window_count;

        // Skip if increment is too small
        if avg_increment < available_for_training / 20 {
            println!(
                "   {} windows: SKIPPED (increment {} < {})",
                window_count,
                avg_increment,
                available_for_training / 20
            );
            continue;
        }

        // Simulate window creation
        let mut train_end = min_train_size;
        let mut total_used = min_train_size;

        for i in 0..window_count {
            let remaining_after_train = available_for_training - train_end;

            if remaining_after_train == 0 {
                break;
            }

            let window_validation_size = if i == window_count - 1 {
                remaining_after_train
            } else {
                let remaining_windows = window_count - i;
                let avg_val_per_remaining = remaining_after_train / remaining_windows;
                std::cmp::min(avg_val_per_remaining, validation_size * 2)
            };

            if train_end + window_validation_size > available_for_training {
                break;
            }

            total_used = train_end + window_validation_size;

            if i < window_count - 1 {
                train_end += avg_increment;
            }
        }

        let utilization = (total_used as f64 / available_for_training as f64) * 100.0;

        // NEW: Efficiency-focused scoring function
        let window_quality_score = if window_count <= 3 {
            window_count as f64 // Linear for 2-3 windows
        } else if window_count <= 5 {
            3.0 + (window_count as f64 - 3.0) * 0.7 // Moderate returns for 4-5 windows
        } else {
            4.4 + (window_count as f64 - 5.0) * 0.3 // Strong diminishing returns for 6+ windows
        };

        // Efficiency bonus: favor 4-5 windows (sweet spot)
        let efficiency_bonus = match window_count {
            4 | 5 => 0.3, // Sweet spot bonus
            3 | 6 => 0.1, // Slight bonus for reasonable choices
            _ => 0.0,     // No bonus for extreme choices
        };

        // Training time penalty for excessive windows
        let time_penalty = if window_count > 5 {
            (window_count as f64 - 5.0) * 0.2
        } else {
            0.0
        };

        // Data utilization bonus
        let utilization_bonus = if utilization > 95.0 {
            0.2
        } else if utilization > 90.0 {
            0.1
        } else {
            0.0
        };

        // Final efficiency-focused score
        let score =
            (utilization * window_quality_score / 100.0) + efficiency_bonus + utilization_bonus
                - time_penalty;

        println!("   {} windows: increment={}, util={:.1}%, quality={:.2}, efficiency_bonus={:.2}, time_penalty={:.2}, score={:.3}",
            window_count, avg_increment, utilization, window_quality_score, efficiency_bonus, time_penalty, score);

        if score > best_score {
            best_score = score;
            best_config = Some((window_count, avg_increment, utilization));
        }
    }

    if let Some((best_windows, best_increment, best_util)) = best_config {
        println!("\n🎯 OPTIMAL RESULT:");
        println!("   Best configuration: {} windows", best_windows);
        println!("   Increment per window: {} samples", best_increment);
        println!("   Data utilization: {:.1}%", best_util);
        println!("   Final score: {:.3}", best_score);
    }
}

#[tokio::test]
async fn test_efficiency_focused_algorithm_different_sizes() {
    println!("🧪 Testing Efficiency-Focused Algorithm Across Different Dataset Sizes");

    let test_cases = vec![
        ("Small", 2000),
        ("Medium", 8000),
        ("Large", 20000),
        ("Very Large", 50000),
    ];

    for (name, total_samples) in test_cases {
        println!("\\n📊 {} Dataset ({} samples):", name, total_samples);

        let available_for_training = (total_samples as f64 * 0.9) as usize; // 90% after test split

        // Current algorithm (40% minimum training ratio)
        let min_train_ratio = 0.4;
        let min_train_size = (available_for_training as f64 * min_train_ratio) as usize;
        let expansion_data = available_for_training - min_train_size;

        // Window constraints
        let max_windows = std::cmp::min(6, available_for_training / 1000);

        println!(
            "   Available for training: {} samples",
            available_for_training
        );
        println!(
            "   Minimum training size: {} samples ({:.0}%)",
            min_train_size,
            min_train_ratio * 100.0
        );
        println!("   Expansion data: {} samples", expansion_data);
        println!("   Maximum windows: {} windows", max_windows);

        // Estimate likely outcome (algorithm favors 4-5 windows)
        let likely_windows = std::cmp::min(5, max_windows);
        let estimated_efficiency = if likely_windows <= 5 {
            "High"
        } else {
            "Medium"
        };

        println!("   Expected outcome: {} windows", likely_windows);
        println!("   Training efficiency: {}", estimated_efficiency);
    }
}

#[test]
fn test_efficiency_focused_scoring_function() {
    println!("🎯 Testing Efficiency-Focused Scoring Function");

    println!("\n📊 Window Quality Scores:");
    for window_count in 2..=8 {
        let quality_score = if window_count <= 3 {
            window_count as f64
        } else if window_count <= 5 {
            3.0 + (window_count as f64 - 3.0) * 0.7
        } else {
            4.4 + (window_count as f64 - 5.0) * 0.3
        };

        let efficiency_bonus = match window_count {
            4 | 5 => 0.3,
            3 | 6 => 0.1,
            _ => 0.0,
        };

        let time_penalty = if window_count > 5 {
            (window_count as f64 - 5.0) * 0.2
        } else {
            0.0
        };

        println!(
            "   {} windows: quality={:.2}, efficiency_bonus={:.1}, time_penalty={:.1}",
            window_count, quality_score, efficiency_bonus, time_penalty
        );
    }

    println!("\n🏆 Expected Sweet Spot Analysis:");
    println!("   Windows 2-3: Linear growth, no bonuses");
    println!("   Windows 4-5: Moderate growth + efficiency bonus (SWEET SPOT)");
    println!("   Windows 6+: Diminishing returns + time penalty");
    println!("   Result: Algorithm should favor 4-5 windows for most datasets");
}
