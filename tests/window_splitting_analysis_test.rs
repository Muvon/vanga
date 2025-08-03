//! Test to analyze window splitting algorithm with different sample sizes

#[tokio::test]
async fn test_window_splitting_with_8000_samples() {
    // Test the window splitting algorithm with 8000 samples
    let total_samples = 8000;

    // Simulate the algorithm logic
    let test_split = 0.1; // 10% for test
    let validation_split = 0.2; // 20% for validation

    // STEP 1: Reserve test set
    let test_size = (total_samples as f64 * test_split) as usize; // 800 samples
    let available_for_training = total_samples - test_size; // 7200 samples

    // STEP 2: Calculate validation size from remaining data
    let validation_size = (available_for_training as f64 * validation_split) as usize; // 1440 samples
    let min_train_size = available_for_training / 2; // 3600 samples (50% minimum)

    println!(
        "📊 Sample Distribution Analysis for {} samples:",
        total_samples
    );
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
    println!("   Minimum train size: {} samples", min_train_size);

    // STEP 3: Simulate the optimal window configuration algorithm
    let data_for_expansion = available_for_training - min_train_size; // 3600 samples
    println!("   Data for expansion: {} samples", data_for_expansion);

    // Test different window counts (2-8 windows)
    println!("\n🧠 Window Configuration Analysis:");
    for window_count in 2..=8 {
        let avg_increment = data_for_expansion / window_count;

        // Skip if increment is too small (less than 5% of available data)
        if avg_increment < available_for_training / 20 {
            println!(
                "   {} windows: SKIPPED (increment {} < {})",
                window_count,
                avg_increment,
                available_for_training / 20
            );
            continue;
        }

        let mut train_end = min_train_size;
        let mut total_used = min_train_size;
        let mut valid_windows = 0;

        for i in 0..window_count {
            let remaining_after_train = available_for_training - train_end;

            if remaining_after_train == 0 {
                break;
            }

            let window_validation_size = if i == window_count - 1 {
                remaining_after_train // Final window uses all remaining
            } else {
                let remaining_windows = window_count - i;
                let avg_val_per_remaining = remaining_after_train / remaining_windows;
                std::cmp::min(avg_val_per_remaining, validation_size * 2)
            };

            if train_end + window_validation_size > available_for_training {
                break;
            }

            total_used = train_end + window_validation_size;
            valid_windows += 1;

            if i < window_count - 1 {
                train_end += avg_increment;
            }
        }

        let utilization = (total_used as f64 / available_for_training as f64) * 100.0;

        // Score function from the actual algorithm
        let window_quality_score = if window_count <= 4 {
            window_count as f64
        } else {
            4.0 + (window_count as f64 - 4.0) * 0.5
        };

        let score = utilization * window_quality_score / 100.0;

        println!(
            "   {} windows: increment={}, utilization={:.1}%, score={:.2}, valid_windows={}",
            window_count, avg_increment, utilization, score, valid_windows
        );
    }

    // Test the actual algorithm behavior
    println!("\n🎯 Expected Optimal Configuration:");
    println!("   Based on the algorithm, with 8000 samples:");
    println!("   - Test reserved: 800 samples");
    println!("   - Available for training: 7200 samples");
    println!("   - Minimum initial training: 3600 samples");
    println!("   - Data for expansion: 3600 samples");
    println!("   - Likely optimal: 4-5 windows (good balance of validation and efficiency)");
    println!("   - Each window increment: ~720-900 samples");

    // Verify the minimum increment threshold
    let min_increment_threshold = available_for_training / 20; // 360 samples
    println!(
        "   - Minimum increment threshold: {} samples",
        min_increment_threshold
    );
    println!("   - All window counts 2-8 should be valid (increments > threshold)");
}

#[tokio::test]
async fn test_window_splitting_edge_cases() {
    println!("🧪 Testing Edge Cases:");

    // Test small dataset
    let small_samples = 1000;
    let test_split = 0.1;
    let validation_split = 0.2;

    let test_size = (small_samples as f64 * test_split) as usize; // 100
    let available = small_samples - test_size; // 900
    let _val_size = (available as f64 * validation_split) as usize; // 180
    let min_train = available / 2; // 450
    let expansion_data = available - min_train; // 450

    println!("\n📊 Small Dataset (1000 samples):");
    println!("   Available for training: {}", available);
    println!("   Min train size: {}", min_train);
    println!("   Data for expansion: {}", expansion_data);
    println!("   Min increment threshold: {}", available / 20); // 45

    for window_count in 2..=8 {
        let avg_increment = expansion_data / window_count;
        let valid = avg_increment >= available / 20;
        println!(
            "   {} windows: increment={}, valid={}",
            window_count, avg_increment, valid
        );
    }

    // Test large dataset
    let large_samples = 50000;
    let test_size_large = (large_samples as f64 * test_split) as usize; // 5000
    let available_large = large_samples - test_size_large; // 45000
    let min_train_large = available_large / 2; // 22500
    let expansion_data_large = available_large - min_train_large; // 22500

    println!("\n📊 Large Dataset (50000 samples):");
    println!("   Available for training: {}", available_large);
    println!("   Min train size: {}", min_train_large);
    println!("   Data for expansion: {}", expansion_data_large);
    println!("   Min increment threshold: {}", available_large / 20); // 2250

    for window_count in 2..=8 {
        let avg_increment = expansion_data_large / window_count;
        let valid = avg_increment >= available_large / 20;
        println!(
            "   {} windows: increment={}, valid={}",
            window_count, avg_increment, valid
        );
    }
}

#[test]
fn test_window_algorithm_scoring() {
    println!("🎯 Window Algorithm Scoring Analysis:");

    // Test the scoring function used in the algorithm
    for window_count in 2..=8 {
        let window_quality_score = if window_count <= 4 {
            window_count as f64
        } else {
            4.0 + (window_count as f64 - 4.0) * 0.5 // Diminishing returns
        };

        // Test with different utilization rates
        for utilization in [85.0, 90.0, 95.0, 98.0] {
            let score = utilization * window_quality_score / 100.0;
            println!(
                "   {} windows, {:.1}% util: quality_score={:.1}, final_score={:.2}",
                window_count, utilization, window_quality_score, score
            );
        }
    }

    println!("\n📈 Scoring Insights:");
    println!("   - 2-4 windows: Linear quality score (2.0, 3.0, 4.0)");
    println!("   - 5+ windows: Diminishing returns (4.5, 5.0, 5.5, 6.0)");
    println!("   - Higher utilization always better");
    println!("   - Sweet spot likely 4-5 windows for most datasets");
}
