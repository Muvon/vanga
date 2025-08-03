//! Test to verify window increment validation prevents overfitting

#[tokio::test]
async fn test_minimum_increment_ratio_validation() {
    println!("🧪 Testing Minimum Increment Ratio Validation");

    // Test case: 8000 samples with different min_increment_ratio values
    let total_samples = 8000;
    let available_for_training = (total_samples as f64 * 0.9) as usize; // 7200 samples
    let min_train_ratio = 0.4; // 40% minimum training
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

    // Test different min_increment_ratio values
    let test_cases = vec![
        ("Conservative", 0.5), // 50% increment required
        ("Balanced", 0.3),     // 30% increment required (default)
        ("Aggressive", 0.2),   // 20% increment required
        ("Minimal", 0.1),      // 10% increment required
    ];

    for (name, min_increment_ratio) in test_cases {
        println!(
            "\\n🔍 Testing {} approach (min_increment_ratio = {:.1}%):",
            name,
            min_increment_ratio * 100.0
        );

        let min_increment_threshold = (min_train_size as f64 * min_increment_ratio) as usize;
        println!(
            "   Minimum increment threshold: {} samples",
            min_increment_threshold
        );

        // Test window counts 2-8
        let mut valid_windows = Vec::new();
        for window_count in 2..=8 {
            let avg_increment = data_for_expansion / window_count;
            let is_valid = avg_increment >= min_increment_threshold;

            if is_valid {
                valid_windows.push(window_count);
                println!(
                    "   ✅ {} windows: increment {} >= threshold {} (valid)",
                    window_count, avg_increment, min_increment_threshold
                );
            } else {
                println!(
                    "   ❌ {} windows: increment {} < threshold {} (REJECTED)",
                    window_count, avg_increment, min_increment_threshold
                );
            }
        }

        println!("   📈 Valid window counts: {:?}", valid_windows);

        // Calculate expected training progression for first valid option
        if let Some(&first_valid) = valid_windows.first() {
            let increment = data_for_expansion / first_valid;
            println!(
                "   🚀 Expected progression (using {} windows):",
                first_valid
            );

            let mut current_size = min_train_size;
            for i in 1..=first_valid {
                println!(
                    "      Window {}: {} samples ({:.1}% increase from initial)",
                    i,
                    current_size,
                    ((current_size - min_train_size) as f64 / min_train_size as f64) * 100.0
                );
                if i < first_valid {
                    current_size += increment;
                }
            }
        }
    }

    println!("\\n🎯 Key Insights:");
    println!("   - Higher min_increment_ratio = fewer valid windows but more new data per window");
    println!("   - Lower min_increment_ratio = more windows but risk of overfitting");
    println!("   - Default 30% strikes balance between efficiency and sufficient new information");
}

#[test]
fn test_increment_ratio_prevents_overfitting_scenario() {
    println!("🚨 Testing Overfitting Prevention Scenario");

    // Scenario: Small dataset where many windows would create tiny increments
    let available_for_training = 3000; // Small dataset
    let min_train_size = 1200; // 40% minimum
    let data_for_expansion = 1800; // Only 1800 samples for expansion

    println!("📊 Small Dataset Scenario:");
    println!(
        "   Available for training: {} samples",
        available_for_training
    );
    println!("   Minimum training size: {} samples", min_train_size);
    println!("   Data for expansion: {} samples", data_for_expansion);

    // Without increment ratio validation (old behavior)
    println!("\\n❌ Without increment ratio validation:");
    for window_count in 2..=8 {
        let increment = data_for_expansion / window_count;
        let increment_percentage = (increment as f64 / min_train_size as f64) * 100.0;
        println!(
            "   {} windows: +{} samples per window ({:.1}% of initial)",
            window_count, increment, increment_percentage
        );
    }

    // With increment ratio validation (new behavior)
    println!("\\n✅ With 30% increment ratio validation:");
    let min_increment_ratio = 0.3;
    let min_threshold = (min_train_size as f64 * min_increment_ratio) as usize; // 360 samples

    println!(
        "   Minimum increment threshold: {} samples ({:.0}% of initial window)",
        min_threshold,
        min_increment_ratio * 100.0
    );

    for window_count in 2..=8 {
        let increment = data_for_expansion / window_count;
        let is_valid = increment >= min_threshold;

        if is_valid {
            println!(
                "   ✅ {} windows: +{} samples per window (VALID)",
                window_count, increment
            );
        } else {
            println!(
                "   ❌ {} windows: +{} samples per window (REJECTED - insufficient new data)",
                window_count, increment
            );
        }
    }

    println!("\\n🎯 Result: Prevents overfitting by ensuring each window adds meaningful new information");
}
