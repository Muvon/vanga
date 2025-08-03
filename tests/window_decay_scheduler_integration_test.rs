//! Integration test for window decay with different learning rate schedulers
//!
//! This test verifies that window decay is properly applied to all scheduler parameters,
//! not just the base learning rate.

use vanga::config::training::{LearningScheduleConfig, TrainingConfig};
use vanga::model::lstm::{create_window_aware_config, WindowAwareLearningRate};

#[test]
fn test_onecycle_window_decay_integration() {
    // Create a config with OneCycle scheduler
    let mut config = TrainingConfig::default();
    config.training.learning_rate = 0.001; // Base learning rate
    config.training.window_decay = 0.8; // 20% decay per window
    config.training.learning_schedule = Some(LearningScheduleConfig::OneCycle {
        max_lr: 0.01, // This should be scaled by window decay
        pct_start: Some(0.3),
        anneal_strategy: Some("cos".to_string()),
        div_factor: Some(25.0),
        final_div_factor: Some(1e4),
    });

    // Test window 0 (no decay)
    let window_0_config = create_window_aware_config(&config, 0).unwrap();
    if let Some(LearningScheduleConfig::OneCycle { max_lr, .. }) =
        &window_0_config.training.learning_schedule
    {
        assert!(
            (max_lr - 0.01).abs() < 1e-6,
            "Window 0 should have original max_lr"
        );
    } else {
        panic!("Expected OneCycle schedule");
    }

    // Test window 1 (first decay: 0.8)
    let window_1_config = create_window_aware_config(&config, 1).unwrap();
    if let Some(LearningScheduleConfig::OneCycle { max_lr, .. }) =
        &window_1_config.training.learning_schedule
    {
        let expected_max_lr = 0.01 * 0.8;
        assert!(
            (max_lr - expected_max_lr).abs() < 1e-6,
            "Window 1 max_lr should be scaled: expected {}, got {}",
            expected_max_lr,
            max_lr
        );
    } else {
        panic!("Expected OneCycle schedule");
    }

    // Test window 2 (second decay: 0.8^2 = 0.64)
    let window_2_config = create_window_aware_config(&config, 2).unwrap();
    if let Some(LearningScheduleConfig::OneCycle { max_lr, .. }) =
        &window_2_config.training.learning_schedule
    {
        let expected_max_lr = 0.01 * 0.8 * 0.8;
        assert!(
            (max_lr - expected_max_lr).abs() < 1e-6,
            "Window 2 max_lr should be scaled: expected {}, got {}",
            expected_max_lr,
            max_lr
        );
    } else {
        panic!("Expected OneCycle schedule");
    }

    println!("✅ OneCycle window decay integration test passed!");
}

#[test]
fn test_cyclical_lr_window_decay_integration() {
    // Create a config with CyclicalLR scheduler
    let mut config = TrainingConfig::default();
    config.training.learning_rate = 0.001; // Base learning rate
    config.training.window_decay = 0.9; // 10% decay per window
    config.training.learning_schedule = Some(LearningScheduleConfig::CyclicalLR {
        base_lr: 1e-5, // This should be scaled by window decay
        max_lr: 1e-3,  // This should be scaled by window decay
        step_size_up: 20,
        step_size_down: Some(20),
        mode: Some("triangular".to_string()),
        gamma: Some(1.0),
    });

    // Test window 0 (no decay)
    let window_0_config = create_window_aware_config(&config, 0).unwrap();
    if let Some(LearningScheduleConfig::CyclicalLR {
        base_lr, max_lr, ..
    }) = &window_0_config.training.learning_schedule
    {
        assert!(
            (base_lr - 1e-5).abs() < 1e-8,
            "Window 0 should have original base_lr"
        );
        assert!(
            (max_lr - 1e-3).abs() < 1e-6,
            "Window 0 should have original max_lr"
        );
    } else {
        panic!("Expected CyclicalLR schedule");
    }

    // Test window 1 (first decay: 0.9)
    let window_1_config = create_window_aware_config(&config, 1).unwrap();
    if let Some(LearningScheduleConfig::CyclicalLR {
        base_lr, max_lr, ..
    }) = &window_1_config.training.learning_schedule
    {
        let expected_base_lr = 1e-5 * 0.9;
        let expected_max_lr = 1e-3 * 0.9;
        assert!(
            (base_lr - expected_base_lr).abs() < 1e-8,
            "Window 1 base_lr should be scaled: expected {}, got {}",
            expected_base_lr,
            base_lr
        );
        assert!(
            (max_lr - expected_max_lr).abs() < 1e-6,
            "Window 1 max_lr should be scaled: expected {}, got {}",
            expected_max_lr,
            max_lr
        );
    } else {
        panic!("Expected CyclicalLR schedule");
    }

    println!("✅ CyclicalLR window decay integration test passed!");
}

#[test]
fn test_cosine_annealing_window_decay_integration() {
    // Create a config with CosineAnnealing scheduler
    let mut config = TrainingConfig::default();
    config.training.learning_rate = 0.001; // This should be scaled by window decay
    config.training.window_decay = 0.7; // 30% decay per window
    config.training.learning_schedule = Some(LearningScheduleConfig::CosineAnnealing {
        t_max: 100,
        eta_min: Some(1e-6), // This should be scaled by window decay
    });

    // Test window 0 (no decay)
    let window_0_config = create_window_aware_config(&config, 0).unwrap();
    assert!(
        (window_0_config.training.learning_rate - 0.001).abs() < 1e-6,
        "Window 0 should have original learning_rate"
    );
    if let Some(LearningScheduleConfig::CosineAnnealing { eta_min, .. }) =
        &window_0_config.training.learning_schedule
    {
        assert!(
            (eta_min.unwrap() - 1e-6).abs() < 1e-9,
            "Window 0 should have original eta_min"
        );
    } else {
        panic!("Expected CosineAnnealing schedule");
    }

    // Test window 3 (third decay: 0.7^3 = 0.343)
    let window_3_config = create_window_aware_config(&config, 3).unwrap();
    let decay_factor = 0.7_f64.powi(3);
    let expected_lr = 0.001 * decay_factor;
    assert!(
        (window_3_config.training.learning_rate - expected_lr).abs() < 1e-6,
        "Window 3 learning_rate should be scaled: expected {}, got {}",
        expected_lr,
        window_3_config.training.learning_rate
    );

    if let Some(LearningScheduleConfig::CosineAnnealing { eta_min, .. }) =
        &window_3_config.training.learning_schedule
    {
        let expected_eta_min = 1e-6 * decay_factor;
        assert!(
            (eta_min.unwrap() - expected_eta_min).abs() < 1e-9,
            "Window 3 eta_min should be scaled: expected {}, got {}",
            expected_eta_min,
            eta_min.unwrap()
        );
    } else {
        panic!("Expected CosineAnnealing schedule");
    }

    println!("✅ CosineAnnealing window decay integration test passed!");
}

#[test]
fn test_constant_schedule_window_decay_integration() {
    // Create a config with Constant scheduler (should only affect base learning rate)
    let mut config = TrainingConfig::default();
    config.training.learning_rate = 0.005; // This should be scaled by window decay
    config.training.window_decay = 0.8; // 20% decay per window
    config.training.learning_schedule = Some(LearningScheduleConfig::Constant);

    // Test window 0 (no decay)
    let window_0_config = create_window_aware_config(&config, 0).unwrap();
    assert!(
        (window_0_config.training.learning_rate - 0.005).abs() < 1e-6,
        "Window 0 should have original learning_rate"
    );

    // Test window 2 (second decay: 0.8^2 = 0.64)
    let window_2_config = create_window_aware_config(&config, 2).unwrap();
    let expected_lr = 0.005 * 0.8 * 0.8;
    assert!(
        (window_2_config.training.learning_rate - expected_lr).abs() < 1e-6,
        "Window 2 learning_rate should be scaled: expected {}, got {}",
        expected_lr,
        window_2_config.training.learning_rate
    );

    println!("✅ Constant schedule window decay integration test passed!");
}

#[test]
fn test_window_aware_learning_rate_description() {
    // Test the description functionality
    let original_schedule = LearningScheduleConfig::OneCycle {
        max_lr: 0.01,
        pct_start: Some(0.3),
        anneal_strategy: Some("cos".to_string()),
        div_factor: Some(25.0),
        final_div_factor: Some(1e4),
    };

    let window_aware = WindowAwareLearningRate::new(original_schedule, 0.8, 2);
    let description = window_aware.get_decay_description();

    // Should contain information about the decay
    assert!(description.contains("OneCycle"));
    assert!(description.contains("64.0%")); // 0.8^2 * 100 = 64%
    assert!(description.contains("max_lr"));

    println!("✅ Window decay description test passed!");
    println!("   Description: {}", description);
}

#[test]
fn test_window_decay_validation_warnings() {
    // Test validation warnings for aggressive decay
    let original_schedule = LearningScheduleConfig::OneCycle {
        max_lr: 1e-4, // Very small max_lr
        pct_start: Some(0.3),
        anneal_strategy: Some("cos".to_string()),
        div_factor: Some(25.0),
        final_div_factor: Some(1e4),
    };

    let window_aware = WindowAwareLearningRate::new(original_schedule, 0.5, 5); // Aggressive decay
    let warnings = window_aware.validate_window_decay_compatibility().unwrap();

    assert!(
        !warnings.is_empty(),
        "Should generate warnings for aggressive decay"
    );
    assert!(warnings
        .iter()
        .any(|w| w.contains("max_lr becomes very small")));

    println!("✅ Window decay validation test passed!");
    println!("   Generated {} warnings", warnings.len());
    for warning in warnings {
        println!("   Warning: {}", warning);
    }
}
