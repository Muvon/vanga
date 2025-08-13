//! Tests for the diagnostics module
//!
//! This module tests the comprehensive diagnostic functions for training pipeline components.

use crate::config::training::OptimizerType;
use crate::utils::diagnostics::TrainingDiagnostics;

#[test]
fn test_optimizer_name_extraction() {
    let adamw = OptimizerType::AdamW {
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };
    assert_eq!(TrainingDiagnostics::get_optimizer_name(&adamw), "AdamW");

    let sgd = OptimizerType::SGD {
        momentum: Some(0.9),
    };
    assert_eq!(TrainingDiagnostics::get_optimizer_name(&sgd), "SGD");

    let adam = OptimizerType::Adam {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: Some(0.01),
        amsgrad: false,
    };
    assert_eq!(TrainingDiagnostics::get_optimizer_name(&adam), "Adam");
}

#[test]
fn test_weight_decay_support() {
    let adamw = OptimizerType::AdamW {
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };
    assert!(TrainingDiagnostics::supports_weight_decay(&adamw));

    let sgd = OptimizerType::SGD {
        momentum: Some(0.9),
    };
    assert!(!TrainingDiagnostics::supports_weight_decay(&sgd));

    let rmsprop = OptimizerType::RMSprop {
        alpha: 0.99,
        eps: 1e-8,
        weight_decay: Some(0.01),
        momentum: 0.0,
        centered: false,
    };
    assert!(TrainingDiagnostics::supports_weight_decay(&rmsprop));
}

#[test]
fn test_weight_decay_extraction() {
    let adamw = OptimizerType::AdamW {
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };
    assert_eq!(TrainingDiagnostics::get_weight_decay(&adamw), Some(0.01));

    let sgd = OptimizerType::SGD {
        momentum: Some(0.9),
    };
    assert_eq!(TrainingDiagnostics::get_weight_decay(&sgd), None);

    let adam_with_wd = OptimizerType::Adam {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: Some(0.005),
        amsgrad: false,
    };
    assert_eq!(
        TrainingDiagnostics::get_weight_decay(&adam_with_wd),
        Some(0.005)
    );

    let adam_without_wd = OptimizerType::Adam {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: None,
        amsgrad: false,
    };
    assert_eq!(
        TrainingDiagnostics::get_weight_decay(&adam_without_wd),
        None
    );
}

#[test]
fn test_all_optimizer_types_coverage() {
    // Test that all optimizer types are properly handled
    let optimizers = vec![
        OptimizerType::AdamW {
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        },
        OptimizerType::SGD {
            momentum: Some(0.9),
        },
        OptimizerType::Adam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: Some(0.01),
            amsgrad: false,
        },
        OptimizerType::AdaDelta {
            rho: 0.95,
            eps: 1e-6,
            weight_decay: Some(0.01),
        },
        OptimizerType::AdaGrad {
            lr_decay: 0.0,
            weight_decay: Some(0.01),
            initial_accumulator_value: 0.0,
            eps: 1e-10,
        },
        OptimizerType::AdaMax {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: Some(0.01),
        },
        OptimizerType::NAdam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: Some(0.01),
            momentum_decay: 0.004,
        },
        OptimizerType::RAdam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: Some(0.01),
        },
        OptimizerType::RMSprop {
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: Some(0.01),
            momentum: 0.0,
            centered: false,
        },
    ];

    let expected_names = vec![
        "AdamW", "SGD", "Adam", "AdaDelta", "AdaGrad", "AdaMax", "NAdam", "RAdam", "RMSprop",
    ];

    for (optimizer, expected_name) in optimizers.iter().zip(expected_names.iter()) {
        assert_eq!(
            TrainingDiagnostics::get_optimizer_name(optimizer),
            *expected_name
        );
    }
}

#[test]
fn test_diagnostics_logging_functions() {
    // Test that the logging functions don't panic
    // Note: These tests don't verify log output, just that functions execute without errors

    let adamw = OptimizerType::AdamW {
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };

    // Test optimizer diagnostics
    TrainingDiagnostics::log_optimizer_config(&adamw, 0.001);

    // Test regularization diagnostics
    TrainingDiagnostics::log_regularization_config(true, Some(0.25));
    TrainingDiagnostics::log_regularization_config(false, None);

    // Test data diagnostics
    TrainingDiagnostics::log_data_config(1000, 200, 32, true);
    TrainingDiagnostics::log_data_config(1000, 0, 32, false);

    // Test capacity assessment
    TrainingDiagnostics::log_capacity_assessment(1000, 60, 50, 100000);
    TrainingDiagnostics::log_capacity_assessment(100, 30, 20, 50000); // Low data density
    TrainingDiagnostics::log_capacity_assessment(0, 60, 50, 100000); // No training data
    TrainingDiagnostics::log_capacity_assessment(1000, 60, 50, 0); // No parameters
}

#[test]
fn test_edge_cases() {
    // Test edge cases for weight decay extraction
    let optimizers_with_optional_wd = vec![
        OptimizerType::Adam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: None,
            amsgrad: false,
        },
        OptimizerType::AdaDelta {
            rho: 0.95,
            eps: 1e-6,
            weight_decay: None,
        },
        OptimizerType::RMSprop {
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: None,
            momentum: 0.0,
            centered: false,
        },
    ];

    for optimizer in optimizers_with_optional_wd {
        assert_eq!(TrainingDiagnostics::get_weight_decay(&optimizer), None);
        assert!(TrainingDiagnostics::supports_weight_decay(&optimizer));
    }
}
