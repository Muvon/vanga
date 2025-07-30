//! Dropout Implementation Testing
//!
//! This module provides comprehensive tests for dropout behavior
//! in both training and inference modes across LSTM and attention components.

#[cfg(test)]
mod tests {
    use crate::model::dropout_consistency::{
        AttentionDropoutComponent, AttentionDropoutConfig, DropoutConsistencyConfig,
        DropoutConsistencyStrategy,
    };

    /// Test dropout consistency configuration validation
    #[test]
    fn test_dropout_consistency_validation() {
        let config = DropoutConsistencyConfig::default();

        // Test valid configuration
        assert!(config
            .validate_comprehensive_dropout_config(true, Some(0.2), Some(0.1), "TestModel")
            .is_ok());

        // Test invalid LSTM dropout rate
        assert!(config
            .validate_comprehensive_dropout_config(true, Some(1.5), Some(0.1), "TestModel")
            .is_err());

        // Test invalid attention dropout rate
        assert!(config
            .validate_comprehensive_dropout_config(true, Some(0.2), Some(-0.1), "TestModel")
            .is_err());
    }

    /// Test effective dropout rate calculation
    #[test]
    fn test_effective_dropout_rates() {
        let mut config = DropoutConsistencyConfig::default();
        let base_rate = 0.3;

        // Test Standard strategy
        config.strategy = DropoutConsistencyStrategy::Standard;
        assert_eq!(
            config.get_effective_dropout_rate(base_rate, true),
            base_rate
        );
        assert_eq!(config.get_effective_dropout_rate(base_rate, false), 0.0);

        // Test Consistent strategy
        config.strategy = DropoutConsistencyStrategy::Consistent;
        assert_eq!(
            config.get_effective_dropout_rate(base_rate, true),
            base_rate
        );
        assert_eq!(
            config.get_effective_dropout_rate(base_rate, false),
            base_rate
        );

        // Test Disabled strategy
        config.strategy = DropoutConsistencyStrategy::Disabled;
        assert_eq!(config.get_effective_dropout_rate(base_rate, true), 0.0);
        assert_eq!(config.get_effective_dropout_rate(base_rate, false), 0.0);
    }

    /// Test attention dropout component decisions
    #[test]
    fn test_attention_dropout_components() {
        let config = DropoutConsistencyConfig::default();
        let base_rate = 0.2;

        // Test attention weights dropout
        assert!(config.should_apply_attention_dropout(
            AttentionDropoutComponent::AttentionWeights,
            true,
            base_rate
        ));

        // Test attention output dropout
        assert!(config.should_apply_attention_dropout(
            AttentionDropoutComponent::AttentionOutput,
            true,
            base_rate
        ));

        // Test projections dropout
        assert!(config.should_apply_attention_dropout(
            AttentionDropoutComponent::Projections,
            true,
            base_rate
        ));

        // Test inference mode with Standard strategy
        assert!(!config.should_apply_attention_dropout(
            AttentionDropoutComponent::AttentionWeights,
            false,
            base_rate
        ));
    }

    /// Test attention dropout rate scaling
    #[test]
    fn test_attention_dropout_scaling() {
        let mut config = DropoutConsistencyConfig::default();
        config.attention_dropout_config.attention_dropout_scale = 0.5;

        let base_rate = 0.4;
        let expected_attention_rate = base_rate * 0.5;

        assert_eq!(
            config.get_effective_attention_dropout_rate(base_rate, true),
            expected_attention_rate
        );
    }

    /// Test dropout behavior logging (integration test)
    #[test]
    fn test_dropout_logging_behavior() {
        let config = DropoutConsistencyConfig {
            strategy: DropoutConsistencyStrategy::Standard,
            log_dropout_changes: true,
            warn_validation_inconsistency: true,
            attention_dropout_config: AttentionDropoutConfig::default(),
        };

        // This test verifies that logging methods don't panic
        config.log_dropout_behavior(true, true);
        config.log_dropout_behavior(false, false);

        // Test validation logging
        let result = config.validate_comprehensive_dropout_config(
            true,
            Some(0.2),
            Some(0.1),
            "TestComponent",
        );
        assert!(result.is_ok());
    }

    /// Test recommended strategy configuration
    #[test]
    fn test_recommended_strategy() {
        let recommended = DropoutConsistencyConfig::get_recommended_strategy();

        assert_eq!(recommended.strategy, DropoutConsistencyStrategy::Consistent);
        assert!(recommended.log_dropout_changes);
        assert!(!recommended.warn_validation_inconsistency);
    }

    /// Test dropout configuration edge cases
    #[test]
    fn test_dropout_edge_cases() {
        let config = DropoutConsistencyConfig::default();

        // Test zero dropout rate
        assert_eq!(config.get_effective_dropout_rate(0.0, true), 0.0);
        assert_eq!(config.get_effective_dropout_rate(0.0, false), 0.0);

        // Test very small dropout rate
        let small_rate = 0.001;
        assert_eq!(
            config.get_effective_dropout_rate(small_rate, true),
            small_rate
        );

        // Test ValidationOnly strategy
        let mut val_only_config = config.clone();
        val_only_config.strategy = DropoutConsistencyStrategy::ValidationOnly;
        assert_eq!(val_only_config.get_effective_dropout_rate(0.3, true), 0.0);
        assert_eq!(val_only_config.get_effective_dropout_rate(0.3, false), 0.3);
    }

    /// Integration test for dropout behavior consistency
    #[test]
    fn test_dropout_integration_consistency() {
        // Test that dropout behavior is consistent across different components
        let config = DropoutConsistencyConfig {
            strategy: DropoutConsistencyStrategy::Consistent,
            log_dropout_changes: false, // Disable logging for test
            warn_validation_inconsistency: false,
            attention_dropout_config: AttentionDropoutConfig {
                apply_to_attention_weights: true,
                apply_to_attention_output: true,
                apply_to_projections: true,
                attention_dropout_scale: 1.0,
            },
        };

        let base_rate = 0.25;

        // Both training and inference should have same behavior with Consistent strategy
        let training_rate = config.get_effective_dropout_rate(base_rate, true);
        let inference_rate = config.get_effective_dropout_rate(base_rate, false);

        assert_eq!(training_rate, inference_rate);
        assert_eq!(training_rate, base_rate);

        // Attention dropout should also be consistent
        let attention_training_rate = config.get_effective_attention_dropout_rate(base_rate, true);
        let attention_inference_rate =
            config.get_effective_attention_dropout_rate(base_rate, false);

        assert_eq!(attention_training_rate, attention_inference_rate);
        assert_eq!(attention_training_rate, base_rate); // Scale is 1.0
    }

    /// Test attention dropout configuration defaults
    #[test]
    fn test_attention_dropout_defaults() {
        let attention_config = AttentionDropoutConfig::default();

        assert!(attention_config.apply_to_attention_weights);
        assert!(attention_config.apply_to_attention_output);
        assert!(attention_config.apply_to_projections);
        assert_eq!(attention_config.attention_dropout_scale, 1.0);
    }

    /// Test comprehensive validation with rate differences
    #[test]
    fn test_rate_difference_warnings() {
        let config = DropoutConsistencyConfig::default();

        // Test large rate difference (should generate warning but not error)
        let result = config.validate_comprehensive_dropout_config(
            true,
            Some(0.1), // LSTM rate
            Some(0.5), // Attention rate - large difference
            "TestModel",
        );

        // Should succeed but generate warning (we can't easily test log output)
        assert!(result.is_ok());
    }
}

/// Performance test for dropout operations
#[cfg(test)]
mod performance_tests {
    use candle_core::{Device, Tensor};
    use std::time::Instant;

    /// Test dropout performance impact
    #[test]
    fn test_dropout_performance_overhead() {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, (100, 50), &device).unwrap();

        // Measure time without dropout
        let start = Instant::now();
        for _ in 0..100 {
            let _result = tensor.clone();
        }
        let no_dropout_time = start.elapsed();

        // Measure time with dropout
        let start = Instant::now();
        for _ in 0..100 {
            let _result = candle_nn::ops::dropout(&tensor, 0.2).unwrap();
        }
        let dropout_time = start.elapsed();

        // Performance test - dropout overhead should be reasonable
        // Note: This is a basic performance check, not a strict requirement
        if dropout_time.as_nanos() > no_dropout_time.as_nanos() * 10000 {
            println!(
                "⚠️  Warning: Dropout overhead is very high: {:.2}x",
                dropout_time.as_nanos() as f64 / no_dropout_time.as_nanos() as f64
            );
        }

        println!(
            "Dropout overhead: {:.2}x (no dropout: {:?}, with dropout: {:?})",
            dropout_time.as_nanos() as f64 / no_dropout_time.as_nanos() as f64,
            no_dropout_time,
            dropout_time
        );
    }
}
