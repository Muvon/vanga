//! Comprehensive tests for the new target interface system
//!
//! This module tests the trait-based target generation interface,
//! ensuring compatibility with existing functionality and proper
//! operation of the new registry and orchestration systems.

use crate::config::model::TargetsConfig;
use crate::targets::adaptive_parameters::{
    AdaptiveTargetParameters, DirectionAdaptiveParams, PriceLevelAdaptiveParams,
    SentimentAdaptiveParams, VolatilityAdaptiveParams, VolumeAdaptiveParams,
};
use crate::targets::interface::{AdaptiveParameters, TargetGenerator};
use crate::targets::registry::TargetRegistry;
use crate::targets::{MultiTargetConfig, TargetGenerator as Orchestrator};
use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Test that all target generators are properly registered
#[test]
fn test_registry_contains_all_targets() {
    let registry = TargetRegistry::new();

    // Check that all 5 target types are registered
    let expected_targets = vec![
        "price_levels",
        "direction",
        "volatility",
        "sentiment",
        "volume",
    ];

    for target_type in expected_targets {
        let generator = registry.get(target_type);
        assert!(
            generator.is_some(),
            "Target type '{}' should be registered",
            target_type
        );

        let gen = generator.unwrap();
        assert_eq!(gen.target_type(), target_type);
        assert!(!gen.target_name().is_empty());
        assert_eq!(gen.class_names().len(), 5); // All targets have 5 classes
    }

    // Check total count
    assert_eq!(registry.len(), 5);
    assert!(!registry.is_empty());
}

/// Test that registry correctly filters enabled targets
#[test]
fn test_registry_enabled_filtering() {
    let registry = TargetRegistry::new();

    // Test with all targets enabled
    let mut config = MultiTargetConfig::default();
    config.price_levels.enabled = true;
    config.direction.enabled = true;
    config.volatility.enabled = true;
    config.sentiment.enabled = true;
    config.volume.enabled = true;

    let enabled = registry.get_enabled_generators(&config);
    assert_eq!(enabled.len(), 5);

    // Test with only some targets enabled
    config.price_levels.enabled = true;
    config.direction.enabled = false;
    config.volatility.enabled = true;
    config.sentiment.enabled = false;
    config.volume.enabled = false;

    let enabled = registry.get_enabled_generators(&config);
    assert_eq!(enabled.len(), 2);

    let enabled_types: Vec<&str> = enabled.iter().map(|g| g.target_type()).collect();
    assert!(enabled_types.contains(&"price_levels"));
    assert!(enabled_types.contains(&"volatility"));
    assert!(!enabled_types.contains(&"direction"));
}

/// Test that all target generators implement the trait correctly
#[tokio::test]
async fn test_all_targets_implement_trait() {
    let registry = TargetRegistry::new();
    let target_types = vec![
        "price_levels",
        "direction",
        "volatility",
        "sentiment",
        "volume",
    ];

    for target_type in target_types {
        let generator = registry
            .get(target_type)
            .unwrap_or_else(|| panic!("Target {} not registered", target_type));

        // Test trait methods
        assert!(!generator.target_type().is_empty());
        assert!(!generator.target_name().is_empty());
        assert_eq!(generator.class_names().len(), 5); // All targets have 5 classes

        // Test that class names are not empty
        for class_name in generator.class_names() {
            assert!(!class_name.is_empty());
        }
    }
}

/// Test adaptive parameter trait implementations
#[test]
fn test_adaptive_parameters_trait() {
    // Test DirectionAdaptiveParams
    let direction_params = DirectionAdaptiveParams::default();
    let boxed: Box<dyn AdaptiveParameters> = Box::new(direction_params.clone());
    let downcasted = boxed.as_any().downcast_ref::<DirectionAdaptiveParams>();
    assert!(downcasted.is_some());
    assert_eq!(
        downcasted.unwrap().base_sensitivity,
        direction_params.base_sensitivity
    );

    // Test PriceLevelAdaptiveParams
    let price_params = PriceLevelAdaptiveParams::default();
    let boxed: Box<dyn AdaptiveParameters> = Box::new(price_params.clone());
    let downcasted = boxed.as_any().downcast_ref::<PriceLevelAdaptiveParams>();
    assert!(downcasted.is_some());
    assert_eq!(
        downcasted.unwrap().bandwidth_size,
        price_params.bandwidth_size
    );

    // Test VolatilityAdaptiveParams
    let volatility_params = VolatilityAdaptiveParams::default();
    let boxed: Box<dyn AdaptiveParameters> = Box::new(volatility_params.clone());
    let downcasted = boxed.as_any().downcast_ref::<VolatilityAdaptiveParams>();
    assert!(downcasted.is_some());
    assert_eq!(
        downcasted.unwrap().bandwidth_size,
        volatility_params.bandwidth_size
    );

    // Test SentimentAdaptiveParams
    let sentiment_params = SentimentAdaptiveParams::default();
    let boxed: Box<dyn AdaptiveParameters> = Box::new(sentiment_params.clone());
    let downcasted = boxed.as_any().downcast_ref::<SentimentAdaptiveParams>();
    assert!(downcasted.is_some());
    assert_eq!(
        downcasted.unwrap().body_sensitivity,
        sentiment_params.body_sensitivity
    );

    // Test VolumeAdaptiveParams
    let volume_params = VolumeAdaptiveParams::default();
    let boxed: Box<dyn AdaptiveParameters> = Box::new(volume_params.clone());
    let downcasted = boxed.as_any().downcast_ref::<VolumeAdaptiveParams>();
    assert!(downcasted.is_some());
    assert_eq!(
        downcasted.unwrap().bandwidth_size,
        volume_params.bandwidth_size
    );
}

/// Test orchestrator with trait-based approach
#[test]
fn test_orchestrator_with_registry() {
    let config = MultiTargetConfig::default();
    let orchestrator = Orchestrator::new(config);

    // Test that registry is properly initialized
    let registry = orchestrator.get_registry();
    assert_eq!(registry.len(), 5);

    // Test that all target types are available
    let target_types = registry.get_all_types();
    assert_eq!(target_types.len(), 5);
    assert!(target_types.contains(&"price_levels".to_string()));
    assert!(target_types.contains(&"direction".to_string()));
    assert!(target_types.contains(&"volatility".to_string()));
    assert!(target_types.contains(&"sentiment".to_string()));
    assert!(target_types.contains(&"volume".to_string()));
}

/// Test adaptive parameter mapping in orchestrator
#[test]
fn test_adaptive_parameter_mapping() {
    let config = MultiTargetConfig::default();
    let orchestrator = Orchestrator::new(config);

    // Create test adaptive parameters
    let adaptive_params = AdaptiveTargetParameters::default();

    // Test parameter mapping for each target type
    let price_param =
        orchestrator.get_adaptive_param_for_target("price_levels", Some(&adaptive_params));
    assert!(price_param.is_some());

    let direction_param =
        orchestrator.get_adaptive_param_for_target("direction", Some(&adaptive_params));
    assert!(direction_param.is_some());

    let volatility_param =
        orchestrator.get_adaptive_param_for_target("volatility", Some(&adaptive_params));
    assert!(volatility_param.is_some());

    let sentiment_param =
        orchestrator.get_adaptive_param_for_target("sentiment", Some(&adaptive_params));
    assert!(sentiment_param.is_some());

    let volume_param = orchestrator.get_adaptive_param_for_target("volume", Some(&adaptive_params));
    assert!(volume_param.is_some());

    // Test unknown target type
    let unknown_param =
        orchestrator.get_adaptive_param_for_target("unknown", Some(&adaptive_params));
    assert!(unknown_param.is_none());

    // Test with no adaptive parameters
    let no_param = orchestrator.get_adaptive_param_for_target("price_levels", None);
    assert!(no_param.is_none());
}

/// Test that class names are consistent across all targets
#[test]
fn test_class_names_consistency() {
    let registry = TargetRegistry::new();

    // Test price levels
    let price_gen = registry.get("price_levels").unwrap();
    let price_classes = price_gen.class_names();
    assert_eq!(price_classes.len(), 5);
    assert!(price_classes.contains(&"Strong Down"));
    assert!(price_classes.contains(&"Neutral"));
    assert!(price_classes.contains(&"Strong Up"));

    // Test direction
    let direction_gen = registry.get("direction").unwrap();
    let direction_classes = direction_gen.class_names();
    assert_eq!(direction_classes.len(), 5);
    assert!(direction_classes.contains(&"DUMP"));
    assert!(direction_classes.contains(&"SIDEWAYS"));
    assert!(direction_classes.contains(&"PUMP"));

    // Test volatility
    let volatility_gen = registry.get("volatility").unwrap();
    let volatility_classes = volatility_gen.class_names();
    assert_eq!(volatility_classes.len(), 5);
    assert!(volatility_classes.contains(&"VeryLow"));
    assert!(volatility_classes.contains(&"Medium"));
    assert!(volatility_classes.contains(&"VeryHigh"));

    // Test sentiment
    let sentiment_gen = registry.get("sentiment").unwrap();
    let sentiment_classes = sentiment_gen.class_names();
    assert_eq!(sentiment_classes.len(), 5);
    assert!(sentiment_classes.contains(&"Strong Panic"));
    assert!(sentiment_classes.contains(&"Neutral"));
    assert!(sentiment_classes.contains(&"Strong Greed"));

    // Test volume
    let volume_gen = registry.get("volume").unwrap();
    let volume_classes = volume_gen.class_names();
    assert_eq!(volume_classes.len(), 5);
    assert!(volume_classes.contains(&"Very Low"));
    assert!(volume_classes.contains(&"Medium"));
    assert!(volume_classes.contains(&"Very High"));
}

/// Test registry extensibility (adding custom targets)
#[test]
fn test_registry_extensibility() {
    let mut registry = TargetRegistry::new();

    // Initial count should be 5
    assert_eq!(registry.len(), 5);

    // Create a mock target generator for testing
    struct MockTargetGenerator;

    impl TargetGenerator for MockTargetGenerator {
        fn target_type(&self) -> &'static str {
            "mock"
        }
        fn target_name(&self) -> &'static str {
            "Mock Target"
        }
        fn class_names(&self) -> Vec<&'static str> {
            vec!["Class0", "Class1", "Class2", "Class3", "Class4"]
        }

        fn generate_targets(
            &self,
            _df: &DataFrame,
            _horizons: &[String],
            _targets_config: &TargetsConfig,
            _sequence_indices: &[usize],
            _sequence_length: usize,
            _adaptive_params: Option<&dyn AdaptiveParameters>,
        ) -> Result<HashMap<String, Vec<i32>>> {
            Ok(HashMap::new())
        }

        fn calibrate_parameters(
            &self,
            _df: &DataFrame,
            _sequence_length: usize,
            _horizon_steps: usize,
            _targets_config: &TargetsConfig,
        ) -> Result<Box<dyn AdaptiveParameters>> {
            Ok(Box::new(DirectionAdaptiveParams::default()))
        }
    }

    // Register custom target
    registry.register("mock", std::sync::Arc::new(MockTargetGenerator));

    // Should now have 6 targets
    assert_eq!(registry.len(), 6);

    // Should be able to retrieve the custom target
    let mock_gen = registry.get("mock");
    assert!(mock_gen.is_some());
    let mock_gen = mock_gen.unwrap();
    assert_eq!(mock_gen.target_type(), "mock");
    assert_eq!(mock_gen.target_name(), "Mock Target");
}

/// Integration test for the complete trait-based system
#[test]
fn test_trait_based_system_integration() {
    // Create orchestrator with default configuration
    let config = MultiTargetConfig::default();
    let orchestrator = Orchestrator::new(config);

    // Verify registry is properly initialized
    let registry = orchestrator.get_registry();
    assert_eq!(registry.len(), 5);

    // Test enabled generator filtering
    let mut test_config = MultiTargetConfig::default();
    test_config.price_levels.enabled = true;
    test_config.direction.enabled = true;
    test_config.volatility.enabled = false;
    test_config.sentiment.enabled = false;
    test_config.volume.enabled = false;

    let enabled_generators = registry.get_enabled_generators(&test_config);
    assert_eq!(enabled_generators.len(), 2);

    let enabled_names = registry.get_enabled_target_names(&test_config);
    assert_eq!(enabled_names.len(), 2);
    assert!(enabled_names.contains(&"Price Levels".to_string()));
    assert!(enabled_names.contains(&"Direction".to_string()));

    // Test that all components work together
    let all_types = registry.get_all_types();
    for target_type in all_types {
        let generator = registry.get(&target_type).unwrap();

        // Each generator should have proper metadata
        assert!(!generator.target_type().is_empty());
        assert!(!generator.target_name().is_empty());
        assert_eq!(generator.class_names().len(), 5);

        // Each generator should be able to handle adaptive parameters
        let adaptive_params = AdaptiveTargetParameters::default();
        let param =
            orchestrator.get_adaptive_param_for_target(&target_type, Some(&adaptive_params));
        assert!(param.is_some());
    }
}
