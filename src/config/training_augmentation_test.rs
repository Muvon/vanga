//! Tests for augmentation configuration

use crate::config::training::DataConfig;

#[test]
fn test_data_config_defaults() {
    let config = DataConfig::default();

    assert_eq!(
        config.sequence_augment, false,
        "Augmentation should be disabled by default"
    );
    assert_eq!(
        config.augment_target_percentile, 0.5,
        "Target percentile should default to 0.5 (median)"
    );
    assert_eq!(
        config.max_synthetic_ratio, 2.0,
        "Max synthetic ratio should default to 2.0"
    );
    assert_eq!(
        config.sequence_overlap, 0.8,
        "Sequence overlap should default to 0.8"
    );
}

#[test]
fn test_data_config_deserialization_with_new_fields() {
    let toml_str = r#"
        normalization = "Robust"
        sequence_overlap = 0.3
        sequence_augment = true
        augment_target_percentile = 0.6
        max_synthetic_ratio = 3.0
        
        [outlier_handling]
        enabled = true
        method = "ModifiedZScore"
        threshold = 3.5
        
        [feature_selection]
        enabled = true
        correlation_threshold = 0.95
        importance_threshold = 0.001
    "#;

    let config: Result<DataConfig, _> = toml::from_str(toml_str);
    assert!(config.is_ok(), "Should deserialize config with new fields");

    let config = config.unwrap();
    assert_eq!(config.sequence_augment, true);
    assert_eq!(config.augment_target_percentile, 0.6);
    assert_eq!(config.max_synthetic_ratio, 3.0);
    assert_eq!(config.sequence_overlap, 0.3);
}

#[test]
fn test_backward_compatibility_without_new_fields() {
    // Old config without augmentation fields
    let toml_str = r#"
        normalization = "Robust"
        sequence_overlap = 0.8
        
        [outlier_handling]
        enabled = true
        method = "ModifiedZScore"
        threshold = 3.5
        
        [feature_selection]
        enabled = true
        correlation_threshold = 0.95
        importance_threshold = 0.001
    "#;

    let config: Result<DataConfig, _> = toml::from_str(toml_str);
    assert!(
        config.is_ok(),
        "Should deserialize old config without new fields"
    );

    let config = config.unwrap();
    // Should use defaults for new fields
    assert_eq!(config.sequence_augment, false, "Should default to false");
    assert_eq!(
        config.augment_target_percentile, 0.5,
        "Should default to 0.5"
    );
    assert_eq!(config.max_synthetic_ratio, 2.0, "Should default to 2.0");
}

#[test]
fn test_augment_percentile_values() {
    // Test various valid percentile values
    let percentiles = vec![0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0];

    for percentile in percentiles {
        let toml_str = format!(
            r#"
            normalization = "Robust"
            sequence_overlap = 0.5
            augment_target_percentile = {}
            
            [outlier_handling]
            enabled = true
            method = "ModifiedZScore"
            threshold = 3.5
            
            [feature_selection]
            enabled = true
            correlation_threshold = 0.95
            importance_threshold = 0.001
            "#,
            percentile
        );

        let config: Result<DataConfig, _> = toml::from_str(&toml_str);
        assert!(config.is_ok(), "Should accept percentile {}", percentile);
        assert_eq!(config.unwrap().augment_target_percentile, percentile);
    }
}

#[test]
fn test_max_synthetic_ratio_values() {
    // Test various valid ratio values
    let ratios = vec![0.5, 1.0, 1.5, 2.0, 3.0, 5.0];

    for ratio in ratios {
        let toml_str = format!(
            r#"
            normalization = "Robust"
            sequence_overlap = 0.5
            max_synthetic_ratio = {}
            
            [outlier_handling]
            enabled = true
            method = "ModifiedZScore"
            threshold = 3.5
            
            [feature_selection]
            enabled = true
            correlation_threshold = 0.95
            importance_threshold = 0.001
            "#,
            ratio
        );

        let config: Result<DataConfig, _> = toml::from_str(&toml_str);
        assert!(config.is_ok(), "Should accept ratio {}", ratio);
        assert_eq!(config.unwrap().max_synthetic_ratio, ratio);
    }
}

#[test]
fn test_augmentation_enabled_with_defaults() {
    let toml_str = r#"
        normalization = "Robust"
        sequence_overlap = 0.3
        sequence_augment = true
        
        [outlier_handling]
        enabled = true
        method = "ModifiedZScore"
        threshold = 3.5
        
        [feature_selection]
        enabled = true
        correlation_threshold = 0.95
        importance_threshold = 0.001
    "#;

    let config: Result<DataConfig, _> = toml::from_str(toml_str);
    assert!(config.is_ok());

    let config = config.unwrap();
    assert_eq!(config.sequence_augment, true);
    // Should use defaults when not specified
    assert_eq!(config.augment_target_percentile, 0.5);
    assert_eq!(config.max_synthetic_ratio, 2.0);
}

#[test]
fn test_conservative_augmentation_config() {
    let toml_str = r#"
        normalization = "Robust"
        sequence_overlap = 0.3
        sequence_augment = true
        augment_target_percentile = 0.4
        max_synthetic_ratio = 1.0
        
        [outlier_handling]
        enabled = true
        method = "ModifiedZScore"
        threshold = 3.5
        
        [feature_selection]
        enabled = true
        correlation_threshold = 0.95
        importance_threshold = 0.001
    "#;

    let config: Result<DataConfig, _> = toml::from_str(toml_str);
    assert!(config.is_ok(), "Conservative config should be valid");

    let config = config.unwrap();
    assert_eq!(
        config.augment_target_percentile, 0.4,
        "Conservative percentile"
    );
    assert_eq!(config.max_synthetic_ratio, 1.0, "Conservative ratio");
}

#[test]
fn test_aggressive_augmentation_config() {
    let toml_str = r#"
        normalization = "Robust"
        sequence_overlap = 0.3
        sequence_augment = true
        augment_target_percentile = 0.7
        max_synthetic_ratio = 3.0
        
        [outlier_handling]
        enabled = true
        method = "ModifiedZScore"
        threshold = 3.5
        
        [feature_selection]
        enabled = true
        correlation_threshold = 0.95
        importance_threshold = 0.001
    "#;

    let config: Result<DataConfig, _> = toml::from_str(toml_str);
    assert!(config.is_ok(), "Aggressive config should be valid");

    let config = config.unwrap();
    assert_eq!(
        config.augment_target_percentile, 0.7,
        "Aggressive percentile"
    );
    assert_eq!(config.max_synthetic_ratio, 3.0, "Aggressive ratio");
}
