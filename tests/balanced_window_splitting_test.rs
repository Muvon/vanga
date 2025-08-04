//! Integration tests for balanced window splitting

use std::io::Write;
use std::path::PathBuf;
use tempfile::NamedTempFile;
use vanga::config::training::{BatchSizeConfig, DeviceConfig, EpochConfig, OptimizerType};
use vanga::config::training::{ClassWeightStrategy, OptimizationMethod, OptimizationMetric};
use vanga::config::training::{
    DataConfig, EarlyStoppingConfig, OptimizationConfig, TrainingParams,
};
use vanga::config::training::{
    FeatureSelectionConfig, NormalizationMethod, OutlierHandling, OutlierMethod,
};
use vanga::config::{FeatureConfig, ModelConfig, TrainingConfig};
use vanga::data::DataPipeline;

/// Create a test CSV file with synthetic data
fn create_test_csv(num_rows: usize) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let mut file = NamedTempFile::with_suffix(".csv")?;

    // Write header
    writeln!(file, "timestamp,open,high,low,close,volume")?;

    // Write synthetic data with patterns that create class imbalance
    let base_price = 100.0;
    let mut timestamp = 1640995200; // 2022-01-01

    for i in 0..num_rows {
        // Create price patterns that lead to imbalanced classes
        let trend = if i < num_rows / 3 {
            1.0 + (i as f64 * 0.001) // Uptrend (fewer samples)
        } else if i < 2 * num_rows / 3 {
            1.0 - ((i - num_rows / 3) as f64 * 0.0005) // Downtrend (more samples)
        } else {
            1.0 + ((i % 10) as f64 - 5.0) * 0.001 // Sideways (moderate samples)
        };

        // Add some noise to make it more realistic
        let noise = ((i * 17) % 100) as f64 / 10000.0; // Deterministic noise

        let open = base_price * trend * (1.0 + noise);
        let close = open * (1.0 + ((i % 5) as f64 - 2.0) * 0.002);
        let high = open.max(close) * (1.001 + noise);
        let low = open.min(close) * (0.999 - noise);
        let volume = 1000.0 + (i % 100) as f64 * 10.0;

        writeln!(
            file,
            "{},{:.6},{:.6},{:.6},{:.6},{:.2}",
            timestamp, open, high, low, close, volume
        )?;

        timestamp += 3600; // 1 hour intervals
    }

    file.flush()?;
    Ok(file)
}

/// Create a minimal feature configuration for tests
fn create_minimal_feature_config() -> FeatureConfig {
    let mut config = FeatureConfig::default();

    // Disable all features that require large windows
    config.technical_indicators.enabled = false;
    config.volatility_features.enabled = false;
    config.cross_asset.enabled = false;
    config.engineering.rolling_features.enabled = false;
    config.engineering.lag_features.enabled = false;

    config
}

/// Create a test training configuration
fn create_test_config(data_path: PathBuf) -> TrainingConfig {
    TrainingConfig {
        symbol: "TESTUSDT".to_string(),
        data_path,
        fresh_training: true,
        continue_training: false,
        horizons: vec!["1h".to_string(), "4h".to_string()],
        features: create_minimal_feature_config(),
        model: ModelConfig::default(),
        training: TrainingParams {
            device: DeviceConfig::CPU,
            epochs: EpochConfig::Fixed(10),
            batch_size: BatchSizeConfig::Fixed(32),
            learning_rate: 0.001,
            optimizer: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: None,
                amsgrad: false,
            },
            warmup_epochs: 0,
            learning_schedule: None,
            validation_split: 0.2,
            validation_gap: "0".to_string(),
            test_split: 0.1,
            early_stopping: EarlyStoppingConfig {
                patience: 5,
                min_delta: 0.0001,
            },
            gradient_clip: Some(1.0),
            print_every: 1,
            class_weight_strategy: ClassWeightStrategy::PerWindow,
            window_decay: 1.0,
            min_train_ratio: 0.4,
            min_increment_ratio: 0.3,
            seed: 42,
        },
        data: DataConfig {
            normalization: NormalizationMethod::Standard,
            sequence_overlap: 0.5, // 50% overlap allowed
            outlier_handling: OutlierHandling {
                enabled: false,
                method: OutlierMethod::IQR,
                threshold: 3.0,
            },
            feature_selection: FeatureSelectionConfig {
                enabled: false,
                max_features: Some(50),
                correlation_threshold: 0.95,
                importance_threshold: 0.01,
            },
        },
        optimization: OptimizationConfig {
            method: OptimizationMethod::None,
            n_trials: 0,
            timeout_seconds: None,
            metric: OptimizationMetric::DirectionalAccuracy,
        },
    }
}

#[tokio::test]
async fn test_balanced_window_creation() {
    // Create test data
    let test_file = create_test_csv(2000).expect("Failed to create test CSV");
    let config = create_test_config(test_file.path().to_path_buf());

    // Create data pipeline
    let pipeline = DataPipeline::new();

    // Prepare training data with balanced approach
    let windows = pipeline
        .prepare_training_data(test_file.path(), &config)
        .await
        .expect("Failed to prepare training data");

    // Verify windows were created
    assert!(!windows.is_empty(), "No windows created");

    // Verify each window has balanced data
    for (idx, window) in windows.iter().enumerate() {
        println!(
            "Window {}: train={} sequences, val={} sequences",
            idx + 1,
            window.train_data.sequences.shape()[0],
            window.val_data.sequences.shape()[0]
        );

        // Check sequence tracking
        assert!(
            !window.train_sequence_indices.is_empty(),
            "Window {} missing train sequence indices",
            idx + 1
        );

        // Verify validation is consistent across windows
        if idx > 0 {
            assert_eq!(
                window.val_sequence_indices.len(),
                windows[0].val_sequence_indices.len(),
                "Validation size inconsistent across windows"
            );
        }

        // Check class weights were calculated
        if matches!(
            config.training.class_weight_strategy,
            ClassWeightStrategy::PerWindow
        ) {
            assert!(
                !window.target_class_weights.is_empty(),
                "Window {} missing class weights",
                idx + 1
            );
        }
    }
}

#[tokio::test]
async fn test_sequence_overlap_handling() {
    // Create test data
    let test_file = create_test_csv(1000).expect("Failed to create test CSV");
    let mut config = create_test_config(test_file.path().to_path_buf());

    // Test with no overlap allowed
    config.data.sequence_overlap = 0.0;

    let pipeline = DataPipeline::new();
    let windows = pipeline
        .prepare_training_data(test_file.path(), &config)
        .await
        .expect("Failed to prepare training data");

    // Verify sequences don't overlap
    for window in &windows {
        // Check that validation sequences don't overlap with any training sequences
        for val_idx in &window.val_sequence_indices {
            for train_indices in window.train_sequence_indices.values() {
                assert!(
                    !train_indices.contains(val_idx),
                    "Validation sequence {} found in training set",
                    val_idx
                );
            }
        }
    }
}

#[tokio::test]
async fn test_target_specific_balancing() {
    // Create test data with known imbalance
    let test_file = create_test_csv(1500).expect("Failed to create test CSV");
    let config = create_test_config(test_file.path().to_path_buf());

    let pipeline = DataPipeline::new();
    let windows = pipeline
        .prepare_training_data(test_file.path(), &config)
        .await
        .expect("Failed to prepare training data");

    // Check that each target has its own balanced set
    for window in &windows {
        let mut target_counts = std::collections::HashMap::new();

        for ((target_type, horizon), indices) in &window.train_sequence_indices {
            target_counts.insert(format!("{:?}_{}", target_type, horizon), indices.len());

            println!(
                "Target {:?} {}: {} sequences selected",
                target_type,
                horizon,
                indices.len()
            );
        }

        // Different targets may have different counts due to different balance requirements
        assert!(
            !target_counts.is_empty(),
            "No target-specific sequences found"
        );
    }
}

#[tokio::test]
async fn test_validation_consistency() {
    // Create test data
    let test_file = create_test_csv(2000).expect("Failed to create test CSV");
    let config = create_test_config(test_file.path().to_path_buf());

    let pipeline = DataPipeline::new();
    let windows = pipeline
        .prepare_training_data(test_file.path(), &config)
        .await
        .expect("Failed to prepare training data");

    assert!(windows.len() > 1, "Need multiple windows for this test");

    // Verify validation set is consistent across all windows
    let first_val_indices = &windows[0].val_sequence_indices;

    for (idx, window) in windows.iter().enumerate().skip(1) {
        assert_eq!(
            window.val_sequence_indices,
            *first_val_indices,
            "Window {} has different validation indices",
            idx + 1
        );

        // Also verify validation data shape is consistent
        assert_eq!(
            window.val_data.sequences.shape(),
            windows[0].val_data.sequences.shape(),
            "Window {} has different validation data shape",
            idx + 1
        );
    }
}

#[tokio::test]
async fn test_window_progression() {
    // Create test data
    let test_file = create_test_csv(2500).expect("Failed to create test CSV");
    let config = create_test_config(test_file.path().to_path_buf());

    let pipeline = DataPipeline::new();
    let windows = pipeline
        .prepare_training_data(test_file.path(), &config)
        .await
        .expect("Failed to prepare training data");

    // Verify windows are expanding
    let mut prev_train_samples = 0;
    for (idx, window) in windows.iter().enumerate() {
        assert!(
            window.train_samples >= prev_train_samples,
            "Window {} has fewer samples ({}) than previous ({})",
            idx + 1,
            window.train_samples,
            prev_train_samples
        );

        if idx > 0 {
            let increment = window.train_samples - prev_train_samples;
            let increment_ratio = increment as f64 / prev_train_samples as f64;

            println!(
                "Window {}: {} samples (+{}, {:.1}% increase)",
                idx + 1,
                window.train_samples,
                increment,
                increment_ratio * 100.0
            );

            // Verify minimum increment ratio (if not the last window)
            if idx < windows.len() - 1 {
                assert!(
                    increment_ratio >= config.training.min_increment_ratio - 0.01,
                    "Window {} increment ratio {:.2} below minimum {:.2}",
                    idx + 1,
                    increment_ratio,
                    config.training.min_increment_ratio
                );
            }
        }

        prev_train_samples = window.train_samples;
    }
}

#[tokio::test]
async fn test_edge_cases() {
    // Test with very small dataset
    let small_file = create_test_csv(500).expect("Failed to create test CSV");
    let config = create_test_config(small_file.path().to_path_buf());

    let pipeline = DataPipeline::new();

    // Should handle small datasets gracefully
    let result = pipeline
        .prepare_training_data(small_file.path(), &config)
        .await;

    match result {
        Ok(windows) => {
            // If it succeeds, should create at least one window
            assert!(!windows.is_empty(), "Should create at least one window");
        }
        Err(e) => {
            // If it fails, should be due to insufficient data
            assert!(
                e.to_string().contains("Insufficient"),
                "Expected insufficient data error, got: {}",
                e
            );
        }
    }
}
