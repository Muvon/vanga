//! Core LSTM forecasting system for cryptocurrency markets
//!
//! This library provides a complete LSTM-based forecasting system specifically
//! designed for cryptocurrency markets with automatic feature engineering,
//! hyperparameter optimization, and multi-target prediction capabilities.
//!
//! # Example
//!
//! ```rust,no_run
//! use vanga::{train_model, predict, TrainingConfig, PredictionConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Train a multi-target model
//! let config = TrainingConfig {
//!     symbol: "BTCUSDT".to_string(),
//!     data_path: "./data/btc_ohlcv.csv".into(),
//!     horizons: vec!["1h".to_string(), "4h".to_string(), "1d".to_string()],
//!     ..TrainingConfig::default()
//! };
//!
//! let model = train_model(config).await?;
//!
//! // Make predictions using the trained multi-target model
//! let pred_config = PredictionConfig {
//!     symbol: "BTCUSDT".to_string(),
//!     input_path: "./data/recent_btc.csv".into(),
//!     ..PredictionConfig::default()
//! };
//! let predictions = model.predict_multi_target(&pred_config).await?;
//! # Ok(())
//! # }
//! ```

pub mod api;
pub mod config;
pub mod data;
pub mod features;
pub mod model;
pub mod optimization; // NEW: Auto-optimization system
pub mod output;
pub mod realtime; // NEW: Real-time streaming prediction
pub mod targets;
pub mod utils;

#[cfg(test)]
mod tests;

// Re-export commonly used types for convenience
pub use config::{FeatureConfig, ModelConfig, PredictionConfig, TrainingConfig};
pub use utils::{Result, VangaError};

// Re-export high-level API functions
pub use api::predict;
pub use api::trainer::train_model;

// Integration test for the complete multi-target LSTM pipeline
//
// This test verifies the end-to-end workflow:
// 1. Target generation (PreparedTargets with HashMap structure)
// 2. Target conversion (HashMap -> Array2<f64> for training)
// 3. Model training with multi-target outputs
// 4. Prediction with structured JSON output
#[cfg(test)]
mod integration_tests {
    use crate::config::model::TargetsConfig;
    use crate::config::ModelConfig;
    use crate::data::TargetConverter;
    use crate::output::MultiTargetParser;
    use crate::targets::PreparedTargets;
    use ndarray::{s, Array2};

    #[test]
    fn test_model_config_creation() {
        let config = create_test_model_config();
        // Test the new TargetsConfig system
        assert_eq!(config.targets.base_sensitivity, 0.02);
        assert_eq!(config.targets.balance_target, 0.2);
        assert_eq!(config.targets.momentum_weighting, 1.2);
        assert_eq!(config.targets.extreme_multiplier, 2.0);
    }

    fn create_test_model_config() -> ModelConfig {
        ModelConfig {
            targets: TargetsConfig::default(),
            ..Default::default()
        }
    }

    fn create_test_targets() -> PreparedTargets {
        let mut targets = PreparedTargets::new(10);

        // Price levels: 5 bins (0-4)
        targets
            .price_levels
            .insert("1h".to_string(), vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);

        // Directions: 5 classes (0=DUMP, 1=DOWN, 2=SIDEWAYS, 3=UP, 4=PUMP)
        targets
            .directions
            .insert("1h".to_string(), vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0]);

        // Volatility: 5 levels (0-4)
        targets
            .volatility
            .insert("1h".to_string(), vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);

        targets.valid_indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        targets
    }

    #[test]
    fn test_end_to_end_target_conversion() {
        let converter = TargetConverter::new();
        let targets = create_test_targets();

        // Test target validation
        let validation_result = converter.validate_targets(&targets, "1h");
        assert!(validation_result.is_ok(), "Target validation should pass");

        // Test target conversion
        let conversion_result =
            converter.convert_to_training_array(&targets, &targets.valid_indices, "1h");
        assert!(
            conversion_result.is_ok(),
            "Target conversion should succeed"
        );

        let training_array = conversion_result.unwrap();

        // Verify dimensions
        assert_eq!(training_array.shape()[0], 10); // 10 samples

        // Calculate expected output size: 5 (price) + 5 (direction) + 5 (volatility)
        let expected_output_size = crate::config::model::NUM_CLASSES * 3; // 3 target types, 5 classes each
        assert_eq!(training_array.shape()[1], expected_output_size);
        assert_eq!(expected_output_size, 15); // 5 + 5 + 5

        // Verify one-hot encoding for first sample
        let first_sample = training_array.row(0);

        // Price levels: target[0] = 0, so first bin should be 1.0
        assert_eq!(first_sample[0], 1.0);
        assert_eq!(first_sample[1], 0.0);
        assert_eq!(first_sample[2], 0.0);
        assert_eq!(first_sample[3], 0.0);
        assert_eq!(first_sample[4], 0.0);

        // Direction: target[0] = 0, so first direction should be 1.0
        assert_eq!(first_sample[5], 1.0);
        assert_eq!(first_sample[6], 0.0);
        assert_eq!(first_sample[7], 0.0);

        // Volatility: target[0] = 0, normalized to 0.0
        assert_eq!(first_sample[8], 0.0);

        println!("✅ End-to-end target conversion test passed!");
    }

    #[test]
    fn test_slicing_debug() {
        use ndarray::{s, Array1};

        // Create test array [0.1, 0.3, 0.4, 0.15, 0.05, 0.2, 0.3, 0.5, 0.7]
        let data = vec![0.1, 0.3, 0.4, 0.15, 0.05, 0.2, 0.3, 0.5, 0.7];
        let array = Array1::from_vec(data);

        // Test slice [0..5] - should give us first 5 elements
        let slice = array.slice(s![0..5]);
        println!(
            "Slice [0..5] length: {}, content: {:?}",
            slice.len(),
            slice.to_vec()
        );

        // Test slice [5..8] - should give us next 3 elements
        let slice2 = array.slice(s![5..8]);
        println!(
            "Slice [5..8] length: {}, content: {:?}",
            slice2.len(),
            slice2.to_vec()
        );

        assert_eq!(slice.len(), 5);
        assert_eq!(slice2.len(), 3);

        println!("✅ Slicing debug test passed!");
    }

    #[test]
    fn test_parser_direct() {
        let parser = MultiTargetParser::new();

        // Create a simple 9-element array matching our expected structure
        let data = vec![0.1, 0.3, 0.4, 0.15, 0.05, 0.2, 0.3, 0.5, 0.7];
        let array = Array2::from_shape_vec((1, 9), data).unwrap();
        let array_view = array.row(0);

        println!("Input array length: {}", array_view.len());
        println!("Input array content: {:?}", array_view.to_vec());

        println!("Volatility classes: 5 (VeryLow, Low, Medium, High, VeryHigh)");

        // Check what segments the parser actually has
        println!("Parser segments: {:?}", parser.segments);

        // Call the parser and catch any error
        match parser.parse_output(array_view) {
            Ok(result) => {
                println!("✅ Parsing succeeded!");
                println!("Price levels: {:?}", result.price_levels);
                println!("Direction: {:?}", result.direction);
                println!("Volatility: {:?}", result.volatility);
            }
            Err(e) => {
                println!("❌ Parsing failed: {:?}", e);

                // Let's manually test the slicing that the parser should be doing
                // Price levels: 0-4, Direction: 5-9, Volatility: 10-14
                let price_start = 0;
                let price_end = crate::config::model::NUM_CLASSES;
                let slice = array_view.slice(s![price_start..price_end]);
                println!(
                    "Manual price slice: start={}, end={}, length={}, content={:?}",
                    price_start,
                    price_end,
                    slice.len(),
                    slice.to_vec()
                );

                panic!("Parser should succeed");
            }
        }
    }

    #[test]
    fn test_multi_target_output_parsing() {
        let parser = MultiTargetParser::new();

        // Debug: Check the segments
        let price_segment = (0, crate::config::model::NUM_CLASSES);
        let direction_segment = (
            crate::config::model::NUM_CLASSES,
            crate::config::model::NUM_CLASSES * 2,
        );
        let volatility_segment = (
            crate::config::model::NUM_CLASSES * 2,
            crate::config::model::NUM_CLASSES * 3,
        );
        println!("Debug - Price segment: {:?}", price_segment);
        println!("Debug - Direction segment: {:?}", direction_segment);
        println!("Debug - Volatility segment: {:?}", volatility_segment);
        println!(
            "Debug - Total output size: {}",
            crate::config::model::NUM_CLASSES * 3
        );

        // Create mock model output with correct dimensions
        let output_size = crate::config::model::NUM_CLASSES * 3; // 3 target types, 5 classes each
        let mut raw_output = vec![0.0; output_size];

        // Set realistic values for price levels (softmax-like)
        raw_output[0] = 0.1; // bin 0
        raw_output[1] = 0.3; // bin 1
        raw_output[2] = 0.4; // bin 2 (highest)
        raw_output[3] = 0.15; // bin 3
        raw_output[4] = 0.05; // bin 4

        // Set direction values
        raw_output[5] = 0.2; // DOWN
        raw_output[6] = 0.3; // SIDEWAYS
        raw_output[7] = 0.5; // UP (highest)

        // Set volatility value
        raw_output[8] = 0.7; // High volatility

        println!(
            "Debug - Raw output length: {}, content: {:?}",
            raw_output.len(),
            raw_output
        );

        // Convert to ArrayView1 for parsing
        let array = Array2::from_shape_vec((1, output_size), raw_output).unwrap();
        let array_view = array.row(0);

        println!("Debug - ArrayView1 length: {}", array_view.len());

        // Test parsing - let's debug the error first
        let parse_result = parser.parse_output(array_view);
        match &parse_result {
            Ok(_) => {
                let parsed = parse_result.unwrap();

                // Verify price levels were parsed
                assert!(
                    parsed.price_levels.is_some(),
                    "Price levels should be parsed"
                );

                // Verify direction was parsed
                assert!(parsed.direction.is_some(), "Direction should be parsed");

                // Verify volatility was parsed
                assert!(parsed.volatility.is_some(), "Volatility should be parsed");

                println!("✅ Multi-target output parsing test passed!");
            }
            Err(e) => {
                println!("❌ Parsing failed with error: {:?}", e);
                panic!("Output parsing should succeed");
            }
        }
    }

    #[test]
    fn test_model_output_size_configuration() {
        let total_size = 15;

        // Expected: 5 (price bins) + 5 (direction classes) + 5 (volatility classes per horizon)
        assert_eq!(total_size, 15);

        // Test segments - using new unified targets approach
        let price_segment = (0, crate::config::model::NUM_CLASSES);
        let direction_segment = (
            crate::config::model::NUM_CLASSES,
            crate::config::model::NUM_CLASSES * 2,
        );
        let volatility_segment = (
            crate::config::model::NUM_CLASSES * 2,
            crate::config::model::NUM_CLASSES * 3,
        );

        // Price levels segment
        assert_eq!(price_segment.0, 0);
        assert_eq!(price_segment.1, 5);

        // Direction segment
        assert_eq!(direction_segment.0, 5);
        assert_eq!(direction_segment.1, 10); // 5 classes for direction

        // Volatility segment
        assert_eq!(volatility_segment.0, 10);
        assert_eq!(volatility_segment.1, 15); // 5 classes for volatility

        println!("✅ Model output size configuration test passed!");
    }

    #[test]
    fn test_target_converter_validation_errors() {
        let converter = TargetConverter::new();

        // Test with missing targets
        let mut incomplete_targets = PreparedTargets::new(5);
        // Only add price levels, missing directions and volatility
        incomplete_targets
            .price_levels
            .insert("1h".to_string(), vec![0, 1, 2, 3, 4]);

        let validation_result = converter.validate_targets(&incomplete_targets, "1h");
        assert!(
            validation_result.is_err(),
            "Validation should fail with missing targets"
        );

        // Test with empty valid indices
        let complete_targets = create_test_targets();
        let conversion_result = converter.convert_to_training_array(&complete_targets, &[], "1h");
        assert!(
            conversion_result.is_err(),
            "Conversion should fail with empty indices"
        );

        println!("✅ Target converter validation error test passed!");
    }
}

#[cfg(test)]
mod attention_integration_tests {

    include!("tests/attention_integration.rs");
}

#[cfg(test)]
mod metrics_tests {
    include!("utils/metrics_test.rs");
}
