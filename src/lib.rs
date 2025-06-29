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
//! // Train a model
//! let config = TrainingConfig::default()
//!     .symbol("BTCUSDT")
//!     .data_path("./data/btc_ohlcv.csv")
//!     .horizons(vec!["1h".to_string(), "4h".to_string(), "1d".to_string()]);
//!
//! let model = train_model(config).await?;
//!
//! // Make predictions
//! let pred_config = PredictionConfig::default()
//!     .symbol("BTCUSDT")
//!     .input_path("./data/recent_btc.csv");
//! let predictions = predict(pred_config, &model).await?;
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
pub mod targets;
pub mod utils;

// Re-export commonly used types for convenience
pub use config::{FeatureConfig, ModelConfig, PredictionConfig, TrainingConfig};
pub use utils::{Result, VangaError};

// Re-export high-level API functions
pub use api::predictor::predict;
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
    use crate::config::model::{
        DirectionHead, DistributionType, OutputHeadsConfig, PriceLevelHead, VolatilityHead,
        VolatilityPredictionMethod,
    };
    use crate::config::ModelConfig;
    use crate::data::TargetConverter;
    use crate::output::MultiTargetParser;
    use crate::targets::PreparedTargets;
    use ndarray::{s, Array2};

    fn create_test_output_heads() -> OutputHeadsConfig {
        OutputHeadsConfig {
            price_levels: PriceLevelHead {
                enabled: true,
                bins: 5,
                range_percent: 0.1,
                distribution_type: DistributionType::Categorical,
            },
            direction: DirectionHead {
                enabled: true,
                threshold: 0.02,
                confidence_calibration: false,
            },
            volatility: VolatilityHead {
                enabled: true,
                method: VolatilityPredictionMethod::Direct,
                horizons: vec!["1h".to_string()],
            },
        }
    }

    #[test]
    fn test_model_config_creation() {
        let config = create_test_model_config();
        assert!(config.output_heads.price_levels.enabled);
        assert!(config.output_heads.direction.enabled);
        assert!(config.output_heads.volatility.enabled);
        assert_eq!(config.output_heads.price_levels.bins, 5);
    }

    fn create_test_model_config() -> ModelConfig {
        ModelConfig {
            output_heads: create_test_output_heads(),
            ..Default::default()
        }
    }

    fn create_test_targets() -> PreparedTargets {
        let mut targets = PreparedTargets::new(10);

        // Price levels: 5 bins (0-4)
        targets
            .price_levels
            .insert("1h".to_string(), vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);

        // Directions: 3 classes (0=DOWN, 1=SIDEWAYS, 2=UP)
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
        let output_heads = create_test_output_heads();
        let converter = TargetConverter::new(output_heads.clone());
        let targets = create_test_targets();

        // Test target validation
        let validation_result = converter.validate_targets(&targets);
        assert!(validation_result.is_ok(), "Target validation should pass");

        // Test target conversion
        let conversion_result =
            converter.convert_to_training_array(&targets, &targets.valid_indices);
        assert!(
            conversion_result.is_ok(),
            "Target conversion should succeed"
        );

        let training_array = conversion_result.unwrap();

        // Verify dimensions
        assert_eq!(training_array.shape()[0], 10); // 10 samples

        // Calculate expected output size: 5 (price) + 3 (direction) + 1 (volatility)
        let expected_output_size = output_heads.calculate_total_output_size();
        assert_eq!(training_array.shape()[1], expected_output_size);
        assert_eq!(expected_output_size, 9); // 5 + 3 + 1

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
    fn test_segment_calculation() {
        let output_heads = create_test_output_heads();

        // Test segment calculation
        let segments = output_heads.get_output_segments();
        println!("Segments: {:?}", segments);

        // Test if the parser is using the right segments
        let parser = MultiTargetParser::new(output_heads.clone());

        // Verify parser can handle test data - create a single row view
        let test_predictions = Array2::from_shape_vec(
            (1, 9),
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, // Single sample
            ],
        )
        .unwrap();

        let single_row = test_predictions.row(0);
        let parsed_result = parser.parse_output(single_row);
        assert!(
            parsed_result.is_ok(),
            "Parser should handle test predictions"
        );

        let result = parsed_result.unwrap();
        println!("Parsed result: {:?}", result);

        // Access the segments field (we need to make it public or add a getter)
        // For now, let's just verify the calculation logic

        // Manually verify segments
        let mut offset = 0;
        if output_heads.price_levels.enabled {
            let size = output_heads.price_levels.bins as usize;
            println!(
                "Price levels: offset={}, size={}, range=({}, {})",
                offset,
                size,
                offset,
                offset + size
            );
            offset += size;
        }
        if output_heads.direction.enabled {
            let size = 3;
            println!(
                "Direction: offset={}, size={}, range=({}, {})",
                offset,
                size,
                offset,
                offset + size
            );
            offset += size;
        }
        if output_heads.volatility.enabled {
            let size = output_heads.volatility.horizons.len();
            println!(
                "Volatility: offset={}, size={}, range=({}, {})",
                offset,
                size,
                offset,
                offset + size
            );
        }

        println!("✅ Segment calculation test passed!");
    }

    #[test]
    fn test_parser_direct() {
        let output_heads = create_test_output_heads();
        let parser = MultiTargetParser::new(output_heads.clone());

        // Create a simple 9-element array matching our expected structure
        let data = vec![0.1, 0.3, 0.4, 0.15, 0.05, 0.2, 0.3, 0.5, 0.7];
        let array = Array2::from_shape_vec((1, 9), data).unwrap();
        let array_view = array.row(0);

        println!("Input array length: {}", array_view.len());
        println!("Input array content: {:?}", array_view.to_vec());

        // Test that our output_heads configuration is correct
        println!(
            "Price levels enabled: {}",
            output_heads.price_levels.enabled
        );
        println!("Price levels bins: {}", output_heads.price_levels.bins);
        println!("Direction enabled: {}", output_heads.direction.enabled);
        println!("Volatility enabled: {}", output_heads.volatility.enabled);
        println!(
            "Volatility horizons: {:?}",
            output_heads.volatility.horizons
        );

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
                let segments = output_heads.get_output_segments();
                if let Some((start, end)) = segments.price_levels {
                    let slice = array_view.slice(s![start..end]);
                    println!(
                        "Manual price slice: start={}, end={}, length={}, content={:?}",
                        start,
                        end,
                        slice.len(),
                        slice.to_vec()
                    );
                }

                panic!("Parser should succeed");
            }
        }
    }

    #[test]
    fn test_multi_target_output_parsing() {
        let output_heads = create_test_output_heads();
        let parser = MultiTargetParser::new(output_heads.clone());

        // Debug: Check the segments
        let segments = output_heads.get_output_segments();
        println!("Debug - Segments: {:?}", segments);
        println!(
            "Debug - Total output size: {}",
            output_heads.calculate_total_output_size()
        );

        // Create mock model output with correct dimensions
        let output_size = output_heads.calculate_total_output_size();
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
        let output_heads = create_test_output_heads();
        let total_size = output_heads.calculate_total_output_size();

        // Expected: 5 (price bins) + 3 (direction classes) + 1 (volatility horizons)
        assert_eq!(total_size, 9);

        // Test segments
        let segments = output_heads.get_output_segments();

        if let Some((start, end)) = segments.price_levels {
            assert_eq!(start, 0);
            assert_eq!(end, 5);
        } else {
            panic!("Price levels segment should be defined");
        }

        if let Some((start, end)) = segments.direction {
            assert_eq!(start, 5);
            assert_eq!(end, 8);
        } else {
            panic!("Direction segment should be defined");
        }

        if let Some((start, end)) = segments.volatility {
            assert_eq!(start, 8);
            assert_eq!(end, 9);
        } else {
            panic!("Volatility segment should be defined");
        }

        println!("✅ Model output size configuration test passed!");
    }

    #[test]
    fn test_target_converter_validation_errors() {
        let output_heads = create_test_output_heads();
        let converter = TargetConverter::new(output_heads);

        // Test with missing targets
        let mut incomplete_targets = PreparedTargets::new(5);
        // Only add price levels, missing directions and volatility
        incomplete_targets
            .price_levels
            .insert("1h".to_string(), vec![0, 1, 2, 3, 4]);

        let validation_result = converter.validate_targets(&incomplete_targets);
        assert!(
            validation_result.is_err(),
            "Validation should fail with missing targets"
        );

        // Test with empty valid indices
        let complete_targets = create_test_targets();
        let conversion_result = converter.convert_to_training_array(&complete_targets, &[]);
        assert!(
            conversion_result.is_err(),
            "Conversion should fail with empty indices"
        );

        println!("✅ Target converter validation error test passed!");
    }

    #[test]
    fn test_backward_compatibility() {
        // Test with all heads disabled (should work like single-output model)
        let mut disabled_heads = OutputHeadsConfig {
            price_levels: PriceLevelHead {
                enabled: false,
                bins: 5,
                range_percent: 0.1,
                distribution_type: DistributionType::Categorical,
            },
            direction: DirectionHead {
                enabled: false,
                threshold: 0.02,
                confidence_calibration: false,
            },
            volatility: VolatilityHead {
                enabled: false,
                method: VolatilityPredictionMethod::Direct,
                horizons: vec!["1h".to_string()],
            },
        };

        let total_size = disabled_heads.calculate_total_output_size();
        assert_eq!(
            total_size, 1,
            "Disabled heads should default to single output for backward compatibility"
        );

        // Enable only price levels
        disabled_heads.price_levels.enabled = true;
        let price_only_size = disabled_heads.calculate_total_output_size();
        assert_eq!(
            price_only_size, 5,
            "Only price levels should give 5 outputs"
        );

        println!("✅ Backward compatibility test passed!");
    }
}
