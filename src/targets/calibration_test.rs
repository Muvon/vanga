//! Tests for the clean calibration system

use crate::data::structures::MarketDataRow;
use crate::targets::calibration::*;

#[tokio::test]
async fn test_parameter_calibrator_basic() {
    let calibrator = ParameterCalibrator::new();

    // Create minimal test data
    let test_data = create_test_ohlcv_data(100);
    let horizons = vec!["1h".to_string()];

    let result = calibrator
        .calibrate(&test_data, 20, &horizons, Some(50), 0.8, 60)
        .await;


    assert!(result.is_ok());
    let params = result.unwrap();

    // Verify all parameters are within reasonable ranges
    assert!(params.direction.get("1h").unwrap().sensitivity > 0.0);
    assert!(params.direction.get("1h").unwrap().extreme_multiplier > 1.0);
    assert!(params.price_levels.get("1h").unwrap().bandwidth > 0.0);
    assert!(params.volatility.get("1h").unwrap().bandwidth > 0.0);
    assert!(params.sentiment.get("1h").unwrap().body_sensitivity > 0.0);
    assert!(params.volume.get("1h").unwrap().bandwidth > 0.0);

    // Verify metadata
    assert_eq!(params.metadata.data_length, 100);
    assert_eq!(params.metadata.sequence_length, 20);
    assert!(params.metadata.optimization_time_ms > 0);
}

#[tokio::test]
async fn test_calibration_with_insufficient_data() {
    let calibrator = ParameterCalibrator::new();

    let result = calibrator
        .calibrate(&test_data, 20, &horizons, Some(50), 0.8, 60)
        .await;


    let result = calibrator
        .calibrate(&test_data, 20, &horizons, Some(50), 0.8, 60, None)
        .await;

    // Should still work but with default parameters
    assert!(result.is_ok());
}

#[test]
fn test_class_balance_calculation() {
    let calibrator = ParameterCalibrator::new();

    // Test balanced distribution
    let balanced_counts = vec![20, 20, 20, 20, 20];
    let balance = calibrator.calculate_balance(&balanced_counts, 100).unwrap();

    assert_eq!(balance.total_samples, 100);
    assert!(balance.balance_score < 1.0); // Should be very low for perfect balance
    assert!(balance.imbalance_ratio < 1.5); // Should be close to 1.0

    // Test imbalanced distribution
    let imbalanced_counts = vec![50, 10, 10, 10, 20];
    let balance = calibrator
        .calculate_balance(&imbalanced_counts, 100)
        .unwrap();

    assert!(balance.balance_score > 5.0); // Should be higher for imbalance
    assert!(balance.imbalance_ratio > 2.0); // Should show significant imbalance
}

#[test]
fn test_calibrated_parameters_serialization() {
    let params = CalibratedParameters::default();

    // Test serialization
    let serialized = serde_json::to_string(&params);
    assert!(serialized.is_ok());

    // Test deserialization
    let deserialized: Result<CalibratedParameters, _> = serde_json::from_str(&serialized.unwrap());
    assert!(deserialized.is_ok());
}

#[test]
fn test_class_balance_default() {
    let balance = ClassBalance::default();

    assert_eq!(balance.total_samples, 0);
    assert_eq!(balance.balance_score, f64::INFINITY);
    assert_eq!(balance.imbalance_ratio, f64::INFINITY);
    assert_eq!(balance.target_balance, 0.2);
}

/// Helper function to create test OHLCV data
fn create_test_ohlcv_data(count: usize) -> Vec<MarketDataRow> {
    let mut data = Vec::new();
    let mut price = 100.0;

    for i in 0..count {
        let volatility = 0.02 * (i as f64 * 0.1).sin(); // Varying volatility
        let change = volatility * ((i as f64 * 0.3).sin() - 0.5);

        price *= 1.0 + change;
        let high = price * (1.0 + volatility.abs());
        let low = price * (1.0 - volatility.abs());
        let volume = 1000000.0 * (1.0 + 0.5 * (i as f64 * 0.2).cos());

        data.push(MarketDataRow {
            timestamp: i as i64 * 3600, // Hourly data
            open: price,
            high,
            low,
            close: price,
            volume,
        });
    }

    data
}
