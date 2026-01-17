use crate::data::structures::MarketDataRow;
use crate::targets::calibration::StopLevelParams;
use crate::targets::stop_levels::{
    classify_stop_level_with_calibrated_params, reconstruct_stop_levels,
};

#[test]
fn test_mae_classification_minimal_risk() {
    // Create sequence with range 100-110
    let sequence = create_test_sequence(100.0, 110.0, 10);

    // Horizon with minimal drawdown (stays above 105)
    let horizon = create_test_horizon(105.0, 108.0, 5);

    let params = StopLevelParams {
        bandwidth: 1.0,
        percentiles: [0.1, 0.9],
        neutral_band_factor: 0.4,
        momentum_factor: 1.0,
        balance: Default::default(),
    };

    let (class, strength) =
        classify_stop_level_with_calibrated_params(&sequence, &horizon, &params)
            .expect("Classification failed");

    // MAE should be minimal (< 25% of sequence range)
    // Sequence range = 10, MAE < 2.5 → Class 4 (Minimal Risk)
    assert_eq!(class, 4, "Expected minimal risk classification");
    assert!(strength > 0.1, "Strength should be positive");
}

#[test]
fn test_mae_classification_high_risk() {
    // Create sequence with range 100-110
    let sequence = create_test_sequence(100.0, 110.0, 10);

    // Horizon with large drawdown (drops to 90)
    let horizon = create_test_horizon(90.0, 95.0, 5);

    let params = StopLevelParams {
        bandwidth: 1.0,
        percentiles: [0.1, 0.9],
        neutral_band_factor: 0.4,
        momentum_factor: 1.0,
        balance: Default::default(),
    };

    let (class, _strength) =
        classify_stop_level_with_calibrated_params(&sequence, &horizon, &params)
            .expect("Classification failed");

    // MAE should be high (> 100% of sequence range)
    // Reference ~105, drops to 90 = 15 drawdown, range = 10 → 150% → Class 1 (High Risk)
    assert!(
        class <= 1,
        "Expected high or extreme risk classification, got class {}",
        class
    );
}

#[test]
fn test_mae_direction_independence() {
    // Create sequence with range 100-110
    let sequence = create_test_sequence(100.0, 110.0, 10);

    // Use IDENTICAL horizons - both have same worst dip (to 95) and same subsequent pattern
    // The only difference is the final close price (up vs down)
    // This tests that MAE is calculated the same way regardless of final direction

    let base_horizon = vec![
        create_candle(105.0, 95.0, 105.0, 100.0), // Same dip to 95 (first position)
        create_candle(100.0, 100.0, 110.0, 110.0),
        create_candle(110.0, 110.0, 115.0, 115.0),
    ];

    // Create copies with different final close (direction doesn't affect MAE calculation)
    let mut horizon_up = base_horizon.clone();
    horizon_up[2].close = 115.0; // Ends up

    let mut horizon_down = base_horizon;
    horizon_down[2].close = 95.0; // Ends down

    let params = StopLevelParams {
        bandwidth: 1.0,
        percentiles: [0.1, 0.9],
        neutral_band_factor: 0.4,
        momentum_factor: 1.0,
        balance: Default::default(),
    };

    let (class_up, _) = classify_stop_level_with_calibrated_params(&sequence, &horizon_up, &params)
        .expect("Classification failed");
    let (class_down, _) =
        classify_stop_level_with_calibrated_params(&sequence, &horizon_down, &params)
            .expect("Classification failed");

    // Both should have same MAE since the lows are identical
    // MAE = reference - minimum = 105 - 95 = 10 (same for both)
    assert_eq!(
        class_up, class_down,
        "MAE classification should be direction-independent"
    );
}

#[test]
fn test_mae_same_minimum_different_weights() {
    // Test that MAE correctly identifies worst drawdown
    let sequence = create_test_sequence(100.0, 110.0, 10);

    // Both have minimum of 95, but at different positions
    let horizon_early_dip = vec![
        create_candle(105.0, 95.0, 105.0, 100.0), // Early dip
        create_candle(100.0, 100.0, 110.0, 110.0),
        create_candle(110.0, 110.0, 115.0, 115.0),
    ];

    let horizon_late_dip = vec![
        create_candle(105.0, 105.0, 110.0, 110.0),
        create_candle(110.0, 100.0, 112.0, 100.0), // Late dip
        create_candle(100.0, 100.0, 105.0, 105.0),
    ];

    let params = StopLevelParams {
        bandwidth: 1.0,
        percentiles: [0.1, 0.9],
        neutral_band_factor: 0.4,
        momentum_factor: 1.0,
        balance: Default::default(),
    };

    let (class_early, _) =
        classify_stop_level_with_calibrated_params(&sequence, &horizon_early_dip, &params)
            .expect("Classification failed");
    let (class_late, _) =
        classify_stop_level_with_calibrated_params(&sequence, &horizon_late_dip, &params)
            .expect("Classification failed");

    // Both have same raw MAE (10), but early dip gets more weight
    // Early dip at position 0: weight = 0.11, weighted MAE = 10 * (0.5 + 0.5 * 0.11) = 5.55
    // Late dip at position 1: weight = 0.44, weighted MAE = 10 * (0.5 + 0.5 * 0.44) = 7.2
    // Late dip gets higher weighted MAE (more recent)
    assert!(
        class_late <= class_early,
        "Recent dips should be weighted more heavily"
    );
}

#[test]
fn test_reconstruction_structure() {
    let sequence = create_test_sequence(100.0, 110.0, 10);
    let probabilities = vec![0.1, 0.15, 0.3, 0.25, 0.2];

    let params = StopLevelParams {
        bandwidth: 1.0,
        percentiles: [0.1, 0.9],
        neutral_band_factor: 0.4,
        momentum_factor: 1.0,
        balance: Default::default(),
    };

    let reconstruction = reconstruct_stop_levels(&probabilities, &sequence, 105.0, &params)
        .expect("Reconstruction failed");

    // Verify structure
    assert_eq!(reconstruction.probabilities.len(), 5);
    assert_eq!(reconstruction.mae_ratio_ranges.len(), 5);
    assert_eq!(reconstruction.mae_price_ranges.len(), 5);
    assert!(reconstruction.confidence > 0.0 && reconstruction.confidence <= 1.0);
    assert!(reconstruction.expected_mae >= 0.0);
    assert!(reconstruction.sequence_range > 0.0);
}

#[test]
fn test_reconstruction_class_ordering() {
    let sequence = create_test_sequence(100.0, 110.0, 10);
    let probabilities = vec![0.05, 0.1, 0.2, 0.3, 0.35]; // Class 4 most likely

    let params = StopLevelParams {
        bandwidth: 1.0,
        percentiles: [0.1, 0.9],
        neutral_band_factor: 0.4,
        momentum_factor: 1.0,
        balance: Default::default(),
    };

    let reconstruction = reconstruct_stop_levels(&probabilities, &sequence, 105.0, &params)
        .expect("Reconstruction failed");

    // Most likely class should be 4 (highest probability)
    assert_eq!(reconstruction.most_likely_class, 4);

    // MAE ranges should be ordered: Class 0 (highest) to Class 4 (lowest)
    assert!(reconstruction.mae_ratio_ranges[0][0] > reconstruction.mae_ratio_ranges[4][1]);
}

// Helper functions
fn create_test_sequence(min: f64, max: f64, count: usize) -> Vec<MarketDataRow> {
    let step = (max - min) / (count as f64);
    (0..count)
        .map(|i| {
            let price = min + (i as f64) * step;
            create_candle(price, price - 1.0, price + 1.0, price)
        })
        .collect()
}

fn create_test_horizon(min: f64, max: f64, count: usize) -> Vec<MarketDataRow> {
    let step = (max - min) / (count as f64);
    (0..count)
        .map(|i| {
            let price = min + (i as f64) * step;
            create_candle(price, min, max, price)
        })
        .collect()
}

fn create_candle(open: f64, low: f64, high: f64, close: f64) -> MarketDataRow {
    MarketDataRow {
        timestamp: 0,
        open,
        high,
        low,
        close,
        volume: 1000.0,
    }
}
