use crate::data::structures::MarketDataRow;
use crate::targets::calibration::StopLevelParams;
use crate::targets::stop_levels::{
    classify_stop_level_with_calibrated_params, reconstruct_stop_levels,
};

#[test]
fn test_adverse_classification_minimal_risk() {
    // Sequence: 100-110 with lows around 99-109
    let sequence = create_test_sequence(100.0, 110.0, 10);

    // Horizon: bullish move (105 → 108), minimal adverse (lows stay within sequence range)
    // The lows (104.9, 105.8, 106.9) are all within the sequence's low range
    let horizon = vec![
        create_candle(105.0, 104.9, 106.0, 106.0),
        create_candle(106.0, 105.8, 107.0, 107.0),
        create_candle(107.0, 106.9, 108.0, 108.0),
    ];

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

    // With boundary-based classification, the adverse (lowest low = 104.9)
    // is compared against sequence adverse boundaries
    // Class 2 (moderate) or higher is acceptable for this scenario
    assert!(
        class >= 2,
        "Expected moderate or lower risk classification, got class {}",
        class
    );
    assert!(strength > 0.1, "Strength should be positive");
}

#[test]
fn test_adverse_classification_high_risk() {
    // Sequence: 100-110
    let sequence = create_test_sequence(100.0, 110.0, 10);

    // Horizon: bullish move (105 → 110), but large adverse (drops to 90)
    let horizon = vec![
        create_candle(105.0, 90.0, 106.0, 100.0),
        create_candle(100.0, 95.0, 105.0, 105.0),
        create_candle(105.0, 100.0, 110.0, 110.0),
    ];

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

    // Large adverse movement (reference ~105, worst low 90)
    assert!(
        class <= 1,
        "Expected high or extreme risk classification, got class {}",
        class
    );
}

#[test]
fn test_adverse_direction_awareness() {
    // Sequence: 100-110
    let sequence = create_test_sequence(100.0, 110.0, 10);

    // Bullish horizon: adverse = lows
    let horizon_bullish = vec![
        create_candle(105.0, 95.0, 110.0, 110.0), // Low 95 is adverse
        create_candle(110.0, 105.0, 115.0, 115.0),
    ];

    // Bearish horizon: adverse = highs
    let horizon_bearish = vec![
        create_candle(105.0, 95.0, 115.0, 95.0), // High 115 is adverse
        create_candle(95.0, 90.0, 100.0, 90.0),
    ];

    let params = StopLevelParams {
        bandwidth: 1.0,
        percentiles: [0.1, 0.9],
        neutral_band_factor: 0.4,
        momentum_factor: 1.0,
        balance: Default::default(),
    };

    let (class_bull, _) =
        classify_stop_level_with_calibrated_params(&sequence, &horizon_bullish, &params)
            .expect("Classification failed");
    let (class_bear, _) =
        classify_stop_level_with_calibrated_params(&sequence, &horizon_bearish, &params)
            .expect("Classification failed");

    // Both should detect high risk (different adverse directions)
    assert!(
        class_bull <= 2,
        "Bullish adverse should be high risk, got class {}",
        class_bull
    );
    assert!(
        class_bear <= 2,
        "Bearish adverse should be high risk, got class {}",
        class_bear
    );
}

#[test]
fn test_adverse_exponential_weighting() {
    // Test that recent adverse prices are weighted more heavily
    let sequence = create_test_sequence(100.0, 110.0, 10);

    // Both have same worst adverse, but at different positions
    let horizon_early_adverse = vec![
        create_candle(105.0, 95.0, 105.0, 100.0), // Early adverse
        create_candle(100.0, 100.0, 110.0, 110.0),
        create_candle(110.0, 105.0, 115.0, 115.0),
    ];

    let horizon_late_adverse = vec![
        create_candle(105.0, 105.0, 110.0, 110.0),
        create_candle(110.0, 105.0, 112.0, 110.0),
        create_candle(110.0, 95.0, 115.0, 115.0), // Late adverse (weighted more)
    ];

    let params = StopLevelParams {
        bandwidth: 1.0,
        percentiles: [0.1, 0.9],
        neutral_band_factor: 0.4,
        momentum_factor: 1.0,
        balance: Default::default(),
    };

    let (class_early, _) =
        classify_stop_level_with_calibrated_params(&sequence, &horizon_early_adverse, &params)
            .expect("Classification failed");
    let (class_late, _) =
        classify_stop_level_with_calibrated_params(&sequence, &horizon_late_adverse, &params)
            .expect("Classification failed");

    // Both have same worst adverse, but late adverse gets more weight
    assert!(
        class_late <= class_early,
        "Recent adverse movements should be weighted more heavily"
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

    let reconstruction = reconstruct_stop_levels(&probabilities, &sequence, 105.0, &params, None)
        .expect("Reconstruction failed");

    // Verify structure
    assert_eq!(reconstruction.probabilities.len(), 5);
    assert_eq!(reconstruction.adverse_price_ranges.len(), 5);
    assert!(reconstruction.confidence > 0.0 && reconstruction.confidence <= 1.0);
    assert!(reconstruction.reference_price > 0.0);
    // No direction probs → defaults to bullish (legacy behavior).
    assert!(reconstruction.is_bullish);
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

    let reconstruction = reconstruct_stop_levels(&probabilities, &sequence, 105.0, &params, None)
        .expect("Reconstruction failed");

    // Most likely class should be 4 (highest probability)
    assert_eq!(reconstruction.most_likely_class, 4);

    // For bullish: Class 0 has lowest prices (deepest dip), Class 4 has highest (shallowest)
    // So adverse_price_ranges[0] should have lower values than adverse_price_ranges[4]
    assert!(
        reconstruction.adverse_price_ranges[0][1] < reconstruction.adverse_price_ranges[4][0],
        "Class 0 (extreme) should have lower price range than Class 4 (minimal)"
    );
}

#[test]
fn test_reconstruction_direction_awareness() {
    let sequence = create_test_sequence(100.0, 110.0, 10);
    let probabilities = vec![0.2, 0.2, 0.2, 0.2, 0.2];

    let params = StopLevelParams {
        bandwidth: 1.0,
        percentiles: [0.1, 0.9],
        neutral_band_factor: 0.4,
        momentum_factor: 1.0,
        balance: Default::default(),
    };

    // Bullish direction probs: UP+PUMP dominant
    let bull_dir = vec![0.05, 0.05, 0.10, 0.40, 0.40];
    let bull = reconstruct_stop_levels(&probabilities, &sequence, 105.0, &params, Some(&bull_dir))
        .expect("Bullish reconstruction failed");
    assert!(bull.is_bullish);

    // Bearish direction probs: DUMP+DOWN dominant
    let bear_dir = vec![0.40, 0.40, 0.10, 0.05, 0.05];
    let bear = reconstruct_stop_levels(&probabilities, &sequence, 105.0, &params, Some(&bear_dir))
        .expect("Bearish reconstruction failed");
    assert!(!bear.is_bullish);

    // Bullish class 0 is the deepest dip — should be BELOW the bearish class 0 (a high bounce).
    let bull_class0_mid = (bull.adverse_price_ranges[0][0] + bull.adverse_price_ranges[0][1]) / 2.0;
    let bear_class0_mid = (bear.adverse_price_ranges[0][0] + bear.adverse_price_ranges[0][1]) / 2.0;
    assert!(
        bull_class0_mid < bear_class0_mid,
        "Bullish extreme (deep dip) midpoint {} should be below bearish extreme (high bounce) midpoint {}",
        bull_class0_mid,
        bear_class0_mid
    );

    // Every class range must be non-degenerate (width > 0).
    for (i, [lo, hi]) in bull.adverse_price_ranges.iter().enumerate() {
        assert!(
            hi > lo,
            "Bullish class {} range [{}, {}] is non-positive",
            i,
            lo,
            hi
        );
    }
    for (i, [lo, hi]) in bear.adverse_price_ranges.iter().enumerate() {
        assert!(
            hi > lo,
            "Bearish class {} range [{}, {}] is non-positive",
            i,
            lo,
            hi
        );
    }
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
