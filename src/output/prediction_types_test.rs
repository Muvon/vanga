// src/output/prediction_types_test.rs
// Separate test file as per project rules (no inline #[cfg(test)] in source files)

use crate::output::prediction_types::DirectionPrediction;

#[test]
fn direction_from_probabilities_aggregates_and_predicts() {
    // Strong pump scenario
    let pred = DirectionPrediction::from_probabilities(0.05, 0.05, 0.1, 0.1, 0.7);

    // Aggregations
    assert!((pred.up_probability_aggregated - (0.1 + 0.7)).abs() < 1e-9);
    assert!((pred.down_probability_aggregated - (0.05 + 0.05)).abs() < 1e-9);
    assert!((pred.sideways_probability_aggregated - 0.1).abs() < 1e-9);

    // Most likely class
    assert_eq!(pred.prediction, "PUMP");

    // Confidence should be valid and within expected calibrated bounds
    assert!(pred.confidence.is_finite());
    assert!(pred.confidence >= 0.2 && pred.confidence <= 0.98);
}

#[test]
fn direction_balanced_probabilities_confidence_near_baseline() {
    // Uniform distribution across 5 classes
    let pred = DirectionPrediction::from_probabilities(0.2, 0.2, 0.2, 0.2, 0.2);

    // Confidence should be close to baseline, but we allow a safe range
    assert!(pred.confidence.is_finite());
    assert!(pred.confidence >= 0.2 && pred.confidence <= 0.4);
}

#[test]
fn direction_horizon_adaptive_metrics_update() {
    let mut pred = DirectionPrediction::from_probabilities(0.05, 0.15, 0.2, 0.3, 0.3);
    // bandwidth 4%, horizon label and seq length typical
    pred.calculate_horizon_adaptive_metrics(4.0, "4h".to_string(), 60);

    // Expected derived fields should be finite
    assert!(pred.expected_upside_percent.is_finite());
    assert!(pred.expected_downside_percent.is_finite());
    assert!(pred.risk_reward_ratio.is_finite());
    assert!(pred.breakout_probability.is_finite());

    // Non-negative checks where applicable
    assert!(pred.expected_upside_percent >= 0.0);
    assert!(pred.expected_downside_percent >= 0.0);
    assert!(pred.breakout_probability >= 0.0 && pred.breakout_probability <= 1.0);

    // Metadata filled
    assert_eq!(pred.training_horizon, "4h".to_string());
    assert_eq!(pred.sequence_length, 60);
    assert!((pred.sequence_bandwidth_percent - 4.0).abs() < 1e-9);
}
