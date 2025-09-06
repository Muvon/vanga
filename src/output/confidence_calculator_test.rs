//! Comprehensive tests for unified confidence calculation system
//!
//! Tests validate the research-based calibration function with realistic probability distributions
//! and ensure consistent confidence scores across all target types.

use crate::output::confidence_calculator::calibrate_5_class_confidence;

#[test]
fn test_unified_calibration_realistic_probabilities() {
    // Test realistic probability ranges based on neural network research

    // POOR MODEL (barely better than random)
    let poor_max_prob = 0.22;
    let poor_confidence = calibrate_5_class_confidence(poor_max_prob);
    assert!(
        (0.25..=0.45).contains(&poor_confidence),
        "Poor model (max_prob={}) should have low confidence, got {}",
        poor_max_prob,
        poor_confidence
    );

    // TYPICAL GOOD MODEL (common in well-calibrated models)
    let typical_max_prob = 0.35;
    let typical_confidence = calibrate_5_class_confidence(typical_max_prob);
    assert!(
        (0.65..=0.75).contains(&typical_confidence),
        "Typical good model (max_prob={}) should have moderate confidence, got {}",
        typical_max_prob,
        typical_confidence
    );

    // EXCELLENT MODEL (top 15% of predictions)
    let excellent_max_prob = 0.42;
    let excellent_confidence = calibrate_5_class_confidence(excellent_max_prob);
    assert!(
        (0.75..=0.85).contains(&excellent_confidence),
        "Excellent model (max_prob={}) should have high confidence, got {}",
        excellent_max_prob,
        excellent_confidence
    );

    // EXCEPTIONAL MODEL (very rare, top 5%)
    let exceptional_max_prob = 0.48;
    let exceptional_confidence = calibrate_5_class_confidence(exceptional_max_prob);
    assert!(
        (0.82..=0.88).contains(&exceptional_confidence),
        "Exceptional model (max_prob={}) should have very high confidence, got {}",
        exceptional_max_prob,
        exceptional_confidence
    );

    // POTENTIALLY OVERFITTED (suspicious if common)
    let overfitted_max_prob = 0.65;
    let overfitted_confidence = calibrate_5_class_confidence(overfitted_max_prob);
    assert!(
        (0.89..=0.95).contains(&overfitted_confidence),
        "Potentially overfitted model (max_prob={}) should have very high but capped confidence, got {}",
        overfitted_max_prob, overfitted_confidence
    );
}

#[test]
fn test_calibration_monotonicity() {
    // Confidence should increase monotonically with probability
    let probabilities = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70];
    let mut prev_confidence = 0.0;

    for &prob in &probabilities {
        let confidence = calibrate_5_class_confidence(prob);
        assert!(
            confidence > prev_confidence,
            "Confidence should increase monotonically: prob={}, confidence={}, prev={}",
            prob,
            confidence,
            prev_confidence
        );
        prev_confidence = confidence;
    }
}

#[test]
fn test_calibration_boundary_conditions() {
    // Test edge cases and boundary conditions

    // Random baseline (uniform distribution)
    let random_confidence = calibrate_5_class_confidence(0.20);
    assert!(
        (0.25..=0.30).contains(&random_confidence),
        "Random baseline should have minimal confidence, got {}",
        random_confidence
    );

    // Below random (edge case) - the function handles this by scaling up
    let below_random_confidence = calibrate_5_class_confidence(0.15);
    assert!(
        (0.20..=0.35).contains(&below_random_confidence),
        "Below random should still yield reasonable confidence due to scaling, got {}",
        below_random_confidence
    );

    // Perfect prediction (theoretical maximum)
    let perfect_confidence = calibrate_5_class_confidence(1.0);
    assert!(
        (0.95..=1.0).contains(&perfect_confidence),
        "Perfect prediction should have maximum confidence, got {}",
        perfect_confidence
    );

    // Zero probability (edge case)
    let zero_confidence = calibrate_5_class_confidence(0.0);
    assert!(
        (0.0..=0.1).contains(&zero_confidence),
        "Zero probability should have minimal confidence, got {}",
        zero_confidence
    );
}

#[test]
fn test_realistic_probability_distributions() {
    // Test with realistic probability arrays that LSTM models actually produce

    // Typical good model output
    let typical_probs = [0.15, 0.18, 0.22, 0.35, 0.10];
    let max_prob = typical_probs.iter().fold(0.0_f64, |a, &b| a.max(b));
    let confidence = calibrate_5_class_confidence(max_prob);
    assert!(
        (0.65..=0.75).contains(&confidence),
        "Typical distribution should yield moderate-high confidence, got {}",
        confidence
    );

    // Excellent model output (rare)
    let excellent_probs = [0.08, 0.12, 0.15, 0.42, 0.23];
    let max_prob = excellent_probs.iter().fold(0.0_f64, |a, &b| a.max(b));
    let confidence = calibrate_5_class_confidence(max_prob);
    assert!(
        (0.75..=0.85).contains(&confidence),
        "Excellent distribution should yield high confidence, got {}",
        confidence
    );

    // Poor model output (barely better than random)
    let poor_probs = [0.18, 0.19, 0.21, 0.22, 0.20];
    let max_prob = poor_probs.iter().fold(0.0_f64, |a, &b| a.max(b));
    let confidence = calibrate_5_class_confidence(max_prob);
    assert!(
        (0.25..=0.45).contains(&confidence),
        "Poor distribution should yield low confidence, got {}",
        confidence
    );

    // Uniform distribution (worst case)
    let uniform_probs = [0.20, 0.20, 0.20, 0.20, 0.20];
    let max_prob = uniform_probs.iter().fold(0.0_f64, |a, &b| a.max(b));
    let confidence = calibrate_5_class_confidence(max_prob);
    assert!(
        (0.25..=0.35).contains(&confidence),
        "Uniform distribution should yield minimal confidence, got {}",
        confidence
    );
}

#[test]
fn test_confidence_ranges() {
    // Test that confidence values stay within reasonable ranges for crypto trading
    let test_probabilities = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80];

    for &prob in &test_probabilities {
        let confidence = calibrate_5_class_confidence(prob);

        // All confidence values should be between 0.2 and 0.95 (conservative for crypto)
        assert!(
            (0.20..=0.95).contains(&confidence),
            "Confidence for prob={} should be in range [0.20, 0.95], got {}",
            prob,
            confidence
        );

        // Confidence should never be lower than the probability itself (sanity check)
        if prob >= 0.20 {
            assert!(
                confidence >= prob,
                "Confidence ({}) should not be lower than probability ({}) for well-calibrated models",
                confidence, prob
            );
        }
    }
}

#[test]
fn test_research_alignment() {
    // Test alignment with neural network calibration research findings

    // Research shows that max_prob of 0.35 is actually excellent performance
    let research_excellent = 0.35;
    let confidence = calibrate_5_class_confidence(research_excellent);
    assert!(
        confidence >= 0.65,
        "Research-based excellent performance (0.35) should have high confidence, got {}",
        confidence
    );

    // Research shows that max_prob > 0.5 is rare and potentially indicates overfitting
    let potentially_overfit = 0.55;
    let confidence = calibrate_5_class_confidence(potentially_overfit);
    assert!(
        confidence <= 0.90,
        "Potentially overfitted model (0.55) should have capped confidence, got {}",
        confidence
    );

    // Research shows that well-calibrated models rarely exceed 0.4-0.5 max probability
    let realistic_good = 0.40;
    let confidence = calibrate_5_class_confidence(realistic_good);
    assert!(
        (0.70..=0.80).contains(&confidence),
        "Realistic good model (0.40) should have solid confidence, got {}",
        confidence
    );
}

#[test]
fn test_crypto_trading_appropriateness() {
    // Test that confidence levels are appropriate for crypto trading decisions

    // Low confidence should discourage trading
    let low_prob = 0.25;
    let low_confidence = calibrate_5_class_confidence(low_prob);
    assert!(
        low_confidence <= 0.50,
        "Low probability should yield low confidence for risk management, got {}",
        low_confidence
    );

    // Moderate confidence should allow cautious trading
    let moderate_prob = 0.35;
    let moderate_confidence = calibrate_5_class_confidence(moderate_prob);
    assert!(
        (0.60..=0.75).contains(&moderate_confidence),
        "Moderate probability should yield moderate confidence for cautious trading, got {}",
        moderate_confidence
    );

    // High confidence should enable more aggressive trading
    let high_prob = 0.45;
    let high_confidence = calibrate_5_class_confidence(high_prob);
    assert!(
        (0.80..=0.90).contains(&high_confidence),
        "High probability should yield high confidence for aggressive trading, got {}",
        high_confidence
    );

    // Very high confidence should be rare and capped to prevent overconfidence
    let very_high_prob = 0.65;
    let very_high_confidence = calibrate_5_class_confidence(very_high_prob);
    assert!(
        very_high_confidence <= 0.95,
        "Very high probability should be capped to prevent dangerous overconfidence, got {}",
        very_high_confidence
    );
}
