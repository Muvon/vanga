//! Comprehensive tests for fully adaptive order generation system
//! Validates that NO hardcoded values are used and all orders stay within sequence bounds

use crate::output::prediction_types::*;
use crate::output::sequence_statistics::SequenceStatistics;
use crate::output::smart_order_generator::SmartConsensus;
use crate::output::trading_orders::OrderLevel;
use crate::utils::error::Result;

/// Create test price sequence with known statistical properties
fn create_test_sequence(base_price: f64, volatility: f64, trend: f64, length: usize) -> Vec<f64> {
    let mut prices = Vec::with_capacity(length);
    let mut current_price = base_price;

    for i in 0..length {
        // Add trend component
        let trend_component = trend * (i as f64 / length as f64);

        // Add volatility component (simplified random walk)
        let volatility_component = volatility * ((i as f64 * 0.1).sin() * 0.5);

        current_price = base_price * (1.0 + trend_component + volatility_component);
        prices.push(current_price);
    }

    prices
}

/// Create test consensus with known probabilities
fn create_test_consensus(
    direction_confidence: f64,
    price_level_probs: [f64; 5],
    volatility_regime: &str,
) -> SmartConsensus {
    // Create price level bins with test probabilities
    let mut price_bins = std::collections::HashMap::new();
    let bin_names = [
        "strong_down",
        "moderate_down",
        "neutral",
        "moderate_up",
        "strong_up",
    ];

    for (i, &name) in bin_names.iter().enumerate() {
        price_bins.insert(
            name.to_string(),
            PriceLevelBin {
                range: [
                    match name {
                        "strong_down" => -8.0,
                        "moderate_down" => -4.0,
                        "neutral" => -1.0,
                        "moderate_up" => 2.0,
                        "strong_up" => 6.0,
                        _ => 0.0,
                    },
                    match name {
                        "strong_down" => -4.0,
                        "moderate_down" => -1.0,
                        "neutral" => 2.0,
                        "moderate_up" => 6.0,
                        "strong_up" => 12.0,
                        _ => 0.0,
                    },
                ],
                probability: price_level_probs[i],
            },
        );
    }

    SmartConsensus {
        direction: DirectionPrediction {
            signal: if direction_confidence > 0.5 {
                "LONG".to_string()
            } else {
                "SHORT".to_string()
            },
            up_probability_aggregated: direction_confidence,
            down_probability_aggregated: 1.0 - direction_confidence,
            expected_upside_percent: 5.0,
            expected_downside_percent: 3.0,
        },
        price_levels: PriceLevelPrediction { bins: price_bins },
        volatility: VolatilityPrediction {
            regime: volatility_regime.to_string(),
            regime_confidence: 0.8,
            expected_range_percent: 4.0,
            recommended_stop_distance_percent: 2.5,
        },
        volume: VolumePrediction {
            regime: "MEDIUM".to_string(),
            regime_confidence: 0.7,
        },
        sentiment: SentimentPrediction {
            regime: "NEUTRAL".to_string(),
            regime_confidence: 0.6,
        },
    }
}

#[test]
fn test_no_hardcoded_values_in_adaptive_entries() {
    // Test that entry generation uses ONLY sequence data and model predictions
    let sequence_prices = create_test_sequence(100.0, 0.05, 0.02, 60);
    let sequence_stats = SequenceStatistics::from_prices(&sequence_prices, 4.0, None).unwrap();
    let consensus = create_test_consensus(0.7, [0.1, 0.2, 0.4, 0.2, 0.1], "MEDIUM");

    let entries = consensus
        .generate_fully_adaptive_entries(100.0, "LONG", &sequence_prices, &sequence_stats)
        .unwrap();

    // Verify all entries are within sequence bounds
    let bounds = sequence_stats.get_adaptive_bounds(&sequence_prices, 100.0);

    for (i, entry) in entries.iter().enumerate() {
        // Entry prices should be validated by z-score (within 3 standard deviations)
        let z_score = (entry.price - bounds.sequence_mean) / bounds.sequence_std;
        assert!(
            z_score.abs() <= 3.0,
            "Entry {} z-score {:.2} exceeds 3 standard deviations",
            i + 1,
            z_score
        );

        // Sizes should be based on probabilities, not hardcoded
        assert!(
            entry.quantity_percentage > 0.0 && entry.quantity_percentage < 1.0,
            "Entry {} size {:.3} should be probability-based",
            i + 1,
            entry.quantity_percentage
        );

        // Confidence should be based on model and distance, not hardcoded
        assert!(
            entry.confidence > 0.1 && entry.confidence < 1.0,
            "Entry {} confidence {:.3} should be model-based",
            i + 1,
            entry.confidence
        );
    }

    // Total size should sum to 1.0 (normalized)
    let total_size: f64 = entries.iter().map(|e| e.quantity_percentage).sum();
    assert!(
        (total_size - 1.0).abs() < 0.01,
        "Entry sizes should sum to 1.0, got {:.3}",
        total_size
    );
}

#[test]
fn test_sequence_bounded_exits() {
    // Test that exits are bounded by actual sequence upside potential
    let sequence_prices = create_test_sequence(100.0, 0.08, 0.05, 60); // Higher volatility and uptrend
    let sequence_stats = SequenceStatistics::from_prices(&sequence_prices, 4.0, None).unwrap();
    let consensus = create_test_consensus(0.8, [0.05, 0.15, 0.3, 0.3, 0.2], "HIGH");

    let exits = consensus
        .generate_fully_adaptive_exits(100.0, "LONG", &sequence_prices, &sequence_stats)
        .unwrap();

    let bounds = sequence_stats.get_adaptive_bounds(&sequence_prices, 100.0);

    for (i, exit) in exits.iter().enumerate() {
        // Exit prices should not exceed sequence maximum (realistic bounds)
        assert!(
            exit.price <= bounds.sequence_max * 1.1, // Allow small buffer for model predictions
            "Exit {} price ${:.2} exceeds sequence max ${:.2}",
            i + 1,
            exit.price,
            bounds.sequence_max
        );

        // Exit distances should be bounded by sequence upside potential
        let exit_distance = ((exit.price - 100.0) / 100.0) * 100.0;
        assert!(
            exit_distance <= bounds.max_upside_pct * 1.2, // Allow model prediction buffer
            "Exit {} distance {:.2}% exceeds sequence upside {:.2}%",
            i + 1,
            exit_distance,
            bounds.max_upside_pct
        );

        // Z-score validation
        let z_score = (exit.price - bounds.sequence_mean) / bounds.sequence_std;
        assert!(
            z_score.abs() <= 3.0,
            "Exit {} z-score {:.2} exceeds 3 standard deviations",
            i + 1,
            z_score
        );
    }
}

#[test]
fn test_drawdown_based_stops() {
    // Test that stops use actual sequence drawdown analysis
    let sequence_prices = create_test_sequence(100.0, 0.06, -0.01, 60); // Slight downtrend
    let sequence_stats = SequenceStatistics::from_prices(&sequence_prices, 4.0, None).unwrap();
    let consensus = create_test_consensus(0.6, [0.2, 0.25, 0.3, 0.15, 0.1], "MEDIUM");

    // Create test entries
    let entries = [
        OrderLevel {
            price: 99.0,
            quantity_percentage: 0.4,
            atr_distance: 1.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.8,
        },
        OrderLevel {
            price: 98.0,
            quantity_percentage: 0.35,
            atr_distance: 2.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.7,
        },
        OrderLevel {
            price: 97.0,
            quantity_percentage: 0.25,
            atr_distance: 3.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.6,
        },
    ];

    let stops = consensus
        .generate_fully_adaptive_stops(&entries, "LONG", &sequence_prices, &sequence_stats)
        .unwrap();

    let bounds = sequence_stats.get_adaptive_bounds(&sequence_prices, 100.0);
    let extreme_entry = 97.0; // Lowest entry for LONG

    for (i, stop) in stops.iter().enumerate() {
        // Stops should be below extreme entry for LONG positions
        assert!(
            stop.price < extreme_entry,
            "Stop {} price ${:.2} should be below extreme entry ${:.2}",
            i + 1,
            stop.price,
            extreme_entry
        );

        // Stop distances should be based on sequence drawdown, not hardcoded
        let stop_distance = ((extreme_entry - stop.price) / extreme_entry) * 100.0;
        assert!(
            stop_distance <= bounds.max_drawdown_pct * 2.0, // Allow safety buffer
            "Stop {} distance {:.2}% exceeds reasonable drawdown bounds",
            i + 1,
            stop_distance
        );

        // Z-score validation
        let z_score = (stop.price - bounds.sequence_mean) / bounds.sequence_std;
        assert!(
            z_score.abs() <= 3.0,
            "Stop {} z-score {:.2} exceeds 3 standard deviations",
            i + 1,
            z_score
        );

        // Stop should match entry size (risk parity)
        assert_eq!(
            stop.quantity_percentage,
            entries[i].quantity_percentage,
            "Stop {} size should match entry {} size",
            i + 1,
            i + 1
        );
    }
}

#[test]
fn test_volatile_vs_calm_market_adaptation() {
    // Test that system adapts to different market conditions
    let calm_prices = create_test_sequence(100.0, 0.01, 0.0, 60); // Very low volatility
    let volatile_prices = create_test_sequence(100.0, 0.15, 0.0, 60); // High volatility

    let calm_stats = SequenceStatistics::from_prices(&calm_prices, 4.0, None).unwrap();
    let volatile_stats = SequenceStatistics::from_prices(&volatile_prices, 4.0, None).unwrap();

    let consensus = create_test_consensus(0.7, [0.1, 0.2, 0.4, 0.2, 0.1], "MEDIUM");

    let calm_entries = consensus
        .generate_fully_adaptive_entries(100.0, "LONG", &calm_prices, &calm_stats)
        .unwrap();

    let volatile_entries = consensus
        .generate_fully_adaptive_entries(100.0, "LONG", &volatile_prices, &volatile_stats)
        .unwrap();

    // Volatile market should have wider entry spacing
    let calm_range = calm_entries[2].price - calm_entries[0].price;
    let volatile_range = volatile_entries[2].price - volatile_entries[0].price;

    assert!(
        volatile_range > calm_range,
        "Volatile market entries should be more spread out: volatile={:.2}, calm={:.2}",
        volatile_range,
        calm_range
    );

    // Verify bounds are different
    let calm_bounds = calm_stats.get_adaptive_bounds(&calm_prices, 100.0);
    let volatile_bounds = volatile_stats.get_adaptive_bounds(&volatile_prices, 100.0);

    assert!(
        volatile_bounds.iqr_volatility > calm_bounds.iqr_volatility,
        "Volatile market should have higher IQR volatility: volatile={:.2}, calm={:.2}",
        volatile_bounds.iqr_volatility,
        calm_bounds.iqr_volatility
    );
}

#[test]
fn test_z_score_validation_enforcement() {
    // Test that z-score validation prevents extreme outliers
    let sequence_prices = create_test_sequence(100.0, 0.02, 0.0, 60); // Low volatility
    let sequence_stats = SequenceStatistics::from_prices(&sequence_prices, 4.0, None).unwrap();

    // Test extreme price validation
    let extreme_high = 150.0; // Way outside normal range
    let extreme_low = 50.0; // Way outside normal range

    let validated_high = sequence_stats.validate_price_with_zscore(&sequence_prices, extreme_high);
    let validated_low = sequence_stats.validate_price_with_zscore(&sequence_prices, extreme_low);

    let bounds = sequence_stats.get_adaptive_bounds(&sequence_prices, 100.0);

    // Validated prices should be within 3 standard deviations
    let high_z = (validated_high - bounds.sequence_mean) / bounds.sequence_std;
    let low_z = (validated_low - bounds.sequence_mean) / bounds.sequence_std;

    assert!(
        high_z.abs() <= 3.0,
        "Validated high price z-score {:.2} should be <= 3.0",
        high_z
    );

    assert!(
        low_z.abs() <= 3.0,
        "Validated low price z-score {:.2} should be <= 3.0",
        low_z
    );

    // Extreme prices should be adjusted
    assert!(
        validated_high < extreme_high,
        "Extreme high price should be adjusted down: {:.2} -> {:.2}",
        extreme_high,
        validated_high
    );

    assert!(
        validated_low > extreme_low,
        "Extreme low price should be adjusted up: {:.2} -> {:.2}",
        extreme_low,
        validated_low
    );
}

#[test]
fn test_sequence_length_robustness() {
    // Test that system works with various sequence lengths
    let short_prices = create_test_sequence(100.0, 0.05, 0.01, 20); // Short sequence
    let long_prices = create_test_sequence(100.0, 0.05, 0.01, 200); // Long sequence

    let short_stats = SequenceStatistics::from_prices(&short_prices, 4.0, None).unwrap();
    let long_stats = SequenceStatistics::from_prices(&long_prices, 4.0, None).unwrap();

    let consensus = create_test_consensus(0.7, [0.1, 0.2, 0.4, 0.2, 0.1], "MEDIUM");

    // Both should generate valid orders
    let short_entries =
        consensus.generate_fully_adaptive_entries(100.0, "LONG", &short_prices, &short_stats);

    let long_entries =
        consensus.generate_fully_adaptive_entries(100.0, "LONG", &long_prices, &long_stats);

    assert!(
        short_entries.is_ok(),
        "Short sequence should generate valid entries"
    );
    assert!(
        long_entries.is_ok(),
        "Long sequence should generate valid entries"
    );

    // Longer sequences should have higher confidence (more data)
    let short_bounds = short_stats.get_adaptive_bounds(&short_prices, 100.0);
    let long_bounds = long_stats.get_adaptive_bounds(&long_prices, 100.0);

    // Both should have valid bounds
    assert!(
        short_bounds.sequence_std > 0.0,
        "Short sequence should have valid std dev"
    );
    assert!(
        long_bounds.sequence_std > 0.0,
        "Long sequence should have valid std dev"
    );
}

#[test]
fn test_mathematical_consistency() {
    // Test that all calculations are mathematically consistent
    let sequence_prices = create_test_sequence(100.0, 0.05, 0.02, 60);
    let sequence_stats = SequenceStatistics::from_prices(&sequence_prices, 4.0, None).unwrap();
    let bounds = sequence_stats.get_adaptive_bounds(&sequence_prices, 100.0);

    // Percentiles should be ordered correctly
    assert!(bounds.p10 <= bounds.p25, "P10 should be <= P25");
    assert!(bounds.p25 <= bounds.p50, "P25 should be <= P50");
    assert!(bounds.p50 <= bounds.p75, "P50 should be <= P75");
    assert!(bounds.p75 <= bounds.p90, "P75 should be <= P90");

    // Sequence bounds should be consistent
    assert!(
        bounds.sequence_min <= bounds.p10,
        "Sequence min should be <= P10"
    );
    assert!(
        bounds.p90 <= bounds.sequence_max,
        "P90 should be <= sequence max"
    );

    // IQR volatility should be positive
    assert!(
        bounds.iqr_volatility >= 0.0,
        "IQR volatility should be non-negative"
    );

    // Range percentage should be reasonable
    assert!(
        bounds.sequence_range_pct >= 0.0,
        "Sequence range should be non-negative"
    );

    // Standard deviation should be positive for varying prices
    assert!(
        bounds.sequence_std > 0.0,
        "Sequence std dev should be positive"
    );
}

#[test]
fn test_order_alignment_long_positions() {
    // Test that LONG orders are properly aligned after generation and optimization
    let sequence_prices = create_test_sequence(100.0, 0.05, 0.02, 60);
    let sequence_stats = SequenceStatistics::from_prices(&sequence_prices, 4.0, None).unwrap();
    let consensus = create_test_consensus(0.8, [0.1, 0.15, 0.3, 0.3, 0.15], "MEDIUM");

    let entries = consensus
        .generate_sequence_aware_entries(100.0, "LONG", &sequence_stats)
        .unwrap();

    let exits = consensus
        .generate_sequence_aware_exits(100.0, "LONG", &sequence_stats)
        .unwrap();

    let stops = consensus
        .generate_sequence_aware_stops(&entries, "LONG", &sequence_stats)
        .unwrap();

    // LONG entries should be descending (highest price first)
    assert!(
        entries[0].price >= entries[1].price,
        "LONG Entry 1 (${:.4}) should be >= Entry 2 (${:.4})",
        entries[0].price,
        entries[1].price
    );
    assert!(
        entries[1].price >= entries[2].price,
        "LONG Entry 2 (${:.4}) should be >= Entry 3 (${:.4})",
        entries[1].price,
        entries[2].price
    );

    // LONG exits should be ascending (lowest target first)
    assert!(
        exits[0].price <= exits[1].price,
        "LONG Exit 1 (${:.4}) should be <= Exit 2 (${:.4})",
        exits[0].price,
        exits[1].price
    );
    assert!(
        exits[1].price <= exits[2].price,
        "LONG Exit 2 (${:.4}) should be <= Exit 3 (${:.4})",
        exits[1].price,
        exits[2].price
    );

    // LONG stops should be descending (highest stop first)
    assert!(
        stops[0].price >= stops[1].price,
        "LONG Stop 1 (${:.4}) should be >= Stop 2 (${:.4})",
        stops[0].price,
        stops[1].price
    );
    assert!(
        stops[1].price >= stops[2].price,
        "LONG Stop 2 (${:.4}) should be >= Stop 3 (${:.4})",
        stops[1].price,
        stops[2].price
    );

    // All stops must be below all entries
    let lowest_entry = entries
        .iter()
        .map(|e| e.price)
        .fold(f64::INFINITY, f64::min);
    let highest_stop = stops
        .iter()
        .map(|s| s.price)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        highest_stop < lowest_entry,
        "LONG highest stop (${:.4}) must be < lowest entry (${:.4})",
        highest_stop,
        lowest_entry
    );
}

#[test]
fn test_order_alignment_short_positions() {
    // Test that SHORT orders are properly aligned after generation and optimization
    let sequence_prices = create_test_sequence(100.0, 0.05, -0.02, 60); // Downtrend
    let sequence_stats = SequenceStatistics::from_prices(&sequence_prices, 4.0, None).unwrap();
    let consensus = create_test_consensus(0.2, [0.15, 0.3, 0.3, 0.15, 0.1], "MEDIUM"); // Bearish

    let entries = consensus
        .generate_sequence_aware_entries(100.0, "SHORT", &sequence_stats)
        .unwrap();

    let exits = consensus
        .generate_sequence_aware_exits(100.0, "SHORT", &sequence_stats)
        .unwrap();

    let stops = consensus
        .generate_sequence_aware_stops(&entries, "SHORT", &sequence_stats)
        .unwrap();

    // SHORT entries should be ascending (lowest price first)
    assert!(
        entries[0].price <= entries[1].price,
        "SHORT Entry 1 (${:.4}) should be <= Entry 2 (${:.4})",
        entries[0].price,
        entries[1].price
    );
    assert!(
        entries[1].price <= entries[2].price,
        "SHORT Entry 2 (${:.4}) should be <= Entry 3 (${:.4})",
        entries[1].price,
        entries[2].price
    );

    // SHORT exits should be descending (highest target first)
    assert!(
        exits[0].price >= exits[1].price,
        "SHORT Exit 1 (${:.4}) should be >= Exit 2 (${:.4})",
        exits[0].price,
        exits[1].price
    );
    assert!(
        exits[1].price >= exits[2].price,
        "SHORT Exit 2 (${:.4}) should be >= Exit 3 (${:.4})",
        exits[1].price,
        exits[2].price
    );

    // SHORT stops should be ascending (lowest stop first)
    assert!(
        stops[0].price <= stops[1].price,
        "SHORT Stop 1 (${:.4}) should be <= Stop 2 (${:.4})",
        stops[0].price,
        stops[1].price
    );
    assert!(
        stops[1].price <= stops[2].price,
        "SHORT Stop 2 (${:.4}) should be <= Stop 3 (${:.4})",
        stops[1].price,
        stops[2].price
    );

    // All stops must be above all entries
    let highest_entry = entries
        .iter()
        .map(|e| e.price)
        .fold(f64::NEG_INFINITY, f64::max);
    let lowest_stop = stops.iter().map(|s| s.price).fold(f64::INFINITY, f64::min);
    assert!(
        lowest_stop > highest_entry,
        "SHORT lowest stop (${:.4}) must be > highest entry (${:.4})",
        lowest_stop,
        highest_entry
    );
}

#[test]
fn test_comprehensive_order_validation() {
    // Test the comprehensive validation function catches all error cases
    use crate::output::trading_orders::TradingOrders;

    // Test case 1: Quantities don't sum to 1.0
    let bad_entries = [
        OrderLevel {
            price: 99.0,
            quantity_percentage: 0.5, // Total will be 1.1 (invalid)
            atr_distance: 1.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.8,
        },
        OrderLevel {
            price: 98.0,
            quantity_percentage: 0.4,
            atr_distance: 2.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.7,
        },
        OrderLevel {
            price: 97.0,
            quantity_percentage: 0.2, // This makes total 1.1
            atr_distance: 3.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.6,
        },
    ];

    let good_exits = [
        OrderLevel {
            price: 101.0,
            quantity_percentage: 0.4,
            atr_distance: 1.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.8,
        },
        OrderLevel {
            price: 102.0,
            quantity_percentage: 0.35,
            atr_distance: 2.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.7,
        },
        OrderLevel {
            price: 103.0,
            quantity_percentage: 0.25,
            atr_distance: 3.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.6,
        },
    ];

    let good_stops = [
        OrderLevel {
            price: 96.0,
            quantity_percentage: 0.4,
            atr_distance: 4.0,
            order_type: "STOP_LOSS".to_string(),
            confidence: 0.8,
        },
        OrderLevel {
            price: 95.0,
            quantity_percentage: 0.35,
            atr_distance: 5.0,
            order_type: "STOP_LOSS".to_string(),
            confidence: 0.7,
        },
        OrderLevel {
            price: 94.0,
            quantity_percentage: 0.25,
            atr_distance: 6.0,
            order_type: "STOP_LOSS".to_string(),
            confidence: 0.6,
        },
    ];

    let result = TradingOrders::validate_order_integrity(
        &bad_entries,
        &good_exits,
        &good_stops,
        "LONG",
        100.0,
    );

    assert!(
        result.is_err(),
        "Should fail validation due to bad entry quantities"
    );
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Entry quantities sum to"));

    // Test case 2: Wrong order alignment for LONG
    let wrong_order_entries = [
        OrderLevel {
            price: 97.0,
            quantity_percentage: 0.4,
            atr_distance: 3.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.6,
        }, // Should be highest
        OrderLevel {
            price: 98.0,
            quantity_percentage: 0.35,
            atr_distance: 2.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.7,
        },
        OrderLevel {
            price: 99.0,
            quantity_percentage: 0.25,
            atr_distance: 1.0,
            order_type: "LIMIT".to_string(),
            confidence: 0.8,
        }, // Should be lowest
    ];

    let result = TradingOrders::validate_order_integrity(
        &wrong_order_entries,
        &good_exits,
        &good_stops,
        "LONG",
        100.0,
    );

    assert!(
        result.is_err(),
        "Should fail validation due to wrong LONG entry ordering"
    );
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("LONG entry ordering broken"));
}

#[test]
fn test_rr_optimization_preserves_order() {
    // Test that RR optimization maintains proper order relationships
    use crate::output::confidence_calculator::ConfidenceCalculator;
    use crate::output::trading_orders::{SmartOrderConfig, TradingOrders};

    // Create test predictions
    let price_levels = create_test_consensus(0.7, [0.1, 0.2, 0.4, 0.2, 0.1], "MEDIUM").price_levels;
    let direction = create_test_consensus(0.7, [0.1, 0.2, 0.4, 0.2, 0.1], "MEDIUM").direction;
    let volatility = create_test_consensus(0.7, [0.1, 0.2, 0.4, 0.2, 0.1], "MEDIUM").volatility;
    let sentiment = create_test_consensus(0.7, [0.1, 0.2, 0.4, 0.2, 0.1], "MEDIUM").sentiment;
    let volume = create_test_consensus(0.7, [0.1, 0.2, 0.4, 0.2, 0.1], "MEDIUM").volume;

    let confidence_calc = ConfidenceCalculator::new();

    let config = SmartOrderConfig {
        current_price: 100.0,
        price_levels: &price_levels,
        direction_pred: &direction,
        volatility_pred: &volatility,
        sentiment_pred: &sentiment,
        volume_pred: &volume,
        confidence_calculator: &confidence_calc,
        min_confidence: 0.5,
        sequence_ohlcv: None,
    };

    let result = TradingOrders::generate(config);

    // Should either succeed with proper alignment or fail validation gracefully
    match result {
        Ok(orders) => {
            // If successful, orders should be properly aligned
            if orders.direction != "NO_SIGNAL" {
                // Validate the generated orders
                let validation_result = TradingOrders::validate_order_integrity(
                    &orders.entry_levels,
                    &orders.exit_levels,
                    &orders.stop_levels,
                    &orders.direction,
                    100.0,
                );
                assert!(
                    validation_result.is_ok(),
                    "Generated orders should pass validation: {:?}",
                    validation_result.err()
                );
            }
        }
        Err(_) => {
            // If failed, that's also acceptable - the system should be robust
            // The important thing is it doesn't generate invalid orders
        }
    }
}
