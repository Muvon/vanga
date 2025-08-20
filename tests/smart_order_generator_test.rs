//! Tests for SMART order generation system

use std::collections::HashMap;
use vanga::output::prediction_types::*;
use vanga::output::smart_order_generator::SmartConsensus;
use vanga::output::trading_orders::TradingOrders;
use vanga::output::{OrderConfig, PriceBin};

/// Create test price levels with realistic bins
fn create_test_price_levels() -> PriceLevelPrediction {
    let mut bins = HashMap::new();

    // Realistic price bins for testing
    bins.insert(
        "strong_down".to_string(),
        PriceBin {
            range: [-1.591, -1.307],
            vwap_range: [-1.434, -1.150],
            price: [114377.50, 114707.86],
            probability: 0.190,
        },
    );

    bins.insert(
        "moderate_down".to_string(),
        PriceBin {
            range: [-1.307, -0.195],
            vwap_range: [-1.150, -0.036],
            price: [114707.98, 116000.65],
            probability: 0.201,
        },
    );

    bins.insert(
        "neutral".to_string(),
        PriceBin {
            range: [-0.195, 0.909],
            vwap_range: [-0.036, 1.070],
            price: [116000.77, 117283.73],
            probability: 0.221,
        },
    );

    bins.insert(
        "moderate_up".to_string(),
        PriceBin {
            range: [0.909, 2.021],
            vwap_range: [1.070, 2.184],
            price: [117283.85, 118576.52],
            probability: 0.198,
        },
    );

    bins.insert(
        "strong_up".to_string(),
        PriceBin {
            range: [2.022, 2.306],
            vwap_range: [2.184, 2.469],
            price: [118576.64, 118907.12],
            probability: 0.191,
        },
    );

    PriceLevelPrediction {
        bins,
        most_likely_range: [-0.195, 0.909],
        confidence: 0.221,
    }
}

/// Create test direction prediction
fn create_test_direction() -> DirectionPrediction {
    let mut pred = DirectionPrediction::from_probabilities(
        0.195, // dump
        0.231, // down
        0.206, // sideways
        0.186, // up
        0.181, // pump
    );

    pred.calculate_horizon_adaptive_metrics(
        0.560, // bandwidth
        "10h".to_string(),
        40,
    );

    pred
}

/// Create test volatility prediction
fn create_test_volatility() -> VolatilityPrediction {
    let mut pred = VolatilityPrediction::from_probabilities(
        0.178, // very_low
        0.186, // low
        0.200, // medium
        0.230, // high
        0.206, // very_high
    );

    pred.training_horizon = "10h".to_string();
    pred.expected_range_percent = 0.699;
    pred.volatility_percentile = 61.54;
    pred.recommended_stop_distance_percent = 0.419;
    pred.position_size_multiplier = 0.8;
    pred.regime_confidence = 0.230;
    pred.regime = "HIGH".to_string();

    pred
}

/// Create test sentiment prediction
fn create_test_sentiment() -> SentimentPrediction {
    let mut pred = SentimentPrediction::from_probabilities(
        0.184, // very_bearish
        0.195, // bearish
        0.202, // neutral
        0.218, // bullish
        0.200, // very_bullish
    );

    pred.training_horizon = "10h".to_string();
    pred.regime = "BULLISH".to_string();
    pred.confidence = 0.218;

    pred
}

/// Create test volume prediction
fn create_test_volume() -> VolumePrediction {
    let mut pred = VolumePrediction::from_probabilities(
        0.205, // very_low
        0.243, // low
        0.196, // medium
        0.182, // high
        0.175, // very_high
    );

    pred.training_horizon = "10h".to_string();
    pred.regime = "LOW".to_string();
    pred.confidence = 0.243;

    pred
}

#[test]
fn test_smart_order_generation_no_magic_numbers() {
    let current_price = 116227.05;

    let price_levels = create_test_price_levels();
    let direction = create_test_direction();
    let volatility = create_test_volatility();
    let sentiment = create_test_sentiment();
    let volume = create_test_volume();

    // Generate SMART orders
    let orders = TradingOrders::generate_smart(
        current_price,
        &price_levels,
        &direction,
        &volatility,
        &sentiment,
        &volume,
    )
    .unwrap();

    // Verify no magic numbers - all confidence values should come from model outputs
    for entry in &orders.entry_levels {
        assert!(entry.confidence > 0.0 && entry.confidence <= 1.0);
        // Confidence should be derived from model probabilities, not hardcoded
        assert_ne!(entry.confidence, 0.9); // Not the old hardcoded value
        assert_ne!(entry.confidence, 0.7);
        assert_ne!(entry.confidence, 0.6);
    }

    // Verify position sizes are dynamic and sum to 1.0
    let entry_sum: f64 = orders
        .entry_levels
        .iter()
        .map(|l| l.quantity_percentage)
        .sum();
    assert!(
        (entry_sum - 1.0).abs() < 0.01,
        "Entry sizes should sum to 1.0"
    );

    let exit_sum: f64 = orders
        .exit_levels
        .iter()
        .map(|l| l.quantity_percentage)
        .sum();
    assert!(
        (exit_sum - 1.0).abs() < 0.01,
        "Exit sizes should sum to 1.0"
    );
}

#[test]
fn test_short_order_validation() {
    let current_price = 116227.05;

    let price_levels = create_test_price_levels();
    let direction = create_test_direction();
    let volatility = create_test_volatility();
    let sentiment = create_test_sentiment();
    let volume = create_test_volume();

    // Generate SHORT orders (direction shows down probability > up)
    let orders = TradingOrders::generate_smart(
        current_price,
        &price_levels,
        &direction,
        &volatility,
        &sentiment,
        &volume,
    )
    .unwrap();

    if orders.direction == "SHORT" {
        // Verify SHORT order constraints
        for entry in &orders.entry_levels {
            assert!(
                entry.price > current_price,
                "SHORT entry {:.2} must be above current price {:.2}",
                entry.price,
                current_price
            );
        }

        for exit in &orders.exit_levels {
            assert!(
                exit.price < current_price,
                "SHORT exit {:.2} must be below current price {:.2}",
                exit.price,
                current_price
            );
        }

        // CRITICAL: Verify stops don't intersect ANY entry
        // Find the highest entry price
        let highest_entry = orders
            .entry_levels
            .iter()
            .map(|e| e.price)
            .fold(f64::NEG_INFINITY, f64::max);

        for (i, stop) in orders.stop_levels.iter().enumerate() {
            // Check against ALL entries, not just the corresponding one
            for (j, entry) in orders.entry_levels.iter().enumerate() {
                assert!(
                    stop.price > entry.price,
                    "❌ CRITICAL FAILURE: SHORT stop {} at {:.2} intersects with entry {} at {:.2}. Stop must be above ALL entries (highest: {:.2})",
                    i + 1,
                    stop.price,
                    j + 1,
                    entry.price,
                    highest_entry
                );
            }

            // Also verify stop is above the highest entry
            assert!(
                stop.price > highest_entry,
                "SHORT stop {} at {:.2} must be above highest entry {:.2}",
                i + 1,
                stop.price,
                highest_entry
            );
        }
    }
}

#[test]
fn test_long_order_validation() {
    let current_price = 116227.05;

    // Create bullish predictions for LONG test
    let price_levels = create_test_price_levels();
    let mut direction = create_test_direction();
    // Make it bullish
    direction.up_probability_aggregated = 0.6;
    direction.down_probability_aggregated = 0.3;

    let volatility = create_test_volatility();
    let sentiment = create_test_sentiment();
    let volume = create_test_volume();

    // Generate LONG orders
    let orders = TradingOrders::generate_smart(
        current_price,
        &price_levels,
        &direction,
        &volatility,
        &sentiment,
        &volume,
    )
    .unwrap();

    if orders.direction == "LONG" {
        // Verify LONG order constraints
        for entry in &orders.entry_levels {
            assert!(
                entry.price < current_price,
                "LONG entry {:.2} must be below current price {:.2}",
                entry.price,
                current_price
            );
        }

        for exit in &orders.exit_levels {
            assert!(
                exit.price > current_price,
                "LONG exit {:.2} must be above current price {:.2}",
                exit.price,
                current_price
            );
        }

        // CRITICAL: Verify stops don't intersect entries
        for (i, (entry, stop)) in orders
            .entry_levels
            .iter()
            .zip(orders.stop_levels.iter())
            .enumerate()
        {
            assert!(
                stop.price < entry.price,
                "LONG stop {} at {:.2} must be below entry {:.2} (NO INTERSECTION!)",
                i + 1,
                stop.price,
                entry.price
            );
        }
    }
}

#[test]
fn test_entry_levels_use_price_bins() {
    let current_price = 116227.05;

    let consensus = SmartConsensus {
        direction: create_test_direction(),
        price_levels: create_test_price_levels(),
        volatility: create_test_volatility(),
        sentiment: create_test_sentiment(),
        volume: create_test_volume(),
    };

    // Test SHORT entries
    let short_entries = consensus
        .generate_smart_entries(current_price, "SHORT")
        .unwrap();

    // Entry 1 should be close to neutral upper bound (0.909%)
    let entry_1_pct = (short_entries[0].price / current_price - 1.0) * 100.0;
    assert!(
        entry_1_pct > 0.0 && entry_1_pct < 2.0,
        "Entry 1 should be close to market"
    );

    // Entries should be progressively further
    assert!(short_entries[1].price > short_entries[0].price);
    assert!(short_entries[2].price > short_entries[1].price);

    // Test LONG entries
    let long_entries = consensus
        .generate_smart_entries(current_price, "LONG")
        .unwrap();

    // LONG entries should be below current price
    assert!(long_entries[0].price < current_price);
    assert!(long_entries[1].price < long_entries[0].price);
    assert!(long_entries[2].price < long_entries[1].price);
}

#[test]
fn test_stop_distances_from_volatility() {
    let current_price = 116227.05;

    let consensus = SmartConsensus {
        direction: create_test_direction(),
        price_levels: create_test_price_levels(),
        volatility: create_test_volatility(),
        sentiment: create_test_sentiment(),
        volume: create_test_volume(),
    };

    // Generate entries first
    let entries = consensus
        .generate_smart_entries(current_price, "SHORT")
        .unwrap();

    // Generate stops based on volatility
    let stops = consensus.generate_smart_stops(&entries, "SHORT").unwrap();

    // Verify stops use volatility model's recommended distance
    let expected_stop_distance = consensus.volatility.recommended_stop_distance_percent;

    // Find the highest entry (for SHORT)
    let highest_entry = entries
        .iter()
        .map(|e| e.price)
        .fold(f64::NEG_INFINITY, f64::max);

    for (i, stop) in stops.iter().enumerate() {
        // CRITICAL: Stop must be above ALL entries for SHORT
        assert!(
            stop.price > highest_entry,
            "Stop {} at {:.2} must be above highest entry {:.2}",
            i + 1,
            stop.price,
            highest_entry
        );

        // The distance from the highest entry should be based on recommended distance
        let distance_from_highest = ((stop.price - highest_entry) / highest_entry) * 100.0;

        // Should be positive and reasonable
        assert!(
            distance_from_highest > 0.0,
            "Stop {} distance from highest entry should be positive",
            i + 1
        );

        // Should be within reasonable range (considering adjustments and safety buffer)
        assert!(
            distance_from_highest < expected_stop_distance * 5.0,
            "Stop {} distance {:.2}% from highest entry should be reasonable (base recommended: {:.2}%)",
            i + 1,
            distance_from_highest,
            expected_stop_distance
        );
    }
}

#[test]
fn test_position_sizing_normalization() {
    let consensus = SmartConsensus {
        direction: create_test_direction(),
        price_levels: create_test_price_levels(),
        volatility: create_test_volatility(),
        sentiment: create_test_sentiment(),
        volume: create_test_volume(),
    };

    let current_price = 116227.05;

    // Generate all order levels
    let mut entries = consensus
        .generate_smart_entries(current_price, "SHORT")
        .unwrap();
    let mut exits = consensus
        .generate_smart_exits(current_price, "SHORT")
        .unwrap();

    // Normalize sizes
    SmartConsensus::normalize_sizes(&mut entries);
    SmartConsensus::normalize_sizes(&mut exits);

    // Verify normalization
    let entry_sum: f64 = entries.iter().map(|l| l.quantity_percentage).sum();
    assert!(
        (entry_sum - 1.0).abs() < 0.001,
        "Entry sizes should sum to 1.0 after normalization"
    );

    let exit_sum: f64 = exits.iter().map(|l| l.quantity_percentage).sum();
    assert!(
        (exit_sum - 1.0).abs() < 0.001,
        "Exit sizes should sum to 1.0 after normalization"
    );
}

#[test]
fn test_confidence_calculation_from_models() {
    let consensus = SmartConsensus {
        direction: create_test_direction(),
        price_levels: create_test_price_levels(),
        volatility: create_test_volatility(),
        sentiment: create_test_sentiment(),
        volume: create_test_volume(),
    };

    let overall_confidence = consensus.calculate_overall_confidence();

    // Should be weighted average of model confidences
    assert!(overall_confidence > 0.0 && overall_confidence <= 1.0);

    // Should not be a magic number
    assert_ne!(overall_confidence, 0.5);
    assert_ne!(overall_confidence, 0.75);
    assert_ne!(overall_confidence, 1.0);

    // Should be reasonable based on input confidences
    assert!(overall_confidence > 0.1 && overall_confidence < 0.5);
}

#[test]
fn test_no_stop_entry_intersection_critical() {
    // This test specifically checks the critical bug where stops intersect with entries
    let current_price = 116227.05;

    let consensus = SmartConsensus {
        direction: create_test_direction(),
        price_levels: create_test_price_levels(),
        volatility: create_test_volatility(),
        sentiment: create_test_sentiment(),
        volume: create_test_volume(),
    };

    // Test SHORT orders
    let short_entries = consensus
        .generate_smart_entries(current_price, "SHORT")
        .unwrap();
    let short_stops = consensus
        .generate_smart_stops(&short_entries, "SHORT")
        .unwrap();

    // Find highest entry
    let highest_entry = short_entries
        .iter()
        .map(|e| e.price)
        .fold(f64::NEG_INFINITY, f64::max);

    println!(
        "SHORT Entries: {:.2}, {:.2}, {:.2}",
        short_entries[0].price, short_entries[1].price, short_entries[2].price
    );
    println!(
        "SHORT Stops: {:.2}, {:.2}, {:.2}",
        short_stops[0].price, short_stops[1].price, short_stops[2].price
    );
    println!("Highest Entry: {:.2}", highest_entry);

    // CRITICAL: Every stop must be above EVERY entry for SHORT
    for (i, stop) in short_stops.iter().enumerate() {
        for (j, entry) in short_entries.iter().enumerate() {
            assert!(
                stop.price > entry.price,
                "❌ INTERSECTION BUG: SHORT stop {} ({:.2}) <= entry {} ({:.2})",
                i + 1,
                stop.price,
                j + 1,
                entry.price
            );
        }

        // Double check: stop must be above highest entry
        assert!(
            stop.price > highest_entry,
            "SHORT stop {} ({:.2}) must be above highest entry ({:.2})",
            i + 1,
            stop.price,
            highest_entry
        );
    }

    // Test LONG orders
    let mut bullish_direction = create_test_direction();
    bullish_direction.up_probability_aggregated = 0.6;
    bullish_direction.down_probability_aggregated = 0.3;

    let bullish_consensus = SmartConsensus {
        direction: bullish_direction,
        price_levels: create_test_price_levels(),
        volatility: create_test_volatility(),
        sentiment: create_test_sentiment(),
        volume: create_test_volume(),
    };

    let long_entries = bullish_consensus
        .generate_smart_entries(current_price, "LONG")
        .unwrap();
    let long_stops = bullish_consensus
        .generate_smart_stops(&long_entries, "LONG")
        .unwrap();

    // Find lowest entry
    let lowest_entry = long_entries
        .iter()
        .map(|e| e.price)
        .fold(f64::INFINITY, f64::min);

    println!(
        "\nLONG Entries: {:.2}, {:.2}, {:.2}",
        long_entries[0].price, long_entries[1].price, long_entries[2].price
    );
    println!(
        "LONG Stops: {:.2}, {:.2}, {:.2}",
        long_stops[0].price, long_stops[1].price, long_stops[2].price
    );
    println!("Lowest Entry: {:.2}", lowest_entry);

    // CRITICAL: Every stop must be below EVERY entry for LONG
    for (i, stop) in long_stops.iter().enumerate() {
        for (j, entry) in long_entries.iter().enumerate() {
            assert!(
                stop.price < entry.price,
                "❌ INTERSECTION BUG: LONG stop {} ({:.2}) >= entry {} ({:.2})",
                i + 1,
                stop.price,
                j + 1,
                entry.price
            );
        }

        // Double check: stop must be below lowest entry
        assert!(
            stop.price < lowest_entry,
            "LONG stop {} ({:.2}) must be below lowest entry ({:.2})",
            i + 1,
            stop.price,
            lowest_entry
        );
    }
}
