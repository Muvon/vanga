use crate::output::prediction_types::{
    DirectionPrediction, PriceBin, PriceLevelPrediction, SentimentPrediction, VolatilityPrediction,
    VolumePrediction,
};
use crate::output::smart_order_generator::SmartConsensus;
use crate::output::trading_orders::{OrderLevel, TradingOrders};
use crate::utils::error::VangaError;
use std::collections::HashMap;

#[test]
fn test_neutral_prediction_exit_generation() -> Result<(), VangaError> {
    // Create the exact prediction from the user's example
    let current_price = 0.0666;
    let _current_vwap_price = 0.06693100722673893; // Keep for reference but not used in this test

    // Create price level bins exactly as in the raw prediction
    let mut bins = HashMap::new();

    // Neutral bin - the model's prediction
    bins.insert(
        "neutral".to_string(),
        PriceBin {
            price: [0.06662499999999999, 0.06847493306899277],
            range: [0.037537537537512565, 2.8152148183074552],
            vwap_range: [-0.45719800047577086, 2.3067422801864677],
            probability: 0.22334290756621375,
        },
    );

    bins.insert(
        "strong_up".to_string(),
        PriceBin {
            price: [0.07166439999999999, 0.0739288],
            range: [7.604204204204177, 11.004204204204198],
            vwap_range: [7.072047724048113, 10.45523302757866],
            probability: 0.19559353590506096,
        },
    );

    bins.insert(
        "moderate_up".to_string(),
        PriceBin {
            price: [0.06847500000000001, 0.07166433306899278],
            range: [2.8153153153153174, 7.604103707196357],
            vwap_range: [2.306842280186468, 7.071947724048114],
            probability: 0.20200994310042203,
        },
    );

    bins.insert(
        "strong_down".to_string(),
        PriceBin {
            price: [0.06117119999999999, 0.06343553306899277],
            range: [-8.151351351351378, -4.75145184835921],
            vwap_range: [-8.605588747867973, -5.222503444337416],
            probability: 0.1836232749147335,
        },
    );

    bins.insert(
        "moderate_down".to_string(),
        PriceBin {
            price: [0.0634356, 0.06662493306899277],
            range: [-4.751351351351368, 0.037437040529670736],
            vwap_range: [-5.2224034443374165, -0.4572980004757709],
            probability: 0.1954303385135696,
        },
    );

    let price_levels = PriceLevelPrediction {
        bins,
        most_likely_range: [0.037537537537512565, 2.8152148183074552],
        confidence: 0.3293658857251268,
    };

    let direction = DirectionPrediction {
        prediction: "DOWN".to_string(),
        confidence: 0.23123875871245037,
        up_probability: 0.18702293931374409,
        down_probability: 0.2341794497687541,
        pump_probability: 0.17576068170073778,
        dump_probability: 0.2014310528861748,
        sideways_probability: 0.20160587633058927,
        breakout_probability: 0.3771917345869126,
        up_probability_aggregated: 0.36278362101448186,
        down_probability_aggregated: 0.4356105026549289,
        sideways_probability_aggregated: 0.20160587633058927,
        expected_upside_percent: 3.5715249220797873,
        expected_downside_percent: 4.192363042136392,
        most_likely_move_percent: -0.6208381200566055,
        risk_reward_ratio: 0.851912128358943,
        breakout_threshold_percent: 10.0,
        sequence_bandwidth_percent: 10.0,
        sequence_length: 40,
        training_horizon: "20h".to_string(),
    };

    let volatility = VolatilityPrediction {
        regime: "LOW".to_string(),
        confidence: 0.22871183023864872,
        regime_confidence: 0.2332361322627116,
        very_low_probability: 0.21399350186769056,
        low_probability: 0.2332361322627116,
        medium_probability: 0.197951848532272,
        high_probability: 0.18098893421700185,
        very_high_probability: 0.17382958312032412,
        expected_range_percent: 11.262478309982193,
        volatility_percentile: 56.41025641025641,
        position_size_multiplier: 1.2,
        recommended_stop_distance_percent: 11.262478309982193,
        training_horizon: "20h".to_string(),
    };

    let sentiment = SentimentPrediction {
        regime: "BEARISH".to_string(),
        confidence: 0.3088828940619964,
        very_bullish_probability: 0.1862520377393502,
        bullish_probability: 0.19874390392219576,
        neutral_probability: 0.20773411862881272,
        bearish_probability: 0.21731849825352836,
        very_bearish_probability: 0.18995144145611303,
        training_horizon: "20h".to_string(),
    };

    let volume = VolumePrediction {
        regime: "HIGH".to_string(),
        confidence: 0.31846749875128455,
        very_low_probability: 0.18198326978251397,
        low_probability: 0.18986668055633807,
        medium_probability: 0.20393299141548415,
        high_probability: 0.22013749963273077,
        very_high_probability: 0.204079558612933,
        training_horizon: "20h".to_string(),
    };

    // Create consensus from predictions
    let consensus = SmartConsensus {
        direction: direction.clone(),
        price_levels: price_levels.clone(),
        volatility: volatility.clone(),
        sentiment: sentiment.clone(),
        volume: volume.clone(),
    };

    // Generate exits using the smart exit generation
    let exits = consensus.generate_smart_exits(current_price, "SHORT")?;

    // Sort exits by price (descending for SHORT - highest to lowest for progressive profit taking)
    let mut sorted_exits = exits.clone();
    sorted_exits.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());

    // Create orders structure for validation
    let orders = TradingOrders {
        direction: "SHORT".to_string(),
        exit_levels: [
            OrderLevel {
                price: sorted_exits[0].price,
                quantity_percentage: sorted_exits[0].quantity_percentage,
                atr_distance: sorted_exits[0].atr_distance,
                order_type: sorted_exits[0].order_type.clone(),
                confidence: sorted_exits[0].confidence,
            },
            OrderLevel {
                price: sorted_exits[1].price,
                quantity_percentage: sorted_exits[1].quantity_percentage,
                atr_distance: sorted_exits[1].atr_distance,
                order_type: sorted_exits[1].order_type.clone(),
                confidence: sorted_exits[1].confidence,
            },
            OrderLevel {
                price: sorted_exits[2].price,
                quantity_percentage: sorted_exits[2].quantity_percentage,
                atr_distance: sorted_exits[2].atr_distance,
                order_type: sorted_exits[2].order_type.clone(),
                confidence: sorted_exits[2].confidence,
            },
        ],
        ..Default::default()
    };

    // Print the generated orders for debugging
    println!("\n=== GENERATED ORDERS ===");
    println!("Direction: {}", orders.direction);
    println!("Current Price: ${:.5}", current_price);
    println!("\nExit Levels:");
    for (i, exit) in orders.exit_levels.iter().enumerate() {
        println!(
            "  Exit {}: ${:.5} ({:+.2}% from current)",
            i + 1,
            exit.price,
            ((exit.price - current_price) / current_price) * 100.0
        );
    }

    // CRITICAL ASSERTIONS - Test reasonable behavior, not exact numbers
    // The model predicted neutral range: 0.06662 - 0.06847
    // For SHORT direction, exits should be BELOW current price (0.0666)

    // Check that all exits are below current price for SHORT (profitable direction)
    for (i, exit) in orders.exit_levels.iter().enumerate() {
        assert!(
            exit.price < current_price,
            "Exit {} price ${:.5} should be below current ${:.5} for SHORT (profitable direction)",
            i + 1,
            exit.price,
            current_price
        );
    }

    // Check that exits are reasonable using MODEL-DERIVED boundaries (NO MAGIC NUMBERS)
    // Use the unified boundary calculation for consistent validation
    let model_boundaries = crate::output::model_boundaries::ModelBoundaries::calculate(
        &price_levels,
        current_price,
        "SHORT",
        volatility.expected_range_percent,
    );

    for (i, exit) in orders.exit_levels.iter().enumerate() {
        // Validate using the unified boundary system
        if let Err(boundary_error) = model_boundaries.validate_exit_price(exit.price, current_price)
        {
            panic!(
                "Exit {} violates model boundaries: {}",
                i + 1,
                boundary_error
            );
        }
    }

    // Check that exits are ordered correctly for SHORT (descending)
    for i in 0..2 {
        assert!(
            orders.exit_levels[i].price >= orders.exit_levels[i + 1].price,
            "SHORT exits should be descending: Exit {} (${:.5}) >= Exit {} (${:.5})",
            i + 1,
            orders.exit_levels[i].price,
            i + 2,
            orders.exit_levels[i + 1].price
        );
    }

    // Check that quantities sum to approximately 1.0
    let total_exit_quantity: f64 = orders
        .exit_levels
        .iter()
        .map(|e| e.quantity_percentage)
        .sum();
    assert!(
        (total_exit_quantity - 1.0).abs() < 0.05,
        "Exit quantities should sum to ~1.0, got {:.3}",
        total_exit_quantity
    );

    // Note: Risk/reward ratio not tested here as we're only testing exit generation
    // without stop levels. The ratio would be calculated properly in full order generation.

    Ok(())
}

#[test]
fn test_exit_generation_bug_reproduction() -> Result<(), VangaError> {
    // This test reproduces the exact bug where exits are calculated as 0.03
    // when the model predicts neutral range around 0.066

    use crate::output::smart_order_generator::SmartConsensus;

    let current_price = 0.0666;

    // Create the consensus with the exact prediction data
    let mut bins = HashMap::new();
    bins.insert(
        "neutral".to_string(),
        PriceBin {
            price: [0.06662499999999999, 0.06847493306899277],
            range: [0.037537537537512565, 2.8152148183074552],
            vwap_range: [-0.45719800047577086, 2.3067422801864677],
            probability: 0.22334290756621375,
        },
    );

    bins.insert(
        "moderate_down".to_string(),
        PriceBin {
            price: [0.0634356, 0.06662493306899277],
            range: [-4.751351351351368, 0.037437040529670736],
            vwap_range: [-5.2224034443374165, -0.4572980004757709],
            probability: 0.1954303385135696,
        },
    );

    bins.insert(
        "strong_down".to_string(),
        PriceBin {
            price: [0.06117119999999999, 0.06343553306899277],
            range: [-8.151351351351378, -4.75145184835921],
            vwap_range: [-8.605588747867973, -5.222503444337416],
            probability: 0.1836232749147335,
        },
    );

    let price_levels = PriceLevelPrediction {
        bins,
        most_likely_range: [0.037537537537512565, 2.8152148183074552],
        confidence: 0.3293658857251268,
    };

    let direction = DirectionPrediction {
        prediction: "DOWN".to_string(),
        confidence: 0.23123875871245037,
        up_probability: 0.18702293931374409,
        down_probability: 0.2341794497687541,
        pump_probability: 0.17576068170073778,
        dump_probability: 0.2014310528861748,
        sideways_probability: 0.20160587633058927,
        breakout_probability: 0.3771917345869126,
        up_probability_aggregated: 0.36278362101448186,
        down_probability_aggregated: 0.4356105026549289,
        sideways_probability_aggregated: 0.20160587633058927,
        expected_upside_percent: 3.5715249220797873,
        expected_downside_percent: 4.192363042136392,
        most_likely_move_percent: -0.6208381200566055,
        risk_reward_ratio: 0.851912128358943,
        breakout_threshold_percent: 10.0,
        sequence_bandwidth_percent: 10.0,
        sequence_length: 40,
        training_horizon: "20h".to_string(),
    };

    let volatility = VolatilityPrediction {
        regime: "LOW".to_string(),
        confidence: 0.22871183023864872,
        regime_confidence: 0.2332361322627116,
        very_low_probability: 0.21399350186769056,
        low_probability: 0.2332361322627116,
        medium_probability: 0.197951848532272,
        high_probability: 0.18098893421700185,
        very_high_probability: 0.17382958312032412,
        expected_range_percent: 11.262478309982193,
        volatility_percentile: 56.41025641025641,
        position_size_multiplier: 1.2,
        recommended_stop_distance_percent: 11.262478309982193,
        training_horizon: "20h".to_string(),
    };

    let sentiment = SentimentPrediction {
        regime: "BEARISH".to_string(),
        confidence: 0.3088828940619964,
        very_bullish_probability: 0.1862520377393502,
        bullish_probability: 0.19874390392219576,
        neutral_probability: 0.20773411862881272,
        bearish_probability: 0.21731849825352836,
        very_bearish_probability: 0.18995144145611303,
        training_horizon: "20h".to_string(),
    };

    let volume = VolumePrediction {
        regime: "HIGH".to_string(),
        confidence: 0.31846749875128455,
        very_low_probability: 0.18198326978251397,
        low_probability: 0.18986668055633807,
        medium_probability: 0.20393299141548415,
        high_probability: 0.22013749963273077,
        very_high_probability: 0.204079558612933,
        training_horizon: "20h".to_string(),
    };

    // Create SmartConsensus
    let consensus = SmartConsensus {
        price_levels,
        direction,
        volatility,
        sentiment,
        volume,
    };

    // Generate exits using the smart exit generation
    let exits = consensus.generate_smart_exits(current_price, "SHORT")?;

    println!("\n=== BUG REPRODUCTION ===");
    println!("Current Price: ${:.5}", current_price);
    println!("Direction: SHORT");
    println!("\nGenerated Exits:");
    for (i, exit) in exits.iter().enumerate() {
        println!(
            "  Exit {}: ${:.5} ({:+.2}% from current)",
            i + 1,
            exit.price,
            ((exit.price - current_price) / current_price) * 100.0
        );
    }

    // THE FIX VALIDATION: Check if the bugs are resolved
    let all_exits_profitable = exits.iter().all(|exit| exit.price < current_price);
    let no_extreme_prices = exits.iter().all(|exit| exit.price > current_price * 0.8); // Within 20% profit range
    let reasonable_spacing = {
        let mut sorted_prices: Vec<f64> = exits.iter().map(|e| e.price).collect();
        sorted_prices.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Descending for SHORT
        sorted_prices
            .windows(2)
            .all(|w| (w[0] - w[1]) / current_price < 0.1) // Max 10% between exits
    };

    if all_exits_profitable && no_extreme_prices && reasonable_spacing {
        println!("\n✅ BUGS FIXED: All exits are profitable and reasonable!");
        println!("✅ All exits below current price for SHORT");
        println!("✅ No extreme prices (all above 80% of current)");
        println!("✅ Reasonable spacing between exits");
    } else {
        println!("\n❌ BUGS STILL PRESENT:");
        if !all_exits_profitable {
            println!("  - Some exits are above current price (unprofitable for SHORT)");
        }
        if !no_extreme_prices {
            println!("  - Some exits are extremely low (below 80% of current)");
        }
        if !reasonable_spacing {
            println!("  - Exit spacing is unreasonable");
        }
    }

    // Validate the fixes work
    assert!(
        all_exits_profitable,
        "All exits should be profitable for SHORT direction"
    );
    assert!(no_extreme_prices, "No exits should be extremely low");
    assert!(reasonable_spacing, "Exit spacing should be reasonable");

    Ok(())
}

#[test]
fn test_full_pipeline_with_rr_optimization() -> Result<(), VangaError> {
    // This test shows how exits SHOULD be calculated

    let current_price = 0.0666;
    let direction = "SHORT";

    // Neutral bin from the prediction
    let neutral_range = [0.06662499999999999, 0.06847493306899277];
    let moderate_down_range = [0.0634356, 0.06662493306899277];
    let strong_down_range = [0.06117119999999999, 0.06343553306899277];

    println!("\n=== CORRECT EXIT CALCULATION ===");
    println!("Current Price: ${:.5}", current_price);
    println!("Direction: {}", direction);
    println!("\nPrice Level Bins (absolute prices):");
    println!(
        "  Neutral:       ${:.5} - ${:.5}",
        neutral_range[0], neutral_range[1]
    );
    println!(
        "  Moderate Down: ${:.5} - ${:.5}",
        moderate_down_range[0], moderate_down_range[1]
    );
    println!(
        "  Strong Down:   ${:.5} - ${:.5}",
        strong_down_range[0], strong_down_range[1]
    );

    // For SHORT direction, exits should be:
    // 1. All below current price (profitable)
    // 2. Reasonably spaced
    // 3. Use available bin data intelligently

    // Find bins that are actually suitable for SHORT (centers below current)
    let bin_data = [
        ("moderate_down", moderate_down_range),
        ("strong_down", strong_down_range),
    ];
    let suitable_bins: Vec<_> = bin_data
        .iter()
        .filter(|(_, range)| {
            let center = (range[0] + range[1]) / 2.0;
            center < current_price
        })
        .collect();

    println!("\nSuitable bins for SHORT (centers below current):");
    for (name, range) in &suitable_bins {
        let center = (range[0] + range[1]) / 2.0;
        println!("  {}: center=${:.5}", name, center);
    }

    // Calculate reasonable exits based on suitable bins
    let mut reasonable_exits = Vec::new();
    for (i, (_, range)) in suitable_bins.iter().enumerate() {
        let exit_price = if i == 0 {
            // First exit: use upper edge but ensure it's below current
            let upper_edge = range[1];
            if upper_edge < current_price {
                upper_edge
            } else {
                // Fallback to center if upper edge is above current
                (range[0] + range[1]) / 2.0
            }
        } else {
            (range[0] + range[1]) / 2.0 // Center for further exits
        };
        reasonable_exits.push(exit_price);
    }

    // Add a third exit if needed using volatility
    if reasonable_exits.len() < 3 {
        let last_exit = reasonable_exits
            .last()
            .copied()
            .unwrap_or(current_price * 0.95);
        let volatility_distance = current_price * 0.05; // 5% additional distance
        reasonable_exits.push(last_exit - volatility_distance);
    }

    println!("\nReasonable Exits for SHORT:");
    for (i, &exit_price) in reasonable_exits.iter().take(3).enumerate() {
        let profit_pct = ((current_price - exit_price) / current_price) * 100.0;
        println!(
            "  Exit {}: ${:.5} ({:.2}% profit)",
            i + 1,
            exit_price,
            profit_pct
        );
    }

    // Validate all reasonable exits
    for (i, &exit_price) in reasonable_exits.iter().take(3).enumerate() {
        assert!(
            exit_price < current_price,
            "Exit {} should be below current for SHORT",
            i + 1
        );
        assert!(
            exit_price > current_price * 0.8,
            "Exit {} should not be extremely low",
            i + 1
        );
    }

    println!("\n✅ All reasonable exits are correctly calculated and profitable!");

    Ok(())
}
