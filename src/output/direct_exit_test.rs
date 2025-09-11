use crate::output::prediction_types::{
    DirectionPrediction, PriceBin, PriceLevelPrediction, SentimentPrediction, VolatilityPrediction,
    VolumePrediction,
};
use crate::output::smart_order_generator::SmartConsensus;
use crate::utils::error::VangaError;
use std::collections::HashMap;

#[test]
fn test_direct_smart_exits_generation() -> Result<(), VangaError> {
    println!("\n🔍 DIRECT TEST OF generate_smart_exits()");

    let current_price = 0.0666;

    // Create the exact bins from the user's prediction
    let mut bins = HashMap::new();

    bins.insert(
        "neutral".to_string(),
        PriceBin {
            price: [0.06662499999999999, 0.06847493306899277], // ABOVE current price!
            range: [0.037537537537512565, 2.8152148183074552],
            vwap_range: [-0.45719800047577086, 2.3067422801864677],
            probability: 0.22334290756621375,
        },
    );

    bins.insert(
        "moderate_down".to_string(),
        PriceBin {
            price: [0.0634356, 0.06662493306899277], // Partially below current
            range: [-4.751351351351368, 0.037437040529670736],
            vwap_range: [-5.2224034443374165, -0.4572980004757709],
            probability: 0.1954303385135696,
        },
    );

    bins.insert(
        "strong_down".to_string(),
        PriceBin {
            price: [0.06117119999999999, 0.06343553306899277], // BELOW current price
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

    // Create minimal other predictions
    let direction = DirectionPrediction {
        prediction: "DOWN".to_string(),
        confidence: 0.23,
        up_probability: 0.18,
        down_probability: 0.23,
        pump_probability: 0.17,
        dump_probability: 0.20,
        sideways_probability: 0.20,
        breakout_probability: 0.37,
        up_probability_aggregated: 0.36,
        down_probability_aggregated: 0.43,
        sideways_probability_aggregated: 0.20,
        expected_upside_percent: 3.57,
        expected_downside_percent: 4.19,
        most_likely_move_percent: -0.62,
        risk_reward_ratio: 0.85,
        breakout_threshold_percent: 10.0,
        sequence_bandwidth_percent: 10.0,
        sequence_length: 40,
        training_horizon: "20h".to_string(),
    };

    let volatility = VolatilityPrediction {
        regime: "LOW".to_string(),
        confidence: 0.22,
        regime_confidence: 0.23,
        very_low_probability: 0.21,
        low_probability: 0.23,
        medium_probability: 0.19,
        high_probability: 0.18,
        very_high_probability: 0.17,
        expected_range_percent: 11.26,
        volatility_percentile: 56.41,
        position_size_multiplier: 1.2,
        recommended_stop_distance_percent: 11.26,
        training_horizon: "20h".to_string(),
    };

    let sentiment = SentimentPrediction {
        regime: "BEARISH".to_string(),
        confidence: 0.30,
        very_bullish_probability: 0.18,
        bullish_probability: 0.19,
        neutral_probability: 0.20,
        bearish_probability: 0.21,
        very_bearish_probability: 0.18,
        training_horizon: "20h".to_string(),
    };

    let volume = VolumePrediction {
        regime: "HIGH".to_string(),
        confidence: 0.31,
        very_low_probability: 0.18,
        low_probability: 0.18,
        medium_probability: 0.20,
        high_probability: 0.22,
        very_high_probability: 0.20,
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

    println!("Current Price: ${:.5}", current_price);
    println!("Direction: SHORT");

    // DIRECTLY test the generate_smart_exits method
    let exits = consensus.generate_smart_exits(current_price, "SHORT")?;

    println!("\n=== DIRECT generate_smart_exits() RESULTS ===");
    for (i, exit) in exits.iter().enumerate() {
        let profit_pct = ((current_price - exit.price) / current_price) * 100.0;
        let vs_current = if exit.price < current_price {
            "BELOW (PROFIT)"
        } else {
            "ABOVE (LOSS)"
        };

        println!(
            "  Exit {}: ${:.5} ({:+.2}% from current) - {}",
            i + 1,
            exit.price,
            profit_pct,
            vs_current
        );
    }

    // Test the fix
    let all_profitable = exits.iter().all(|exit| exit.price < current_price);
    let no_extreme_prices = exits.iter().all(|exit| exit.price > current_price * 0.8);

    if all_profitable && no_extreme_prices {
        println!("\n✅ FIXED: All exits are profitable and reasonable!");
    } else {
        println!("\n❌ STILL BROKEN:");
        if !all_profitable {
            println!("  - Some exits are above current price (unprofitable)");
        }
        if !no_extreme_prices {
            println!("  - Some exits are extremely low");
        }
    }

    // Assertions
    assert!(
        all_profitable,
        "All exits should be below current price for SHORT"
    );
    assert!(no_extreme_prices, "No exits should be extremely low");

    Ok(())
}
