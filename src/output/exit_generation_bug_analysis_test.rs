use crate::output::prediction_types::{PriceBin, PriceLevelPrediction};
use crate::utils::error::VangaError;
use std::collections::HashMap;

#[test]
fn test_exit_generation_logic_bugs() -> Result<(), VangaError> {
    println!("\n🔍 ANALYZING EXIT GENERATION BUGS");

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

    println!("Current Price: ${:.5}", current_price);
    println!("\n📊 PRICE BIN ANALYSIS:");

    // Analyze each bin relative to current price
    for (name, bin) in &price_levels.bins {
        let lower = bin.price[0];
        let upper = bin.price[1];
        let center = (lower + upper) / 2.0;

        let lower_vs_current = if lower > current_price {
            "ABOVE"
        } else {
            "BELOW"
        };
        let upper_vs_current = if upper > current_price {
            "ABOVE"
        } else {
            "BELOW"
        };
        let center_vs_current = if center > current_price {
            "ABOVE"
        } else {
            "BELOW"
        };

        println!(
            "  {}: ${:.5}-${:.5} (center: ${:.5})",
            name, lower, upper, center
        );
        println!(
            "    Lower edge: {} current | Upper edge: {} current | Center: {} current",
            lower_vs_current, upper_vs_current, center_vs_current
        );
    }

    println!("\n🎯 CURRENT EXIT GENERATION LOGIC FOR SHORT:");

    // Test the current logic
    let direction = "SHORT";
    let favorable_bins = vec!["neutral", "moderate_down", "strong_down"];

    println!("Favorable bins: {:?}", favorable_bins);

    let mut bin_targets = Vec::new();
    for bin_name in &favorable_bins {
        if let Some(bin) = price_levels.bins.get(*bin_name) {
            let target_price = if bin_targets.is_empty() && *bin_name == "neutral" {
                // CURRENT LOGIC: First exit uses neutral edge
                if direction == "LONG" {
                    bin.price[1] // Upper edge of neutral
                } else {
                    bin.price[0] // Lower edge of neutral - THIS IS THE BUG!
                }
            } else {
                // Further exits: use bin center
                (bin.price[0] + bin.price[1]) / 2.0
            };

            let vs_current = if target_price > current_price {
                "ABOVE"
            } else {
                "BELOW"
            };
            let profit_loss = if direction == "SHORT" {
                if target_price < current_price {
                    "PROFIT"
                } else {
                    "LOSS"
                }
            } else if target_price > current_price {
                "PROFIT"
            } else {
                "LOSS"
            };
            println!(
                "  {} -> ${:.5} ({} current, {} for {})",
                bin_name, target_price, vs_current, profit_loss, direction
            );

            bin_targets.push((target_price, bin.probability));
        }
    }

    println!("\n❌ IDENTIFIED BUGS:");

    // Bug 1: Neutral bin is above current price
    if let Some(neutral_bin) = price_levels.bins.get("neutral") {
        if neutral_bin.price[0] > current_price {
            println!(
                "1. NEUTRAL BIN POSITIONING: Neutral lower edge (${:.5}) > current price (${:.5})",
                neutral_bin.price[0], current_price
            );
            println!("   This means 'neutral' is actually BULLISH relative to current price!");
        }
    }

    // Bug 2: First exit logic for SHORT
    if let Some((first_target, _)) = bin_targets.first() {
        if direction == "SHORT" && *first_target >= current_price {
            println!(
                "2. SHORT EXIT LOGIC: First exit (${:.5}) >= current price (${:.5})",
                first_target, current_price
            );
            println!("   For SHORT, exits should be BELOW current price to take profit!");
        }
    }

    // Bug 3: Favorable bins selection
    println!("3. FAVORABLE BINS FOR SHORT: {:?}", favorable_bins);
    println!("   For SHORT positions, we want to exit at LOWER prices");
    println!("   But 'neutral' bin is ABOVE current price - this is contradictory!");

    println!("\n✅ PROPOSED FIXES:");

    println!("1. FOR SHORT EXITS: Use bins that are BELOW current price");
    println!("   - Skip 'neutral' if it's above current price");
    println!("   - Prioritize 'moderate_down' and 'strong_down'");
    println!("   - Use upper edges of down bins (closer to current price for quick profit)");

    println!(
        "2. DYNAMIC BIN SELECTION: Choose bins based on their position relative to current price"
    );
    println!("   - SHORT: Use bins with centers BELOW current price");
    println!("   - LONG: Use bins with centers ABOVE current price");

    println!("3. EDGE SELECTION LOGIC:");
    println!("   - SHORT first exit: Use UPPER edge of the highest down bin (quick profit)");
    println!("   - SHORT further exits: Use centers of lower down bins");

    // Test the proposed fix
    println!("\n🔧 TESTING PROPOSED FIX:");

    let mut fixed_targets = Vec::new();

    // For SHORT, find bins that are actually below current price
    let suitable_bins: Vec<_> = price_levels
        .bins
        .iter()
        .filter(|(_, bin)| {
            let center = (bin.price[0] + bin.price[1]) / 2.0;
            center < current_price // Only bins with centers below current price
        })
        .collect();

    println!("Suitable bins for SHORT (centers below current):");
    for (name, bin) in &suitable_bins {
        let center = (bin.price[0] + bin.price[1]) / 2.0;
        println!(
            "  {}: center=${:.5}, prob={:.1}%",
            name,
            center,
            bin.probability * 100.0
        );
    }

    // Sort by probability (highest first) and distance from current (closest first)
    let mut sorted_bins = suitable_bins;
    sorted_bins.sort_by(|a, b| {
        let a_center = (a.1.price[0] + a.1.price[1]) / 2.0;
        let b_center = (b.1.price[0] + b.1.price[1]) / 2.0;
        let a_distance = (current_price - a_center).abs();
        let b_distance = (current_price - b_center).abs();

        // First sort by probability (descending), then by distance (ascending)
        b.1.probability
            .partial_cmp(&a.1.probability)
            .unwrap()
            .then(a_distance.partial_cmp(&b_distance).unwrap())
    });

    // Generate fixed exits
    for (i, (_name, bin)) in sorted_bins.iter().take(3).enumerate() {
        let target_price = if i == 0 {
            // First exit: use upper edge for quick profit
            bin.price[1].min(current_price * 0.999) // Ensure it's below current
        } else {
            // Further exits: use center
            (bin.price[0] + bin.price[1]) / 2.0
        };

        let profit_pct = ((current_price - target_price) / current_price) * 100.0;
        println!(
            "  Fixed Exit {}: ${:.5} ({:.2}% profit)",
            i + 1,
            target_price,
            profit_pct
        );

        fixed_targets.push(target_price);
    }

    // Verify all fixed exits are below current price
    let all_below_current = fixed_targets.iter().all(|&price| price < current_price);

    if all_below_current {
        println!("\n✅ FIXED EXITS: All exits are below current price for SHORT");
    } else {
        println!("\n❌ FIXED EXITS: Some exits are still above current price");
    }

    Ok(())
}

#[test]
fn test_rr_scaling_bug() -> Result<(), VangaError> {
    println!("\n🔍 ANALYZING R:R SCALING BUG");

    let current_price = 0.0666;

    // Test the R:R scaling logic that might be causing 0.03 prices
    println!("Current Price: ${:.5}", current_price);

    // Simulate the scaling logic from trading_orders.rs
    let target_risk_reward = 3.0;
    let initial_rr = 2.0; // Below target

    println!("Target R:R: {:.1}", target_risk_reward);
    println!("Initial R:R: {:.1}", initial_rr);

    // This is the scaling factor calculation from the code
    let scale_factor = target_risk_reward / initial_rr;
    println!("Scale Factor: {:.2}", scale_factor);

    // Test what happens to exit prices
    let initial_exits: [f64; 3] = [0.0662, 0.0650, 0.0635]; // Reasonable exits

    println!("\nInitial Exits:");
    for (i, &price) in initial_exits.iter().enumerate() {
        println!("  Exit {}: ${:.5}", i + 1, price);
    }

    // Apply scaling
    println!("\nAfter R:R Scaling:");
    for (i, &initial_price) in initial_exits.iter().enumerate() {
        let distance_from_current: f64 = (initial_price - current_price).abs();
        let scaled_distance = distance_from_current * scale_factor;

        let scaled_price = if initial_price < current_price {
            // SHORT exit - move further down
            current_price - scaled_distance
        } else {
            // LONG exit - move further up
            current_price + scaled_distance
        };

        println!(
            "  Exit {}: ${:.5} -> ${:.5} (scale: {:.2}x)",
            i + 1,
            initial_price,
            scaled_price,
            scale_factor
        );

        if scaled_price < 0.05 {
            println!("    ❌ UNREALISTIC PRICE DETECTED!");
        }
    }

    println!("\n🔍 POTENTIAL BUG: If scale_factor > 1.5, exits can become unrealistic");
    println!("The scaling should be capped or use a different approach");

    Ok(())
}
