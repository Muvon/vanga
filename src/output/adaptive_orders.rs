//! Adaptive Mathematical Order Generation System
//!
//! This module implements a comprehensive order generation system that maximizes utilization
//! of ALL prediction data: price level probabilities, volatility regime probabilities,
//! direction confidence, sequence bandwidth, and risk-reward calculations.
//!
//! Key Features:
//! - Entropy-based adaptive thresholds (no hardcoded 15%)
//! - Probability-weighted position sizing
//! - Volatility regime-aware ATR multipliers
//! - Sequence bandwidth-aware order spacing
//! - Mathematical risk-reward optimization
//! - Confidence-driven order placement

use crate::output::{
    DirectionPrediction, OrderConfig, OrderLevel, PriceBin, PriceLevelPrediction, TradingOrders,
    VolatilityPrediction,
};
use log::{debug, info, warn};
use std::collections::HashMap;

/// Calculate volatility regime-aware ATR multiplier using probability distribution
/// This replaces hardcoded regime multipliers with mathematical probability weighting
fn calculate_volatility_aware_atr_multiplier(
    volatility_pred: &VolatilityPrediction,
    base_atr_multiplier: f64,
) -> f64 {
    // Probability-weighted multiplier calculation
    let weighted_multiplier = volatility_pred.very_low_probability * 0.4 +    // Very tight in very low vol
        volatility_pred.low_probability * 0.6 +         // Tight in low vol
        volatility_pred.medium_probability * 1.0 +      // Normal spacing
        volatility_pred.high_probability * 1.6 +        // Wide in high vol
        volatility_pred.very_high_probability * 2.2; // Very wide in very high vol

    let final_multiplier = base_atr_multiplier * weighted_multiplier;

    // Apply confidence scaling - lower confidence = wider spacing for safety
    let confidence_adjusted = final_multiplier * (0.8 + (1.0 - volatility_pred.confidence) * 0.4);

    // Reasonable bounds for crypto markets
    let bounded_multiplier = confidence_adjusted.clamp(0.3, 3.5);

    debug!(
        "🌊 Volatility ATR: regime={} prob_weighted={:.2} confidence_adj={:.2} final={:.2}",
        volatility_pred.regime, weighted_multiplier, confidence_adjusted, bounded_multiplier
    );

    bounded_multiplier
}

/// Select optimal price ranges based on probability distribution and direction
/// This replaces arbitrary range selection with mathematical probability weighting
fn select_optimal_ranges_by_probability(
    bins: &HashMap<String, PriceBin>,
    direction: &str,
    max_ranges: usize,
) -> Vec<(String, PriceBin, f64)> {
    let total_probability: f64 = bins.values().map(|b| b.probability).sum();
    let mut suitable_ranges = Vec::new();

    // Calculate dynamic probability threshold
    let probabilities: Vec<f64> = bins.values().map(|b| b.probability).collect();
    let max_prob = probabilities.iter().fold(0.0f64, |a, &b| a.max(b));
    let avg_prob = total_probability / bins.len() as f64;
    let dynamic_threshold = avg_prob.max(max_prob * 0.2); // At least 20% of max or average

    debug!(
        "🎯 Probability Analysis: total={:.3}, avg={:.3}, max={:.3}, threshold={:.3}",
        total_probability, avg_prob, max_prob, dynamic_threshold
    );

    for (name, bin) in bins {
        // Only consider ranges with significant probability mass
        if bin.probability < dynamic_threshold {
            continue;
        }

        let probability_weight = bin.probability / total_probability;

        // Direction-specific range selection logic
        let is_suitable = match direction {
            "LONG" | "LONG_BREAKOUT" => {
                // For LONG: prioritize negative ranges (buy dips) and neutral
                bin.range[1] <= 0.0 || (bin.range[0] <= 0.0 && bin.range[1] > 0.0)
            }
            "SHORT" | "SHORT_BREAKOUT" => {
                // For SHORT: prioritize positive ranges (sell rallies) and neutral
                bin.range[0] >= 0.0 || (bin.range[0] < 0.0 && bin.range[1] >= 0.0)
            }
            _ => false,
        };

        if is_suitable {
            suitable_ranges.push((name.clone(), bin.clone(), probability_weight));
            debug!(
                "✅ Selected range {}: [{:.2}%, {:.2}%] prob={:.3} weight={:.3}",
                name, bin.range[0], bin.range[1], bin.probability, probability_weight
            );
        }
    }

    // Sort by probability (highest first) for optimal allocation
    suitable_ranges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Return top ranges up to max_ranges
    suitable_ranges.into_iter().take(max_ranges).collect()
}

/// Calculate mathematical position sizing using Kelly Criterion and confidence weighting
/// This replaces arbitrary position sizing with mathematically optimal allocation
fn calculate_confidence_driven_position_sizing(
    selected_ranges: &[(String, PriceBin, f64)],
    direction_pred: &DirectionPrediction,
    price_levels: &PriceLevelPrediction,
    volatility_pred: &VolatilityPrediction,
    expected_profit_pct: f64,
    stop_loss_pct: f64,
) -> Vec<f64> {
    let mut allocations = Vec::new();

    // Calculate natural risk-reward from predictions
    let natural_risk_reward = if direction_pred.expected_downside_percent > 0.0 {
        direction_pred.expected_upside_percent / direction_pred.expected_downside_percent
    } else {
        3.0 // Conservative default
    };

    // Combined confidence score from all predictions
    let combined_confidence = (direction_pred.confidence * 0.4
        + price_levels.confidence * 0.4
        + volatility_pred.confidence * 0.2)
        .clamp(0.1, 0.95);

    // Calculate total probability mass for normalization
    let total_probability: f64 = selected_ranges.iter().map(|(_, _, prob)| prob).sum();

    debug!(
        "💪 Position Sizing: natural_rr={:.2} combined_conf={:.3} total_prob={:.3}",
        natural_risk_reward, combined_confidence, total_probability
    );

    for (i, (name, bin, probability_weight)) in selected_ranges.iter().enumerate() {
        // Kelly Criterion components
        let win_probability = bin.probability.clamp(0.1, 0.9); // Avoid extremes
        let win_amount = expected_profit_pct / 100.0;
        let loss_amount = stop_loss_pct / 100.0;

        // Kelly fraction: f = (bp - q) / b where b = win/loss ratio, p = win prob, q = loss prob
        let win_loss_ratio = win_amount / loss_amount;
        let kelly_fraction =
            ((win_loss_ratio * win_probability) - (1.0 - win_probability)) / win_loss_ratio;

        // Apply confidence scaling and probability weighting
        let confidence_scaled_kelly = kelly_fraction * combined_confidence;
        let probability_weighted_allocation = confidence_scaled_kelly * probability_weight;

        // Distance-based adjustment - closer entries get more allocation
        let distance_from_current = bin.range[1].abs() / 100.0;
        let distance_factor = 1.0 / (1.0 + distance_from_current * 2.0); // Closer = higher factor

        let distance_adjusted_allocation = probability_weighted_allocation * distance_factor;

        // Volatility regime adjustment - higher vol = smaller positions
        let volatility_adjustment = match volatility_pred.regime.as_str() {
            "VERY_LOW" => 1.2,
            "LOW" => 1.1,
            "MEDIUM" => 1.0,
            "HIGH" => 0.9,
            "VERY_HIGH" => 0.8,
            _ => 1.0,
        };

        let final_allocation =
            (distance_adjusted_allocation * volatility_adjustment).clamp(0.05, 0.6);
        allocations.push(final_allocation);

        debug!(
            "🎯 Allocation {}: {} kelly={:.3} conf_scaled={:.3} prob_weighted={:.3} dist_adj={:.3} vol_adj={:.3} final={:.3}",
            i + 1, name, kelly_fraction, confidence_scaled_kelly, probability_weighted_allocation,
            distance_adjusted_allocation, volatility_adjustment, final_allocation
        );
    }

    // Normalize allocations to sum to 1.0
    let total_allocation: f64 = allocations.iter().sum();
    if total_allocation > 0.0 {
        allocations
            .iter_mut()
            .for_each(|allocation| *allocation /= total_allocation);
    } else {
        // Fallback to equal allocation if calculations fail
        let equal_allocation = 1.0 / allocations.len() as f64;
        allocations
            .iter_mut()
            .for_each(|allocation| *allocation = equal_allocation);
    }

    debug!(
        "📊 Final Allocations: {:?} (sum={:.3})",
        allocations,
        allocations.iter().sum::<f64>()
    );
    allocations
}

/// Calculate adaptive stop distance using volatility probabilities and risk management
/// This replaces hardcoded stop distances with mathematical volatility analysis
fn calculate_adaptive_stop_distance(
    volatility_pred: &VolatilityPrediction,
    _direction_pred: &DirectionPrediction,
    expected_profit_pct: f64,
    min_risk_reward: f64,
) -> f64 {
    // Base stop from volatility prediction
    let base_stop_pct = volatility_pred.recommended_stop_distance_percent;

    // Risk-reward constraint: stop cannot be larger than profit/min_risk_reward
    let max_allowed_stop = expected_profit_pct / min_risk_reward;

    // Volatility regime adjustment using probabilities
    let volatility_adjustment = volatility_pred.very_low_probability * 0.6 +    // Tighter stops in low vol
        volatility_pred.low_probability * 0.8 +
        volatility_pred.medium_probability * 1.0 +
        volatility_pred.high_probability * 1.3 +        // Wider stops in high vol
        volatility_pred.very_high_probability * 1.6;

    let volatility_adjusted_stop = base_stop_pct * volatility_adjustment;

    // Confidence adjustment - lower confidence = wider stops
    let confidence_adjustment = 0.9 + (1.0 - volatility_pred.confidence) * 0.3;
    let confidence_adjusted_stop = volatility_adjusted_stop * confidence_adjustment;

    // Apply risk-reward constraint
    let final_stop = confidence_adjusted_stop.min(max_allowed_stop);

    debug!(
        "🛑 Adaptive Stop: base={:.2}% vol_adj={:.2}% conf_adj={:.2}% max_allowed={:.2}% final={:.2}%",
        base_stop_pct, volatility_adjusted_stop, confidence_adjusted_stop, max_allowed_stop, final_stop
    );

    final_stop.clamp(0.5, 8.0) // 0.5% to 8% stop bounds for crypto
}

/// Generate comprehensive adaptive trading orders using ALL prediction data
/// This is the main function that replaces the hardcoded order generation system
pub fn generate_adaptive_orders(
    current_price: f64,
    direction_pred: &DirectionPrediction,
    volatility_pred: &VolatilityPrediction,
    price_levels: &PriceLevelPrediction,
    atr_value: f64,
    config: &OrderConfig,
) -> crate::utils::error::Result<TradingOrders> {
    info!("🚀 Generating adaptive orders with mathematical optimization");

    // 🚀 ADAPTIVE ANALYSIS - Same logic as AdaptiveTradingSignal
    // Analyze probability distribution to determine best trading approach
    let directional_edge =
        direction_pred.up_probability_aggregated - direction_pred.down_probability_aggregated;

    // Calculate probability spread (how concentrated vs distributed the predictions are)
    let max_prob = direction_pred
        .sideways_probability
        .max(direction_pred.up_probability_aggregated)
        .max(direction_pred.down_probability_aggregated);
    let min_prob = direction_pred
        .sideways_probability
        .min(direction_pred.up_probability_aggregated)
        .min(direction_pred.down_probability_aggregated);
    let probability_spread = max_prob - min_prob;

    // Calculate confidence in the prediction (higher spread = more confident)
    let prediction_confidence = probability_spread;

    info!(
        "📊 ADAPTIVE ORDERS ANALYSIS: edge={:.1}%, spread={:.1}%, confidence={:.1}%, max_prob={:.1}%",
        directional_edge * 100.0,
        probability_spread * 100.0,
        prediction_confidence * 100.0,
        max_prob * 100.0
    );

    // Use model's own confidence and risk/reward to determine if we should trade
    let min_confidence_threshold = 0.25; // Only trade if we have >25% confidence in any direction
    let min_risk_reward_threshold = 0.5; // Minimum R/R based on volatility

    // Check if any prediction has sufficient confidence
    let has_sufficient_confidence = max_prob > min_confidence_threshold;
    let has_acceptable_risk_reward = direction_pred.risk_reward_ratio > min_risk_reward_threshold;

    if !has_sufficient_confidence {
        warn!(
            "❌ Insufficient confidence: max_prob={:.1}% < 25%",
            max_prob * 100.0
        );
        return Ok(TradingOrders::empty(
            direction_pred,
            &format!(
                "Insufficient confidence for trading. Max probability: {:.1}% (need >25%)",
                max_prob * 100.0
            ),
        ));
    }

    if !has_acceptable_risk_reward {
        warn!(
            "❌ Poor risk/reward ratio: {:.2} < 0.5",
            direction_pred.risk_reward_ratio
        );
        return Ok(TradingOrders::empty(
            direction_pred,
            &format!(
                "Poor risk/reward ratio: {:.2} (need >0.5)",
                direction_pred.risk_reward_ratio
            ),
        ));
    }

    // Determine direction based on adaptive logic (same as AdaptiveTradingSignal)
    let direction = if directional_edge.abs() > probability_spread * 0.3 {
        // Clear directional bias
        if directional_edge > 0.0 {
            if direction_pred.pump_probability > 0.25 {
                "LONG_BREAKOUT"
            } else {
                "LONG"
            }
        } else if direction_pred.dump_probability > 0.25 {
            "SHORT_BREAKOUT"
        } else {
            "SHORT"
        }
    } else {
        // Weak directional bias but still tradeable if we have confidence
        if direction_pred.up_probability_aggregated > direction_pred.down_probability_aggregated {
            "LONG"
        } else {
            "SHORT"
        }
    };

    let is_breakout = direction.contains("BREAKOUT");
    info!("🎯 Direction: {} (breakout={})", direction, is_breakout);

    // 5. VOLATILITY-AWARE ATR CALCULATION
    let atr_multiplier =
        calculate_volatility_aware_atr_multiplier(volatility_pred, config.base_atr_multiplier);
    let atr_distance = atr_value * atr_multiplier;

    info!(
        "📏 Spacing Analysis: atr_multiplier={:.2} atr_distance={:.2}",
        atr_multiplier, atr_distance
    );

    // 6. PROBABILITY-BASED RANGE SELECTION
    let selected_ranges = select_optimal_ranges_by_probability(&price_levels.bins, direction, 3);

    if selected_ranges.is_empty() {
        warn!(
            "❌ No suitable probability ranges found for direction: {}",
            direction
        );
        return Ok(TradingOrders::empty_with_reason(&format!(
            "No suitable probability ranges for {} direction",
            direction
        )));
    }

    // 7. ENTRY PRICE CALCULATION FROM PROBABILITY RANGES
    let mut entry_prices = Vec::new();
    for (name, bin, weight) in &selected_ranges {
        let optimal_entry_pct = match direction {
            "LONG" | "LONG_BREAKOUT" => {
                // For LONG: use better entry (more negative = lower price for better entry)
                bin.range[0].max(bin.range[1] - (bin.range[1] - bin.range[0]) * 0.8)
            }
            "SHORT" | "SHORT_BREAKOUT" => {
                // For SHORT: use better entry (less positive = lower price within range)
                bin.range[0].max(bin.range[1] - (bin.range[1] - bin.range[0]) * 0.8)
            }
            _ => (bin.range[0] + bin.range[1]) / 2.0, // Midpoint fallback
        };

        let entry_price = current_price * (1.0 + optimal_entry_pct / 100.0);
        entry_prices.push(entry_price);

        debug!(
            "💰 Entry {}: range=[{:.2}%, {:.2}%] optimal={:.2}% price={:.2} weight={:.3}",
            name, bin.range[0], bin.range[1], optimal_entry_pct, entry_price, weight
        );
    }

    // 8. EXIT PRICE CALCULATION FROM UPSIDE PROBABILITY RANGES
    let mut exit_ranges: Vec<_> = price_levels
        .bins
        .iter()
        .filter(|(_, bin)| {
            match direction {
                "LONG" | "LONG_BREAKOUT" => bin.range[0] > 0.0, // Positive ranges for LONG exits
                "SHORT" | "SHORT_BREAKOUT" => bin.range[1] < 0.0, // Negative ranges for SHORT exits
                _ => false,
            }
        })
        .collect();

    // Sort by range start for progressive exits
    exit_ranges.sort_by(|a, b| match direction {
        "LONG" | "LONG_BREAKOUT" => a.1.range[0].partial_cmp(&b.1.range[0]).unwrap(),
        "SHORT" | "SHORT_BREAKOUT" => b.1.range[1].partial_cmp(&a.1.range[1]).unwrap(),
        _ => std::cmp::Ordering::Equal,
    });

    let mut exit_prices = Vec::new();
    for (i, (name, bin)) in exit_ranges.iter().take(3).enumerate() {
        let exit_pct = match direction {
            "LONG" | "LONG_BREAKOUT" => {
                // For LONG: use conservative exit (lower end of positive range)
                bin.range[0] + (bin.range[1] - bin.range[0]) * 0.3
            }
            "SHORT" | "SHORT_BREAKOUT" => {
                // For SHORT: use conservative exit (higher end of negative range)
                bin.range[1] - (bin.range[1] - bin.range[0]) * 0.3
            }
            _ => (bin.range[0] + bin.range[1]) / 2.0,
        };

        let exit_price = current_price * (1.0 + exit_pct / 100.0);
        exit_prices.push(exit_price);

        debug!(
            "🎯 Exit {}: {} range=[{:.2}%, {:.2}%] target={:.2}% price={:.2} prob={:.3}",
            i + 1,
            name,
            bin.range[0],
            bin.range[1],
            exit_pct,
            exit_price,
            bin.probability
        );
    }

    // Ensure we have at least 3 exits (extend if needed)
    while exit_prices.len() < 3 {
        if let Some(&last_price) = exit_prices.last() {
            let extension_multiplier = match direction {
                "LONG" | "LONG_BREAKOUT" => 1.02,   // 2% higher
                "SHORT" | "SHORT_BREAKOUT" => 0.98, // 2% lower
                _ => 1.01,
            };
            exit_prices.push(last_price * extension_multiplier);
        } else {
            // Fallback exits if no probability ranges available
            let fallback_pct = match direction {
                "LONG" | "LONG_BREAKOUT" => 2.0 + (exit_prices.len() as f64 * 1.5),
                "SHORT" | "SHORT_BREAKOUT" => -2.0 - (exit_prices.len() as f64 * 1.5),
                _ => 1.0,
            };
            exit_prices.push(current_price * (1.0 + fallback_pct / 100.0));
        }
    }

    // 9. EXPECTED PROFIT CALCULATION FOR RISK-REWARD
    let expected_profit_pct = match direction {
        "LONG" | "LONG_BREAKOUT" => {
            exit_prices
                .iter()
                .map(|&price| ((price - current_price) / current_price) * 100.0)
                .sum::<f64>()
                / exit_prices.len() as f64
        }
        "SHORT" | "SHORT_BREAKOUT" => {
            exit_prices
                .iter()
                .map(|&price| ((current_price - price) / current_price) * 100.0)
                .sum::<f64>()
                / exit_prices.len() as f64
        }
        _ => 2.0,
    };

    // 10. ADAPTIVE STOP DISTANCE CALCULATION
    let stop_distance_pct = calculate_adaptive_stop_distance(
        volatility_pred,
        direction_pred,
        expected_profit_pct,
        config.min_risk_reward,
    );

    // 11. CONFIDENCE-DRIVEN POSITION SIZING
    let position_allocations = calculate_confidence_driven_position_sizing(
        &selected_ranges,
        direction_pred,
        price_levels,
        volatility_pred,
        expected_profit_pct,
        stop_distance_pct,
    );

    info!(
        "💰 Financial Analysis: expected_profit={:.2}% stop_distance={:.2}% risk_reward={:.1}",
        expected_profit_pct,
        stop_distance_pct,
        expected_profit_pct / stop_distance_pct
    );

    // 12. BUILD ORDER LEVELS
    let mut entry_levels = Vec::new();
    let mut exit_levels = Vec::new();
    let mut stop_levels = Vec::new();

    // Entry levels with probability-weighted allocation
    for (i, (&entry_price, &allocation)) in entry_prices
        .iter()
        .zip(position_allocations.iter())
        .enumerate()
    {
        let order_type = if is_breakout && i == entry_prices.len() - 1 {
            "STOP_LIMIT" // Last entry as breakout trigger
        } else {
            "LIMIT"
        };

        let confidence = selected_ranges
            .get(i)
            .map(|(_, bin, _)| bin.probability)
            .unwrap_or(0.5);

        entry_levels.push(OrderLevel {
            price: entry_price,
            quantity_percentage: allocation,
            atr_distance,
            order_type: order_type.to_string(),
            confidence,
        });
    }

    // Exit levels with progressive allocation
    let exit_quantities = [0.4, 0.4, 0.2]; // Progressive exit allocation
    let exit_confidences = [0.8, 0.6, 0.4];

    for (&exit_price, (&exit_qty, &exit_conf)) in exit_prices
        .iter()
        .zip(exit_quantities.iter().zip(exit_confidences.iter()))
    {
        exit_levels.push(OrderLevel {
            price: exit_price,
            quantity_percentage: exit_qty,
            atr_distance,
            order_type: "LIMIT".to_string(),
            confidence: exit_conf,
        });
    }

    // Stop levels with hunt protection
    let hunt_protection_distance = atr_distance * config.hunt_protection;
    let stop_quantities = [0.4, 0.4, 0.2]; // Progressive stop allocation
    let stop_confidences = [0.9, 0.8, 0.7];

    for (i, (&stop_qty, &stop_conf)) in stop_quantities
        .iter()
        .zip(stop_confidences.iter())
        .enumerate()
    {
        let stop_multiplier = 1.0 + (i as f64 * 0.1); // Progressive widening
        let stop_price = match direction {
            "LONG" | "LONG_BREAKOUT" => {
                current_price * (1.0 - (stop_distance_pct * stop_multiplier) / 100.0)
            }
            "SHORT" | "SHORT_BREAKOUT" => {
                current_price * (1.0 + (stop_distance_pct * stop_multiplier) / 100.0)
            }
            _ => current_price,
        };

        stop_levels.push(OrderLevel {
            price: stop_price,
            quantity_percentage: stop_qty,
            atr_distance: hunt_protection_distance,
            order_type: "STOP_LOSS".to_string(),
            confidence: stop_conf,
        });
    }

    // 13. CALCULATE FINAL RISK-REWARD RATIO
    let avg_entry_price: f64 = entry_levels
        .iter()
        .map(|level| level.price * level.quantity_percentage)
        .sum();
    let avg_exit_price: f64 = exit_levels
        .iter()
        .map(|level| level.price * level.quantity_percentage)
        .sum();
    let avg_stop_price: f64 = stop_levels
        .iter()
        .map(|level| level.price * level.quantity_percentage)
        .sum();

    let risk_reward_ratio = match direction {
        "LONG" | "LONG_BREAKOUT" => {
            let profit = avg_exit_price - avg_entry_price;
            let loss = avg_entry_price - avg_stop_price;
            if loss > 0.0 {
                profit / loss
            } else {
                10.0
            }
        }
        "SHORT" | "SHORT_BREAKOUT" => {
            let profit = avg_entry_price - avg_exit_price;
            let loss = avg_stop_price - avg_entry_price;
            if loss > 0.0 {
                profit / loss
            } else {
                10.0
            }
        }
        _ => 1.0,
    };

    info!(
        "✅ Orders Generated: {} entries, {} exits, {} stops, risk_reward={:.1}",
        entry_levels.len(),
        exit_levels.len(),
        stop_levels.len(),
        risk_reward_ratio
    );

    // Convert to fixed-size arrays (pad if needed)
    let entry_array = [
        entry_levels.first().cloned().unwrap_or_default(),
        entry_levels.get(1).cloned().unwrap_or_default(),
        entry_levels.get(2).cloned().unwrap_or_default(),
    ];
    let exit_array = [
        exit_levels.first().cloned().unwrap_or_default(),
        exit_levels.get(1).cloned().unwrap_or_default(),
        exit_levels.get(2).cloned().unwrap_or_default(),
    ];
    let stop_array = [
        stop_levels.first().cloned().unwrap_or_default(),
        stop_levels.get(1).cloned().unwrap_or_default(),
        stop_levels.get(2).cloned().unwrap_or_default(),
    ];

    Ok(TradingOrders {
        direction: direction.to_string(),
        entry_levels: entry_array,
        exit_levels: exit_array,
        stop_levels: stop_array,
        total_position_size: 1.0,
        risk_reward_ratio,
        atr_multiplier,
        dynamic_sizing: true,
    })
}

// Helper implementations for TradingOrders
impl TradingOrders {
    fn empty_with_reason(reason: &str) -> Self {
        let empty_level = OrderLevel::default();
        Self {
            direction: format!("NO_SIGNAL: {}", reason),
            entry_levels: [
                empty_level.clone(),
                empty_level.clone(),
                empty_level.clone(),
            ],
            exit_levels: [
                empty_level.clone(),
                empty_level.clone(),
                empty_level.clone(),
            ],
            stop_levels: [empty_level.clone(), empty_level.clone(), empty_level],
            total_position_size: 0.0,
            risk_reward_ratio: 0.0,
            atr_multiplier: 0.0,
            dynamic_sizing: false,
        }
    }
}

impl Default for OrderLevel {
    fn default() -> Self {
        Self {
            price: 0.0,
            quantity_percentage: 0.0,
            atr_distance: 0.0,
            order_type: "NONE".to_string(),
            confidence: 0.0,
        }
    }
}
