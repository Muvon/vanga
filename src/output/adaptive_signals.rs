//! Adaptive trading signals generation
//!
//! This module handles the generation of adaptive trading signals that work across
//! different time horizons and market conditions, using probability-based decision making.

use serde::{Deserialize, Serialize};

// Import prediction types from other modules
use super::prediction_types::{DirectionPrediction, VolatilityPrediction};

/// Generate adaptive trading signal that works for any horizon
pub fn generate_adaptive_trading_signal(
    direction_pred: &DirectionPrediction,
    volatility_pred: &VolatilityPrediction,
    current_price: f64,
) -> AdaptiveTradingSignal {
    // 1. BREAKOUT SIGNALS (works for any horizon)
    if direction_pred.breakout_probability > 0.4 {
        if direction_pred.pump_probability > 0.25 {
            return AdaptiveTradingSignal::StrongLong {
                entry_price: current_price,
                target_price: current_price
                    * (1.0 + direction_pred.expected_upside_percent / 100.0),
                stop_loss: current_price
                    * (1.0 - volatility_pred.recommended_stop_distance_percent / 100.0),
                position_size: volatility_pred.position_size_multiplier * 1.3, // Boost for breakouts
                horizon: direction_pred.training_horizon.clone(),
                risk_reward: direction_pred.risk_reward_ratio,
                confidence: direction_pred.pump_probability,
            };
        }
        if direction_pred.dump_probability > 0.25 {
            return AdaptiveTradingSignal::StrongShort {
                entry_price: current_price,
                target_price: current_price
                    * (1.0 - direction_pred.expected_downside_percent / 100.0),
                stop_loss: current_price
                    * (1.0 + volatility_pred.recommended_stop_distance_percent / 100.0),
                position_size: volatility_pred.position_size_multiplier * 1.3,
                horizon: direction_pred.training_horizon.clone(),
                risk_reward: direction_pred.risk_reward_ratio,
                confidence: direction_pred.dump_probability,
            };
        }
    }

    // 2. INTELLIGENT ADAPTIVE SIGNAL GENERATION
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

    log::info!(
        "📊 ADAPTIVE ANALYSIS: edge={:.1}%, spread={:.1}%, confidence={:.1}%, max_prob={:.1}%",
        directional_edge * 100.0,
        probability_spread * 100.0,
        prediction_confidence * 100.0,
        max_prob * 100.0
    );

    // 🚀 ADAPTIVE DECISION LOGIC - No hardcoded thresholds!
    // Use model's own confidence and risk/reward to determine if we should trade

    // Minimum confidence threshold based on model's own assessment
    let min_confidence_threshold = 0.25; // Only trade if we have >25% confidence in any direction
    let min_risk_reward_threshold = 0.8; // Minimum R/R based on volatility

    // Check if any prediction has sufficient confidence
    let has_sufficient_confidence = max_prob > min_confidence_threshold;
    let has_acceptable_risk_reward = direction_pred.risk_reward_ratio > min_risk_reward_threshold;

    if has_sufficient_confidence && has_acceptable_risk_reward {
        // Determine the best trading strategy based on probability distribution

        // Strategy 1: Clear directional bias (one aggregated probability significantly higher)
        if directional_edge.abs() > probability_spread * 0.3 {
            // Edge is significant relative to spread
            if directional_edge > 0.0 {
                log::info!(
                    "📈 DIRECTIONAL LONG: edge={:.1}% is significant",
                    directional_edge * 100.0
                );
                return AdaptiveTradingSignal::Long {
                    entry_price: current_price,
                    target_price: current_price
                        * (1.0 + direction_pred.expected_upside_percent / 100.0),
                    stop_loss: current_price
                        * (1.0 - volatility_pred.recommended_stop_distance_percent / 100.0),
                    position_size: volatility_pred.position_size_multiplier
                        * (1.0 + prediction_confidence),
                    horizon: direction_pred.training_horizon.clone(),
                    risk_reward: direction_pred.risk_reward_ratio,
                    confidence: direction_pred.up_probability_aggregated,
                };
            } else {
                log::info!(
                    "📉 DIRECTIONAL SHORT: edge={:.1}% is significant",
                    directional_edge * 100.0
                );
                return AdaptiveTradingSignal::Short {
                    entry_price: current_price,
                    target_price: current_price
                        * (1.0 - direction_pred.expected_downside_percent / 100.0),
                    stop_loss: current_price
                        * (1.0 + volatility_pred.recommended_stop_distance_percent / 100.0),
                    position_size: volatility_pred.position_size_multiplier
                        * (1.0 + prediction_confidence),
                    horizon: direction_pred.training_horizon.clone(),
                    risk_reward: direction_pred.risk_reward_ratio,
                    confidence: direction_pred.down_probability_aggregated,
                };
            }
        }

        // Strategy 2: SIDEWAYS is competitive (within reasonable range of other probabilities)
        // Even if not dominant, if SIDEWAYS is close to other probabilities, it's a valid strategy
        let sideways_competitiveness = direction_pred.sideways_probability / max_prob;
        if sideways_competitiveness > 0.8 {
            // SIDEWAYS is within 20% of the highest probability

            // Determine bias within sideways movement
            let sideways_direction = if direction_pred.up_probability_aggregated
                > direction_pred.down_probability_aggregated
            {
                "LONG"
            } else {
                "SHORT"
            };

            let _bias_strength = directional_edge.abs();

            // Use sequence bandwidth for more conservative targets in sideways markets
            let target_percent = if sideways_direction == "LONG" {
                (direction_pred.expected_upside_percent / 100.0)
                    .min(direction_pred.sequence_bandwidth_percent / 100.0 * 0.6)
            } else {
                (direction_pred.expected_downside_percent / 100.0)
                    .min(direction_pred.sequence_bandwidth_percent / 100.0 * 0.6)
            };

            let stop_percent = (direction_pred.sequence_bandwidth_percent / 100.0 * 0.4)
                .max(volatility_pred.recommended_stop_distance_percent / 100.0);

            let target_price = if sideways_direction == "LONG" {
                current_price * (1.0 + target_percent)
            } else {
                current_price * (1.0 - target_percent)
            };

            let stop_loss = if sideways_direction == "LONG" {
                current_price * (1.0 - stop_percent)
            } else {
                current_price * (1.0 + stop_percent)
            };

            let risk_reward = if stop_percent > 0.0 {
                target_percent / stop_percent
            } else {
                0.0
            };

            log::info!(
                "🔄 SIDEWAYS COMPETITIVE: sideways={:.1}% (competitiveness={:.1}%), bias={}, R/R={:.2}",
                direction_pred.sideways_probability * 100.0,
                sideways_competitiveness * 100.0,
                sideways_direction,
                risk_reward
            );

            if sideways_direction == "LONG" {
                return AdaptiveTradingSignal::SidewaysLong {
                    entry_price: current_price,
                    target_price,
                    stop_loss,
                    position_size: volatility_pred.position_size_multiplier
                        * (0.7 + prediction_confidence * 0.3),
                    horizon: direction_pred.training_horizon.clone(),
                    risk_reward,
                    confidence: direction_pred.sideways_probability,
                    sideways_probability: direction_pred.sideways_probability,
                    sequence_bias: format!("{}_BIAS", sideways_direction),
                };
            } else {
                return AdaptiveTradingSignal::SidewaysShort {
                    entry_price: current_price,
                    target_price,
                    stop_loss,
                    position_size: volatility_pred.position_size_multiplier
                        * (0.7 + prediction_confidence * 0.3),
                    horizon: direction_pred.training_horizon.clone(),
                    risk_reward,
                    confidence: direction_pred.sideways_probability,
                    sideways_probability: direction_pred.sideways_probability,
                    sequence_bias: format!("{}_BIAS", sideways_direction),
                };
            }
        }

        // Strategy 3: Weak directional bias but still tradeable
        // If we have acceptable confidence but no clear strategy above, trade the bias
        if max_prob > 0.3 {
            // At least 30% confidence in some direction
            if direction_pred.up_probability_aggregated > direction_pred.down_probability_aggregated
            {
                log::info!(
                    "📈 WEAK LONG: up_agg={:.1}% > down_agg={:.1}%",
                    direction_pred.up_probability_aggregated * 100.0,
                    direction_pred.down_probability_aggregated * 100.0
                );
                return AdaptiveTradingSignal::Long {
                    entry_price: current_price,
                    target_price: current_price
                        * (1.0 + direction_pred.expected_upside_percent / 100.0),
                    stop_loss: current_price
                        * (1.0 - volatility_pred.recommended_stop_distance_percent / 100.0),
                    position_size: volatility_pred.position_size_multiplier * 0.8, // Reduced size for weak signals
                    horizon: direction_pred.training_horizon.clone(),
                    risk_reward: direction_pred.risk_reward_ratio,
                    confidence: direction_pred.up_probability_aggregated,
                };
            } else {
                log::info!(
                    "📉 WEAK SHORT: down_agg={:.1}% > up_agg={:.1}%",
                    direction_pred.down_probability_aggregated * 100.0,
                    direction_pred.up_probability_aggregated * 100.0
                );
                return AdaptiveTradingSignal::Short {
                    entry_price: current_price,
                    target_price: current_price
                        * (1.0 - direction_pred.expected_downside_percent / 100.0),
                    stop_loss: current_price
                        * (1.0 + volatility_pred.recommended_stop_distance_percent / 100.0),
                    position_size: volatility_pred.position_size_multiplier * 0.8,
                    horizon: direction_pred.training_horizon.clone(),
                    risk_reward: direction_pred.risk_reward_ratio,
                    confidence: direction_pred.down_probability_aggregated,
                };
            }
        }
    }

    AdaptiveTradingSignal::NoSignal {
        reason: format!(
            "Insufficient confidence for trading. Max probability: {:.1}% (need >25%), R/R: {:.2} (need >0.8), Spread: {:.1}%",
            max_prob * 100.0,
            direction_pred.risk_reward_ratio,
            probability_spread * 100.0
        ),
        horizon: direction_pred.training_horizon.clone(),
        confidence: direction_pred.confidence,
    }
}

/// Horizon-agnostic adaptive trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptiveTradingSignal {
    StrongLong {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
    },
    Long {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
    },
    StrongShort {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
    },
    Short {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
    },
    SidewaysLong {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
        sideways_probability: f64,
        sequence_bias: String, // "LONG_BIAS" or "SHORT_BIAS"
    },
    SidewaysShort {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
        sideways_probability: f64,
        sequence_bias: String, // "LONG_BIAS" or "SHORT_BIAS"
    },
    NoSignal {
        reason: String,
        horizon: String,
        confidence: f64,
    },
}
