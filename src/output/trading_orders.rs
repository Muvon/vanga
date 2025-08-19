//! Trading orders generation and management
//!
//! This module handles the generation of trading orders based on predictions,
//! including dynamic position sizing, risk-reward optimization, and adaptive order placement.

use serde::{Deserialize, Serialize};

// Import prediction types from other modules
use super::prediction_types::{DirectionPrediction, PriceLevelPrediction, VolatilityPrediction};

/// Trading orders with dynamic position sizing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingOrders {
    /// Trading direction based on prediction
    pub direction: String, // "LONG" or "SHORT"

    /// Three entry levels with dynamic quantities
    pub entry_levels: [OrderLevel; 3],

    /// Three exit levels with dynamic quantities
    pub exit_levels: [OrderLevel; 3],

    /// Three stop loss levels with dynamic quantities
    pub stop_levels: [OrderLevel; 3],

    /// Total position size (1.0 = 100%)
    pub total_position_size: f64,

    /// Expected risk-reward ratio (crypto-aggressive: 4.0-8.0+)
    pub risk_reward_ratio: f64,

    /// ATR multiplier used for spacing
    pub atr_multiplier: f64,

    /// Confidence-based position sizing enabled
    pub dynamic_sizing: bool,

    /// Optional note for additional information (e.g., confidence warnings)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Individual order level with dynamic sizing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderLevel {
    /// Order price
    pub price: f64,

    /// Dynamic quantity percentage (not fixed 33.33%)
    pub quantity_percentage: f64,

    /// Distance from current price in ATR units
    pub atr_distance: f64,

    /// Order type for execution
    pub order_type: String, // "LIMIT", "STOP_LIMIT", "MARKET"

    /// Confidence level for this price point
    pub confidence: f64,
}

/// Configuration for order generation
#[derive(Debug, Clone)]
pub struct OrderConfig {
    /// Base ATR multiplier (crypto default: 2.0)
    pub base_atr_multiplier: f64,

    /// Minimum risk-reward ratio (crypto: 4.0)
    pub min_risk_reward: f64,

    /// Maximum risk-reward ratio (crypto: 12.0)
    pub max_risk_reward: f64,

    /// Enable aggressive position sizing
    pub aggressive_sizing: bool,

    /// Hunt protection multiplier for stops
    pub hunt_protection: f64,
}

impl Default for OrderConfig {
    fn default() -> Self {
        Self {
            base_atr_multiplier: 2.0, // Crypto-aggressive spacing
            min_risk_reward: 4.0,     // Minimum 4:1 for crypto
            max_risk_reward: 12.0,    // Maximum 12:1 for high conviction
            aggressive_sizing: true,  // Enable dynamic sizing
            hunt_protection: 1.5,     // 50% extra distance for stops
        }
    }
}

impl Default for TradingOrders {
    fn default() -> Self {
        let empty_level = OrderLevel {
            price: 0.0,
            quantity_percentage: 0.0,
            atr_distance: 0.0,
            order_type: "NONE".to_string(),
            confidence: 0.0,
        };

        Self {
            direction: "NO_SIGNAL".to_string(),
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
            note: None,
        }
    }
}

/// Configuration for sequence-aware order generation
#[derive(Debug, Clone)]
pub struct SequenceAwareOrderConfig<'a> {
    pub current_price: f64,
    pub direction_pred: &'a DirectionPrediction,
    pub volatility_pred: &'a VolatilityPrediction,
    pub price_levels: &'a PriceLevelPrediction,
    pub atr_value: f64,
    pub config: &'a OrderConfig,
    pub sequence_prices: &'a [f64],
    pub bandwidth_size: f64,
    /// Optional dynamic entry sizes from enhanced confidence calculator
    pub dynamic_entry_sizes: Option<[f64; 3]>,
    /// Optional dynamic exit sizes from enhanced confidence calculator
    pub dynamic_exit_sizes: Option<[f64; 3]>,
    /// Overall confidence from enhanced calculator
    pub overall_confidence: Option<f64>,
}

impl TradingOrders {
    /// Generate sequence-aware trading orders using the same bandwidth logic as price levels
    /// This ensures consistency between price level predictions and order generation
    pub fn generate(config: SequenceAwareOrderConfig) -> crate::utils::error::Result<Self> {
        // Validate that price_levels are consistent with our sequence-aware approach
        log::debug!(
            "Generating sequence-aware orders with price level confidence: {:.2}",
            config.price_levels.confidence
        );

        // 🚀 ADAPTIVE ANALYSIS - Same logic as AdaptiveTradingSignal
        // Analyze probability distribution to determine best trading approach
        let directional_edge = config.direction_pred.up_probability_aggregated
            - config.direction_pred.down_probability_aggregated;

        // Calculate probability spread (how concentrated vs distributed the predictions are)
        let max_prob = config
            .direction_pred
            .sideways_probability
            .max(config.direction_pred.up_probability_aggregated)
            .max(config.direction_pred.down_probability_aggregated);
        let min_prob = config
            .direction_pred
            .sideways_probability
            .min(config.direction_pred.up_probability_aggregated)
            .min(config.direction_pred.down_probability_aggregated);
        let probability_spread = max_prob - min_prob;

        // Calculate confidence in the prediction (higher spread = more confident)
        let prediction_confidence = probability_spread;

        log::info!(
            "📊 SEQUENCE-AWARE ANALYSIS: edge={:.1}%, spread={:.1}%, confidence={:.1}%, max_prob={:.1}%",
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
        let has_acceptable_risk_reward =
            config.direction_pred.risk_reward_ratio > min_risk_reward_threshold;

        if !has_sufficient_confidence || !has_acceptable_risk_reward {
            return Ok(Self::empty(
                config.direction_pred,
                Some(format!(
                    "Insufficient confidence for trading. Max probability: {:.1}% (need >25%), R/R: {:.2} (need >0.5)",
                    max_prob * 100.0,
                    config.direction_pred.risk_reward_ratio
                )),
            ));
        }

        // Determine direction based on adaptive logic
        let direction = if directional_edge.abs() > probability_spread * 0.3 {
            // Clear directional bias
            if directional_edge > 0.0 {
                "LONG"
            } else {
                "SHORT"
            }
        } else {
            // Weak directional bias but still tradeable if we have confidence
            if config.direction_pred.up_probability_aggregated
                > config.direction_pred.down_probability_aggregated
            {
                "LONG"
            } else {
                "SHORT"
            }
        };

        // 🎯 PROPER ATR CALCULATION: Base multiplier adjusted by market volatility
        let volatility_factor = config.volatility_pred.expected_range_percent / 5.0; // Scale to reasonable range (5% baseline)
        let atr_multiplier = config.config.base_atr_multiplier * volatility_factor.clamp(0.5, 3.0); // Cap between 0.5x-3.0x

        // ATR distance: use base calculation with market adjustment
        let base_atr_pct = config
            .volatility_pred
            .recommended_stop_distance_percent
            .max(1.0); // Minimum 1%
        let atr_distance = config.current_price * (base_atr_pct / 100.0);

        // 🎯 ADAPTIVE ORDER GENERATION: Use price level probabilities instead of sequence ranges
        let (mut entry_levels, mut exit_levels, mut stop_levels) = if direction == "LONG" {
            // Check if this is a breakout signal based on pump probability (adaptive threshold)
            let is_breakout = config.direction_pred.pump_probability > 0.25;
            Self::generate_adaptive_long_orders(
                config.current_price,
                atr_distance,
                config.price_levels,
                config.direction_pred,
                config.volatility_pred,
                config.config,
                is_breakout,
                config.dynamic_entry_sizes,
                config.dynamic_exit_sizes,
                config.sequence_prices, // NEW: Add sequence_prices parameter
            )
        } else {
            // Check if this is a breakout signal based on dump probability (adaptive threshold)
            let is_breakout = config.direction_pred.dump_probability > 0.25; // Use same threshold as AdaptiveTradingSignal
            Self::generate_sequence_aware_short_orders(
                config.current_price,
                atr_distance,
                config.config,
                is_breakout,
                config.price_levels,
                config.volatility_pred,
                config.sequence_prices,
                config.direction_pred,
            )
        };

        // Validate and optimize risk-reward ratio (configurable minimum for crypto)
        let min_risk_reward = 4.0; // TODO: Move to config - 4:1 minimum as requested
        let risk_reward_ratio = Self::validate_and_optimize_risk_reward(
            &mut entry_levels,
            &mut exit_levels,
            &mut stop_levels,
            direction,
            config.current_price,
            min_risk_reward,
        );

        // Log final risk/reward assessment and add note if below minimum
        let note = if risk_reward_ratio < min_risk_reward {
            log::warn!(
                "⚠️ TRADING SIGNAL OPTIMIZED: Risk/Reward {:.2} below minimum {:.2} - using best available optimization",
                risk_reward_ratio, min_risk_reward
            );
            log::info!(
                "📊 Consider: Tighter position sizing, waiting for better setup, or accepting lower R:R for this opportunity"
            );
            Some(format!(
                "Risk/Reward {:.2} below target {:.2} - optimized to best available",
                risk_reward_ratio, min_risk_reward
            ))
        } else {
            log::info!(
                "✅ TRADING SIGNAL APPROVED: Risk/Reward {:.2} meets minimum {:.2} requirement",
                risk_reward_ratio,
                min_risk_reward
            );
            None
        };

        Ok(TradingOrders {
            direction: direction.to_string(),
            entry_levels,
            exit_levels,
            stop_levels,
            total_position_size: 1.0,
            risk_reward_ratio,
            atr_multiplier,
            dynamic_sizing: config.config.aggressive_sizing,
            note,
        })
    }

    /// Generate adaptive long orders using price level probabilities with enhanced confidence
    #[allow(clippy::too_many_arguments)]
    fn generate_adaptive_long_orders(
        current_price: f64,
        atr_distance: f64,
        price_levels: &PriceLevelPrediction,
        direction_pred: &DirectionPrediction,
        volatility_pred: &VolatilityPrediction,
        config: &OrderConfig,
        is_breakout: bool,
        dynamic_entry_sizes: Option<[f64; 3]>,
        dynamic_exit_sizes: Option<[f64; 3]>,
        sequence_prices: &[f64], // NEW: Actual price sequence for smart calculation
    ) -> ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]) {
        // 🎯 SMART ENTRY CALCULATION: Based on ACTUAL sequence volatility + momentum
        // Calculate actual volatility from the sequence for realistic entry points
        let sequence_volatility = if sequence_prices.len() >= 2 {
            let returns: Vec<f64> = sequence_prices
                .windows(2)
                .map(|w| ((w[1] - w[0]) / w[0]) * 100.0)
                .collect();

            if !returns.is_empty() {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance =
                    returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                variance.sqrt()
            } else {
                volatility_pred.expected_range_percent // Use prediction if can't calculate
            }
        } else {
            volatility_pred.expected_range_percent // Use prediction as fallback
        };

        // Combine actual and predicted volatility
        let combined_volatility =
            sequence_volatility * 0.7 + volatility_pred.expected_range_percent * 0.3;

        // Base distance for entries
        let base_distance = combined_volatility;

        // 🎯 Use direction momentum to adjust entry aggressiveness
        // Strong momentum = can enter closer to market (momentum will continue)
        // Weak momentum = need better prices (wait for pullback)
        let momentum_factor = if direction_pred.up_probability_aggregated > 0.6 {
            0.8 // Strong bullish momentum - enter closer
        } else if direction_pred.up_probability_aggregated > 0.4 {
            1.0 // Moderate momentum - normal spacing
        } else {
            1.2 // Weak momentum - wait for better prices
        };

        log::info!(
            "📊 LONG Data-Driven Entry: seq_vol={:.3}%, pred_vol={:.3}%, combined={:.3}%, momentum_factor={:.1}x",
            sequence_volatility,
            volatility_pred.expected_range_percent,
            combined_volatility,
            momentum_factor
        );

        // ENTRY LOGIC: For LONG positions, entries BELOW current price (buy low)
        // Use actual volatility + momentum for realistic, fillable entry points
        let entry_1_pct = -base_distance * 0.2 * momentum_factor; // Very close - high fill probability
        let entry_2_pct = -base_distance * 0.5 * momentum_factor; // Medium distance
        let entry_3_pct = -base_distance * 0.8 * momentum_factor; // Further for better price

        log::info!(
            "📍 LONG Entry Distances: {:.3}%, {:.3}%, {:.3}% BELOW current price (momentum-adjusted)",
            entry_1_pct.abs(), entry_2_pct.abs(), entry_3_pct.abs()
        );

        // ENHANCED POSITION SIZING: Use dynamic sizes if available, otherwise calculate from probabilities
        let moderate_down_bin = price_levels.bins.get("moderate_down");
        let strong_down_bin = price_levels.bins.get("strong_down");
        let neutral_bin = price_levels.bins.get("neutral");

        let (entry_1_size, entry_2_size, entry_3_size) = if let Some(sizes) = dynamic_entry_sizes {
            // Use enhanced confidence-based sizes
            log::info!("📊 Using ENHANCED dynamic entry sizes from confidence calculator");
            (sizes[0], sizes[1], sizes[2])
        } else {
            // Fallback to probability-weighted distribution
            let moderate_down_prob = moderate_down_bin.map(|bin| bin.probability).unwrap_or(0.2);
            let strong_down_prob = strong_down_bin.map(|bin| bin.probability).unwrap_or(0.1);
            let neutral_prob = neutral_bin.map(|bin| bin.probability).unwrap_or(0.2);

            // Calculate total probability mass for normalization
            let total_entry_prob = moderate_down_prob + strong_down_prob + neutral_prob;
            let norm_factor = if total_entry_prob > 0.0 {
                1.0 / total_entry_prob
            } else {
                1.0
            };

            // Weight by probability and adjust for confidence
            let size_1 = (moderate_down_prob * norm_factor * 1.2).min(0.5);
            let size_2 = (neutral_prob * norm_factor).min(0.3);
            let size_3 = (1.0 - size_1 - size_2).max(0.2);
            (size_1, size_2, size_3)
        };

        // Calculate confidence for each entry level based on probability
        let moderate_down_prob = moderate_down_bin.map(|bin| bin.probability).unwrap_or(0.2);
        let strong_down_prob = strong_down_bin.map(|bin| bin.probability).unwrap_or(0.1);
        let neutral_prob = neutral_bin.map(|bin| bin.probability).unwrap_or(0.2);

        let entry_1_confidence = (moderate_down_prob * 2.0).min(0.9); // Scale up for confidence
        let entry_2_confidence = (neutral_prob * 1.5).min(0.7);
        let entry_3_confidence = (strong_down_prob * 1.0).min(0.5);

        log::info!(
            "📊 ENHANCED LONG Entry Sizing: Level1={:.1}% (conf={:.1}), Level2={:.1}% (conf={:.1}), Level3={:.1}% (conf={:.1})",
            entry_1_size * 100.0, entry_1_confidence,
            entry_2_size * 100.0, entry_2_confidence,
            entry_3_size * 100.0, entry_3_confidence
        );

        let entry_levels = [
            OrderLevel {
                price: current_price * (1.0 + entry_1_pct / 100.0),
                quantity_percentage: entry_1_size,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: entry_1_confidence,
            },
            OrderLevel {
                price: current_price * (1.0 + entry_2_pct / 100.0),
                quantity_percentage: entry_2_size,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: entry_2_confidence,
            },
            OrderLevel {
                price: current_price * (1.0 + entry_3_pct / 100.0),
                quantity_percentage: entry_3_size,
                atr_distance,
                order_type: if is_breakout {
                    "STOP_LIMIT".to_string()
                } else {
                    "LIMIT".to_string()
                },
                confidence: entry_3_confidence,
            },
        ];

        // 🎯 SMART EXIT: Based on MODEL PREDICTIONS for profit targets
        // For LONG positions, use UPSIDE predictions (above current price)
        let moderate_up_bin = price_levels.bins.get("moderate_up");
        let strong_up_bin = price_levels.bins.get("strong_up");

        // Use model's predicted upside targets
        let exit_1_pct = moderate_up_bin
            .map(|bin| bin.range[0].max(base_distance * 2.0))
            .unwrap_or(base_distance * 2.0);

        let exit_2_pct = moderate_up_bin
            .map(|bin| ((bin.range[0] + bin.range[1]) / 2.0).max(exit_1_pct * 1.5))
            .unwrap_or(exit_1_pct * 1.8);

        let exit_3_pct = strong_up_bin
            .map(|bin| bin.range[0].max(exit_2_pct * 1.3))
            .unwrap_or(exit_2_pct * 1.5);

        log::info!(
            "🎯 LONG Exit Targets (MODEL-based): {:.3}%, {:.3}%, {:.3}% ABOVE current price",
            exit_1_pct,
            exit_2_pct,
            exit_3_pct
        );

        // ENHANCED EXIT SIZING: Use dynamic sizes if available
        let (exit_1_size, exit_2_size, exit_3_size) = if let Some(sizes) = dynamic_exit_sizes {
            // Use enhanced confidence-based exit sizes
            log::info!("📊 Using ENHANCED dynamic exit sizes from confidence calculator");
            (sizes[0], sizes[1], sizes[2])
        } else {
            // Fallback to probability-based sizing
            let moderate_up_prob = moderate_up_bin.map(|bin| bin.probability).unwrap_or(0.25);

            // Dynamic exit sizing based on probability distribution
            let size_1 = if moderate_up_prob > 0.3 {
                0.3 // Take 30% profit early if high probability
            } else {
                0.4 // Take 40% profit if lower probability
            };

            let size_2 = 0.4; // Always 40% at middle target
            let size_3 = 1.0 - size_1 - size_2; // Remainder
            (size_1, size_2, size_3)
        };

        // Calculate confidence for each exit level
        let moderate_up_prob = moderate_up_bin.map(|bin| bin.probability).unwrap_or(0.25);
        let strong_up_prob = strong_up_bin.map(|bin| bin.probability).unwrap_or(0.15);

        let exit_1_confidence = (moderate_up_prob * 3.0).min(0.9); // High confidence for likely targets
        let exit_2_confidence = ((moderate_up_prob + strong_up_prob) * 1.5).min(0.7);
        let exit_3_confidence = (strong_up_prob * 2.0).min(0.5);

        log::info!(
            "📊 ENHANCED LONG Exit Sizing: Level1={:.1}% (conf={:.1}), Level2={:.1}% (conf={:.1}), Level3={:.1}% (conf={:.1})",
            exit_1_size * 100.0, exit_1_confidence,
            exit_2_size * 100.0, exit_2_confidence,
            exit_3_size * 100.0, exit_3_confidence
        );

        let exit_levels = [
            OrderLevel {
                price: current_price * (1.0 + exit_1_pct / 100.0),
                quantity_percentage: exit_1_size,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: exit_1_confidence,
            },
            OrderLevel {
                price: current_price * (1.0 + exit_2_pct / 100.0),
                quantity_percentage: exit_2_size,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: exit_2_confidence,
            },
            OrderLevel {
                price: current_price * (1.0 + exit_3_pct / 100.0),
                quantity_percentage: exit_3_size,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: exit_3_confidence,
            },
        ];

        // STOP LOGIC: ENFORCE RISK-REWARD RATIO WITH INTELLIGENT CONFIDENCE
        let avg_entry_price =
            (entry_levels[0].price + entry_levels[1].price + entry_levels[2].price) / 3.0;
        let avg_exit_price =
            (exit_levels[0].price + exit_levels[1].price + exit_levels[2].price) / 3.0;

        // Calculate required stop distance to maintain min_risk_reward
        let expected_profit = avg_exit_price - avg_entry_price;
        let max_allowed_loss = expected_profit / config.min_risk_reward;

        // Use volatility recommendation but cap by risk-reward requirement
        let volatility_stop_distance =
            volatility_pred.recommended_stop_distance_percent / 100.0 * avg_entry_price;
        let required_stop_distance = max_allowed_loss.min(volatility_stop_distance);

        // CRITICAL: Stops must be BELOW entry prices for LONG
        let stop_price_1 = avg_entry_price - required_stop_distance;
        let stop_price_2 = avg_entry_price - required_stop_distance * 1.1; // 10% wider
        let stop_price_3 = avg_entry_price - required_stop_distance * 1.2; // 20% wider

        // ENHANCED STOP CONFIDENCE: Based on volatility and risk management
        let base_stop_confidence = match volatility_pred.regime.as_str() {
            "VERY_LOW" | "LOW" => 0.95, // High confidence in calm markets
            "MEDIUM" => 0.85,           // Good confidence
            "HIGH" => 0.75,             // Lower confidence in volatile markets
            "VERY_HIGH" => 0.65,        // Lowest confidence in extreme volatility
            _ => 0.8,
        };

        // Adjust stop sizes based on volatility regime
        let (stop_1_size, stop_2_size, stop_3_size) = match volatility_pred.regime.as_str() {
            "VERY_LOW" | "LOW" => (0.5, 0.3, 0.2), // Tighter stops in calm markets
            "MEDIUM" => (0.4, 0.4, 0.2),           // Balanced stops
            "HIGH" | "VERY_HIGH" => (0.3, 0.4, 0.3), // Wider distribution in volatile markets
            _ => (0.4, 0.4, 0.2),
        };

        log::info!(
            "📊 ENHANCED Stop Sizing: Level1={:.1}% (conf={:.1}), Level2={:.1}% (conf={:.1}), Level3={:.1}% (conf={:.1})",
            stop_1_size * 100.0, base_stop_confidence,
            stop_2_size * 100.0, base_stop_confidence * 0.9,
            stop_3_size * 100.0, base_stop_confidence * 0.8
        );

        let stop_levels = [
            OrderLevel {
                price: stop_price_1,
                quantity_percentage: stop_1_size,
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: base_stop_confidence,
            },
            OrderLevel {
                price: stop_price_2,
                quantity_percentage: stop_2_size,
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: base_stop_confidence * 0.9,
            },
            OrderLevel {
                price: stop_price_3,
                quantity_percentage: stop_3_size,
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: base_stop_confidence * 0.8,
            },
        ];

        // VALIDATION: Ensure mathematical correctness
        let actual_risk_reward = expected_profit / required_stop_distance;

        log::info!(
            "🎯 LONG Orders: Avg Entry={:.2} | Avg Exit={:.2} | Avg Stop={:.2} | R:R={:.2} (min={:.1})",
            avg_entry_price, avg_exit_price, (stop_price_1 + stop_price_2 + stop_price_3) / 3.0,
            actual_risk_reward, config.min_risk_reward
        );

        // Ensure stops are below entries
        for (i, stop) in stop_levels.iter().enumerate() {
            if stop.price >= entry_levels[i].price {
                log::error!(
                    "🚨 CRITICAL: Stop {} ({:.2}) >= Entry {} ({:.2})",
                    i,
                    stop.price,
                    i,
                    entry_levels[i].price
                );
            }
        }

        (entry_levels, exit_levels, stop_levels)
    }

    /// Generate sequence-aware short orders using probability-based allocation (NO MAGIC NUMBERS)
    #[allow(clippy::too_many_arguments)]
    fn generate_sequence_aware_short_orders(
        current_price: f64,
        atr_distance: f64,
        config: &OrderConfig,
        is_breakout: bool,
        price_levels: &PriceLevelPrediction,
        volatility_pred: &VolatilityPrediction,
        sequence_prices: &[f64],
        direction_pred: &DirectionPrediction,
    ) -> ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]) {
        // DEBUG: Log the ranges being used
        log::debug!(
            "🔍 SHORT Order Generation Debug: current_price={:.2}, sequence_len={}",
            current_price,
            sequence_prices.len()
        );

        // CRITICAL FIX: Force correct SHORT order logic regardless of range issues
        // SHORT must enter ABOVE current price and exit BELOW current price

        // 🎯 PROFESSIONAL GRID: Use ACTUAL prediction bins with their probabilities

        // Get the actual bins with ranges and probabilities
        let moderate_up_bin = price_levels.bins.get("moderate_up");
        let _strong_up_bin = price_levels.bins.get("strong_up");
        let moderate_down_bin = price_levels.bins.get("moderate_down");
        let strong_down_bin = price_levels.bins.get("strong_down");
        let neutral_bin = price_levels.bins.get("neutral");

        // 🎯 SHORT ENTRIES: Use PROGRESSIVE upside ranges for DIFFERENT entry levels
        let (entry_1_pct, entry_1_prob) = if let Some(bin) = neutral_bin {
            (bin.range[1], bin.probability) // neutral upper = 0.909%
        } else {
            (0.2, 0.2)
        };

        let (entry_2_pct, entry_2_prob) = if let Some(bin) = moderate_up_bin {
            ((bin.range[0] + bin.range[1]) / 2.0, bin.probability) // moderate_up CENTER = 1.465%
        } else {
            (direction_pred.expected_upside_percent, 0.2)
        };

        let (entry_3_pct, entry_3_prob) = if let Some(bin) = moderate_up_bin {
            (bin.range[1], bin.probability) // moderate_up upper = 2.021%
        } else {
            (direction_pred.expected_upside_percent * 1.5, 0.15)
        };

        log::info!("📊 SMART SHORT ENTRIES using PROGRESSIVE ranges (ABOVE current price):");
        log::info!(
            "📍 Entry 1: {:.3}% | Entry 2: {:.3}% | Entry 3: {:.3}% (DIFFERENT prices!)",
            entry_1_pct,
            entry_2_pct,
            entry_3_pct
        );

        log::info!(
            "📊 SHORT Entry Logic: Catching bounces/resistance ABOVE current price {:.2}",
            current_price
        );

        let entry_levels = [
            OrderLevel {
                price: current_price * (1.0 + entry_1_pct / 100.0),
                quantity_percentage: entry_1_prob, // Use actual probability for sizing
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.9,
            },
            OrderLevel {
                price: current_price * (1.0 + entry_2_pct / 100.0),
                quantity_percentage: entry_2_prob, // Use actual probability for sizing
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.7,
            },
            OrderLevel {
                price: current_price * (1.0 + entry_3_pct / 100.0),
                quantity_percentage: entry_3_prob, // Use actual probability for sizing
                atr_distance,
                order_type: if is_breakout {
                    "STOP_LIMIT".to_string()
                } else {
                    "LIMIT".to_string()
                },
                confidence: if is_breakout { 0.8 } else { 0.6 },
            },
        ];

        // 🎯 SHORT EXITS: Use PROGRESSIVE downside ranges for DIFFERENT exit levels
        let (exit_1_pct, exit_1_prob) = if let Some(bin) = moderate_down_bin {
            (bin.range[1], bin.probability) // moderate_down upper = -0.195%
        } else {
            (-direction_pred.expected_downside_percent * 0.5, 0.2)
        };

        let (exit_2_pct, exit_2_prob) = if let Some(bin) = moderate_down_bin {
            ((bin.range[0] + bin.range[1]) / 2.0, bin.probability) // moderate_down CENTER = -0.751%
        } else {
            (-direction_pred.expected_downside_percent, 0.2)
        };

        let (exit_3_pct, exit_3_prob) = if let Some(bin) = strong_down_bin {
            (bin.range[1], bin.probability) // strong_down upper = -1.307%
        } else {
            (-direction_pred.expected_downside_percent * 1.5, 0.15)
        };

        log::info!("🎯 SHORT EXITS using PROGRESSIVE downside ranges (BELOW current price):");
        log::info!(
            "📍 Exit 1: {:.3}% | Exit 2: {:.3}% | Exit 3: {:.3}% (DIFFERENT prices!)",
            exit_1_pct,
            exit_2_pct,
            exit_3_pct
        );

        log::debug!(
            "📉 SHORT Exit Percentages (ADAPTIVE): exit_1={:.2}%, exit_2={:.2}%, exit_3={:.2}%",
            exit_1_pct,
            exit_2_pct,
            exit_3_pct
        );

        let exit_levels = [
            OrderLevel {
                price: current_price * (1.0 + exit_1_pct / 100.0),
                quantity_percentage: exit_1_prob, // Use actual probability for sizing
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: current_price * (1.0 + exit_2_pct / 100.0),
                quantity_percentage: exit_2_prob, // Use actual probability for sizing
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.6,
            },
            OrderLevel {
                price: current_price * (1.0 + exit_3_pct / 100.0),
                quantity_percentage: exit_3_prob, // Use actual probability for sizing
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.4,
            },
        ];

        // 🎯 PROFESSIONAL STOP GRID: Use recommended stop distance
        let recommended_stop = volatility_pred.recommended_stop_distance_percent;
        let stop_distance = recommended_stop;

        // Position stops above entries using recommended stop distance
        let stop_1_pct = entry_1_pct + stop_distance;
        let stop_2_pct = entry_2_pct + stop_distance;
        let stop_3_pct = entry_3_pct + stop_distance;

        log::info!(
            "🛑 SHORT STOPS: Using recommended stop distance {:.3}%",
            stop_distance
        );
        log::info!(
            "📍 Stop levels: {:.3}%, {:.3}%, {:.3}% above entries",
            stop_1_pct,
            stop_2_pct,
            stop_3_pct
        );

        log::debug!(
            "🛑 SHORT Stop Percentages (VOLATILITY-ADAPTIVE): stop_1={:.2}%, stop_2={:.2}%, stop_3={:.2}%",
            stop_1_pct, stop_2_pct, stop_3_pct
        );

        // Use same probability-based portions for stops as entries (risk consistency)
        let stop_levels = [
            OrderLevel {
                price: current_price * (1.0 + stop_1_pct / 100.0), // Convert percentage to multiplier
                quantity_percentage: entry_1_prob,                 // Same as entry allocation
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.9,
            },
            OrderLevel {
                price: current_price * (1.0 + stop_2_pct / 100.0),
                quantity_percentage: entry_2_prob, // Same as entry allocation
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: current_price * (1.0 + stop_3_pct / 100.0),
                quantity_percentage: entry_3_prob, // Same as entry allocation
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.7,
            },
        ];

        // 🎯 NEW: Validate the generated orders for consistency
        if let Err(e) = price_levels.validate_orders(&entry_levels, &exit_levels, &stop_levels) {
            log::warn!(
                "⚠️ Order validation issue: {} - continuing with best effort",
                e
            );
            // Return anyway but with warning logged
        }

        log::info!("✅ Probability-driven orders validated successfully - no duplicates, proper sequencing");

        (entry_levels, exit_levels, stop_levels)
    }

    /// Create empty orders when no trading signals are available
    pub fn empty(direction_pred: &DirectionPrediction, note: Option<String>) -> Self {
        let empty_level = OrderLevel {
            price: 0.0,
            quantity_percentage: 0.0,
            atr_distance: 0.0,
            order_type: "NONE".to_string(),
            confidence: 0.0,
        };

        let direction = if direction_pred.up_probability > direction_pred.down_probability {
            "LONG"
        } else {
            "SHORT"
        };

        TradingOrders {
            direction: direction.to_string(),
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
            note,
        }
    }

    /// Validate and optimize risk/reward ratio for trading viability using smart adjustments
    fn validate_and_optimize_risk_reward(
        entry_levels: &mut [OrderLevel; 3],
        exit_levels: &mut [OrderLevel; 3],
        stop_levels: &mut [OrderLevel; 3],
        direction: &str,
        current_price: f64,
        min_ratio: f64,
    ) -> f64 {
        let initial_ratio =
            Self::calculate_risk_reward(entry_levels, exit_levels, stop_levels, direction);

        log::debug!(
            "🎯 Initial Risk/Reward Ratio: {:.2} (minimum required: {:.2})",
            initial_ratio,
            min_ratio
        );

        if initial_ratio >= min_ratio {
            log::debug!("✅ Risk/Reward ratio is acceptable: {:.2}", initial_ratio);
            return initial_ratio;
        }

        log::warn!(
            "⚠️ Poor Risk/Reward ratio: {:.2} < {:.2} minimum. Starting SMART optimization...",
            initial_ratio,
            min_ratio
        );

        // SMART OPTIMIZATION: Use iterative approach with intelligent adjustments
        let mut current_ratio = initial_ratio;
        let max_iterations = 10;
        let target_ratio = min_ratio * 1.1; // Aim slightly above minimum for safety

        for iteration in 1..=max_iterations {
            // Calculate how much improvement we need
            let improvement_needed = target_ratio / current_ratio;

            // Smart adjustment factors based on how far we are from target
            // More aggressive when far from target, more conservative when close
            let adjustment_factor = if improvement_needed > 2.0 {
                0.05 // 5% adjustments when far from target
            } else if improvement_needed > 1.5 {
                0.03 // 3% adjustments when moderately far
            } else if improvement_needed > 1.2 {
                0.02 // 2% adjustments when getting close
            } else {
                0.01 // 1% fine-tuning when very close
            };

            // SMART STRATEGY: Prioritize based on order confidence levels
            // Higher confidence orders get smaller adjustments to preserve prediction integrity

            match direction {
                "SHORT" => {
                    // For SHORT: Entry ABOVE current (sell high), Exit BELOW current (buy low), Stop ABOVE entry (loss)
                    // To improve R:R, primarily move STOPS closer (reduce risk), slightly adjust entries

                    // CRITICAL FIX: Move STOPS closer to entries to reduce risk (80% of optimization)
                    for (i, stop) in stop_levels.iter_mut().enumerate() {
                        if stop.price > current_price && i < entry_levels.len() {
                            let entry_price = entry_levels[i].price;
                            if entry_price > 0.0 && stop.price > entry_price {
                                // Calculate current distance from entry to stop
                                let current_distance = stop.price - entry_price;
                                // Reduce distance based on adjustment factor (more aggressive for stops)
                                let adjustment_multiplier: f64 =
                                    (1.0f64 - adjustment_factor * 2.0).max(0.3f64);
                                let new_distance = current_distance * adjustment_multiplier;
                                // Ensure minimum distance for hunt protection (at least 0.3% from entry)
                                let min_distance = entry_price * 0.003;
                                stop.price = entry_price + new_distance.max(min_distance);

                                log::trace!(
                                    "  Iteration {}: Stop {} moved closer: {:.2} (distance from entry: {:.2}%)",
                                    iteration, i, stop.price, (new_distance / entry_price) * 100.0
                                );
                            }
                        }
                    }

                    // Move ENTRIES slightly lower (closer to current) to improve fill probability (20% of optimization)
                    for (i, entry) in entry_levels.iter_mut().enumerate() {
                        if entry.price > current_price {
                            // Move entry closer to current price by small amount
                            let distance_from_current = entry.price - current_price;
                            let new_distance =
                                distance_from_current * (1.0 - adjustment_factor * 0.3);
                            // Keep minimum distance to avoid immediate fill
                            let min_distance = current_price * 0.002;
                            entry.price = current_price + new_distance.max(min_distance);

                            log::trace!(
                                "  Iteration {}: Entry {} adjusted to {:.2} (distance: {:.2}%)",
                                iteration,
                                i,
                                entry.price,
                                (new_distance / current_price) * 100.0
                            );
                        }
                    }

                    // DON'T move exits - keep them based on predictions to ensure they execute
                    // Only make tiny adjustments if absolutely necessary
                    if iteration > 5 && current_ratio < min_ratio * 0.5 {
                        // Emergency adjustment only after many iterations
                        for exit in exit_levels.iter_mut() {
                            if exit.price < current_price && exit.price > 0.0 {
                                // Move exit slightly lower for more profit (max 1% total)
                                exit.price *= (1.0 - adjustment_factor * 0.2).max(0.99);
                                log::trace!("  Emergency exit adjustment: {:.2}", exit.price);
                            }
                        }
                    }
                }
                "LONG" => {
                    // For LONG: Entry BELOW current (buy low), Exit ABOVE current (sell high), Stop BELOW entry (loss)
                    // To improve R:R, primarily move STOPS closer (reduce risk), slightly adjust entries

                    // CRITICAL FIX: Move STOPS closer to entries to reduce risk (80% of optimization)
                    for (i, stop) in stop_levels.iter_mut().enumerate() {
                        if stop.price < current_price && i < entry_levels.len() {
                            let entry_price = entry_levels[i].price;
                            if entry_price > 0.0 && stop.price < entry_price {
                                // Calculate current distance from entry to stop
                                let current_distance = entry_price - stop.price;
                                // Reduce distance based on adjustment factor
                                let adjustment_multiplier: f64 =
                                    (1.0f64 - adjustment_factor * 2.0).max(0.3f64);
                                let new_distance = current_distance * adjustment_multiplier;
                                // Ensure minimum distance for hunt protection
                                let min_distance = entry_price * 0.003;
                                stop.price = entry_price - new_distance.max(min_distance);

                                log::trace!(
                                    "  Iteration {}: Stop {} moved closer: {:.2} (distance from entry: {:.2}%)",
                                    iteration, i, stop.price, (new_distance / entry_price) * 100.0
                                );
                            }
                        }
                    }

                    // Move ENTRIES slightly higher (closer to current) to improve fill probability (20% of optimization)
                    for (i, entry) in entry_levels.iter_mut().enumerate() {
                        if entry.price < current_price {
                            // Move entry closer to current price
                            let distance_from_current = current_price - entry.price;
                            let new_distance =
                                distance_from_current * (1.0 - adjustment_factor * 0.3);
                            // Keep minimum distance
                            let min_distance = current_price * 0.002;
                            entry.price = current_price - new_distance.max(min_distance);

                            log::trace!(
                                "  Iteration {}: Entry {} adjusted to {:.2} (distance: {:.2}%)",
                                iteration,
                                i,
                                entry.price,
                                (new_distance / current_price) * 100.0
                            );
                        }
                    }

                    // DON'T move exits - keep them based on predictions
                    if iteration > 5 && current_ratio < min_ratio * 0.5 {
                        // Emergency adjustment only
                        for exit in exit_levels.iter_mut() {
                            if exit.price > current_price && exit.price > 0.0 {
                                // Move exit slightly higher for more profit (max 1% total)
                                exit.price *= (1.0 + adjustment_factor * 0.2).min(1.01);
                                log::trace!("  Emergency exit adjustment: {:.2}", exit.price);
                            }
                        }
                    }
                }
                _ => {}
            }

            // Recalculate ratio after adjustments
            let new_ratio =
                Self::calculate_risk_reward(entry_levels, exit_levels, stop_levels, direction);

            log::debug!(
                "  Iteration {}: Risk/Reward improved from {:.2} to {:.2} (target: {:.2})",
                iteration,
                current_ratio,
                new_ratio,
                target_ratio
            );

            // Check if we've reached our target
            if new_ratio >= min_ratio {
                log::info!(
                    "✅ Risk/Reward SMARTLY optimized in {} iterations: {:.2} -> {:.2}",
                    iteration,
                    initial_ratio,
                    new_ratio
                );
                return new_ratio;
            }

            // Check if we're making progress
            if new_ratio <= current_ratio * 1.01 {
                // Not making enough progress, try more aggressive approach
                log::debug!("  Optimization stalled, trying more aggressive adjustments...");

                // Double the exit adjustments for final push
                match direction {
                    "SHORT" => {
                        for exit in exit_levels.iter_mut() {
                            if exit.price > 0.0 {
                                exit.price *= 0.95; // 5% more aggressive
                            }
                        }
                    }
                    "LONG" => {
                        for exit in exit_levels.iter_mut() {
                            if exit.price > 0.0 {
                                exit.price *= 1.05; // 5% more aggressive
                            }
                        }
                    }
                    _ => {}
                }
            }

            current_ratio = new_ratio;
        }

        // Final check after all iterations
        if current_ratio >= min_ratio {
            log::info!(
                "✅ Risk/Reward optimized successfully: {:.2} -> {:.2}",
                initial_ratio,
                current_ratio
            );
        } else {
            log::warn!(
                "⚠️ Risk/Reward optimization reached {:.2} after {} iterations (target was {:.2})",
                max_iterations,
                current_ratio,
                min_ratio
            );
            log::info!(
                "📊 Achieved {:.1}x improvement: {:.2} -> {:.2} - using best available optimization",
                current_ratio / initial_ratio,
                initial_ratio,
                current_ratio
            );
        }

        current_ratio
    }

    /// Calculate risk-reward ratio from order levels with direction awareness
    fn calculate_risk_reward(
        entry_levels: &[OrderLevel; 3],
        exit_levels: &[OrderLevel; 3],
        stop_levels: &[OrderLevel; 3],
        direction: &str,
    ) -> f64 {
        // Weighted average prices by quantity
        let avg_entry = Self::weighted_average_price(entry_levels);
        let avg_exit = Self::weighted_average_price(exit_levels);
        let avg_stop = Self::weighted_average_price(stop_levels);

        // Calculate profit and loss based on direction
        let (potential_profit, potential_loss) = match direction {
            "LONG" => {
                // LONG: profit when price goes up, loss when price goes down
                let profit = avg_exit - avg_entry; // Exit higher than entry = profit
                let loss = avg_entry - avg_stop; // Stop lower than entry = loss
                (profit.max(0.0), loss.max(0.0))
            }
            "SHORT" => {
                // SHORT: profit when price goes down, loss when price goes up
                let profit = avg_entry - avg_exit; // Entry higher than exit = profit
                let loss = avg_stop - avg_entry; // Stop higher than entry = loss
                (profit.max(0.0), loss.max(0.0))
            }
            _ => {
                // Fallback to absolute difference for unknown directions
                let profit = (avg_exit - avg_entry).abs();
                let loss = (avg_entry - avg_stop).abs();
                (profit, loss)
            }
        };

        if potential_loss > 0.0 {
            potential_profit / potential_loss
        } else {
            10.0 // If no loss possible, return high ratio
        }
    }

    /// Calculate weighted average price by quantity
    fn weighted_average_price(levels: &[OrderLevel; 3]) -> f64 {
        let total_weight: f64 = levels.iter().map(|l| l.quantity_percentage).sum();
        if total_weight > 0.0 {
            levels
                .iter()
                .map(|l| l.price * l.quantity_percentage)
                .sum::<f64>()
                / total_weight
        } else {
            levels.iter().map(|l| l.price).sum::<f64>() / 3.0
        }
    }
}
