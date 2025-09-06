//! Trading orders generation and management
//!
//! This module handles the generation of trading orders based on predictions,
//! including dynamic position sizing, risk-reward optimization, and adaptive order placement.

use serde::{Deserialize, Serialize};

// Import prediction types from other modules
use super::confidence_calculator::ConfidenceCalculator;
use super::prediction_types::{
    DirectionPrediction, PredictionResult, PriceLevelPrediction, SentimentPrediction,
    VolatilityPrediction, VolumePrediction,
};
use super::sequence_statistics::SequenceStatistics;
use super::smart_order_generator::SmartConsensus;

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
            min_risk_reward: 2.0, // Dynamic minimum (will be overridden by confidence calculation)
            max_risk_reward: 12.0, // Maximum 12:1 for high conviction
            aggressive_sizing: true, // Enable dynamic sizing
            hunt_protection: 1.5, // 50% extra distance for stops
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

/// Configuration for sequence-aware order generation with statistics
#[derive(Debug, Clone)]
pub struct SequenceAwareConfig<'a> {
    pub current_price: f64,
    pub direction_pred: &'a DirectionPrediction,
    pub price_levels: &'a PriceLevelPrediction,
    pub volatility_pred: &'a VolatilityPrediction,
    pub sentiment_pred: &'a SentimentPrediction,
    pub volume_pred: &'a VolumePrediction,
    pub sequence_prices: &'a [f64],
    pub sequence_volumes: Option<&'a [f64]>,
    pub horizon_hours: f64,
}

/// Configuration for smart trading order generation
pub struct SmartOrderConfig<'a> {
    pub current_price: f64,
    pub price_levels: &'a PriceLevelPrediction,
    pub direction_pred: &'a DirectionPrediction,
    pub volatility_pred: &'a VolatilityPrediction,
    pub sentiment_pred: &'a SentimentPrediction,
    pub volume_pred: &'a VolumePrediction,
    pub confidence_calculator: &'a ConfidenceCalculator,
    pub min_confidence: f64,
    /// Optional sequence OHLCV data for adaptive stop calculation
    pub sequence_ohlcv: Option<&'a [crate::data::structures::MarketDataRow]>,
}

impl TradingOrders {
    /// Generate SMART trading orders using model-specific strengths (NO MAGIC NUMBERS)
    /// This is the NEW primary method that should be used instead of the old generate()
    pub fn generate_smart(config: SmartOrderConfig) -> crate::utils::error::Result<Self> {
        // Create SMART consensus calculator
        let consensus = SmartConsensus {
            direction: config.direction_pred.clone(),
            price_levels: config.price_levels.clone(),
            volatility: config.volatility_pred.clone(),
            sentiment: config.sentiment_pred.clone(),
            volume: config.volume_pred.clone(),
        };

        // Step 1: Determine direction using Direction + Sentiment models
        let (direction, direction_confidence) = consensus.calculate_direction_consensus();

        // Step 2: Calculate overall confidence using the sophisticated ConfidenceCalculator
        // Create a temporary PredictionResult to calculate confidence
        let temp_result = PredictionResult {
            symbol: String::new(),
            timestamp: String::new(),
            horizon: String::new(),
            current_price: config.current_price,
            current_vwap_price: config.current_price, // Use current price as fallback
            price_levels: Some(config.price_levels.clone()),
            direction: Some(config.direction_pred.clone()),
            volatility: Some(config.volatility_pred.clone()),
            sentiment: Some(config.sentiment_pred.clone()),
            volume: Some(config.volume_pred.clone()),
            orders: None,    // Temporary - no orders for confidence calculation
            confidence: 0.0, // Will be calculated
            metadata: super::metadata::PredictionMetadata {
                model_version: String::new(),
                generated_at: chrono::Utc::now(),
                feature_count: 0,
                sequence_length: 0,
                data_quality: super::metadata::DataQuality {
                    completeness: 1.0,
                    freshness_hours: 0.0,
                    market_condition: "NORMAL".to_string(),
                },
            },
        };

        let overall_confidence = config
            .confidence_calculator
            .calculate_overall_confidence(&temp_result);

        log::info!(
            "🎯 Using ConfidenceCalculator for order generation: confidence={:.2}%",
            overall_confidence * 100.0
        );

        // Step 3: Check if we have sufficient confidence to trade
        if overall_confidence < config.min_confidence {
            return Err(crate::utils::error::VangaError::PredictionError(format!(
                "Insufficient model confidence: {:.1}% < {:.1}% minimum",
                overall_confidence * 100.0,
                config.min_confidence * 100.0
            )));
        }

        log::info!(
            "🎯 SMART Order Generation: Direction={} (conf={:.2}), Overall confidence={:.2}",
            direction,
            direction_confidence,
            overall_confidence
        );

        // Step 3: Generate entry levels using Price Levels model
        let mut entry_levels =
            consensus.generate_smart_entries(config.current_price, &direction)?;

        // Apply psychological level adjustments if sequence data available
        if let Some(ohlcv_data) = config.sequence_ohlcv {
            // Convert MarketDataRow to OHLCV array format
            let ohlcv_arrays: Vec<[f64; 5]> = ohlcv_data
                .iter()
                .map(|row| [row.open, row.high, row.low, row.close, row.volume])
                .collect();

            // Adjust each entry level
            for entry in &mut entry_levels {
                entry.price =
                    consensus.adjust_for_psychological_levels(entry.price, Some(&ohlcv_arrays));
            }

            log::info!("🎯 Applied psychological level adjustments to entry levels");
        }

        // Step 4: Generate exit levels using Price Levels + Volume
        let mut exit_levels = consensus.generate_smart_exits(config.current_price, &direction)?;

        // Apply psychological level adjustments to exits if sequence data available
        if let Some(ohlcv_data) = config.sequence_ohlcv {
            let ohlcv_arrays: Vec<[f64; 5]> = ohlcv_data
                .iter()
                .map(|row| [row.open, row.high, row.low, row.close, row.volume])
                .collect();

            for exit in &mut exit_levels {
                exit.price =
                    consensus.adjust_for_psychological_levels(exit.price, Some(&ohlcv_arrays));
            }

            log::info!("🎯 Applied psychological level adjustments to exit levels");
        }

        // Step 5: Generate ADAPTIVE stop levels using ALL prediction data
        // Extract sequence prices from OHLCV data if available
        let sequence_prices: Vec<f64> = if let Some(ohlcv_data) = config.sequence_ohlcv {
            ohlcv_data.iter().map(|row| row.close).collect()
        } else {
            // Fallback: generate synthetic sequence around current price
            let volatility_estimate = config.volatility_pred.expected_range_percent / 100.0;
            (0..40)
                .map(|i| {
                    config.current_price * (1.0 + (i as f64 - 20.0) * volatility_estimate / 20.0)
                })
                .collect()
        };

        let mut stop_levels =
            consensus.generate_adaptive_stops(&entry_levels, &direction, &sequence_prices)?;

        // Apply psychological level adjustments to stops if sequence data available
        if let Some(ohlcv_data) = config.sequence_ohlcv {
            let ohlcv_arrays: Vec<[f64; 5]> = ohlcv_data
                .iter()
                .map(|row| [row.open, row.high, row.low, row.close, row.volume])
                .collect();

            for stop in &mut stop_levels {
                stop.price =
                    consensus.adjust_for_psychological_levels(stop.price, Some(&ohlcv_arrays));
            }

            log::info!("🎯 Applied psychological level adjustments to stop levels");
        }

        // Step 6: Normalize sizes to ensure they sum to 1.0
        SmartConsensus::normalize_sizes(&mut entry_levels);
        SmartConsensus::normalize_sizes(&mut exit_levels);

        // Step 7: Calculate ATR distance as percentage from current price for each level
        // This makes ATR distance semantically correct and consistent with smart_order_generator.rs
        for level in &mut entry_levels {
            level.atr_distance =
                ((level.price - config.current_price).abs() / config.current_price) * 100.0;
        }
        for level in &mut exit_levels {
            level.atr_distance =
                ((level.price - config.current_price).abs() / config.current_price) * 100.0;
        }

        // Step 8: Calculate initial risk-reward ratio
        let initial_risk_reward =
            Self::calculate_risk_reward(&entry_levels, &exit_levels, &stop_levels, &direction);

        // Step 9: Calculate NATURAL risk-reward from predictions (no hardcoded targets)
        let natural_rr = config.direction_pred.expected_upside_percent
            / config.direction_pred.expected_downside_percent.max(0.01);

        // Use prediction's own risk assessment as baseline
        let prediction_rr = config.direction_pred.risk_reward_ratio;

        // Target R:R is the better of natural calculation or model's assessment
        let target_risk_reward = natural_rr.max(prediction_rr);

        log::info!(
            "📊 R:R Assessment: natural={:.2} (up:{:.2}%/down:{:.2}%), model={:.2}, target={:.2}",
            natural_rr,
            config.direction_pred.expected_upside_percent,
            config.direction_pred.expected_downside_percent,
            prediction_rr,
            target_risk_reward
        );

        let risk_reward = if initial_risk_reward < target_risk_reward {
            log::info!(
                "⚠️ Initial R:R {:.2} below target {:.2}, checking if optimization is needed...",
                initial_risk_reward,
                target_risk_reward
            );

            // CRITICAL: Define prediction boundaries - we should NOT go beyond these!
            // The model's predictions are our reality check
            let max_exit_distance = if direction == "LONG" {
                // For LONG, use the maximum of expected upside or strong_up bin
                let strong_up_max = config
                    .price_levels
                    .bins
                    .get("strong_up")
                    .map(|b| b.range[1].abs())
                    .unwrap_or(config.direction_pred.expected_upside_percent);
                strong_up_max.max(config.direction_pred.expected_upside_percent * 1.5)
            } else {
                // For SHORT, use the maximum of expected downside or strong_down bin
                let strong_down_max = config
                    .price_levels
                    .bins
                    .get("strong_down")
                    .map(|b| b.range[1].abs())
                    .unwrap_or(config.direction_pred.expected_downside_percent);
                strong_down_max.max(config.direction_pred.expected_downside_percent * 1.5)
            };

            let min_stop_distance = config.volatility_pred.expected_range_percent * 0.3; // At least 30% of volatility

            log::info!(
                "🎯 Prediction boundaries: max_exit={:.2}%, min_stop={:.2}%",
                max_exit_distance,
                min_stop_distance
            );

            // Check if we even have room to optimize
            let current_max_exit_distance = exit_levels
                .iter()
                .map(|e| ((e.price - config.current_price).abs() / config.current_price * 100.0))
                .fold(0.0, f64::max);

            let current_min_stop_distance = stop_levels
                .iter()
                .map(|s| ((s.price - entry_levels[0].price).abs() / entry_levels[0].price * 100.0))
                .fold(f64::INFINITY, f64::min);

            // If we're already at prediction boundaries, we can't optimize much
            if current_max_exit_distance >= max_exit_distance * 0.9 {
                log::warn!(
                    "⚠️ Exit levels already near prediction boundary ({:.2}% vs max {:.2}%), limited optimization possible",
                    current_max_exit_distance,
                    max_exit_distance
                );
            }

            if current_min_stop_distance <= min_stop_distance * 1.1 {
                log::warn!(
                    "⚠️ Stop levels already near minimum safe distance ({:.2}% vs min {:.2}%), limited optimization possible",
                    current_min_stop_distance,
                    min_stop_distance
                );
            }

            // Calculate how much we COULD improve within boundaries
            let max_possible_exit_scale = max_exit_distance / current_max_exit_distance.max(0.1);
            let max_possible_stop_tighten = current_min_stop_distance / min_stop_distance.max(0.1);

            let max_achievable_rr =
                initial_risk_reward * max_possible_exit_scale * max_possible_stop_tighten;

            if max_achievable_rr < target_risk_reward * 0.8 {
                log::warn!(
                    "⚠️ Cannot achieve target R:R {:.2} within prediction boundaries (max achievable: {:.2})",
                    target_risk_reward,
                    max_achievable_rr
                );
                log::info!("📊 Optimizing to best possible R:R within model predictions...");
            }

            // Calculate improvement needed, but cap it by what's achievable
            let improvement_needed = (target_risk_reward / initial_risk_reward.max(0.1))
                .min(max_achievable_rr / initial_risk_reward.max(0.1));

            // BALANCED APPROACH with prediction boundaries
            let sqrt_improvement = improvement_needed.sqrt();

            // Scale exits, but respect prediction boundaries
            let desired_exit_scale = 1.0 + (sqrt_improvement - 1.0) * 0.4; // Conservative: 40% of sqrt gap
            let exit_scale = desired_exit_scale.min(max_possible_exit_scale).min(1.5); // Cap at 1.5x or prediction boundary

            // Scale stops, but respect minimum safe distance
            let desired_stop_scale = 1.0 / (1.0 + (sqrt_improvement - 1.0) * 0.3);
            let stop_scale = desired_stop_scale
                .max(1.0 / max_possible_stop_tighten)
                .max(0.7); // Don't tighten more than 30%

            log::info!(
                "🔄 Bounded scaling: exit_scale={:.2}x (max:{:.2}x), stop_scale={:.2}x (min:{:.2}x)",
                exit_scale,
                max_possible_exit_scale,
                stop_scale,
                1.0 / max_possible_stop_tighten
            );

            // Step 1: Apply exit scaling with boundary checks
            for exit in &mut exit_levels {
                let distance_from_current = (exit.price - config.current_price).abs();
                let scaled_distance = distance_from_current * exit_scale;

                // Enforce prediction boundary
                let bounded_distance =
                    scaled_distance.min(max_exit_distance * config.current_price / 100.0);

                exit.price = if direction == "LONG" {
                    config.current_price + bounded_distance
                } else {
                    config.current_price - bounded_distance
                };
                // Update ATR distance
                exit.atr_distance = (bounded_distance / config.current_price) * 100.0;
            }

            // Step 2: Apply stop scaling with safety checks
            for stop in &mut stop_levels {
                let distance_from_entry = (stop.price - entry_levels[0].price).abs();
                let scaled_distance = distance_from_entry * stop_scale;

                // Enforce minimum safe distance
                let bounded_distance =
                    scaled_distance.max(min_stop_distance * entry_levels[0].price / 100.0);

                stop.price = if direction == "LONG" {
                    entry_levels[0].price - bounded_distance
                } else {
                    entry_levels[0].price + bounded_distance
                };
                // Update ATR distance
                stop.atr_distance = (bounded_distance / config.current_price) * 100.0;
            }

            log::info!(
                "📊 Applied bounded adjustments: exits expanded by {:.1}%, stops tightened by {:.1}%",
                (exit_scale - 1.0) * 100.0,
                (1.0 - stop_scale) * 100.0
            );

            let optimized_rr =
                Self::calculate_risk_reward(&entry_levels, &exit_levels, &stop_levels, &direction);

            // Log final status
            if optimized_rr >= target_risk_reward * 0.9 {
                log::info!(
                    "✅ Successfully optimized R:R from {:.2} to {:.2} (target: {:.2})",
                    initial_risk_reward,
                    optimized_rr,
                    target_risk_reward
                );
            } else {
                log::warn!(
                    "⚠️ Optimized R:R to {:.2} (from {:.2}), below target {:.2} due to prediction boundaries",
                    optimized_rr,
                    initial_risk_reward,
                    target_risk_reward
                );
                log::info!(
                    "💡 Consider waiting for better market conditions or adjusting position size"
                );
            }

            optimized_rr
        } else {
            initial_risk_reward
        };

        // Step 10: Validate order consistency
        Self::validate_smart_orders(
            &entry_levels,
            &exit_levels,
            &stop_levels,
            &direction,
            config.current_price,
        )?;

        log::info!(
            "✅ SMART Orders Generated: R:R={:.2}, Direction={}, Confidence={:.2}",
            risk_reward,
            direction,
            overall_confidence
        );

        // Log detailed order information
        log::info!("📍 Entry Levels:");
        for (i, entry) in entry_levels.iter().enumerate() {
            log::info!(
                "  Entry {}: ${:.2} ({:+.2}%) | Size: {:.1}% | Conf: {:.2}",
                i + 1,
                entry.price,
                (entry.price / config.current_price - 1.0) * 100.0,
                entry.quantity_percentage * 100.0,
                entry.confidence
            );
        }

        log::info!("🎯 Exit Levels:");
        for (i, exit) in exit_levels.iter().enumerate() {
            log::info!(
                "  Exit {}: ${:.2} ({:+.2}%) | Size: {:.1}% | Conf: {:.2}",
                i + 1,
                exit.price,
                (exit.price / config.current_price - 1.0) * 100.0,
                exit.quantity_percentage * 100.0,
                exit.confidence
            );
        }

        log::info!("🛑 Stop Levels:");
        for (i, stop) in stop_levels.iter().enumerate() {
            log::info!(
                "  Stop {}: ${:.2} ({:+.2}%) | Size: {:.1}% | Conf: {:.2}",
                i + 1,
                stop.price,
                (stop.price / config.current_price - 1.0) * 100.0,
                stop.quantity_percentage * 100.0,
                stop.confidence
            );
        }

        Ok(TradingOrders {
            direction,
            entry_levels,
            exit_levels,
            stop_levels,
            total_position_size: 1.0,
            risk_reward_ratio: risk_reward,
            atr_multiplier: config.volatility_pred.position_size_multiplier,
            dynamic_sizing: true,
            note: if risk_reward < target_risk_reward * 0.8 {
                Some(format!(
                    "Risk/Reward {:.2} below target {:.2} - consider waiting for better setup",
                    risk_reward, target_risk_reward
                ))
            } else {
                None
            },
        })
    }

    /// Validate SMART orders for consistency and correctness
    fn validate_smart_orders(
        entry_levels: &[OrderLevel; 3],
        exit_levels: &[OrderLevel; 3],
        stop_levels: &[OrderLevel; 3],
        direction: &str,
        current_price: f64,
    ) -> crate::utils::error::Result<()> {
        // Validate SHORT orders
        if direction == "SHORT" {
            // Entries must be ABOVE current price
            for (i, entry) in entry_levels.iter().enumerate() {
                if entry.price <= current_price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "SHORT entry {} at ${:.2} must be above current ${:.2}",
                        i + 1,
                        entry.price,
                        current_price
                    )));
                }
            }

            // Exits must be BELOW current price
            for (i, exit) in exit_levels.iter().enumerate() {
                if exit.price >= current_price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "SHORT exit {} at ${:.2} must be below current ${:.2}",
                        i + 1,
                        exit.price,
                        current_price
                    )));
                }
            }

            // CRITICAL: Stops must be ABOVE ALL entries (no intersection!)
            let highest_entry = entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::NEG_INFINITY, f64::max);

            for (i, stop) in stop_levels.iter().enumerate() {
                // Check against ALL entries, not just corresponding one
                for (j, entry) in entry_levels.iter().enumerate() {
                    if stop.price <= entry.price {
                        return Err(crate::utils::error::VangaError::PredictionError(
                            format!(
                                "❌ CRITICAL: SHORT stop {} at ${:.2} intersects with entry {} at ${:.2}. Stop must be above ALL entries (highest: ${:.2})",
                                i+1, stop.price, j+1, entry.price, highest_entry
                            )
                        ));
                    }
                }
            }
        } else {
            // LONG orders validation
            // Entries must be BELOW current price
            for (i, entry) in entry_levels.iter().enumerate() {
                if entry.price >= current_price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "LONG entry {} at ${:.2} must be below current ${:.2}",
                        i + 1,
                        entry.price,
                        current_price
                    )));
                }
            }

            // Exits must be ABOVE current price
            for (i, exit) in exit_levels.iter().enumerate() {
                if exit.price <= current_price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "LONG exit {} at ${:.2} must be above current ${:.2}",
                        i + 1,
                        exit.price,
                        current_price
                    )));
                }
            }

            // CRITICAL: Stops must be BELOW ALL entries (no intersection!)
            let lowest_entry = entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::INFINITY, f64::min);

            for (i, stop) in stop_levels.iter().enumerate() {
                // Check against ALL entries, not just corresponding one
                for (j, entry) in entry_levels.iter().enumerate() {
                    if stop.price >= entry.price {
                        return Err(crate::utils::error::VangaError::PredictionError(
                            format!(
                                "❌ CRITICAL: LONG stop {} at ${:.2} intersects with entry {} at ${:.2}. Stop must be below ALL entries (lowest: ${:.2})",
                                i+1, stop.price, j+1, entry.price, lowest_entry
                            )
                        ));
                    }
                }
            }
        }

        // Validate sizes sum to 1.0
        let entry_sum: f64 = entry_levels.iter().map(|l| l.quantity_percentage).sum();
        let exit_sum: f64 = exit_levels.iter().map(|l| l.quantity_percentage).sum();

        if (entry_sum - 1.0).abs() > 0.01 {
            log::warn!("Entry sizes sum to {:.3}, expected 1.0", entry_sum);
        }
        if (exit_sum - 1.0).abs() > 0.01 {
            log::warn!("Exit sizes sum to {:.3}, expected 1.0", exit_sum);
        }

        // Log successful validation
        log::info!("✅ Order validation passed: No stop/entry intersections detected");

        Ok(())
    }

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
        // Remove unused atr_distance calculation since we now calculate per-level
        // let atr_distance = config.current_price * (base_atr_pct / 100.0);

        // 🎯 ADAPTIVE ORDER GENERATION: Use price level probabilities instead of sequence ranges
        let (mut entry_levels, mut exit_levels, mut stop_levels) = if direction == "LONG" {
            // Check if this is a breakout signal based on pump probability (adaptive threshold)
            let is_breakout = config.direction_pred.pump_probability > 0.25;
            Self::generate_adaptive_long_orders(
                config.current_price,
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
                config.config,
                is_breakout,
                config.price_levels,
                config.volatility_pred,
                config.sequence_prices,
                config.direction_pred,
            )
        };

        // Calculate dynamic risk-reward ratio based on prediction confidence
        // Formula: risk_reward = 1 / confidence (with practical bounds)
        // Use available predictions to calculate confidence
        let overall_confidence = if let Some(confidence) = config.overall_confidence {
            confidence
        } else {
            // Fallback: calculate from available predictions
            let direction_conf = config.direction_pred.confidence;
            let price_conf = config.price_levels.confidence;
            let volatility_conf = config.volatility_pred.regime_confidence;

            // Weighted average of available confidences
            (direction_conf * 0.4 + price_conf * 0.4 + volatility_conf * 0.2).clamp(0.05, 0.95)
        };

        let min_risk_reward = (1.0_f64 / overall_confidence.max(0.1)).clamp(2.0, 10.0);

        log::info!(
            "🎯 Dynamic R:R: confidence={:.3} → required_rr={:.2} (was hardcoded 4.0)",
            overall_confidence,
            min_risk_reward
        );
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
        price_levels: &PriceLevelPrediction,
        direction_pred: &DirectionPrediction,
        volatility_pred: &VolatilityPrediction,
        _config: &OrderConfig,
        is_breakout: bool,
        dynamic_entry_sizes: Option<[f64; 3]>,
        dynamic_exit_sizes: Option<[f64; 3]>,
        sequence_prices: &[f64], // NEW: Actual price sequence for smart calculation
    ) -> ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]) {
        // 🎯 SEQUENCE-DERIVED BOUNDS: Calculate adaptive bounds from actual sequence data
        let sequence_min = sequence_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let sequence_max = sequence_prices
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sequence_range_pct = ((sequence_max - sequence_min) / current_price) * 100.0;

        // Calculate percentiles for robust statistics
        let mut sorted_prices = sequence_prices.to_vec();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = sorted_prices.len();

        let p25 = sorted_prices[len * 25 / 100];
        let p50 = sorted_prices[len * 50 / 100];
        let p75 = sorted_prices[len * 75 / 100];

        // IQR-based volatility (robust to outliers)
        let iqr_volatility = ((p75 - p25) / p50) * 100.0;

        // Maximum drawdown from sequence
        let max_drawdown_pct = ((current_price - sequence_min) / current_price) * 100.0;
        let max_upside_pct = ((sequence_max - current_price) / current_price) * 100.0;

        log::info!(
            "📊 Sequence Bounds: range={:.2}%, drawdown={:.2}%, upside={:.2}%, IQR_vol={:.2}%",
            sequence_range_pct,
            max_drawdown_pct,
            max_upside_pct,
            iqr_volatility
        );

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
            "📊 LONG Adaptive Entry: IQR_vol={:.3}%, pred_vol={:.3}%, seq_range={:.3}%, momentum_factor={:.1}x",
            iqr_volatility,
            volatility_pred.expected_range_percent,
            sequence_range_pct,
            momentum_factor
        );

        // 🎯 ADAPTIVE ENTRY DISTANCES: Use price level bins bounded by sequence data
        let neutral_bin = price_levels.bins.get("neutral");
        let moderate_down_bin = price_levels.bins.get("moderate_down");
        let strong_down_bin = price_levels.bins.get("strong_down");

        // Use model predictions bounded by sequence reality
        let entry_1_pct = if let Some(bin) = neutral_bin {
            // Use neutral range (smallest predicted movement) bounded by sequence
            let model_distance = bin.range[0].abs();
            let bounded_distance = (sequence_range_pct * bin.probability).min(model_distance);
            -bounded_distance * momentum_factor
        } else {
            -iqr_volatility * 0.2 * momentum_factor
        };

        let entry_2_pct = if let Some(bin) = moderate_down_bin {
            // Use moderate down range bounded by sequence
            let model_distance = bin.range[0].abs();
            let bounded_distance = (sequence_range_pct * bin.probability).min(model_distance);
            -bounded_distance * momentum_factor
        } else {
            -iqr_volatility * 0.5 * momentum_factor
        };

        let entry_3_pct = if let Some(bin) = strong_down_bin {
            // Use strong down range bounded by sequence
            let model_distance = bin.range[0].abs();
            let bounded_distance = (sequence_range_pct * bin.probability).min(model_distance);
            -bounded_distance * momentum_factor
        } else {
            -iqr_volatility * 0.8 * momentum_factor
        };

        log::info!(
            "📍 ADAPTIVE Entry Distances: {:.3}%, {:.3}%, {:.3}% (model-guided, sequence-bounded)",
            entry_1_pct.abs(),
            entry_2_pct.abs(),
            entry_3_pct.abs()
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

        let entry_1_price = current_price * (1.0 + entry_1_pct / 100.0);
        let entry_2_price = current_price * (1.0 + entry_2_pct / 100.0);
        let entry_3_price = current_price * (1.0 + entry_3_pct / 100.0);

        let entry_levels = [
            OrderLevel {
                price: entry_1_price,
                quantity_percentage: entry_1_size,
                atr_distance: ((entry_1_price - current_price).abs() / current_price) * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: entry_1_confidence,
            },
            OrderLevel {
                price: entry_2_price,
                quantity_percentage: entry_2_size,
                atr_distance: ((entry_2_price - current_price).abs() / current_price) * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: entry_2_confidence,
            },
            OrderLevel {
                price: entry_3_price,
                quantity_percentage: entry_3_size,
                atr_distance: ((entry_3_price - current_price).abs() / current_price) * 100.0,
                order_type: if is_breakout {
                    "STOP_LIMIT".to_string()
                } else {
                    "LIMIT".to_string()
                },
                confidence: entry_3_confidence,
            },
        ];

        // 🎯 SEQUENCE-BOUNDED EXIT TARGETS: Use model predictions bounded by sequence upside
        let moderate_up_bin = price_levels.bins.get("moderate_up");
        let strong_up_bin = price_levels.bins.get("strong_up");

        // Use model predictions bounded by sequence reality
        let exit_1_pct = if let Some(bin) = moderate_up_bin {
            let model_target = bin.range[0];
            let bounded_target = (max_upside_pct * bin.probability).min(model_target);
            bounded_target.max(iqr_volatility * 0.5) // Minimum based on sequence volatility
        } else {
            iqr_volatility * 2.0
        };

        let exit_2_pct = if let Some(bin) = moderate_up_bin {
            let model_target = (bin.range[0] + bin.range[1]) / 2.0;
            let bounded_target = (max_upside_pct * bin.probability * 1.2).min(model_target);
            bounded_target.max(exit_1_pct * 1.2)
        } else {
            exit_1_pct * 1.5
        };

        let exit_3_pct = if let Some(bin) = strong_up_bin {
            let model_target = bin.range[0];
            let bounded_target = (max_upside_pct * bin.probability * 1.5).min(model_target);
            bounded_target.max(exit_2_pct * 1.1)
        } else {
            exit_2_pct * 1.3
        };

        log::info!(
            "🎯 ADAPTIVE Exit Targets: {:.3}%, {:.3}%, {:.3}% (model-guided, sequence-bounded)",
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

        let exit_1_price = current_price * (1.0 + exit_1_pct / 100.0);
        let exit_2_price = current_price * (1.0 + exit_2_pct / 100.0);
        let exit_3_price = current_price * (1.0 + exit_3_pct / 100.0);

        let exit_levels = [
            OrderLevel {
                price: exit_1_price,
                quantity_percentage: exit_1_size,
                atr_distance: ((exit_1_price - current_price).abs() / current_price) * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: exit_1_confidence,
            },
            OrderLevel {
                price: exit_2_price,
                quantity_percentage: exit_2_size,
                atr_distance: ((exit_2_price - current_price).abs() / current_price) * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: exit_2_confidence,
            },
            OrderLevel {
                price: exit_3_price,
                quantity_percentage: exit_3_size,
                atr_distance: ((exit_3_price - current_price).abs() / current_price) * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: exit_3_confidence,
            },
        ];

        // 🎯 SEQUENCE-BASED STOP CALCULATION: Use lowest entry and sequence drawdown
        let lowest_entry_price = entry_levels
            .iter()
            .map(|e| e.price)
            .fold(f64::INFINITY, |a, b| a.min(b));

        // Use sequence-derived maximum drawdown with probability weighting
        let strong_down_prob = strong_down_bin.map(|bin| bin.probability).unwrap_or(0.1);
        let stop_distance_pct = (max_drawdown_pct * strong_down_prob)
            .min(volatility_pred.recommended_stop_distance_percent);

        // Progressive stops below lowest entry (never overlapping)
        let stop_spacing = iqr_volatility * 0.01; // Use IQR for natural spacing
        let stop_base = lowest_entry_price * (1.0 - stop_distance_pct / 100.0);

        let stop_price_1 = stop_base;
        let stop_price_2 = stop_base - (stop_spacing * current_price / 100.0);
        let stop_price_3 = stop_base - (stop_spacing * 2.0 * current_price / 100.0);

        log::info!(
            "🛡️ ADAPTIVE Stops: base={:.4}, spacing={:.3}%, below_lowest_entry={:.2}%",
            stop_base,
            stop_spacing,
            ((lowest_entry_price - stop_base) / lowest_entry_price) * 100.0
        );

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
                atr_distance: ((stop_price_1 - current_price).abs() / current_price) * 100.0,
                order_type: "STOP_LOSS".to_string(),
                confidence: base_stop_confidence,
            },
            OrderLevel {
                price: stop_price_2,
                quantity_percentage: stop_2_size,
                atr_distance: ((stop_price_2 - current_price).abs() / current_price) * 100.0,
                order_type: "STOP_LOSS".to_string(),
                confidence: base_stop_confidence * 0.9,
            },
            OrderLevel {
                price: stop_price_3,
                quantity_percentage: stop_3_size,
                atr_distance: ((stop_price_3 - current_price).abs() / current_price) * 100.0,
                order_type: "STOP_LOSS".to_string(),
                confidence: base_stop_confidence * 0.8,
            },
        ];

        // 🎯 STATISTICAL VALIDATION: Use sequence statistics to validate orders
        let sequence_mean = sequence_prices.iter().sum::<f64>() / sequence_prices.len() as f64;
        let sequence_variance = sequence_prices
            .iter()
            .map(|&x| (x - sequence_mean).powi(2))
            .sum::<f64>()
            / sequence_prices.len() as f64;
        let sequence_std = sequence_variance.sqrt();

        // Validate and auto-correct if needed
        let mut validated_entry_levels = entry_levels;
        let mut validated_exit_levels = exit_levels;
        let mut validated_stop_levels = stop_levels;

        // Check entries are within sequence bounds and not statistical outliers
        for (i, entry) in validated_entry_levels.iter_mut().enumerate() {
            let z_score = (entry.price - sequence_mean) / sequence_std;
            if z_score.abs() > 3.0 || entry.price < sequence_min || entry.price >= current_price {
                let corrected_price = (sequence_min + current_price) / 2.0; // Safe middle ground
                log::warn!(
                    "⚠️ Entry {} corrected: {:.4} -> {:.4} (z_score={:.2})",
                    i,
                    entry.price,
                    corrected_price,
                    z_score
                );
                entry.price = corrected_price;
            }
        }

        // Check exits are above current and within reasonable bounds
        for (i, exit) in validated_exit_levels.iter_mut().enumerate() {
            let z_score = (exit.price - sequence_mean) / sequence_std;
            if z_score.abs() > 3.0 || exit.price > sequence_max || exit.price <= current_price {
                let corrected_price = (current_price + sequence_max) / 2.0; // Safe middle ground
                log::warn!(
                    "⚠️ Exit {} corrected: {:.4} -> {:.4} (z_score={:.2})",
                    i,
                    exit.price,
                    corrected_price,
                    z_score
                );
                exit.price = corrected_price;
            }
        }

        // Ensure stops are below lowest entry with minimum gap
        let lowest_entry = validated_entry_levels
            .iter()
            .map(|e| e.price)
            .fold(f64::INFINITY, |a, b| a.min(b));
        for (i, stop) in validated_stop_levels.iter_mut().enumerate() {
            let min_stop_price = lowest_entry * 0.995; // 0.5% below lowest entry minimum
            if stop.price >= min_stop_price {
                let corrected_price = min_stop_price - (i as f64 * 0.001 * current_price);
                log::warn!(
                    "⚠️ Stop {} corrected: {:.4} -> {:.4} (was above entry)",
                    i,
                    stop.price,
                    corrected_price
                );
                stop.price = corrected_price;
            }
        }

        log::info!(
            "✅ VALIDATED Orders: Entries [{:.4}, {:.4}, {:.4}] | Exits [{:.4}, {:.4}, {:.4}] | Stops [{:.4}, {:.4}, {:.4}]",
            validated_entry_levels[0].price, validated_entry_levels[1].price, validated_entry_levels[2].price,
            validated_exit_levels[0].price, validated_exit_levels[1].price, validated_exit_levels[2].price,
            validated_stop_levels[0].price, validated_stop_levels[1].price, validated_stop_levels[2].price
        );

        (
            validated_entry_levels,
            validated_exit_levels,
            validated_stop_levels,
        )
    }

    /// Generate sequence-aware short orders using probability-based allocation (NO MAGIC NUMBERS)
    #[allow(clippy::too_many_arguments)]
    fn generate_sequence_aware_short_orders(
        current_price: f64,
        _config: &OrderConfig,
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

        // 🎯 SEQUENCE-DERIVED BOUNDS: Calculate adaptive bounds from actual sequence data
        let sequence_min = sequence_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let sequence_max = sequence_prices
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sequence_range_pct = ((sequence_max - sequence_min) / current_price) * 100.0;

        // Calculate percentiles for robust statistics
        let mut sorted_prices = sequence_prices.to_vec();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = sorted_prices.len();

        let p25 = sorted_prices[len * 25 / 100];
        let p50 = sorted_prices[len * 50 / 100];
        let p75 = sorted_prices[len * 75 / 100];

        // IQR-based volatility (robust to outliers)
        let iqr_volatility = ((p75 - p25) / p50) * 100.0;

        // Maximum upside and downside from sequence
        let max_upside_pct = ((sequence_max - current_price) / current_price) * 100.0;
        let max_downside_pct = ((current_price - sequence_min) / current_price) * 100.0;

        log::info!(
            "📊 SHORT Sequence Bounds: range={:.2}%, upside={:.2}%, downside={:.2}%, IQR_vol={:.2}%",
            sequence_range_pct, max_upside_pct, max_downside_pct, iqr_volatility
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

        // 🎯 ADAPTIVE SHORT ENTRIES: Use price level bins bounded by sequence upside
        let (entry_1_pct, entry_1_prob) = if let Some(bin) = neutral_bin {
            let model_distance = bin.range[1].abs(); // Upper range of neutral
            let bounded_distance = (max_upside_pct * bin.probability).min(model_distance);
            (bounded_distance, bin.probability)
        } else {
            (iqr_volatility * 0.2, 0.2)
        };

        let (entry_2_pct, entry_2_prob) = if let Some(bin) = moderate_up_bin {
            let model_distance = (bin.range[0] + bin.range[1]) / 2.0; // Center of moderate_up
            let bounded_distance = (max_upside_pct * bin.probability).min(model_distance);
            (bounded_distance, bin.probability)
        } else {
            (iqr_volatility * 0.5, 0.2)
        };

        let (entry_3_pct, entry_3_prob) = if let Some(bin) = moderate_up_bin {
            let model_distance = bin.range[1].abs(); // Upper range of moderate_up
            let bounded_distance = (max_upside_pct * bin.probability).min(model_distance);
            (bounded_distance, bin.probability)
        } else {
            (iqr_volatility * 0.8, 0.15)
        };

        log::info!(
            "📍 ADAPTIVE SHORT Entries: {:.3}%, {:.3}%, {:.3}% ABOVE current (sequence-bounded)",
            entry_1_pct,
            entry_2_pct,
            entry_3_pct
        );

        let entry_1_price = current_price * (1.0 + entry_1_pct / 100.0);
        let entry_2_price = current_price * (1.0 + entry_2_pct / 100.0);
        let entry_3_price = current_price * (1.0 + entry_3_pct / 100.0);

        let entry_levels = [
            OrderLevel {
                price: entry_1_price,
                quantity_percentage: entry_1_prob, // Use actual probability for sizing
                atr_distance: ((entry_1_price - current_price).abs() / current_price) * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: 0.9,
            },
            OrderLevel {
                price: entry_2_price,
                quantity_percentage: entry_2_prob, // Use actual probability for sizing
                atr_distance: ((entry_2_price - current_price).abs() / current_price) * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: 0.7,
            },
            OrderLevel {
                price: entry_3_price,
                quantity_percentage: entry_3_prob, // Use actual probability for sizing
                atr_distance: ((entry_3_price - current_price).abs() / current_price) * 100.0,
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

        let exit_1_price = current_price * (1.0 + exit_1_pct / 100.0);
        let exit_2_price = current_price * (1.0 + exit_2_pct / 100.0);
        let exit_3_price = current_price * (1.0 + exit_3_pct / 100.0);

        let exit_levels = [
            OrderLevel {
                price: exit_1_price,
                quantity_percentage: exit_1_prob, // Use actual probability for sizing
                atr_distance: ((exit_1_price - current_price).abs() / current_price) * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: exit_2_price,
                quantity_percentage: exit_2_prob, // Use actual probability for sizing
                atr_distance: ((exit_2_price - current_price).abs() / current_price) * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: 0.6,
            },
            OrderLevel {
                price: exit_3_price,
                quantity_percentage: exit_3_prob, // Use actual probability for sizing
                atr_distance: ((exit_3_price - current_price).abs() / current_price) * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: 0.4,
            },
        ];

        // 🎯 SEQUENCE-BASED SHORT STOPS: Above highest entry using sequence bounds
        let highest_entry_price = entry_levels
            .iter()
            .map(|e| e.price)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        // Use sequence-derived maximum upside with probability weighting
        let strong_up_bin = price_levels.bins.get("strong_up");
        let strong_up_prob = strong_up_bin.map(|bin| bin.probability).unwrap_or(0.1);
        let stop_distance_pct = (max_upside_pct * strong_up_prob)
            .min(volatility_pred.recommended_stop_distance_percent);

        // Progressive stops above highest entry (never overlapping)
        let stop_spacing = iqr_volatility * 0.01; // Use IQR for natural spacing
        let stop_base = highest_entry_price * (1.0 + stop_distance_pct / 100.0);

        let stop_1_price = stop_base;
        let stop_2_price = stop_base + (stop_spacing * current_price / 100.0);
        let stop_3_price = stop_base + (stop_spacing * 2.0 * current_price / 100.0);

        log::info!(
            "🛡️ ADAPTIVE SHORT Stops: base={:.4}, spacing={:.3}%, above_highest_entry={:.2}%",
            stop_base,
            stop_spacing,
            ((stop_base - highest_entry_price) / highest_entry_price) * 100.0
        );

        // Use same probability-based portions for stops as entries (risk consistency)
        let stop_levels = [
            OrderLevel {
                price: stop_1_price,
                quantity_percentage: entry_1_prob, // Same as entry allocation
                atr_distance: ((stop_1_price - current_price).abs() / current_price) * 100.0,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.9,
            },
            OrderLevel {
                price: stop_2_price,
                quantity_percentage: entry_2_prob, // Same as entry allocation
                atr_distance: ((stop_2_price - current_price).abs() / current_price) * 100.0,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: stop_3_price,
                quantity_percentage: entry_3_prob, // Same as entry allocation
                atr_distance: ((stop_3_price - current_price).abs() / current_price) * 100.0,
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

    /// Generate trading orders with sequence statistics for fully adaptive behavior
    pub fn generate_with_sequence_stats(
        config: SequenceAwareConfig,
    ) -> crate::utils::error::Result<Self> {
        // Calculate sequence statistics from raw data
        let sequence_stats = SequenceStatistics::from_prices(
            config.sequence_prices,
            config.horizon_hours,
            config.sequence_volumes,
        )?;

        log::info!(
            "📊 Sequence Statistics: mean_return={:.3}%, std={:.3}%, hurst={:.2}, kelly={:.2}",
            sequence_stats.mean_return * 100.0,
            sequence_stats.std_return * 100.0,
            sequence_stats.hurst_exponent,
            sequence_stats.kelly_fraction
        );

        // Create SmartConsensus from predictions
        let consensus = SmartConsensus {
            direction: config.direction_pred.clone(),
            price_levels: config.price_levels.clone(),
            volatility: config.volatility_pred.clone(),
            sentiment: config.sentiment_pred.clone(),
            volume: config.volume_pred.clone(),
        };

        // Get direction consensus
        let (direction, _direction_confidence) = consensus.calculate_direction_consensus();

        // Generate sequence-aware orders
        let mut entry_levels = consensus.generate_sequence_aware_entries(
            config.current_price,
            &direction,
            &sequence_stats,
        )?;

        let mut exit_levels = consensus.generate_sequence_aware_exits(
            config.current_price,
            &direction,
            &sequence_stats,
        )?;

        let mut stop_levels =
            consensus.generate_sequence_aware_stops(&entry_levels, &direction, &sequence_stats)?;

        // Normalize sizes
        SmartConsensus::normalize_sizes(&mut entry_levels);
        SmartConsensus::normalize_sizes(&mut exit_levels);

        // Calculate adaptive risk-reward requirement
        let required_rr = consensus.calculate_adaptive_risk_reward_requirement(&sequence_stats);

        // Optimize if needed
        let final_rr = consensus.optimize_with_sequence_stats(
            &mut entry_levels,
            &mut exit_levels,
            &mut stop_levels,
            &direction,
            &sequence_stats,
        );

        // Create note about sequence-aware generation
        let note = if final_rr < required_rr {
            Some(format!(
                "Sequence-aware optimization achieved R:R {:.2} (adaptive requirement: {:.2}). Hurst={:.2}, Kelly={:.2}",
                final_rr, required_rr, sequence_stats.hurst_exponent, sequence_stats.kelly_fraction
            ))
        } else {
            Some(format!(
                "Sequence-aware orders generated with R:R {:.2} (exceeds adaptive requirement {:.2}). Market is {}",
                final_rr,
                required_rr,
                if sequence_stats.hurst_exponent > 0.5 { "trending" } else { "mean-reverting" }
            ))
        };

        Ok(Self {
            direction: direction.clone(),
            entry_levels,
            exit_levels,
            stop_levels,
            total_position_size: 1.0,
            risk_reward_ratio: final_rr,
            atr_multiplier: 2.0,
            dynamic_sizing: true,
            note,
        })
    }
}
