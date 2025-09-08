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
            min_risk_reward: 4.0,     // Crypto minimum R:R target
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
    pub fn generate(config: SmartOrderConfig) -> crate::utils::error::Result<Self> {
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

        // Step 3: Generate SEQUENCE-AWARE entry levels when sequence data available
        let mut entry_levels = if let Some(ohlcv_data) = config.sequence_ohlcv {
            // Extract sequence prices and calculate statistics
            let sequence_prices: Vec<f64> = ohlcv_data.iter().map(|row| row.close).collect();

            // Calculate sequence statistics for adaptive generation
            let sequence_stats = SequenceStatistics::from_prices(
                &sequence_prices,
                4.0, // Default horizon hours, adjust as needed
                None,
            )?;

            log::info!(
                "📊 Using SEQUENCE-AWARE entry generation with statistics: volatility={:.3}%, hurst={:.3}, kelly={:.3}",
                sequence_stats.std_return * 100.0,
                sequence_stats.hurst_exponent,
                sequence_stats.kelly_fraction
            );

            // Use the sequence-aware method that works properly
            let entries = consensus.generate_sequence_aware_entries(
                config.current_price,
                &direction,
                &sequence_stats,
            )?;

            // Convert MarketDataRow to OHLCV array format for psychological adjustments
            let ohlcv_arrays: Vec<[f64; 5]> = ohlcv_data
                .iter()
                .map(|row| [row.open, row.high, row.low, row.close, row.volume])
                .collect();

            // Apply psychological level adjustments
            let mut adjusted_entries = entries;
            for entry in &mut adjusted_entries {
                entry.price = consensus.adjust_for_psychological_levels(
                    entry.price,
                    Some(&ohlcv_arrays),
                    false, // is_stop_loss = false for entries
                    None,  // direction not needed for entries
                );
            }

            log::info!("🎯 Generated SEQUENCE-AWARE entries with psychological adjustments");
            adjusted_entries
        } else {
            // Fallback to standard generation when no sequence data available
            log::warn!("⚠️ No sequence data available, using standard entry generation");
            consensus.generate_smart_entries(config.current_price, &direction)?
        };

        // Step 4: Generate SEQUENCE-AWARE exit levels when sequence data available
        let mut exit_levels = if let Some(ohlcv_data) = config.sequence_ohlcv {
            // Reuse sequence prices if already calculated, or extract them
            let sequence_prices: Vec<f64> = ohlcv_data.iter().map(|row| row.close).collect();

            // Calculate sequence statistics if not already done
            let sequence_stats = SequenceStatistics::from_prices(
                &sequence_prices,
                4.0, // Default horizon hours
                None,
            )?;

            log::info!("📊 Using SEQUENCE-AWARE exit generation with MFE distribution");

            // Use sequence-aware exit generation with MFE distribution
            let exits = consensus.generate_sequence_aware_exits(
                config.current_price,
                &direction,
                &sequence_stats,
            )?;

            // Convert for psychological adjustments
            let ohlcv_arrays: Vec<[f64; 5]> = ohlcv_data
                .iter()
                .map(|row| [row.open, row.high, row.low, row.close, row.volume])
                .collect();

            // Apply psychological level adjustments
            let mut adjusted_exits = exits;
            for exit in &mut adjusted_exits {
                exit.price = consensus.adjust_for_psychological_levels(
                    exit.price,
                    Some(&ohlcv_arrays),
                    false, // is_stop_loss = false for targets
                    None,  // direction not needed for targets
                );
            }

            log::info!("🎯 Generated SEQUENCE-AWARE exits with psychological adjustments");
            adjusted_exits
        } else {
            // Fallback to standard generation
            log::warn!("⚠️ No sequence data available, using standard exit generation");
            consensus.generate_smart_exits(config.current_price, &direction)?
        };

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
                // Use enhanced existing method with stop-specific parameters
                stop.price = consensus.adjust_for_psychological_levels(
                    stop.price,
                    Some(&ohlcv_arrays),
                    true,             // is_stop_loss = true
                    Some(&direction), // Pass direction for stop-specific logic
                );
            }

            log::info!("🛡️ Applied stop hunt protection via psychological level adjustments");
        }

        // Step 6: Normalize sizes to ensure they sum to 1.0
        SmartConsensus::normalize_sizes(&mut entry_levels);
        SmartConsensus::normalize_sizes(&mut exit_levels);
        SmartConsensus::normalize_sizes(&mut stop_levels); // FIX: Normalize stops too!

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

        // Step 9: Use CONFIGURED minimum R:R as base target, enhanced by model confidence
        const MIN_RR_TARGET: f64 = 4.0; // CONSTANT minimum R:R target
        let order_config = OrderConfig::default();
        let configured_min_rr = if order_config.min_risk_reward >= 2.0 {
            order_config.min_risk_reward
        } else {
            MIN_RR_TARGET // Use constant if config is not properly set
        };

        // Model's natural R:R for reference (but not as target)
        let natural_rr = config.direction_pred.expected_upside_percent
            / config.direction_pred.expected_downside_percent.max(0.01);
        let prediction_rr = config.direction_pred.risk_reward_ratio;

        // Target R:R uses configured minimum, can be enhanced if model shows strong opportunity
        // But NEVER go below the configured minimum
        let target_risk_reward = if prediction_rr > configured_min_rr {
            // Model sees exceptional opportunity, use it
            prediction_rr.min(order_config.max_risk_reward)
        } else {
            // Use configured minimum as floor
            configured_min_rr
        };

        log::info!(
            "📊 R:R Assessment: configured_min={:.2}, model_natural={:.2} (up:{:.2}%/down:{:.2}%), model_rr={:.2}, TARGET={:.2}",
            configured_min_rr,
            natural_rr,
            config.direction_pred.expected_upside_percent,
            config.direction_pred.expected_downside_percent,
            prediction_rr,
            target_risk_reward
        );

        let risk_reward = if initial_risk_reward < target_risk_reward {
            log::info!(
                "⚠️ Initial R:R {:.2} below target {:.2}, optimizing with PROPORTIONAL scaling...",
                initial_risk_reward,
                target_risk_reward
            );

            // Calculate average stop distance for R:R calculation
            let avg_stop_distance = stop_levels
                .iter()
                .map(|s| ((s.price - entry_levels[0].price).abs() / entry_levels[0].price) * 100.0)
                .sum::<f64>()
                / stop_levels.len() as f64;

            // Define boundaries based on price level predictions
            // IMPROVED: Use bin centers as targets, with buffer for R:R achievement
            let max_exit_boundary = if direction == "LONG" {
                // For LONG, find the most optimistic reasonable target
                let strong_up_target = config
                    .price_levels
                    .bins
                    .get("strong_up")
                    .map(|b| {
                        // Use bin CENTER as primary target
                        let center_price = (b.price[0] + b.price[1]) / 2.0;
                        ((center_price - config.current_price) / config.current_price) * 100.0
                    })
                    .unwrap_or(3.0); // Default 3% if no bin

                // Allow extension beyond bin center for R:R, but be reasonable
                // Use max of: strong_up center, 3x stop distance, or 3%
                let min_for_rr = (avg_stop_distance * target_risk_reward).max(3.0);
                strong_up_target.max(min_for_rr)
            } else {
                // For SHORT, find the most optimistic reasonable target
                let strong_down_target = config
                    .price_levels
                    .bins
                    .get("strong_down")
                    .map(|b| {
                        // Use bin CENTER as primary target
                        let center_price = (b.price[0] + b.price[1]) / 2.0;
                        ((config.current_price - center_price) / config.current_price) * 100.0
                    })
                    .unwrap_or(3.0); // Default 3% if no bin

                // Allow extension for R:R achievement
                let min_for_rr = (avg_stop_distance * target_risk_reward).max(3.0);
                strong_down_target.max(min_for_rr)
            };

            log::info!(
                "🎯 Optimization boundaries: max_exit={:.2}% (from predictions)",
                max_exit_boundary
            );

            // Calculate current distances for all levels
            let current_exit_distances: Vec<f64> = exit_levels
                .iter()
                .map(|e| ((e.price - config.current_price).abs() / config.current_price) * 100.0)
                .collect();

            // Current middle exit distance (this will be our limiting factor)
            let current_middle_exit_distance = current_exit_distances[1];

            // Calculate the maximum scale factor based on middle exit reaching boundary

            // FIXED: Use prediction boundary as primary constraint, but respect middle exit rule
            // Middle exit should reach towards prediction boundary, not be limited by original exits
            let effective_max_middle_distance = max_exit_boundary;

            // FIXED: Middle exit should reach FULL prediction boundary for maximum R:R
            // This is the max we go as you specified
            let max_for_middle = max_exit_boundary; // Use FULL boundary, not 70%

            // Maximum scale is when middle exit reaches FULL boundary
            let max_scale_factor = max_for_middle / current_middle_exit_distance.max(0.1);

            log::info!(
                "📏 Middle exit constraint: current={:.2}%, max_for_middle={:.2}% (FULL boundary {:.2}%)",
                current_middle_exit_distance,
                max_for_middle,
                max_exit_boundary
            );

            // Calculate desired scale to achieve target R:R
            // Direct ratio, not sqrt - we want to actually reach the target!
            let desired_scale = target_risk_reward / initial_risk_reward.max(0.1);

            // CRITICAL: Apply the scale to reach target R:R, capped by boundary
            let final_scale = desired_scale.min(max_scale_factor);

            log::info!(
                "🔄 PROPORTIONAL scaling: desired={:.2}x, max_allowed={:.2}x, final={:.2}x",
                desired_scale,
                max_scale_factor,
                final_scale
            );

            // Apply PROPORTIONAL scaling to ALL exit levels
            for (i, exit) in exit_levels.iter_mut().enumerate() {
                let distance_from_current = (exit.price - config.current_price).abs();
                let scaled_distance = distance_from_current * final_scale;

                exit.price = if direction == "LONG" {
                    config.current_price + scaled_distance
                } else {
                    config.current_price - scaled_distance
                };

                // Update ATR distance
                exit.atr_distance = (scaled_distance / config.current_price) * 100.0;

                log::debug!(
                    "  Exit {}: {:.4} → {:.4} (distance: {:.2}% → {:.2}%)",
                    i + 1,
                    if direction == "LONG" {
                        config.current_price + distance_from_current
                    } else {
                        config.current_price - distance_from_current
                    },
                    exit.price,
                    current_exit_distances[i],
                    exit.atr_distance
                );
            }

            // STOPS NEVER MOVE - they stay exactly where they were calculated
            // Only exits are scaled to improve R:R ratio
            log::info!("🛑 STOPS REMAIN UNCHANGED - never moved during optimization");

            // Verify middle exit constraint
            let new_middle_exit_distance = ((exit_levels[1].price - config.current_price).abs()
                / config.current_price)
                * 100.0;

            log::info!(
                "✅ PROPORTIONAL optimization complete: exits scaled {:.1}x, stops UNCHANGED",
                final_scale
            );

            log::info!(
                "📊 Middle exit moved: {:.2}% → {:.2}% (boundary: {:.2}%)",
                current_middle_exit_distance,
                new_middle_exit_distance,
                effective_max_middle_distance
            );

            // Calculate initial optimized R:R after scaling
            let optimized_rr =
                Self::calculate_risk_reward(&entry_levels, &exit_levels, &stop_levels, &direction);

            let mut final_rr = optimized_rr;

            // ADDITIONAL OPTIMIZATION: If R:R still below target, try adjusting entries
            if optimized_rr < target_risk_reward * 0.9 {
                log::info!(
                    "🔧 R:R still below target ({:.2} < {:.2}), trying entry adjustment...",
                    optimized_rr,
                    target_risk_reward
                );

                // For R:R improvement, we need to:
                // - For LONG: Move entries LOWER (better entry prices) OR keep exits same
                // - For SHORT: Move entries HIGHER (better entry prices) OR keep exits same

                // Calculate current weighted averages for analysis
                let current_avg_entry = Self::weighted_average_price(&entry_levels);
                let current_avg_exit = Self::weighted_average_price(&exit_levels);
                let current_avg_stop = Self::weighted_average_price(&stop_levels);

                log::debug!(
                    "Current averages: entry=${:.4}, exit=${:.4}, stop=${:.4}",
                    current_avg_entry,
                    current_avg_exit,
                    current_avg_stop
                );

                // Try to improve R:R by adjusting entries (but not MARKET orders)
                let mut entries_adjusted = false;

                // VOLATILITY-BASED entry adjustment - NO MAGIC NUMBERS
                // Get volatility recommendation from the actual model (like 0.33% we see in logs)
                let volatility_stop_distance =
                    config.volatility_pred.recommended_stop_distance_percent;

                // Entry movement limited by volatility - can't move more than volatility allows
                let max_entry_movement = volatility_stop_distance; // Use actual volatility recommendation

                log::debug!(
                    "🔧 Volatility-based entry adjustment: max_movement={:.2}% (from volatility model)",
                    max_entry_movement
                );

                for (i, entry) in entry_levels.iter_mut().enumerate() {
                    if entry.order_type == "MARKET" {
                        continue; // Don't adjust market orders at current price
                    }

                    // Calculate how much we can safely move this entry (limited by volatility)
                    let current_distance = (entry.price - config.current_price).abs();

                    // Move entry by maximum volatility allows (e.g., 0.33%)
                    let movement_amount = config.current_price * (max_entry_movement / 100.0);
                    let new_distance = current_distance + movement_amount;

                    let new_entry_price = if direction == "LONG" {
                        // LONG: move entries LOWER (better entry prices)
                        config.current_price - new_distance
                    } else {
                        // SHORT: move entries HIGHER (better entry prices)
                        config.current_price + new_distance
                    };

                    // CRITICAL: Check that new entry doesn't intersect with stops
                    let safe_from_stops = if direction == "LONG" {
                        // For LONG: entry must be ABOVE all stops
                        let highest_stop = stop_levels
                            .iter()
                            .map(|s| s.price)
                            .fold(f64::NEG_INFINITY, f64::max);
                        new_entry_price > highest_stop * 1.02 // 2% safety buffer
                    } else {
                        // For SHORT: entry must be BELOW all stops
                        let lowest_stop = stop_levels
                            .iter()
                            .map(|s| s.price)
                            .fold(f64::INFINITY, f64::min);
                        new_entry_price < lowest_stop * 0.98 // 2% safety buffer
                    };

                    // Validate the new entry price makes sense AND is safe from stops
                    let price_makes_sense = if direction == "LONG" {
                        new_entry_price < config.current_price && new_entry_price > 0.0
                    } else {
                        new_entry_price > config.current_price
                    };

                    if price_makes_sense && safe_from_stops {
                        entry.price = new_entry_price;
                        entry.atr_distance = (new_distance / config.current_price) * 100.0;
                        entries_adjusted = true;

                        log::debug!(
                            "  Adjusted Entry {}: ${:.4} → ${:.4} (moved {:.2}% further from current)",
                            i + 1,
                            config.current_price - current_distance * if direction == "LONG" { 1.0 } else { -1.0 },
                            new_entry_price,
                            (new_distance - current_distance) / config.current_price * 100.0
                        );
                    }
                }

                if entries_adjusted {
                    // CRITICAL: Stops should maintain their ORIGINAL absolute price, not move at all
                    // Only log that entries moved but stops stay exactly where they were
                    log::info!(
                        "🛑 Stops remain at original prices - entries moved but stops UNCHANGED"
                    );

                    // Recalculate R:R after both entry and stop adjustment
                    final_rr = Self::calculate_risk_reward(
                        &entry_levels,
                        &exit_levels,
                        &stop_levels,
                        &direction,
                    );

                    log::info!(
                        "🔧 Entry adjustment: R:R improved from {:.2} to {:.2} (stops unchanged)",
                        optimized_rr,
                        final_rr
                    );
                } else {
                    log::info!(
                        "🔧 No entry adjustments possible, keeping R:R at {:.2}",
                        optimized_rr
                    );
                }
            }

            // Step 4: SMART Quantity Rebalancing (if R:R still below target)
            if final_rr < target_risk_reward {
                // First try entry quantity optimization
                let entry_optimized_rr = Self::optimize_entry_quantities(
                    &mut entry_levels,
                    &exit_levels,
                    &stop_levels,
                    &direction,
                    final_rr,
                    target_risk_reward,
                );

                if entry_optimized_rr > final_rr {
                    log::info!(
                        "🎯 Entry quantity rebalancing: R:R improved from {:.2} to {:.2}",
                        final_rr,
                        entry_optimized_rr
                    );
                    final_rr = entry_optimized_rr;
                }

                // If still below target, try exit quantity optimization
                if final_rr < target_risk_reward {
                    let exit_optimized_rr = Self::optimize_exit_quantities(
                        &entry_levels,
                        &mut exit_levels,
                        &stop_levels,
                        &direction,
                        final_rr,
                        target_risk_reward,
                    );

                    if exit_optimized_rr > final_rr {
                        log::info!(
                            "🎯 Exit quantity rebalancing: R:R improved from {:.2} to {:.2}",
                            final_rr,
                            exit_optimized_rr
                        );
                        final_rr = exit_optimized_rr;
                    }
                }

                // If still below target, try stop quantity optimization
                if final_rr < target_risk_reward {
                    let stop_optimized_rr = Self::optimize_stop_quantities(
                        &entry_levels,
                        &exit_levels,
                        &mut stop_levels,
                        &direction,
                        final_rr,
                        target_risk_reward,
                    );

                    if stop_optimized_rr > final_rr {
                        log::info!(
                            "🎯 Stop quantity rebalancing: R:R improved from {:.2} to {:.2}",
                            final_rr,
                            stop_optimized_rr
                        );
                        final_rr = stop_optimized_rr;
                    }
                }

                if final_rr >= target_risk_reward {
                    log::info!(
                        "🎯 Quantity optimization reached target R:R {:.2}!",
                        target_risk_reward
                    );
                }
            }

            // Log final status
            if final_rr >= target_risk_reward * 0.9 {
                log::info!(
                    "✅ Successfully optimized R:R from {:.2} to {:.2} (target: {:.2})",
                    initial_risk_reward,
                    final_rr,
                    target_risk_reward
                );
            } else if final_rr > initial_risk_reward {
                log::info!(
                    "📈 Improved R:R from {:.2} to {:.2} (target: {:.2} limited by boundaries)",
                    initial_risk_reward,
                    final_rr,
                    target_risk_reward
                );
            } else {
                log::warn!(
                    "⚠️ Could not improve R:R (remains at {:.2}, target was {:.2})",
                    final_rr,
                    target_risk_reward
                );
            }

            final_rr
        } else {
            log::info!(
                "✅ Initial R:R {:.2} already meets/exceeds target {:.2}, no optimization needed",
                initial_risk_reward,
                target_risk_reward
            );
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
            // Entries must be BELOW current price (except MARKET orders which can be AT current price)
            for (i, entry) in entry_levels.iter().enumerate() {
                if entry.order_type == "MARKET" {
                    // MARKET orders can be AT current price for immediate execution
                    if (entry.price - current_price).abs() > current_price * 0.001 {
                        return Err(crate::utils::error::VangaError::PredictionError(format!(
                            "LONG MARKET entry {} at ${:.4} must be at current price ${:.4} (within 0.1%)",
                            i + 1,
                            entry.price,
                            current_price
                        )));
                    }
                } else if entry.price >= current_price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "LONG LIMIT entry {} at ${:.4} must be below current ${:.4}",
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
            levels[0].price // Fallback to first level price
        }
    }

    /// SMART Entry Quantity Rebalancing - Optimize R:R by adjusting entry quantity distribution
    /// Constraints: No quantity below MIN_QUANTITY (0.2) for reasonable lot sizes
    fn optimize_entry_quantities(
        entry_levels: &mut [OrderLevel; 3],
        exit_levels: &[OrderLevel; 3],
        stop_levels: &[OrderLevel; 3],
        direction: &str,
        current_rr: f64,
        target_rr: f64,
    ) -> f64 {
        const MIN_QUANTITY: f64 = 0.2; // Minimum quantity per level (20%)
        const MAX_ITERATIONS: usize = 10; // Prevent infinite loops
        const IMPROVEMENT_THRESHOLD: f64 = 0.01; // Stop if improvement less than 1%

        // Save original quantities for rollback if needed
        let original_quantities: [f64; 3] = [
            entry_levels[0].quantity_percentage,
            entry_levels[1].quantity_percentage,
            entry_levels[2].quantity_percentage,
        ];

        // For LONG: Move quantity from market (worst) to limit orders (better)
        // For SHORT: Move quantity from market (worst) to limit orders (better)
        let mut best_rr = current_rr;
        let mut iterations = 0;
        let mut improved = false;

        // Determine which entry has best price (lowest for LONG, highest for SHORT)
        let best_entry_idx = if direction == "LONG" {
            // For LONG: lowest price is best
            if entry_levels[1].price < entry_levels[2].price {
                1
            } else {
                2
            }
        } else {
            // For SHORT: highest price is best
            if entry_levels[1].price > entry_levels[2].price {
                1
            } else {
                2
            }
        };

        // Determine which entry has second best price
        let second_best_idx = if best_entry_idx == 1 { 2 } else { 1 };

        // Market entry is always index 0
        let market_idx = 0;

        while iterations < MAX_ITERATIONS {
            // Try moving 5% quantity from market to best entry
            let shift_amount = 0.05; // 5% shift per iteration

            // Ensure we don't go below minimum quantity
            if entry_levels[market_idx].quantity_percentage - shift_amount >= MIN_QUANTITY {
                // Move quantity from market to best entry
                entry_levels[market_idx].quantity_percentage -= shift_amount;
                entry_levels[best_entry_idx].quantity_percentage += shift_amount;

                // Normalize to ensure sum is 1.0
                let total: f64 = entry_levels.iter().map(|l| l.quantity_percentage).sum();
                for level in entry_levels.iter_mut() {
                    level.quantity_percentage /= total;
                }

                // Calculate new R:R
                let new_rr =
                    Self::calculate_risk_reward(entry_levels, exit_levels, stop_levels, direction);

                // If improved, keep going
                if new_rr > best_rr + IMPROVEMENT_THRESHOLD {
                    log::debug!(
                        "  Quantity shift: Market -{:.1}% → Best entry +{:.1}%, R:R: {:.2} → {:.2}",
                        shift_amount * 100.0,
                        shift_amount * 100.0,
                        best_rr,
                        new_rr
                    );
                    best_rr = new_rr;
                    improved = true;

                    // If we've reached target, stop
                    if new_rr >= target_rr {
                        log::info!(
                            "🎯 Quantity optimization reached target R:R {:.2}",
                            target_rr
                        );
                        break;
                    }
                } else {
                    // Try moving to second best entry instead
                    entry_levels[market_idx].quantity_percentage += shift_amount; // Undo
                    entry_levels[best_entry_idx].quantity_percentage -= shift_amount; // Undo

                    // Now try second best
                    if entry_levels[market_idx].quantity_percentage - shift_amount >= MIN_QUANTITY {
                        entry_levels[market_idx].quantity_percentage -= shift_amount;
                        entry_levels[second_best_idx].quantity_percentage += shift_amount;

                        // Normalize
                        let total: f64 = entry_levels.iter().map(|l| l.quantity_percentage).sum();
                        for level in entry_levels.iter_mut() {
                            level.quantity_percentage /= total;
                        }

                        // Calculate new R:R
                        let new_rr = Self::calculate_risk_reward(
                            entry_levels,
                            exit_levels,
                            stop_levels,
                            direction,
                        );

                        if new_rr > best_rr + IMPROVEMENT_THRESHOLD {
                            log::debug!(
                                "  Quantity shift: Market -{:.1}% → Second best +{:.1}%, R:R: {:.2} → {:.2}",
                                shift_amount * 100.0,
                                shift_amount * 100.0,
                                best_rr,
                                new_rr
                            );
                            best_rr = new_rr;
                            improved = true;
                        } else {
                            // Undo and stop - no more improvements possible
                            entry_levels[market_idx].quantity_percentage += shift_amount;
                            entry_levels[second_best_idx].quantity_percentage -= shift_amount;
                            break;
                        }
                    } else {
                        // Can't shift more from market - stop
                        break;
                    }
                }
            } else {
                // Can't shift more from market - try shifting between limit orders
                if entry_levels[second_best_idx].quantity_percentage - shift_amount >= MIN_QUANTITY
                {
                    entry_levels[second_best_idx].quantity_percentage -= shift_amount;
                    entry_levels[best_entry_idx].quantity_percentage += shift_amount;

                    // Normalize
                    let total: f64 = entry_levels.iter().map(|l| l.quantity_percentage).sum();
                    for level in entry_levels.iter_mut() {
                        level.quantity_percentage /= total;
                    }

                    // Calculate new R:R
                    let new_rr = Self::calculate_risk_reward(
                        entry_levels,
                        exit_levels,
                        stop_levels,
                        direction,
                    );

                    if new_rr > best_rr + IMPROVEMENT_THRESHOLD {
                        log::debug!(
                            "  Quantity shift: Second best -{:.1}% → Best +{:.1}%, R:R: {:.2} → {:.2}",
                            shift_amount * 100.0,
                            shift_amount * 100.0,
                            best_rr,
                            new_rr
                        );
                        best_rr = new_rr;
                        improved = true;
                    } else {
                        // Undo and stop - no more improvements possible
                        entry_levels[second_best_idx].quantity_percentage += shift_amount;
                        entry_levels[best_entry_idx].quantity_percentage -= shift_amount;
                        break;
                    }
                } else {
                    // Can't shift more - stop
                    break;
                }
            }

            iterations += 1;
        }

        // If no improvement, roll back to original quantities
        if !improved {
            entry_levels[0].quantity_percentage = original_quantities[0];
            entry_levels[1].quantity_percentage = original_quantities[1];
            entry_levels[2].quantity_percentage = original_quantities[2];
            return current_rr;
        }

        // Final normalization to ensure sum is exactly 1.0
        let total: f64 = entry_levels.iter().map(|l| l.quantity_percentage).sum();
        for level in entry_levels.iter_mut() {
            level.quantity_percentage /= total;
        }

        // Log final distribution
        log::info!(
            "🎯 Optimized quantity distribution: Market: {:.0}%, Limit1: {:.0}%, Limit2: {:.0}%",
            entry_levels[0].quantity_percentage * 100.0,
            entry_levels[1].quantity_percentage * 100.0,
            entry_levels[2].quantity_percentage * 100.0
        );

        best_rr
    }

    /// SMART Exit Quantity Rebalancing - Optimize R:R by adjusting exit quantity distribution
    /// Constraints: No quantity below MIN_QUANTITY (0.2) for reasonable lot sizes
    fn optimize_exit_quantities(
        entry_levels: &[OrderLevel; 3],
        exit_levels: &mut [OrderLevel; 3],
        stop_levels: &[OrderLevel; 3],
        direction: &str,
        current_rr: f64,
        target_rr: f64,
    ) -> f64 {
        const MIN_QUANTITY: f64 = 0.2; // Minimum quantity per level (20%)
        const MAX_ITERATIONS: usize = 10; // Prevent infinite loops
        const IMPROVEMENT_THRESHOLD: f64 = 0.01; // Stop if improvement less than 1%

        // Save original quantities for rollback if needed
        let original_quantities: [f64; 3] = [
            exit_levels[0].quantity_percentage,
            exit_levels[1].quantity_percentage,
            exit_levels[2].quantity_percentage,
        ];

        // For LONG: Move quantity from first exit (worst) to further exits (better)
        // For SHORT: Move quantity from first exit (worst) to further exits (better)
        let mut best_rr = current_rr;
        let mut iterations = 0;
        let mut improved = false;

        // First exit is always index 0, furthest exit is index 2
        let worst_exit_idx = 0;
        let best_exit_idx = 2;
        let middle_exit_idx = 1;

        while iterations < MAX_ITERATIONS {
            // Try moving 5% quantity from first exit to furthest exit
            let shift_amount = 0.05; // 5% shift per iteration

            // Ensure we don't go below minimum quantity
            if exit_levels[worst_exit_idx].quantity_percentage - shift_amount >= MIN_QUANTITY {
                // Move quantity from first exit to furthest exit
                exit_levels[worst_exit_idx].quantity_percentage -= shift_amount;
                exit_levels[best_exit_idx].quantity_percentage += shift_amount;

                // Normalize to ensure sum is 1.0
                let total: f64 = exit_levels.iter().map(|l| l.quantity_percentage).sum();
                for level in exit_levels.iter_mut() {
                    level.quantity_percentage /= total;
                }

                // Calculate new R:R
                let new_rr =
                    Self::calculate_risk_reward(entry_levels, exit_levels, stop_levels, direction);

                // If improved, keep going
                if new_rr > best_rr + IMPROVEMENT_THRESHOLD {
                    log::debug!(
                        "  Exit quantity shift: First -{:.1}% → Furthest +{:.1}%, R:R: {:.2} → {:.2}",
                        shift_amount * 100.0,
                        shift_amount * 100.0,
                        best_rr,
                        new_rr
                    );
                    best_rr = new_rr;
                    improved = true;

                    // If we've reached target, stop
                    if new_rr >= target_rr {
                        log::info!(
                            "ud83cudfaf Exit quantity optimization reached target R:R {:.2}",
                            target_rr
                        );
                        break;
                    }
                } else {
                    // Try moving to middle exit instead
                    exit_levels[worst_exit_idx].quantity_percentage += shift_amount; // Undo
                    exit_levels[best_exit_idx].quantity_percentage -= shift_amount; // Undo

                    // Now try middle exit
                    if exit_levels[worst_exit_idx].quantity_percentage - shift_amount
                        >= MIN_QUANTITY
                    {
                        exit_levels[worst_exit_idx].quantity_percentage -= shift_amount;
                        exit_levels[middle_exit_idx].quantity_percentage += shift_amount;

                        // Normalize
                        let total: f64 = exit_levels.iter().map(|l| l.quantity_percentage).sum();
                        for level in exit_levels.iter_mut() {
                            level.quantity_percentage /= total;
                        }

                        // Calculate new R:R
                        let new_rr = Self::calculate_risk_reward(
                            entry_levels,
                            exit_levels,
                            stop_levels,
                            direction,
                        );

                        if new_rr > best_rr + IMPROVEMENT_THRESHOLD {
                            log::debug!(
                                "  Exit quantity shift: First -{:.1}% → Middle +{:.1}%, R:R: {:.2} → {:.2}",
                                shift_amount * 100.0,
                                shift_amount * 100.0,
                                best_rr,
                                new_rr
                            );
                            best_rr = new_rr;
                            improved = true;
                        } else {
                            // Undo and stop - no more improvements possible
                            exit_levels[worst_exit_idx].quantity_percentage += shift_amount;
                            exit_levels[middle_exit_idx].quantity_percentage -= shift_amount;
                            break;
                        }
                    } else {
                        // Can't shift more from first exit - stop
                        break;
                    }
                }
            } else {
                // Can't shift more from first exit - try shifting between middle and furthest
                if exit_levels[middle_exit_idx].quantity_percentage - shift_amount >= MIN_QUANTITY {
                    exit_levels[middle_exit_idx].quantity_percentage -= shift_amount;
                    exit_levels[best_exit_idx].quantity_percentage += shift_amount;

                    // Normalize
                    let total: f64 = exit_levels.iter().map(|l| l.quantity_percentage).sum();
                    for level in exit_levels.iter_mut() {
                        level.quantity_percentage /= total;
                    }

                    // Calculate new R:R
                    let new_rr = Self::calculate_risk_reward(
                        entry_levels,
                        exit_levels,
                        stop_levels,
                        direction,
                    );

                    if new_rr > best_rr + IMPROVEMENT_THRESHOLD {
                        log::debug!(
                            "  Exit quantity shift: Middle -{:.1}% → Furthest +{:.1}%, R:R: {:.2} → {:.2}",
                            shift_amount * 100.0,
                            shift_amount * 100.0,
                            best_rr,
                            new_rr
                        );
                        best_rr = new_rr;
                        improved = true;
                    } else {
                        // Undo and stop - no more improvements possible
                        exit_levels[middle_exit_idx].quantity_percentage += shift_amount;
                        exit_levels[best_exit_idx].quantity_percentage -= shift_amount;
                        break;
                    }
                } else {
                    // Can't shift more - stop
                    break;
                }
            }

            iterations += 1;
        }

        // If no improvement, roll back to original quantities
        if !improved {
            exit_levels[0].quantity_percentage = original_quantities[0];
            exit_levels[1].quantity_percentage = original_quantities[1];
            exit_levels[2].quantity_percentage = original_quantities[2];
            return current_rr;
        }

        // Final normalization to ensure sum is exactly 1.0
        let total: f64 = exit_levels.iter().map(|l| l.quantity_percentage).sum();
        for level in exit_levels.iter_mut() {
            level.quantity_percentage /= total;
        }

        // Log final distribution
        log::info!(
            "ud83cudfaf Optimized exit distribution: First: {:.0}%, Middle: {:.0}%, Furthest: {:.0}%",
            exit_levels[0].quantity_percentage * 100.0,
            exit_levels[1].quantity_percentage * 100.0,
            exit_levels[2].quantity_percentage * 100.0
        );

        best_rr
    }

    /// SMART Stop Quantity Rebalancing - Optimize R:R by adjusting stop quantity distribution
    /// Constraints: No quantity below MIN_QUANTITY (0.2) for reasonable lot sizes
    fn optimize_stop_quantities(
        entry_levels: &[OrderLevel; 3],
        exit_levels: &[OrderLevel; 3],
        stop_levels: &mut [OrderLevel; 3],
        direction: &str,
        current_rr: f64,
        target_rr: f64,
    ) -> f64 {
        const MIN_QUANTITY: f64 = 0.2; // Minimum quantity per level (20%)
        const MAX_ITERATIONS: usize = 10; // Prevent infinite loops
        const IMPROVEMENT_THRESHOLD: f64 = 0.01; // Stop if improvement less than 1%

        // Save original quantities for rollback if needed
        let original_quantities: [f64; 3] = [
            stop_levels[0].quantity_percentage,
            stop_levels[1].quantity_percentage,
            stop_levels[2].quantity_percentage,
        ];

        // For LONG: Move quantity from furthest stop (worst) to closest stop (better)
        // For SHORT: Move quantity from furthest stop (worst) to closest stop (better)
        let mut best_rr = current_rr;
        let mut iterations = 0;
        let mut improved = false;

        // Closest stop is index 0, furthest stop is index 2
        let best_stop_idx = 0; // Closest stop (best for R:R)
        let worst_stop_idx = 2; // Furthest stop (worst for R:R)
        let middle_stop_idx = 1;

        while iterations < MAX_ITERATIONS {
            // Try moving 5% quantity from furthest stop to closest stop
            let shift_amount = 0.05; // 5% shift per iteration

            // Ensure we don't go below minimum quantity
            if stop_levels[worst_stop_idx].quantity_percentage - shift_amount >= MIN_QUANTITY {
                // Move quantity from furthest stop to closest stop
                stop_levels[worst_stop_idx].quantity_percentage -= shift_amount;
                stop_levels[best_stop_idx].quantity_percentage += shift_amount;

                // Normalize to ensure sum is 1.0
                let total: f64 = stop_levels.iter().map(|l| l.quantity_percentage).sum();
                for level in stop_levels.iter_mut() {
                    level.quantity_percentage /= total;
                }

                // Calculate new R:R
                let new_rr =
                    Self::calculate_risk_reward(entry_levels, exit_levels, stop_levels, direction);

                // If improved, keep going
                if new_rr > best_rr + IMPROVEMENT_THRESHOLD {
                    log::debug!(
                        "  Stop quantity shift: Furthest -{:.1}% → Closest +{:.1}%, R:R: {:.2} → {:.2}",
                        shift_amount * 100.0,
                        shift_amount * 100.0,
                        best_rr,
                        new_rr
                    );
                    best_rr = new_rr;
                    improved = true;

                    // If we've reached target, stop
                    if new_rr >= target_rr {
                        log::info!(
                            "ud83cudfaf Stop quantity optimization reached target R:R {:.2}",
                            target_rr
                        );
                        break;
                    }
                } else {
                    // Try moving to middle stop instead
                    stop_levels[worst_stop_idx].quantity_percentage += shift_amount; // Undo
                    stop_levels[best_stop_idx].quantity_percentage -= shift_amount; // Undo

                    // Now try middle stop
                    if stop_levels[worst_stop_idx].quantity_percentage - shift_amount
                        >= MIN_QUANTITY
                    {
                        stop_levels[worst_stop_idx].quantity_percentage -= shift_amount;
                        stop_levels[middle_stop_idx].quantity_percentage += shift_amount;

                        // Normalize
                        let total: f64 = stop_levels.iter().map(|l| l.quantity_percentage).sum();
                        for level in stop_levels.iter_mut() {
                            level.quantity_percentage /= total;
                        }

                        // Calculate new R:R
                        let new_rr = Self::calculate_risk_reward(
                            entry_levels,
                            exit_levels,
                            stop_levels,
                            direction,
                        );

                        if new_rr > best_rr + IMPROVEMENT_THRESHOLD {
                            log::debug!(
                                "  Stop quantity shift: Furthest -{:.1}% → Middle +{:.1}%, R:R: {:.2} → {:.2}",
                                shift_amount * 100.0,
                                shift_amount * 100.0,
                                best_rr,
                                new_rr
                            );
                            best_rr = new_rr;
                            improved = true;
                        } else {
                            // Undo and stop - no more improvements possible
                            stop_levels[worst_stop_idx].quantity_percentage += shift_amount;
                            stop_levels[middle_stop_idx].quantity_percentage -= shift_amount;
                            break;
                        }
                    } else {
                        // Can't shift more from furthest stop - stop
                        break;
                    }
                }
            } else {
                // Can't shift more from furthest stop - try shifting between middle and closest
                if stop_levels[middle_stop_idx].quantity_percentage - shift_amount >= MIN_QUANTITY {
                    stop_levels[middle_stop_idx].quantity_percentage -= shift_amount;
                    stop_levels[best_stop_idx].quantity_percentage += shift_amount;

                    // Normalize
                    let total: f64 = stop_levels.iter().map(|l| l.quantity_percentage).sum();
                    for level in stop_levels.iter_mut() {
                        level.quantity_percentage /= total;
                    }

                    // Calculate new R:R
                    let new_rr = Self::calculate_risk_reward(
                        entry_levels,
                        exit_levels,
                        stop_levels,
                        direction,
                    );

                    if new_rr > best_rr + IMPROVEMENT_THRESHOLD {
                        log::debug!(
                            "  Stop quantity shift: Middle -{:.1}% → Closest +{:.1}%, R:R: {:.2} → {:.2}",
                            shift_amount * 100.0,
                            shift_amount * 100.0,
                            best_rr,
                            new_rr
                        );
                        best_rr = new_rr;
                        improved = true;
                    } else {
                        // Undo and stop - no more improvements possible
                        stop_levels[middle_stop_idx].quantity_percentage += shift_amount;
                        stop_levels[best_stop_idx].quantity_percentage -= shift_amount;
                        break;
                    }
                } else {
                    // Can't shift more - stop
                    break;
                }
            }

            iterations += 1;
        }

        // If no improvement, roll back to original quantities
        if !improved {
            stop_levels[0].quantity_percentage = original_quantities[0];
            stop_levels[1].quantity_percentage = original_quantities[1];
            stop_levels[2].quantity_percentage = original_quantities[2];
            return current_rr;
        }

        // Final normalization to ensure sum is exactly 1.0
        let total: f64 = stop_levels.iter().map(|l| l.quantity_percentage).sum();
        for level in stop_levels.iter_mut() {
            level.quantity_percentage /= total;
        }

        // Log final distribution
        log::info!(
            "ud83cudfaf Optimized stop distribution: Closest: {:.0}%, Middle: {:.0}%, Furthest: {:.0}%",
            stop_levels[0].quantity_percentage * 100.0,
            stop_levels[1].quantity_percentage * 100.0,
            stop_levels[2].quantity_percentage * 100.0
        );

        best_rr
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
        SmartConsensus::normalize_sizes(&mut stop_levels); // FIX: Normalize stops too!

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
