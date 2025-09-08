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
                entry.price =
                    consensus.adjust_for_psychological_levels(entry.price, Some(&ohlcv_arrays));
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
                exit.price =
                    consensus.adjust_for_psychological_levels(exit.price, Some(&ohlcv_arrays));
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
