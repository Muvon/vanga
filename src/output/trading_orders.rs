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

/// Type alias for order level arrays (entry, exit, stop)
type OrderLevelArrays = ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]);

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
            min_risk_reward: 3.5,     // Crypto minimum R:R target
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

        // Step 1: Determine direction using Direction + Sentiment models with alignment validation
        let (direction, direction_confidence) = match consensus.calculate_direction_consensus() {
            Ok(result) => result,
            Err(alignment_error) => {
                return Err(crate::utils::error::VangaError::PredictionError(format!(
                    "Direction-Price Level alignment failed: {}",
                    alignment_error
                )));
            }
        };

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

            // Parse horizon using the existing utility function
            let horizon_hours = crate::utils::parser::parse_horizon_to_steps(
                &config.volatility_pred.training_horizon,
            )
            .unwrap_or(4) as f64;

            // Calculate sequence statistics for adaptive generation
            let sequence_stats = SequenceStatistics::from_prices(
                &sequence_prices,
                horizon_hours, // Use actual model horizon
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

        // Step 4: Generate MODEL-BASED exit levels using LSTM predictions
        let mut exit_levels = {
            log::info!("📊 Using MODEL-BASED exit generation with price level predictions");

            // ALWAYS use model predictions for exits - this is why we trained the LSTM!
            let mut exits = consensus.generate_smart_exits(config.current_price, &direction)?;

            // Apply psychological level adjustments using sequence data
            let ohlcv_arrays: Vec<[f64; 5]> = config
                .sequence_ohlcv
                .expect("Sequence OHLCV data should always be available")
                .iter()
                .map(|row| [row.open, row.high, row.low, row.close, row.volume])
                .collect();

            // Apply separation-aware psychological adjustments to exits
            Self::apply_separation_aware_psychological_adjustments(
                &mut exits,
                &consensus,
                &ohlcv_arrays,
                false, // is_stop_loss = false
                Some(&direction),
                config.current_price,
            );

            log::info!("🎯 Generated MODEL-BASED exits with psychological adjustments");
            exits
        };

        // Step 5: Generate ADAPTIVE stop levels using ALL prediction data
        let sequence_prices: Vec<f64> = config
            .sequence_ohlcv
            .expect("Sequence OHLCV data should always be available")
            .iter()
            .map(|row| row.close)
            .collect();

        let mut stop_levels =
            consensus.generate_adaptive_stops(&entry_levels, &direction, &sequence_prices)?;

        // Apply separation-aware psychological level adjustments to stops
        let ohlcv_arrays: Vec<[f64; 5]> = config
            .sequence_ohlcv
            .expect("Sequence OHLCV data should always be available")
            .iter()
            .map(|row| [row.open, row.high, row.low, row.close, row.volume])
            .collect();

        Self::apply_separation_aware_psychological_adjustments(
            &mut stop_levels,
            &consensus,
            &ohlcv_arrays,
            true, // is_stop_loss = true
            Some(&direction),
            config.current_price,
        );

        log::info!("🛡️ Applied stop hunt protection via psychological level adjustments");

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

        // Step 8: ATR-based spacing validation and adjustment (post-processing)
        // Calculate sequence statistics from OHLCV data
        let sequence_ohlcv = config
            .sequence_ohlcv
            .expect("Sequence OHLCV data should always be available");

        let prices: Vec<f64> = sequence_ohlcv.iter().map(|row| row.close).collect();
        let volumes: Vec<f64> = sequence_ohlcv.iter().map(|row| row.volume).collect();

        // Parse horizon using the existing utility function
        let horizon_hours =
            crate::utils::parser::parse_horizon_to_steps(&config.volatility_pred.training_horizon)
                .unwrap_or(24) as f64;

        let sequence_stats = crate::output::sequence_statistics::SequenceStatistics::from_prices(
            &prices,
            horizon_hours,
            Some(&volumes), // Use actual volume data from OHLCV
        )?;

        let atr_adjustment_result = Self::apply_atr_spacing_validation(
            entry_levels,
            exit_levels,
            stop_levels,
            &direction,
            &sequence_stats,
            config.current_price,
        )?;

        let (mut entry_levels, mut exit_levels, mut stop_levels) = atr_adjustment_result;
        if let Err(validation_error) = Self::validate_order_integrity(
            &entry_levels,
            &exit_levels,
            &stop_levels,
            &direction,
            config.current_price,
        ) {
            log::error!("❌ Order validation failed: {}", validation_error);
            log::error!(
                "🚫 Skipping prediction due to validation failure - no signal will be output"
            );

            // Return error instead of empty orders - this will cause the prediction to be skipped entirely
            return Err(crate::utils::error::VangaError::PredictionError(format!(
                "Order validation failed: {}. No prediction will be output to avoid zero prices.",
                validation_error
            )));
        }

        // Step 9: Calculate initial risk-reward ratio
        let initial_risk_reward =
            Self::calculate_risk_reward(&entry_levels, &exit_levels, &stop_levels, &direction);

        // Step 10: Use CONFIGURED minimum R:R as base target, enhanced by model confidence
        let order_config = OrderConfig::default();
        let configured_min_rr = order_config.min_risk_reward;

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

            log::debug!("Average stop distance: {:.2}%", avg_stop_distance);

            // Calculate unified model boundaries for consistent optimization
            let model_boundaries = crate::output::model_boundaries::ModelBoundaries::calculate(
                config.price_levels,
                config.current_price,
                &direction,
                config.volatility_pred.expected_range_percent,
            );

            log::info!(
                "🎯 Model Boundaries: max_exit={:.2}% (${:.5}), absolute={:.2}% (${:.5}), suitable_bins={}",
                model_boundaries.max_exit_boundary_percent,
                model_boundaries.max_exit_boundary_price,
                model_boundaries.absolute_boundary_percent,
                model_boundaries.absolute_boundary_price,
                model_boundaries.suitable_bins.len()
            );

            // Use model boundary as the optimization limit
            let max_exit_boundary = model_boundaries.max_exit_boundary_percent;

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
            log::debug!(
                "Using full boundary for middle exit: {:.2}%",
                max_exit_boundary
            );

            // Maximum scale is when middle exit reaches model boundary (not exceeds)
            let max_scale_factor =
                model_boundaries.get_max_scale_factor(current_middle_exit_distance);

            log::info!(
                "📏 Middle exit constraint: current={:.2}%, model_boundary={:.2}%, max_scale={:.2}x",
                current_middle_exit_distance,
                model_boundaries.max_exit_boundary_percent,
                max_scale_factor
            );

            let desired_scale = target_risk_reward / initial_risk_reward.max(0.1);

            // CRITICAL: Apply the scale to reach target R:R, capped by model boundary
            // The boundary ensures middle exit doesn't exceed the most predicted class center
            let final_scale = desired_scale.min(max_scale_factor);

            log::info!(
                "🔄 Model-bounded scaling: desired={:.2}x, model_max={:.2}x, final={:.2}x",
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

                // CRITICAL: Validate exit respects model boundaries
                if let Err(boundary_error) =
                    model_boundaries.validate_exit_price(exit.price, config.current_price)
                {
                    log::warn!(
                        "⚠️ Exit {} violates model boundary: {}. Adjusting to boundary.",
                        i + 1,
                        boundary_error
                    );

                    // Adjust to stay within absolute boundary
                    exit.price = if direction == "LONG" {
                        exit.price.min(model_boundaries.absolute_boundary_price)
                    } else {
                        exit.price.max(model_boundaries.absolute_boundary_price)
                    };
                }

                // Update ATR distance
                exit.atr_distance =
                    ((exit.price - config.current_price).abs() / config.current_price) * 100.0;

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

                // FIXED: Apply proportional adjustment to maintain entry order relationships
                // Calculate the maximum safe movement for all entries
                let movement_percentage = max_entry_movement / 100.0;

                // Check safety constraints for ALL entries before applying any changes
                let mut can_adjust_all = true;
                let mut new_entry_prices = Vec::new();

                for entry in entry_levels.iter() {
                    if entry.order_type == "MARKET" {
                        new_entry_prices.push(entry.price); // Keep market orders unchanged
                        continue;
                    }

                    // Calculate new price with proportional movement
                    let current_distance = (entry.price - config.current_price).abs();
                    let new_distance = current_distance * (1.0 + movement_percentage);

                    let new_entry_price = if direction == "LONG" {
                        // LONG: move entries LOWER (better entry prices)
                        config.current_price - new_distance
                    } else {
                        // SHORT: move entries HIGHER (better entry prices)
                        config.current_price + new_distance
                    };

                    // Validate this new price
                    let price_makes_sense = if direction == "LONG" {
                        new_entry_price < config.current_price && new_entry_price > 0.0
                    } else {
                        new_entry_price > config.current_price
                    };

                    if !price_makes_sense {
                        can_adjust_all = false;
                        break;
                    }

                    new_entry_prices.push(new_entry_price);
                }

                // Check that new entries don't intersect with stops
                if can_adjust_all {
                    let extreme_new_entry = if direction == "LONG" {
                        new_entry_prices
                            .iter()
                            .fold(f64::INFINITY, |a, &b| a.min(b))
                    } else {
                        new_entry_prices
                            .iter()
                            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                    };

                    let safe_from_stops = if direction == "LONG" {
                        let highest_stop = stop_levels
                            .iter()
                            .map(|s| s.price)
                            .fold(f64::NEG_INFINITY, f64::max);
                        extreme_new_entry > highest_stop * 1.02 // 2% safety buffer
                    } else {
                        let lowest_stop = stop_levels
                            .iter()
                            .map(|s| s.price)
                            .fold(f64::INFINITY, f64::min);
                        extreme_new_entry < lowest_stop * 0.98 // 2% safety buffer
                    };

                    if safe_from_stops {
                        // Apply all adjustments - they maintain relative order
                        for (i, entry) in entry_levels.iter_mut().enumerate() {
                            if entry.order_type != "MARKET" {
                                let old_price = entry.price;
                                entry.price = new_entry_prices[i];
                                entry.atr_distance = ((entry.price - config.current_price).abs()
                                    / config.current_price)
                                    * 100.0;
                                entries_adjusted = true;

                                log::debug!(
                                    "  Proportionally Adjusted Entry {}: ${:.4} → ${:.4} (moved {:.2}% further)",
                                    i + 1,
                                    old_price,
                                    entry.price,
                                    movement_percentage * 100.0
                                );
                            }
                        }
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

        // Step 11: Recalculate FINAL risk-reward ratio after ALL adjustments (including ATR)
        // This ensures the reported R:R matches the actual orders being generated
        let final_risk_reward =
            Self::calculate_risk_reward(&entry_levels, &exit_levels, &stop_levels, &direction);

        // Log if R:R changed due to ATR adjustment
        if (final_risk_reward - risk_reward).abs() > 0.01 {
            log::info!(
                "📊 Final R:R after ATR adjustment: {:.2} (was {:.2} before ATR validation)",
                final_risk_reward,
                risk_reward
            );
        }

        // Use the FINAL risk_reward for reporting and the struct
        let risk_reward = final_risk_reward;

        log::info!(
            "✅ SMART Orders Generated: FINAL R:R={:.2}, Direction={}, Confidence={:.2}",
            risk_reward,
            direction,
            overall_confidence
        );

        // Log detailed order information with weighted averages for transparency
        let weighted_avg_entry = Self::weighted_average_price(&entry_levels);
        let weighted_avg_exit = Self::weighted_average_price(&exit_levels);
        let weighted_avg_stop = Self::weighted_average_price(&stop_levels);

        log::info!("📊 Weighted Average Prices (used for R:R calculation):");
        log::info!("  Avg Entry: ${:.2}", weighted_avg_entry);
        log::info!("  Avg Exit:  ${:.2}", weighted_avg_exit);
        log::info!("  Avg Stop:  ${:.2}", weighted_avg_stop);

        if direction == "LONG" {
            let profit = weighted_avg_exit - weighted_avg_entry;
            let loss = weighted_avg_entry - weighted_avg_stop;
            log::info!(
                "  Potential Profit: ${:.2} ({:.2}%)",
                profit,
                (profit / weighted_avg_entry) * 100.0
            );
            log::info!(
                "  Potential Loss:   ${:.2} ({:.2}%)",
                loss,
                (loss / weighted_avg_entry) * 100.0
            );
        } else {
            let profit = weighted_avg_entry - weighted_avg_exit;
            let loss = weighted_avg_stop - weighted_avg_entry;
            log::info!(
                "  Potential Profit: ${:.2} ({:.2}%)",
                profit,
                (profit / weighted_avg_entry) * 100.0
            );
            log::info!(
                "  Potential Loss:   ${:.2} ({:.2}%)",
                loss,
                (loss / weighted_avg_entry) * 100.0
            );
        }

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

    /// Comprehensive validation of order integrity - called at the END after all generation
    /// Returns detailed error if any issues found, Ok(()) if all checks pass
    pub fn validate_order_integrity(
        entry_levels: &[OrderLevel; 3],
        exit_levels: &[OrderLevel; 3],
        stop_levels: &[OrderLevel; 3],
        direction: &str,
        current_price: f64,
    ) -> crate::utils::error::Result<()> {
        // Check (a): All arrays have exactly 3 levels
        if entry_levels.len() != 3 || exit_levels.len() != 3 || stop_levels.len() != 3 {
            return Err(crate::utils::error::VangaError::PredictionError(
                "Order arrays must have exactly 3 levels each".to_string(),
            ));
        }

        // Check (b): Quantities sum to 1.0 (±0.01 tolerance)
        let entry_sum: f64 = entry_levels.iter().map(|l| l.quantity_percentage).sum();
        let exit_sum: f64 = exit_levels.iter().map(|l| l.quantity_percentage).sum();
        let stop_sum: f64 = stop_levels.iter().map(|l| l.quantity_percentage).sum();

        if (entry_sum - 1.0).abs() > 0.01 {
            return Err(crate::utils::error::VangaError::PredictionError(format!(
                "Entry quantities sum to {:.3}, expected 1.0 (±0.01)",
                entry_sum
            )));
        }
        if (exit_sum - 1.0).abs() > 0.01 {
            return Err(crate::utils::error::VangaError::PredictionError(format!(
                "Exit quantities sum to {:.3}, expected 1.0 (±0.01)",
                exit_sum
            )));
        }
        if (stop_sum - 1.0).abs() > 0.01 {
            return Err(crate::utils::error::VangaError::PredictionError(format!(
                "Stop quantities sum to {:.3}, expected 1.0 (±0.01)",
                stop_sum
            )));
        }

        // Check (c): No duplicate prices within same order type
        let entry_prices: Vec<f64> = entry_levels.iter().map(|l| l.price).collect();
        let exit_prices: Vec<f64> = exit_levels.iter().map(|l| l.price).collect();
        let stop_prices: Vec<f64> = stop_levels.iter().map(|l| l.price).collect();

        for i in 0..3 {
            for j in (i + 1)..3 {
                // Prices are duplicate if they are exactly equal (floating point equality)
                if entry_prices[i] == entry_prices[j] {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "Duplicate entry prices: Entry {} and {} both at ${}",
                        i + 1,
                        j + 1,
                        entry_prices[i]
                    )));
                }
                if exit_prices[i] == exit_prices[j] {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "Duplicate exit prices: Exit {} and {} both at ${}",
                        i + 1,
                        j + 1,
                        exit_prices[i]
                    )));
                }
                if stop_prices[i] == stop_prices[j] {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "Duplicate stop prices: Stop {} and {} both at ${}",
                        i + 1,
                        j + 1,
                        stop_prices[i]
                    )));
                }
            }
        }

        // Check (d): All prices > 0
        for (i, entry) in entry_levels.iter().enumerate() {
            if entry.price <= 0.0 {
                return Err(crate::utils::error::VangaError::PredictionError(format!(
                    "Entry {} price ${:.4} must be > 0",
                    i + 1,
                    entry.price
                )));
            }
        }
        for (i, exit) in exit_levels.iter().enumerate() {
            if exit.price <= 0.0 {
                return Err(crate::utils::error::VangaError::PredictionError(format!(
                    "Exit {} price ${:.4} must be > 0",
                    i + 1,
                    exit.price
                )));
            }
        }
        for (i, stop) in stop_levels.iter().enumerate() {
            if stop.price <= 0.0 {
                return Err(crate::utils::error::VangaError::PredictionError(format!(
                    "Stop {} price ${:.4} must be > 0",
                    i + 1,
                    stop.price
                )));
            }
        }

        // Check (e): Proper ordering based on direction
        if direction == "LONG" {
            // LONG entries should be descending (highest first for best opportunity)
            for i in 0..2 {
                if entry_levels[i].price < entry_levels[i + 1].price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "LONG entry ordering broken: Entry {} (${:.4}) should be >= Entry {} (${:.4})",
                        i + 1,
                        entry_levels[i].price,
                        i + 2,
                        entry_levels[i + 1].price
                    )));
                }
            }
            // LONG exits should be ascending (lowest target first)
            for i in 0..2 {
                if exit_levels[i].price > exit_levels[i + 1].price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "LONG exit ordering broken: Exit {} (${:.4}) should be <= Exit {} (${:.4})",
                        i + 1,
                        exit_levels[i].price,
                        i + 2,
                        exit_levels[i + 1].price
                    )));
                }
            }
            // LONG stops should be descending (highest stop first)
            for i in 0..2 {
                if stop_levels[i].price < stop_levels[i + 1].price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "LONG stop ordering broken: Stop {} (${:.4}) should be >= Stop {} (${:.4})",
                        i + 1,
                        stop_levels[i].price,
                        i + 2,
                        stop_levels[i + 1].price
                    )));
                }
            }
        } else if direction == "SHORT" {
            // SHORT entries should be ascending (lowest first for best opportunity)
            for i in 0..2 {
                if entry_levels[i].price > entry_levels[i + 1].price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "SHORT entry ordering broken: Entry {} (${:.4}) should be <= Entry {} (${:.4})",
                        i + 1,
                        entry_levels[i].price,
                        i + 2,
                        entry_levels[i + 1].price
                    )));
                }
            }
            // SHORT exits should be descending (highest target first)
            for i in 0..2 {
                if exit_levels[i].price < exit_levels[i + 1].price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "SHORT exit ordering broken: Exit {} (${:.4}) should be >= Exit {} (${:.4})",
                        i + 1,
                        exit_levels[i].price,
                        i + 2,
                        exit_levels[i + 1].price
                    )));
                }
            }
            // SHORT stops should be ascending (lowest stop first)
            for i in 0..2 {
                if stop_levels[i].price > stop_levels[i + 1].price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "SHORT stop ordering broken: Stop {} (${:.4}) should be <= Stop {} (${:.4})",
                        i + 1,
                        stop_levels[i].price,
                        i + 2,
                        stop_levels[i + 1].price
                    )));
                }
            }
        }

        // Check (f): No intersection between entries and stops
        if direction == "LONG" {
            // For LONG: all stops must be below all entries
            let lowest_entry = entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::INFINITY, f64::min);
            let highest_stop = stop_levels
                .iter()
                .map(|s| s.price)
                .fold(f64::NEG_INFINITY, f64::max);

            if highest_stop >= lowest_entry {
                return Err(crate::utils::error::VangaError::PredictionError(format!(
                    "LONG stop-entry intersection: highest stop ${:.4} >= lowest entry ${:.4}",
                    highest_stop, lowest_entry
                )));
            }

            // All entries must be below or at current price (except MARKET orders)
            for (i, entry) in entry_levels.iter().enumerate() {
                if entry.order_type != "MARKET" && entry.price > current_price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "LONG entry {} at ${:.4} must be <= current price ${:.4}",
                        i + 1,
                        entry.price,
                        current_price
                    )));
                }
            }

            // All exits must be above current price
            for (i, exit) in exit_levels.iter().enumerate() {
                if exit.price <= current_price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "LONG exit {} at ${:.4} must be > current price ${:.4}",
                        i + 1,
                        exit.price,
                        current_price
                    )));
                }
            }
        } else if direction == "SHORT" {
            // For SHORT: all stops must be above all entries
            let highest_entry = entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::NEG_INFINITY, f64::max);
            let lowest_stop = stop_levels
                .iter()
                .map(|s| s.price)
                .fold(f64::INFINITY, f64::min);

            if lowest_stop <= highest_entry {
                return Err(crate::utils::error::VangaError::PredictionError(format!(
                    "SHORT stop-entry intersection: lowest stop ${:.4} <= highest entry ${:.4}",
                    lowest_stop, highest_entry
                )));
            }

            // All entries must be above or at current price (except MARKET orders)
            for (i, entry) in entry_levels.iter().enumerate() {
                if entry.order_type != "MARKET" && entry.price < current_price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "SHORT entry {} at ${:.4} must be >= current price ${:.4}",
                        i + 1,
                        entry.price,
                        current_price
                    )));
                }
            }

            // All exits must be below current price
            for (i, exit) in exit_levels.iter().enumerate() {
                if exit.price >= current_price {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "SHORT exit {} at ${:.4} must be < current price ${:.4}",
                        i + 1,
                        exit.price,
                        current_price
                    )));
                }
            }
        }

        log::info!("✅ Order integrity validation passed - all orders properly aligned");
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

        // Get direction consensus with error handling
        let (direction, _direction_confidence) = consensus.calculate_direction_consensus()?;

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

    /// ATR-based spacing validation and adjustment (post-processing)
    /// Ensures professional spacing standards while preserving model intelligence
    fn apply_atr_spacing_validation(
        entry_levels: [OrderLevel; 3],
        exit_levels: [OrderLevel; 3],
        stop_levels: [OrderLevel; 3],
        direction: &str,
        sequence_stats: &crate::output::sequence_statistics::SequenceStatistics,
        current_price: f64,
    ) -> Result<OrderLevelArrays, crate::utils::error::VangaError> {
        // Calculate true ATR from sequence data (convert to percentage)
        let true_atr_percent = sequence_stats.std_return * 100.0;

        // Calculate maximum entry spacing
        let max_entry_spacing = Self::calculate_max_entry_spacing(&entry_levels, current_price);

        // Professional minimum: stop distance >= max(entry_spacing, 2x ATR)
        let min_stop_distance = max_entry_spacing.max(true_atr_percent * 2.0);

        // Calculate current minimum stop distance from worst entry
        let current_min_stop_distance = Self::calculate_min_stop_distance(
            &entry_levels,
            &stop_levels,
            direction,
            current_price,
        );

        log::info!(
            "🔍 ATR Spacing Analysis: true_ATR={:.2}%, max_entry_spacing={:.2}%, min_required_stop={:.2}%, current_stop={:.2}%",
            true_atr_percent, max_entry_spacing, min_stop_distance, current_min_stop_distance
        );

        // Check if spacing violation exists
        if current_min_stop_distance < min_stop_distance {
            let scale_factor = min_stop_distance / current_min_stop_distance.max(0.01);

            log::info!(
                "⚠️ Spacing violation detected: stop distance {:.2}% < required {:.2}%, scaling by {:.2}x",
                current_min_stop_distance, min_stop_distance, scale_factor
            );

            // Scale stops proportionally to meet minimum distance
            // CRITICAL: Scale each stop individually to maintain their relative spacing
            let mut adjusted_stops = stop_levels;
            let worst_entry_price = Self::get_worst_entry_price(&entry_levels, direction);

            // Calculate the current closest stop distance as percentage
            let closest_stop_distance_pct = adjusted_stops
                .iter()
                .map(|stop| ((stop.price - worst_entry_price).abs() / current_price) * 100.0)
                .fold(f64::INFINITY, f64::min);

            // Apply scaling to each stop individually to maintain their relative proportions
            for stop in adjusted_stops.iter_mut() {
                let current_distance_pct =
                    ((stop.price - worst_entry_price).abs() / current_price) * 100.0;

                // Each stop maintains its relative distance ratio
                // If stop was 2x further than closest, it remains 2x further after scaling
                let distance_ratio = current_distance_pct / closest_stop_distance_pct;
                let new_distance_pct = min_stop_distance * distance_ratio;
                let new_distance = (new_distance_pct / 100.0) * current_price;

                stop.price = if direction == "SHORT" {
                    worst_entry_price + new_distance
                } else {
                    worst_entry_price - new_distance
                };

                // Update ATR distance
                stop.atr_distance = new_distance_pct;
            }

            // Scale exits proportionally to maintain R:R AND preserve relative spacing
            let mut adjusted_exits = exit_levels;

            // CRITICAL: Scale the ENTIRE exit structure proportionally
            // Find the reference point (closest exit to current price)
            let reference_exit_distance = adjusted_exits
                .iter()
                .map(|exit| (exit.price - current_price).abs())
                .fold(f64::INFINITY, f64::min);

            // Calculate the scaling anchor point
            let reference_exit_price = if direction == "LONG" {
                // For LONG, find the lowest exit (closest to current)
                adjusted_exits
                    .iter()
                    .map(|exit| exit.price)
                    .fold(f64::INFINITY, f64::min)
            } else {
                // For SHORT, find the highest exit (closest to current)
                adjusted_exits
                    .iter()
                    .map(|exit| exit.price)
                    .fold(0.0, f64::max)
            };

            // Scale the reference distance
            let scaled_reference_distance = reference_exit_distance * scale_factor;
            let new_reference_price = if direction == "LONG" {
                current_price + scaled_reference_distance
            } else {
                current_price - scaled_reference_distance
            };

            // Calculate the scaling factor for the exit structure
            let structure_scale_factor = if reference_exit_price != current_price {
                (new_reference_price - current_price) / (reference_exit_price - current_price)
            } else {
                scale_factor
            };

            // Apply proportional scaling to ALL exits
            for exit in adjusted_exits.iter_mut() {
                // Scale relative to current price, maintaining proportional relationships
                let original_offset = exit.price - current_price;
                let scaled_offset = original_offset * structure_scale_factor;
                exit.price = current_price + scaled_offset;

                // Update ATR distance
                exit.atr_distance = (scaled_offset.abs() / current_price) * 100.0;
            }

            log::info!(
                "📊 Proportional exit scaling: reference_distance={:.2}% → {:.2}%, structure_factor={:.2}x",
                reference_exit_distance / current_price * 100.0,
                scaled_reference_distance / current_price * 100.0,
                structure_scale_factor
            );

            log::info!(
                "✅ ATR spacing adjustment applied: stops and exits scaled {:.2}x to maintain mathematical consistency",
                scale_factor
            );

            Ok((entry_levels, adjusted_exits, adjusted_stops))
        } else {
            log::info!("✅ ATR spacing validation passed: no adjustment needed");
            Ok((entry_levels, exit_levels, stop_levels))
        }
    }

    /// Calculate maximum spacing between consecutive entry levels
    fn calculate_max_entry_spacing(entry_levels: &[OrderLevel; 3], current_price: f64) -> f64 {
        let mut max_spacing = 0.0;

        for i in 0..entry_levels.len() - 1 {
            let spacing =
                ((entry_levels[i + 1].price - entry_levels[i].price).abs() / current_price) * 100.0;
            max_spacing = if spacing > max_spacing {
                spacing
            } else {
                max_spacing
            };
        }

        max_spacing
    }

    /// Calculate minimum stop distance from worst entry
    fn calculate_min_stop_distance(
        entry_levels: &[OrderLevel; 3],
        stop_levels: &[OrderLevel; 3],
        direction: &str,
        current_price: f64,
    ) -> f64 {
        let worst_entry_price = Self::get_worst_entry_price(entry_levels, direction);

        // Find closest stop to worst entry
        let min_stop_distance = stop_levels
            .iter()
            .map(|stop| ((stop.price - worst_entry_price).abs() / current_price) * 100.0)
            .fold(f64::INFINITY, f64::min);

        min_stop_distance
    }

    /// Get worst entry price (furthest from current price in trade direction)
    fn get_worst_entry_price(entry_levels: &[OrderLevel; 3], direction: &str) -> f64 {
        if direction == "SHORT" {
            // For SHORT, worst entry is highest price
            entry_levels.iter().map(|e| e.price).fold(0.0, f64::max)
        } else {
            // For LONG, worst entry is lowest price
            entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::INFINITY, f64::min)
        }
    }

    /// Apply psychological level adjustments while maintaining separation between orders
    fn apply_separation_aware_psychological_adjustments(
        orders: &mut [OrderLevel; 3],
        consensus: &SmartConsensus,
        ohlcv_arrays: &Vec<[f64; 5]>,
        is_stop_loss: bool,
        direction: Option<&str>,
        current_price: f64,
    ) {
        let min_separation = current_price * 0.001; // 0.1% minimum separation

        // Apply psychological adjustments one by one, checking separation
        for i in 0..orders.len() {
            let original_price = orders[i].price;

            // Apply psychological adjustment
            let adjusted_price = consensus.adjust_for_psychological_levels(
                original_price,
                Some(ohlcv_arrays),
                is_stop_loss,
                direction,
            );

            // Check if adjustment would create duplicates with previous orders
            let mut final_price = adjusted_price;
            for (j, other_order) in orders.iter().enumerate().take(i) {
                if (final_price - other_order.price).abs() < min_separation {
                    // Adjust to maintain separation
                    let adjustment_direction = if is_stop_loss {
                        // For stops, spread them further apart in the safe direction
                        if direction == Some("LONG") {
                            -1.0 // LONG stops go lower
                        } else {
                            1.0 // SHORT stops go higher
                        }
                    } else {
                        // For exits, spread them in the profitable direction
                        if direction == Some("LONG") {
                            1.0 // LONG exits go higher
                        } else {
                            -1.0 // SHORT exits go lower
                        }
                    };

                    final_price = other_order.price
                        + (min_separation * (i as f64 + 1.0) * adjustment_direction);

                    log::info!(
                        "🔧 Separation-aware adjustment: Order {} moved to ${:.5} to maintain separation from Order {}",
                        i + 1, final_price, j + 1
                    );
                    break;
                }
            }

            orders[i].price = final_price;

            // Update ATR distance
            orders[i].atr_distance = ((final_price - current_price).abs() / current_price) * 100.0;
        }
    }
}
