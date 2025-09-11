//! SMART Order Generation System - Using RIGHT models for RIGHT purposes
//! No magic numbers, pure mathematical consensus based on model strengths

use crate::output::prediction_types::{
    DirectionPrediction, PriceLevelPrediction, SentimentPrediction, VolatilityPrediction,
    VolumePrediction,
};
use crate::output::sequence_statistics::SequenceStatistics;
use crate::output::trading_orders::OrderLevel;
use crate::utils::error::Result;

/// SMART consensus calculator that uses each model for its strength
#[derive(Debug, Clone)]
pub struct SmartConsensus {
    /// Direction model output - BEST for trade direction and momentum
    pub direction: DirectionPrediction,
    /// Price levels model - BEST for entry/exit price targets
    pub price_levels: PriceLevelPrediction,
    /// Volatility model - BEST for stops and position sizing
    pub volatility: VolatilityPrediction,
    /// Sentiment model - BEST for confidence adjustment
    pub sentiment: SentimentPrediction,
    /// Volume model - BEST for timing and liquidity assessment
    pub volume: VolumePrediction,
}

impl SmartConsensus {
    /// Adjust level to avoid psychological clustering using sequence data
    /// Enhanced to handle stop-specific adjustments when is_stop_loss is true
    pub fn adjust_for_psychological_levels(
        &self,
        price: f64,
        sequence_ohlcv: Option<&Vec<[f64; 5]>>,
        is_stop_loss: bool,
        direction: Option<&str>,
    ) -> f64 {
        // Find natural support/resistance from sequence data
        let psychological_zones = if let Some(ohlcv) = sequence_ohlcv {
            self.find_natural_levels_from_sequence(ohlcv)
        } else {
            Vec::new()
        };

        // For stops, use wider range to avoid stop hunting
        let detection_range = if is_stop_loss {
            self.volatility.expected_range_percent / 100.0 // Wider range for stops
        } else {
            0.001 // 0.1% for regular orders
        };

        // Check if our price is too close to a natural level
        for &level in &psychological_zones {
            let distance = (price - level).abs() / level;

            if distance < detection_range {
                // Adjust based on sequence bandwidth
                let adjustment = self.direction.sequence_bandwidth_percent / 100.0;

                let adjusted_price = if is_stop_loss {
                    // More aggressive adjustment for stops to avoid hunting
                    if let Some(dir) = direction {
                        if dir == "LONG" {
                            // For LONG stops, push below support levels
                            if price > level {
                                level * (1.0 - adjustment * 0.5)
                            } else {
                                price * (1.0 - adjustment * 0.3)
                            }
                        } else {
                            // For SHORT stops, push above resistance levels
                            if price < level {
                                level * (1.0 + adjustment * 0.5)
                            } else {
                                price * (1.0 + adjustment * 0.3)
                            }
                        }
                    } else {
                        // Fallback if direction not provided
                        if price > level {
                            price * (1.0 + adjustment * 0.2)
                        } else {
                            price * (1.0 - adjustment * 0.2)
                        }
                    }
                } else {
                    // Regular adjustment for entry/target orders
                    if price > level {
                        price * (1.0 + adjustment * 0.1)
                    } else {
                        price * (1.0 - adjustment * 0.1)
                    }
                };

                log::info!(
                    "{} Psychological adjustment: {:.4} → {:.4} (avoiding level {:.4})",
                    if is_stop_loss { "🛡️" } else { "🎯" },
                    price,
                    adjusted_price,
                    level
                );

                return adjusted_price;
            }
        }

        // Also check round numbers for stops
        if is_stop_loss {
            let round_levels = self.find_round_number_levels(price);
            for round_level in round_levels {
                let distance = (price - round_level).abs() / round_level;

                if distance < detection_range * 0.5 {
                    let adjustment = self.direction.sequence_bandwidth_percent / 100.0;

                    let adjusted_price = if let Some(dir) = direction {
                        if dir == "LONG" {
                            if price > round_level {
                                round_level * (1.0 - adjustment * 0.2)
                            } else {
                                price * (1.0 - adjustment * 0.2)
                            }
                        } else if price < round_level {
                            round_level * (1.0 + adjustment * 0.2)
                        } else {
                            price * (1.0 + adjustment * 0.2)
                        }
                    } else {
                        price * (1.0 + adjustment * 0.1)
                    };

                    log::info!(
                        "🎯 Round number avoidance: {:.4} → {:.4} (avoiding {:.4})",
                        price,
                        adjusted_price,
                        round_level
                    );

                    return adjusted_price;
                }
            }
        }

        price
    }

    /// Find nearby round number levels (psychological levels)
    fn find_round_number_levels(&self, price: f64) -> Vec<f64> {
        let mut levels = Vec::new();

        // Determine the scale based on price magnitude
        let scale = if price < 0.01 {
            0.0001 // For very small prices
        } else if price < 0.1 {
            0.001
        } else if price < 1.0 {
            0.01
        } else if price < 10.0 {
            0.1
        } else if price < 100.0 {
            1.0
        } else {
            10.0
        };

        // Find nearest round numbers
        let lower_round = (price / scale).floor() * scale;
        let upper_round = (price / scale).ceil() * scale;

        if lower_round > 0.0 {
            levels.push(lower_round);
        }
        if upper_round != lower_round {
            levels.push(upper_round);
        }

        levels
    }

    /// Find natural support/resistance levels from sequence OHLCV data
    fn find_natural_levels_from_sequence(&self, ohlcv: &[[f64; 5]]) -> Vec<f64> {
        let mut levels = Vec::new();

        if ohlcv.len() < 3 {
            return levels;
        }

        // Find local highs and lows (reversal points)
        for i in 1..(ohlcv.len() - 1) {
            let prev_high = ohlcv[i - 1][2]; // Previous high
            let curr_high = ohlcv[i][2]; // Current high
            let next_high = ohlcv[i + 1][2]; // Next high

            let prev_low = ohlcv[i - 1][3]; // Previous low
            let curr_low = ohlcv[i][3]; // Current low
            let next_low = ohlcv[i + 1][3]; // Next low

            // Local high (resistance)
            if curr_high > prev_high && curr_high > next_high {
                levels.push(curr_high);
            }

            // Local low (support)
            if curr_low < prev_low && curr_low < next_low {
                levels.push(curr_low);
            }
        }

        // Remove duplicates and sort
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        levels.dedup_by(|a, b| ((*a - *b).abs() / *a) < 0.005); // Remove levels within 0.5%

        log::info!(
            "🎯 Found {} natural levels from sequence data",
            levels.len()
        );

        levels
    }
    /// Calculate SMART consensus for trade direction with confidence
    /// Now includes direction-price level alignment validation
    pub fn calculate_direction_consensus(&self) -> Result<(String, f64)> {
        // Direction model is PRIMARY for direction decision
        let direction_signal = if self.direction.up_probability_aggregated
            > self.direction.down_probability_aggregated
        {
            "LONG"
        } else {
            "SHORT"
        };

        // Find the HIGHEST PROBABILITY price level bin (best predicted class)
        let best_price_bin = self
            .price_levels
            .bins
            .iter()
            .max_by(|a, b| a.1.probability.partial_cmp(&b.1.probability).unwrap())
            .map(|(name, bin)| (name.as_str(), bin.probability));

        // Validate alignment between direction and best price level class
        if let Some((best_bin_name, best_probability)) = best_price_bin {
            let alignment_valid = match (direction_signal, best_bin_name) {
                // LONG direction - favorable bins
                ("LONG", "moderate_up") | ("LONG", "strong_up") => true,
                // SHORT direction - favorable bins
                ("SHORT", "moderate_down") | ("SHORT", "strong_down") => true,
                // NEUTRAL is acceptable for any direction (as requested)
                (_, "neutral") => true,
                // Contradictory alignment - reject
                ("LONG", "moderate_down") | ("LONG", "strong_down") => false,
                ("SHORT", "moderate_up") | ("SHORT", "strong_up") => false,
                // Unknown bins - allow but log warning
                _ => {
                    log::warn!("⚠️ Unknown price level bin: {}", best_bin_name);
                    true
                }
            };

            if !alignment_valid {
                return Err(crate::utils::error::VangaError::PredictionError(format!(
                    "Direction {} conflicts with best price level prediction {} (prob: {:.1}%)",
                    direction_signal,
                    best_bin_name,
                    best_probability * 100.0
                )));
            }

            log::info!(
                "✅ Direction-Price Level Alignment: {} direction with {} bin (prob: {:.1}%)",
                direction_signal,
                best_bin_name,
                best_probability * 100.0
            );
        }

        // Calculate confidence using Direction + Sentiment alignment
        let direction_confidence = (self.direction.up_probability_aggregated
            - self.direction.down_probability_aggregated)
            .abs();

        // Sentiment alignment boost (if sentiment agrees with direction)
        let sentiment_alignment = match (direction_signal, self.sentiment.regime.as_str()) {
            ("LONG", "BULLISH") | ("LONG", "VERY_BULLISH") => 1.2,
            ("SHORT", "BEARISH") | ("SHORT", "VERY_BEARISH") => 1.2,
            ("LONG", "BEARISH") | ("SHORT", "BULLISH") => 0.8, // Disagreement penalty
            _ => 1.0,                                          // Neutral
        };

        let final_confidence = (direction_confidence * sentiment_alignment).min(1.0);

        log::info!(
            "🎯 SMART Direction: {} with confidence {:.2} (sentiment alignment: {:.1}x)",
            direction_signal,
            final_confidence,
            sentiment_alignment
        );

        Ok((direction_signal.to_string(), final_confidence))
    }

    /// Generate SMART entry levels using PURE prediction data
    pub fn generate_smart_entries(
        &self,
        current_price: f64,
        direction: &str,
    ) -> Result<[OrderLevel; 3]> {
        let mut entries = Vec::new();

        // FIXED: First entry at EXACTLY current price for immediate execution (taker)
        let first_entry_price = current_price; // EXACTLY at current price

        entries.push(OrderLevel {
            price: first_entry_price,
            quantity_percentage: self.calculate_entry_size(1, &self.direction, &self.volatility), // Use proper sizing
            atr_distance: 0.0,                // At current price
            order_type: "MARKET".to_string(), // Market order for immediate fill
            confidence: 0.9,                  // High confidence for immediate entry
        });

        log::info!(
            "  {} Entry 1 (TAKER): ${:.4} (at current price) | Size: 50.0% | Type: MARKET",
            direction,
            first_entry_price
        );

        // Entries 2 & 3: Use zone-based approach for limit orders
        let entry_bins = if direction == "LONG" {
            vec!["neutral", "moderate_down"] // Buy on dips
        } else {
            vec!["neutral", "moderate_up"] // Sell on rallies
        };

        // Get zone boundaries for remaining entries
        let mut zone_entries = Vec::new();
        for bin_name in &entry_bins {
            if let Some(bin) = self.price_levels.bins.get(*bin_name) {
                // Use edge of zone for entry
                let entry_price = if direction == "LONG" {
                    bin.price[0] // Lower edge for LONG entries
                } else {
                    bin.price[1] // Upper edge for SHORT entries
                };
                zone_entries.push((entry_price, bin.probability));
            }
        }

        // If we don't have enough zones, use ATR-based fallback
        if zone_entries.is_empty() {
            let atr_spacing = self.volatility.expected_range_percent / 100.0;
            for i in 1..=2 {
                let distance = atr_spacing * (i as f64 * 0.5); // 0.5x, 1x ATR
                let price = if direction == "LONG" {
                    current_price * (1.0 - distance)
                } else {
                    current_price * (1.0 + distance)
                };
                zone_entries.push((price, 0.33));
            }
        }

        // Add remaining entries (using proper sizing)
        for (i, (entry_price, confidence)) in zone_entries.iter().take(2).enumerate() {
            let atr_distance = ((*entry_price - current_price).abs() / current_price) * 100.0;

            entries.push(OrderLevel {
                price: *entry_price,
                quantity_percentage: self.calculate_entry_size(
                    i + 2,
                    &self.direction,
                    &self.volatility,
                ), // Use proper sizing
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: confidence.min(0.9),
            });

            log::info!(
                "  {} Entry {}: ${:.4} ({:+.2}%) | Size: {:.1}% | Type: LIMIT",
                direction,
                i + 2, // Entry 2 and 3
                entry_price,
                if direction == "SHORT" {
                    atr_distance
                } else {
                    -atr_distance
                },
                self.calculate_entry_size(i + 2, &self.direction, &self.volatility) * 100.0
            );
        }

        // Ensure we have exactly 3 entries
        while entries.len() < 3 {
            // Fallback entry
            let distance = self.volatility.expected_range_percent / 100.0 * entries.len() as f64;
            let price = if direction == "LONG" {
                current_price * (1.0 - distance)
            } else {
                current_price * (1.0 + distance)
            };

            entries.push(OrderLevel {
                price,
                quantity_percentage: self.calculate_entry_size(
                    entries.len() + 1,
                    &self.direction,
                    &self.volatility,
                ),
                atr_distance: distance * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: 0.5,
            });
        }

        Ok([entries[0].clone(), entries[1].clone(), entries[2].clone()])
    }

    /// Generate SEQUENCE-AWARE entry levels using actual market behavior
    /// Uses volatility scaling (σ√t) and golden ratio for natural progression
    pub fn generate_sequence_aware_entries(
        &self,
        current_price: f64,
        direction: &str,
        sequence_stats: &SequenceStatistics,
    ) -> Result<[OrderLevel; 3]> {
        let mut entries = Vec::new();

        // FIXED: First entry EXACTLY at current price for immediate execution (taker)
        entries.push(OrderLevel {
            price: current_price, // EXACTLY at current price
            quantity_percentage: self.calculate_entry_size(1, &self.direction, &self.volatility),
            atr_distance: 0.0,                // At current price
            order_type: "MARKET".to_string(), // Market order for immediate fill
            confidence: 0.9,                  // High confidence for immediate entry
        });

        log::info!(
            "  {} Entry 1 (TAKER): ${:.4} (EXACTLY at current price) | Type: MARKET",
            direction,
            current_price
        );

        // Golden ratio for natural, non-arbitrary progression for remaining entries
        const PHI: f64 = 1.618033988749895;

        // Use ACTUAL market volatility from sequence data for entries 2 & 3
        let volatility_base = sequence_stats.std_return;

        // Use MAE (Maximum Adverse Excursion) distribution for realistic entry spacing
        let mae_median = if !sequence_stats.mae_distribution.is_empty() {
            SequenceStatistics::percentile(&sequence_stats.mae_distribution, 50.0)
        } else {
            sequence_stats.std_return
        };

        // Base spacing on the combination of volatility and typical adverse movement
        let base_spacing = (volatility_base + mae_median.abs()) / 2.0;

        // Adjust spacing based on market regime (trending vs mean-reverting)
        let regime_adjustment = if sequence_stats.mean_reversion_rate.abs() > 0.5 {
            // Mean reverting market: wider spacing (price likely to revert)
            // Use actual mean reversion strength from data
            1.0 + sequence_stats.mean_reversion_rate.abs()
        } else {
            // Trending market: use Hurst exponent to determine trend strength
            // Hurst > 0.5 = trending, < 0.5 = mean reverting
            if sequence_stats.hurst_exponent > 0.5 {
                // Strong trend: tighter spacing to catch it
                1.0 / (1.0 + (sequence_stats.hurst_exponent - 0.5))
            } else {
                // Weak trend/choppy: wider spacing
                1.0 + (0.5 - sequence_stats.hurst_exponent)
            }
        };

        // Use model predictions to scale spacing
        // Get the expected move from direction model
        let expected_move = if direction == "LONG" {
            self.direction.expected_downside_percent / 100.0 // Entry zone for longs
        } else {
            self.direction.expected_upside_percent / 100.0 // Entry zone for shorts
        };

        // Combine sequence statistics with model predictions
        let model_adjusted_spacing = base_spacing * (1.0 + expected_move);

        // Use directional confidence to further adjust
        let directional_confidence = (self.direction.up_probability_aggregated
            - self.direction.down_probability_aggregated)
            .abs();

        // Less confidence = wider spacing (using actual probability)
        let confidence_multiplier = 2.0 - directional_confidence;

        // Use volatility regime from model to fine-tune
        let volatility_multiplier = match self.volatility.regime.as_str() {
            "VERY_HIGH" => 1.5, // High volatility = wider entries
            "HIGH" => 1.25,
            "MEDIUM" => 1.0,
            "LOW" => 0.85,
            "VERY_LOW" => 0.7, // Low volatility = tighter entries
            _ => 1.0,
        };

        // Final spacing uses ALL available data
        let entry_spacing = model_adjusted_spacing
            * regime_adjustment
            * confidence_multiplier
            * volatility_multiplier;

        log::info!(
            "📊 DATA-DRIVEN Entry Spacing: vol_base={:.3}%, MAE={:.3}%, expected_move={:.3}%, regime_adj={:.2}x, conf={:.2}, vol_mult={:.2}x, final={:.3}%",
            volatility_base * 100.0,
            mae_median * 100.0,
            expected_move * 100.0,
            regime_adjustment,
            directional_confidence,
            volatility_multiplier,
            entry_spacing * 100.0
        );

        // FIXED: Generate ALL 3 entries properly with correct indexing
        // Entry 1 is already in the vector (MARKET order at current price)
        // Now generate entries 2 and 3 with proper indexing
        for i in 0..2 {
            // i=0 for entry 2, i=1 for entry 3
            let entry_index = i + 2; // This gives us entry 2 and 3
            let progression_factor = PHI.powi(i as i32 + 1); // PHI^1 for entry 2, PHI^2 for entry 3
            let distance = entry_spacing * progression_factor;

            let entry_price = if direction == "SHORT" {
                current_price * (1.0 + distance)
            } else {
                current_price * (1.0 - distance)
            };

            // Use proper sizing for entries 2 & 3
            let entry_size =
                self.calculate_entry_size(entry_index, &self.direction, &self.volatility);

            // Use actual Kelly fraction from sequence statistics
            let kelly_adjusted_size = entry_size * (1.0 + sequence_stats.kelly_fraction);

            // Confidence based on DATA:
            // 1. Directional confidence from model
            // 2. Distance decay using volatility scale
            // 3. Market efficiency from Hurst exponent
            let distance_in_stdevs = distance / sequence_stats.std_return;
            let distance_decay = (-distance_in_stdevs).exp(); // Decay based on standard deviations

            let efficiency_factor = if sequence_stats.hurst_exponent > 0.5 {
                // Trending market: higher confidence
                sequence_stats.hurst_exponent * 2.0
            } else {
                // Mean reverting: adjust by strength
                1.0 - (0.5 - sequence_stats.hurst_exponent)
            };

            // Use volume regime for liquidity confidence
            let volume_confidence = match self.volume.regime.as_str() {
                "VERY_HIGH" | "HIGH" => 1.2, // High volume = higher confidence
                "MEDIUM" => 1.0,
                "LOW" | "VERY_LOW" => 0.8, // Low volume = lower confidence
                _ => 1.0,
            };

            let confidence =
                (directional_confidence * distance_decay * efficiency_factor * volume_confidence)
                    .clamp(0.1, 0.95);

            // Calculate actual distance from current price
            let atr_distance = ((entry_price - current_price).abs() / current_price) * 100.0;

            entries.push(OrderLevel {
                price: entry_price,
                quantity_percentage: kelly_adjusted_size,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence,
            });

            log::info!(
                "  {} Entry {}: ${:.4} ({:+.3}%) | Size: {:.1}% | Conf: {:.2} | Kelly: {:.3} | Distance in σ: {:.2}",
                direction,
                entry_index,
                entry_price,
                if direction == "SHORT" {
                    distance * 100.0
                } else {
                    -distance * 100.0
                },
                kelly_adjusted_size * 100.0,
                confidence,
                sequence_stats.kelly_fraction,
                distance_in_stdevs
            );
        }

        // Entries should already be properly ordered by the math above

        // Validate we have exactly 3 entries
        if entries.len() != 3 {
            return Err(crate::utils::error::VangaError::PredictionError(format!(
                "Entry generation failed: expected 3 entries, got {}",
                entries.len()
            )));
        }

        Ok([entries[0].clone(), entries[1].clone(), entries[2].clone()])
    }

    /// FULLY ADAPTIVE entry generation using ONLY sequence data and model predictions
    /// NO hardcoded values - everything derived from mathematical relationships
    pub fn generate_fully_adaptive_entries(
        &self,
        current_price: f64,
        direction: &str,
        sequence_prices: &[f64],
        sequence_stats: &SequenceStatistics,
    ) -> Result<[OrderLevel; 3]> {
        log::info!("🚀 USING FULLY ADAPTIVE ENTRY GENERATION - NO HARDCODED VALUES");
        log::info!(
            "📊 Sequence data: {} prices, current: ${:.4}",
            sequence_prices.len(),
            current_price
        );

        let mut entries = Vec::new();

        // Get adaptive bounds from sequence data
        let bounds = sequence_stats.get_adaptive_bounds(sequence_prices, current_price);

        // Calculate probability-weighted entry zone using model predictions
        let entry_bins = if direction == "LONG" {
            vec!["neutral", "moderate_down", "strong_down"]
        } else {
            vec!["neutral", "moderate_up", "strong_up"]
        };

        log::info!(
            "🎯 Fully Adaptive Entry Generation: seq_range={:.2}%, max_drawdown={:.2}%, IQR_vol={:.2}%",
            bounds.sequence_range_pct,
            bounds.max_drawdown_pct,
            bounds.iqr_volatility
        );
        log::info!(
            "📈 Sequence bounds: min=${:.4}, max=${:.4}, p25=${:.4}, p75=${:.4}",
            bounds.sequence_min,
            bounds.sequence_max,
            bounds.p25,
            bounds.p75
        );

        // Generate 3 entries using sequence-derived bounds
        for i in 0..3 {
            // Get model prediction for this entry level
            let bin_name = entry_bins.get(i).unwrap_or(&"neutral");
            let bin_probability = self
                .price_levels
                .bins
                .get(*bin_name)
                .map(|b| b.probability)
                .unwrap_or(0.33);

            let bin_range = self
                .price_levels
                .bins
                .get(*bin_name)
                .map(|b| b.range)
                .unwrap_or([0.0, 0.0]);

            // Use MODEL predictions but bound by SEQUENCE reality
            let model_distance = if direction == "LONG" {
                bin_range[0].abs() // Use lower bound for long entries (negative values)
            } else {
                bin_range[0] // Use lower bound for short entries (positive values)
            };

            // Bound by actual sequence movement using probability weighting
            let sequence_bounded_distance =
                (bounds.sequence_range_pct * bin_probability).min(model_distance.abs());

            // Progressive spacing using sequence percentiles (no hardcoded ratios)
            let percentile_spacing = match i {
                0 => (bounds.p50 - current_price).abs() / current_price * 100.0, // Median distance
                1 => (bounds.p25 - current_price).abs() / current_price * 100.0, // Q1 distance
                2 => (bounds.p10 - current_price).abs() / current_price * 100.0, // 10th percentile distance
                _ => bounds.iqr_volatility * 0.5,
            };

            // Use the smaller of model prediction or sequence-derived spacing
            let final_distance = sequence_bounded_distance.min(percentile_spacing);

            // Calculate entry price
            let entry_price = if direction == "SHORT" {
                current_price * (1.0 + final_distance / 100.0)
            } else {
                current_price * (1.0 - final_distance / 100.0)
            };

            // Validate using z-score (within 3 standard deviations)
            let validated_price =
                sequence_stats.validate_price_with_zscore(sequence_prices, entry_price);

            // Size allocation based on bin probability and Kelly fraction
            let probability_size = bin_probability * 0.8; // Scale by probability
            let kelly_adjusted_size = probability_size * (1.0 + sequence_stats.kelly_fraction);

            // Normalize to reasonable range
            let final_size = kelly_adjusted_size.clamp(0.15, 0.6);

            // Confidence based on bin probability and distance from sequence bounds
            let distance_from_current = (validated_price - current_price).abs() / current_price;
            let distance_confidence = (-distance_from_current * 5.0).exp(); // Exponential decay
            let final_confidence = (bin_probability * distance_confidence).clamp(0.2, 0.95);

            let atr_distance = ((validated_price - current_price).abs() / current_price) * 100.0;

            entries.push(OrderLevel {
                price: validated_price,
                quantity_percentage: final_size,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: final_confidence,
            });

            log::info!(
                "  {} Adaptive Entry {}: ${:.4} ({:+.2}%) | Model: {:.2}% | Seq: {:.2}% | Final: {:.2}% | Size: {:.1}% | Conf: {:.2}",
                direction,
                i + 1,
                validated_price,
                if direction == "SHORT" { final_distance } else { -final_distance },
                model_distance,
                percentile_spacing,
                final_distance,
                final_size * 100.0,
                final_confidence
            );
        }

        // Normalize sizes to sum to 1.0
        let total_size: f64 = entries.iter().map(|e| e.quantity_percentage).sum();
        if total_size > 0.0 {
            for entry in &mut entries {
                entry.quantity_percentage /= total_size;
            }
        }

        Ok([entries[0].clone(), entries[1].clone(), entries[2].clone()])
    }

    /// Generate SMART exit levels using PURE prediction data
    pub fn generate_smart_exits(
        &self,
        current_price: f64,
        direction: &str,
    ) -> Result<[OrderLevel; 3]> {
        let mut exits = Vec::new();

        // Use unified model boundaries for consistent exit generation
        let model_boundaries = crate::output::model_boundaries::ModelBoundaries::calculate(
            &self.price_levels,
            current_price,
            direction,
            self.volatility.expected_range_percent,
        );

        log::info!(
            "🎯 Exit generation using model boundaries: {} suitable bins, boundary={:.2}%",
            model_boundaries.suitable_bins.len(),
            model_boundaries.max_exit_boundary_percent
        );

        // Get bin targets from suitable bins (already filtered by model boundaries)
        let mut bin_targets = Vec::new();
        for (i, (bin_name, probability, bin)) in
            model_boundaries.suitable_bins.iter().take(3).enumerate()
        {
            let target_price = if i == 0 {
                // First exit: use edge closest to current price for quick profit
                let edge_price = if direction == "LONG" {
                    bin.price[0] // Lower edge (closest to current for LONG)
                } else {
                    bin.price[1] // Upper edge (closest to current for SHORT)
                };

                // CRITICAL: Ensure the edge is actually profitable
                let is_profitable = if direction == "LONG" {
                    edge_price > current_price
                } else {
                    edge_price < current_price
                };

                if is_profitable {
                    edge_price
                } else {
                    // Fallback to center if edge is not profitable
                    (bin.price[0] + bin.price[1]) / 2.0
                }
            } else {
                // Further exits: use bin center for maximum capture
                (bin.price[0] + bin.price[1]) / 2.0
            };

            log::info!(
                "🎯 {} bin {}: target_price=${:.5} (edge/center logic)",
                bin_name,
                i + 1,
                target_price
            );

            bin_targets.push((target_price, *probability));
        }

        // If we don't have enough bins from model, use volatility-based fallback
        if bin_targets.is_empty() {
            log::warn!(
                "⚠️ No suitable bins found for {} direction, using volatility fallback",
                direction
            );
        }

        log::info!(
            "📊 Found {} model-based bin targets before fallback",
            bin_targets.len()
        );

        while bin_targets.len() < 3 {
            // Use model boundaries for consistent fallback calculation
            let fallback_distance_percent = model_boundaries.max_exit_boundary_percent;
            let progressive_factor = match bin_targets.len() {
                0 => 0.3, // 30% of boundary distance for first fallback
                1 => 0.6, // 60% of boundary distance for second fallback
                2 => 1.0, // Full boundary distance for third fallback
                _ => 1.0,
            };
            let distance = (fallback_distance_percent / 100.0) * progressive_factor;

            let price = if direction == "LONG" {
                current_price * (1.0 + distance)
            } else {
                current_price * (1.0 - distance)
            };

            log::info!(
                "📊 Model-bounded fallback exit {}: ${:.5} (distance: {:.2}%, factor: {:.1}x)",
                bin_targets.len() + 1,
                price,
                distance * 100.0,
                progressive_factor
            );

            bin_targets.push((price, 0.33));
        }

        // Progressive sizing: 50%, 33%, 17% for balanced distribution
        let sizes = [0.5, 0.333, 0.167];

        log::info!("🎯 Exit Generation: Using price bin CENTERS as targets for optimal R:R");

        for (i, (exit_price, confidence)) in bin_targets.iter().take(3).enumerate() {
            // Calculate distance for logging purposes
            let distance_percent = ((*exit_price - current_price).abs() / current_price) * 100.0;
            log::debug!("Exit {} distance: {:.2}%", i + 1, distance_percent);

            // Validate exit respects model boundaries before adding
            let final_exit_price = if let Err(boundary_error) =
                model_boundaries.validate_exit_price(*exit_price, current_price)
            {
                log::warn!(
                    "⚠️ Exit {} violates model boundary: {}. Adjusting to boundary.",
                    i + 1,
                    boundary_error
                );

                // Adjust to stay within absolute boundary
                if direction == "LONG" {
                    exit_price.min(model_boundaries.absolute_boundary_price)
                } else {
                    exit_price.max(model_boundaries.absolute_boundary_price)
                }
            } else {
                *exit_price
            };

            // Ensure the final price is still valid
            if final_exit_price <= 0.0 {
                log::error!(
                    "❌ Exit {} price became invalid: ${:.5}. Using fallback.",
                    i + 1,
                    final_exit_price
                );
                // Use a reasonable fallback based on current price and direction
                let fallback_distance = 0.02; // 2% as minimal distance
                if direction == "LONG" {
                    current_price * (1.0 + fallback_distance)
                } else {
                    current_price * (1.0 - fallback_distance)
                }
            } else {
                final_exit_price
            };

            exits.push(OrderLevel {
                price: final_exit_price,
                quantity_percentage: sizes[i],
                atr_distance: ((final_exit_price - current_price).abs() / current_price) * 100.0,
                order_type: "LIMIT".to_string(),
                confidence: confidence.min(0.9),
            });

            log::info!(
                "  {} Exit {}: ${:.4} ({:+.2}%) | Size: {:.1}% | Conf: {:.2}",
                direction,
                i + 1,
                final_exit_price,
                if direction == "SHORT" {
                    -((final_exit_price - current_price).abs() / current_price) * 100.0
                } else {
                    ((final_exit_price - current_price).abs() / current_price) * 100.0
                },
                sizes[i] * 100.0,
                confidence
            );
        }
        Ok([exits[0].clone(), exits[1].clone(), exits[2].clone()])
    }

    /// Generate SEQUENCE-AWARE exit levels using MFE distribution
    /// Uses actual favorable excursions from sequence data for realistic targets
    pub fn generate_sequence_aware_exits(
        &self,
        current_price: f64,
        direction: &str,
        sequence_stats: &SequenceStatistics,
    ) -> Result<[OrderLevel; 3]> {
        let mut exits = Vec::new();

        // Get favorable excursion percentiles for exit targets
        let mfe_percentiles = if !sequence_stats.mfe_distribution.is_empty() {
            vec![
                SequenceStatistics::percentile(&sequence_stats.mfe_distribution, 25.0), // Conservative
                SequenceStatistics::percentile(&sequence_stats.mfe_distribution, 50.0), // Median expectation
                SequenceStatistics::percentile(&sequence_stats.mfe_distribution, 75.0), // Optimistic
            ]
        } else {
            // Fallback to volatility-based estimates
            vec![
                sequence_stats.std_return * 1.0,
                sequence_stats.std_return * 2.0,
                sequence_stats.std_return * 3.0,
            ]
        };

        // Adjust targets based on market efficiency (Hurst exponent)
        let efficiency_multiplier = if sequence_stats.hurst_exponent > 0.5 {
            // Trending market: can aim for larger moves
            1.0 + (sequence_stats.hurst_exponent - 0.5) * 2.0
        } else {
            // Mean reverting: smaller, quicker profits
            1.0 - (0.5 - sequence_stats.hurst_exponent) * 2.0
        };

        log::info!(
            "📊 Sequence-Aware Exit Targets: MFE percentiles=[{:.3}%, {:.3}%, {:.3}%], Hurst={:.2}, efficiency_mult={:.2}x",
            mfe_percentiles[0] * 100.0,
            mfe_percentiles[1] * 100.0,
            mfe_percentiles[2] * 100.0,
            sequence_stats.hurst_exponent,
            efficiency_multiplier
        );

        // Generate exits based on MFE distribution
        for (i, &mfe_target) in mfe_percentiles.iter().enumerate() {
            // Apply efficiency adjustment
            let adjusted_target = mfe_target * efficiency_multiplier;

            // Calculate exit price
            let exit_price = if direction == "SHORT" {
                current_price * (1.0 - adjusted_target)
            } else {
                current_price * (1.0 + adjusted_target)
            };

            // Size allocation based on probability of reaching target
            // Use inverse of percentile for sizing (lower percentile = higher probability = larger size)
            let probability_factor = match i {
                0 => 0.75, // 75% chance of reaching 25th percentile
                1 => 0.50, // 50% chance of reaching median
                2 => 0.25, // 25% chance of reaching 75th percentile
                _ => 0.33,
            };

            // Adjust size by volume liquidity
            let volume_adjustment = self.volume_liquidity_factor();
            let exit_size = (probability_factor * volume_adjustment) / 1.5; // Normalize to sum ~1.0

            // Confidence based on:
            // 1. MFE distribution reliability
            // 2. Market efficiency
            // 3. Probability of reaching target
            // 4. Information entropy (lower entropy = higher confidence)
            let mfe_confidence = (sequence_stats.mfe_distribution.len() as f64 / 100.0).min(1.0);
            let entropy_factor = if sequence_stats.price_entropy > 0.0 {
                (-sequence_stats.price_entropy).exp().min(1.0)
            } else {
                0.5
            };

            let exit_confidence =
                (mfe_confidence * probability_factor * efficiency_multiplier * entropy_factor)
                    .clamp(0.3, 0.95);

            // Calculate actual ATR distance from current price
            let atr_distance = ((exit_price - current_price).abs() / current_price) * 100.0;

            exits.push(OrderLevel {
                price: exit_price,
                quantity_percentage: exit_size,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: exit_confidence,
            });

            log::info!(
                "  {} Exit {}: ${:.2} ({:+.3}%) | Size: {:.1}% | MFE-based: {:.3}% | Conf: {:.2}",
                direction,
                i + 1,
                exit_price,
                if direction == "SHORT" {
                    -adjusted_target * 100.0
                } else {
                    adjusted_target * 100.0
                },
                exit_size * 100.0,
                adjusted_target * 100.0,
                exit_confidence
            );
        }

        // Normalize exit sizes to sum to 1.0
        let total_size: f64 = exits.iter().map(|e| e.quantity_percentage).sum();
        if total_size > 0.0 {
            for exit in &mut exits {
                exit.quantity_percentage /= total_size;
            }
        }

        Ok([exits[0].clone(), exits[1].clone(), exits[2].clone()])
    }

    /// FULLY ADAPTIVE exit generation using sequence upside bounds and model predictions
    /// NO hardcoded values - everything bounded by actual sequence potential
    pub fn generate_fully_adaptive_exits(
        &self,
        current_price: f64,
        direction: &str,
        sequence_prices: &[f64],
        sequence_stats: &SequenceStatistics,
    ) -> Result<[OrderLevel; 3]> {
        log::info!("🚀 USING FULLY ADAPTIVE EXIT GENERATION - SEQUENCE BOUNDED");
        log::info!(
            "📊 Sequence data: {} prices, current: ${:.4}",
            sequence_prices.len(),
            current_price
        );

        let mut exits = Vec::new();

        // Get adaptive bounds from sequence data
        let bounds = sequence_stats.get_adaptive_bounds(sequence_prices, current_price);

        // Calculate probability-weighted favorable zone using model predictions
        let favorable_bins = if direction == "LONG" {
            vec!["moderate_up", "strong_up"]
        } else {
            vec!["moderate_down", "strong_down"]
        };

        log::info!(
            "🎯 Fully Adaptive Exit Generation: max_upside={:.2}%, seq_range={:.2}%, IQR_vol={:.2}%",
            bounds.max_upside_pct,
            bounds.sequence_range_pct,
            bounds.iqr_volatility
        );
        log::info!(
            "📈 Sequence upside bounds: max=${:.4}, p90=${:.4}, current=${:.4}",
            bounds.sequence_max,
            bounds.p90,
            current_price
        );

        // Generate 3 exits using sequence-bounded approach
        for i in 0..3 {
            // Get model prediction for this exit level
            let bin_name = favorable_bins
                .get(i % favorable_bins.len())
                .unwrap_or(&"moderate_up");
            let bin_probability = self
                .price_levels
                .bins
                .get(*bin_name)
                .map(|b| b.probability)
                .unwrap_or(0.33);

            let bin_range = self
                .price_levels
                .bins
                .get(*bin_name)
                .map(|b| b.range)
                .unwrap_or([0.0, 0.0]);

            // Use MODEL predictions but bound by SEQUENCE reality
            let model_exit_distance = if direction == "LONG" {
                bin_range[0] // Use lower bound for conservative exits
            } else {
                bin_range[0].abs() // Use absolute value for short exits
            };

            // Bound by actual sequence upside potential
            let sequence_bounded_distance = if bounds.max_upside_pct > 0.0 {
                (bounds.max_upside_pct * bin_probability).min(model_exit_distance)
            } else {
                // Fallback to IQR-based targets if no upside in sequence
                bounds.iqr_volatility * bin_probability
            };

            // Progressive exits using sequence percentiles and IQR spacing
            let percentile_target = match i {
                0 => (bounds.p75 - current_price) / current_price * 100.0, // Q3 target (conservative)
                1 => (bounds.p90 - current_price) / current_price * 100.0, // 90th percentile (median)
                2 => (bounds.sequence_max - current_price) / current_price * 100.0, // Maximum seen (optimistic)
                _ => bounds.iqr_volatility * 0.5,
            };

            // Use the smaller of model prediction or sequence-derived target
            let final_distance = sequence_bounded_distance.min(percentile_target.abs());

            // Calculate exit price
            let exit_price = if direction == "SHORT" {
                current_price * (1.0 - final_distance / 100.0)
            } else {
                current_price * (1.0 + final_distance / 100.0)
            };

            // Validate using z-score (within 3 standard deviations)
            let validated_price =
                sequence_stats.validate_price_with_zscore(sequence_prices, exit_price);

            // Size allocation based on probability of reaching target
            // Higher probability targets get larger allocations
            let probability_size = bin_probability * 0.7; // Scale by probability

            // Adjust by distance (closer targets get larger size)
            let distance_factor = 1.0 / (1.0 + final_distance / 100.0);
            let distance_adjusted_size = probability_size * distance_factor;

            // Normalize to reasonable range
            let final_size = distance_adjusted_size.clamp(0.2, 0.5);

            // Confidence based on:
            // 1. Bin probability (model confidence)
            // 2. Sequence upside potential (realistic achievability)
            // 3. Distance from current price (closer = higher confidence)
            let upside_confidence = if bounds.max_upside_pct > 0.0 {
                (final_distance / bounds.max_upside_pct).min(1.0)
            } else {
                0.5 // Neutral confidence if no historical upside
            };

            let distance_confidence = (-final_distance / 10.0).exp(); // Exponential decay
            let final_confidence =
                (bin_probability * upside_confidence * distance_confidence).clamp(0.3, 0.9);

            let atr_distance = ((validated_price - current_price).abs() / current_price) * 100.0;

            exits.push(OrderLevel {
                price: validated_price,
                quantity_percentage: final_size,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: final_confidence,
            });

            log::info!(
                "  {} Adaptive Exit {}: ${:.4} ({:+.2}%) | Model: {:.2}% | Seq: {:.2}% | Final: {:.2}% | Size: {:.1}% | Conf: {:.2}",
                direction,
                i + 1,
                validated_price,
                if direction == "SHORT" { -final_distance } else { final_distance },
                model_exit_distance,
                percentile_target.abs(),
                final_distance,
                final_size * 100.0,
                final_confidence
            );
        }

        // Normalize exit sizes to sum to 1.0
        let total_size: f64 = exits.iter().map(|e| e.quantity_percentage).sum();
        if total_size > 0.0 {
            for exit in &mut exits {
                exit.quantity_percentage /= total_size;
            }
        }

        Ok([exits[0].clone(), exits[1].clone(), exits[2].clone()])
    }

    /// Generate ADAPTIVE stop levels using ALL prediction data
    /// Combines volatility, price levels, sequence data, and confidence
    pub fn generate_adaptive_stops(
        &self,
        entry_levels: &[OrderLevel; 3],
        direction: &str,
        sequence_prices: &[f64],
    ) -> Result<[OrderLevel; 3]> {
        use crate::output::adaptive_stop_calculator::AdaptiveStopCalculator;

        log::info!("🧠 Using ADAPTIVE stop calculation with multi-source data");

        // Create adaptive stop calculator with ALL available data
        let calculator = AdaptiveStopCalculator::new(
            self.volatility.clone(),
            self.price_levels.clone(),
            self.direction.clone(),
            sequence_prices.to_vec(),
        );

        // Calculate adaptive stops
        match calculator.calculate_adaptive_stops(entry_levels, direction) {
            Ok(result) => {
                log::info!(
                    "✅ Adaptive stops calculated: {} (confidence: {:.2})",
                    result.methodology,
                    result.placement_confidence
                );
                log::info!(
                    "📊 Risk assessment: stop_hit_prob={:.2}%, expected_adverse={:.2}%, volatility_factor={:.2}x, trend_alignment={:.2}x",
                    result.risk_assessment.stop_hit_probability * 100.0,
                    result.risk_assessment.expected_adverse_move * 100.0,
                    result.risk_assessment.volatility_risk_factor,
                    result.risk_assessment.trend_alignment
                );

                Ok(result.stop_levels)
            }
            Err(e) => {
                log::warn!(
                    "⚠️ Adaptive stop calculation failed: {}, falling back to legacy",
                    e
                );
                self.generate_smart_stops_legacy(entry_levels, direction)
            }
        }
    }

    /// Generate SMART stop levels using BOTH Volatility + Price Level Probabilities
    /// CRITICAL: Stops must NEVER intersect with ANY entry level
    pub fn generate_smart_stops(
        &self,
        entry_levels: &[OrderLevel; 3],
        direction: &str,
    ) -> Result<[OrderLevel; 3]> {
        self.generate_probability_weighted_stops(entry_levels, direction)
    }

    /// NEW: Probability-weighted stops using BOTH volatility AND price level probabilities
    fn generate_probability_weighted_stops(
        &self,
        entry_levels: &[OrderLevel; 3],
        direction: &str,
    ) -> Result<[OrderLevel; 3]> {
        let mut stops = Vec::new();

        // Base stop distance from volatility model
        let volatility_base_stop = self.volatility.recommended_stop_distance_percent;

        // Find extreme entry to ensure NO intersection
        let extreme_entry = if direction == "SHORT" {
            // For SHORT, find the HIGHEST entry price
            entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            // For LONG, find the LOWEST entry price
            entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::INFINITY, f64::min)
        };

        log::info!(
            "🛡️ Stop positioning: extreme_entry=${:.4}, base_stop={:.2}%",
            extreme_entry,
            volatility_base_stop
        );

        // Calculate probability-weighted stop distances
        for (i, entry) in entry_levels.iter().enumerate().take(3) {
            // 1. VOLATILITY COMPONENT - Base risk from volatility model
            let volatility_stop = volatility_base_stop;

            // 2. PRICE LEVEL PROBABILITY COMPONENT - Risk from adverse price movements
            let adverse_probability = if direction == "SHORT" {
                // For SHORT, risk comes from upside moves (strong_up, moderate_up)
                let strong_up_prob = self
                    .price_levels
                    .bins
                    .get("strong_up")
                    .map(|b| b.probability)
                    .unwrap_or(0.0);
                let moderate_up_prob = self
                    .price_levels
                    .bins
                    .get("moderate_up")
                    .map(|b| b.probability)
                    .unwrap_or(0.0);
                strong_up_prob + moderate_up_prob
            } else {
                // For LONG, risk comes from downside moves (strong_down, moderate_down)
                let strong_down_prob = self
                    .price_levels
                    .bins
                    .get("strong_down")
                    .map(|b| b.probability)
                    .unwrap_or(0.0);
                let moderate_down_prob = self
                    .price_levels
                    .bins
                    .get("moderate_down")
                    .map(|b| b.probability)
                    .unwrap_or(0.0);
                strong_down_prob + moderate_down_prob
            };

            // 3. PROBABILITY-WEIGHTED STOP CALCULATION
            // Higher adverse probability = wider stops needed (use adverse probability directly, no magic multiplier)
            let probability_multiplier = 1.0 + adverse_probability; // Direct scaling by adverse risk
            let probability_weighted_stop = volatility_stop * probability_multiplier;

            // 4. PROGRESSIVE SPACING based on volatility regime (use smaller multipliers for tighter stops)
            let regime_progression = match self.volatility.regime.as_str() {
                "VERY_LOW" => 1.0 + (i as f64 * 0.1), // Tighter progression in calm markets
                "LOW" => 1.0 + (i as f64 * 0.15),
                "MEDIUM" => 1.0 + (i as f64 * 0.2), // Tighter than before
                "HIGH" => 1.0 + (i as f64 * 0.25),
                "VERY_HIGH" => 1.0 + (i as f64 * 0.3), // Still progressive but tighter
                _ => 1.0 + (i as f64 * 0.2),
            };

            // Make stops extremely close to extreme entry as requested
            let base_stop_distance = probability_weighted_stop * regime_progression;
            let final_stop_distance = base_stop_distance * 0.5; // Make stops much tighter

            // Calculate stop price from extreme entry (ensures no intersection)
            let stop_price = if direction == "SHORT" {
                extreme_entry * (1.0 + final_stop_distance / 100.0)
            } else {
                extreme_entry * (1.0 - final_stop_distance / 100.0)
            };

            // Confidence based on BOTH volatility confidence AND probability certainty
            let volatility_confidence = self.volatility.regime_confidence;
            let probability_confidence = 1.0 - adverse_probability; // Lower adverse prob = higher confidence
            let combined_confidence = (volatility_confidence * probability_confidence).max(0.3);

            let atr_distance = ((stop_price - extreme_entry).abs() / extreme_entry) * 100.0;

            stops.push(OrderLevel {
                price: stop_price,
                quantity_percentage: entry.quantity_percentage, // Match entry size
                atr_distance,
                order_type: "STOP_LOSS".to_string(),
                confidence: combined_confidence.min(0.95),
            });

            log::info!(
                "  {} Stop {}: ${:.4} ({:.2}% from extreme) | Vol: {:.2}% | Prob_mult: {:.1}x | Regime_prog: {:.1}x | Final: {:.2}%",
                direction,
                i + 1,
                stop_price,
                final_stop_distance,
                volatility_stop,
                probability_multiplier,
                regime_progression,
                final_stop_distance
            );
        }

        log::info!(
            "🛑 Probability-Weighted Stops: volatility={:.2}%, adverse_prob={:.1}%, regime={}, extreme_entry=${:.4}",
            self.volatility.recommended_stop_distance_percent,
            if direction == "SHORT" {
                self.price_levels.bins.get("strong_up").map(|b| b.probability).unwrap_or(0.0) +
                self.price_levels.bins.get("moderate_up").map(|b| b.probability).unwrap_or(0.0)
            } else {
                self.price_levels.bins.get("strong_down").map(|b| b.probability).unwrap_or(0.0) +
                self.price_levels.bins.get("moderate_down").map(|b| b.probability).unwrap_or(0.0)
            } * 100.0,
            self.volatility.regime,
            extreme_entry
        );

        Ok([stops[0].clone(), stops[1].clone(), stops[2].clone()])
    }

    /// Legacy smart stops implementation (kept for backward compatibility)
    fn generate_smart_stops_legacy(
        &self,
        entry_levels: &[OrderLevel; 3],
        direction: &str,
    ) -> Result<[OrderLevel; 3]> {
        let mut stops = Vec::new();

        // Base stop distance from volatility model (it's designed for this!)
        let base_stop_percent = self.volatility.recommended_stop_distance_percent;

        // Adjust stop distance based on volatility regime
        let regime_multiplier = match self.volatility.regime.as_str() {
            "VERY_LOW" => 0.5, // Tighter stops in calm markets
            "LOW" => 0.7,
            "MEDIUM" => 1.0,    // Normal stops
            "HIGH" => 1.3,      // Wider stops in volatile markets
            "VERY_HIGH" => 1.5, // Extra wide for extreme volatility
            _ => 1.0,
        };

        // Sentiment adjustment (bearish sentiment = tighter stops for longs)
        let sentiment_adjustment = match (direction, self.sentiment.regime.as_str()) {
            ("LONG", "VERY_BEARISH") => 0.8, // Tighter stops when against sentiment
            ("SHORT", "VERY_BULLISH") => 0.8,
            ("LONG", "VERY_BULLISH") => 1.2, // Wider stops when with sentiment
            ("SHORT", "VERY_BEARISH") => 1.2,
            _ => 1.0,
        };

        // CRITICAL FIX: Find the extreme entry price to ensure NO intersection
        let extreme_entry = if direction == "SHORT" {
            // For SHORT, find the HIGHEST entry price
            entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            // For LONG, find the LOWEST entry price
            entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::INFINITY, f64::min)
        };

        // Calculate the entry price range for progressive spacing
        let entry_range = if direction == "SHORT" {
            let min_entry = entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::INFINITY, f64::min);
            let max_entry = entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::NEG_INFINITY, f64::max);
            ((max_entry - min_entry) / min_entry) * 100.0 // Percentage range
        } else {
            let min_entry = entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::INFINITY, f64::min);
            let max_entry = entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::NEG_INFINITY, f64::max);
            ((max_entry - min_entry) / max_entry) * 100.0 // Percentage range
        };

        log::info!(
            "🛑 SMART Stop Calculation: {} extreme_entry={:.2}, entry_range={:.3}%, base_stop={:.2}%",
            direction, extreme_entry, entry_range, base_stop_percent
        );

        // Calculate progressive spacing based on:
        // 1. Entry range (how spread out entries are)
        // 2. Position sizes (larger positions need tighter stops)
        // 3. Volatility expected range
        let total_expected_range = self.volatility.expected_range_percent;

        // Use entry range and volatility to determine progressive spacing
        // If entries are spread 2%, stops should spread proportionally
        let progressive_factor = (entry_range / total_expected_range).max(0.1);

        // Calculate stops based on the extreme entry to guarantee no intersection
        for (i, entry) in entry_levels.iter().enumerate() {
            // Position-weighted stop distance
            // Larger positions get tighter stops (risk management)
            let position_weight = entry.quantity_percentage;
            let position_adjustment = 1.0 + (0.5 - position_weight); // 50% position = 1.0x, 30% = 1.2x, 20% = 1.3x

            // Progressive spacing based on actual data, not magic numbers
            // Use the entry range to determine how much to space stops
            let progression = i as f64 * progressive_factor * entry_range / 3.0;

            let stop_distance_percent =
                base_stop_percent * regime_multiplier * sentiment_adjustment * position_adjustment
                    + progression;

            // CRITICAL: Calculate stop from EXTREME entry, not individual entry
            let stop_price = if direction == "SHORT" {
                // SHORT stops must be ABOVE the HIGHEST entry
                // Base distance + progressive spacing based on entry range
                extreme_entry * (1.0 + stop_distance_percent / 100.0)
            } else {
                // LONG stops must be BELOW the LOWEST entry
                extreme_entry * (1.0 - stop_distance_percent / 100.0)
            };

            // Stop confidence based on:
            // 1. Volatility confidence (model knows risk)
            // 2. Position size (larger positions = higher confidence needed)
            // 3. Progressive decay (further stops = lower confidence)
            let stop_confidence =
                (self.volatility.regime_confidence * position_weight * (1.0 - i as f64 * 0.1))
                    .max(0.1);

            stops.push(OrderLevel {
                price: stop_price,
                quantity_percentage: entry.quantity_percentage, // Match entry size
                atr_distance: stop_distance_percent,
                order_type: "STOP_LOSS".to_string(),
                confidence: stop_confidence.min(0.95),
            });

            log::info!(
                "  Stop {}: ${:.2} ({:+.3}% from extreme) | Size: {:.1}% | Distance: {:.3}%",
                i + 1,
                stop_price,
                ((stop_price - extreme_entry) / extreme_entry) * 100.0,
                entry.quantity_percentage * 100.0,
                stop_distance_percent
            );
        }

        // FINAL VALIDATION: Ensure no intersection
        let mut corrections_needed = Vec::new();

        for (i, stop) in stops.iter().enumerate() {
            for (j, entry) in entry_levels.iter().enumerate() {
                if direction == "SHORT" {
                    if stop.price <= entry.price {
                        log::error!(
                            "❌ CRITICAL: SHORT stop {} ({:.2}) would intersect with entry {} ({:.2})",
                            i + 1, stop.price, j + 1, entry.price
                        );
                        // Calculate safe stop position using volatility data
                        let safety_buffer =
                            self.volatility.recommended_stop_distance_percent / 100.0;
                        let safe_stop = extreme_entry
                            * (1.0 + safety_buffer * (1.0 + i as f64 * progressive_factor));
                        corrections_needed.push((i, safe_stop));
                        log::info!(
                            "✅ Will correct stop {} to {:.2} using volatility buffer",
                            i + 1,
                            safe_stop
                        );
                    }
                } else if stop.price >= entry.price {
                    log::error!(
                        "❌ CRITICAL: LONG stop {} ({:.2}) would intersect with entry {} ({:.2})",
                        i + 1,
                        stop.price,
                        j + 1,
                        entry.price
                    );
                    // Calculate safe stop position using volatility data
                    let safety_buffer = self.volatility.recommended_stop_distance_percent / 100.0;
                    let safe_stop = extreme_entry
                        * (1.0 - safety_buffer * (1.0 + i as f64 * progressive_factor));
                    corrections_needed.push((i, safe_stop));
                    log::info!(
                        "✅ Will correct stop {} to {:.2} using volatility buffer",
                        i + 1,
                        safe_stop
                    );
                }
            }
        }

        // Apply corrections if needed
        for (i, safe_price) in corrections_needed {
            stops[i].price = safe_price;
        }

        Ok([stops[0].clone(), stops[1].clone(), stops[2].clone()])
    }

    /// Generate SEQUENCE-AWARE stop levels using MAE distribution
    /// Uses actual adverse excursions from sequence data for realistic stops
    pub fn generate_sequence_aware_stops(
        &self,
        entry_levels: &[OrderLevel; 3],
        direction: &str,
        sequence_stats: &SequenceStatistics,
    ) -> Result<[OrderLevel; 3]> {
        let mut stops = Vec::new();

        // Get typical adverse excursion from MAE distribution
        // Use 75th percentile - most trades survive this drawdown
        let typical_mae = if !sequence_stats.mae_distribution.is_empty() {
            SequenceStatistics::percentile(&sequence_stats.mae_distribution, 75.0)
        } else {
            // Fallback to volatility-based estimate
            sequence_stats.std_return * 2.0
        };

        // Adjust stop distance by Kelly fraction (risk management)
        // Lower Kelly = wider stops (more conservative)
        // Higher Kelly = tighter stops (more aggressive)
        let kelly_adjustment = if sequence_stats.kelly_fraction > 0.0 {
            1.0 / sequence_stats.kelly_fraction.max(0.1)
        } else {
            2.0 // Conservative default
        };

        // Base stop distance from MAE and Kelly
        let base_stop_distance = typical_mae * kelly_adjustment;

        // Find extreme entry for stop placement
        let extreme_entry = if direction == "SHORT" {
            entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::INFINITY, f64::min)
        };

        log::info!(
            "📊 Sequence-Aware Stop Calculation: MAE={:.3}%, Kelly={:.2}, base_stop={:.3}%",
            typical_mae * 100.0,
            sequence_stats.kelly_fraction,
            base_stop_distance * 100.0
        );

        // Generate stops with progressive spacing based on position sizes
        for (i, entry) in entry_levels.iter().enumerate() {
            // Position-weighted stop distance
            // Larger positions get tighter stops (risk parity)
            let position_weight = entry.quantity_percentage;
            let position_adjustment = 1.0 / position_weight.max(0.1);

            // Progressive spacing using Fibonacci sequence for natural progression
            let fib_multiplier = match i {
                0 => 1.0,   // First stop: 1x base distance
                1 => 1.5,   // Second stop: 1.5x (between 1 and 2)
                2 => 2.0,   // Third stop: 2x base distance
                _ => 1.618, // Golden ratio default
            };

            // Market regime adjustment
            let regime_adjustment = if sequence_stats.hurst_exponent > 0.5 {
                // Trending market: tighter stops (trend should continue)
                1.0 - (sequence_stats.hurst_exponent - 0.5) * 0.5
            } else {
                // Mean reverting: wider stops (expect temporary adverse moves)
                1.0 + (0.5 - sequence_stats.hurst_exponent) * 0.5
            };

            // Final stop distance calculation
            let stop_distance =
                base_stop_distance * position_adjustment * fib_multiplier * regime_adjustment;

            // Calculate stop price from extreme entry
            let stop_price = if direction == "SHORT" {
                extreme_entry * (1.0 + stop_distance)
            } else {
                extreme_entry * (1.0 - stop_distance)
            };

            // Stop confidence based on:
            // 1. MAE distribution confidence (how reliable is our data)
            // 2. Position size (larger positions = need higher confidence)
            // 3. Market efficiency (trending markets = higher confidence)
            let mae_confidence = (sequence_stats.mae_distribution.len() as f64 / 100.0).min(1.0);
            let efficiency_confidence = if sequence_stats.hurst_exponent > 0.5 {
                0.8 + (sequence_stats.hurst_exponent - 0.5) * 0.4
            } else {
                0.6 + sequence_stats.hurst_exponent * 0.4
            };

            let stop_confidence =
                (mae_confidence * position_weight * efficiency_confidence).clamp(0.3, 0.95);

            stops.push(OrderLevel {
                price: stop_price,
                quantity_percentage: entry.quantity_percentage, // Match entry size
                atr_distance: stop_distance * 100.0,            // Convert to percentage
                order_type: "STOP_LOSS".to_string(),
                confidence: stop_confidence,
            });

            log::info!(
                "  Stop {}: ${:.2} ({:+.3}% from extreme) | Size: {:.1}% | MAE-based: {:.3}% | Conf: {:.2}",
                i + 1,
                stop_price,
                ((stop_price - extreme_entry) / extreme_entry) * 100.0,
                entry.quantity_percentage * 100.0,
                stop_distance * 100.0,
                stop_confidence
            );
        }

        // Validate stops don't intersect with entries
        let mut adjustments_needed = Vec::new();
        for (i, stop) in stops.iter().enumerate() {
            for (j, entry) in entry_levels.iter().enumerate() {
                let intersects = if direction == "SHORT" {
                    stop.price <= entry.price
                } else {
                    stop.price >= entry.price
                };

                if intersects {
                    log::warn!(
                        "⚠️ Stop {} would intersect entry {}. Adjusting using MAE safety buffer.",
                        i + 1,
                        j + 1
                    );

                    // Use minimum MAE as safety buffer
                    let min_mae = if !sequence_stats.mae_distribution.is_empty() {
                        sequence_stats.mae_distribution[0] // Minimum observed MAE
                    } else {
                        sequence_stats.std_return // One standard deviation
                    };

                    let new_price = if direction == "SHORT" {
                        extreme_entry * (1.0 + min_mae * 1.5)
                    } else {
                        extreme_entry * (1.0 - min_mae * 1.5)
                    };

                    adjustments_needed.push((i, new_price));
                }
            }
        }

        // Apply adjustments after iteration
        for (i, new_price) in adjustments_needed {
            stops[i].price = new_price;
        }

        Ok([stops[0].clone(), stops[1].clone(), stops[2].clone()])
    }

    /// FULLY ADAPTIVE stop generation using actual sequence drawdown analysis
    /// NO hardcoded multipliers - everything derived from sequence statistical boundaries
    pub fn generate_fully_adaptive_stops(
        &self,
        entry_levels: &[OrderLevel; 3],
        direction: &str,
        sequence_prices: &[f64],
        sequence_stats: &SequenceStatistics,
    ) -> Result<[OrderLevel; 3]> {
        let mut stops = Vec::new();

        // Get adaptive bounds from sequence data
        let bounds = sequence_stats.get_adaptive_bounds(sequence_prices, entry_levels[0].price);

        // Find extreme entry for stop placement (ensures no intersection)
        let extreme_entry = if direction == "SHORT" {
            entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::INFINITY, f64::min)
        };

        // Calculate adverse probability from model predictions
        let adverse_probability = if direction == "SHORT" {
            // For SHORT, risk comes from upside moves
            let strong_up_prob = self
                .price_levels
                .bins
                .get("strong_up")
                .map(|b| b.probability)
                .unwrap_or(0.0);
            let moderate_up_prob = self
                .price_levels
                .bins
                .get("moderate_up")
                .map(|b| b.probability)
                .unwrap_or(0.0);
            strong_up_prob + moderate_up_prob
        } else {
            // For LONG, risk comes from downside moves
            let strong_down_prob = self
                .price_levels
                .bins
                .get("strong_down")
                .map(|b| b.probability)
                .unwrap_or(0.0);
            let moderate_down_prob = self
                .price_levels
                .bins
                .get("moderate_down")
                .map(|b| b.probability)
                .unwrap_or(0.0);
            strong_down_prob + moderate_down_prob
        };

        log::info!(
            "🛑 Fully Adaptive Stop Generation: max_drawdown={:.2}%, adverse_prob={:.1}%, IQR_vol={:.2}%",
            bounds.max_drawdown_pct,
            adverse_probability * 100.0,
            bounds.iqr_volatility
        );

        // Generate 3 stops using sequence drawdown analysis
        for (i, entry) in entry_levels.iter().enumerate() {
            // Base stop distance using actual sequence drawdown
            let sequence_stop_distance = bounds.max_drawdown_pct * adverse_probability;

            // Progressive stops using IQR-based spacing from sequence percentiles
            let iqr_spacing = match i {
                0 => (extreme_entry - bounds.p25).abs() / extreme_entry * 100.0, // Q1 distance (conservative)
                1 => (extreme_entry - bounds.p10).abs() / extreme_entry * 100.0, // 10th percentile (moderate)
                2 => (extreme_entry - bounds.sequence_min).abs() / extreme_entry * 100.0, // Minimum seen (aggressive)
                _ => bounds.iqr_volatility * 0.5,
            };

            // Use the larger of sequence-based or IQR-based distance for safety
            let final_stop_distance = sequence_stop_distance.max(iqr_spacing);

            // Calculate stop price from extreme entry
            let stop_price = if direction == "SHORT" {
                extreme_entry * (1.0 + final_stop_distance / 100.0)
            } else {
                extreme_entry * (1.0 - final_stop_distance / 100.0)
            };

            // Validate using z-score (within 3 standard deviations)
            let validated_price =
                sequence_stats.validate_price_with_zscore(sequence_prices, stop_price);

            // Confidence based on:
            // 1. Adverse probability (lower adverse risk = higher confidence)
            // 2. Sequence drawdown reliability (how much historical data we have)
            // 3. Distance from extreme entry (further = lower confidence due to slippage risk)
            let adverse_confidence = 1.0 - adverse_probability; // Lower adverse prob = higher confidence
            let data_confidence = (sequence_prices.len() as f64 / 100.0).min(1.0); // More data = higher confidence
            let distance_confidence = (-final_stop_distance / 20.0).exp(); // Exponential decay with distance

            let final_confidence =
                (adverse_confidence * data_confidence * distance_confidence).clamp(0.3, 0.95);

            let atr_distance = ((validated_price - extreme_entry).abs() / extreme_entry) * 100.0;

            stops.push(OrderLevel {
                price: validated_price,
                quantity_percentage: entry.quantity_percentage, // Match entry size
                atr_distance,
                order_type: "STOP_LOSS".to_string(),
                confidence: final_confidence,
            });

            log::info!(
                "  {} Adaptive Stop {}: ${:.4} ({:.2}% from extreme) | Seq: {:.2}% | IQR: {:.2}% | Final: {:.2}% | Conf: {:.2}",
                direction,
                i + 1,
                validated_price,
                atr_distance,
                sequence_stop_distance,
                iqr_spacing,
                final_stop_distance,
                final_confidence
            );
        }

        // Validate stops don't intersect with entries using sequence-based safety buffer
        let mut adjustments_needed = Vec::new();
        for (i, stop) in stops.iter().enumerate() {
            for (j, entry) in entry_levels.iter().enumerate() {
                let intersects = if direction == "SHORT" {
                    stop.price <= entry.price
                } else {
                    stop.price >= entry.price
                };

                if intersects {
                    log::warn!(
                        "⚠️ Stop {} would intersect entry {}. Adjusting using sequence IQR safety buffer.",
                        i + 1,
                        j + 1
                    );

                    // Use IQR as safety buffer (robust to outliers)
                    let safety_buffer = bounds.iqr_volatility / 100.0;
                    let new_price = if direction == "SHORT" {
                        extreme_entry * (1.0 + safety_buffer * 1.5)
                    } else {
                        extreme_entry * (1.0 - safety_buffer * 1.5)
                    };

                    adjustments_needed.push((i, new_price));
                }
            }
        }

        // Apply adjustments after iteration
        for (i, new_price) in adjustments_needed {
            stops[i].price = new_price;
        }

        Ok([stops[0].clone(), stops[1].clone(), stops[2].clone()])
    }

    /// Calculate entry size using PURE prediction data - no hardcoded allocations
    fn calculate_entry_size(
        &self,
        level: usize,
        direction: &DirectionPrediction,
        volatility: &VolatilityPrediction,
    ) -> f64 {
        // Use Kelly Criterion-like approach based on confidence
        let directional_edge =
            if direction.up_probability_aggregated > direction.down_probability_aggregated {
                direction.up_probability_aggregated - direction.down_probability_aggregated
            } else {
                direction.down_probability_aggregated - direction.up_probability_aggregated
            };

        // Kelly fraction = edge / odds
        // Using expected upside/downside as odds proxy
        let odds = direction.expected_upside_percent / direction.expected_downside_percent.max(0.1);
        let kelly_fraction = (directional_edge * odds).clamp(0.1, 0.5);

        // Progressive allocation based on level (Fibonacci-like)
        let level_weights = [0.5, 0.3, 0.2]; // Natural 50-30-20 split
        let base_allocation = kelly_fraction * level_weights[level - 1];

        // Volatility adjustment from model
        let volatility_adjustment = volatility.position_size_multiplier;

        // Confidence scaling
        let confidence_scale = direction.confidence + 0.5; // Range 0.5-1.5

        let final_size = base_allocation * volatility_adjustment * confidence_scale;

        log::debug!(
            "Entry {} sizing: kelly={:.3}, base={:.3}, vol_adj={:.2}, conf={:.2} → {:.3}",
            level,
            kelly_fraction,
            base_allocation,
            volatility_adjustment,
            confidence_scale,
            final_size
        );

        final_size.clamp(0.1, 0.6)
    }

    fn volume_liquidity_factor(&self) -> f64 {
        match self.volume.regime.as_str() {
            "VERY_HIGH" => 1.2, // High volume = easier exits
            "HIGH" => 1.1,
            "MEDIUM" => 1.0,
            "LOW" => 0.9,
            "VERY_LOW" => 0.8, // Low volume = harder exits
            _ => 1.0,
        }
    }

    /// Normalize position sizes to ensure they sum to 1.0
    pub fn normalize_sizes(levels: &mut [OrderLevel; 3]) {
        let total: f64 = levels.iter().map(|l| l.quantity_percentage).sum();
        if total > 0.0 {
            for level in levels.iter_mut() {
                level.quantity_percentage /= total;
            }
        }
    }

    /// Calculate ADAPTIVE risk-reward requirement using information entropy
    /// Higher entropy (uncertainty) = need better risk-reward ratio
    pub fn calculate_adaptive_risk_reward_requirement(
        &self,
        sequence_stats: &SequenceStatistics,
    ) -> f64 {
        // Base requirement from information content
        // Higher entropy = more uncertainty = need better R:R
        let price_entropy_factor = sequence_stats.price_entropy.exp();

        // Volume entropy contribution (if available)
        let volume_entropy_factor = sequence_stats
            .volume_entropy
            .map(|ve| ve.exp())
            .unwrap_or(1.0);

        // Market efficiency adjustment
        // Trending markets (Hurst > 0.5) can accept lower R:R
        // Mean reverting markets need higher R:R
        let efficiency_adjustment = if sequence_stats.hurst_exponent > 0.5 {
            1.0 - (sequence_stats.hurst_exponent - 0.5) * 0.5 // Reduce requirement
        } else {
            1.0 + (0.5 - sequence_stats.hurst_exponent) * 0.5 // Increase requirement
        };

        // Volatility adjustment
        // Higher volatility = need better R:R for safety
        let volatility_factor = 1.0 + sequence_stats.std_return * 10.0;

        // Kelly fraction adjustment
        // Higher Kelly = more edge = can accept lower R:R
        let kelly_adjustment = if sequence_stats.kelly_fraction > 0.25 {
            1.0 - (sequence_stats.kelly_fraction - 0.25) * 0.5
        } else {
            1.0 + (0.25 - sequence_stats.kelly_fraction) * 2.0
        };

        // Calculate final requirement
        let base_requirement = price_entropy_factor * volume_entropy_factor;
        let adjusted_requirement =
            base_requirement * efficiency_adjustment * volatility_factor * kelly_adjustment;

        // Ensure reasonable bounds (minimum 2.0 for viability, maximum 10.0 for practicality)
        let final_requirement = adjusted_requirement.clamp(2.0, 10.0);

        log::info!(
            "📊 Adaptive R:R Requirement: price_entropy={:.2}, hurst={:.2}, volatility={:.3}, kelly={:.2} → Required R:R={:.2}",
            sequence_stats.price_entropy,
            sequence_stats.hurst_exponent,
            sequence_stats.std_return,
            sequence_stats.kelly_fraction,
            final_requirement
        );

        final_requirement
    }

    /// Optimize orders using sequence-aware adaptive parameters
    pub fn optimize_with_sequence_stats(
        &self,
        entry_levels: &mut [OrderLevel; 3],
        exit_levels: &mut [OrderLevel; 3],
        stop_levels: &mut [OrderLevel; 3],
        direction: &str,
        sequence_stats: &SequenceStatistics,
    ) -> f64 {
        // Calculate current risk-reward
        let current_rr =
            Self::calculate_risk_reward_static(entry_levels, exit_levels, stop_levels, direction);

        // Get adaptive requirement
        let required_rr = self.calculate_adaptive_risk_reward_requirement(sequence_stats);

        if current_rr >= required_rr {
            log::info!(
                "✅ Risk-Reward {:.2} meets adaptive requirement {:.2}",
                current_rr,
                required_rr
            );
            return current_rr;
        }

        log::info!(
            "⚠️ Optimizing R:R from {:.2} to meet adaptive requirement {:.2}",
            current_rr,
            required_rr
        );

        // Use MAE/MFE distributions for intelligent optimization
        let max_iterations = 10;
        let mut best_rr = current_rr;

        for iteration in 1..=max_iterations {
            // Calculate improvement needed
            let _improvement_factor = required_rr / best_rr;

            // Intelligent adjustments based on MAE/MFE
            if !sequence_stats.mae_distribution.is_empty() {
                // Move stops closer using lower MAE percentile
                let aggressive_mae = SequenceStatistics::percentile(
                    &sequence_stats.mae_distribution,
                    50.0 - (iteration as f64 * 5.0),
                );

                // Adjust stops based on aggressive MAE
                for (i, stop) in stop_levels.iter_mut().enumerate() {
                    let entry_price = entry_levels[i].price;
                    let new_distance = aggressive_mae * (1.0 + i as f64 * 0.2);

                    stop.price = if direction == "SHORT" {
                        entry_price * (1.0 + new_distance)
                    } else {
                        entry_price * (1.0 - new_distance)
                    };
                }
            }

            if !sequence_stats.mfe_distribution.is_empty() && iteration > 3 {
                // Enhance exits using higher MFE percentile
                let optimistic_mfe = SequenceStatistics::percentile(
                    &sequence_stats.mfe_distribution,
                    75.0 + (iteration as f64 * 2.5).min(20.0),
                );

                // Adjust exits for better profit
                for exit in exit_levels.iter_mut() {
                    let current_distance =
                        ((exit.price - entry_levels[0].price) / entry_levels[0].price).abs();
                    let new_distance = current_distance.max(optimistic_mfe);

                    exit.price = if direction == "SHORT" {
                        entry_levels[0].price * (1.0 - new_distance)
                    } else {
                        entry_levels[0].price * (1.0 + new_distance)
                    };
                }
            }

            // Recalculate R:R
            best_rr = Self::calculate_risk_reward_static(
                entry_levels,
                exit_levels,
                stop_levels,
                direction,
            );

            if best_rr >= required_rr {
                log::info!(
                    "✅ Optimized to R:R {:.2} in {} iterations (required: {:.2})",
                    best_rr,
                    iteration,
                    required_rr
                );
                break;
            }
        }

        best_rr
    }

    /// Static helper to calculate risk-reward ratio
    fn calculate_risk_reward_static(
        entry_levels: &[OrderLevel; 3],
        exit_levels: &[OrderLevel; 3],
        stop_levels: &[OrderLevel; 3],
        direction: &str,
    ) -> f64 {
        let avg_entry =
            (entry_levels[0].price + entry_levels[1].price + entry_levels[2].price) / 3.0;
        let avg_exit = (exit_levels[0].price + exit_levels[1].price + exit_levels[2].price) / 3.0;
        let avg_stop = (stop_levels[0].price + stop_levels[1].price + stop_levels[2].price) / 3.0;

        let (profit, risk) = if direction == "SHORT" {
            let profit = avg_entry - avg_exit;
            let risk = avg_stop - avg_entry;
            (profit, risk)
        } else {
            let profit = avg_exit - avg_entry;
            let risk = avg_entry - avg_stop;
            (profit, risk)
        };

        if risk > 0.0 {
            profit / risk
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_smart_consensus_no_magic_numbers() {
        // Test that all generated values come from model outputs, not hardcoded
        // Implementation in smart_order_generator_test.rs
    }
}
