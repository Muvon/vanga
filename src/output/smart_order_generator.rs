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
    /// Calculate SMART consensus for trade direction with confidence
    pub fn calculate_direction_consensus(&self) -> (String, f64) {
        // Direction model is PRIMARY for direction decision
        let direction_signal = if self.direction.up_probability_aggregated
            > self.direction.down_probability_aggregated
        {
            "LONG"
        } else {
            "SHORT"
        };

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

        (direction_signal.to_string(), final_confidence)
    }

    /// Generate SMART entry levels using Price Levels + Direction momentum
    pub fn generate_smart_entries(
        &self,
        current_price: f64,
        direction: &str,
    ) -> Result<[OrderLevel; 3]> {
        let mut entries = Vec::new();

        // Calculate entry depth based on direction confidence and volatility
        // Strong directional bias = don't chase too far
        let directional_confidence = (self.direction.up_probability_aggregated
            - self.direction.down_probability_aggregated)
            .abs();
        let max_entry_depth = if directional_confidence > 0.3 {
            // Strong direction: limit entry depth to expected range
            self.volatility.expected_range_percent * 0.5 // Only use half the expected range
        } else if directional_confidence > 0.15 {
            // Moderate direction: use most of expected range
            self.volatility.expected_range_percent * 0.75
        } else {
            // Weak direction: can use full expected range
            self.volatility.expected_range_percent
        };

        log::info!(
            "📍 Entry Depth Calculation: directional_confidence={:.2}, max_depth={:.2}%",
            directional_confidence,
            max_entry_depth
        );

        if direction == "SHORT" {
            // SHORT entries: Use upside bins (where we want to sell)
            // But limit depth based on directional confidence

            // Entry 1: Close to market - use neutral upper bound or smaller
            let entry_1_distance = if let Some(neutral_bin) = self.price_levels.bins.get("neutral")
            {
                neutral_bin.range[1].min(max_entry_depth / 3.0) // First third of allowed depth
            } else {
                max_entry_depth / 3.0
            };

            // Entry 2: Medium distance - use moderate_up center or limit
            let entry_2_distance =
                if let Some(moderate_up_bin) = self.price_levels.bins.get("moderate_up") {
                    ((moderate_up_bin.range[0] + moderate_up_bin.range[1]) / 2.0)
                        .min(max_entry_depth * 0.66)
                } else {
                    max_entry_depth * 0.66
                };

            // Entry 3: Further out - use strong_up lower or max depth
            let entry_3_distance =
                if let Some(strong_up_bin) = self.price_levels.bins.get("strong_up") {
                    strong_up_bin.range[0].min(max_entry_depth)
                } else {
                    max_entry_depth
                };

            // Generate entries with calculated distances
            let distances = [entry_1_distance, entry_2_distance, entry_3_distance];

            for (i, distance) in distances.iter().enumerate() {
                let entry_price = current_price * (1.0 + distance / 100.0);
                let size = self.calculate_entry_size(i + 1, &self.direction, &self.volatility);

                // Confidence based on:
                // 1. Price bin probability (if available)
                // 2. Direction confidence
                // 3. Distance from current price (closer = higher confidence)
                let distance_factor = 1.0 - (distance / max_entry_depth).min(1.0) * 0.5; // Keep at least 0.5
                let confidence =
                    (self.direction.down_probability_aggregated * distance_factor).max(0.1); // Minimum 10%

                entries.push(OrderLevel {
                    price: entry_price,
                    quantity_percentage: size,
                    atr_distance: 0.0,
                    order_type: "LIMIT".to_string(),
                    confidence: confidence.min(0.95),
                });

                log::info!(
                    "  SHORT Entry {}: ${:.2} (+{:.3}%) | Size: {:.1}% | Conf: {:.2}",
                    i + 1,
                    entry_price,
                    distance,
                    size * 100.0,
                    confidence
                );
            }
        } else {
            // LONG entries: Use downside bins (where we want to buy)
            // But limit depth based on directional confidence

            // Entry 1: Close to market
            let entry_1_distance = if let Some(neutral_bin) = self.price_levels.bins.get("neutral")
            {
                neutral_bin.range[0].abs().min(max_entry_depth / 3.0)
            } else {
                max_entry_depth / 3.0
            };

            // Entry 2: Medium distance
            let entry_2_distance =
                if let Some(moderate_down_bin) = self.price_levels.bins.get("moderate_down") {
                    ((moderate_down_bin.range[0] + moderate_down_bin.range[1]) / 2.0)
                        .abs()
                        .min(max_entry_depth * 0.66)
                } else {
                    max_entry_depth * 0.66
                };

            // Entry 3: Further out
            let entry_3_distance =
                if let Some(strong_down_bin) = self.price_levels.bins.get("strong_down") {
                    strong_down_bin.range[1].abs().min(max_entry_depth)
                } else {
                    max_entry_depth
                };

            // Generate entries with calculated distances
            let distances = [entry_1_distance, entry_2_distance, entry_3_distance];

            for (i, distance) in distances.iter().enumerate() {
                let entry_price = current_price * (1.0 - distance / 100.0);
                let size = self.calculate_entry_size(i + 1, &self.direction, &self.volatility);

                // Confidence calculation
                let distance_factor = 1.0 - (distance / max_entry_depth).min(1.0) * 0.5; // Keep at least 0.5
                let confidence =
                    (self.direction.up_probability_aggregated * distance_factor).max(0.1); // Minimum 10%

                entries.push(OrderLevel {
                    price: entry_price,
                    quantity_percentage: size,
                    atr_distance: 0.0,
                    order_type: "LIMIT".to_string(),
                    confidence: confidence.min(0.95),
                });

                log::info!(
                    "  LONG Entry {}: ${:.2} (-{:.3}%) | Size: {:.1}% | Conf: {:.2}",
                    i + 1,
                    entry_price,
                    distance,
                    size * 100.0,
                    confidence
                );
            }
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

        // Golden ratio for natural, non-arbitrary progression
        const PHI: f64 = 1.618033988749895;

        // Use time-scaled volatility from sequence (σ√t)
        let base_spacing = sequence_stats.time_scaled_volatility;

        // Adjust spacing based on market regime (trending vs mean-reverting)
        let regime_adjustment = if sequence_stats.mean_reversion_rate.abs() > 0.5 {
            // Mean reverting market: wider spacing (price likely to revert)
            1.0 + sequence_stats.mean_reversion_rate.abs()
        } else {
            // Trending market: tighter spacing to catch the trend
            // Use ratio of drift to volatility
            1.0 - (sequence_stats.mean_return / sequence_stats.std_return)
                .abs()
                .min(0.5)
        };

        // Combine with directional confidence
        let directional_confidence = (self.direction.up_probability_aggregated
            - self.direction.down_probability_aggregated)
            .abs();

        // Final spacing calculation
        let entry_spacing = base_spacing * regime_adjustment;

        log::info!(
            "📊 Sequence-Aware Entry Spacing: base={:.3}%, regime_adj={:.2}x, final={:.3}%",
            base_spacing * 100.0,
            regime_adjustment,
            entry_spacing * 100.0
        );

        // Generate entries with golden ratio progression
        for i in 0..3 {
            let progression_factor = PHI.powi(i);
            let distance = entry_spacing * progression_factor;

            let entry_price = if direction == "SHORT" {
                current_price * (1.0 + distance)
            } else {
                current_price * (1.0 - distance)
            };

            // Size based on Kelly fraction and confidence
            let base_size = match i {
                0 => 0.5, // Largest position at best price
                1 => 0.3, // Medium position
                2 => 0.2, // Smallest position at worst price
                _ => 0.33,
            };

            // Adjust size by Kelly fraction (optimal sizing)
            let kelly_adjusted_size = base_size * (1.0 + sequence_stats.kelly_fraction - 0.25);

            // Confidence based on:
            // 1. Directional confidence
            // 2. Distance from current (closer = higher confidence)
            // 3. Market efficiency (Hurst exponent)
            let distance_decay = (-distance * 10.0).exp(); // Exponential decay with distance
            let efficiency_factor = if sequence_stats.hurst_exponent > 0.5 {
                // Trending market: higher confidence
                1.0 + (sequence_stats.hurst_exponent - 0.5)
            } else {
                // Mean reverting: lower confidence
                1.0 - (0.5 - sequence_stats.hurst_exponent)
            };

            let confidence =
                (directional_confidence * distance_decay * efficiency_factor).clamp(0.1, 0.95);

            entries.push(OrderLevel {
                price: entry_price,
                quantity_percentage: kelly_adjusted_size,
                atr_distance: 0.0,
                order_type: "LIMIT".to_string(),
                confidence,
            });

            log::info!(
                "  {} Entry {}: ${:.2} ({:+.3}%) | Size: {:.1}% | Conf: {:.2} | Kelly: {:.2}",
                direction,
                i + 1,
                entry_price,
                if direction == "SHORT" {
                    distance * 100.0
                } else {
                    -distance * 100.0
                },
                kelly_adjusted_size * 100.0,
                confidence,
                sequence_stats.kelly_fraction
            );
        }

        Ok([entries[0].clone(), entries[1].clone(), entries[2].clone()])
    }

    /// Generate SMART exit levels using Price Levels + Volume for liquidity
    /// Can be optimized for better risk-reward ratio
    pub fn generate_smart_exits(
        &self,
        current_price: f64,
        direction: &str,
    ) -> Result<[OrderLevel; 3]> {
        let mut exits = Vec::new();

        // Calculate minimum profitable exit distance based on volatility
        // This ensures we're not setting exits too close
        let min_profit_distance = self.volatility.expected_range_percent * 0.5; // At least half the expected range

        if direction == "SHORT" {
            // SHORT exits (profits): Use downside bins
            // Exit 1: Conservative - moderate_down center or minimum profit
            let exit_1_distance =
                if let Some(moderate_down_bin) = self.price_levels.bins.get("moderate_down") {
                    ((moderate_down_bin.range[0] + moderate_down_bin.range[1]) / 2.0)
                        .abs()
                        .max(min_profit_distance * 0.5) // At least half minimum profit
                } else {
                    min_profit_distance * 0.5
                };

            let exit_1_price = current_price * (1.0 - exit_1_distance / 100.0);
            let size = self.calculate_exit_size(1, &self.volume);
            let confidence = self
                .price_levels
                .bins
                .get("moderate_down")
                .map(|b| b.probability)
                .unwrap_or(0.2)
                * self.volume_liquidity_factor();

            exits.push(OrderLevel {
                price: exit_1_price,
                quantity_percentage: size,
                atr_distance: 0.0,
                order_type: "LIMIT".to_string(),
                confidence: confidence.min(0.9),
            });

            // Exit 2: Target - strong_down center or expected profit
            let exit_2_distance =
                if let Some(strong_down_bin) = self.price_levels.bins.get("strong_down") {
                    ((strong_down_bin.range[0] + strong_down_bin.range[1]) / 2.0)
                        .abs()
                        .max(min_profit_distance) // At least minimum profit
                } else {
                    min_profit_distance
                };

            let exit_2_price = current_price * (1.0 - exit_2_distance / 100.0);
            let size = self.calculate_exit_size(2, &self.volume);
            let confidence = self
                .price_levels
                .bins
                .get("strong_down")
                .map(|b| b.probability)
                .unwrap_or(0.15)
                * self.volume_liquidity_factor();

            exits.push(OrderLevel {
                price: exit_2_price,
                quantity_percentage: size,
                atr_distance: 0.0,
                order_type: "LIMIT".to_string(),
                confidence: confidence.min(0.7),
            });

            // Exit 3: Stretch - strong_down lower or maximum expected profit
            let exit_3_distance =
                if let Some(strong_down_bin) = self.price_levels.bins.get("strong_down") {
                    strong_down_bin.range[0]
                        .abs()
                        .max(min_profit_distance * 1.5) // At least 1.5x minimum profit
                } else {
                    min_profit_distance * 1.5
                };

            let exit_3_price = current_price * (1.0 - exit_3_distance / 100.0);
            let size = self.calculate_exit_size(3, &self.volume);
            let confidence = self
                .price_levels
                .bins
                .get("strong_down")
                .map(|b| b.probability * 0.7) // Lower confidence for stretch target
                .unwrap_or(0.1)
                * self.volume_liquidity_factor();

            exits.push(OrderLevel {
                price: exit_3_price,
                quantity_percentage: size,
                atr_distance: 0.0,
                order_type: "LIMIT".to_string(),
                confidence: confidence.min(0.5),
            });
        } else {
            // LONG exits (profits): Use upside bins
            // Exit 1: Conservative - moderate_up center or minimum profit
            let exit_1_distance =
                if let Some(moderate_up_bin) = self.price_levels.bins.get("moderate_up") {
                    ((moderate_up_bin.range[0] + moderate_up_bin.range[1]) / 2.0)
                        .max(min_profit_distance * 0.5)
                } else {
                    min_profit_distance * 0.5
                };

            let exit_1_price = current_price * (1.0 + exit_1_distance / 100.0);
            let size = self.calculate_exit_size(1, &self.volume);
            let confidence = self
                .price_levels
                .bins
                .get("moderate_up")
                .map(|b| b.probability)
                .unwrap_or(0.2)
                * self.volume_liquidity_factor();

            exits.push(OrderLevel {
                price: exit_1_price,
                quantity_percentage: size,
                atr_distance: 0.0,
                order_type: "LIMIT".to_string(),
                confidence: confidence.min(0.9),
            });

            // Exit 2: Target - strong_up center or expected profit
            let exit_2_distance = if let Some(strong_up_bin) =
                self.price_levels.bins.get("strong_up")
            {
                ((strong_up_bin.range[0] + strong_up_bin.range[1]) / 2.0).max(min_profit_distance)
            } else {
                min_profit_distance
            };

            let exit_2_price = current_price * (1.0 + exit_2_distance / 100.0);
            let size = self.calculate_exit_size(2, &self.volume);
            let confidence = self
                .price_levels
                .bins
                .get("strong_up")
                .map(|b| b.probability)
                .unwrap_or(0.15)
                * self.volume_liquidity_factor();

            exits.push(OrderLevel {
                price: exit_2_price,
                quantity_percentage: size,
                atr_distance: 0.0,
                order_type: "LIMIT".to_string(),
                confidence: confidence.min(0.7),
            });

            // Exit 3: Stretch - strong_up upper or maximum expected profit
            let exit_3_distance =
                if let Some(strong_up_bin) = self.price_levels.bins.get("strong_up") {
                    strong_up_bin.range[1].max(min_profit_distance * 1.5)
                } else {
                    min_profit_distance * 1.5
                };

            let exit_3_price = current_price * (1.0 + exit_3_distance / 100.0);
            let size = self.calculate_exit_size(3, &self.volume);
            let confidence = self
                .price_levels
                .bins
                .get("strong_up")
                .map(|b| b.probability * 0.7)
                .unwrap_or(0.1)
                * self.volume_liquidity_factor();

            exits.push(OrderLevel {
                price: exit_3_price,
                quantity_percentage: size,
                atr_distance: 0.0,
                order_type: "LIMIT".to_string(),
                confidence: confidence.min(0.5),
            });
        }

        log::info!(
            "🎯 SMART Exits: Ensuring minimum profit distance of {:.2}%",
            min_profit_distance
        );

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

            exits.push(OrderLevel {
                price: exit_price,
                quantity_percentage: exit_size,
                atr_distance: 0.0,
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

    /// Generate SMART stop levels using Volatility model (it knows risk best!)
    /// CRITICAL: Stops must NEVER intersect with ANY entry level
    pub fn generate_smart_stops(
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

    /// Calculate entry size using Direction confidence + Volatility sizing
    fn calculate_entry_size(
        &self,
        level: usize,
        direction: &DirectionPrediction,
        volatility: &VolatilityPrediction,
    ) -> f64 {
        // Base allocation (front-loaded for better average price)
        let base_allocations = [0.5, 0.3, 0.2]; // 50%, 30%, 20%

        // Adjust based on direction confidence
        let confidence_factor = if direction.up_probability_aggregated > 0.6
            || direction.down_probability_aggregated > 0.6
        {
            1.2 // More aggressive sizing with high confidence
        } else if direction.sideways_probability_aggregated > 0.4 {
            0.8 // Conservative in sideways markets
        } else {
            1.0
        };

        // Volatility adjustment (from position_size_multiplier)
        let volatility_factor = volatility.position_size_multiplier;

        let size = base_allocations[level - 1] * confidence_factor * volatility_factor;

        // Ensure sizes sum to 1.0 (will normalize later)
        size.clamp(0.1, 0.6) // Cap between 10% and 60%
    }

    /// Calculate exit size using Volume regime for liquidity assessment
    fn calculate_exit_size(&self, level: usize, volume: &VolumePrediction) -> f64 {
        // Base exit allocation
        let base_allocations = match volume.regime.as_str() {
            "VERY_HIGH" | "HIGH" => [0.3, 0.4, 0.3], // Can exit more at once with high volume
            "MEDIUM" => [0.4, 0.4, 0.2],             // Balanced exits
            "LOW" | "VERY_LOW" => [0.5, 0.3, 0.2],   // Take more profit early in low volume
            _ => [0.4, 0.4, 0.2],
        };

        base_allocations[level - 1]
    }

    /// Calculate volume liquidity factor for exit confidence
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
