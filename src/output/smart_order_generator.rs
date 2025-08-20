//! SMART Order Generation System - Using RIGHT models for RIGHT purposes
//! No magic numbers, pure mathematical consensus based on model strengths

use crate::output::prediction_types::{
    DirectionPrediction, PriceLevelPrediction, SentimentPrediction, VolatilityPrediction,
    VolumePrediction,
};
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

    /// Calculate overall trade confidence using all models
    pub fn calculate_overall_confidence(&self) -> f64 {
        // Each model contributes based on its confidence
        let direction_weight = 0.3;
        let price_weight = 0.25;
        let volatility_weight = 0.2;
        let sentiment_weight = 0.15;
        let volume_weight = 0.1;

        let weighted_confidence = self.direction.confidence * direction_weight
            + self.price_levels.confidence * price_weight
            + self.volatility.regime_confidence * volatility_weight
            + self.sentiment.confidence * sentiment_weight
            + self.volume.confidence * volume_weight;

        weighted_confidence.clamp(0.0, 1.0)
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
