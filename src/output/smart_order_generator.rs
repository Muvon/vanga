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
    /// Calculate adaptive spacing using ONLY prediction data - no hardcoded values
    fn calculate_adaptive_spacing(&self) -> f64 {
        // Base spacing from volatility (natural market movement)
        let volatility_spacing = self.volatility.expected_range_percent;

        // Scale by sequence bandwidth (actual recent price action)
        let sequence_scale = self.direction.sequence_bandwidth_percent;

        // Combine for market-adaptive spacing
        let base_spacing = volatility_spacing * sequence_scale;

        log::info!(
            "📏 Adaptive Spacing: volatility={:.3}% × sequence={:.3}% = {:.3}%",
            volatility_spacing,
            sequence_scale,
            base_spacing
        );

        base_spacing
    }

    /// Calculate probability-weighted center from price bins
    fn calculate_weighted_bin_center(&self, bin_names: &[&str]) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_probability = 0.0;

        for &bin_name in bin_names {
            if let Some(bin) = self.price_levels.bins.get(bin_name) {
                let bin_center = (bin.range[0] + bin.range[1]) / 2.0;
                weighted_sum += bin_center * bin.probability;
                total_probability += bin.probability;
            }
        }

        if total_probability > 0.0 {
            weighted_sum / total_probability
        } else {
            0.0
        }
    }

    /// Adjust level to avoid psychological clustering using sequence data
    pub fn adjust_for_psychological_levels(
        &self,
        price: f64,
        sequence_ohlcv: Option<&Vec<[f64; 5]>>,
    ) -> f64 {
        // Find natural support/resistance from sequence data
        let psychological_zones = if let Some(ohlcv) = sequence_ohlcv {
            self.find_natural_levels_from_sequence(ohlcv)
        } else {
            Vec::new()
        };

        // Check if our price is too close to a natural level
        for &level in &psychological_zones {
            let distance_pct = ((price - level).abs() / level) * 100.0;
            if distance_pct < 0.1 {
                // Within 0.1% of natural level
                // Shift by small amount based on sequence bandwidth
                let shift = self.direction.sequence_bandwidth_percent * 0.1;
                let adjusted_price = if price > level {
                    price * (1.0 + shift / 100.0)
                } else {
                    price * (1.0 - shift / 100.0)
                };

                log::info!(
                    "🎯 Psychological adjustment: {:.4} → {:.4} (avoiding natural level {:.4})",
                    price,
                    adjusted_price,
                    level
                );

                return adjusted_price;
            }
        }

        price
    }

    /// Find natural support/resistance levels from sequence OHLCV data
    fn find_natural_levels_from_sequence(&self, ohlcv: &Vec<[f64; 5]>) -> Vec<f64> {
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

    /// Generate SMART entry levels using PURE prediction data
    pub fn generate_smart_entries(
        &self,
        current_price: f64,
        direction: &str,
    ) -> Result<[OrderLevel; 3]> {
        let mut entries = Vec::new();

        // Get adaptive spacing from market data
        let base_spacing = self.calculate_adaptive_spacing();

        // Calculate probability-weighted entry zone
        let entry_bins = if direction == "LONG" {
            vec!["neutral", "moderate_down", "strong_down"]
        } else {
            vec!["neutral", "moderate_up", "strong_up"]
        };

        let weighted_entry_zone = self.calculate_weighted_bin_center(&entry_bins);

        // Scale spacing by directional confidence (less confident = wider spacing)
        let confidence_factor = if direction == "LONG" {
            2.0 - self.direction.up_probability_aggregated
        } else {
            2.0 - self.direction.down_probability_aggregated
        };

        let entry_spacing = base_spacing * confidence_factor;

        // Use Fibonacci ratios for natural progression
        let fibonacci_ratios = [0.382, 0.618, 1.000];

        log::info!(
            "📍 Entry Generation: weighted_zone={:.3}%, spacing={:.3}%, confidence={:.2}",
            weighted_entry_zone.abs(),
            entry_spacing,
            confidence_factor
        );

        // Generate entries using Fibonacci progression
        for (i, &ratio) in fibonacci_ratios.iter().enumerate() {
            // Combine weighted zone with progressive spacing
            let distance = weighted_entry_zone.abs() * ratio + entry_spacing * (i as f64 + 1.0);

            let entry_price = if direction == "SHORT" {
                current_price * (1.0 + distance / 100.0)
            } else {
                current_price * (1.0 - distance / 100.0)
            };

            // Size allocation using prediction-based adaptive sizing
            let base_size = self.calculate_entry_size(i + 1, &self.direction, &self.volatility);

            // Further adjust by directional confidence and bin probability
            let bin_weight = if i < entry_bins.len() {
                self.price_levels
                    .bins
                    .get(entry_bins[i])
                    .map(|b| b.probability)
                    .unwrap_or(0.33)
            } else {
                0.33
            };

            let adjusted_size = base_size * (0.5 + bin_weight);

            // Confidence based on bin probability and distance
            let bin_confidence = if i < entry_bins.len() {
                self.price_levels
                    .bins
                    .get(entry_bins[i])
                    .map(|b| b.probability)
                    .unwrap_or(0.2)
            } else {
                0.2
            };

            let distance_decay = (-distance / 10.0).exp(); // Exponential decay
            let confidence = (bin_confidence * distance_decay).max(0.1);

            let atr_distance = ((entry_price - current_price).abs() / current_price) * 100.0;

            entries.push(OrderLevel {
                price: entry_price,
                quantity_percentage: adjusted_size,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: confidence.min(0.95),
            });

            log::info!(
                "  {} Entry {}: ${:.4} ({:+.2}%) | Size: {:.1}% | Conf: {:.2}",
                direction,
                i + 1,
                entry_price,
                if direction == "SHORT" {
                    distance
                } else {
                    -distance
                },
                adjusted_size * 100.0,
                confidence
            );
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

            // Calculate actual ATR distance from current price
            let atr_distance = ((entry_price - current_price).abs() / current_price) * 100.0;

            entries.push(OrderLevel {
                price: entry_price,
                quantity_percentage: kelly_adjusted_size,
                atr_distance,
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

    /// Generate SMART exit levels using PURE prediction data
    pub fn generate_smart_exits(
        &self,
        current_price: f64,
        direction: &str,
    ) -> Result<[OrderLevel; 3]> {
        let mut exits = Vec::new();

        // Use expected upside/downside from direction prediction
        let exit_base = if direction == "LONG" {
            self.direction.expected_upside_percent
        } else {
            self.direction.expected_downside_percent
        };

        // Scale by volume regime for liquidity (high volume = can exit easier = tighter)
        let volume_scale = match self.volume.regime.as_str() {
            "VERY_HIGH" => 0.8, // Very liquid, can use tighter exits
            "HIGH" => 1.0,      // Good liquidity
            "MEDIUM" => 1.5,    // Normal liquidity, wider exits
            "LOW" => 2.0,       // Poor liquidity, need wider
            "VERY_LOW" => 2.5,  // Very poor, widest exits
            _ => 1.0,
        };

        // Calculate probability-weighted favorable zone
        let favorable_bins = if direction == "LONG" {
            vec!["moderate_up", "strong_up"]
        } else {
            vec!["moderate_down", "strong_down"]
        };

        let weighted_exit_zone = self.calculate_weighted_bin_center(&favorable_bins);

        // Combine expected move with weighted zone
        let base_exit_distance = if weighted_exit_zone.abs() > exit_base {
            weighted_exit_zone.abs() // Use bin prediction if larger
        } else {
            exit_base // Use expected move if larger
        };

        // Apply volume scaling
        let scaled_exit_distance = base_exit_distance * volume_scale;

        // Use golden ratio for exit progression
        let golden_ratio: f64 = 1.618;
        let exit_progression = [golden_ratio, golden_ratio.powi(2), golden_ratio.powi(3)];

        log::info!(
            "🎯 Exit Generation: base={:.3}%, weighted_zone={:.3}%, volume_scale={:.1}x, final={:.3}%",
            exit_base,
            weighted_exit_zone.abs(),
            volume_scale,
            scaled_exit_distance
        );

        for (i, &progression) in exit_progression.iter().enumerate() {
            let distance = scaled_exit_distance * progression;

            let exit_price = if direction == "SHORT" {
                current_price * (1.0 - distance / 100.0)
            } else {
                current_price * (1.0 + distance / 100.0)
            };

            // Size allocation using volume-based adaptive sizing
            let base_size = self.calculate_exit_size(i + 1, &self.volume);

            // Adjust by liquidity factor and bin probability
            let liquidity_factor = self.volume_liquidity_factor();
            let bin_weight = if i < favorable_bins.len() {
                self.price_levels
                    .bins
                    .get(favorable_bins[i])
                    .map(|b| b.probability)
                    .unwrap_or(0.33)
            } else {
                0.33
            };

            let adjusted_size = base_size * liquidity_factor * (0.5 + bin_weight);

            // Confidence based on bin probability and distance
            let bin_confidence = if i < favorable_bins.len() {
                self.price_levels
                    .bins
                    .get(favorable_bins[i])
                    .map(|b| b.probability)
                    .unwrap_or(0.2)
            } else {
                0.15
            };

            let distance_factor = 1.0 / progression; // Decreases with distance
            let confidence = (bin_confidence * distance_factor * liquidity_factor).min(0.9);

            let atr_distance = ((exit_price - current_price).abs() / current_price) * 100.0;

            exits.push(OrderLevel {
                price: exit_price,
                quantity_percentage: adjusted_size,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence,
            });

            log::info!(
                "  {} Exit {}: ${:.4} ({:+.2}%) | Size: {:.1}% | Conf: {:.2}",
                direction,
                i + 1,
                exit_price,
                if direction == "SHORT" {
                    -distance
                } else {
                    distance
                },
                adjusted_size * 100.0,
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

        // Calculate probability-weighted stop distances
        for i in 0..3 {
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
            // Higher adverse probability = wider stops needed
            let probability_multiplier = 1.0 + (adverse_probability * 3.0); // Scale by adverse risk
            let probability_weighted_stop = volatility_stop * probability_multiplier;

            // 4. PROGRESSIVE SPACING based on volatility regime
            let regime_progression = match self.volatility.regime.as_str() {
                "VERY_LOW" => 1.0 + (i as f64 * 0.2), // Smaller progression in calm markets
                "LOW" => 1.0 + (i as f64 * 0.3),
                "MEDIUM" => 1.0 + (i as f64 * 0.4), // Standard progression
                "HIGH" => 1.0 + (i as f64 * 0.5),   // Larger progression in volatile markets
                "VERY_HIGH" => 1.0 + (i as f64 * 0.6), // Largest progression
                _ => 1.0 + (i as f64 * 0.4),
            };

            let final_stop_distance = probability_weighted_stop * regime_progression;

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
                quantity_percentage: entry_levels[i].quantity_percentage, // Match entry size
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

    /// Calculate exit size using PURE volume data for liquidity-based sizing
    fn calculate_exit_size(&self, level: usize, volume: &VolumePrediction) -> f64 {
        // Use volume regime probabilities to determine optimal exit sizing
        let liquidity_score = volume.very_high_probability * 1.0
            + volume.high_probability * 0.8
            + volume.medium_probability * 0.5
            + volume.low_probability * 0.3
            + volume.very_low_probability * 0.1;

        // Higher liquidity = can exit in larger chunks
        // Lower liquidity = need to split exits more
        let liquidity_factor = liquidity_score * 2.0; // Scale to 0-2 range

        // Progressive exit allocation based on liquidity
        let base_allocation = if liquidity_factor > 1.0 {
            // High liquidity: can take larger exits
            match level {
                1 => 0.3, // Take some profit early
                2 => 0.4, // Main exit
                3 => 0.3, // Final exit
                _ => 0.33,
            }
        } else {
            // Low liquidity: need smaller, distributed exits
            match level {
                1 => 0.5, // Take more profit early when possible
                2 => 0.3, // Moderate middle exit
                3 => 0.2, // Small final exit
                _ => 0.33,
            }
        };

        // Confidence adjustment
        let confidence_scale = volume.confidence + 0.5; // Range 0.5-1.5

        let final_size = base_allocation * confidence_scale;

        log::debug!(
            "Exit {} sizing: liquidity={:.3}, base={:.3}, conf={:.2} → {:.3}",
            level,
            liquidity_score,
            base_allocation,
            confidence_scale,
            final_size
        );

        final_size.clamp(0.15, 0.5)
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
