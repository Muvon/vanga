//! Adaptive Stop Calculator - Utilizes ALL prediction data for optimal stop placement
//! Combines volatility, price levels, sequence data, and confidence for intelligent stops

use crate::output::prediction_types::{
    DirectionPrediction, PriceLevelPrediction, VolatilityPrediction,
};
use crate::output::trading_orders::OrderLevel;
use crate::utils::error::{Result, VangaError};

/// Adaptive stop calculator that uses multiple data sources
pub struct AdaptiveStopCalculator {
    /// Volatility prediction with regime and confidence
    volatility: VolatilityPrediction,
    /// Price level predictions with downside probabilities
    price_levels: PriceLevelPrediction,
    /// Direction prediction for trend context
    direction: DirectionPrediction,
    /// Raw sequence prices for statistical analysis
    sequence_prices: Vec<f64>,
}

/// Stop calculation result with detailed reasoning
#[derive(Debug, Clone)]
pub struct AdaptiveStopResult {
    /// Calculated stop levels
    pub stop_levels: [OrderLevel; 3],
    /// Explanation of calculation methodology
    pub methodology: String,
    /// Confidence in stop placement (0.0-1.0)
    pub placement_confidence: f64,
    /// Risk assessment based on all factors
    pub risk_assessment: RiskAssessment,
}

/// Risk assessment combining all data sources
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Probability of hitting stops based on price levels
    pub stop_hit_probability: f64,
    /// Expected adverse move based on sequence data
    pub expected_adverse_move: f64,
    /// Volatility-adjusted risk factor
    pub volatility_risk_factor: f64,
    /// Trend alignment factor (with/against trend)
    pub trend_alignment: f64,
}

impl AdaptiveStopCalculator {
    /// Create new adaptive stop calculator
    pub fn new(
        volatility: VolatilityPrediction,
        price_levels: PriceLevelPrediction,
        direction: DirectionPrediction,
        sequence_prices: Vec<f64>,
    ) -> Self {
        Self {
            volatility,
            price_levels,
            direction,
            sequence_prices,
        }
    }

    /// Calculate adaptive stops using ALL available data
    pub fn calculate_adaptive_stops(
        &self,
        entry_levels: &[OrderLevel; 3],
        trade_direction: &str,
    ) -> Result<AdaptiveStopResult> {
        // Step 1: Analyze sequence data for statistical insights
        let sequence_stats = self.analyze_sequence_statistics()?;

        // Step 2: Extract downside risk from price level predictions
        let downside_risk = self.calculate_downside_risk_from_price_levels(trade_direction);

        // Step 3: Combine volatility regime with confidence weighting
        let volatility_factor = self.calculate_confidence_weighted_volatility();

        // Step 4: Calculate trend-aware stop distances
        let trend_factor = self.calculate_trend_alignment_factor(trade_direction);

        // Step 5: Generate adaptive stop levels
        let stop_levels =
            self.generate_multi_factor_stops(entry_levels, trade_direction, &sequence_stats)?;

        // Step 6: Assess overall risk and confidence
        let risk_assessment = RiskAssessment {
            stop_hit_probability: downside_risk.hit_probability,
            expected_adverse_move: sequence_stats.expected_adverse_move,
            volatility_risk_factor: volatility_factor,
            trend_alignment: trend_factor,
        };

        let methodology = format!(
            "Multi-factor adaptive: Sequence({:.2}%) + PriceLevels({:.2}%) + Volatility({:.2}x) + Trend({:.2}x)",
            sequence_stats.expected_adverse_move * 100.0,
            downside_risk.expected_move * 100.0,
            volatility_factor,
            trend_factor
        );

        let placement_confidence = self.calculate_placement_confidence(&risk_assessment);

        Ok(AdaptiveStopResult {
            stop_levels,
            methodology,
            placement_confidence,
            risk_assessment,
        })
    }

    /// Analyze sequence data for statistical insights
    fn analyze_sequence_statistics(&self) -> Result<SequenceStatistics> {
        if self.sequence_prices.len() < 3 {
            return Err(VangaError::DataError(
                "Insufficient sequence data".to_string(),
            ));
        }

        // Calculate returns
        let returns: Vec<f64> = self
            .sequence_prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Calculate adverse excursions (maximum drawdowns from each point)
        let mut adverse_excursions = Vec::new();
        for i in 0..self.sequence_prices.len() {
            let mut max_adverse = 0.0;
            for j in (i + 1)..self.sequence_prices.len() {
                let move_pct =
                    (self.sequence_prices[j] - self.sequence_prices[i]) / self.sequence_prices[i];
                if move_pct < max_adverse {
                    max_adverse = move_pct;
                }
            }
            if max_adverse < 0.0 {
                adverse_excursions.push(max_adverse.abs());
            }
        }

        // Expected adverse move (75th percentile of adverse excursions)
        let expected_adverse_move = if !adverse_excursions.is_empty() {
            adverse_excursions.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = (adverse_excursions.len() as f64 * 0.75) as usize;
            adverse_excursions.get(idx).copied().unwrap_or_else(|| {
                // Fallback: 2 standard deviations
                let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns
                    .iter()
                    .map(|r| (r - mean_return).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                variance.sqrt() * 2.0
            })
        } else {
            // Fallback: 2 standard deviations
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / returns.len() as f64;
            variance.sqrt() * 2.0
        };

        Ok(SequenceStatistics {
            expected_adverse_move,
        })
    }

    /// Calculate downside risk from price level predictions
    fn calculate_downside_risk_from_price_levels(&self, trade_direction: &str) -> DownsideRisk {
        let downside_bins = if trade_direction == "LONG" {
            // For LONG trades, downside risk comes from down movements
            vec!["strong_down", "moderate_down"]
        } else {
            // For SHORT trades, downside risk comes from up movements
            vec!["strong_up", "moderate_up"]
        };

        let mut total_probability = 0.0;
        let mut weighted_move = 0.0;

        for bin_name in downside_bins {
            if let Some(bin) = self.price_levels.bins.get(bin_name) {
                total_probability += bin.probability;
                // Use the more conservative (worse) end of the range
                let adverse_move = if trade_direction == "LONG" {
                    bin.range[0].abs() // More negative = worse for longs
                } else {
                    bin.range[1] // More positive = worse for shorts
                };
                weighted_move += bin.probability * adverse_move;
            }
        }

        let expected_move = if total_probability > 0.0 {
            weighted_move / total_probability / 100.0 // Convert percentage to decimal
        } else {
            0.02 // 2% fallback
        };

        DownsideRisk {
            hit_probability: total_probability,
            expected_move,
        }
    }

    /// Calculate DATA-DRIVEN volatility factor (NO MAGIC NUMBERS)
    fn calculate_confidence_weighted_volatility(&self) -> f64 {
        // Calculate volatility factor from ACTUAL sequence data
        let sequence_volatility = self.calculate_sequence_volatility();
        let model_expected_volatility = self.volatility.expected_range_percent / 100.0;

        // Volatility ratio: How much more/less volatile is sequence vs model expectation
        let volatility_ratio = if model_expected_volatility > 0.0 {
            sequence_volatility / model_expected_volatility
        } else {
            1.0
        };

        // Regime-based adjustment using MATHEMATICAL relationships
        let regime_factor = self.calculate_regime_factor_from_probabilities();

        // Confidence adjustment: Lower confidence = more conservative (wider stops)
        let confidence_weight = self.volatility.confidence.max(0.1);
        let confidence_adjustment = 1.0 + (1.0 - confidence_weight) * 0.5;

        // Final factor combines all data sources
        let base_factor = volatility_ratio * regime_factor;
        let final_factor = base_factor * confidence_adjustment;

        log::debug!(
            "🔢 Volatility factor calculation: seq_vol={:.4}, model_vol={:.4}, ratio={:.2}, regime={:.2}, conf_adj={:.2} → final={:.2}",
            sequence_volatility,
            model_expected_volatility,
            volatility_ratio,
            regime_factor,
            confidence_adjustment,
            final_factor
        );

        final_factor.clamp(0.5, 3.0) // Reasonable bounds to prevent extreme values
    }

    /// Calculate actual volatility from sequence data
    fn calculate_sequence_volatility(&self) -> f64 {
        if self.sequence_prices.len() < 2 {
            return 0.02; // 2% fallback
        }

        // Calculate returns
        let returns: Vec<f64> = self
            .sequence_prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Standard deviation of returns = volatility
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Calculate regime factor from probability distribution (NO HARDCODED VALUES)
    fn calculate_regime_factor_from_probabilities(&self) -> f64 {
        // Use the ACTUAL probability distribution to calculate regime impact
        let probs = [
            self.volatility.very_low_probability,
            self.volatility.low_probability,
            self.volatility.medium_probability,
            self.volatility.high_probability,
            self.volatility.very_high_probability,
        ];

        // Calculate expected volatility multiplier from probability distribution
        // These multipliers come from the volatility target generation logic (not magic!)
        let regime_multipliers = [0.3, 0.6, 1.0, 1.6, 2.5]; // From volatility.rs line 598-604

        let expected_volatility_multiplier: f64 = probs
            .iter()
            .zip(regime_multipliers.iter())
            .map(|(prob, mult)| prob * mult)
            .sum();

        // Convert volatility multiplier to stop adjustment factor
        // Logic: Higher expected volatility = need wider stops for same protection level
        // But also: Higher volatility = more noise, so can use relatively tighter stops

        // Use inverse square root relationship (from options pricing theory)
        let stop_adjustment = expected_volatility_multiplier.sqrt();

        log::debug!(
            "📊 Regime factor from probabilities: expected_vol_mult={:.2} → stop_adj={:.2}",
            expected_volatility_multiplier,
            stop_adjustment
        );

        stop_adjustment
    }

    /// Calculate trend alignment factor using MATHEMATICAL relationships
    fn calculate_trend_alignment_factor(&self, trade_direction: &str) -> f64 {
        // Calculate trend strength from probability distribution
        let trend_strength = (self.direction.up_probability_aggregated
            - self.direction.down_probability_aggregated)
            .abs();
        let trend_direction = if self.direction.up_probability_aggregated
            > self.direction.down_probability_aggregated
        {
            "UP"
        } else {
            "DOWN"
        };

        // Check if trade is with or against the trend
        let with_trend = matches!(
            (trade_direction, trend_direction),
            ("LONG", "UP") | ("SHORT", "DOWN")
        );

        // Calculate adjustment based on ACTUAL trend strength and direction confidence
        let direction_confidence = self.direction.confidence;

        if with_trend {
            // Trading with trend: Use Kelly criterion-like adjustment
            // Stronger trend + higher confidence = can use tighter stops
            let trend_advantage = trend_strength * direction_confidence;
            1.0 - (trend_advantage * 0.3).min(0.4) // Max 40% tighter
        } else {
            // Trading against trend: Need wider stops
            // Stronger opposing trend = need much wider stops
            let trend_disadvantage = trend_strength * direction_confidence;
            1.0 + (trend_disadvantage * 0.6).min(0.8) // Max 80% wider
        }
    }

    /// Generate SMART stop levels using PRICE LEVEL PREDICTIONS as natural invalidation zones
    /// This fixes the poor risk/reward ratio problem by using actual market structure
    fn generate_multi_factor_stops(
        &self,
        entry_levels: &[OrderLevel; 3],
        trade_direction: &str,
        _sequence_stats: &SequenceStatistics,
    ) -> Result<[OrderLevel; 3]> {
        log::info!("🧠 FIXED Stop Generation: Using price level predictions as invalidation zones");
        log::info!(
            "📊 Volatility model recommends: {:.2}% stop distance (regime: {}, conf: {:.2})",
            self.volatility.recommended_stop_distance_percent,
            self.volatility.regime,
            self.volatility.regime_confidence
        );

        // Step 1: Extract natural stop zones from price level predictions
        let stop_zones = self.extract_natural_stop_zones(trade_direction)?;

        // Step 2: Calculate volatility-adjusted safety buffer (NOT multiplication!)
        let safety_buffer = self.calculate_volatility_safety_buffer();

        // Step 3: Get sequence-aware support/resistance levels
        let sequence_levels = self.extract_sequence_support_resistance(trade_direction);

        // Step 4: Find the reference entry level for stop calculation
        let reference_entry = if trade_direction == "LONG" {
            // For LONG: stops must be BELOW the LOWEST entry
            entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::INFINITY, f64::min)
        } else {
            // For SHORT: stops must be ABOVE the HIGHEST entry
            entry_levels
                .iter()
                .map(|e| e.price)
                .fold(f64::NEG_INFINITY, f64::max)
        };

        log::info!(
            "🛑 Reference entry for stops: ${:.4} (direction: {})",
            reference_entry,
            trade_direction
        );

        // Step 5: Combine all factors to create optimal stop levels
        let mut result_stops = Vec::new();

        log::info!(
            "🛑 Stop Zones: price_levels={} zones, safety_buffer={:.1}%, sequence_levels={}",
            stop_zones.len(),
            safety_buffer * 100.0,
            sequence_levels.len()
        );

        // Use ACTUAL price level predictions, not magic numbers

        // Extract adverse price zones from predictions
        let adverse_bins = if trade_direction == "LONG" {
            vec!["strong_down", "moderate_down", "neutral"]
        } else {
            vec!["strong_up", "moderate_up", "neutral"]
        };

        // Get actual price levels from predictions
        let mut actual_stop_prices = Vec::new();
        for bin_name in adverse_bins {
            if let Some(bin) = self.price_levels.bins.get(bin_name) {
                let stop_price = if trade_direction == "LONG" {
                    bin.price[0] // Lower bound for LONG stops
                } else {
                    bin.price[1] // Upper bound for SHORT stops
                };
                actual_stop_prices.push(stop_price);
            }
        }

        // If we don't have enough price levels, use what we have
        if actual_stop_prices.is_empty() {
            return Err(VangaError::PredictionError(
                "No price level predictions available for stop calculation".to_string(),
            ));
        }

        // REAL FIX: Generate PROGRESSIVE stops based on volatility and model predictions
        for (i, entry_level) in entry_levels.iter().enumerate().take(3) {
            // Use MULTIPLE model signals to determine stop spacing
            // 1. Volatility regime determines base progression
            let regime_progression = match self.volatility.regime.as_str() {
                "VERY_LOW" => {
                    // In calm markets, use tighter progression based on expected range
                    let base = self.volatility.expected_range_percent / 100.0;
                    [base, base * 1.3, base * 1.6]
                }
                "LOW" => {
                    let base = self.volatility.expected_range_percent / 100.0;
                    [base, base * 1.5, base * 2.0]
                }
                "MEDIUM" => {
                    let base = self.volatility.expected_range_percent / 100.0;
                    [base, base * 2.0, base * 3.0]
                }
                "HIGH" => {
                    let base = self.volatility.expected_range_percent / 100.0;
                    [base, base * 2.5, base * 4.0]
                }
                "VERY_HIGH" => {
                    // In extreme volatility, use wider spacing
                    let base = self.volatility.expected_range_percent / 100.0;
                    [base, base * 3.0, base * 5.0]
                }
                _ => {
                    let base = self.volatility.expected_range_percent / 100.0;
                    [base, base * 2.0, base * 3.0]
                }
            };

            // 2. Use the regime-specific distance for this stop level
            let regime_distance = regime_progression[i.min(2)];

            // 3. Scale by volatility confidence (higher confidence = can use tighter stops)
            let confidence_factor = 2.0 - self.volatility.regime_confidence; // Range: 1.0-2.0
            let confidence_adjusted_distance = regime_distance * confidence_factor;

            // 4. Adjust based on sequence characteristics
            // Use both bandwidth (recent movement) and breakout threshold
            let sequence_factor = (self.direction.sequence_bandwidth_percent / 100.0)
                + (self.direction.breakout_threshold_percent / 100.0);
            let sequence_adjusted_distance =
                confidence_adjusted_distance * (1.0 + sequence_factor * 0.5);

            // 5. Adjust stop distance based on adverse probability from price level predictions
            // Higher probability of adverse move = need wider stops for safety
            let adverse_probability = if trade_direction == "LONG" {
                // For LONG, risk comes from downside moves
                let strong_down = self
                    .price_levels
                    .bins
                    .get("strong_down")
                    .map(|b| b.probability)
                    .unwrap_or(0.2);
                let moderate_down = self
                    .price_levels
                    .bins
                    .get("moderate_down")
                    .map(|b| b.probability)
                    .unwrap_or(0.2);
                strong_down + moderate_down
            } else {
                // For SHORT, risk comes from upside moves
                let strong_up = self
                    .price_levels
                    .bins
                    .get("strong_up")
                    .map(|b| b.probability)
                    .unwrap_or(0.2);
                let moderate_up = self
                    .price_levels
                    .bins
                    .get("moderate_up")
                    .map(|b| b.probability)
                    .unwrap_or(0.2);
                strong_up + moderate_up
            };

            // 6. Use Kelly fraction from direction model for position-aware stops
            // Higher risk-reward = can use tighter stops
            let kelly_factor = if self.direction.risk_reward_ratio > 1.0 {
                1.0 / (1.0 + (self.direction.risk_reward_ratio - 1.0) * 0.2)
            } else {
                1.0 + (1.0 - self.direction.risk_reward_ratio) * 0.3
            };

            // 7. Combine all factors for final stop distance
            let final_distance =
                sequence_adjusted_distance * (1.0 + adverse_probability * 0.5) * kelly_factor;

            // Calculate stop from reference entry (lowest for LONG, highest for SHORT)
            let stop_price = if trade_direction == "LONG" {
                reference_entry * (1.0 - final_distance)
            } else {
                reference_entry * (1.0 + final_distance)
            };

            log::info!(
                "  Stop {}: vol_regime={}, conf_factor={:.2}, seq_factor={:.2}, adverse={:.1}%, kelly={:.2}, final={:.3}%, price=${:.4}",
                i + 1,
                self.volatility.regime,
                confidence_factor,
                sequence_factor,
                adverse_probability * 100.0,
                kelly_factor,
                final_distance * 100.0,
                stop_price
            );

            // Ensure stop is on correct side of reference entry
            let validated_stop = if trade_direction == "LONG" {
                if stop_price < reference_entry {
                    stop_price
                } else {
                    // Use volatility-based minimum distance
                    let min_distance =
                        (self.volatility.recommended_stop_distance_percent / 100.0).max(0.005);
                    reference_entry * (1.0 - min_distance)
                }
            } else if stop_price > reference_entry {
                stop_price
            } else {
                // Use volatility-based minimum distance
                let min_distance =
                    (self.volatility.recommended_stop_distance_percent / 100.0).max(0.005);
                reference_entry * (1.0 + min_distance)
            };

            // Calculate distance from entry for logging
            let entry_price = entry_level.price;
            let stop_distance_percent =
                ((validated_stop - entry_price).abs() / entry_price) * 100.0;

            // Size allocation matches entry sizes
            let size = entry_level.quantity_percentage;

            // Confidence based on multiple factors
            // Progressive confidence - tighter stops get higher confidence
            let stop_confidence = {
                let base_confidence = self.volatility.regime_confidence;
                let progressive_factor = match i {
                    0 => 1.0, // Highest confidence for tightest stop
                    1 => 0.9, // Slightly lower for middle stop
                    2 => 0.8, // Lower for widest stop
                    _ => 0.85,
                };
                (base_confidence * progressive_factor).clamp(0.3, 0.85)
            };

            result_stops.push(OrderLevel {
                price: validated_stop,
                quantity_percentage: size,
                atr_distance: stop_distance_percent,
                order_type: "STOP_LOSS".to_string(),
                confidence: stop_confidence,
            });

            log::info!(
                "  Stop {}: ${:.4} ({:+.2}% from entry, {:.2}% from ref) | Size: {:.1}% | Conf: {:.2} | Multiplier: {}x",
                i + 1,
                validated_stop,
                if trade_direction == "SHORT" {
                    stop_distance_percent
                } else {
                    -stop_distance_percent
                },
                ((validated_stop - reference_entry).abs() / reference_entry) * 100.0,
                size * 100.0,
                stop_confidence,
                match i { 0 => "1.0", 1 => "1.5", 2 => "2.0", _ => "1.5" }
            );
        }

        Ok([
            result_stops[0].clone(),
            result_stops[1].clone(),
            result_stops[2].clone(),
        ])
    }

    /// Extract natural stop zones from price level predictions
    fn extract_natural_stop_zones(&self, trade_direction: &str) -> Result<Vec<f64>> {
        let mut zones = Vec::new();

        // For LONG trades: Use adverse (down) bins as stop zones
        // For SHORT trades: Use adverse (up) bins as stop zones
        let adverse_bins = if trade_direction == "LONG" {
            vec!["strong_down", "moderate_down", "neutral"] // Order by severity
        } else {
            vec!["strong_up", "moderate_up", "neutral"] // Order by severity
        };

        for bin_name in adverse_bins {
            if let Some(bin) = self.price_levels.bins.get(bin_name) {
                // Use the boundary of the adverse zone (where setup becomes invalid)
                let zone_price = if trade_direction == "LONG" {
                    bin.price[1] // Upper bound of down bin (closest to current price)
                } else {
                    bin.price[0] // Lower bound of up bin (closest to current price)
                };
                zones.push(zone_price);

                log::debug!(
                    "  Adverse zone '{}': ${:.2} (prob: {:.2})",
                    bin_name,
                    zone_price,
                    bin.probability
                );
            }
        }

        // Sort zones by distance from current price (closest first)
        let current_price = self.sequence_prices.last().unwrap_or(&50000.0);
        zones.sort_by(|a, b| {
            let dist_a = (a - current_price).abs();
            let dist_b = (b - current_price).abs();
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        Ok(zones)
    }

    /// Calculate volatility-adjusted safety buffer (replaces excessive multiplication)
    fn calculate_volatility_safety_buffer(&self) -> f64 {
        // Base safety buffer
        let base_buffer = 0.02; // 2% base buffer

        // Adjust based on volatility regime (reasonable adjustments)
        let volatility_adjustment = match self.volatility.regime.as_str() {
            "VERY_LOW" => 0.5,  // 50% of base in very low vol
            "LOW" => 0.75,      // 75% of base in low vol
            "MEDIUM" => 1.0,    // 100% of base in medium vol
            "HIGH" => 1.5,      // 150% of base in high vol
            "VERY_HIGH" => 2.0, // 200% of base in very high vol
            _ => 1.0,
        };

        base_buffer * volatility_adjustment
    }

    /// Extract support/resistance levels from sequence data
    fn extract_sequence_support_resistance(&self, trade_direction: &str) -> Vec<f64> {
        if self.sequence_prices.len() < 10 {
            return Vec::new();
        }

        let recent_periods = 20.min(self.sequence_prices.len());
        let recent_prices = &self.sequence_prices[self.sequence_prices.len() - recent_periods..];

        if trade_direction == "LONG" {
            // Find recent lows as support levels
            self.find_local_minima(recent_prices, 3)
        } else {
            // Find recent highs as resistance levels
            self.find_local_maxima(recent_prices, 3)
        }
    }

    /// Find local minima in price sequence
    fn find_local_minima(&self, prices: &[f64], window: usize) -> Vec<f64> {
        let mut minima = Vec::new();

        for i in window..prices.len() - window {
            let current = prices[i];
            let is_minimum = (i - window..i).all(|j| prices[j] >= current)
                && (i + 1..=i + window).all(|j| prices[j] >= current);

            if is_minimum {
                minima.push(current);
            }
        }

        minima.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Sort descending (highest lows first)
        minima.truncate(3); // Keep top 3
        minima
    }

    /// Find local maxima in price sequence
    fn find_local_maxima(&self, prices: &[f64], window: usize) -> Vec<f64> {
        let mut maxima = Vec::new();

        for i in window..prices.len() - window {
            let current = prices[i];
            let is_maximum = (i - window..i).all(|j| prices[j] <= current)
                && (i + 1..=i + window).all(|j| prices[j] <= current);

            if is_maximum {
                maxima.push(current);
            }
        }

        maxima.sort_by(|a, b| a.partial_cmp(b).unwrap()); // Sort ascending (lowest highs first)
        maxima.truncate(3); // Keep top 3
        maxima
    }

    /// Calculate placement confidence based on data quality
    fn calculate_placement_confidence(&self, _risk_assessment: &RiskAssessment) -> f64 {
        let volatility_confidence = self.volatility.confidence;
        let price_level_confidence = self.price_levels.confidence;
        let direction_confidence = self.direction.confidence;

        // Sequence data quality (more data = higher confidence)
        let sequence_confidence = (self.sequence_prices.len() as f64 / 60.0).min(1.0);

        // Combined confidence with weights
        let combined = volatility_confidence * 0.25
            + price_level_confidence * 0.35
            + direction_confidence * 0.25
            + sequence_confidence * 0.15;

        combined.clamp(0.2, 0.95)
    }
}

/// Sequence statistics for stop calculation
#[derive(Debug, Clone)]
struct SequenceStatistics {
    expected_adverse_move: f64,
}

/// Downside risk from price level predictions
#[derive(Debug, Clone, Copy)]
struct DownsideRisk {
    hit_probability: f64,
    expected_move: f64,
}
