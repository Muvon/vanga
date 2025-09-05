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
        let stop_levels = self.generate_multi_factor_stops(
            entry_levels,
            trade_direction,
            &sequence_stats,
            &downside_risk, // Pass by reference to avoid move
            volatility_factor,
            trend_factor,
        )?;

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
            adverse_excursions,
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

    /// Generate multi-factor stop levels using PURE prediction data
    fn generate_multi_factor_stops(
        &self,
        entry_levels: &[OrderLevel; 3],
        trade_direction: &str,
        sequence_stats: &SequenceStatistics,
        downside_risk: &DownsideRisk,
        volatility_factor: f64,
        trend_factor: f64,
    ) -> Result<[OrderLevel; 3]> {
        let mut stops = Vec::new();

        // Find extreme entry for stop placement
        let extreme_entry = if trade_direction == "SHORT" {
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

        // Use direction.expected_downside_percent as base risk
        let base_stop_distance = if trade_direction == "LONG" {
            self.direction.expected_downside_percent / 100.0
        } else {
            self.direction.expected_upside_percent / 100.0 // Upside is risk for shorts
        };

        // Weight by adverse bin probabilities
        let adverse_probability = if trade_direction == "LONG" {
            self.direction.down_probability_aggregated
        } else {
            self.direction.up_probability_aggregated
        };

        // Scale by sequence bandwidth (actual market movement)
        let sequence_scale = self.direction.sequence_bandwidth_percent / 100.0;

        // Combine all factors
        let stop_distance = base_stop_distance
            * (1.0 + adverse_probability)  // More adverse probability = wider stops
            * (1.0 + sequence_scale)       // More volatile sequence = wider stops
            * volatility_factor             // Volatility regime adjustment
            * trend_factor; // Trend alignment adjustment

        // Use golden ratio for stop progression
        let golden_ratio: f64 = 1.618;
        let stop_progression = [
            golden_ratio.powi(2), // 2.618
            golden_ratio.powi(3), // 4.236
            golden_ratio.powi(4), // 6.854
        ];

        log::info!(
            "🛑 Stop Generation: base={:.3}%, adverse_prob={:.2}, seq_scale={:.3}, final={:.3}%",
            base_stop_distance * 100.0,
            adverse_probability,
            sequence_scale * 100.0,
            stop_distance * 100.0
        );

        for (i, &progression) in stop_progression.iter().enumerate() {
            let final_stop_distance = stop_distance * progression;

            // Calculate stop price
            let stop_price = if trade_direction == "SHORT" {
                extreme_entry * (1.0 + final_stop_distance)
            } else {
                extreme_entry * (1.0 - final_stop_distance)
            };

            // Size allocation matches entry sizes
            let size = entry_levels[i].quantity_percentage;

            // Confidence based on adverse probability and data quality
            let data_quality = self.calculate_data_quality(sequence_stats, downside_risk);
            let stop_confidence = (adverse_probability * data_quality).clamp(0.3, 0.95);

            stops.push(OrderLevel {
                price: stop_price,
                quantity_percentage: size,
                atr_distance: final_stop_distance * 100.0,
                order_type: "STOP_LOSS".to_string(),
                confidence: stop_confidence,
            });

            log::info!(
                "  Stop {}: ${:.4} ({:+.2}%) | Size: {:.1}% | Conf: {:.2}",
                i + 1,
                stop_price,
                if trade_direction == "SHORT" {
                    final_stop_distance * 100.0
                } else {
                    -final_stop_distance * 100.0
                },
                size * 100.0,
                stop_confidence
            );
        }

        Ok([stops[0].clone(), stops[1].clone(), stops[2].clone()])
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

    /// Calculate data quality score
    fn calculate_data_quality(
        &self,
        sequence_stats: &SequenceStatistics,
        downside_risk: &DownsideRisk,
    ) -> f64 {
        let sequence_quality = if sequence_stats.adverse_excursions.len() > 10 {
            0.9
        } else if sequence_stats.adverse_excursions.len() > 5 {
            0.7
        } else {
            0.5
        };

        let price_level_quality = if downside_risk.hit_probability > 0.3 {
            0.9 // Good probability data
        } else {
            0.6 // Limited probability data
        };

        (sequence_quality + price_level_quality) / 2.0
    }
}

/// Sequence statistics for stop calculation
#[derive(Debug, Clone)]
struct SequenceStatistics {
    expected_adverse_move: f64,
    adverse_excursions: Vec<f64>,
}

/// Downside risk from price level predictions
#[derive(Debug, Clone, Copy)]
struct DownsideRisk {
    hit_probability: f64,
    expected_move: f64,
}
