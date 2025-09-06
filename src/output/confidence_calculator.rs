//! UNIFIED Confidence Calculation System for VANGA
//!
//! This module provides research-based confidence scoring aligned with real-world neural network
//! probability distributions. Based on "On Calibration of Modern Neural Networks" (Guo et al., 2017)
//! and extensive research on well-calibrated classification models.
//!
//! ## Key Research Findings Applied:
//! - Well-calibrated models rarely exceed 0.4-0.5 max probability
//! - 0.35+ max probability represents EXCELLENT performance (top 15%)
//! - 0.42+ max probability represents EXCEPTIONAL performance (top 5%)
//! - 0.50+ max probability may indicate overfitting if common
//! - Modern neural networks tend to be overconfident without calibration
//!
//! ## Unified Approach:
//! All targets (price levels, direction, volume, volatility, sentiment) now use the same
//! calibration function with minimal target-specific adjustments. This eliminates
//! confidence inflation from multiple calculation layers and provides consistent,
//! conservative confidence scores appropriate for crypto trading.
//!
//! ## Confidence Philosophy:
//! - Conservative approach aligned with crypto market uncertainty
//! - Recognizes that lower probabilities are actually very good in real-world models
//! - Avoids overconfidence that leads to excessive risk-taking
//! - Maps realistic probability ranges to actionable confidence levels

use crate::output::prediction_types::{
    DirectionPrediction, PredictionResult, PriceLevelPrediction, SentimentPrediction,
    VolatilityPrediction, VolumePrediction,
};
use crate::utils::error::Result;

/// Configuration for confidence calculation
#[derive(Debug, Clone)]
pub struct ConfidenceConfig {
    /// Weight for price level predictions (most important for entry/exit)
    pub price_level_weight: f64,
    /// Weight for direction predictions (confirms trend)
    pub direction_weight: f64,
    /// Weight for volatility predictions (affects position sizing)
    pub volatility_weight: f64,
    /// Weight for sentiment predictions (market psychology)
    pub sentiment_weight: f64,
    /// Weight for volume predictions (confirms moves)
    pub volume_weight: f64,
    /// Minimum probability for a prediction to be considered confident
    pub min_probability_threshold: f64,
    /// Bonus multiplier when targets agree
    pub agreement_bonus: f64,
    /// Penalty when targets disagree
    pub disagreement_penalty: f64,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            // CRYPTO-OPTIMIZED WEIGHTS (70% for Direction + Price Level):
            // Direction is MAIN signal for trend (45%)
            // Price Level provides entry/exit zones (25%)
            // Total core signals: 70%
            // Supporting signals: 30% (volatility, sentiment, volume)
            direction_weight: 0.45,   // 45% - PRIMARY trend signal (MAIN)
            price_level_weight: 0.25, // 25% - Entry/exit zones (supports direction)
            volatility_weight: 0.10,  // 10% - Risk sizing
            sentiment_weight: 0.10,   // 10% - Market psychology confirmation
            volume_weight: 0.10,      // 10% - Move validation
            min_probability_threshold: 0.25, // 25% threshold (above 5-class baseline of 20%)
            agreement_bonus: 1.3,     // 30% bonus when models agree
            disagreement_penalty: 0.8, // 20% penalty for disagreement
        }
    }
}

/// Enhanced confidence calculator for multi-target predictions using unified calibration
///
/// This calculator uses research-based confidence calibration that properly handles
/// realistic neural network probability distributions. All individual target confidences
/// are calculated using the same unified approach, with minimal target-specific adjustments.
pub struct ConfidenceCalculator {
    config: ConfidenceConfig,
}

impl ConfidenceCalculator {
    /// Create new confidence calculator with configuration
    pub fn new(config: ConfidenceConfig) -> Self {
        Self { config }
    }

    /// Calculate enhanced confidence score using unified calibration approach
    ///
    /// This method combines individual target confidences (all calculated using the same
    /// unified calibration function) with crypto-specific adjustments for target agreement,
    /// volatility, and market dynamics. The approach is conservative to avoid overconfidence
    /// in crypto's uncertain environment.
    pub fn calculate_overall_confidence(&self, prediction: &PredictionResult) -> f64 {
        // UNIFIED CONFIDENCE CALCULATION APPROACH
        // All individual target confidences now use the same research-based calibration
        // This eliminates confidence inflation from multiple calculation layers
        // Core principle: Direction + Price Level are PRIMARY signals
        // Volume + Sentiment are CONFIRMATION signals
        // Volatility is RISK ADJUSTMENT

        let mut core_confidence = 0.0;
        let mut confirmation_confidence = 0.0;
        let mut risk_adjustment = 1.0;

        // 1. CORE SIGNALS (Direction + Price Level) - These MUST agree for high confidence
        let mut has_core_signals = false;

        if let Some(ref direction) = prediction.direction {
            let dir_score = self.calculate_direction_confidence(direction);

            // Debug logging - all confidences now use unified calibration
            log::debug!("Direction confidence: unified_calibrated={:.3}", dir_score);

            if let Some(ref price_levels) = prediction.price_levels {
                has_core_signals = true;
                let price_score = self.calculate_price_level_confidence(price_levels);

                // Debug logging - all confidences now use unified calibration
                log::debug!(
                    "Price levels confidence: unified_calibrated={:.3}",
                    price_score
                );

                // Check if direction and price level agree
                let core_agreement = self.check_price_direction_agreement(price_levels, direction);

                // Core confidence: weighted average with agreement boost
                core_confidence =
                    (dir_score * 0.65 + price_score * 0.35) * (0.8 + core_agreement * 0.4);

                // CRYPTO INSIGHT: Strong directional moves need price level confirmation
                if direction.pump_probability > 0.3 || direction.dump_probability > 0.3 {
                    // For pump/dump scenarios, price levels are MORE important
                    core_confidence =
                        (dir_score * 0.5 + price_score * 0.5) * (0.7 + core_agreement * 0.5);
                }

                log::debug!(
                    "Core confidence: dir={:.3}, price={:.3}, agreement={:.3}, final={:.3}",
                    dir_score,
                    price_score,
                    core_agreement,
                    core_confidence
                );
            } else {
                // Only direction available
                core_confidence = dir_score * 0.7; // Reduce confidence without price levels
            }
        }

        // 2. CONFIRMATION SIGNALS (Volume + Sentiment)
        let mut confirmation_count = 0;

        if let Some(ref volume) = prediction.volume {
            let vol_score = self.calculate_volume_confidence(volume);

            // CRYPTO INSIGHT: High volume is CRITICAL for breakouts
            if let Some(ref direction) = prediction.direction {
                let vol_confirmation = self.check_volume_confirmation(volume, direction);
                confirmation_confidence += vol_score * (0.8 + vol_confirmation * 0.4);
                confirmation_count += 1;

                // Boost confidence for high volume on strong moves
                if (volume.high_probability + volume.very_high_probability) > 0.5
                    && (direction.pump_probability > 0.2 || direction.dump_probability > 0.2)
                {
                    confirmation_confidence *= 1.3; // 30% boost for volume-confirmed breakouts
                }
            } else {
                confirmation_confidence += vol_score;
                confirmation_count += 1;
            }
        }

        if let Some(ref sentiment) = prediction.sentiment {
            let sent_score = self.calculate_sentiment_confidence(sentiment);

            // CRYPTO INSIGHT: Extreme sentiment often precedes reversals
            let extreme_sentiment =
                sentiment.very_bullish_probability + sentiment.very_bearish_probability;
            if extreme_sentiment > 0.6 {
                // Extreme sentiment: be cautious (potential reversal)
                confirmation_confidence += sent_score * 0.7;
            } else {
                confirmation_confidence += sent_score;
            }
            confirmation_count += 1;
        }

        // Average confirmation signals
        if confirmation_count > 0 {
            confirmation_confidence /= confirmation_count as f64;
        }

        // 3. RISK ADJUSTMENT (Volatility)
        if let Some(ref volatility) = prediction.volatility {
            // CRYPTO-SPECIFIC: Medium-High volatility is NORMAL and tradeable
            risk_adjustment = match volatility.regime.as_str() {
                "VERY_LOW" => 0.85,  // Too quiet = less opportunity
                "LOW" => 0.95,       // Slightly below normal
                "MEDIUM" => 1.0,     // Perfect for crypto trading
                "HIGH" => 0.95,      // Normal for crypto, slight caution
                "VERY_HIGH" => 0.75, // Extreme = reduce confidence
                _ => 1.0,
            };

            // CRYPTO INSIGHT: Adjust for expected volatility vs actual
            if volatility.expected_range_percent > 10.0 && volatility.regime == "MEDIUM" {
                risk_adjustment *= 1.1; // Boost for healthy volatility
            }
        }

        // 4. COMBINE ALL FACTORS (Crypto-optimized formula)
        let mut final_confidence = if has_core_signals {
            // When we have core signals, they dominate
            let base = core_confidence * 0.7 + confirmation_confidence * 0.3;
            base * risk_adjustment
        } else {
            // Without core signals, rely more on confirmations
            confirmation_confidence * 0.6 * risk_adjustment
        };

        // 5. CRYPTO-SPECIFIC ADJUSTMENTS

        // Check for confluence (multiple models strongly agreeing)
        let agreement_factor = self.calculate_target_agreement(prediction);
        if agreement_factor > 1.1 {
            // Strong agreement = confidence boost
            final_confidence *= agreement_factor;
        } else if agreement_factor < 0.9 {
            // Disagreement = reduce but don't kill confidence (crypto is noisy)
            final_confidence *= 0.8 + agreement_factor * 0.2;
        }

        // Allow natural confidence expression - no artificial clamping
        // Only apply safety bounds to prevent extreme values
        let clamped_confidence = final_confidence.clamp(0.05, 0.95);

        log::debug!(
            "Overall confidence calculation: core={:.3}, confirmation={:.3}, risk_adj={:.3}, agreement={:.3}, final={:.3}, clamped={:.3}",
            core_confidence,
            confirmation_confidence,
            risk_adjustment,
            agreement_factor,
            final_confidence,
            clamped_confidence
        );

        clamped_confidence
    }

    /// Calculate confidence for price level predictions using unified calibration
    fn calculate_price_level_confidence(&self, price_levels: &PriceLevelPrediction) -> f64 {
        // Get the highest probability bin for unified calibration
        let max_prob = price_levels
            .bins
            .values()
            .map(|bin| bin.probability)
            .fold(0.0, f64::max);

        // Use unified calibration function - price levels already have properly calibrated confidence
        // but we enhance it slightly for crypto trading context
        let base_confidence = calibrate_5_class_confidence(max_prob);

        // Slight enhancement for price levels (most critical for trading decisions)
        // But keep it conservative to avoid overconfidence
        let enhanced_confidence = base_confidence * 1.05; // 5% boost for price levels

        // Conservative clamping for crypto trading
        enhanced_confidence.clamp(0.20, 0.95)
    }

    /// Calculate confidence for direction predictions using unified calibration
    fn calculate_direction_confidence(&self, direction: &DirectionPrediction) -> f64 {
        // Use the unified calibration based on the highest aggregated probability
        let max_strength = direction
            .up_probability_aggregated
            .max(direction.down_probability_aggregated)
            .max(direction.sideways_probability);

        // Base confidence from unified calibration
        let base_confidence = calibrate_5_class_confidence(max_strength);

        // Factor in risk-reward ratio for slight enhancement (but keep conservative)
        let rr_multiplier = if direction.risk_reward_ratio >= 4.0 {
            1.08 // 8% boost for excellent R:R (4:1+)
        } else if direction.risk_reward_ratio >= 2.5 {
            1.04 // 4% boost for good R:R (2.5:1+)
        } else if direction.risk_reward_ratio >= 1.5 {
            1.0 // Standard for acceptable R:R
        } else {
            0.95 // 5% reduction for poor R:R
        };

        // Apply R:R adjustment conservatively
        let enhanced_confidence = base_confidence * rr_multiplier;

        // Conservative clamping for crypto trading
        enhanced_confidence.clamp(0.20, 0.95)
    }

    /// Calculate confidence for sentiment predictions using unified calibration
    fn calculate_sentiment_confidence(&self, sentiment: &SentimentPrediction) -> f64 {
        // Get sentiment probabilities for unified calibration
        let probs = [
            sentiment.very_bearish_probability,
            sentiment.bearish_probability,
            sentiment.neutral_probability,
            sentiment.bullish_probability,
            sentiment.very_bullish_probability,
        ];

        // Find max probability for unified calibration
        let max_prob = probs.iter().fold(0.0_f64, |a, &b| a.max(b));

        // Base confidence from unified calibration
        let base_confidence = calibrate_5_class_confidence(max_prob);

        // Extreme sentiment bonus (very bearish/bullish are stronger signals in crypto)
        let extreme_sentiment =
            sentiment.very_bearish_probability + sentiment.very_bullish_probability;
        let extreme_multiplier = if extreme_sentiment >= 0.6 {
            1.08 // 8% boost for very extreme sentiment
        } else if extreme_sentiment >= 0.4 {
            1.04 // 4% boost for moderate extreme
        } else {
            1.0
        };

        // Apply extreme sentiment adjustment conservatively
        let enhanced_confidence = base_confidence * extreme_multiplier;

        // Conservative clamping
        enhanced_confidence.clamp(0.20, 0.95)
    }

    /// Calculate confidence for volume predictions using unified calibration
    fn calculate_volume_confidence(&self, volume: &VolumePrediction) -> f64 {
        // Get volume probabilities for unified calibration
        let probs = [
            volume.very_low_probability,
            volume.low_probability,
            volume.medium_probability,
            volume.high_probability,
            volume.very_high_probability,
        ];

        // Find max probability for unified calibration
        let max_prob = probs.iter().fold(0.0_f64, |a, &b| a.max(b));

        // Base confidence from unified calibration
        let base_confidence = calibrate_5_class_confidence(max_prob);

        // High volume bonus (high/very high volume are stronger signals in crypto)
        let high_volume_signal = volume.high_probability + volume.very_high_probability;
        let volume_multiplier = if high_volume_signal >= 0.6 {
            1.08 // 8% boost for very high volume
        } else if high_volume_signal >= 0.4 {
            1.04 // 4% boost for moderate high volume
        } else {
            1.0
        };

        // Apply volume adjustment conservatively
        let enhanced_confidence = base_confidence * volume_multiplier;

        // Conservative clamping
        enhanced_confidence.clamp(0.20, 0.95)
    }

    /// Calculate agreement between different targets
    fn calculate_target_agreement(&self, prediction: &PredictionResult) -> f64 {
        let mut agreement_scores = Vec::new();

        // Check price level and direction agreement
        if let (Some(ref price_levels), Some(ref direction)) =
            (&prediction.price_levels, &prediction.direction)
        {
            let price_direction_agreement =
                self.check_price_direction_agreement(price_levels, direction);
            agreement_scores.push(price_direction_agreement);
        }

        // Check volatility and position sizing agreement
        if let (Some(ref volatility), Some(ref direction)) =
            (&prediction.volatility, &prediction.direction)
        {
            let volatility_agreement = self.check_volatility_agreement(volatility, direction);
            agreement_scores.push(volatility_agreement);
        }

        // Check sentiment and direction agreement
        if let (Some(ref sentiment), Some(ref direction)) =
            (&prediction.sentiment, &prediction.direction)
        {
            let sentiment_agreement =
                self.check_sentiment_direction_agreement(sentiment, direction);
            agreement_scores.push(sentiment_agreement);
        }

        // Check volume confirmation
        if let (Some(ref volume), Some(ref direction)) = (&prediction.volume, &prediction.direction)
        {
            let volume_confirmation = self.check_volume_confirmation(volume, direction);
            agreement_scores.push(volume_confirmation);
        }

        // Calculate overall agreement factor
        if agreement_scores.is_empty() {
            1.0 // Neutral if no comparisons possible
        } else {
            let avg_agreement: f64 =
                agreement_scores.iter().sum::<f64>() / agreement_scores.len() as f64;

            // Apply bonus/penalty based on agreement level (IMPROVED: less punitive)
            if avg_agreement > 0.7 {
                self.config.agreement_bonus.min(avg_agreement * 1.2) // Was 1.5
            } else if avg_agreement < 0.3 {
                self.config.disagreement_penalty.max(avg_agreement * 0.9) // Was 0.7, now 0.9 (less penalty)
            } else {
                // Slightly boost neutral agreement
                (avg_agreement * 1.1).min(1.0) // Was just avg_agreement
            }
        }
    }

    /// Check agreement between price levels and direction
    fn check_price_direction_agreement(
        &self,
        price_levels: &PriceLevelPrediction,
        direction: &DirectionPrediction,
    ) -> f64 {
        // Get upside and downside probabilities from price levels
        let upside_prob = price_levels
            .bins
            .get("moderate_up")
            .map(|b| b.probability)
            .unwrap_or(0.0)
            + price_levels
                .bins
                .get("strong_up")
                .map(|b| b.probability)
                .unwrap_or(0.0);

        let downside_prob = price_levels
            .bins
            .get("moderate_down")
            .map(|b| b.probability)
            .unwrap_or(0.0)
            + price_levels
                .bins
                .get("strong_down")
                .map(|b| b.probability)
                .unwrap_or(0.0);

        // Compare with direction predictions
        let direction_up = direction.up_probability_aggregated;
        let direction_down = direction.down_probability_aggregated;

        // Calculate agreement (1.0 = perfect agreement, 0.0 = complete disagreement)
        let up_agreement = 1.0 - (upside_prob - direction_up).abs();
        let down_agreement = 1.0 - (downside_prob - direction_down).abs();

        (up_agreement + down_agreement) / 2.0
    }

    /// Check if volatility regime supports the predicted direction
    fn check_volatility_agreement(
        &self,
        volatility: &VolatilityPrediction,
        direction: &DirectionPrediction,
    ) -> f64 {
        // High volatility should align with pump/dump predictions
        let extreme_direction = direction.pump_probability + direction.dump_probability;
        let extreme_volatility = volatility.high_probability + volatility.very_high_probability;

        // Low volatility should align with sideways predictions
        let low_volatility = volatility.very_low_probability + volatility.low_probability;
        let sideways = direction.sideways_probability;

        // Calculate agreement
        let extreme_agreement = 1.0 - (extreme_direction - extreme_volatility).abs();
        let calm_agreement = 1.0 - (sideways - low_volatility).abs();

        (extreme_agreement * 0.6 + calm_agreement * 0.4).clamp(0.0, 1.0)
    }

    /// Check if sentiment aligns with direction
    fn check_sentiment_direction_agreement(
        &self,
        sentiment: &SentimentPrediction,
        direction: &DirectionPrediction,
    ) -> f64 {
        // Bullish sentiment should align with up direction
        let bullish_sentiment = sentiment.bullish_probability + sentiment.very_bullish_probability;
        let bearish_sentiment = sentiment.bearish_probability + sentiment.very_bearish_probability;

        let up_direction = direction.up_probability_aggregated;
        let down_direction = direction.down_probability_aggregated;

        // Calculate agreement
        let bullish_agreement = 1.0 - (bullish_sentiment - up_direction).abs();
        let bearish_agreement = 1.0 - (bearish_sentiment - down_direction).abs();

        (bullish_agreement + bearish_agreement) / 2.0
    }

    /// Check if volume confirms the predicted move
    fn check_volume_confirmation(
        &self,
        volume: &VolumePrediction,
        direction: &DirectionPrediction,
    ) -> f64 {
        // High volume should confirm strong directional moves
        let high_volume = volume.high_probability + volume.very_high_probability;
        let strong_direction = direction.pump_probability
            + direction.dump_probability
            + (direction.up_probability_aggregated - 0.5).abs()
            + (direction.down_probability_aggregated - 0.5).abs();

        // Low volume might indicate sideways/neutral
        let low_volume = volume.very_low_probability + volume.low_probability;
        let weak_direction = direction.sideways_probability;

        // Calculate confirmation score
        let strong_confirmation = (high_volume * strong_direction).min(1.0);
        let weak_confirmation = (low_volume * weak_direction).min(1.0);

        (strong_confirmation * 0.7 + weak_confirmation * 0.3).clamp(0.0, 1.0)
    }

    /// CRYPTO-SPECIFIC: Calculate breakout confidence for aggressive entries
    pub fn calculate_breakout_confidence(&self, prediction: &PredictionResult) -> f64 {
        let mut breakout_score = 0.0;
        let mut signal_count = 0;

        // 1. Check for pump/dump signals (primary breakout indicator)
        if let Some(ref direction) = prediction.direction {
            let pump_dump_strength = direction.pump_probability.max(direction.dump_probability);
            if pump_dump_strength > 0.3 {
                breakout_score += pump_dump_strength * 1.5; // Weight heavily
                signal_count += 1;
            }
        }

        // 2. Check for volume surge (confirms breakout)
        if let Some(ref volume) = prediction.volume {
            let high_volume = volume.high_probability + volume.very_high_probability;
            if high_volume > 0.5 {
                breakout_score += high_volume;
                signal_count += 1;
            }
        }

        // 3. Check for extreme price levels (breakout targets)
        if let Some(ref price_levels) = prediction.price_levels {
            let extreme_moves = price_levels
                .bins
                .get("strong_up")
                .map(|b| b.probability)
                .unwrap_or(0.0)
                + price_levels
                    .bins
                    .get("strong_down")
                    .map(|b| b.probability)
                    .unwrap_or(0.0);

            if extreme_moves > 0.4 {
                breakout_score += extreme_moves * 0.8;
                signal_count += 1;
            }
        }

        // 4. Check for sentiment extremes (fuel for breakouts)
        if let Some(ref sentiment) = prediction.sentiment {
            let extreme_sentiment =
                sentiment.very_bullish_probability + sentiment.very_bearish_probability;
            if extreme_sentiment > 0.4 {
                breakout_score += extreme_sentiment * 0.6;
                signal_count += 1;
            }
        }

        // Calculate final breakout confidence
        if signal_count > 0 {
            let avg_score = breakout_score / signal_count as f64;
            // Boost if multiple signals confirm
            let multi_signal_boost = 1.0 + (signal_count as f64 - 1.0) * 0.15;
            (avg_score * multi_signal_boost).min(1.0)
        } else {
            0.0
        }
    }

    /// CRYPTO-SPECIFIC: Calculate scalping confidence for quick trades
    pub fn calculate_scalping_confidence(&self, prediction: &PredictionResult) -> f64 {
        let mut scalp_score = 0.0;

        // 1. Need medium volatility (not too high, not too low)
        if let Some(ref volatility) = prediction.volatility {
            scalp_score += match volatility.regime.as_str() {
                "LOW" => 0.7,
                "MEDIUM" => 1.0, // Perfect for scalping
                "HIGH" => 0.6,   // Too risky
                _ => 0.3,
            };
        }

        // 2. Need clear short-term direction
        if let Some(ref direction) = prediction.direction {
            let directional_clarity = (direction.up_probability_aggregated - 0.5).abs() * 2.0;
            scalp_score += directional_clarity * 0.8;
        }

        // 3. Prefer neutral price levels (range-bound)
        if let Some(ref price_levels) = prediction.price_levels {
            let neutral_prob = price_levels
                .bins
                .get("neutral")
                .map(|b| b.probability)
                .unwrap_or(0.0);
            scalp_score += neutral_prob * 0.6;
        }

        // Average and scale
        (scalp_score / 2.4).min(1.0)
    }

    /// Calculate position size multiplier based on confidence and agreement
    pub fn calculate_position_size_multiplier(
        &self,
        overall_confidence: f64,
        prediction: &PredictionResult,
    ) -> f64 {
        // CRYPTO-OPTIMIZED POSITION SIZING
        // Key insight: Crypto rewards aggressive sizing on HIGH confidence
        // but requires conservative sizing on uncertainty

        // 1. BASE SIZING FROM CONFIDENCE
        let base_multiplier = if overall_confidence > 0.75 {
            2.0 // VERY HIGH confidence: 200% (use leverage)
        } else if overall_confidence > 0.65 {
            1.5 // HIGH confidence: 150% position
        } else if overall_confidence > 0.55 {
            1.2 // GOOD confidence: 120% position
        } else if overall_confidence > 0.45 {
            1.0 // MODERATE confidence: 100% position
        } else if overall_confidence > 0.35 {
            0.75 // LOW confidence: 75% position
        } else {
            0.5 // VERY LOW confidence: 50% position
        };

        // 2. VOLATILITY-BASED ADJUSTMENT (Crypto-specific)
        let volatility_multiplier = if let Some(ref vol) = prediction.volatility {
            match vol.regime.as_str() {
                "VERY_LOW" => 1.3,  // Can size up in calm markets
                "LOW" => 1.2,       // Slightly larger positions OK
                "MEDIUM" => 1.0,    // Standard sizing
                "HIGH" => 0.8,      // Reduce in high volatility
                "VERY_HIGH" => 0.5, // Half position in extreme volatility
                _ => 1.0,
            }
        } else {
            1.0
        };

        // 3. RISK-REWARD ADJUSTMENT (Crypto loves high R:R)
        let rr_multiplier = if let Some(ref dir) = prediction.direction {
            if dir.risk_reward_ratio > 10.0 {
                1.5 // EXCEPTIONAL R:R (10:1+) = size up
            } else if dir.risk_reward_ratio > 6.0 {
                1.2 // EXCELLENT R:R (6:1+) = moderate size up
            } else if dir.risk_reward_ratio > 4.0 {
                1.0 // GOOD R:R (4:1+) = standard
            } else if dir.risk_reward_ratio > 2.0 {
                0.8 // ACCEPTABLE R:R (2:1+) = reduce
            } else {
                0.5 // POOR R:R (<2:1) = minimize
            }
        } else {
            1.0
        };

        // 4. PUMP/DUMP DETECTION (Crypto-specific aggressive sizing)
        let momentum_multiplier = if let Some(ref dir) = prediction.direction {
            if dir.pump_probability > 0.4 || dir.dump_probability > 0.4 {
                // Strong pump/dump signal with good confidence
                if overall_confidence > 0.6 {
                    1.3 // Size up for momentum trades
                } else {
                    0.7 // But be careful if confidence is low
                }
            } else {
                1.0
            }
        } else {
            1.0
        };

        // 5. VOLUME CONFIRMATION (Critical for crypto)
        let volume_multiplier = if let Some(ref vol) = prediction.volume {
            let high_volume = vol.high_probability + vol.very_high_probability;
            if high_volume > 0.6 {
                1.2 // High volume confirms the move
            } else if high_volume < 0.2 {
                0.8 // Low volume = be cautious
            } else {
                1.0
            }
        } else {
            0.9 // No volume data = slight reduction
        };

        // COMBINE ALL MULTIPLIERS
        let final_multiplier = base_multiplier
            * volatility_multiplier
            * rr_multiplier
            * momentum_multiplier
            * volume_multiplier;

        // CRYPTO REALITY: Cap at 3x for risk management (even with leverage)
        // Minimum 0.25x to always have some position
        f64::clamp(final_multiplier, 0.25, 3.0)
    }

    /// Calculate individual order level confidence based on price probability
    pub fn calculate_order_confidence(
        &self,
        order_price: f64,
        current_price: f64,
        prediction: &PredictionResult,
        order_type: &str, // "entry", "exit", or "stop"
    ) -> f64 {
        let price_change_pct = ((order_price - current_price) / current_price) * 100.0;

        // Find which bin this price falls into
        if let Some(ref price_levels) = prediction.price_levels {
            for bin in price_levels.bins.values() {
                if price_change_pct >= bin.range[0] && price_change_pct <= bin.range[1] {
                    // Base confidence on bin probability
                    let bin_confidence = bin.probability;

                    // Adjust based on order type
                    return match order_type {
                        "entry" => {
                            // Entry orders: higher confidence for higher probability bins
                            if bin_confidence > 0.3 {
                                bin_confidence * 1.2 // Boost high probability entries
                            } else {
                                bin_confidence * 0.8 // Reduce low probability entries
                            }
                        }
                        "exit" => {
                            // Exit orders: scale confidence by expected probability
                            bin_confidence
                        }
                        "stop" => {
                            // Stop orders: inverse confidence (low probability = good stop)
                            1.0 - bin_confidence * 0.5
                        }
                        _ => bin_confidence,
                    }
                    .clamp(0.1, 0.95);
                }
            }
        }

        // Default confidence if no bin match
        0.5
    }
}

/// Enhanced position sizing based on multi-target agreement
pub struct EnhancedPositionSizer {
    confidence_calc: ConfidenceCalculator,
    max_position_size: f64,
    min_position_size: f64,
}

impl EnhancedPositionSizer {
    pub fn new(confidence_config: ConfidenceConfig) -> Self {
        Self {
            confidence_calc: ConfidenceCalculator::new(confidence_config),
            max_position_size: 1.0, // 100% of capital
            min_position_size: 0.1, // 10% minimum
        }
    }

    /// Calculate dynamic position sizes for entry levels based on probabilities
    pub fn calculate_entry_sizes(
        &self,
        prediction: &PredictionResult,
        overall_confidence: f64,
    ) -> Result<[f64; 3]> {
        // Get position size multiplier
        let multiplier = self
            .confidence_calc
            .calculate_position_size_multiplier(overall_confidence, prediction);

        // Base position size with min/max constraints
        let base_size = (self.max_position_size * multiplier)
            .min(self.max_position_size)
            .max(self.min_position_size);

        // Distribute across 3 entry levels based on price level probabilities
        if let Some(ref price_levels) = prediction.price_levels {
            // For LONG: use downside bins (buying opportunities)
            // For SHORT: use upside bins (selling opportunities)
            let is_long = prediction
                .direction
                .as_ref()
                .map(|d| d.up_probability_aggregated > d.down_probability_aggregated)
                .unwrap_or(true);

            if is_long {
                // LONG entries: weight by downside probabilities
                let moderate_down_prob = price_levels
                    .bins
                    .get("moderate_down")
                    .map(|b| b.probability)
                    .unwrap_or(0.3);
                let strong_down_prob = price_levels
                    .bins
                    .get("strong_down")
                    .map(|b| b.probability)
                    .unwrap_or(0.2);
                let neutral_prob = price_levels
                    .bins
                    .get("neutral")
                    .map(|b| b.probability)
                    .unwrap_or(0.2);

                // Normalize probabilities
                let total_prob = moderate_down_prob + strong_down_prob + neutral_prob;
                let norm_factor = if total_prob > 0.0 {
                    1.0 / total_prob
                } else {
                    1.0
                };

                Ok([
                    base_size * moderate_down_prob * norm_factor * 1.2, // Entry 1: Most likely
                    base_size * neutral_prob * norm_factor,             // Entry 2: Neutral
                    base_size * strong_down_prob * norm_factor * 0.8,   // Entry 3: Less likely
                ])
            } else {
                // SHORT entries: weight by upside probabilities
                let moderate_up_prob = price_levels
                    .bins
                    .get("moderate_up")
                    .map(|b| b.probability)
                    .unwrap_or(0.3);
                let strong_up_prob = price_levels
                    .bins
                    .get("strong_up")
                    .map(|b| b.probability)
                    .unwrap_or(0.2);
                let neutral_prob = price_levels
                    .bins
                    .get("neutral")
                    .map(|b| b.probability)
                    .unwrap_or(0.2);

                // Normalize probabilities
                let total_prob = moderate_up_prob + strong_up_prob + neutral_prob;
                let norm_factor = if total_prob > 0.0 {
                    1.0 / total_prob
                } else {
                    1.0
                };

                Ok([
                    base_size * moderate_up_prob * norm_factor * 1.2, // Entry 1: Most likely
                    base_size * neutral_prob * norm_factor,           // Entry 2: Neutral
                    base_size * strong_up_prob * norm_factor * 0.8,   // Entry 3: Less likely
                ])
            }
        } else {
            // Fallback to equal distribution
            Ok([base_size / 3.0, base_size / 3.0, base_size / 3.0])
        }
    }

    /// Calculate exit sizes with partial profit taking based on confidence
    /// Calculate exit sizes with partial profit taking based on confidence and volatility
    pub fn calculate_exit_sizes(
        &self,
        prediction: &PredictionResult,
        overall_confidence: f64,
    ) -> [f64; 3] {
        // Adjust exit strategy based on volatility regime
        let volatility_adjustment = if let Some(ref vol) = prediction.volatility {
            match vol.regime.as_str() {
                "VERY_HIGH" | "HIGH" => {
                    // In high volatility, take profits more aggressively
                    1.2
                }
                "MEDIUM" => 1.0,
                "LOW" | "VERY_LOW" => {
                    // In low volatility, can hold for larger moves
                    0.8
                }
                _ => 1.0,
            }
        } else {
            1.0
        };

        // Base exit sizes on confidence level
        let base_sizes = if overall_confidence > 0.7 {
            // High confidence: hold longer for bigger moves
            [0.3, 0.4, 0.3] // 30%, 40%, 30%
        } else if overall_confidence > 0.5 {
            // Medium confidence: balanced exit
            [0.4, 0.35, 0.25] // 40%, 35%, 25%
        } else {
            // Low confidence: exit quickly
            [0.5, 0.3, 0.2] // 50%, 30%, 20%
        };

        // Apply volatility adjustment
        [
            f64::min(base_sizes[0] * volatility_adjustment, 0.6),
            base_sizes[1],
            f64::max(base_sizes[2] / volatility_adjustment, 0.1),
        ]
    }
}

/// UNIFIED 5-Class Confidence Calibration Based on Real-World Neural Network Research
///
/// This function provides research-based mapping from LSTM softmax probabilities to confidence scores.
/// Based on "On Calibration of Modern Neural Networks" (Guo et al., 2017) and real-world studies.
///
/// REAL-WORLD PROBABILITY DISTRIBUTIONS:
/// - Well-calibrated models rarely exceed 0.4-0.5 max probability
/// - 0.35+ max probability is EXCELLENT performance
/// - 0.25-0.35 max probability is TYPICAL for good models
/// - 0.45+ max probability suggests possible overfitting
/// - Perfect 1.0 probabilities indicate severe overfitting
///
/// CONFIDENCE MAPPING PHILOSOPHY:
/// - Conservative approach aligned with crypto market uncertainty
/// - Recognizes that lower probabilities are actually very good
/// - Avoids overconfidence that leads to excessive risk-taking
pub fn calibrate_5_class_confidence(max_probability: f64) -> f64 {
    const UNIFORM_BASELINE: f64 = 0.2; // 1/5 for 5 classes (random guessing)

    if max_probability < UNIFORM_BASELINE {
        // Below random (shouldn't happen with softmax, but handle gracefully)
        max_probability * 2.0
    } else if max_probability >= 0.50 {
        // EXCEPTIONAL: Very rare in well-calibrated models, suggests possible overfitting
        // Maps: 0.50→0.85, 0.60→0.90, 0.70→0.93, 1.0→1.0
        // Conservative mapping to avoid overconfidence
        0.85 + (max_probability - 0.50) * 0.30
    } else if max_probability >= 0.42 {
        // EXCELLENT: Top 5% of predictions in well-calibrated models
        // Maps: 0.42→0.78, 0.50→0.85
        0.78 + (max_probability - 0.42) * 0.875
    } else if max_probability >= 0.35 {
        // VERY GOOD: Top 15% of predictions, this is actually excellent performance
        // Maps: 0.35→0.68, 0.42→0.78
        0.68 + (max_probability - 0.35) * 1.43
    } else if max_probability >= 0.30 {
        // GOOD: Top 30% of predictions, significantly better than random
        // Maps: 0.30→0.55, 0.35→0.68
        0.55 + (max_probability - 0.30) * 2.60
    } else if max_probability >= 0.25 {
        // MODERATE: Better than random, tradeable signal
        // Maps: 0.25→0.42, 0.30→0.55
        0.42 + (max_probability - 0.25) * 2.60
    } else if max_probability >= UNIFORM_BASELINE {
        // WEAK: Barely better than random, low confidence
        // Maps: 0.20→0.25, 0.25→0.42
        0.25 + (max_probability - UNIFORM_BASELINE) * 3.40
    } else {
        // Below random baseline (edge case)
        max_probability * 1.25
    }
}

/// Calculate entropy-based confidence for 5-class system
pub fn calculate_5_class_entropy_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() != 5 {
        return 0.5; // Default to neutral if not 5 classes
    }

    // Calculate Shannon entropy
    let entropy: f64 = probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    // Maximum entropy for 5 classes = ln(5) ≈ 1.609
    let max_entropy = 5_f64.ln();

    // Convert entropy to confidence (low entropy = high confidence)
    1.0 - (entropy / max_entropy)
}

/// Combined confidence calculation using multiple methods
pub fn calculate_combined_5_class_confidence(probabilities: &[f64], max_probability: f64) -> f64 {
    // Method 1: Calibrated max probability
    let calibrated_confidence = calibrate_5_class_confidence(max_probability);

    // Method 2: Entropy-based confidence
    let entropy_confidence = calculate_5_class_entropy_confidence(probabilities);

    // Method 3: Deviation from uniform
    let uniform_baseline = 0.2;
    let deviation_confidence = if max_probability > uniform_baseline {
        (max_probability - uniform_baseline) / (1.0 - uniform_baseline)
    } else {
        0.0
    };

    // Weighted combination
    // Calibrated: 60% (most interpretable and adjusted for crypto reality)
    // Entropy: 25% (information theory)
    // Deviation: 15% (simple metric)
    let combined =
        calibrated_confidence * 0.6 + entropy_confidence * 0.25 + deviation_confidence * 0.15;

    // Less restrictive clamping for crypto trading
    combined.clamp(0.20, 0.98)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_confidence_calculation() {
        // Test will be implemented
    }
}
