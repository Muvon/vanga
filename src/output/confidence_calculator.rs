//! Enhanced confidence calculation using multi-target agreement and probability distributions
//!
//! This module provides sophisticated confidence scoring by analyzing agreement between
//! multiple prediction targets and using probability distributions for weighted confidence.

use crate::output::prediction_types::{
    DirectionPrediction, PredictionResult, PriceLevelPrediction, SentimentPrediction,
    VolatilityPrediction, VolumePrediction,
};
use crate::utils::error::Result;
use std::collections::HashMap;

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
            price_level_weight: 0.25, // 25% - Show levels, comes after direction importance
            direction_weight: 0.35,   // 35% - Confirms trend - most important
            volatility_weight: 0.1,   // 10% - Risk management
            sentiment_weight: 0.15,   // 15% - Market psychology
            volume_weight: 0.15,      // 15% - Confirmation
            min_probability_threshold: 0.15, // 15% minimum for confidence (was 0.25)
            agreement_bonus: 1.2,     // 20% bonus for agreement (was 1.5)
            disagreement_penalty: 0.85, // 15% penalty for disagreement (was 0.7)
        }
    }
}

/// Enhanced confidence calculator for multi-target predictions
pub struct ConfidenceCalculator {
    config: ConfidenceConfig,
}

impl ConfidenceCalculator {
    /// Create new confidence calculator with configuration
    pub fn new(config: ConfidenceConfig) -> Self {
        Self { config }
    }

    /// Calculate enhanced confidence score using all available targets
    pub fn calculate_overall_confidence(&self, prediction: &PredictionResult) -> f64 {
        let mut weighted_scores = Vec::new();
        let mut weights = Vec::new();

        // Calculate individual target confidence scores
        if let Some(ref price_levels) = prediction.price_levels {
            let score = self.calculate_price_level_confidence(price_levels);
            weighted_scores.push(score);
            weights.push(self.config.price_level_weight);
        }

        if let Some(ref direction) = prediction.direction {
            let score = self.calculate_direction_confidence(direction);
            weighted_scores.push(score);
            weights.push(self.config.direction_weight);
        }

        if let Some(ref volatility) = prediction.volatility {
            let score = self.calculate_volatility_confidence(volatility);
            weighted_scores.push(score);
            weights.push(self.config.volatility_weight);
        }

        if let Some(ref sentiment) = prediction.sentiment {
            let score = self.calculate_sentiment_confidence(sentiment);
            weighted_scores.push(score);
            weights.push(self.config.sentiment_weight);
        }

        if let Some(ref volume) = prediction.volume {
            let score = self.calculate_volume_confidence(volume);
            weighted_scores.push(score);
            weights.push(self.config.volume_weight);
        }

        // Calculate base weighted average
        let total_weight: f64 = weights.iter().sum();
        let mut base_confidence = if total_weight > 0.0 {
            weighted_scores
                .iter()
                .zip(weights.iter())
                .map(|(score, weight)| score * weight)
                .sum::<f64>()
                / total_weight
        } else {
            0.5 // Default neutral confidence
        };

        // Apply agreement/disagreement adjustments
        let agreement_factor = self.calculate_target_agreement(prediction);
        base_confidence *= agreement_factor;

        // Apply risk adjustments based on volatility
        if let Some(ref volatility) = prediction.volatility {
            base_confidence = self.apply_volatility_adjustment(base_confidence, volatility);
        }

        // IMPROVED: More generous clamping to ensure reasonable minimum confidence
        // Even poor predictions should have at least 25% confidence to avoid filtering
        base_confidence.clamp(0.25, 0.95) // Was 0.1, now 0.25 minimum
    }

    /// Calculate confidence for price level predictions using probability distribution
    fn calculate_price_level_confidence(&self, price_levels: &PriceLevelPrediction) -> f64 {
        // Get the highest probability bin
        let max_prob = price_levels
            .bins
            .values()
            .map(|bin| bin.probability)
            .fold(0.0, f64::max);

        // Calculate entropy-based confidence (lower entropy = higher confidence)
        let entropy = self.calculate_entropy_from_bins(&price_levels.bins);
        let entropy_confidence = 1.0 - (entropy / 2.3); // log2(5) ≈ 2.3 for 5 classes

        // IMPROVED: More generous probability confidence calculation
        let prob_confidence = if max_prob > self.config.min_probability_threshold {
            // Scale from threshold to 1.0 more generously
            0.5 + (max_prob - self.config.min_probability_threshold)
                / (1.0 - self.config.min_probability_threshold)
                * 0.5
        } else {
            // Still give reasonable confidence for lower probabilities
            max_prob * 2.0 // Was 0.5, now 2.0 to be less punitive
        };

        // Weight between probability and entropy measures
        (prob_confidence * 0.6 + entropy_confidence * 0.4).clamp(0.2, 1.0) // Min 0.2 instead of 0.0
    }

    /// Calculate confidence for direction predictions
    fn calculate_direction_confidence(&self, direction: &DirectionPrediction) -> f64 {
        // Calculate directional clarity (how much one direction dominates)
        let up_strength = direction.up_probability_aggregated;
        let down_strength = direction.down_probability_aggregated;
        let sideways_strength = direction.sideways_probability;

        // Find dominant direction
        let max_strength = up_strength.max(down_strength).max(sideways_strength);

        // IMPROVED: More generous confidence calculation
        let dominance_confidence = if max_strength > self.config.min_probability_threshold {
            // Scale more generously
            0.5 + (max_strength - self.config.min_probability_threshold)
                / (1.0 - self.config.min_probability_threshold)
                * 0.5
        } else {
            // Still give reasonable confidence
            max_strength * 2.0 // Was 0.5, now 2.0
        };

        // Factor in risk-reward ratio (higher R/R = more confidence)
        let rr_confidence = (direction.risk_reward_ratio / 6.0).min(1.0); // Was /10.0, now /6.0 for better scaling

        // Combine factors
        (dominance_confidence * 0.7 + rr_confidence * 0.3).clamp(0.2, 1.0) // Min 0.2
    }

    /// Calculate confidence for volatility predictions
    fn calculate_volatility_confidence(&self, volatility: &VolatilityPrediction) -> f64 {
        // Get regime probabilities
        let probs = vec![
            volatility.very_low_probability,
            volatility.low_probability,
            volatility.medium_probability,
            volatility.high_probability,
            volatility.very_high_probability,
        ];

        // Find max probability
        let max_prob = probs.iter().fold(0.0_f64, |a, &b| a.max(b));

        // Calculate entropy for uncertainty
        let entropy = self.calculate_entropy(&probs);
        let entropy_confidence = 1.0 - (entropy / 2.3);

        // Volatility-specific: Medium volatility is actually good for trading
        let regime_quality = match volatility.regime.as_str() {
            "VERY_LOW" => 0.6,  // Too quiet, less opportunities
            "LOW" => 0.8,       // Good for trading
            "MEDIUM" => 1.0,    // Ideal for trading
            "HIGH" => 0.7,      // Risky but tradeable
            "VERY_HIGH" => 0.4, // Too risky
            _ => 0.5,
        };

        // Combine factors
        (max_prob * 0.4 + entropy_confidence * 0.3 + regime_quality * 0.3).clamp(0.0, 1.0)
    }

    /// Calculate confidence for sentiment predictions
    fn calculate_sentiment_confidence(&self, sentiment: &SentimentPrediction) -> f64 {
        // Get sentiment probabilities
        let probs = [
            sentiment.very_bearish_probability,
            sentiment.bearish_probability,
            sentiment.neutral_probability,
            sentiment.bullish_probability,
            sentiment.very_bullish_probability,
        ];

        // Find max probability
        let max_prob = probs.iter().fold(0.0_f64, |a, &b| a.max(b));

        // Strong sentiment (very bearish/bullish) is more actionable
        let extreme_sentiment =
            sentiment.very_bearish_probability + sentiment.very_bullish_probability;

        // Calculate confidence
        if max_prob > self.config.min_probability_threshold {
            (max_prob * 0.7 + extreme_sentiment * 0.3).min(1.0)
        } else {
            max_prob * 0.5
        }
    }

    /// Calculate confidence for volume predictions
    fn calculate_volume_confidence(&self, volume: &VolumePrediction) -> f64 {
        // Get volume probabilities
        let probs = [
            volume.very_low_probability,
            volume.low_probability,
            volume.medium_probability,
            volume.high_probability,
            volume.very_high_probability,
        ];

        // Find max probability
        let max_prob = probs.iter().fold(0.0_f64, |a, &b| a.max(b));

        // High volume confirms moves
        let high_volume_signal = volume.high_probability + volume.very_high_probability;

        // Calculate confidence
        if max_prob > self.config.min_probability_threshold {
            (max_prob * 0.6 + high_volume_signal * 0.4).min(1.0)
        } else {
            max_prob * 0.5
        }
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

    /// Apply volatility-based risk adjustment to confidence
    fn apply_volatility_adjustment(
        &self,
        confidence: f64,
        volatility: &VolatilityPrediction,
    ) -> f64 {
        // Adjust confidence based on volatility regime
        let adjustment = match volatility.regime.as_str() {
            "VERY_LOW" => 0.9,  // Slightly reduce - might be too quiet
            "LOW" => 1.0,       // Ideal for trading
            "MEDIUM" => 1.0,    // Ideal for trading
            "HIGH" => 0.85,     // Reduce due to risk
            "VERY_HIGH" => 0.7, // Significantly reduce due to extreme risk
            _ => 1.0,
        };

        confidence * adjustment
    }

    /// Calculate entropy from probability distribution
    fn calculate_entropy(&self, probabilities: &[f64]) -> f64 {
        let mut entropy = 0.0;
        for &p in probabilities {
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Calculate entropy from bins
    fn calculate_entropy_from_bins(
        &self,
        bins: &HashMap<String, crate::output::prediction_types::PriceBin>,
    ) -> f64 {
        let probs: Vec<f64> = bins.values().map(|bin| bin.probability).collect();
        self.calculate_entropy(&probs)
    }

    /// Calculate position size multiplier based on confidence and agreement
    pub fn calculate_position_size_multiplier(
        &self,
        overall_confidence: f64,
        prediction: &PredictionResult,
    ) -> f64 {
        // Base multiplier from confidence
        let base_multiplier = if overall_confidence > 0.8 {
            1.5 // High confidence: 150% position
        } else if overall_confidence > 0.6 {
            1.0 // Medium confidence: 100% position
        } else if overall_confidence > 0.4 {
            0.75 // Low confidence: 75% position
        } else {
            0.5 // Very low confidence: 50% position
        };

        // Adjust for volatility regime
        let volatility_multiplier = if let Some(ref vol) = prediction.volatility {
            match vol.regime.as_str() {
                "VERY_LOW" => 1.2, // Can take larger position in calm markets
                "LOW" => 1.1,
                "MEDIUM" => 1.0,
                "HIGH" => 0.8,      // Reduce position in volatile markets
                "VERY_HIGH" => 0.5, // Significantly reduce in extreme volatility
                _ => 1.0,
            }
        } else {
            1.0
        };

        // Adjust for risk-reward ratio
        let rr_multiplier = if let Some(ref dir) = prediction.direction {
            if dir.risk_reward_ratio > 8.0 {
                1.2 // Excellent R/R
            } else if dir.risk_reward_ratio > 4.0 {
                1.0 // Good R/R
            } else if dir.risk_reward_ratio > 2.0 {
                0.8 // Acceptable R/R
            } else {
                0.5 // Poor R/R
            }
        } else {
            1.0
        };

        // Combine all multipliers
        f64::clamp(
            base_multiplier * volatility_multiplier * rr_multiplier,
            0.25,
            2.0,
        )
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

#[cfg(test)]
mod tests {
    #[test]
    fn test_confidence_calculation() {
        // Test will be implemented
    }
}
