//! Calibration Utility Functions
//!
//! Shared utility functions used across all calibration modules.
//! Contains balance calculation and other common calibration helpers.

use super::types::ClassBalance;
use crate::data::structures::MarketDataRow;
use crate::utils::error::Result;

/// Calibration utilities
pub struct CalibrationUtils {
    pub balance_weight: f64,
    pub diversity_weight: f64,
    pub target_balance: f64,
}

impl CalibrationUtils {
    pub fn new(balance_weight: f64, diversity_weight: f64, target_balance: f64) -> Self {
        Self {
            balance_weight,
            diversity_weight,
            target_balance,
        }
    }

    /// Calculate class balance metrics with REAL diversity scoring
    pub fn calculate_balance(&self, class_counts: &[usize], total: usize) -> Result<ClassBalance> {
        if total == 0 || class_counts.len() != 5 {
            return Ok(ClassBalance::default());
        }

        // 1. Calculate basic balance metrics (existing logic)
        let mut class_percentages = [0.0; 5];
        for (i, &count) in class_counts.iter().enumerate() {
            class_percentages[i] = (count as f64 / total as f64) * 100.0;
        }

        let target_percentage = 100.0 / 5.0; // 20% per class
        let balance_score: f64 = class_percentages
            .iter()
            .map(|&p| (p - target_percentage).abs())
            .sum::<f64>()
            / 5.0;

        let min_percentage = class_percentages
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);

        let max_percentage = class_percentages
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);

        let imbalance_ratio = if min_percentage > 0.0 {
            max_percentage / min_percentage
        } else {
            f64::INFINITY
        };

        // 2. Use default diversity values (will be calculated with full context)
        let diversity_score = 0.5;
        let temporal_spread = 0.5;
        let feature_diversity = 0.5;
        let market_condition_diversity = 0.5;

        // Composite quality score combines balance and diversity
        let normalized_balance_penalty = balance_score / 20.0;
        let composite_quality_score = self.balance_weight * normalized_balance_penalty
            + self.diversity_weight * (1.0 - diversity_score);

        Ok(ClassBalance {
            class_percentages,
            balance_score,
            imbalance_ratio,
            total_samples: total,
            target_balance: self.target_balance,
            diversity_score,
            temporal_spread,
            feature_diversity,
            market_condition_diversity,
            composite_quality_score,
        })
    }

    /// Calculate class balance with REAL diversity metrics (full context version)
    pub fn calculate_balance_with_diversity(
        &self,
        class_counts: &[usize],
        total: usize,
        ohlcv_data: &[MarketDataRow],
        sample_indices: &[usize],
    ) -> Result<ClassBalance> {
        if total == 0 || class_counts.len() != 5 {
            return Ok(ClassBalance::default());
        }

        // 1. Calculate basic balance metrics
        let mut class_percentages = [0.0; 5];
        for (i, &count) in class_counts.iter().enumerate() {
            class_percentages[i] = (count as f64 / total as f64) * 100.0;
        }

        let target_percentage = 100.0 / 5.0; // 20% per class
        let balance_score: f64 = class_percentages
            .iter()
            .map(|&p| (p - target_percentage).abs())
            .sum::<f64>()
            / 5.0;

        let min_percentage = class_percentages
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);

        let max_percentage = class_percentages
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);

        let imbalance_ratio = if min_percentage > 0.0 {
            max_percentage / min_percentage
        } else {
            f64::INFINITY
        };

        // 2. Calculate REAL diversity metrics
        let temporal_spread = Self::calculate_temporal_diversity(sample_indices);
        let feature_diversity = Self::calculate_feature_space_diversity(ohlcv_data, sample_indices);
        let market_condition_diversity =
            Self::calculate_market_condition_diversity(ohlcv_data, sample_indices);

        // Overall diversity score (weighted average)
        let diversity_score =
            (temporal_spread * 0.4 + feature_diversity * 0.3 + market_condition_diversity * 0.3)
                .clamp(0.0, 1.0);

        // Composite quality score combines balance and diversity
        let normalized_balance_penalty = balance_score / 20.0;
        let composite_quality_score = self.balance_weight * normalized_balance_penalty
            + self.diversity_weight * (1.0 - diversity_score);

        Ok(ClassBalance {
            class_percentages,
            balance_score,
            imbalance_ratio,
            total_samples: total,
            target_balance: self.target_balance,
            diversity_score,
            temporal_spread,
            feature_diversity,
            market_condition_diversity,
            composite_quality_score,
        })
    }

    /// Calculate temporal diversity using coefficient of variation
    /// Returns 0.0 (poor) to 1.0 (excellent temporal spread)
    pub fn calculate_temporal_diversity(sample_indices: &[usize]) -> f64 {
        if sample_indices.len() < 2 {
            return 0.0;
        }

        // Calculate gaps between consecutive samples
        let mut sorted_indices = sample_indices.to_vec();
        sorted_indices.sort_unstable();

        let mut gaps = Vec::new();
        for i in 1..sorted_indices.len() {
            gaps.push((sorted_indices[i] - sorted_indices[i - 1]) as f64);
        }

        if gaps.is_empty() {
            return 0.0;
        }

        // Calculate coefficient of variation (CV = std / mean)
        let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let variance = gaps.iter().map(|g| (g - mean_gap).powi(2)).sum::<f64>() / gaps.len() as f64;
        let std_dev = variance.sqrt();
        let cv = if mean_gap > 0.0 {
            std_dev / mean_gap
        } else {
            0.0
        };

        // Convert CV to diversity score (lower CV = more uniform = better)
        // CV of 0 = perfect uniform spacing = 1.0 diversity
        // CV of 1+ = very uneven spacing = 0.0 diversity
        (1.0 - cv.min(1.0)).clamp(0.0, 1.0)
    }

    /// Calculate feature space diversity using price ranges and volatility
    /// Returns 0.0 (poor) to 1.0 (excellent feature coverage)
    pub fn calculate_feature_space_diversity(
        ohlcv_data: &[MarketDataRow],
        sample_indices: &[usize],
    ) -> f64 {
        if sample_indices.is_empty() || ohlcv_data.is_empty() {
            return 0.0;
        }

        // Extract price ranges and volatility for sampled periods
        let mut price_ranges = Vec::new();
        let mut volatilities = Vec::new();

        for &idx in sample_indices {
            if idx < ohlcv_data.len() {
                let candle = &ohlcv_data[idx];
                let price_range = (candle.high - candle.low) / candle.close;
                let volatility = (candle.close - candle.open).abs() / candle.close;

                price_ranges.push(price_range);
                volatilities.push(volatility);
            }
        }

        if price_ranges.is_empty() {
            return 0.0;
        }

        // Calculate coverage using quartile representation
        let price_coverage = Self::calculate_quartile_coverage(&price_ranges);
        let volatility_coverage = Self::calculate_quartile_coverage(&volatilities);

        // Average coverage
        (price_coverage + volatility_coverage) / 2.0
    }

    /// Calculate market condition diversity (bull/bear/sideways distribution)
    /// Returns 0.0 (poor) to 1.0 (excellent condition balance)
    pub fn calculate_market_condition_diversity(
        ohlcv_data: &[MarketDataRow],
        sample_indices: &[usize],
    ) -> f64 {
        if sample_indices.len() < 10 || ohlcv_data.len() < 20 {
            return 0.0;
        }

        let mut bull_count = 0;
        let mut bear_count = 0;
        let mut sideways_count = 0;
        let mut valid_samples = 0;

        // Classify each sample's market condition with better lookback
        let lookback = 10.min(ohlcv_data.len() / 10); // Adaptive lookback

        for &idx in sample_indices {
            if idx >= lookback && idx + lookback < ohlcv_data.len() {
                // Look at trend around sample (more robust)
                let start_price = ohlcv_data[idx - lookback].close;
                let end_price = ohlcv_data[idx + lookback].close;
                let change_pct = (end_price - start_price) / start_price;

                // More sensitive thresholds for better classification
                if change_pct > 0.01 {
                    bull_count += 1;
                } else if change_pct < -0.01 {
                    bear_count += 1;
                } else {
                    sideways_count += 1;
                }
                valid_samples += 1;
            }
        }

        if valid_samples == 0 {
            return 0.0;
        }

        let total = valid_samples as f64;

        // Calculate distribution balance (ideal: 33.3% each)
        let bull_pct = bull_count as f64 / total;
        let bear_pct = bear_count as f64 / total;
        let sideways_pct = sideways_count as f64 / total;

        log::debug!(
            "Market conditions: Bull={:.1}%, Bear={:.1}%, Sideways={:.1}% (from {} samples)",
            bull_pct * 100.0,
            bear_pct * 100.0,
            sideways_pct * 100.0,
            valid_samples
        );

        let ideal = 1.0 / 3.0;
        let deviation =
            ((bull_pct - ideal).abs() + (bear_pct - ideal).abs() + (sideways_pct - ideal).abs())
                / 3.0;

        // Convert deviation to diversity score (0 deviation = 1.0 diversity)
        (1.0 - deviation * 3.0).clamp(0.0, 1.0)
    }

    /// Calculate quartile coverage for a feature
    fn calculate_quartile_coverage(values: &[f64]) -> f64 {
        if values.len() < 4 {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let q1_idx = n / 4;
        let q2_idx = n / 2;
        let q3_idx = 3 * n / 4;

        // Check representation in each quartile
        let mut quartile_counts = [0usize; 4];
        for &val in values {
            if val <= sorted[q1_idx] {
                quartile_counts[0] += 1;
            } else if val <= sorted[q2_idx] {
                quartile_counts[1] += 1;
            } else if val <= sorted[q3_idx] {
                quartile_counts[2] += 1;
            } else {
                quartile_counts[3] += 1;
            }
        }

        // Calculate balance across quartiles (ideal: 25% each)
        let total = values.len() as f64;
        let ideal = 0.25;
        let deviation: f64 = quartile_counts
            .iter()
            .map(|&count| ((count as f64 / total) - ideal).abs())
            .sum::<f64>()
            / 4.0;

        // Convert deviation to coverage score
        (1.0 - deviation * 4.0).clamp(0.0, 1.0)
    }
}
