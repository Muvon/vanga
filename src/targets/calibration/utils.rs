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

        // CRITICAL: Penalize missing classes HEAVILY (imbalance_ratio = f64::INFINITY)
        // Missing classes are UNACCEPTABLE for training - must have all 5 classes
        let missing_class_penalty = if imbalance_ratio.is_infinite() {
            log::debug!(
                "⚠️  REJECTED: Parameters produce missing classes (imbalance_ratio=∞) - adding penalty=1000.0"
            );
            1000.0 // Massive penalty to reject parameters that eliminate classes
        } else if imbalance_ratio > 10.0 {
            // Also penalize severe imbalance (one class 10x another)
            let penalty = (imbalance_ratio - 10.0) * 10.0;
            log::debug!(
                "⚠️  Severe imbalance detected (ratio={:.2}) - adding penalty={:.2}",
                imbalance_ratio,
                penalty
            );
            penalty
        } else {
            0.0
        };

        // Composite quality score combines balance, diversity, and missing class penalty
        let normalized_balance_penalty = balance_score / 20.0;
        let composite_quality_score = self.balance_weight * normalized_balance_penalty
            + self.diversity_weight * (1.0 - diversity_score)
            + missing_class_penalty; // CRITICAL: Add missing class penalty

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
        sequence_length: usize,
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
            Self::calculate_market_condition_diversity(ohlcv_data, sample_indices, sequence_length);

        // Overall diversity score (weighted average)
        let diversity_score =
            (temporal_spread * 0.4 + feature_diversity * 0.3 + market_condition_diversity * 0.3)
                .clamp(0.0, 1.0);

        // CRITICAL: Penalize missing classes HEAVILY (imbalance_ratio = f64::INFINITY)
        // Missing classes are UNACCEPTABLE for training - must have all 5 classes
        let missing_class_penalty = if imbalance_ratio.is_infinite() {
            log::debug!(
                "⚠️  REJECTED: Parameters produce missing classes (imbalance_ratio=∞) - adding penalty=1000.0"
            );
            1000.0 // Massive penalty to reject parameters that eliminate classes
        } else if imbalance_ratio > 10.0 {
            // Also penalize severe imbalance (one class 10x another)
            let penalty = (imbalance_ratio - 10.0) * 10.0;
            log::debug!(
                "⚠️  Severe imbalance detected (ratio={:.2}) - adding penalty={:.2}",
                imbalance_ratio,
                penalty
            );
            penalty
        } else {
            0.0
        };

        // Composite quality score combines balance, diversity, and missing class penalty
        let normalized_balance_penalty = balance_score / 20.0;
        let composite_quality_score = self.balance_weight * normalized_balance_penalty
            + self.diversity_weight * (1.0 - diversity_score)
            + missing_class_penalty; // CRITICAL: Add missing class penalty

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

    /// Calculate feature space diversity using PCA-like coverage across multiple features
    /// Uses quintile-based sampling (5 regions) to measure feature space coverage
    /// Returns 0.0 (poor) to 1.0 (excellent feature coverage)
    ///
    /// **Key Improvement**: Uses 8 derived features and quintile (5-bin) coverage
    /// to ensure calibration samples cover diverse regions of feature space.
    /// This helps find parameters that work well across different market patterns.
    pub fn calculate_feature_space_diversity(
        ohlcv_data: &[MarketDataRow],
        sample_indices: &[usize],
    ) -> f64 {
        if sample_indices.is_empty() || ohlcv_data.is_empty() {
            return 0.0;
        }

        // Extract multiple derived features for each sample
        // Using 8 features for better coverage of the feature space
        let mut feature_vectors: Vec<Vec<f64>> = Vec::new();

        for &idx in sample_indices {
            if idx < ohlcv_data.len() {
                let candle = &ohlcv_data[idx];

                // Feature 1: Price range (normalized)
                let price_range = if candle.close > 0.0 {
                    (candle.high - candle.low) / candle.close
                } else {
                    0.0
                };

                // Feature 2: Body size (normalized)
                let body_size = if candle.close > 0.0 {
                    (candle.close - candle.open).abs() / candle.close
                } else {
                    0.0
                };

                // Feature 3: Upper wick
                let upper_wick = if candle.high > 0.0 && candle.close > 0.0 {
                    (candle.high - candle.close.max(candle.open)) / candle.close
                } else {
                    0.0
                };

                // Feature 4: Lower wick
                let lower_wick = if candle.low > 0.0 && candle.close > 0.0 {
                    (candle.open.min(candle.close) - candle.low) / candle.close
                } else {
                    0.0
                };

                // Feature 5: Volume intensity (normalized log)
                let volume_intensity = if candle.volume > 0.0 {
                    candle.volume.ln()
                } else {
                    0.0
                };

                // Feature 6: Price position in candle (0-1, top of wick to bottom)
                let price_position = if candle.high > candle.low {
                    (candle.close - candle.low) / (candle.high - candle.low)
                } else {
                    0.5
                };

                // Feature 7: Trend direction (body sign)
                let trend_direction = if candle.close > candle.open { 1.0 } else { 0.0 };

                // Feature 8: Candle size relative to range (0-1)
                let candle_relative_size = if price_range > 0.0 {
                    body_size / price_range
                } else {
                    0.5
                };

                feature_vectors.push(vec![
                    price_range,
                    body_size,
                    upper_wick,
                    lower_wick,
                    volume_intensity,
                    price_position,
                    trend_direction,
                    candle_relative_size,
                ]);
            }
        }

        if feature_vectors.is_empty() {
            return 0.0;
        }

        // Calculate quintile (5-bin) coverage for each feature dimension
        let num_features = feature_vectors[0].len();
        let mut total_quintile_coverage = 0.0;

        for dim in 0..num_features {
            let mut values: Vec<f64> = feature_vectors.iter().map(|v| v[dim]).collect();
            let coverage = Self::calculate_quintile_coverage(&mut values);
            total_quintile_coverage += coverage;
        }

        // Average coverage across all features
        total_quintile_coverage / num_features as f64
    }

    /// Calculate quintile (5-bin) coverage for diversity measurement
    /// Returns 0.0 (all samples in same quintile) to 1.0 (uniformly distributed)
    ///
    /// **Why Quintiles?**
    /// - Matches our 5-class classification system
    /// - Ensures samples cover all regions of feature space
    /// - Penalizes clustering in one or few regions
    fn calculate_quintile_coverage(values: &mut [f64]) -> f64 {
        if values.len() < 5 {
            return 0.0;
        }

        // Clone first since sort consumes
        let min_val = *values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);
        let max_val = *values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&1.0);
        let range = max_val - min_val + 1e-10;

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // After sort, use index-based access to avoid borrowing issues
        let n = values.len();
        let mut quintile_counts = [0usize; 5];

        // Use indices instead of iterating to avoid borrowing conflict after sort
        for i in 0..n {
            let percentile = (values[i] - min_val) / range;
            let bin = (percentile * 5.0).floor() as usize;
            let bin = bin.min(4);
            quintile_counts[bin] += 1;
        }

        // Calculate how evenly distributed samples are across quintiles
        // Ideal: equal distribution (20% each)
        let total = values.len() as f64;
        let ideal = 0.2;

        // Calculate deviation from ideal distribution
        let deviation: f64 = quintile_counts
            .iter()
            .map(|&count| ((count as f64 / total) - ideal).abs())
            .sum::<f64>()
            / 5.0;

        // Convert deviation to coverage score
        // 0 deviation = 1.0 (perfect coverage)
        // 0.2 deviation = 0.0 (all in one bin)
        (1.0 - deviation * 5.0).clamp(0.0, 1.0)
    }

    /// Calculate market condition diversity (bull/bear/sideways distribution)
    /// Uses ADAPTIVE percentile-based thresholds for symbol-agnostic classification
    /// Returns 0.0 (poor) to 1.0 (excellent condition balance)
    pub fn calculate_market_condition_diversity(
        ohlcv_data: &[MarketDataRow],
        sample_indices: &[usize],
        sequence_length: usize,
    ) -> f64 {
        if sample_indices.len() < 10 || ohlcv_data.len() < 20 {
            return 0.0;
        }

        // Step 1: Calculate ALL sequence trends to find adaptive thresholds
        let mut all_changes: Vec<f64> = Vec::new();
        for &idx in sample_indices {
            if idx + sequence_length <= ohlcv_data.len() {
                let start_price = ohlcv_data[idx].close;
                let end_price = ohlcv_data[idx + sequence_length - 1].close;
                let change_pct = (end_price - start_price) / start_price;
                all_changes.push(change_pct);
            }
        }

        if all_changes.is_empty() {
            return 0.0;
        }

        // Step 2: Find adaptive thresholds using percentiles (33rd and 67th)
        all_changes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p33_idx = (all_changes.len() as f64 * 0.33) as usize;
        let p67_idx = (all_changes.len() as f64 * 0.67) as usize;

        let bear_threshold = all_changes[p33_idx]; // Bottom 33% = bear
        let bull_threshold = all_changes[p67_idx]; // Top 33% = bull

        // Step 3: Classify using adaptive thresholds
        let mut bull_count = 0;
        let mut bear_count = 0;
        let mut sideways_count = 0;

        for &change_pct in &all_changes {
            if change_pct >= bull_threshold {
                bull_count += 1;
            } else if change_pct <= bear_threshold {
                bear_count += 1;
            } else {
                sideways_count += 1;
            }
        }

        let total = all_changes.len() as f64;
        let bull_pct = bull_count as f64 / total;
        let bear_pct = bear_count as f64 / total;
        let sideways_pct = sideways_count as f64 / total;

        log::debug!(
            "Market conditions (adaptive): Bull={:.1}%, Bear={:.1}%, Sideways={:.1}% (thresholds: bear<{:.4}, bull>{:.4}, {} samples)",
            bull_pct * 100.0,
            bear_pct * 100.0,
            sideways_pct * 100.0,
            bear_threshold,
            bull_threshold,
            all_changes.len()
        );

        // Step 4: Calculate diversity (ideal: 33.3% each)
        let ideal = 1.0 / 3.0;
        let deviation =
            ((bull_pct - ideal).abs() + (bear_pct - ideal).abs() + (sideways_pct - ideal).abs())
                / 3.0;

        // Convert deviation to diversity score (0 deviation = 1.0 diversity)
        (1.0 - deviation * 3.0).clamp(0.0, 1.0)
    }
}
