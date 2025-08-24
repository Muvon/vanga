//! Calibration Utility Functions
//!
//! Shared utility functions used across all calibration modules.
//! Contains balance calculation and other common calibration helpers.

use super::types::ClassBalance;
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

    /// Calculate class balance metrics with diversity scoring
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

        // Calculate diversity metrics with default values
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
}
