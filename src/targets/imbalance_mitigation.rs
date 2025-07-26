//! Class imbalance mitigation strategies for price level classification
//!
//! This module provides advanced techniques to handle severe class imbalance
//! in cryptocurrency price level prediction, including adaptive bandwidth sizing,
//! sophisticated class weighting, and data augmentation strategies.

use crate::targets::price_levels::PriceLevelConfig;
use crate::utils::error::Result;

/// Imbalance severity levels for different mitigation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ImbalanceSeverity {
    /// Mild imbalance (2-10x ratio) - standard class weighting sufficient
    Mild,
    /// Moderate imbalance (10-50x ratio) - enhanced weighting + minor adjustments
    Moderate,
    /// Severe imbalance (50-500x ratio) - adaptive bandwidth + advanced weighting
    Severe,
    /// Extreme imbalance (500x+ ratio) - comprehensive mitigation required
    Extreme,
}

/// Advanced class weighting strategies for extreme imbalance
#[derive(Debug, Clone)]
pub enum AdvancedWeightingStrategy {
    /// Standard sklearn-style balanced weighting
    Balanced,
    /// Focal loss inspired weighting with gamma parameter
    FocalLoss { gamma: f32 },
    /// Effective number of samples weighting
    EffectiveNumber { beta: f32 },
    /// Custom capped weighting to prevent gradient explosion
    CappedBalanced { max_weight_ratio: f32 },
}

/// Configuration for imbalance mitigation
#[derive(Debug, Clone)]
pub struct ImbalanceMitigationConfig {
    /// Maximum allowed imbalance ratio before intervention
    pub max_imbalance_ratio: f64,
    /// Minimum samples per class before considering it problematic
    pub min_samples_per_class: usize,
    /// Advanced weighting strategy to use
    pub weighting_strategy: AdvancedWeightingStrategy,
    /// Whether to enable adaptive bandwidth sizing
    pub enable_adaptive_bandwidth: bool,
    /// Target class distribution balance (0.0 = no adjustment, 1.0 = perfect balance)
    pub balance_target: f32,
}

impl Default for ImbalanceMitigationConfig {
    fn default() -> Self {
        Self {
            max_imbalance_ratio: 100.0, // Trigger mitigation at 100x imbalance
            min_samples_per_class: 10,  // Need at least 10 samples per class
            weighting_strategy: AdvancedWeightingStrategy::CappedBalanced {
                max_weight_ratio: 50.0,
            },
            enable_adaptive_bandwidth: true,
            balance_target: 0.3, // Moderate balancing
        }
    }
}

/// Class distribution analysis results
#[derive(Debug, Clone)]
pub struct ClassDistributionAnalysis {
    pub class_counts: Vec<usize>,
    pub class_percentages: Vec<f64>,
    pub total_samples: usize,
    pub imbalance_ratio: f64,
    pub severity: ImbalanceSeverity,
    pub empty_classes: Vec<usize>,
    pub rare_classes: Vec<usize>, // Classes with < min_samples_per_class
}

impl ClassDistributionAnalysis {
    /// Analyze class distribution and determine severity
    pub fn analyze(
        targets: &[i32],
        num_classes: usize,
        config: &ImbalanceMitigationConfig,
    ) -> Self {
        let mut class_counts = vec![0usize; num_classes];
        let mut total_samples = 0;

        // Count class occurrences
        for &target in targets {
            if target >= 0 && target < num_classes as i32 {
                class_counts[target as usize] += 1;
                total_samples += 1;
            }
        }

        // Calculate percentages
        let class_percentages: Vec<f64> = class_counts
            .iter()
            .map(|&count| (count as f64 / total_samples as f64) * 100.0)
            .collect();

        // Find min/max for imbalance ratio
        let min_count = class_counts.iter().filter(|&&c| c > 0).min().unwrap_or(&1);
        let max_count = class_counts.iter().max().unwrap_or(&1);
        let imbalance_ratio = *max_count as f64 / *min_count as f64;

        // Determine severity
        let severity = match imbalance_ratio {
            r if r < 10.0 => ImbalanceSeverity::Mild,
            r if r < 50.0 => ImbalanceSeverity::Moderate,
            r if r < 500.0 => ImbalanceSeverity::Severe,
            _ => ImbalanceSeverity::Extreme,
        };

        // Identify problematic classes
        let empty_classes: Vec<usize> = class_counts
            .iter()
            .enumerate()
            .filter(|(_, &count)| count == 0)
            .map(|(idx, _)| idx)
            .collect();

        let rare_classes: Vec<usize> = class_counts
            .iter()
            .enumerate()
            .filter(|(_, &count)| count > 0 && count < config.min_samples_per_class)
            .map(|(idx, _)| idx)
            .collect();

        Self {
            class_counts,
            class_percentages,
            total_samples,
            imbalance_ratio,
            severity,
            empty_classes,
            rare_classes,
        }
    }
}

/// Adaptive bandwidth sizing based on class distribution
pub struct AdaptiveBandwidthSizer;

impl AdaptiveBandwidthSizer {
    /// Calculate optimal bandwidth_size to improve class balance
    pub fn calculate_adaptive_bandwidth(
        analysis: &ClassDistributionAnalysis,
        current_bandwidth: f64,
        config: &ImbalanceMitigationConfig,
    ) -> f64 {
        if !config.enable_adaptive_bandwidth {
            return current_bandwidth;
        }

        match analysis.severity {
            ImbalanceSeverity::Mild => current_bandwidth,
            ImbalanceSeverity::Moderate => {
                // Slightly reduce bandwidth to create more breakouts
                current_bandwidth * 0.8
            }
            ImbalanceSeverity::Severe => {
                // Significantly reduce bandwidth for more balanced distribution
                current_bandwidth * 0.6
            }
            ImbalanceSeverity::Extreme => {
                // Aggressive bandwidth reduction for extreme cases
                let breakout_classes_empty =
                    analysis.empty_classes.contains(&0) || analysis.empty_classes.contains(&5);
                if breakout_classes_empty {
                    // Very aggressive reduction to force breakout generation
                    current_bandwidth * 0.4
                } else {
                    current_bandwidth * 0.5
                }
            }
        }
    }

    /// Suggest bandwidth_size range for experimentation
    pub fn suggest_bandwidth_range(analysis: &ClassDistributionAnalysis) -> (f64, f64, f64) {
        match analysis.severity {
            ImbalanceSeverity::Mild => (0.8, 1.0, 1.2), // Conservative range
            ImbalanceSeverity::Moderate => (0.6, 0.8, 1.0), // Moderate reduction
            ImbalanceSeverity::Severe => (0.4, 0.6, 0.8), // Significant reduction
            ImbalanceSeverity::Extreme => (0.2, 0.4, 0.6), // Aggressive reduction
        }
    }
}

/// Advanced class weighting calculator
pub struct AdvancedClassWeighter;

impl AdvancedClassWeighter {
    /// Calculate class weights using advanced strategies
    pub fn calculate_weights(
        analysis: &ClassDistributionAnalysis,
        strategy: &AdvancedWeightingStrategy,
    ) -> Result<Vec<f32>> {
        match strategy {
            AdvancedWeightingStrategy::Balanced => Self::calculate_balanced_weights(analysis),
            AdvancedWeightingStrategy::FocalLoss { gamma } => {
                Self::calculate_focal_weights(analysis, *gamma)
            }
            AdvancedWeightingStrategy::EffectiveNumber { beta } => {
                Self::calculate_effective_number_weights(analysis, *beta)
            }
            AdvancedWeightingStrategy::CappedBalanced { max_weight_ratio } => {
                Self::calculate_capped_balanced_weights(analysis, *max_weight_ratio)
            }
        }
    }

    /// Standard balanced weighting (sklearn style)
    fn calculate_balanced_weights(analysis: &ClassDistributionAnalysis) -> Result<Vec<f32>> {
        let mut weights = vec![1.0f32; analysis.class_counts.len()];
        let total_samples = analysis.total_samples as f32;
        let num_classes = analysis.class_counts.len() as f32;

        for (i, &count) in analysis.class_counts.iter().enumerate() {
            if count > 0 {
                weights[i] = total_samples / (num_classes * count as f32);
            } else {
                // Handle empty classes with very high weight
                weights[i] = total_samples / num_classes; // Fallback weight
            }
        }

        Ok(weights)
    }

    /// Capped balanced weighting to prevent gradient explosion
    fn calculate_capped_balanced_weights(
        analysis: &ClassDistributionAnalysis,
        max_weight_ratio: f32,
    ) -> Result<Vec<f32>> {
        let balanced_weights = Self::calculate_balanced_weights(analysis)?;

        // Find minimum weight to calculate ratios
        let min_weight = balanced_weights
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);

        // Cap weights to prevent extreme ratios
        let capped_weights: Vec<f32> = balanced_weights
            .iter()
            .map(|&weight| {
                let ratio = weight / min_weight;
                if ratio > max_weight_ratio {
                    min_weight * max_weight_ratio
                } else {
                    weight
                }
            })
            .collect();

        Ok(capped_weights)
    }

    /// Focal loss inspired weighting
    fn calculate_focal_weights(
        analysis: &ClassDistributionAnalysis,
        gamma: f32,
    ) -> Result<Vec<f32>> {
        let mut weights = vec![1.0f32; analysis.class_counts.len()];

        for (i, &percentage) in analysis.class_percentages.iter().enumerate() {
            let p = percentage / 100.0; // Convert to probability
            if p > 0.0 {
                // Focal loss weighting: (1-p)^gamma
                weights[i] = (1.0 - p as f32).powf(gamma);
            } else {
                weights[i] = 1.0; // Maximum weight for empty classes
            }
        }

        Ok(weights)
    }

    /// Effective number of samples weighting
    fn calculate_effective_number_weights(
        analysis: &ClassDistributionAnalysis,
        beta: f32,
    ) -> Result<Vec<f32>> {
        let mut weights = vec![1.0f32; analysis.class_counts.len()];

        for (i, &count) in analysis.class_counts.iter().enumerate() {
            if count > 0 {
                // Effective number: (1 - beta^n) / (1 - beta)
                let n = count as f32;
                let effective_num = (1.0 - beta.powf(n)) / (1.0 - beta);
                weights[i] = 1.0 / effective_num;
            } else {
                weights[i] = 1.0; // High weight for empty classes
            }
        }

        // Normalize weights
        let weights_sum: f32 = weights.iter().sum();
        let weights_len = weights.len() as f32;
        for weight in &mut weights {
            *weight = (*weight / weights_sum) * weights_len;
        }

        Ok(weights)
    }
}

/// Comprehensive imbalance mitigation recommendations
pub struct ImbalanceMitigator;

impl ImbalanceMitigator {
    /// Generate comprehensive mitigation recommendations
    pub fn generate_recommendations(
        analysis: &ClassDistributionAnalysis,
        current_config: &PriceLevelConfig,
        mitigation_config: &ImbalanceMitigationConfig,
    ) -> ImbalanceMitigationRecommendations {
        let adaptive_bandwidth = AdaptiveBandwidthSizer::calculate_adaptive_bandwidth(
            analysis,
            current_config.bandwidth_size,
            mitigation_config,
        );

        let bandwidth_range = AdaptiveBandwidthSizer::suggest_bandwidth_range(analysis);

        let recommended_weights = AdvancedClassWeighter::calculate_weights(
            analysis,
            &mitigation_config.weighting_strategy,
        )
        .unwrap_or_else(|_| vec![1.0; analysis.class_counts.len()]);

        ImbalanceMitigationRecommendations {
            severity: analysis.severity.clone(),
            current_imbalance_ratio: analysis.imbalance_ratio,
            recommended_bandwidth_size: adaptive_bandwidth,
            bandwidth_experimentation_range: bandwidth_range,
            recommended_class_weights: recommended_weights,
            empty_classes: analysis.empty_classes.clone(),
            rare_classes: analysis.rare_classes.clone(),
            mitigation_strategies: Self::suggest_strategies(&analysis.severity),
        }
    }

    fn suggest_strategies(severity: &ImbalanceSeverity) -> Vec<String> {
        match severity {
            ImbalanceSeverity::Mild => vec![
                "Use standard class weighting".to_string(),
                "Monitor class distribution".to_string(),
            ],
            ImbalanceSeverity::Moderate => vec![
                "Apply enhanced class weighting".to_string(),
                "Consider slight bandwidth reduction".to_string(),
                "Monitor training stability".to_string(),
            ],
            ImbalanceSeverity::Severe => vec![
                "Implement adaptive bandwidth sizing".to_string(),
                "Use capped balanced class weighting".to_string(),
                "Consider data augmentation for rare classes".to_string(),
                "Implement focal loss weighting".to_string(),
            ],
            ImbalanceSeverity::Extreme => vec![
                "URGENT: Aggressive bandwidth reduction required".to_string(),
                "Implement comprehensive class weighting strategy".to_string(),
                "Consider synthetic data generation for empty classes".to_string(),
                "Use effective number of samples weighting".to_string(),
                "Implement class-aware data augmentation".to_string(),
                "Consider ensemble methods with different bandwidth sizes".to_string(),
            ],
        }
    }
}

/// Comprehensive mitigation recommendations
#[derive(Debug, Clone)]
pub struct ImbalanceMitigationRecommendations {
    pub severity: ImbalanceSeverity,
    pub current_imbalance_ratio: f64,
    pub recommended_bandwidth_size: f64,
    pub bandwidth_experimentation_range: (f64, f64, f64), // (min, recommended, max)
    pub recommended_class_weights: Vec<f32>,
    pub empty_classes: Vec<usize>,
    pub rare_classes: Vec<usize>,
    pub mitigation_strategies: Vec<String>,
}

impl ImbalanceMitigationRecommendations {
    /// Log comprehensive recommendations
    pub fn log_recommendations(&self, horizon: &str) {
        log::warn!(
            "🚨 IMBALANCE MITIGATION RECOMMENDATIONS for horizon {}: {:.1}x ratio ({:?})",
            horizon,
            self.current_imbalance_ratio,
            self.severity
        );

        log::info!(
            "📊 Recommended bandwidth_size: {:.2} (current range: {:.2}-{:.2})",
            self.recommended_bandwidth_size,
            self.bandwidth_experimentation_range.0,
            self.bandwidth_experimentation_range.2
        );

        if !self.empty_classes.is_empty() {
            log::warn!(
                "⚠️  Empty classes detected: {:?} - consider aggressive bandwidth reduction",
                self.empty_classes
            );
        }

        if !self.rare_classes.is_empty() {
            log::warn!(
                "⚠️  Rare classes detected: {:?} - consider data augmentation",
                self.rare_classes
            );
        }

        log::info!(
            "🎯 Recommended class weights: {:?}",
            self.recommended_class_weights
        );

        log::info!("💡 Mitigation strategies:");
        for (i, strategy) in self.mitigation_strategies.iter().enumerate() {
            log::info!("   {}. {}", i + 1, strategy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_class_distribution_analysis() {
        let targets = vec![2, 2, 2, 3, 3, 5]; // Imbalanced distribution
        let config = ImbalanceMitigationConfig::default();
        let analysis = ClassDistributionAnalysis::analyze(&targets, 6, &config);

        assert_eq!(analysis.total_samples, 6);
        assert_eq!(analysis.class_counts[2], 3);
        assert_eq!(analysis.class_counts[5], 1);
        assert!(analysis.imbalance_ratio > 1.0);
        assert!(analysis.empty_classes.contains(&0));
    }

    #[test]
    fn test_adaptive_bandwidth_sizing() {
        // Create extreme imbalance: 999 samples in class 2, 1 sample in class 0
        let mut targets = vec![2; 999];
        targets.push(0);

        let config = ImbalanceMitigationConfig::default();
        let analysis = ClassDistributionAnalysis::analyze(&targets, 6, &config);

        let adaptive_bandwidth =
            AdaptiveBandwidthSizer::calculate_adaptive_bandwidth(&analysis, 1.0, &config);

        assert!(adaptive_bandwidth < 1.0); // Should reduce bandwidth
        assert_eq!(analysis.severity, ImbalanceSeverity::Extreme);
    }

    #[test]
    fn test_capped_balanced_weights() {
        let targets = vec![2, 2, 2, 2, 2, 5]; // 5:1 ratio
        let config = ImbalanceMitigationConfig::default();
        let analysis = ClassDistributionAnalysis::analyze(&targets, 6, &config);

        let weights =
            AdvancedClassWeighter::calculate_capped_balanced_weights(&analysis, 10.0).unwrap();

        // Verify weights are capped
        let min_weight = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_weight = weights.iter().cloned().fold(0.0f32, f32::max);
        assert!((max_weight / min_weight) <= 10.0);
    }
}
