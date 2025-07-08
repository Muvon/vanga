// TFT Auto-Optimization - Intelligent parameter tuning for Variable Selection and Quantile Regression
use crate::config::model::{TFTQuantileOutputConfig, TFTVariableSelectionConfig};
use crate::utils::error::{Result, VangaError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Auto-optimization configuration for TFT components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TFTAutoOptimizerConfig {
    /// Enable automatic parameter tuning
    pub enabled: bool,
    /// Variable selection optimization settings
    pub variable_selection: VariableSelectionOptimizer,
    /// Quantile regression optimization settings
    pub quantile_regression: QuantileRegressionOptimizer,
    /// Training integration settings
    pub training_integration: TrainingIntegrationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableSelectionOptimizer {
    /// Auto-tune selection threshold based on feature importance distribution
    pub auto_tune_threshold: bool,
    /// Dynamically adjust top_k_features based on data characteristics
    pub dynamic_top_k: bool,
    /// Minimum threshold value
    pub min_threshold: f64,
    /// Maximum threshold value
    pub max_threshold: f64,
    /// Minimum number of features to select
    pub min_features: usize,
    /// Maximum number of features to select
    pub max_features: usize,
    /// Feature importance analysis window
    pub analysis_window: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileRegressionOptimizer {
    /// Auto-select optimal quantile levels based on data distribution
    pub auto_select_quantiles: bool,
    /// Dynamically adjust loss weighting based on prediction accuracy
    pub dynamic_loss_weighting: bool,
    /// Minimum number of quantiles
    pub min_quantiles: usize,
    /// Maximum number of quantiles
    pub max_quantiles: usize,
    /// Quantile selection strategy
    pub selection_strategy: QuantileSelectionStrategy,
    /// Loss weighting adaptation rate
    pub weighting_adaptation_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantileSelectionStrategy {
    /// Symmetric quantiles around median (e.g., 0.1, 0.25, 0.5, 0.75, 0.9)
    Symmetric,
    /// Focus on extreme quantiles for risk management
    ExtremeWeighted,
    /// Adaptive based on data volatility
    VolatilityAdaptive,
    /// Custom quantile levels
    Custom(Vec<f64>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingIntegrationConfig {
    /// Enable TFT features during training
    pub enable_during_training: bool,
    /// Validation-based parameter adjustment
    pub validation_based_tuning: bool,
    /// Early stopping based on TFT metrics
    pub tft_early_stopping: bool,
    /// Model comparison with baseline LSTM
    pub baseline_comparison: bool,
    /// Performance tracking interval (epochs)
    pub tracking_interval: usize,
}

impl Default for TFTAutoOptimizerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            variable_selection: VariableSelectionOptimizer::default(),
            quantile_regression: QuantileRegressionOptimizer::default(),
            training_integration: TrainingIntegrationConfig::default(),
        }
    }
}

impl Default for VariableSelectionOptimizer {
    fn default() -> Self {
        Self {
            auto_tune_threshold: true,
            dynamic_top_k: true,
            min_threshold: 0.05,
            max_threshold: 0.3,
            min_features: 5,
            max_features: 50,
            analysis_window: 100,
        }
    }
}

impl Default for QuantileRegressionOptimizer {
    fn default() -> Self {
        Self {
            auto_select_quantiles: true,
            dynamic_loss_weighting: true,
            min_quantiles: 3,
            max_quantiles: 9,
            selection_strategy: QuantileSelectionStrategy::VolatilityAdaptive,
            weighting_adaptation_rate: 0.01,
        }
    }
}

impl Default for TrainingIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_during_training: true,
            validation_based_tuning: true,
            tft_early_stopping: true,
            baseline_comparison: true,
            tracking_interval: 10,
        }
    }
}

/// TFT Auto-Optimizer for intelligent parameter tuning
#[allow(dead_code)]
pub struct TFTAutoOptimizer {
    config: TFTAutoOptimizerConfig,
    /// Feature importance history for threshold tuning
    importance_history: Vec<Vec<f64>>,
    /// Quantile performance tracking
    quantile_performance: HashMap<String, f64>,
    /// Training metrics history
    training_metrics: Vec<TrainingMetrics>,
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub validation_loss: f64,
    pub tft_variable_selection_score: f64,
    pub quantile_coverage: f64,
    pub feature_importance_entropy: f64,
}

impl TFTAutoOptimizer {
    /// Create new auto-optimizer
    pub fn new(config: TFTAutoOptimizerConfig) -> Self {
        log::info!("Initializing TFT Auto-Optimizer with config: {:?}", config);

        Self {
            config,
            importance_history: Vec::new(),
            quantile_performance: HashMap::new(),
            training_metrics: Vec::new(),
        }
    }

    /// Optimize variable selection configuration based on data characteristics
    pub fn optimize_variable_selection(
        &mut self,
        base_config: &TFTVariableSelectionConfig,
        feature_importance: &[f64],
        data_characteristics: &DataCharacteristics,
    ) -> Result<TFTVariableSelectionConfig> {
        if !self.config.variable_selection.auto_tune_threshold {
            return Ok(base_config.clone());
        }

        let mut optimized_config = base_config.clone();

        // Auto-tune selection threshold based on feature importance distribution
        let optimal_threshold = self.calculate_optimal_threshold(feature_importance)?;
        optimized_config.selection_threshold = optimal_threshold;

        // Dynamic top_k adjustment based on data characteristics
        if self.config.variable_selection.dynamic_top_k {
            let optimal_k =
                self.calculate_optimal_top_k(feature_importance, data_characteristics)?;
            optimized_config.top_k_features = Some(optimal_k);
        }

        // Store importance history for future optimization
        self.importance_history.push(feature_importance.to_vec());
        if self.importance_history.len() > self.config.variable_selection.analysis_window {
            self.importance_history.remove(0);
        }

        log::debug!(
            "Optimized variable selection: threshold={:.3}, top_k={:?}",
            optimized_config.selection_threshold,
            optimized_config.top_k_features
        );

        Ok(optimized_config)
    }

    /// Optimize quantile regression configuration
    pub fn optimize_quantile_regression(
        &mut self,
        base_config: &TFTQuantileOutputConfig,
        data_characteristics: &DataCharacteristics,
        validation_metrics: Option<&QuantileValidationMetrics>,
    ) -> Result<TFTQuantileOutputConfig> {
        if !self.config.quantile_regression.auto_select_quantiles {
            return Ok(base_config.clone());
        }

        let mut optimized_config = base_config.clone();

        // Auto-select optimal quantile levels
        let optimal_quantiles = self.select_optimal_quantiles(
            data_characteristics,
            &self.config.quantile_regression.selection_strategy,
        )?;
        optimized_config.quantiles = optimal_quantiles;

        // Dynamic loss weighting adjustment
        if self.config.quantile_regression.dynamic_loss_weighting {
            if let Some(metrics) = validation_metrics {
                let optimal_weighting = self.adjust_loss_weighting(metrics)?;
                optimized_config.loss_weighting = optimal_weighting;
            }
        }

        log::debug!(
            "Optimized quantile regression: {} quantiles, weighting={:?}",
            optimized_config.quantiles.len(),
            optimized_config.loss_weighting
        );

        Ok(optimized_config)
    }

    /// Calculate optimal threshold based on feature importance distribution
    fn calculate_optimal_threshold(&self, importance: &[f64]) -> Result<f64> {
        if importance.is_empty() {
            return Err(VangaError::ConfigError(
                "Empty feature importance array".to_string(),
            ));
        }

        // Calculate statistics
        let mean = importance.iter().sum::<f64>() / importance.len() as f64;
        let variance =
            importance.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / importance.len() as f64;
        let std_dev = variance.sqrt();

        // Use mean + 0.5 * std_dev as threshold (captures significant features)
        let threshold = mean + 0.5 * std_dev;

        // Clamp to configured bounds
        let clamped_threshold = threshold
            .max(self.config.variable_selection.min_threshold)
            .min(self.config.variable_selection.max_threshold);

        Ok(clamped_threshold)
    }

    /// Calculate optimal number of top features
    fn calculate_optimal_top_k(
        &self,
        importance: &[f64],
        data_characteristics: &DataCharacteristics,
    ) -> Result<usize> {
        // Base calculation on data size and feature count
        let base_k = (data_characteristics.feature_count as f64 * 0.4) as usize;

        // Adjust based on data quality
        let quality_multiplier = if data_characteristics.data_quality > 0.8 {
            1.2 // More features for high-quality data
        } else if data_characteristics.data_quality < 0.5 {
            0.7 // Fewer features for low-quality data
        } else {
            1.0
        };

        let adjusted_k = (base_k as f64 * quality_multiplier) as usize;

        // Clamp to configured bounds
        let clamped_k = adjusted_k
            .max(self.config.variable_selection.min_features)
            .min(self.config.variable_selection.max_features)
            .min(importance.len()); // Can't select more features than available

        Ok(clamped_k)
    }

    /// Select optimal quantile levels based on strategy
    fn select_optimal_quantiles(
        &self,
        data_characteristics: &DataCharacteristics,
        strategy: &QuantileSelectionStrategy,
    ) -> Result<Vec<f64>> {
        match strategy {
            QuantileSelectionStrategy::Symmetric => {
                // Standard symmetric quantiles
                Ok(vec![0.05, 0.25, 0.5, 0.75, 0.95])
            }
            QuantileSelectionStrategy::ExtremeWeighted => {
                // Focus on extreme quantiles for risk management
                Ok(vec![0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
            }
            QuantileSelectionStrategy::VolatilityAdaptive => {
                // Adapt based on data volatility
                let volatility = data_characteristics.volatility;
                if volatility > 0.1 {
                    // High volatility: more extreme quantiles
                    Ok(vec![0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
                } else if volatility > 0.05 {
                    // Medium volatility: standard quantiles
                    Ok(vec![0.05, 0.25, 0.5, 0.75, 0.95])
                } else {
                    // Low volatility: fewer quantiles
                    Ok(vec![0.1, 0.5, 0.9])
                }
            }
            QuantileSelectionStrategy::Custom(quantiles) => Ok(quantiles.clone()),
        }
    }

    /// Adjust loss weighting based on validation metrics
    fn adjust_loss_weighting(&mut self, metrics: &QuantileValidationMetrics) -> Result<String> {
        // Analyze quantile coverage and adjust weighting
        let coverage_deviation = metrics.coverage_deviation();

        if coverage_deviation > 0.1 {
            // Poor coverage: increase extreme quantile weighting
            Ok("extreme_weighted".to_string())
        } else if coverage_deviation < 0.02 {
            // Good coverage: use balanced weighting
            Ok("balanced".to_string())
        } else {
            // Medium coverage: use moderate weighting
            Ok("moderate_weighted".to_string())
        }
    }

    /// Update training metrics for optimization
    pub fn update_training_metrics(&mut self, metrics: TrainingMetrics) {
        self.training_metrics.push(metrics);

        // Keep only recent metrics for efficiency
        if self.training_metrics.len() > 1000 {
            self.training_metrics.remove(0);
        }
    }

    /// Check if early stopping should be triggered based on TFT metrics
    pub fn should_early_stop(&self, patience: usize) -> bool {
        if !self.config.training_integration.tft_early_stopping {
            return false;
        }

        if self.training_metrics.len() < patience + 1 {
            return false;
        }

        // Check if TFT metrics have stopped improving
        let recent_metrics = &self.training_metrics[self.training_metrics.len() - patience..];
        let best_score = recent_metrics
            .iter()
            .map(|m| m.tft_variable_selection_score)
            .fold(f64::NEG_INFINITY, f64::max);

        let latest_score = recent_metrics.last().unwrap().tft_variable_selection_score;

        // Stop if no improvement in recent epochs
        latest_score < best_score * 0.995 // 0.5% improvement threshold
    }
}

/// Data characteristics for optimization
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    pub feature_count: usize,
    pub sample_count: usize,
    pub volatility: f64,
    pub data_quality: f64, // 0.0 to 1.0
    pub missing_data_ratio: f64,
    pub correlation_structure: f64, // Average feature correlation
}

/// Quantile validation metrics
#[derive(Debug, Clone)]
pub struct QuantileValidationMetrics {
    pub quantile_coverages: HashMap<String, f64>,
    pub prediction_intervals: HashMap<String, f64>,
    pub calibration_scores: HashMap<String, f64>,
}

impl QuantileValidationMetrics {
    /// Calculate overall coverage deviation from expected
    pub fn coverage_deviation(&self) -> f64 {
        self.quantile_coverages
            .values()
            .map(|&coverage| (coverage - 0.5).abs()) // Deviation from 50% coverage
            .sum::<f64>()
            / self.quantile_coverages.len() as f64
    }
}

/// Factory for creating optimized TFT configurations
pub struct TFTOptimizerFactory;

impl TFTOptimizerFactory {
    /// Create crypto-optimized auto-optimizer
    pub fn crypto_optimized() -> TFTAutoOptimizerConfig {
        TFTAutoOptimizerConfig {
            enabled: true,
            variable_selection: VariableSelectionOptimizer {
                auto_tune_threshold: true,
                dynamic_top_k: true,
                min_threshold: 0.1, // Higher threshold for crypto noise
                max_threshold: 0.4,
                min_features: 8,
                max_features: 30, // Focused feature set for crypto
                analysis_window: 50,
            },
            quantile_regression: QuantileRegressionOptimizer {
                auto_select_quantiles: true,
                dynamic_loss_weighting: true,
                min_quantiles: 5,
                max_quantiles: 9,
                selection_strategy: QuantileSelectionStrategy::ExtremeWeighted,
                weighting_adaptation_rate: 0.02, // Faster adaptation for crypto
            },
            training_integration: TrainingIntegrationConfig {
                enable_during_training: true,
                validation_based_tuning: true,
                tft_early_stopping: true,
                baseline_comparison: true,
                tracking_interval: 5, // More frequent tracking
            },
        }
    }

    /// Create conservative auto-optimizer for stable assets
    pub fn conservative() -> TFTAutoOptimizerConfig {
        TFTAutoOptimizerConfig {
            enabled: true,
            variable_selection: VariableSelectionOptimizer {
                auto_tune_threshold: true,
                dynamic_top_k: false, // Fixed feature count for stability
                min_threshold: 0.05,
                max_threshold: 0.2,
                min_features: 10,
                max_features: 25,
                analysis_window: 200,
            },
            quantile_regression: QuantileRegressionOptimizer {
                auto_select_quantiles: true,
                dynamic_loss_weighting: false, // Fixed weighting for stability
                min_quantiles: 3,
                max_quantiles: 7,
                selection_strategy: QuantileSelectionStrategy::Symmetric,
                weighting_adaptation_rate: 0.005,
            },
            training_integration: TrainingIntegrationConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tft_auto_optimizer_creation() {
        let config = TFTAutoOptimizerConfig::default();
        let optimizer = TFTAutoOptimizer::new(config);
        assert!(optimizer.importance_history.is_empty());
    }

    #[test]
    fn test_optimal_threshold_calculation() {
        let config = TFTAutoOptimizerConfig::default();
        let optimizer = TFTAutoOptimizer::new(config);

        let importance = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let threshold = optimizer.calculate_optimal_threshold(&importance).unwrap();

        assert!(threshold > 0.0);
        assert!(threshold <= 0.3); // Should be within max threshold
    }

    #[test]
    fn test_quantile_selection_strategies() {
        let config = TFTAutoOptimizerConfig::default();
        let optimizer = TFTAutoOptimizer::new(config);

        let data_chars = DataCharacteristics {
            feature_count: 20,
            sample_count: 1000,
            volatility: 0.15, // High volatility
            data_quality: 0.9,
            missing_data_ratio: 0.01,
            correlation_structure: 0.3,
        };

        let quantiles = optimizer
            .select_optimal_quantiles(&data_chars, &QuantileSelectionStrategy::VolatilityAdaptive)
            .unwrap();

        // High volatility should result in more quantiles
        assert!(quantiles.len() >= 7);
    }

    #[test]
    fn test_factory_crypto_optimized() {
        let config = TFTOptimizerFactory::crypto_optimized();
        assert!(config.enabled);
        assert!(config.variable_selection.auto_tune_threshold);
        assert_eq!(config.variable_selection.min_threshold, 0.1);
    }

    #[test]
    fn test_factory_conservative() {
        let config = TFTOptimizerFactory::conservative();
        assert!(config.enabled);
        assert!(!config.variable_selection.dynamic_top_k);
        assert!(!config.quantile_regression.dynamic_loss_weighting);
    }
}
