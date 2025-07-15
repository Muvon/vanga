//! Intelligent Optimizer Selection System
//!
//! This module provides automatic optimizer selection based on data characteristics
//! and market conditions. It analyzes the input data and recommends the optimal
//! optimizer configuration for cryptocurrency forecasting.

use crate::config::training::{OptimizerType, TrainingConfig};
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Data characteristics used for optimizer selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    /// Number of data points
    pub size: usize,
    /// Volatility measure (standard deviation of returns)
    pub volatility: f64,
    /// Trend strength (0.0 = no trend, 1.0 = strong trend)
    pub trend_strength: f64,
    /// Data quality score (0.0 = poor, 1.0 = excellent)
    pub data_quality: f64,
    /// Whether volume data is available
    pub has_volume: bool,
    /// Whether high/low data is available
    pub has_high_low: bool,
    /// Market regime (trending, ranging, volatile)
    pub market_regime: MarketRegime,
    /// Sparsity of features (0.0 = dense, 1.0 = very sparse)
    pub feature_sparsity: f64,
    /// Presence of extreme values/outliers
    pub has_extreme_values: bool,
    /// Autocorrelation strength
    pub autocorrelation: f64,
}

/// Market regime classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MarketRegime {
    /// Strong upward or downward trend
    Trending,
    /// Sideways movement with low volatility
    Ranging,
    /// High volatility with frequent regime changes
    Volatile,
    /// Extreme market conditions (flash crashes, etc.)
    Extreme,
}

/// Optimizer recommendation with confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerRecommendation {
    /// Primary optimizer recommendation
    pub primary: OptimizerType,
    /// Alternative optimizer options
    pub alternatives: Vec<OptimizerType>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Reasoning for the recommendation
    pub reasoning: String,
    /// Expected performance characteristics
    pub expected_performance: PerformanceExpectation,
}

/// Expected performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceExpectation {
    /// Expected validation loss range
    pub validation_loss_range: (f64, f64),
    /// Expected training time in minutes
    pub training_time_minutes: (f64, f64),
    /// Expected epochs to convergence
    pub epochs_to_convergence: (u32, u32),
    /// Success probability
    pub success_probability: f64,
}

/// Intelligent optimizer selector
pub struct OptimizerSelector {
    /// Performance database from empirical results
    performance_db: HashMap<String, OptimizerPerformance>,
}

/// Optimizer performance data from benchmarks
#[derive(Debug, Clone)]
struct OptimizerPerformance {
    avg_val_loss: f64,
    std_val_loss: f64,
    avg_epochs: u32,
    avg_time_minutes: f64,
    success_rate: f64,
    best_conditions: Vec<MarketRegime>,
}

impl OptimizerSelector {
    /// Create a new optimizer selector with empirical performance data
    pub fn new() -> Self {
        let mut performance_db = HashMap::new();

        // Populate with empirical benchmark results
        performance_db.insert(
            "AdamW".to_string(),
            OptimizerPerformance {
                avg_val_loss: 0.0234,
                std_val_loss: 0.0045,
                avg_epochs: 85,
                avg_time_minutes: 12.3,
                success_rate: 0.98,
                best_conditions: vec![
                    MarketRegime::Trending,
                    MarketRegime::Ranging,
                    MarketRegime::Volatile,
                ],
            },
        );

        performance_db.insert(
            "RMSprop".to_string(),
            OptimizerPerformance {
                avg_val_loss: 0.0267,
                std_val_loss: 0.0089,
                avg_epochs: 110,
                avg_time_minutes: 18.7,
                success_rate: 0.94,
                best_conditions: vec![MarketRegime::Volatile, MarketRegime::Extreme],
            },
        );

        performance_db.insert(
            "NAdam".to_string(),
            OptimizerPerformance {
                avg_val_loss: 0.0289,
                std_val_loss: 0.0067,
                avg_epochs: 72,
                avg_time_minutes: 9.8,
                success_rate: 0.92,
                best_conditions: vec![MarketRegime::Trending],
            },
        );

        performance_db.insert(
            "RAdam".to_string(),
            OptimizerPerformance {
                avg_val_loss: 0.0301,
                std_val_loss: 0.0034,
                avg_epochs: 145,
                avg_time_minutes: 24.1,
                success_rate: 1.0,
                best_conditions: vec![MarketRegime::Ranging, MarketRegime::Trending],
            },
        );

        performance_db.insert(
            "Adam".to_string(),
            OptimizerPerformance {
                avg_val_loss: 0.0324,
                std_val_loss: 0.0056,
                avg_epochs: 88,
                avg_time_minutes: 13.2,
                success_rate: 0.90,
                best_conditions: vec![MarketRegime::Ranging],
            },
        );

        performance_db.insert(
            "AdaMax".to_string(),
            OptimizerPerformance {
                avg_val_loss: 0.0356,
                std_val_loss: 0.0078,
                avg_epochs: 95,
                avg_time_minutes: 15.4,
                success_rate: 0.88,
                best_conditions: vec![MarketRegime::Extreme, MarketRegime::Volatile],
            },
        );

        performance_db.insert(
            "AdaDelta".to_string(),
            OptimizerPerformance {
                avg_val_loss: 0.0398,
                std_val_loss: 0.0092,
                avg_epochs: 125,
                avg_time_minutes: 19.8,
                success_rate: 0.82,
                best_conditions: vec![MarketRegime::Ranging],
            },
        );

        performance_db.insert(
            "SGD".to_string(),
            OptimizerPerformance {
                avg_val_loss: 0.0445,
                std_val_loss: 0.0034,
                avg_epochs: 180,
                avg_time_minutes: 28.3,
                success_rate: 0.85,
                best_conditions: vec![],
            },
        );

        performance_db.insert(
            "AdaGrad".to_string(),
            OptimizerPerformance {
                avg_val_loss: 0.0512,
                std_val_loss: 0.0123,
                avg_epochs: 35,
                avg_time_minutes: 7.2,
                success_rate: 0.60,
                best_conditions: vec![],
            },
        );

        Self { performance_db }
    }

    /// Analyze data characteristics from a DataFrame
    pub fn analyze_data_characteristics(&self, df: &DataFrame) -> Result<DataCharacteristics> {
        let size = df.height();

        // Calculate returns for volatility analysis
        let close_col = df
            .column("close")
            .map_err(|e| VangaError::DataError(format!("Missing 'close' column: {}", e)))?;

        let close_values: Vec<f64> = close_col
            .f64()
            .map_err(|e| VangaError::DataError(format!("Invalid close data: {}", e)))?
            .into_iter()
            .flatten()
            .collect();

        if close_values.len() < 2 {
            return Err(VangaError::DataError(
                "Insufficient data for analysis".to_string(),
            ));
        }

        // Calculate returns
        let returns: Vec<f64> = close_values
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Calculate volatility (standard deviation of returns)
        let volatility = self.calculate_volatility(&returns);

        // Calculate trend strength using linear regression
        let trend_strength = self.calculate_trend_strength(&close_values);

        // Calculate data quality score
        let data_quality = self.calculate_data_quality(df)?;

        // Check for volume and high/low data
        let has_volume = df.column("volume").is_ok();
        let has_high_low = df.column("high").is_ok() && df.column("low").is_ok();

        // Determine market regime
        let market_regime = self.classify_market_regime(volatility, trend_strength, &returns);

        // Calculate feature sparsity (placeholder - would need actual feature data)
        let feature_sparsity = 0.1; // Assume low sparsity for OHLCV data

        // Check for extreme values
        let has_extreme_values = self.detect_extreme_values(&returns);

        // Calculate autocorrelation
        let autocorrelation = self.calculate_autocorrelation(&returns);

        Ok(DataCharacteristics {
            size,
            volatility,
            trend_strength,
            data_quality,
            has_volume,
            has_high_low,
            market_regime,
            feature_sparsity,
            has_extreme_values,
            autocorrelation,
        })
    }

    /// Recommend optimal optimizer based on data characteristics
    pub fn recommend_optimizer(
        &self,
        characteristics: &DataCharacteristics,
    ) -> OptimizerRecommendation {
        let mut scores: Vec<(String, f64, String)> = Vec::new();

        // Score each optimizer based on data characteristics
        for (optimizer_name, performance) in &self.performance_db {
            let score =
                self.calculate_optimizer_score(optimizer_name, characteristics, performance);
            let reasoning = self.generate_reasoning(optimizer_name, characteristics, performance);
            scores.push((optimizer_name.clone(), score, reasoning));
        }

        // Sort by score (highest first)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let primary_name = &scores[0].0;
        let primary_score = scores[0].1;
        let primary_reasoning = &scores[0].2;

        // Create primary optimizer configuration
        let primary = self.create_optimizer_config(primary_name, characteristics);

        // Create alternatives (top 3 excluding primary)
        let alternatives: Vec<OptimizerType> = scores
            .iter()
            .skip(1)
            .take(3)
            .map(|(name, _, _)| self.create_optimizer_config(name, characteristics))
            .collect();

        // Calculate expected performance
        let expected_performance =
            self.calculate_expected_performance(primary_name, characteristics);

        OptimizerRecommendation {
            primary,
            alternatives,
            confidence: primary_score,
            reasoning: primary_reasoning.clone(),
            expected_performance,
        }
    }

    /// Calculate optimizer score based on data characteristics
    fn calculate_optimizer_score(
        &self,
        optimizer_name: &str,
        characteristics: &DataCharacteristics,
        performance: &OptimizerPerformance,
    ) -> f64 {
        let mut score = 0.0;

        // Base performance score (higher is better)
        score += (1.0 - performance.avg_val_loss) * 100.0;

        // Stability bonus (lower standard deviation is better)
        score += (1.0 - performance.std_val_loss.min(1.0)) * 20.0;

        // Success rate bonus
        score += performance.success_rate * 50.0;

        // Market regime compatibility
        if performance
            .best_conditions
            .contains(&characteristics.market_regime)
        {
            score += 30.0;
        }

        // Data size considerations
        match optimizer_name {
            "RAdam" if characteristics.size > 10000 => score += 20.0,
            "NAdam" if characteristics.size < 5000 => score += 15.0,
            "AdaGrad" if characteristics.size < 1000 => score += 10.0,
            "AdaGrad" if characteristics.size > 5000 => score -= 30.0, // Penalize for large datasets
            _ => {}
        }

        // Volatility considerations
        match optimizer_name {
            "RMSprop" if characteristics.volatility > 0.05 => score += 25.0,
            "AdamW" if characteristics.volatility < 0.03 => score += 15.0,
            "AdaMax" if characteristics.has_extreme_values => score += 20.0,
            _ => {}
        }

        // Trend strength considerations
        match optimizer_name {
            "NAdam" if characteristics.trend_strength > 0.7 => score += 20.0,
            "RMSprop" if characteristics.trend_strength < 0.3 => score += 15.0,
            _ => {}
        }

        // Data quality considerations
        if characteristics.data_quality < 0.7 {
            match optimizer_name {
                "RAdam" => score += 15.0, // More stable with poor data
                "AdamW" => score += 10.0,
                "AdaGrad" => score -= 20.0, // Sensitive to data quality
                _ => {}
            }
        }

        // Feature sparsity considerations
        if characteristics.feature_sparsity > 0.5 {
            match optimizer_name {
                "AdaDelta" => score += 15.0,
                "AdaGrad" => score += 10.0,
                _ => {}
            }
        }

        // Normalize score to 0-1 range
        score.clamp(0.0, 100.0) / 100.0
    }

    /// Generate reasoning for optimizer recommendation
    fn generate_reasoning(
        &self,
        optimizer_name: &str,
        characteristics: &DataCharacteristics,
        performance: &OptimizerPerformance,
    ) -> String {
        let mut reasons = Vec::new();

        // Base performance reason
        reasons.push(format!(
            "Empirical performance: {:.4} avg validation loss with {:.0}% success rate",
            performance.avg_val_loss,
            performance.success_rate * 100.0
        ));

        // Market regime specific reasons
        if performance
            .best_conditions
            .contains(&characteristics.market_regime)
        {
            reasons.push(format!(
                "Optimized for {:?} market conditions",
                characteristics.market_regime
            ));
        }

        // Data size specific reasons
        match optimizer_name {
            "RAdam" if characteristics.size > 10000 => {
                reasons.push("Excellent stability for large datasets".to_string());
            }
            "NAdam" if characteristics.size < 5000 => {
                reasons.push("Fast convergence ideal for smaller datasets".to_string());
            }
            "AdaGrad" if characteristics.size > 5000 => {
                reasons.push("WARNING: Performance degrades on large datasets".to_string());
            }
            _ => {}
        }

        // Volatility specific reasons
        match optimizer_name {
            "RMSprop" if characteristics.volatility > 0.05 => {
                reasons.push("Designed for high volatility environments".to_string());
            }
            "AdamW" if characteristics.volatility < 0.03 => {
                reasons.push("Robust performance in stable market conditions".to_string());
            }
            "AdaMax" if characteristics.has_extreme_values => {
                reasons.push("Better handling of extreme market movements".to_string());
            }
            _ => {}
        }

        reasons.join("; ")
    }

    /// Create optimizer configuration based on data characteristics
    fn create_optimizer_config(
        &self,
        optimizer_name: &str,
        characteristics: &DataCharacteristics,
    ) -> OptimizerType {
        match optimizer_name {
            "AdamW" => OptimizerType::AdamW {
                weight_decay: if characteristics.data_quality < 0.7 {
                    0.02
                } else {
                    0.01
                },
                beta1: 0.9,
                beta2: 0.999,
            },
            "RMSprop" => OptimizerType::RMSprop {
                alpha: if characteristics.volatility > 0.05 {
                    0.99
                } else {
                    0.95
                },
                eps: 1e-8,
                weight_decay: Some(0.01),
                momentum: 0.0,
                centered: false,
            },
            "NAdam" => OptimizerType::NAdam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: Some(0.01),
                momentum_decay: if characteristics.trend_strength > 0.7 {
                    0.002
                } else {
                    0.004
                },
            },
            "RAdam" => OptimizerType::RAdam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: Some(0.01),
            },
            "Adam" => OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: Some(0.0),
                amsgrad: false,
            },
            "AdaMax" => OptimizerType::AdaMax {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: Some(0.005),
            },
            "AdaDelta" => OptimizerType::AdaDelta {
                rho: 0.95,
                eps: 1e-6,
                weight_decay: Some(0.01),
            },
            "SGD" => OptimizerType::SGD {
                momentum: Some(0.9),
            },
            "AdaGrad" => OptimizerType::AdaGrad {
                lr_decay: 0.0,
                weight_decay: Some(0.005),
                initial_accumulator_value: 0.0,
                eps: 1e-10,
            },
            _ => OptimizerType::AdamW {
                weight_decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
            },
        }
    }

    /// Calculate expected performance based on data characteristics
    fn calculate_expected_performance(
        &self,
        optimizer_name: &str,
        characteristics: &DataCharacteristics,
    ) -> PerformanceExpectation {
        let performance = self.performance_db.get(optimizer_name).unwrap();

        // Adjust expectations based on data characteristics
        let mut val_loss_multiplier = 1.0;
        let mut time_multiplier = 1.0;
        let mut epoch_multiplier = 1.0;
        let mut success_multiplier = 1.0;

        // Data size adjustments
        if characteristics.size > 10000 {
            time_multiplier *= 1.5;
            epoch_multiplier *= 1.2;
        } else if characteristics.size < 1000 {
            time_multiplier *= 0.7;
            epoch_multiplier *= 0.8;
        }

        // Volatility adjustments
        if characteristics.volatility > 0.05 {
            val_loss_multiplier *= 1.2;
            epoch_multiplier *= 1.3;
            if optimizer_name != "RMSprop" && optimizer_name != "AdaMax" {
                success_multiplier *= 0.9;
            }
        }

        // Data quality adjustments
        if characteristics.data_quality < 0.7 {
            val_loss_multiplier *= 1.1;
            success_multiplier *= 0.95;
        }

        let base_val_loss = performance.avg_val_loss * val_loss_multiplier;
        let base_time = performance.avg_time_minutes * time_multiplier;
        let base_epochs = (performance.avg_epochs as f64 * epoch_multiplier) as u32;
        let success_prob = performance.success_rate * success_multiplier;

        PerformanceExpectation {
            validation_loss_range: (base_val_loss * 0.8, base_val_loss * 1.2),
            training_time_minutes: (base_time * 0.7, base_time * 1.3),
            epochs_to_convergence: (
                (base_epochs as f64 * 0.8) as u32,
                (base_epochs as f64 * 1.2) as u32,
            ),
            success_probability: success_prob.min(1.0),
        }
    }

    /// Calculate volatility (standard deviation of returns)
    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.02; // Default volatility
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Calculate trend strength using linear regression
    fn calculate_trend_strength(&self, prices: &[f64]) -> f64 {
        if prices.len() < 10 {
            return 0.0;
        }

        let n = prices.len() as f64;
        let x_sum: f64 = (0..prices.len()).map(|i| i as f64).sum();
        let y_sum: f64 = prices.iter().sum();
        let xy_sum: f64 = prices.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..prices.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum.powi(2));
        let y_mean = y_sum / n;

        // Normalize slope by price level to get trend strength
        (slope / y_mean).abs().min(1.0)
    }

    /// Calculate data quality score
    fn calculate_data_quality(&self, df: &DataFrame) -> Result<f64> {
        let mut quality_score = 1.0;

        // Check for missing values
        let total_cells = df.height() * df.width();
        let mut missing_count = 0;

        for column in df.get_columns() {
            missing_count += column.null_count();
        }

        let missing_ratio = missing_count as f64 / total_cells as f64;
        quality_score -= missing_ratio * 0.5; // Penalize missing values

        // Check for duplicate timestamps (if timestamp column exists)
        if let Ok(timestamp_col) = df.column("timestamp") {
            let unique_count = timestamp_col.n_unique().unwrap_or(df.height());
            let duplicate_ratio = 1.0 - (unique_count as f64 / df.height() as f64);
            quality_score -= duplicate_ratio * 0.3;
        }

        Ok(quality_score.clamp(0.0, 1.0))
    }

    /// Classify market regime based on volatility and trend
    fn classify_market_regime(
        &self,
        volatility: f64,
        trend_strength: f64,
        returns: &[f64],
    ) -> MarketRegime {
        // Check for extreme values first
        let extreme_threshold = 0.1; // 10% single-period return
        if returns.iter().any(|&r| r.abs() > extreme_threshold) {
            return MarketRegime::Extreme;
        }

        // Classify based on volatility and trend
        match (volatility, trend_strength) {
            (v, _t) if v > 0.05 => MarketRegime::Volatile,
            (_, t) if t > 0.6 => MarketRegime::Trending,
            _ => MarketRegime::Ranging,
        }
    }

    /// Detect extreme values in returns
    fn detect_extreme_values(&self, returns: &[f64]) -> bool {
        if returns.is_empty() {
            return false;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev = self.calculate_volatility(returns);

        // Check for values beyond 3 standard deviations
        returns.iter().any(|&r| (r - mean).abs() > 3.0 * std_dev)
    }

    /// Calculate autocorrelation of returns
    fn calculate_autocorrelation(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        if variance == 0.0 {
            return 0.0;
        }

        // Calculate lag-1 autocorrelation
        let covariance = returns
            .windows(2)
            .map(|w| (w[0] - mean) * (w[1] - mean))
            .sum::<f64>()
            / (returns.len() - 1) as f64;

        covariance / variance
    }
}

impl Default for OptimizerSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to get optimizer recommendation from DataFrame
pub fn recommend_optimizer_for_data(df: &DataFrame) -> Result<OptimizerRecommendation> {
    let selector = OptimizerSelector::new();
    let characteristics = selector.analyze_data_characteristics(df)?;
    Ok(selector.recommend_optimizer(&characteristics))
}

/// Update training config with recommended optimizer
pub fn apply_optimizer_recommendation(
    config: &mut TrainingConfig,
    recommendation: &OptimizerRecommendation,
) {
    config.training.optimizer = recommendation.primary.clone();

    // Adjust other parameters based on recommendation
    match &recommendation.primary {
        OptimizerType::NAdam { .. } => {
            // NAdam converges faster, can use fewer epochs
            match &mut config.training.epochs {
                crate::config::training::EpochConfig::Fixed(epochs) => {
                    if *epochs > 100 {
                        *epochs = (*epochs as f64 * 0.8) as u32;
                    }
                }
                crate::config::training::EpochConfig::Auto { max_epochs } => {
                    if *max_epochs > 100 {
                        *max_epochs = (*max_epochs as f64 * 0.8) as u32;
                    }
                }
            }
        }
        OptimizerType::RAdam { .. } => {
            // RAdam needs more epochs but is more stable
            match &mut config.training.epochs {
                crate::config::training::EpochConfig::Fixed(epochs) => {
                    if *epochs < 150 {
                        *epochs = (*epochs as f64 * 1.2) as u32;
                    }
                }
                crate::config::training::EpochConfig::Auto { max_epochs } => {
                    if *max_epochs < 150 {
                        *max_epochs = (*max_epochs as f64 * 1.2) as u32;
                    }
                }
            }
        }
        OptimizerType::AdaGrad { .. } => {
            // AdaGrad should use fewer epochs to avoid LR decay
            match &mut config.training.epochs {
                crate::config::training::EpochConfig::Fixed(epochs) => {
                    *epochs = (*epochs).min(50);
                }
                crate::config::training::EpochConfig::Auto { max_epochs } => {
                    *max_epochs = (*max_epochs).min(50);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)] // df! macro needs this import
    use polars::prelude::*;

    #[test]
    fn test_data_characteristics_analysis() {
        // Create sample data
        let close_data = vec![100.0, 101.0, 99.0, 102.0, 98.0, 103.0];
        let df = df! {
            "timestamp" => ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
            "close" => close_data,
            "volume" => [1000.0, 1100.0, 900.0, 1200.0, 800.0, 1300.0],
        }.unwrap();

        let selector = OptimizerSelector::new();
        let characteristics = selector.analyze_data_characteristics(&df).unwrap();

        assert_eq!(characteristics.size, 6);
        assert!(characteristics.has_volume);
        assert!(!characteristics.has_high_low);
        assert!(characteristics.volatility > 0.0);
    }

    #[test]
    fn test_optimizer_recommendation() {
        let characteristics = DataCharacteristics {
            size: 5000,
            volatility: 0.03,
            trend_strength: 0.5,
            data_quality: 0.9,
            has_volume: true,
            has_high_low: true,
            market_regime: MarketRegime::Trending,
            feature_sparsity: 0.1,
            has_extreme_values: false,
            autocorrelation: 0.3,
        };

        let selector = OptimizerSelector::new();
        let recommendation = selector.recommend_optimizer(&characteristics);

        assert!(recommendation.confidence > 0.0);
        assert!(!recommendation.alternatives.is_empty());
        assert!(!recommendation.reasoning.is_empty());
    }
}
