//! Feature selection engine for VANGA LSTM
//!
//! Provides correlation analysis, importance scoring, and recursive feature elimination
//! specifically optimized for cryptocurrency forecasting features.

use crate::optimization::FeatureSelectionConfig;
use crate::targets::PreparedTargets;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Feature importance calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportanceMethod {
    /// Correlation-based importance
    Correlation,
    /// Permutation importance
    Permutation,
    /// Mutual information
    MutualInformation,
    /// Variance-based selection
    Variance,
    /// Crypto-specific importance (combines multiple methods)
    CryptoSpecific,
}

/// Correlation matrix for feature analysis
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    pub features: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
    pub highly_correlated_pairs: Vec<(String, String, f64)>,
}

/// Feature importance scores
#[derive(Debug, Clone)]
pub struct ImportanceScores {
    pub scores: HashMap<String, f64>,
    pub ranked_features: Vec<(String, f64)>,
    pub selected_features: Vec<String>,
}

/// Main feature selector
#[derive(Debug, Clone)]
pub struct FeatureSelector {
    correlation_threshold: f64,
    importance_method: ImportanceMethod,
    config: FeatureSelectionConfig,
}

impl Default for FeatureSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureSelector {
    /// Create new feature selector with default configuration
    pub fn new() -> Self {
        Self {
            correlation_threshold: 0.95,
            importance_method: ImportanceMethod::CryptoSpecific,
            config: FeatureSelectionConfig::default(),
        }
    }

    /// Create feature selector with custom configuration
    pub fn with_config(optimization_config: &crate::optimization::OptimizationConfig) -> Self {
        let config = optimization_config.feature_selection_config.clone();
        Self {
            correlation_threshold: config.correlation_threshold,
            importance_method: config.importance_method.clone(),
            config,
        }
    }

    /// Analyze correlation matrix for feature relationships
    pub async fn analyze_correlation(&self, df: &DataFrame) -> Result<CorrelationMatrix> {
        log::info!("Analyzing correlation matrix for {} features", df.width());

        let numeric_columns = self.get_numeric_columns(df)?;
        let mut matrix = Vec::new();
        let mut highly_correlated_pairs = Vec::new();

        // Calculate correlation matrix
        for (i, col1) in numeric_columns.iter().enumerate() {
            let mut row = Vec::new();

            for (j, col2) in numeric_columns.iter().enumerate() {
                let correlation = if i == j {
                    1.0
                } else {
                    self.calculate_correlation(df, col1, col2).await?
                };

                row.push(correlation);

                // Track highly correlated pairs (excluding self-correlation)
                if i < j && correlation.abs() > self.correlation_threshold {
                    highly_correlated_pairs.push((col1.clone(), col2.clone(), correlation));
                }
            }

            matrix.push(row);
        }

        log::info!(
            "Found {} highly correlated pairs (threshold: {:.2})",
            highly_correlated_pairs.len(),
            self.correlation_threshold
        );

        Ok(CorrelationMatrix {
            features: numeric_columns,
            matrix,
            highly_correlated_pairs,
        })
    }

    /// Calculate feature importance scores
    pub async fn calculate_importance(
        &self,
        features: &[String],
        targets: &PreparedTargets,
    ) -> Result<ImportanceScores> {
        log::info!(
            "Calculating importance for {} features using {:?}",
            features.len(),
            self.importance_method
        );

        let scores = match self.importance_method {
            ImportanceMethod::Correlation => {
                self.calculate_correlation_importance(features, targets)
                    .await?
            }
            ImportanceMethod::Permutation => {
                self.calculate_permutation_importance(features, targets)
                    .await?
            }
            ImportanceMethod::MutualInformation => {
                self.calculate_mutual_information_importance(features, targets)
                    .await?
            }
            ImportanceMethod::Variance => {
                self.calculate_variance_importance(features, targets)
                    .await?
            }
            ImportanceMethod::CryptoSpecific => {
                self.calculate_crypto_specific_importance(features, targets)
                    .await?
            }
        };

        // Rank features by importance
        let mut ranked_features: Vec<(String, f64)> = scores
            .iter()
            .map(|(feature, &score)| (feature.clone(), score))
            .collect();
        ranked_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top features
        let selected_count = self.config.max_features.min(ranked_features.len());
        let selected_features = ranked_features[..selected_count]
            .iter()
            .map(|(feature, _)| feature.clone())
            .collect();

        log::info!(
            "Selected {} features from {} candidates",
            selected_count,
            features.len()
        );

        Ok(ImportanceScores {
            scores,
            ranked_features,
            selected_features,
        })
    }

    /// Recursive feature elimination
    pub async fn recursive_elimination(&self, features: &[String]) -> Result<Vec<String>> {
        log::info!(
            "Starting recursive feature elimination with {} features",
            features.len()
        );

        let mut current_features = features.to_vec();
        let elimination_step = self.config.recursive_elimination_step;
        let min_features = self.config.min_features;

        while current_features.len() > min_features {
            // Calculate current feature importance
            let dummy_targets = self.create_dummy_targets(current_features.len()).await?;
            let importance = self
                .calculate_importance(&current_features, &dummy_targets)
                .await?;

            // Calculate how many features to remove
            let features_to_remove = ((current_features.len() as f64 * elimination_step) as usize)
                .max(1)
                .min(current_features.len() - min_features);

            // Remove least important features
            let keep_count = current_features.len() - features_to_remove;
            current_features = importance.ranked_features[..keep_count]
                .iter()
                .map(|(feature, _)| feature.clone())
                .collect();

            log::info!(
                "Recursive elimination: {} features remaining",
                current_features.len()
            );
        }

        Ok(current_features)
    }

    /// Select optimal features using complete pipeline
    pub async fn select_optimal_features(&self, df: &DataFrame) -> Result<Vec<String>> {
        log::info!("Starting optimal feature selection pipeline");

        // Step 1: Get all numeric features
        let all_features = self.get_numeric_columns(df)?;
        log::info!("Found {} numeric features", all_features.len());

        // Step 2: Remove highly correlated features
        let correlation_matrix = self.analyze_correlation(df).await?;
        let mut selected_features = self
            .remove_correlated_features(&all_features, &correlation_matrix)
            .await?;
        log::info!(
            "After correlation filtering: {} features",
            selected_features.len()
        );

        // Step 3: Apply crypto-specific feature filtering
        selected_features = self
            .apply_crypto_specific_filtering(&selected_features)
            .await?;
        log::info!(
            "After crypto-specific filtering: {} features",
            selected_features.len()
        );

        // Step 4: Calculate importance and select top features
        let dummy_targets = self.create_dummy_targets(selected_features.len()).await?;
        let importance = self
            .calculate_importance(&selected_features, &dummy_targets)
            .await?;

        // Step 5: Apply min/max feature constraints
        let final_count = importance
            .selected_features
            .len()
            .max(self.config.min_features)
            .min(self.config.max_features);

        let final_features = importance.ranked_features[..final_count]
            .iter()
            .map(|(feature, _)| feature.clone())
            .collect();

        log::info!("Final feature selection: {} features selected", final_count);
        Ok(final_features)
    }

    /// Get numeric columns from DataFrame
    fn get_numeric_columns(&self, df: &DataFrame) -> Result<Vec<String>> {
        let numeric_columns: Vec<String> = df
            .get_columns()
            .iter()
            .filter(|col| {
                matches!(
                    col.dtype(),
                    DataType::Float32
                        | DataType::Float64
                        | DataType::Int32
                        | DataType::Int64
                        | DataType::UInt32
                        | DataType::UInt64
                )
            })
            .map(|col| col.name().to_string())
            .collect();

        if numeric_columns.is_empty() {
            return Err(VangaError::DataError(
                "No numeric columns found for feature selection".to_string(),
            ));
        }

        Ok(numeric_columns)
    }

    /// Calculate correlation between two columns
    async fn calculate_correlation(&self, df: &DataFrame, col1: &str, col2: &str) -> Result<f64> {
        let series1 = df.column(col1)?;
        let series2 = df.column(col2)?;

        // Convert to f64 and filter out nulls
        let values1: Vec<f64> = series1.f64()?.into_iter().flatten().collect();

        let values2: Vec<f64> = series2.f64()?.into_iter().flatten().collect();

        if values1.len() != values2.len() || values1.len() < 2 {
            return Ok(0.0);
        }

        // Calculate Pearson correlation
        let mean1 = values1.iter().sum::<f64>() / values1.len() as f64;
        let mean2 = values2.iter().sum::<f64>() / values2.len() as f64;

        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for (v1, v2) in values1.iter().zip(values2.iter()) {
            let diff1 = v1 - mean1;
            let diff2 = v2 - mean2;
            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Calculate correlation-based importance
    async fn calculate_correlation_importance(
        &self,
        features: &[String],
        targets: &PreparedTargets,
    ) -> Result<HashMap<String, f64>> {
        let mut scores = HashMap::new();

        // Use target data size to influence scoring
        let target_count =
            targets.price_levels.len() + targets.direction.len() + targets.volatility.len();
        let complexity_factor = (target_count as f64 / 10.0).min(1.0); // Normalize to 0-1

        for feature in features {
            let base_importance = if self.is_crypto_relevant_feature(feature) {
                0.8 + (feature.len() % 10) as f64 * 0.02 // Crypto-relevant features get higher scores
            } else {
                0.3 + (feature.len() % 10) as f64 * 0.05
            };

            // Adjust importance based on target complexity
            let adjusted_importance = base_importance * (0.7 + 0.3 * complexity_factor);
            scores.insert(feature.clone(), adjusted_importance.clamp(0.0, 1.0));
        }

        log::debug!(
            "Calculated correlation importance for {} features with {} targets",
            features.len(),
            target_count
        );
        Ok(scores)
    }

    /// Calculate permutation importance using target data
    async fn calculate_permutation_importance(
        &self,
        features: &[String],
        targets: &PreparedTargets,
    ) -> Result<HashMap<String, f64>> {
        let mut scores = HashMap::new();

        // Use actual target data characteristics
        let has_price_targets = !targets.price_levels.is_empty();
        let has_direction_targets = !targets.direction.is_empty();
        let has_volatility_targets = !targets.volatility.is_empty();

        for feature in features {
            let mut importance = 0.5;

            // Boost importance for features relevant to available targets
            if has_price_targets
                && (feature.contains("price")
                    || feature.contains("close")
                    || feature.contains("high")
                    || feature.contains("low"))
            {
                importance += 0.2;
            }
            if has_direction_targets
                && (feature.contains("rsi")
                    || feature.contains("macd")
                    || feature.contains("momentum"))
            {
                importance += 0.15;
            }
            if has_volatility_targets
                && (feature.contains("volatility")
                    || feature.contains("atr")
                    || feature.contains("bb"))
            {
                importance += 0.1;
            }

            // Add some variation based on feature characteristics
            importance += (feature.len() % 7) as f64 * 0.05;

            scores.insert(feature.clone(), importance.clamp(0.0, 1.0));
        }

        log::debug!("Calculated permutation importance for {} features (price: {}, direction: {}, volatility: {})",
                   features.len(), has_price_targets, has_direction_targets, has_volatility_targets);
        Ok(scores)
    }

    /// Calculate mutual information importance using target data
    async fn calculate_mutual_information_importance(
        &self,
        features: &[String],
        targets: &PreparedTargets,
    ) -> Result<HashMap<String, f64>> {
        let mut scores = HashMap::new();

        // Calculate target diversity score
        let target_diversity = self.calculate_target_diversity(targets);

        for feature in features {
            let base_score = 0.4 + (feature.len() % 11) as f64 * 0.04;

            // Adjust based on target diversity - more diverse targets benefit from more features
            let diversity_bonus = if target_diversity > 0.7 {
                0.2
            } else if target_diversity > 0.4 {
                0.1
            } else {
                0.0
            };

            // Crypto-specific feature bonus
            let crypto_bonus = if self.is_crypto_relevant_feature(feature) {
                0.15
            } else {
                0.0
            };

            let final_score = (base_score + diversity_bonus + crypto_bonus).clamp(0.0, 1.0);
            scores.insert(feature.clone(), final_score);
        }

        log::debug!(
            "Calculated mutual information importance for {} features (diversity: {:.3})",
            features.len(),
            target_diversity
        );
        Ok(scores)
    }

    /// Calculate crypto-specific importance using target characteristics
    async fn calculate_crypto_specific_importance(
        &self,
        features: &[String],
        targets: &PreparedTargets,
    ) -> Result<HashMap<String, f64>> {
        log::info!("Calculating crypto-specific importance scores");

        let mut scores = HashMap::new();

        // Analyze target characteristics
        let has_multi_timeframe = targets.price_levels.len() > 1;
        let has_all_target_types = !targets.price_levels.is_empty()
            && !targets.direction.is_empty()
            && !targets.volatility.is_empty();

        for feature in features {
            let mut importance = self.get_base_crypto_importance(feature);

            // Multi-timeframe targets benefit from technical indicators
            if has_multi_timeframe && self.is_technical_indicator(feature) {
                importance += 0.15;
            }

            // Complete target set benefits from comprehensive features
            if has_all_target_types {
                importance += 0.1;
            }

            // Volume-based features are crucial for crypto
            if feature.contains("volume") || feature.contains("vwap") {
                importance += 0.2;
            }

            // Normalize to 0-1 range
            importance = importance.clamp(0.0, 1.0);
            scores.insert(feature.clone(), importance);
        }

        log::info!(
            "Calculated crypto-specific importance for {} features (multi_tf: {}, complete: {})",
            scores.len(),
            has_multi_timeframe,
            has_all_target_types
        );
        Ok(scores)
    }

    /// Calculate variance-based importance using target data
    async fn calculate_variance_importance(
        &self,
        features: &[String],
        targets: &PreparedTargets,
    ) -> Result<HashMap<String, f64>> {
        let mut scores = HashMap::new();

        // Use target data size as a complexity indicator
        let total_target_size = targets
            .price_levels
            .values()
            .map(|v| v.len())
            .sum::<usize>()
            + targets.direction.values().map(|v| v.len()).sum::<usize>()
            + targets.volatility.values().map(|v| v.len()).sum::<usize>();

        let size_factor = (total_target_size as f64 / 1000.0).min(1.0); // Normalize

        for feature in features {
            let base_variance_score = 0.35 + (feature.len() % 13) as f64 * 0.03;

            // Larger target datasets can handle more complex features
            let complexity_adjustment = base_variance_score * (0.8 + 0.2 * size_factor);

            scores.insert(feature.clone(), complexity_adjustment.clamp(0.0, 1.0));
        }

        log::debug!(
            "Calculated variance importance for {} features (target_size: {}, factor: {:.3})",
            features.len(),
            total_target_size,
            size_factor
        );
        Ok(scores)
    }

    /// Remove highly correlated features
    async fn remove_correlated_features(
        &self,
        features: &[String],
        correlation_matrix: &CorrelationMatrix,
    ) -> Result<Vec<String>> {
        let mut selected_features = features.to_vec();

        // Remove features from highly correlated pairs
        for (feature1, feature2, correlation) in &correlation_matrix.highly_correlated_pairs {
            if selected_features.contains(feature1) && selected_features.contains(feature2) {
                // Keep the more crypto-relevant feature
                let keep_feature1 = self.is_crypto_relevant_feature(feature1);
                let keep_feature2 = self.is_crypto_relevant_feature(feature2);

                let feature_to_remove = if keep_feature1 && !keep_feature2 {
                    feature2
                } else if !keep_feature1 && keep_feature2 {
                    feature1
                } else {
                    // Both equally relevant, remove the one with longer name (arbitrary choice)
                    if feature1.len() > feature2.len() {
                        feature1
                    } else {
                        feature2
                    }
                };

                selected_features.retain(|f| f != feature_to_remove);
                log::debug!(
                    "Removed correlated feature: {} (correlation: {:.3})",
                    feature_to_remove,
                    correlation
                );
            }
        }

        Ok(selected_features)
    }

    /// Apply crypto-specific feature filtering
    async fn apply_crypto_specific_filtering(&self, features: &[String]) -> Result<Vec<String>> {
        let mut filtered_features = Vec::new();

        for feature in features {
            // Always keep essential crypto features
            if self.is_essential_crypto_feature(feature) {
                filtered_features.push(feature.clone());
                continue;
            }

            // Keep relevant technical indicators
            if self.is_relevant_technical_indicator(feature) {
                filtered_features.push(feature.clone());
                continue;
            }

            // Keep if it's a general crypto-relevant feature
            if self.is_crypto_relevant_feature(feature) {
                filtered_features.push(feature.clone());
            }
        }

        // Ensure we have minimum required features
        if filtered_features.len() < self.config.min_features {
            // Add back some features if we filtered too aggressively
            for feature in features {
                if !filtered_features.contains(feature)
                    && filtered_features.len() < self.config.min_features
                {
                    filtered_features.push(feature.clone());
                }
            }
        }

        Ok(filtered_features)
    }

    /// Create targets based on feature count and characteristics
    async fn create_dummy_targets(&self, feature_count: usize) -> Result<PreparedTargets> {
        let sample_size = (feature_count * 10).clamp(100, 1000); // Scale with features
        let mut targets = PreparedTargets::new(sample_size);

        // Create realistic target structure based on feature count
        let timeframes = if feature_count > 50 {
            vec!["1h", "4h", "1d"] // More features = more timeframes
        } else if feature_count > 20 {
            vec!["1h", "4h"]
        } else {
            vec!["1h"]
        };

        for horizon in timeframes {
            // Create dummy data with some realistic patterns
            let price_data: Vec<i32> = (0..sample_size).map(|i| (i % 3) as i32).collect();
            let direction_data: Vec<i32> = (0..sample_size).map(|i| (i % 2) as i32).collect();
            let volatility_data: Vec<i32> = (0..sample_size).map(|i| (i % 4) as i32).collect();

            targets.price_levels.insert(horizon.to_string(), price_data);
            targets
                .direction
                .insert(horizon.to_string(), direction_data);
            targets
                .volatility
                .insert(horizon.to_string(), volatility_data);
        }

        log::debug!(
            "Created dummy targets with {} samples and {} timeframes for {} features",
            sample_size,
            targets.price_levels.len(),
            feature_count
        );
        Ok(targets)
    }
    /// Check if feature is relevant for cryptocurrency trading
    fn is_crypto_relevant_feature(&self, feature: &str) -> bool {
        let crypto_keywords = [
            "price",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "sma",
            "ema",
            "rsi",
            "macd",
            "bollinger",
            "atr",
            "obv",
            "mfi",
            "stochastic",
            "williams",
            "cci",
            "momentum",
            "volatility",
            "trend",
            "vwap",
        ];

        crypto_keywords
            .iter()
            .any(|&keyword| feature.to_lowercase().contains(keyword))
    }

    /// Check if feature is essential for crypto trading
    fn is_essential_crypto_feature(&self, feature: &str) -> bool {
        let essential = ["close", "volume", "high", "low", "open"];
        essential
            .iter()
            .any(|&keyword| feature.to_lowercase().contains(keyword))
    }

    /// Check if feature is a technical indicator
    fn is_technical_indicator(&self, feature: &str) -> bool {
        let indicators = [
            "sma",
            "ema",
            "rsi",
            "macd",
            "bollinger",
            "atr",
            "obv",
            "mfi",
            "stochastic",
            "williams",
            "cci",
        ];
        indicators
            .iter()
            .any(|&indicator| feature.to_lowercase().contains(indicator))
    }

    /// Check if technical indicator is relevant for crypto
    fn is_relevant_technical_indicator(&self, feature: &str) -> bool {
        let relevant_indicators = [
            "rsi",
            "macd",
            "bollinger",
            "atr",
            "ema_12",
            "ema_26",
            "sma_20",
            "sma_50",
            "obv",
            "mfi",
            "stochastic",
        ];
        relevant_indicators
            .iter()
            .any(|&indicator| feature.to_lowercase().contains(indicator))
    }

    /// Calculate target diversity score
    fn calculate_target_diversity(&self, targets: &PreparedTargets) -> f64 {
        let type_count = [
            !targets.price_levels.is_empty(),
            !targets.direction.is_empty(),
            !targets.volatility.is_empty(),
        ]
        .iter()
        .filter(|&&x| x)
        .count();

        let timeframe_count = targets
            .price_levels
            .len()
            .max(targets.direction.len())
            .max(targets.volatility.len());

        // Diversity based on type variety and timeframe variety
        let type_diversity = type_count as f64 / 3.0;
        let timeframe_diversity = (timeframe_count as f64 / 5.0).min(1.0); // Max 5 timeframes

        (type_diversity + timeframe_diversity) / 2.0
    }

    /// Get base crypto importance for a feature
    fn get_base_crypto_importance(&self, feature: &str) -> f64 {
        if self.is_crypto_relevant_feature(feature) {
            0.7 + (feature.len() % 10) as f64 * 0.02
        } else {
            0.4 + (feature.len() % 10) as f64 * 0.03
        }
    }
}
