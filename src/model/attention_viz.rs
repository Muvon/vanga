// Attention weight visualization and interpretability for VANGA LSTM
use crate::utils::error::{Result, VangaError};
use candle_core::Tensor;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for attention visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionVisualizationConfig {
    /// Whether to save attention heatmaps
    pub save_heatmaps: bool,

    /// Whether to calculate feature importance scores
    pub calculate_feature_importance: bool,

    /// Whether to track temporal attention patterns
    pub track_temporal_patterns: bool,

    /// Output directory for visualizations
    pub output_dir: String,

    /// Top-K features to highlight in visualizations
    pub top_k_features: usize,

    /// Attention threshold for highlighting (0.0-1.0)
    pub attention_threshold: f64,
}

impl Default for AttentionVisualizationConfig {
    fn default() -> Self {
        Self {
            save_heatmaps: true,
            calculate_feature_importance: true,
            track_temporal_patterns: true,
            output_dir: "attention_viz".to_string(),
            top_k_features: 10,
            attention_threshold: 0.1,
        }
    }
}

/// Attention pattern analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionAnalysis {
    /// Feature importance scores (feature_name -> importance)
    pub feature_importance: HashMap<String, f64>,

    /// Temporal attention patterns (timestep -> attention_score)
    pub temporal_patterns: Vec<f64>,

    /// Attention entropy (measure of attention concentration)
    pub attention_entropy: f64,

    /// Most attended features
    pub top_features: Vec<(String, f64)>,

    /// Attention statistics
    pub attention_stats: AttentionStatistics,
}

/// Statistical measures of attention patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionStatistics {
    pub mean_attention: f64,
    pub std_attention: f64,
    pub max_attention: f64,
    pub min_attention: f64,
    pub attention_sparsity: f64, // Percentage of near-zero attention weights
}

/// Attention weight visualizer for model interpretability
pub struct AttentionVisualizer {
    config: AttentionVisualizationConfig,
    feature_names: Vec<String>,
    attention_history: Vec<Array2<f64>>,
}

impl AttentionVisualizer {
    /// Create new attention visualizer
    pub fn new(config: AttentionVisualizationConfig, feature_names: Vec<String>) -> Result<Self> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&config.output_dir).map_err(|e| {
            VangaError::IoError(format!("Failed to create output directory: {}", e))
        })?;

        log::info!(
            "✅ Attention visualizer initialized with {} features, output dir: {}",
            feature_names.len(),
            config.output_dir
        );

        Ok(Self {
            config,
            feature_names,
            attention_history: Vec::new(),
        })
    }

    /// Process attention weights and generate visualizations
    pub fn process_attention_weights(
        &mut self,
        attention_weights: &Tensor,
        prediction_step: usize,
    ) -> Result<AttentionAnalysis> {
        let sequence_features = attention_weights.dims2()?.1;

        // Validate dimensions
        if sequence_features != self.feature_names.len() {
            return Err(VangaError::ModelError(format!(
                "Attention weights feature dimension {} doesn't match expected {}",
                sequence_features,
                self.feature_names.len()
            )));
        }

        // Convert tensor to ndarray for processing
        let attention_array = self.tensor_to_array2(attention_weights)?;

        // Store in history for temporal analysis
        self.attention_history.push(attention_array.clone());

        // Calculate feature importance
        let feature_importance = if self.config.calculate_feature_importance {
            self.calculate_feature_importance(&attention_array)?
        } else {
            HashMap::new()
        };

        // Calculate temporal patterns
        let temporal_patterns = if self.config.track_temporal_patterns {
            self.calculate_temporal_patterns(&attention_array)?
        } else {
            Vec::new()
        };

        // Calculate attention entropy
        let attention_entropy = self.calculate_attention_entropy(&attention_array)?;

        // Get top features
        let top_features = self.get_top_features(&feature_importance)?;

        // Calculate attention statistics
        let attention_stats = self.calculate_attention_statistics(&attention_array)?;

        // Generate visualizations if enabled
        if self.config.save_heatmaps {
            self.save_attention_heatmap(&attention_array, prediction_step)?;
        }

        let analysis = AttentionAnalysis {
            feature_importance,
            temporal_patterns,
            attention_entropy,
            top_features,
            attention_stats,
        };

        // Save analysis results
        self.save_analysis_results(&analysis, prediction_step)?;

        Ok(analysis)
    }

    /// Convert Candle tensor to ndarray for processing
    fn tensor_to_array2(&self, tensor: &Tensor) -> Result<Array2<f64>> {
        // Get tensor data as Vec<f32>
        let tensor_data = tensor.to_vec2::<f32>().map_err(|e| {
            VangaError::ModelError(format!("Failed to convert tensor to vec: {}", e))
        })?;

        let shape = tensor.shape();
        if shape.dims().len() < 2 {
            return Err(VangaError::ModelError(
                "Attention tensor must be at least 2D".to_string(),
            ));
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        // Convert to f64 and create Array2
        let data: Vec<f64> = tensor_data
            .into_iter()
            .flatten()
            .map(|x| x as f64)
            .collect();

        Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| VangaError::ModelError(format!("Failed to create attention array: {}", e)))
    }

    /// Calculate feature importance from attention weights
    fn calculate_feature_importance(
        &self,
        attention_weights: &Array2<f64>,
    ) -> Result<HashMap<String, f64>> {
        let mut feature_importance = HashMap::new();

        // Calculate mean attention for each feature across all timesteps
        for (feature_idx, feature_name) in self.feature_names.iter().enumerate() {
            if feature_idx < attention_weights.ncols() {
                let feature_attention: f64 =
                    attention_weights.column(feature_idx).iter().sum::<f64>()
                        / attention_weights.nrows() as f64;

                feature_importance.insert(feature_name.clone(), feature_attention);
            }
        }

        Ok(feature_importance)
    }

    /// Calculate temporal attention patterns
    fn calculate_temporal_patterns(&self, attention_weights: &Array2<f64>) -> Result<Vec<f64>> {
        let mut temporal_patterns = Vec::new();

        // Calculate mean attention for each timestep across all features
        for row_idx in 0..attention_weights.nrows() {
            let timestep_attention: f64 = attention_weights.row(row_idx).iter().sum::<f64>()
                / attention_weights.ncols() as f64;

            temporal_patterns.push(timestep_attention);
        }

        Ok(temporal_patterns)
    }

    /// Calculate attention entropy (measure of attention concentration)
    fn calculate_attention_entropy(&self, attention_weights: &Array2<f64>) -> Result<f64> {
        let mut total_entropy = 0.0;
        let epsilon = 1e-10; // Small value to avoid log(0)

        for row in attention_weights.rows() {
            let row_sum: f64 = row.iter().sum();
            if row_sum > epsilon {
                let mut row_entropy = 0.0;
                for &weight in row.iter() {
                    let prob = weight / row_sum;
                    if prob > epsilon {
                        row_entropy -= prob * prob.ln();
                    }
                }
                total_entropy += row_entropy;
            }
        }

        Ok(total_entropy / attention_weights.nrows() as f64)
    }

    /// Get top-K most important features
    fn get_top_features(
        &self,
        feature_importance: &HashMap<String, f64>,
    ) -> Result<Vec<(String, f64)>> {
        let mut features: Vec<(String, f64)> = feature_importance
            .iter()
            .map(|(name, &importance)| (name.clone(), importance))
            .collect();

        // Sort by importance (descending)
        features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-K features
        features.truncate(self.config.top_k_features);

        Ok(features)
    }

    /// Calculate attention statistics
    fn calculate_attention_statistics(
        &self,
        attention_weights: &Array2<f64>,
    ) -> Result<AttentionStatistics> {
        let all_weights: Vec<f64> = attention_weights.iter().cloned().collect();

        let mean_attention = all_weights.iter().sum::<f64>() / all_weights.len() as f64;

        let variance = all_weights
            .iter()
            .map(|&x| (x - mean_attention).powi(2))
            .sum::<f64>()
            / all_weights.len() as f64;
        let std_attention = variance.sqrt();

        let max_attention = all_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let min_attention = all_weights.iter().cloned().fold(f64::INFINITY, f64::min);

        // Calculate sparsity (percentage of weights below threshold)
        let sparse_count = all_weights
            .iter()
            .filter(|&&x| x < self.config.attention_threshold)
            .count();
        let attention_sparsity = sparse_count as f64 / all_weights.len() as f64;

        Ok(AttentionStatistics {
            mean_attention,
            std_attention,
            max_attention,
            min_attention,
            attention_sparsity,
        })
    }

    /// Save attention heatmap as CSV for external visualization
    fn save_attention_heatmap(&self, attention_weights: &Array2<f64>, step: usize) -> Result<()> {
        let filename = format!(
            "{}/attention_heatmap_step_{}.csv",
            self.config.output_dir, step
        );
        let path = Path::new(&filename);

        let mut csv_content = String::new();

        // Header with feature names
        csv_content.push_str("timestep");
        for feature_name in &self.feature_names {
            csv_content.push_str(&format!(",{}", feature_name));
        }
        csv_content.push('\n');

        // Data rows
        for (row_idx, row) in attention_weights.rows().into_iter().enumerate() {
            csv_content.push_str(&format!("{}", row_idx));
            for &weight in row.iter() {
                csv_content.push_str(&format!(",{:.6}", weight));
            }
            csv_content.push('\n');
        }

        std::fs::write(path, csv_content)
            .map_err(|e| VangaError::IoError(format!("Failed to save attention heatmap: {}", e)))?;

        log::debug!("Saved attention heatmap to: {}", filename);
        Ok(())
    }

    /// Save analysis results as JSON
    fn save_analysis_results(&self, analysis: &AttentionAnalysis, step: usize) -> Result<()> {
        let filename = format!(
            "{}/attention_analysis_step_{}.json",
            self.config.output_dir, step
        );
        let path = Path::new(&filename);

        let json_content = serde_json::to_string_pretty(analysis)
            .map_err(|e| VangaError::ConfigError(format!("Failed to serialize analysis: {}", e)))?;

        std::fs::write(path, json_content)
            .map_err(|e| VangaError::IoError(format!("Failed to save analysis results: {}", e)))?;

        log::debug!("Saved attention analysis to: {}", filename);
        Ok(())
    }

    /// Generate comprehensive attention report
    pub fn generate_attention_report(&self) -> Result<AttentionReport> {
        if self.attention_history.is_empty() {
            return Err(VangaError::ModelError(
                "No attention history available for report".to_string(),
            ));
        }

        // Aggregate feature importance across all steps
        let mut aggregated_importance = HashMap::new();
        let mut temporal_evolution = Vec::new();
        let mut entropy_evolution = Vec::new();

        for attention_weights in &self.attention_history {
            // Feature importance for this step
            let step_importance = self.calculate_feature_importance(attention_weights)?;
            for (feature, importance) in step_importance {
                *aggregated_importance.entry(feature).or_insert(0.0) += importance;
            }

            // Temporal patterns for this step
            let temporal_pattern = self.calculate_temporal_patterns(attention_weights)?;
            temporal_evolution.push(temporal_pattern);

            // Entropy for this step
            let entropy = self.calculate_attention_entropy(attention_weights)?;
            entropy_evolution.push(entropy);
        }

        // Normalize aggregated importance
        let num_steps = self.attention_history.len() as f64;
        for (_, importance) in aggregated_importance.iter_mut() {
            *importance /= num_steps;
        }

        // Get top features overall
        let top_features_overall = self.get_top_features(&aggregated_importance)?;

        // Calculate attention stability (how consistent attention patterns are)
        let attention_stability = self.calculate_attention_stability()?;

        let report = AttentionReport {
            total_steps: self.attention_history.len(),
            aggregated_feature_importance: aggregated_importance,
            top_features_overall,
            temporal_evolution,
            entropy_evolution,
            attention_stability,
            feature_consistency: self.calculate_feature_consistency()?,
        };

        // Save comprehensive report
        self.save_attention_report(&report)?;

        Ok(report)
    }

    /// Calculate attention stability across prediction steps
    fn calculate_attention_stability(&self) -> Result<f64> {
        if self.attention_history.len() < 2 {
            return Ok(1.0); // Perfect stability with only one step
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for i in 1..self.attention_history.len() {
            let prev_attention = &self.attention_history[i - 1];
            let curr_attention = &self.attention_history[i];

            // Calculate cosine similarity between attention patterns
            let similarity = self.calculate_cosine_similarity(prev_attention, curr_attention)?;
            total_similarity += similarity;
            comparisons += 1;
        }

        Ok(total_similarity / comparisons as f64)
    }

    /// Calculate cosine similarity between two attention matrices
    fn calculate_cosine_similarity(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<f64> {
        if a.shape() != b.shape() {
            return Ok(0.0); // No similarity if shapes don't match
        }

        let a_flat: Vec<f64> = a.iter().cloned().collect();
        let b_flat: Vec<f64> = b.iter().cloned().collect();

        let dot_product: f64 = a_flat.iter().zip(b_flat.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a_flat.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b_flat.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm_a * norm_b))
        }
    }

    /// Calculate feature consistency (how consistently each feature is attended to)
    fn calculate_feature_consistency(&self) -> Result<HashMap<String, f64>> {
        let mut feature_consistency = HashMap::new();

        for (feature_idx, feature_name) in self.feature_names.iter().enumerate() {
            let mut feature_attentions = Vec::new();

            for attention_weights in &self.attention_history {
                if feature_idx < attention_weights.ncols() {
                    let feature_attention: f64 =
                        attention_weights.column(feature_idx).iter().sum::<f64>()
                            / attention_weights.nrows() as f64;
                    feature_attentions.push(feature_attention);
                }
            }

            if !feature_attentions.is_empty() {
                let mean = feature_attentions.iter().sum::<f64>() / feature_attentions.len() as f64;
                let variance = feature_attentions
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / feature_attentions.len() as f64;
                let std_dev = variance.sqrt();

                // Consistency = 1 - coefficient_of_variation
                let consistency = if mean > 0.0 {
                    1.0 - (std_dev / mean)
                } else {
                    0.0
                };
                feature_consistency.insert(feature_name.clone(), consistency.max(0.0));
            }
        }

        Ok(feature_consistency)
    }

    /// Save comprehensive attention report
    fn save_attention_report(&self, report: &AttentionReport) -> Result<()> {
        let filename = format!(
            "{}/comprehensive_attention_report.json",
            self.config.output_dir
        );
        let path = Path::new(&filename);

        let json_content = serde_json::to_string_pretty(report)
            .map_err(|e| VangaError::ConfigError(format!("Failed to serialize report: {}", e)))?;

        std::fs::write(path, json_content)
            .map_err(|e| VangaError::IoError(format!("Failed to save attention report: {}", e)))?;

        log::info!("Saved comprehensive attention report to: {}", filename);
        Ok(())
    }

    /// Get visualization configuration
    pub fn get_config(&self) -> &AttentionVisualizationConfig {
        &self.config
    }

    /// Clear attention history to save memory
    pub fn clear_history(&mut self) {
        self.attention_history.clear();
        log::debug!("Cleared attention history");
    }
}

/// Comprehensive attention analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionReport {
    pub total_steps: usize,
    pub aggregated_feature_importance: HashMap<String, f64>,
    pub top_features_overall: Vec<(String, f64)>,
    pub temporal_evolution: Vec<Vec<f64>>,
    pub entropy_evolution: Vec<f64>,
    pub attention_stability: f64,
    pub feature_consistency: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_visualization_config_defaults() {
        let config = AttentionVisualizationConfig::default();
        assert!(config.save_heatmaps);
        assert!(config.calculate_feature_importance);
        assert_eq!(config.top_k_features, 10);
    }

    #[test]
    fn test_attention_visualizer_creation() {
        let config = AttentionVisualizationConfig::default();
        let features = vec!["close".to_string(), "volume".to_string(), "rsi".to_string()];

        // Create temp directory for test
        let temp_dir = tempfile::tempdir().unwrap();
        let mut config = config;
        config.output_dir = temp_dir.path().to_string_lossy().to_string();

        let visualizer = AttentionVisualizer::new(config, features);
        assert!(visualizer.is_ok());
    }

    #[tokio::test]
    async fn test_attention_statistics_calculation() {
        let config = AttentionVisualizationConfig::default();
        let features = vec!["feature1".to_string(), "feature2".to_string()];

        let temp_dir = tempfile::tempdir().unwrap();
        let mut config = config;
        config.output_dir = temp_dir.path().to_string_lossy().to_string();

        let visualizer = AttentionVisualizer::new(config, features).unwrap();

        let test_weights = Array2::from_shape_vec((2, 2), vec![0.1, 0.9, 0.3, 0.7]).unwrap();
        let stats = visualizer
            .calculate_attention_statistics(&test_weights)
            .unwrap();

        assert!(stats.mean_attention > 0.0);
        assert!(stats.max_attention >= stats.min_attention);
    }
}
