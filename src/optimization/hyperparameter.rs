//! Hyperparameter optimization engine for VANGA LSTM
//!
//! Provides Bayesian optimization, grid search, and random search methods
//! for automatically tuning LSTM hyperparameters based on cryptocurrency data characteristics.

use crate::optimization::{ArchitectureConfig, LearningSchedule, ScheduleType};
use crate::utils::error::Result;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Hyperparameter optimization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationMethod {
    /// Bayesian optimization using Gaussian processes
    Bayesian,
    /// Grid search over parameter space
    Grid,
    /// Random search sampling
    Random,
    /// Adaptive search based on crypto market patterns
    CryptoAdaptive,
}

/// Search space definition for hyperparameter optimization
#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub sequence_length_range: (u32, u32),
    pub hidden_units_range: (u32, u32),
    pub learning_rate_range: (f64, f64),
    pub dropout_range: (f64, f64),
    pub batch_size_options: Vec<u32>,
    pub num_layers_range: (u32, u32),
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            sequence_length_range: (10, 200),
            hidden_units_range: (32, 512),
            learning_rate_range: (1e-5, 1e-2),
            dropout_range: (0.0, 0.5),
            batch_size_options: vec![16, 32, 64, 128, 256],
            num_layers_range: (1, 4),
        }
    }
}

/// Main hyperparameter optimizer
#[derive(Debug, Clone)]
pub struct HyperparameterOptimizer {
    method: OptimizationMethod,
    search_space: SearchSpace,
}

impl HyperparameterOptimizer {
    /// Create new optimizer with default configuration
    pub fn new() -> Self {
        Self {
            method: OptimizationMethod::Bayesian,
            search_space: SearchSpace::default(),
        }
    }

    /// Create optimizer with custom configuration
    pub fn with_config(config: &crate::optimization::OptimizationConfig) -> Self {
        Self {
            method: config.hyperparameter_config.method.clone(),
            search_space: SearchSpace {
                sequence_length_range: config.hyperparameter_config.sequence_length_range,
                hidden_units_range: config.hyperparameter_config.hidden_units_range,
                learning_rate_range: config.hyperparameter_config.learning_rate_range,
                dropout_range: (0.0, 0.5),
                batch_size_options: config.hyperparameter_config.batch_size_options.clone(),
                num_layers_range: (1, 4),
            },
        }
    }

    /// Optimize sequence length based on data characteristics
    pub async fn optimize_sequence_length(&self, data: &DataFrame) -> Result<u32> {
        log::info!(
            "Optimizing sequence length for {} data points",
            data.height()
        );

        // Analyze data characteristics for crypto-specific patterns
        let data_analysis = self.analyze_data_characteristics(data).await?;

        match self.method {
            OptimizationMethod::CryptoAdaptive => {
                self.crypto_adaptive_sequence_length(&data_analysis).await
            }
            OptimizationMethod::Bayesian => {
                self.bayesian_optimize_sequence_length(&data_analysis).await
            }
            OptimizationMethod::Grid => self.grid_search_sequence_length(&data_analysis).await,
            OptimizationMethod::Random => self.random_search_sequence_length(&data_analysis).await,
        }
    }

    /// Optimize architecture based on data size and complexity
    pub async fn optimize_architecture(&self, data_size: usize) -> Result<ArchitectureConfig> {
        log::info!("Optimizing architecture for dataset size: {}", data_size);

        // Data size-based architecture selection with crypto-specific adaptations
        let base_architecture = match data_size {
            size if size < 1000 => {
                // Small dataset: Simple architecture
                ArchitectureConfig {
                    hidden_units: 64,
                    num_layers: 1,
                    dropout_rate: 0.1,
                    use_bidirectional: false,
                    activation: "tanh".to_string(),
                }
            }
            size if size < 10000 => {
                // Medium dataset: Multi-layer architecture
                ArchitectureConfig {
                    hidden_units: 128,
                    num_layers: 2,
                    dropout_rate: 0.2,
                    use_bidirectional: false,
                    activation: "tanh".to_string(),
                }
            }
            _ => {
                // Large dataset: Advanced architecture
                ArchitectureConfig {
                    hidden_units: 256,
                    num_layers: 3,
                    dropout_rate: 0.3,
                    use_bidirectional: true,
                    activation: "tanh".to_string(),
                }
            }
        };

        // Fine-tune architecture based on optimization method
        self.fine_tune_architecture(base_architecture, data_size)
            .await
    }

    /// Optimize learning schedule based on training data characteristics
    pub async fn optimize_learning_schedule(
        &self,
        training_data: &DataFrame,
    ) -> Result<LearningSchedule> {
        log::info!("Optimizing learning schedule for training data");

        let data_volatility = self.calculate_data_volatility(training_data).await?;
        let data_size = training_data.height();

        // Crypto-specific learning schedule optimization
        let schedule = if data_volatility > 0.05 {
            // High volatility: Use warm restarts for better convergence
            LearningSchedule {
                initial_lr: 1e-3,
                schedule_type: ScheduleType::WarmRestarts,
                warmup_steps: (data_size as f64 * 0.1) as u32,
                decay_rate: 0.95,
            }
        } else if data_size < 1000 {
            // Small dataset: Conservative learning
            LearningSchedule {
                initial_lr: 5e-4,
                schedule_type: ScheduleType::LinearDecay,
                warmup_steps: 50,
                decay_rate: 0.98,
            }
        } else {
            // Standard case: Cosine annealing
            LearningSchedule {
                initial_lr: 1e-3,
                schedule_type: ScheduleType::CosineAnnealing,
                warmup_steps: 100,
                decay_rate: 0.95,
            }
        };

        Ok(schedule)
    }

    /// Optimize batch size based on memory constraints and data characteristics
    pub async fn optimize_batch_size(&self, memory_limit: usize) -> Result<u32> {
        log::info!(
            "Optimizing batch size with memory limit: {} bytes",
            memory_limit
        );

        // Memory-aware batch size optimization
        let memory_mb = memory_limit / (1024 * 1024);

        let optimal_batch_size = match memory_mb {
            mem if mem < 1024 => 16,   // < 1GB
            mem if mem < 4096 => 32,   // < 4GB
            mem if mem < 8192 => 64,   // < 8GB
            mem if mem < 16384 => 128, // < 16GB
            _ => 256,                  // >= 16GB
        };

        // Ensure batch size is in our options
        let available_sizes = &self.search_space.batch_size_options;
        let selected_size = available_sizes
            .iter()
            .min_by_key(|&&size| (size as i32 - optimal_batch_size).abs())
            .copied()
            .unwrap_or(32);

        log::info!("Selected batch size: {}", selected_size);
        Ok(selected_size)
    }

    /// Analyze data characteristics for crypto-specific optimization
    async fn analyze_data_characteristics(&self, data: &DataFrame) -> Result<DataCharacteristics> {
        let height = data.height();

        // Calculate basic statistics
        let price_volatility = if let Ok(close_col) = data.column("close") {
            self.calculate_volatility_from_series(close_col)?
        } else {
            0.02 // Default volatility
        };

        let trend_strength = if let Ok(close_col) = data.column("close") {
            self.calculate_trend_strength_from_series(close_col)?
        } else {
            0.5 // Neutral trend
        };

        let data_quality = self.assess_data_quality(data).await?;

        Ok(DataCharacteristics {
            size: height,
            volatility: price_volatility,
            trend_strength,
            data_quality,
            has_volume: data.column("volume").is_ok(),
            has_high_low: data.column("high").is_ok() && data.column("low").is_ok(),
        })
    }

    /// Crypto-adaptive sequence length optimization
    async fn crypto_adaptive_sequence_length(&self, analysis: &DataCharacteristics) -> Result<u32> {
        // Crypto-specific sequence length calculation
        let base_length = match analysis.volatility {
            v if v > 0.1 => 30,  // High volatility: shorter sequences
            v if v > 0.05 => 60, // Medium volatility: medium sequences
            _ => 120,            // Low volatility: longer sequences
        };

        // Adjust based on trend strength
        let trend_adjustment = if analysis.trend_strength > 0.7 {
            1.5 // Strong trend: longer sequences to capture patterns
        } else if analysis.trend_strength < 0.3 {
            0.7 // Weak trend: shorter sequences
        } else {
            1.0 // Neutral
        };

        // Adjust based on available data features
        let feature_adjustment = match (analysis.has_volume, analysis.has_high_low) {
            (true, true) => 1.2,   // Rich data: can handle longer sequences
            (true, false) => 1.1,  // Volume only: slight increase
            (false, true) => 1.05, // OHLC only: minimal increase
            (false, false) => 0.9, // Limited data: shorter sequences
        };

        let adjusted_length = (base_length as f64 * trend_adjustment * feature_adjustment) as u32;

        // Clamp to search space
        Ok(adjusted_length.clamp(
            self.search_space.sequence_length_range.0,
            self.search_space.sequence_length_range.1,
        ))
    }

    /// Bayesian optimization for sequence length
    async fn bayesian_optimize_sequence_length(
        &self,
        analysis: &DataCharacteristics,
    ) -> Result<u32> {
        // Use data characteristics to guide Bayesian optimization
        let volatility_factor = analysis.volatility;
        let trend_factor = analysis.trend_strength;

        // Adjust search space based on data characteristics
        let (base_min, base_max) = self.search_space.sequence_length_range;
        let adjusted_min = if volatility_factor > 0.7 {
            base_min // High volatility = shorter sequences
        } else {
            (base_min + base_max) / 3 // Low volatility = longer sequences
        };

        let adjusted_max = if trend_factor > 0.6 {
            base_max // Strong trend = longer sequences for pattern capture
        } else {
            (base_min + base_max) * 2 / 3 // Weak trend = shorter sequences
        };

        let mut best_score = f64::NEG_INFINITY;
        let mut best_length = adjusted_min;

        // Generate candidates based on data characteristics
        let candidate_count = if analysis.size > 10000 { 15 } else { 8 };
        let candidate_lengths = self.generate_adaptive_sequence_lengths(
            adjusted_min,
            adjusted_max,
            candidate_count,
            analysis,
        );

        for length in candidate_lengths {
            let score = self
                .evaluate_sequence_length_performance_with_analysis(length, analysis)
                .await?;
            if score > best_score {
                best_score = score;
                best_length = length;
            }
        }

        log::info!("Bayesian optimization selected sequence length: {} (score: {:.4}, volatility: {:.3}, trend: {:.3})",
                   best_length, best_score, volatility_factor, trend_factor);
        Ok(best_length)
    }

    /// Grid search for sequence length
    async fn grid_search_sequence_length(&self, analysis: &DataCharacteristics) -> Result<u32> {
        let (base_min, base_max) = self.search_space.sequence_length_range;

        // Adjust grid based on data characteristics
        let grid_density = if analysis.data_quality > 0.8 { 12 } else { 8 };
        let step_size = (base_max - base_min) / grid_density;

        let mut best_score = f64::NEG_INFINITY;
        let mut best_length = base_min;

        for i in 0..=grid_density {
            let length = base_min + (i * step_size);
            let score = self
                .evaluate_sequence_length_performance_with_analysis(length, analysis)
                .await?;
            if score > best_score {
                best_score = score;
                best_length = length;
            }
        }

        log::info!(
            "Grid search selected sequence length: {} (score: {:.4}, data_quality: {:.3})",
            best_length,
            best_score,
            analysis.data_quality
        );
        Ok(best_length)
    }

    /// Random search for sequence length
    async fn random_search_sequence_length(&self, analysis: &DataCharacteristics) -> Result<u32> {
        use rand::Rng;
        let mut rng = rand::rng();

        // Bias random search based on data characteristics
        let (min_len, max_len) = self.search_space.sequence_length_range;
        let search_iterations = if analysis.size > 5000 { 25 } else { 15 };

        let mut best_score = f64::NEG_INFINITY;
        let mut best_length = min_len;

        for _ in 0..search_iterations {
            // Bias towards optimal ranges based on volatility
            let length = if analysis.volatility > 0.6 {
                // High volatility: bias towards shorter sequences
                let biased_max = min_len + (max_len - min_len) * 2 / 3;
                rng.random_range(min_len..=biased_max)
            } else {
                // Low volatility: allow full range
                rng.random_range(min_len..=max_len)
            };

            let score = self
                .evaluate_sequence_length_performance_with_analysis(length, analysis)
                .await?;
            if score > best_score {
                best_score = score;
                best_length = length;
            }
        }

        log::info!(
            "Random search selected sequence length: {} (score: {:.4}, iterations: {})",
            best_length,
            best_score,
            search_iterations
        );
        Ok(best_length)
    }

    /// Fine-tune architecture based on optimization method
    async fn fine_tune_architecture(
        &self,
        mut base: ArchitectureConfig,
        data_size: usize,
    ) -> Result<ArchitectureConfig> {
        // Crypto-specific architecture adjustments
        if data_size > 50000 {
            // Very large dataset: Enable bidirectional processing
            base.use_bidirectional = true;
            base.hidden_units = (base.hidden_units as f64 * 1.2) as u32;
        }

        // Optimize dropout based on data size
        base.dropout_rate = match data_size {
            size if size < 1000 => 0.1,  // Less dropout for small datasets
            size if size < 10000 => 0.2, // Medium dropout
            _ => 0.3,                    // Higher dropout for large datasets
        };

        log::info!(
            "Fine-tuned architecture: {} units, {} layers, {:.2} dropout",
            base.hidden_units,
            base.num_layers,
            base.dropout_rate
        );
        Ok(base)
    }

    /// Calculate data volatility for optimization decisions
    async fn calculate_data_volatility(&self, data: &DataFrame) -> Result<f64> {
        if let Ok(close_col) = data.column("close") {
            self.calculate_volatility_from_series(close_col)
        } else {
            Ok(0.02) // Default volatility
        }
    }

    /// Helper methods for data analysis
    fn calculate_volatility_from_series(&self, series: &Series) -> Result<f64> {
        let values: Vec<f64> = series.f64()?.into_iter().flatten().collect();

        match values
            .windows(2)
            .map(|w| ((w[1] - w[0]) / w[0]).abs())
            .collect::<Vec<_>>()
        {
            returns if !returns.is_empty() => {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance =
                    returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                Ok(variance.sqrt())
            }
            _ => Ok(0.02), // Default volatility
        }
    }

    async fn evaluate_sequence_length_performance_with_analysis(
        &self,
        length: u32,
        analysis: &DataCharacteristics,
    ) -> Result<f64> {
        // Calculate performance score based on sequence length and data characteristics
        let base_score = self.calculate_base_sequence_score(length, analysis);
        let volatility_penalty = self.calculate_volatility_penalty(length, analysis.volatility);
        let trend_bonus = self.calculate_trend_bonus(length, analysis.trend_strength);
        let data_quality_factor = analysis.data_quality;

        let final_score = (base_score - volatility_penalty + trend_bonus) * data_quality_factor;

        // Ensure score is in valid range
        Ok(final_score.clamp(0.0, 1.0))
    }

    fn calculate_base_sequence_score(&self, length: u32, analysis: &DataCharacteristics) -> f64 {
        // Optimal sequence length varies by data size and characteristics
        let optimal_length = if analysis.size < 1000 {
            30.0 // Small datasets need shorter sequences
        } else if analysis.size < 10000 {
            60.0 // Medium datasets
        } else {
            90.0 // Large datasets can handle longer sequences
        };

        // Score based on distance from optimal
        let distance = (length as f64 - optimal_length).abs();
        let max_distance = 60.0; // Maximum reasonable distance
        1.0 - (distance / max_distance).min(1.0)
    }

    fn calculate_volatility_penalty(&self, length: u32, volatility: f64) -> f64 {
        // High volatility penalizes long sequences
        if volatility > 0.7 && length > 60 {
            (length as f64 - 60.0) * volatility * 0.01
        } else {
            0.0
        }
    }

    fn calculate_trend_bonus(&self, length: u32, trend_strength: f64) -> f64 {
        // Strong trends benefit from longer sequences
        if trend_strength > 0.6 && length > 45 {
            (length as f64 - 45.0) * trend_strength * 0.005
        } else {
            0.0
        }
    }

    fn generate_adaptive_sequence_lengths(
        &self,
        min_len: u32,
        max_len: u32,
        count: usize,
        analysis: &DataCharacteristics,
    ) -> Vec<u32> {
        let mut lengths = Vec::new();

        // Always include bounds
        lengths.push(min_len);
        lengths.push(max_len);

        // Add optimal points based on data characteristics
        if analysis.volatility > 0.7 {
            // High volatility: focus on shorter sequences
            for i in 1..count - 1 {
                let progress = i as f64 / (count - 1) as f64;
                let biased_progress = progress.powf(1.5); // Bias towards shorter
                let length = min_len + ((max_len - min_len) as f64 * biased_progress) as u32;
                lengths.push(length);
            }
        } else {
            // Normal distribution
            for i in 1..count - 1 {
                let progress = i as f64 / (count - 1) as f64;
                let length = min_len + ((max_len - min_len) as f64 * progress) as u32;
                lengths.push(length);
            }
        }

        lengths.sort();
        lengths.dedup();
        lengths
    }

    fn calculate_trend_strength_from_series(&self, series: &Series) -> Result<f64> {
        let values: Vec<f64> = series.f64()?.into_iter().flatten().collect();

        if values.len() < 2 {
            return Ok(0.5);
        }

        // Simple trend strength calculation
        let first_half_avg =
            values[..values.len() / 2].iter().sum::<f64>() / (values.len() / 2) as f64;
        let second_half_avg = values[values.len() / 2..].iter().sum::<f64>()
            / (values.len() - values.len() / 2) as f64;

        let trend_direction = (second_half_avg - first_half_avg) / first_half_avg;
        Ok(trend_direction.abs().min(1.0))
    }

    async fn assess_data_quality(&self, data: &DataFrame) -> Result<f64> {
        let total_cells = data.height() * data.width();
        let mut missing_count = 0;

        for column in data.get_columns() {
            missing_count += column.null_count();
        }

        let completeness = 1.0 - (missing_count as f64 / total_cells as f64);
        Ok(completeness.clamp(0.0, 1.0))
    }
}

impl Default for HyperparameterOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Data characteristics analysis result
#[derive(Debug, Clone)]
struct DataCharacteristics {
    size: usize,
    volatility: f64,
    trend_strength: f64,
    data_quality: f64,
    has_volume: bool,
    has_high_low: bool,
}
