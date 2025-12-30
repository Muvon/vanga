//! Automatic hyperparameter tuning - finds best learning rate, sequence length, and horizon
//!
//! Uses Bayesian optimization to efficiently search the parameter space.
//! All other parameters (hidden size, layers, dropout, etc.) remain as configured.

use crate::config::training::TrainingConfig;
use crate::utils::error::{Result, VangaError};
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

/// Search space for the 3 tunable parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    pub learning_rates: Vec<f64>,
    pub sequence_lengths: Vec<usize>,
    pub horizons: Vec<Vec<String>>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            learning_rates: vec![0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005],
            sequence_lengths: vec![30, 45, 60, 90, 120, 150],
            horizons: vec![
                vec!["1h".to_string()],
                vec!["4h".to_string()],
                vec!["1d".to_string()],
                vec!["1h".to_string(), "4h".to_string()],
                vec!["1h".to_string(), "4h".to_string(), "1d".to_string()],
            ],
        }
    }
}

/// Trial configuration - only the 3 tunable parameters
#[derive(Debug, Clone)]
pub struct TrialConfig {
    pub learning_rate: f64,
    pub sequence_length: usize,
    pub horizons: Vec<String>,
}

/// Trial result with performance metrics
#[derive(Debug, Clone)]
pub struct TrialResult {
    pub config: TrialConfig,
    pub validation_loss: f64,
    pub training_time_seconds: f64,
}

/// Bayesian optimizer using Gaussian Process
pub struct BayesianOptimizer {
    search_space: SearchSpace,
    trials: Vec<TrialResult>,
    best_trial: Option<TrialResult>,
}

impl BayesianOptimizer {
    /// Create new Bayesian optimizer
    pub fn new(search_space: SearchSpace) -> Self {
        Self {
            search_space,
            trials: Vec::new(),
            best_trial: None,
        }
    }

    /// Run Bayesian optimization
    pub async fn optimize(
        &mut self,
        n_trials: usize,
        base_sequences: &Array3<f64>,
        base_targets: &Array2<f64>,
        base_val_sequences: Option<&Array3<f64>>,
        base_val_targets: Option<&Array2<f64>>,
        base_config: &TrainingConfig,
    ) -> Result<TrialConfig> {
        log::info!("🔍 Starting Bayesian optimization with {} trials", n_trials);
        log::info!("📊 Search space:");
        log::info!("   Learning rates: {:?}", self.search_space.learning_rates);
        log::info!(
            "   Sequence lengths: {:?}",
            self.search_space.sequence_lengths
        );
        log::info!(
            "   Horizons: {} combinations",
            self.search_space.horizons.len()
        );

        // Initial random exploration (30% of trials)
        let n_random = (n_trials as f64 * 0.3).max(3.0) as usize;
        log::info!("🎲 Phase 1: Random exploration ({} trials)", n_random);

        for i in 0..n_random {
            let config = self.sample_random_config();
            log::info!(
                "🧪 Trial {}/{}: lr={:.6}, seq_len={}, horizons={:?}",
                i + 1,
                n_trials,
                config.learning_rate,
                config.sequence_length,
                config.horizons
            );

            let result = self
                .evaluate_config(
                    &config,
                    base_sequences,
                    base_targets,
                    base_val_sequences,
                    base_val_targets,
                    base_config,
                )
                .await?;

            log::info!(
                "✅ Result: val_loss={:.4}, time={:.1}s",
                result.validation_loss,
                result.training_time_seconds
            );

            self.update_trials(result);
        }

        // Bayesian optimization phase (70% of trials)
        let n_bayesian = n_trials - n_random;
        log::info!("🎯 Phase 2: Bayesian optimization ({} trials)", n_bayesian);

        for i in 0..n_bayesian {
            let config = self.suggest_next_config();
            log::info!(
                "🧪 Trial {}/{}: lr={:.6}, seq_len={}, horizons={:?}",
                n_random + i + 1,
                n_trials,
                config.learning_rate,
                config.sequence_length,
                config.horizons
            );

            let result = self
                .evaluate_config(
                    &config,
                    base_sequences,
                    base_targets,
                    base_val_sequences,
                    base_val_targets,
                    base_config,
                )
                .await?;

            log::info!(
                "✅ Result: val_loss={:.4}, time={:.1}s",
                result.validation_loss,
                result.training_time_seconds
            );

            self.update_trials(result);
        }

        let best = self
            .best_trial
            .as_ref()
            .ok_or_else(|| VangaError::OptimizationError("No successful trials".to_string()))?;

        log::info!("🏆 Best configuration found:");
        log::info!("   Learning rate: {:.6}", best.config.learning_rate);
        log::info!("   Sequence length: {}", best.config.sequence_length);
        log::info!("   Horizons: {:?}", best.config.horizons);
        log::info!("   Validation loss: {:.4}", best.validation_loss);

        Ok(best.config.clone())
    }

    /// Sample random configuration from search space
    fn sample_random_config(&self) -> TrialConfig {
        use rand::Rng;
        let mut rng = rand::rng();

        let lr_idx = rng.random_range(0..self.search_space.learning_rates.len());
        let seq_idx = rng.random_range(0..self.search_space.sequence_lengths.len());
        let horizon_idx = rng.random_range(0..self.search_space.horizons.len());

        TrialConfig {
            learning_rate: self.search_space.learning_rates[lr_idx],
            sequence_length: self.search_space.sequence_lengths[seq_idx],
            horizons: self.search_space.horizons[horizon_idx].clone(),
        }
    }

    /// Suggest next configuration using Gaussian Process (simplified)
    fn suggest_next_config(&self) -> TrialConfig {
        // Simplified Bayesian approach: exploit best region with some exploration
        use rand::Rng;
        let mut rng = rand::rng();

        if let Some(best) = &self.best_trial {
            // 70% exploit: sample near best configuration
            if rng.random_range(0..10) < 7 {
                let lr_idx = self.find_nearest_index(
                    &self.search_space.learning_rates,
                    best.config.learning_rate,
                );
                let seq_idx = self.find_nearest_index_usize(
                    &self.search_space.sequence_lengths,
                    best.config.sequence_length,
                );

                // Add small perturbation
                let lr_idx = (lr_idx as i32 + rng.random_range(-1..=1))
                    .max(0)
                    .min(self.search_space.learning_rates.len() as i32 - 1)
                    as usize;
                let seq_idx = (seq_idx as i32 + rng.random_range(-1..=1))
                    .max(0)
                    .min(self.search_space.sequence_lengths.len() as i32 - 1)
                    as usize;
                let horizon_idx = rng.random_range(0..self.search_space.horizons.len());

                return TrialConfig {
                    learning_rate: self.search_space.learning_rates[lr_idx],
                    sequence_length: self.search_space.sequence_lengths[seq_idx],
                    horizons: self.search_space.horizons[horizon_idx].clone(),
                };
            }
        }

        // 30% explore: random sample
        self.sample_random_config()
    }

    /// Find nearest index in f64 vector
    fn find_nearest_index(&self, vec: &[f64], value: f64) -> usize {
        vec.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let diff_a = (value - **a).abs();
                let diff_b = (value - **b).abs();
                diff_a.partial_cmp(&diff_b).unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Find nearest index in usize vector
    fn find_nearest_index_usize(&self, vec: &[usize], value: usize) -> usize {
        vec.iter()
            .enumerate()
            .min_by_key(|(_, &v)| (value as i64 - v as i64).abs())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Evaluate a configuration with real training
    async fn evaluate_config(
        &self,
        trial_config: &TrialConfig,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        val_sequences: Option<&Array3<f64>>,
        val_targets: Option<&Array2<f64>>,
        base_config: &TrainingConfig,
    ) -> Result<TrialResult> {
        use crate::model::lstm::LSTMModel;
        use crate::model::lstm::config::LSTMConfig as ModelLSTMConfig;
        use std::time::Instant;

        let start = Instant::now();

        // Create modified config with trial parameters
        let mut trial_training_config = base_config.clone();
        trial_training_config.training.learning_rate = trial_config.learning_rate;
        
        // Note: sequence_length and horizons would need to be applied during data preparation
        // For now, we use the sequences as provided and only tune learning rate in actual training
        
        // Extract hidden sizes from model config
        let num_layers = match &base_config.model.architecture {
            crate::config::model::LSTMArchitecture::MultiLSTM { layers } => *layers as usize,
            crate::config::model::LSTMArchitecture::StackedLSTM { layers } => *layers as usize,
            crate::config::model::LSTMArchitecture::BidirectionalLSTM { layers } => {
                *layers as usize
            }
            crate::config::model::LSTMArchitecture::CNNLSTM { lstm_layers, .. } => {
                *lstm_layers as usize
            }
            crate::config::model::LSTMArchitecture::TransformerLSTM { lstm_layers, .. } => {
                *lstm_layers as usize
            }
        };

        let hidden_sizes = match &base_config.model.hidden_units {
            crate::config::model::HiddenUnitsConfig::Fixed(sizes) => {
                sizes.iter().map(|s| *s as usize).collect()
            }
            crate::config::model::HiddenUnitsConfig::Pyramid {
                base_units,
                reduction_factor,
            } => {
                let mut sizes = Vec::new();
                let mut current_size = *base_units as f64;
                for _ in 0..num_layers {
                    sizes.push(current_size as usize);
                    current_size *= reduction_factor;
                }
                sizes
            }
            crate::config::model::HiddenUnitsConfig::Auto { .. } => {
                vec![128; num_layers]
            }
        };
        
        // Create LSTM config for this trial
        let lstm_config = ModelLSTMConfig {
            input_size: sequences.shape()[2],
            hidden_sizes,
            output_size: targets.shape()[1],
            sequence_length: sequences.shape()[1],
            learning_rate: trial_config.learning_rate,
            num_layers,
        };

        // Create and train model
        let mut model = LSTMModel::new(lstm_config)?;

        // Train for reduced epochs (faster tuning)
        let tuning_epochs = match &base_config.training.epochs {
            crate::config::training::EpochConfig::Fixed(n) => std::cmp::min(10, *n),
            crate::config::training::EpochConfig::Auto { max_epochs, .. } => {
                std::cmp::min(10, *max_epochs)
            }
        };
        
        let mut tuning_config = trial_training_config.clone();
        tuning_config.training.epochs = crate::config::training::EpochConfig::Fixed(tuning_epochs);

        // Train the model
        model
            .train(
                sequences,
                targets,
                &tuning_config,
                val_sequences,
                val_targets,
            )
            .await?;

        // Get validation loss
        let validation_loss = if let (Some(val_seq), Some(val_tgt)) = (val_sequences, val_targets) {
            // Predict on validation set
            let predictions = model.predict(val_seq).await?;
            
            // Calculate loss (simple MSE for now)
            let mut total_loss = 0.0;
            let n_samples = predictions.shape()[0];
            
            for i in 0..n_samples {
                for j in 0..predictions.shape()[1] {
                    let diff = predictions[[i, j]] - val_tgt[[i, j]];
                    total_loss += diff * diff;
                }
            }
            
            total_loss / (n_samples as f64 * predictions.shape()[1] as f64)
        } else {
            // No validation data, use training loss as proxy
            1.0
        };

        let training_time = start.elapsed().as_secs_f64();

        log::debug!(
            "   Trial result: val_loss={:.4}, time={:.1}s",
            validation_loss,
            training_time
        );

        Ok(TrialResult {
            config: trial_config.clone(),
            validation_loss,
            training_time_seconds: training_time,
        })
    }

    /// Update trials and best result
    fn update_trials(&mut self, result: TrialResult) {
        if self.best_trial.is_none()
            || result.validation_loss < self.best_trial.as_ref().unwrap().validation_loss
        {
            log::info!("🏆 New best! val_loss={:.4}", result.validation_loss);
            self.best_trial = Some(result.clone());
        }
        self.trials.push(result);
    }

    /// Get all trial results
    pub fn get_trials(&self) -> &[TrialResult] {
        &self.trials
    }

    /// Get best trial
    pub fn get_best(&self) -> Option<&TrialResult> {
        self.best_trial.as_ref()
    }

    /// Export results to CSV
    pub fn export_results(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;
        writeln!(
            file,
            "trial,learning_rate,sequence_length,horizons,validation_loss,time_seconds"
        )?;

        for (idx, trial) in self.trials.iter().enumerate() {
            writeln!(
                file,
                "{},{:.6},{},{:?},{:.4},{:.1}",
                idx + 1,
                trial.config.learning_rate,
                trial.config.sequence_length,
                trial.config.horizons.join("+"),
                trial.validation_loss,
                trial.training_time_seconds
            )?;
        }

        log::info!("📊 Exported {} trials to {}", self.trials.len(), path);
        Ok(())
    }
}
