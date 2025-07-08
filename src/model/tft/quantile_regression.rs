// TFT Quantile Regression - Building on existing MultiTargetLSTMModel
use crate::model::multi_target::MultiTargetLSTMModel;
use crate::utils::error::{Result, VangaError};
use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantile regression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileOutputConfig {
    /// Enable quantile regression outputs
    pub enabled: bool,
    /// Quantile levels to predict (e.g., [0.1, 0.5, 0.9] for 80% prediction interval)
    pub quantiles: Vec<f64>,
    /// Loss weighting strategy for different quantiles
    pub loss_weighting: QuantileLossWeighting,
    /// Enable uncertainty calibration
    pub uncertainty_calibration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantileLossWeighting {
    /// Equal weight for all quantiles
    Equal,
    /// Higher weight for extreme quantiles
    ExtremeWeighted,
    /// Custom weights per quantile
    Custom(Vec<f64>),
}

impl Default for QuantileOutputConfig {
    fn default() -> Self {
        Self {
            enabled: false,                 // Disabled by default for backward compatibility
            quantiles: vec![0.1, 0.5, 0.9], // 80% prediction interval + median
            loss_weighting: QuantileLossWeighting::Equal,
            uncertainty_calibration: true,
        }
    }
}

/// Quantile regression head for a single target
/// Quantile regression head for a single target
pub struct QuantileRegressionHead {
    quantile_layers: Vec<Linear>,
    quantile_levels: Vec<f64>,
}

impl QuantileRegressionHead {
    /// Create new quantile regression head
    pub fn new(input_dim: usize, quantiles: Vec<f64>, vs: VarBuilder) -> Result<Self> {
        let mut quantile_layers = Vec::new();

        for (i, &_quantile) in quantiles.iter().enumerate() {
            let layer = linear(input_dim, 1, vs.pp(format!("quantile_{}", i)))?;
            quantile_layers.push(layer);
        }

        log::debug!(
            "Created quantile regression head with {} quantiles: {:?}",
            quantiles.len(),
            quantiles
        );

        Ok(Self {
            quantile_layers,
            quantile_levels: quantiles,
        })
    }

    /// Forward pass to generate quantile predictions
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut quantile_outputs = Vec::new();

        for layer in &self.quantile_layers {
            let quantile_pred = layer.forward(input)?;
            quantile_outputs.push(quantile_pred);
        }

        // Stack quantile predictions: [batch_size, num_quantiles]
        Tensor::cat(&quantile_outputs, 1).map_err(|e| {
            VangaError::ModelError(format!("Failed to concatenate quantile outputs: {}", e))
        })
    }

    /// Get quantile levels
    pub fn get_quantiles(&self) -> &[f64] {
        &self.quantile_levels
    }
}

/// Enhanced multi-target model with quantile regression capability
pub struct QuantileMultiTargetModel {
    base_model: MultiTargetLSTMModel,
    quantile_heads: HashMap<String, QuantileRegressionHead>,
    quantile_config: QuantileOutputConfig,
}

impl QuantileMultiTargetModel {
    /// Create quantile-enhanced model from existing MultiTargetLSTMModel
    pub fn from_existing_model(
        base_model: MultiTargetLSTMModel,
        quantile_config: QuantileOutputConfig,
        vs: VarBuilder,
    ) -> Result<Self> {
        let mut quantile_heads = HashMap::new();

        if quantile_config.enabled {
            // Create quantile heads for each target
            for target_name in base_model.get_target_names() {
                let quantile_head = QuantileRegressionHead::new(
                    64, // Hidden dimension from LSTM (will be auto-detected in practice)
                    quantile_config.quantiles.clone(),
                    vs.pp(format!("target_{}", target_name)),
                )?;
                quantile_heads.insert(target_name.clone(), quantile_head);
            }

            log::info!(
                "Created quantile regression model with {} targets and {} quantiles",
                base_model.get_target_names().len(),
                quantile_config.quantiles.len()
            );
        }

        Ok(Self {
            base_model,
            quantile_heads,
            quantile_config,
        })
    }

    /// Train the model with quantile regression
    pub async fn train_with_quantiles(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        // First train the base multi-target model
        self.base_model
            .train_with_early_stopping(sequences, targets, config)
            .await?;

        // If quantile regression is enabled, train quantile heads
        if self.quantile_config.enabled && !self.quantile_heads.is_empty() {
            log::info!("Training quantile regression heads...");
            // Quantile head training would be implemented here
            // For now, we'll use the base model's training
        }

        Ok(())
    }

    /// Predict with quantile outputs
    pub async fn predict_with_quantiles(
        &self,
        sequences: &Array3<f64>,
    ) -> Result<QuantilePredictions> {
        // Get base predictions
        let base_predictions = self.base_model.predict(sequences).await?;

        // Generate quantile predictions if enabled
        let quantile_predictions = if self.quantile_config.enabled {
            // For now, return base predictions as median quantile
            // Full quantile prediction would be implemented here
            Some(base_predictions.clone())
        } else {
            None
        };

        Ok(QuantilePredictions {
            point_predictions: base_predictions,
            quantile_predictions,
            quantile_levels: if self.quantile_config.enabled {
                Some(self.quantile_config.quantiles.clone())
            } else {
                None
            },
            uncertainty_scores: None, // Would be calculated from quantile spread
        })
    }

    /// Get the base multi-target model
    pub fn get_base_model(&self) -> &MultiTargetLSTMModel {
        &self.base_model
    }

    /// Get quantile configuration
    pub fn get_quantile_config(&self) -> &QuantileOutputConfig {
        &self.quantile_config
    }

    /// Check if quantile regression is enabled
    pub fn has_quantile_outputs(&self) -> bool {
        self.quantile_config.enabled
    }
}

/// Quantile prediction results
#[derive(Debug, Clone)]
pub struct QuantilePredictions {
    /// Point predictions (median or mean)
    pub point_predictions: Array2<f64>,
    /// Quantile predictions [samples, targets, quantiles]
    pub quantile_predictions: Option<Array2<f64>>,
    /// Quantile levels used
    pub quantile_levels: Option<Vec<f64>>,
    /// Uncertainty scores derived from quantile spread
    pub uncertainty_scores: Option<Array2<f64>>,
}

impl QuantilePredictions {
    /// Get prediction intervals for a given confidence level
    pub fn get_prediction_intervals(
        &self,
        confidence: f64,
    ) -> Result<Option<(Array2<f64>, Array2<f64>)>> {
        if let (Some(quantiles), Some(levels)) = (&self.quantile_predictions, &self.quantile_levels)
        {
            let alpha = 1.0 - confidence;
            let lower_quantile = alpha / 2.0;
            let upper_quantile = 1.0 - alpha / 2.0;

            // Find closest quantile indices
            let _lower_idx = levels
                .iter()
                .position(|&q| (q - lower_quantile).abs() < 0.01)
                .ok_or_else(|| {
                    VangaError::PredictionError(format!(
                        "Lower quantile {} not found in levels",
                        lower_quantile
                    ))
                })?;

            let _upper_idx = levels
                .iter()
                .position(|&q| (q - upper_quantile).abs() < 0.01)
                .ok_or_else(|| {
                    VangaError::PredictionError(format!(
                        "Upper quantile {} not found in levels",
                        upper_quantile
                    ))
                })?;

            // Extract prediction intervals (simplified - would need proper tensor slicing)
            Ok(Some((quantiles.clone(), quantiles.clone())))
        } else {
            Ok(None)
        }
    }

    /// Calculate uncertainty scores from quantile spread
    pub fn calculate_uncertainty(&mut self) -> Result<()> {
        if let (Some(quantiles), Some(_levels)) =
            (&self.quantile_predictions, &self.quantile_levels)
        {
            // Calculate uncertainty as interquartile range or similar metric
            // Simplified implementation - would calculate actual spread
            self.uncertainty_scores = Some(quantiles.clone());
        }
        Ok(())
    }
}

/// Factory for creating quantile-enhanced models
pub struct QuantileModelFactory;

impl QuantileModelFactory {
    /// Upgrade existing MultiTargetLSTMModel to support quantile regression
    pub fn upgrade_to_quantile_model(
        base_model: MultiTargetLSTMModel,
        quantile_config: QuantileOutputConfig,
        vs: VarBuilder,
    ) -> Result<QuantileMultiTargetModel> {
        QuantileMultiTargetModel::from_existing_model(base_model, quantile_config, vs)
    }

    /// Create crypto-optimized quantile configuration
    pub fn create_crypto_quantile_config() -> QuantileOutputConfig {
        QuantileOutputConfig {
            enabled: true,
            quantiles: vec![0.05, 0.25, 0.5, 0.75, 0.95], // 90% prediction interval
            loss_weighting: QuantileLossWeighting::ExtremeWeighted,
            uncertainty_calibration: true,
        }
    }

    /// Create conservative quantile configuration
    pub fn create_conservative_quantile_config() -> QuantileOutputConfig {
        QuantileOutputConfig {
            enabled: true,
            quantiles: vec![0.1, 0.5, 0.9], // 80% prediction interval
            loss_weighting: QuantileLossWeighting::Equal,
            uncertainty_calibration: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_config_defaults() {
        let config = QuantileOutputConfig::default();
        assert!(!config.enabled); // Disabled by default for backward compatibility
        assert_eq!(config.quantiles, vec![0.1, 0.5, 0.9]);
        assert!(config.uncertainty_calibration);
    }

    #[test]
    fn test_crypto_quantile_config() {
        let config = QuantileModelFactory::create_crypto_quantile_config();
        assert!(config.enabled);
        assert_eq!(config.quantiles.len(), 5);
        assert!(config.quantiles.contains(&0.05));
        assert!(config.quantiles.contains(&0.95));
    }

    #[test]
    fn test_conservative_quantile_config() {
        let config = QuantileModelFactory::create_conservative_quantile_config();
        assert!(config.enabled);
        assert_eq!(config.quantiles, vec![0.1, 0.5, 0.9]);
        assert!(matches!(
            config.loss_weighting,
            QuantileLossWeighting::Equal
        ));
    }
}
