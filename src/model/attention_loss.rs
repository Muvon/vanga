// Attention-weighted loss functions for enhanced VANGA LSTM training
use crate::utils::error::{Result, VangaError};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

/// Configuration for attention-weighted loss functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionLossConfig {
    /// Base loss function type
    pub base_loss: BaseLossType,

    /// Attention weighting factor (0.0 = no attention weighting, 1.0 = full attention weighting)
    pub attention_weight: f64,

    /// Temporal consistency penalty weight
    pub temporal_consistency_weight: f64,

    /// Feature importance regularization weight
    pub feature_importance_weight: f64,

    /// Gradient flow enhancement factor
    pub gradient_flow_factor: f64,

    /// Market regime awareness
    pub regime_aware: bool,
}

impl Default for AttentionLossConfig {
    fn default() -> Self {
        Self {
            base_loss: BaseLossType::MSE,
            attention_weight: 0.3,            // Moderate attention influence
            temporal_consistency_weight: 0.1, // Small temporal penalty
            feature_importance_weight: 0.05,  // Light feature regularization
            gradient_flow_factor: 1.2,        // Slight gradient enhancement
            regime_aware: true,               // Enable regime awareness for crypto
        }
    }
}

/// Base loss function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaseLossType {
    MSE,      // Mean Squared Error
    MAE,      // Mean Absolute Error
    Huber,    // Huber loss (robust to outliers)
    LogCosh,  // Log-cosh loss (smooth, robust)
    Quantile, // Quantile loss for uncertainty
}

/// Attention-weighted loss calculator for cryptocurrency prediction
pub struct AttentionWeightedLoss {
    config: AttentionLossConfig,
}

impl AttentionWeightedLoss {
    /// Create new attention-weighted loss calculator
    pub fn new(config: AttentionLossConfig) -> Self {
        log::info!(
            "✅ Attention-weighted loss initialized with config: {:?}",
            config
        );
        Self { config }
    }

    /// Calculate attention-weighted loss with gradient flow enhancement
    pub fn calculate_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        attention_weights: Option<&Tensor>,
        sequence_features: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Calculate base loss
        let base_loss = self.calculate_base_loss(predictions, targets)?;

        // Apply attention weighting if available
        let attention_weighted_loss = if let Some(attention_weights) = attention_weights {
            self.apply_attention_weighting(&base_loss, attention_weights)?
        } else {
            base_loss
        };

        // Add temporal consistency penalty
        let temporal_penalty = self.calculate_temporal_consistency_penalty(predictions)?;

        // Add feature importance regularization
        let feature_penalty = if let Some(features) = sequence_features {
            self.calculate_feature_importance_penalty(features)?
        } else {
            Tensor::zeros_like(&attention_weighted_loss)?
        };

        // Combine all loss components with auto-optimized weights
        let total_loss = self.combine_loss_components(
            &attention_weighted_loss,
            &temporal_penalty,
            &feature_penalty,
        )?;

        // Apply gradient flow enhancement
        let enhanced_loss = self.enhance_gradient_flow(&total_loss)?;

        Ok(enhanced_loss)
    }

    /// Calculate base loss function
    fn calculate_base_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        match self.config.base_loss {
            BaseLossType::MSE => {
                // Mean Squared Error
                let diff = predictions.sub(targets)?;
                let squared = diff.sqr()?;
                squared.mean_all()
            }
            BaseLossType::MAE => {
                // Mean Absolute Error
                let diff = predictions.sub(targets)?;
                let abs_diff = diff.abs()?;
                abs_diff.mean_all()
            }
            BaseLossType::Huber => {
                // Huber loss (robust to outliers) - important for crypto volatility
                let delta = 1.0; // Huber delta parameter
                let diff = predictions.sub(targets)?;
                let abs_diff = diff.abs()?;

                // For |diff| <= delta: 0.5 * diff^2
                // For |diff| > delta: delta * (|diff| - 0.5 * delta)
                let delta_tensor = Tensor::new(delta as f32, predictions.device())?;
                let half_delta_squared =
                    Tensor::new((0.5 * delta * delta) as f32, predictions.device())?;

                let condition = abs_diff.le(&delta_tensor)?;
                let squared_loss = diff
                    .sqr()?
                    .mul(&Tensor::new(0.5f32, predictions.device())?)?;
                let linear_loss = delta_tensor.mul(&abs_diff)?.sub(&half_delta_squared)?;

                condition
                    .where_cond(&squared_loss, &linear_loss)?
                    .mean_all()
            }
            BaseLossType::LogCosh => {
                // Log-cosh loss (smooth and robust)
                let diff = predictions.sub(targets)?;
                // Log-cosh loss (smooth and robust) - manual implementation
                let cosh_diff = (diff.exp()? + diff.neg()?.exp()?)?
                    .div(&Tensor::new(2.0f32, predictions.device())?)?;
                cosh_diff.log()?.mean_all()
            }
            BaseLossType::Quantile => {
                // Quantile loss for uncertainty estimation
                let quantile = 0.5; // Median quantile
                let diff = targets.sub(predictions)?;
                let quantile_tensor = Tensor::new(quantile as f32, predictions.device())?;
                let one_tensor = Tensor::new(1.0f32, predictions.device())?;

                let condition = diff.ge(&Tensor::zeros_like(&diff)?)?;
                let positive_loss = quantile_tensor.mul(&diff)?;
                let negative_loss = one_tensor.sub(&quantile_tensor)?.mul(&diff.neg()?)?;

                condition
                    .where_cond(&positive_loss, &negative_loss)?
                    .mean_all()
            }
        }
        .map_err(|e| VangaError::ModelError(format!("Base loss calculation failed: {}", e)))
    }

    /// Apply attention weighting to base loss
    fn apply_attention_weighting(
        &self,
        base_loss: &Tensor,
        attention_weights: &Tensor,
    ) -> Result<Tensor> {
        // Calculate attention-based loss weighting
        // Higher attention weights should contribute more to loss
        let attention_sum = attention_weights.sum_all()?;
        let normalized_attention = attention_weights.div(&attention_sum)?;

        // Apply attention weighting factor
        let attention_factor =
            Tensor::new(self.config.attention_weight as f32, base_loss.device())?;
        let base_factor = Tensor::new(
            (1.0 - self.config.attention_weight) as f32,
            base_loss.device(),
        )?;

        // Weighted combination: (1-α) * base_loss + α * attention_weighted_loss
        let attention_weighted = base_loss.mul(&normalized_attention.mean_all()?)?;
        let weighted_loss = base_loss
            .mul(&base_factor)?
            .add(&attention_weighted.mul(&attention_factor)?)?;

        Ok(weighted_loss)
    }

    /// Calculate temporal consistency penalty for smoother predictions
    fn calculate_temporal_consistency_penalty(&self, predictions: &Tensor) -> Result<Tensor> {
        if self.config.temporal_consistency_weight == 0.0 {
            return Ok(Tensor::zeros_like(predictions)?);
        }

        // Calculate differences between consecutive predictions
        let pred_shape = predictions.shape();
        if pred_shape.dims().len() < 2 || pred_shape.dims()[1] < 2 {
            // Not enough temporal dimension for consistency penalty
            return Ok(Tensor::zeros_like(predictions)?);
        }

        let seq_len = pred_shape.dims()[1];
        let current_preds = predictions.narrow(1, 1, seq_len - 1)?;
        let previous_preds = predictions.narrow(1, 0, seq_len - 1)?;

        // Calculate temporal difference penalty
        let temporal_diff = current_preds.sub(&previous_preds)?;
        let temporal_penalty = temporal_diff.sqr()?.mean_all()?;

        // Apply temporal consistency weight
        let weight = Tensor::new(
            self.config.temporal_consistency_weight as f32,
            predictions.device(),
        )?;
        temporal_penalty.mul(&weight).map_err(|e| {
            VangaError::ModelError(format!(
                "Temporal consistency penalty calculation failed: {}",
                e
            ))
        })
    }

    /// Calculate feature importance regularization penalty
    fn calculate_feature_importance_penalty(&self, features: &Tensor) -> Result<Tensor> {
        if self.config.feature_importance_weight == 0.0 {
            return Ok(Tensor::zeros_like(features)?);
        }

        // Calculate L2 regularization on feature importance
        // This encourages the model to use features more evenly
        let feature_variance = features.var_keepdim(2)?; // Variance across features
        let importance_penalty = feature_variance.mean_all()?;

        // Apply feature importance weight
        let weight = Tensor::new(
            self.config.feature_importance_weight as f32,
            features.device(),
        )?;
        importance_penalty.mul(&weight).map_err(|e| {
            VangaError::ModelError(format!(
                "Feature importance penalty calculation failed: {}",
                e
            ))
        })
    }

    /// Combine all loss components with optimized weighting
    fn combine_loss_components(
        &self,
        attention_loss: &Tensor,
        temporal_penalty: &Tensor,
        feature_penalty: &Tensor,
    ) -> Result<Tensor> {
        // Auto-optimize component weights based on their magnitudes
        let attention_magnitude = attention_loss.abs()?.mean_all()?;
        let temporal_magnitude = temporal_penalty.abs()?.mean_all()?;
        let feature_magnitude = feature_penalty.abs()?.mean_all()?;

        // Normalize components to similar scales for balanced training
        let total_magnitude = attention_magnitude
            .add(&temporal_magnitude)?
            .add(&feature_magnitude)?;
        let epsilon = Tensor::new(1e-8f32, attention_loss.device())?;
        let safe_total = total_magnitude.add(&epsilon)?;

        let attention_weight = attention_magnitude.div(&safe_total)?;
        let temporal_weight = temporal_magnitude.div(&safe_total)?;
        let feature_weight = feature_magnitude.div(&safe_total)?;

        // Combine with normalized weights
        let weighted_attention = attention_loss.mul(&attention_weight)?;
        let weighted_temporal = temporal_penalty.mul(&temporal_weight)?;
        let weighted_feature = feature_penalty.mul(&feature_weight)?;

        let combined_loss = weighted_attention
            .add(&weighted_temporal)?
            .add(&weighted_feature)?;

        Ok(combined_loss)
    }

    /// Enhance gradient flow for better training dynamics
    fn enhance_gradient_flow(&self, loss: &Tensor) -> Result<Tensor> {
        if self.config.gradient_flow_factor == 1.0 {
            return Ok(loss.clone());
        }

        // Apply gradient flow enhancement factor
        // This can help with vanishing/exploding gradient problems
        let enhancement_factor =
            Tensor::new(self.config.gradient_flow_factor as f32, loss.device())?;
        let enhanced_loss = loss.mul(&enhancement_factor)?;

        // Optional: Apply gradient clipping-like effect
        let max_loss = Tensor::new(10.0f32, loss.device())?; // Reasonable upper bound
        let min_loss = Tensor::new(-10.0f32, loss.device())?; // Reasonable lower bound

        let clipped_loss = enhanced_loss.clamp(&min_loss, &max_loss)?;

        Ok(clipped_loss)
    }

    /// Calculate market regime-aware loss adjustments
    pub fn calculate_regime_aware_loss(
        &self,
        base_loss: &Tensor,
        market_volatility: f64,
    ) -> Result<Tensor> {
        if !self.config.regime_aware {
            return Ok(base_loss.clone());
        }

        // Adjust loss based on market regime (volatility)
        let regime_factor = match market_volatility {
            v if v < 0.02 => 1.0, // Low volatility: normal loss
            v if v < 0.05 => 1.2, // Medium volatility: slightly higher penalty
            v if v < 0.10 => 1.5, // High volatility: higher penalty
            _ => 2.0,             // Extreme volatility: much higher penalty
        };

        let regime_tensor = Tensor::new(regime_factor as f32, base_loss.device())?;
        base_loss.mul(&regime_tensor).map_err(|e| {
            VangaError::ModelError(format!("Regime-aware loss calculation failed: {}", e))
        })
    }

    /// Get loss configuration
    pub fn get_config(&self) -> &AttentionLossConfig {
        &self.config
    }

    /// Update attention weight dynamically during training
    pub fn update_attention_weight(&mut self, new_weight: f64) {
        self.config.attention_weight = new_weight.clamp(0.0, 1.0);
        log::debug!(
            "Updated attention weight to: {}",
            self.config.attention_weight
        );
    }

    /// Auto-optimize loss configuration based on training progress
    pub fn auto_optimize_for_crypto(&mut self, training_epoch: usize, validation_loss: f64) {
        // Auto-adjust attention weight based on training progress
        if training_epoch < 50 {
            // Early training: lower attention weight
            self.config.attention_weight = 0.2;
        } else if training_epoch < 200 {
            // Mid training: moderate attention weight
            self.config.attention_weight = 0.4;
        } else {
            // Late training: higher attention weight
            self.config.attention_weight = 0.6;
        }

        // Auto-adjust temporal consistency based on validation loss
        if validation_loss > 1.0 {
            // High loss: increase temporal consistency penalty
            self.config.temporal_consistency_weight = 0.2;
        } else if validation_loss < 0.1 {
            // Low loss: reduce temporal consistency penalty
            self.config.temporal_consistency_weight = 0.05;
        }

        // Auto-adjust gradient flow factor
        if validation_loss > 10.0 {
            // Very high loss: reduce gradient flow to prevent instability
            self.config.gradient_flow_factor = 0.8;
        } else if validation_loss < 0.01 {
            // Very low loss: increase gradient flow for fine-tuning
            self.config.gradient_flow_factor = 1.5;
        }

        log::debug!(
            "Auto-optimized loss config for epoch {}: attention_weight={:.3}, temporal_weight={:.3}, gradient_factor={:.3}",
            training_epoch,
            self.config.attention_weight,
            self.config.temporal_consistency_weight,
            self.config.gradient_flow_factor
        );
    }
}

/// Factory for creating attention-weighted loss functions
pub struct AttentionLossFactory;

impl AttentionLossFactory {
    /// Create attention-weighted loss for cryptocurrency prediction
    pub fn create_crypto_optimized() -> AttentionWeightedLoss {
        let config = AttentionLossConfig {
            base_loss: BaseLossType::Huber,    // Robust to crypto volatility
            attention_weight: 0.4,             // Moderate attention influence
            temporal_consistency_weight: 0.15, // Higher for crypto smoothness
            feature_importance_weight: 0.1,    // Encourage feature diversity
            gradient_flow_factor: 1.3,         // Enhanced gradient flow
            regime_aware: true,                // Essential for crypto markets
        };

        AttentionWeightedLoss::new(config)
    }

    /// Create attention-weighted loss for high-frequency trading
    pub fn create_high_frequency() -> AttentionWeightedLoss {
        let config = AttentionLossConfig {
            base_loss: BaseLossType::MAE,      // Fast convergence
            attention_weight: 0.6,             // High attention influence
            temporal_consistency_weight: 0.05, // Low temporal penalty for responsiveness
            feature_importance_weight: 0.02,   // Minimal feature regularization
            gradient_flow_factor: 1.5,         // High gradient flow for fast learning
            regime_aware: true,
        };

        AttentionWeightedLoss::new(config)
    }

    /// Create attention-weighted loss for long-term prediction
    pub fn create_long_term() -> AttentionWeightedLoss {
        let config = AttentionLossConfig {
            base_loss: BaseLossType::MSE,      // Standard for long-term
            attention_weight: 0.3,             // Moderate attention
            temporal_consistency_weight: 0.25, // High temporal consistency
            feature_importance_weight: 0.15,   // High feature regularization
            gradient_flow_factor: 1.1,         // Conservative gradient flow
            regime_aware: true,
        };

        AttentionWeightedLoss::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_attention_loss_config_defaults() {
        let config = AttentionLossConfig::default();
        assert_eq!(config.attention_weight, 0.3);
        assert!(config.regime_aware);
        assert!(matches!(config.base_loss, BaseLossType::MSE));
    }

    #[test]
    fn test_loss_factory_crypto_optimized() {
        let loss = AttentionLossFactory::create_crypto_optimized();
        assert!(matches!(loss.config.base_loss, BaseLossType::Huber));
        assert_eq!(loss.config.attention_weight, 0.4);
        assert!(loss.config.regime_aware);
    }

    #[tokio::test]
    async fn test_attention_weighted_loss_creation() {
        let config = AttentionLossConfig::default();
        let loss = AttentionWeightedLoss::new(config);
        assert_eq!(loss.get_config().attention_weight, 0.3);
    }

    #[tokio::test]
    async fn test_base_loss_calculation() {
        let device = Device::Cpu;
        let predictions = Tensor::new(&[1.0f32, 2.0, 3.0], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let targets = Tensor::new(&[1.1f32, 2.1, 2.9], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let config = AttentionLossConfig::default();
        let loss_calculator = AttentionWeightedLoss::new(config);

        let loss = loss_calculator.calculate_base_loss(&predictions, &targets);
        assert!(loss.is_ok());
    }
}
