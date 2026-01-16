// MoH Attention Wrapper for integration with VANGA training pipeline
// Handles the mutable reference requirement for routing state management

use crate::config::model::{AttentionConfig, AttentionMechanism};
use crate::model::attention::AttentionModule;
use crate::model::attention_moh::MixtureOfHeadAttention;
use crate::utils::error::{Result, VangaError};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use std::cell::RefCell;
use std::collections::HashMap;

/// Thread-safe wrapper for MixtureOfHeadAttention that implements AttentionModule
pub struct MoHAttentionWrapper {
    inner: RefCell<MixtureOfHeadAttention>,
}

impl MoHAttentionWrapper {
    /// Create new MoH attention wrapper
    pub fn new(
        input_dim: usize,
        config: AttentionConfig,
        vs: VarBuilder,
        device: Device,
    ) -> Result<Self> {
        let moh_attention = MixtureOfHeadAttention::new(input_dim, config, vs, device)?;

        Ok(Self {
            inner: RefCell::new(moh_attention),
        })
    }

    /// Get routing statistics (requires mutable access)
    pub fn get_routing_stats(&self) -> HashMap<String, f64> {
        self.inner.borrow().get_routing_stats()
    }

    /// Calculate load balance loss (requires mutable access)
    pub fn calculate_load_balance_loss(&self) -> Result<Tensor> {
        self.inner.borrow().calculate_load_balance_loss()
    }

    /// Clear routing history for memory management
    pub fn clear_routing_history(&self) {
        self.inner.borrow_mut().clear_routing_history()
    }

    /// Get MoH configuration for analysis
    pub fn get_moh_config(&self) -> crate::config::model::MoHConfig {
        self.inner
            .borrow()
            .get_config()
            .moh
            .clone()
            .unwrap_or_default()
    }
}

impl AttentionModule for MoHAttentionWrapper {
    fn forward(&self, input: &Tensor, training: bool) -> Result<(Tensor, Tensor)> {
        // Use interior mutability to handle the mutable reference requirement
        self.inner.borrow_mut().forward(input, training)
    }

    fn get_config(&self) -> AttentionConfig {
        // Return owned config to avoid lifetime issues
        self.inner.borrow().get_config().clone()
    }
}

/// Enhanced AttentionFactory that properly handles MoH creation
pub struct EnhancedAttentionFactory;

impl EnhancedAttentionFactory {
    /// Create attention mechanism with proper MoH support
    pub fn create_attention(
        attention_type: &AttentionMechanism,
        input_dim: usize,
        config: AttentionConfig,
        vs: VarBuilder,
        device: Device,
    ) -> Result<Box<dyn AttentionModule>> {
        match attention_type {
            AttentionMechanism::MixtureOfHeads => {
                // Validate MoH configuration
                if let Some(ref moh_config) = config.moh {
                    moh_config.validate().map_err(|e| {
                        VangaError::ModelError(format!("MoH config validation failed: {}", e))
                    })?;
                } else {
                    return Err(VangaError::ModelError(
                        "MoH configuration required for MixtureOfHeads mechanism".to_string(),
                    ));
                }

                let wrapper = MoHAttentionWrapper::new(input_dim, config, vs, device)?;
                log::info!("✅ Created MixtureOfHeads attention with wrapper");
                Ok(Box::new(wrapper))
            }
            _ => {
                // Delegate to original factory for other attention types
                use crate::model::attention::AttentionFactory;
                AttentionFactory::create_attention(attention_type, input_dim, config, vs, device)
            }
        }
    }
}

/// MoH-aware training loss that includes load balance loss
pub struct MoHTrainingLoss {
    pub task_loss: Tensor,
    pub load_balance_loss: Option<Tensor>,
    pub total_loss: Tensor,
}

impl MoHTrainingLoss {
    /// Create MoH training loss with load balance component
    pub fn new(
        task_loss: Tensor,
        attention_module: Option<&MoHAttentionWrapper>,
        load_balance_weight: f64,
    ) -> Result<Self> {
        let load_balance_loss = if let Some(moh_attention) = attention_module {
            let lb_loss = moh_attention.calculate_load_balance_loss()?;
            Some(lb_loss)
        } else {
            None
        };

        let total_loss = if let Some(ref lb_loss) = load_balance_loss {
            let weight_tensor = Tensor::new(&[load_balance_weight as f32], task_loss.device())?;
            let weighted_lb_loss = lb_loss.broadcast_mul(&weight_tensor)?.contiguous()?;
            (task_loss.clone() + weighted_lb_loss)?.contiguous()?
        } else {
            task_loss.clone()
        };

        Ok(Self {
            task_loss,
            load_balance_loss,
            total_loss,
        })
    }

    /// Get the total loss for backpropagation
    pub fn total(&self) -> &Tensor {
        &self.total_loss
    }

    /// Get task loss value
    pub fn task_loss_value(&self) -> Result<f32> {
        self.task_loss
            .to_scalar::<f32>()
            .map_err(|e| VangaError::ModelError(format!("Task loss extraction failed: {}", e)))
    }

    /// Get load balance loss value
    pub fn load_balance_loss_value(&self) -> Result<Option<f32>> {
        if let Some(ref lb_loss) = self.load_balance_loss {
            Ok(Some(lb_loss.to_scalar::<f32>().map_err(|e| {
                VangaError::ModelError(format!("Load balance loss extraction failed: {}", e))
            })?))
        } else {
            Ok(None)
        }
    }

    /// Get total loss value
    pub fn total_loss_value(&self) -> Result<f32> {
        self.total_loss
            .to_scalar::<f32>()
            .map_err(|e| VangaError::ModelError(format!("Total loss extraction failed: {}", e)))
    }
}

/// MoH performance metrics for analysis
#[derive(Debug, Clone)]
pub struct MoHMetrics {
    pub efficiency_ratio: f64,
    pub active_heads: u32,
    pub total_heads: u32,
    pub routing_entropy: f64,
    pub load_balance_loss: f64,
    pub routing_stability: f64,
}

impl MoHMetrics {
    /// Calculate MoH metrics from attention wrapper
    pub fn from_attention(attention: &MoHAttentionWrapper) -> Result<Self> {
        let stats = attention.get_routing_stats();
        let config = attention.get_moh_config();
        let load_balance_loss = attention
            .calculate_load_balance_loss()?
            .to_scalar::<f32>()? as f64;

        Ok(Self {
            efficiency_ratio: config.efficiency_ratio(),
            active_heads: config.active_heads(),
            total_heads: config.total_heads,
            routing_entropy: stats.get("routing_entropy").copied().unwrap_or(0.0),
            load_balance_loss,
            routing_stability: stats.get("routing_stability").copied().unwrap_or(0.0),
        })
    }

    /// Log metrics for monitoring
    pub fn log_metrics(&self, epoch: usize) {
        log::info!(
            "📊 MoH Metrics [Epoch {}]: efficiency={:.1}%, active_heads={}/{}, lb_loss={:.6}, entropy={:.3}",
            epoch,
            self.efficiency_ratio * 100.0,
            self.active_heads,
            self.total_heads,
            self.load_balance_loss,
            self.routing_entropy
        );
    }
}
