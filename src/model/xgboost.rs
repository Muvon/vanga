//! XGBoost integration for hybrid LSTM+XGBoost models
//!
//! This module implements the XGBoost regression component of the hybrid model
//! as described in the paper. It takes LSTM features (z = h_n) and learns
//! the nonlinear mapping ŷ = f(z) = Σ(m=1 to M) f_m(z).
//!
//! **UPDATED**: Now uses SmartCore backend for better performance and reliability.

use crate::config::model::XGBoostConfig;
use crate::model::smartcore_backend::SmartCoreRegressor;
use crate::utils::error::{Result, VangaError};

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// XGBoost regressor using SmartCore backend
pub struct XGBoostRegressor {
    /// SmartCore backend (replaces problematic xgb crate)
    backend: SmartCoreRegressor,
}

/// XGBoost model metadata for persistence (backward compatibility)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XGBoostMetadata {
    pub config: XGBoostConfig,
    pub feature_dim: usize,
    pub num_classes: Option<usize>,
    pub feature_importance: Option<HashMap<String, f32>>,
}

impl XGBoostRegressor {
    /// Create new XGBoost regressor (now using SmartCore backend)
    pub fn new(config: XGBoostConfig, device: Device) -> Self {
        log::info!("🔄 Creating XGBoost regressor with SmartCore backend");
        Self {
            backend: SmartCoreRegressor::new(config, device),
        }
    }

    /// Train XGBoost model on LSTM features (now using SmartCore)
    ///
    /// # Arguments
    /// * `features` - LSTM feature tensor [batch_size, feature_dim]
    /// * `targets` - Target tensor [batch_size, num_classes] or [batch_size, 1]
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn train(&mut self, features: &Tensor, targets: &Tensor) -> Result<()> {
        log::info!("🔄 Starting XGBoost training phase (SmartCore backend)...");
        self.backend.train(features, targets)
    }

    /// Make predictions using trained XGBoost model (now using SmartCore)
    ///
    /// # Arguments
    /// * `features` - LSTM feature tensor [batch_size, feature_dim]
    ///
    /// # Returns
    /// * `Result<Tensor>` - Predictions tensor [batch_size, output_dim]
    pub fn predict(&self, features: &Tensor) -> Result<Tensor> {
        self.backend.predict(features)
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.backend.is_trained()
    }

    /// Get feature importance scores
    pub fn get_feature_importance(&self) -> Option<&HashMap<String, f32>> {
        self.backend.get_feature_importance()
    }

    /// Get XGBoost configuration
    pub fn get_config(&self) -> &XGBoostConfig {
        self.backend.get_config()
    }

    /// Extract feature names from trained XGBoost model
    pub fn extract_feature_names(&mut self) -> Result<Vec<String>> {
        let feature_dim = self.backend.get_config().feature_dim;
        let feature_names: Vec<String> = (0..feature_dim)
            .map(|i| format!("lstm_feature_{}", i))
            .collect();

        log::info!(
            "📊 Generated {} feature names for SmartCore model",
            feature_names.len()
        );
        Ok(feature_names)
    }

    /// Set feature importance manually (for compatibility)
    pub fn set_feature_importance(&mut self, importance: HashMap<String, f32>) {
        log::info!(
            "📊 Feature importance set manually: {} features",
            importance.len()
        );
        log::warn!("⚠️ Manual feature importance setting not supported with SmartCore backend");
    }

    /// Save model to file
    pub fn save_model(&self, path: &str) -> Result<()> {
        self.backend.save_model(path)
    }

    /// Load model from file
    pub fn load_model(path: &str, device: Device) -> Result<Self> {
        let backend = SmartCoreRegressor::load_model(path, device)?;
        Ok(Self { backend })
    }

    /// Determine number of classes from target tensor shape
    pub fn determine_num_classes(&self, targets: &Tensor) -> Result<usize> {
        let shape = targets.shape();
        let dims = shape.dims();

        if dims.len() == 1 {
            // 1D targets - regression or binary classification
            Ok(1)
        } else if dims.len() == 2 {
            if dims[1] == 1 {
                // [batch_size, 1] - regression or binary classification
                Ok(1)
            } else {
                // [batch_size, num_classes] - multi-class classification
                Ok(dims[1])
            }
        } else {
            Err(VangaError::model(format!(
                "Invalid target tensor shape: {:?}. Expected 1D or 2D tensor",
                dims
            )))
        }
    }
}

/// Utility functions for XGBoost integration
///
/// Determine appropriate XGBoost objective based on target type
pub fn get_objective_for_target(target_name: &str, num_classes: usize) -> String {
    if target_name.contains("price_level") || target_name.contains("direction") {
        // Classification tasks
        if num_classes == 2 {
            "binary:logistic".to_string()
        } else {
            "multi:softprob".to_string()
        }
    } else if target_name.contains("volatility") {
        // Regression task
        "reg:squarederror".to_string()
    } else {
        // Default to regression
        "reg:squarederror".to_string()
    }
}

/// Determine appropriate evaluation metric based on target type
pub fn get_eval_metric_for_target(target_name: &str, num_classes: usize) -> String {
    if target_name.contains("price_level") || target_name.contains("direction") {
        // Classification tasks
        if num_classes == 2 {
            "logloss".to_string()
        } else {
            "mlogloss".to_string()
        }
    } else if target_name.contains("volatility") {
        // Regression task
        "rmse".to_string()
    } else {
        // Default to regression
        "rmse".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_xgboost_config_default() {
        let config = XGBoostConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.feature_dim, 64);
        assert_eq!(config.n_estimators, 100);
        assert_eq!(config.max_depth, 6);
    }

    #[test]
    fn test_objective_selection() {
        assert_eq!(
            get_objective_for_target("price_level_1h", 5),
            "multi:softprob"
        );
        assert_eq!(
            get_objective_for_target("direction_4h", 5),
            "multi:softprob"
        );
        assert_eq!(
            get_objective_for_target("volatility_1d", 1),
            "reg:squarederror"
        );
    }

    #[test]
    fn test_eval_metric_selection() {
        assert_eq!(get_eval_metric_for_target("price_level_1h", 5), "mlogloss");
        assert_eq!(get_eval_metric_for_target("direction_4h", 5), "mlogloss");
        assert_eq!(get_eval_metric_for_target("volatility_1d", 1), "rmse");
    }

    #[tokio::test]
    async fn test_xgboost_regressor_creation() {
        let config = XGBoostConfig::default();
        let device = Device::Cpu;
        let regressor = XGBoostRegressor::new(config, device);

        assert!(!regressor.is_trained());
        assert!(regressor.get_feature_importance().is_none());
    }

    #[test]
    fn test_determine_num_classes() {
        let config = XGBoostConfig::default();
        let device = Device::Cpu;
        let regressor = XGBoostRegressor::new(config, device.clone());

        // Test 1D tensor (regression)
        let targets_1d = Tensor::zeros((10,), candle_core::DType::F32, &device).unwrap();
        assert_eq!(regressor.determine_num_classes(&targets_1d).unwrap(), 1);

        // Test 2D tensor with 1 column (regression/binary)
        let targets_2d_1 = Tensor::zeros((10, 1), candle_core::DType::F32, &device).unwrap();
        assert_eq!(regressor.determine_num_classes(&targets_2d_1).unwrap(), 1);

        // Test 2D tensor with multiple columns (multi-class)
        let targets_2d_5 = Tensor::zeros((10, 5), candle_core::DType::F32, &device).unwrap();
        assert_eq!(regressor.determine_num_classes(&targets_2d_5).unwrap(), 5);
    }
}
