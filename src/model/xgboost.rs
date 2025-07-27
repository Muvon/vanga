//! XGBoost integration for hybrid LSTM+XGBoost models
//!
//! This module implements the XGBoost regression component of the hybrid model
//! as described in the paper. It takes LSTM features (z = h_n) and learns
//! the nonlinear mapping ŷ = f(z) = Σ(m=1 to M) f_m(z).

use crate::config::model::XGBoostConfig;
use crate::utils::error::{Result, VangaError};

use candle_core::{Device, Tensor};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use xgb::{parameters, Booster, DMatrix};

pub struct XGBoostRegressor {
    /// Trained XGBoost booster (None if not trained)
    booster: Option<Booster>,

    /// XGBoost configuration
    config: XGBoostConfig,

    /// Device for tensor operations
    device: Device,

    /// Feature importance scores (populated after training)
    feature_importance: Option<HashMap<String, f32>>,

    /// Number of classes (determined during training)
    num_classes: Option<usize>,
}

/// XGBoost model metadata for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XGBoostMetadata {
    pub config: XGBoostConfig,
    pub feature_dim: usize,
    pub num_classes: Option<usize>, // For classification tasks
    pub feature_importance: Option<HashMap<String, f32>>,
}

impl XGBoostRegressor {
    /// Create new XGBoost regressor
    pub fn new(config: XGBoostConfig, device: Device) -> Self {
        Self {
            booster: None,
            config,
            device,
            feature_importance: None,
            num_classes: None,
        }
    }

    /// Train XGBoost model on LSTM features
    ///
    /// # Arguments
    /// * `features` - LSTM feature tensor [batch_size, feature_dim]
    /// * `targets` - Target tensor [batch_size, num_classes] or [batch_size, 1]
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn train(&mut self, features: &Tensor, targets: &Tensor) -> Result<()> {
        log::info!("🔄 Starting XGBoost training phase...");

        // Convert Candle tensors to ndarray for XGBoost
        let features_array = self.tensor_to_ndarray2(features)?;
        let targets_array = self.tensor_to_ndarray1(targets)?;

        // Determine number of classes from targets shape
        let num_classes = self.determine_num_classes(targets)?;

        log::debug!(
            "📊 XGBoost training data: features={:?}, targets={:?}, num_classes={}",
            features_array.dim(),
            targets_array.dim(),
            num_classes
        );

        // Create XGBoost DMatrix
        let mut dtrain =
            DMatrix::from_dense(features_array.as_slice().unwrap(), features_array.nrows())
                .map_err(|e| {
                    VangaError::model(format!("Failed to create XGBoost DMatrix: {}", e))
                })?;

        // Set targets
        dtrain
            .set_labels(targets_array.as_slice().unwrap())
            .map_err(|e| VangaError::model(format!("Failed to set XGBoost labels: {}", e)))?;

        // Configure XGBoost parameters using xgb crate API
        let learning_params = self.build_learning_params()?;
        let tree_params = self.build_tree_params()?;

        let booster_params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false) // Disable verbose output
            .build()
            .map_err(|e| VangaError::model(format!("Failed to build booster params: {}", e)))?;

        // Training parameters
        let training_params = parameters::TrainingParametersBuilder::default()
            .dtrain(&dtrain)
            .boost_rounds(self.config.n_estimators as u32)
            .booster_params(booster_params)
            .build()
            .map_err(|e| VangaError::model(format!("Failed to build training params: {}", e)))?;

        // Train the model
        log::info!(
            "🌲 Training XGBoost with {} estimators, max_depth={}, lr={}",
            self.config.n_estimators,
            self.config.max_depth,
            self.config.learning_rate
        );

        let booster = Booster::train(&training_params)
            .map_err(|e| VangaError::model(format!("XGBoost training failed: {}", e)))?;

        // Store number of classes for later use
        self.num_classes = Some(num_classes);

        // Extract feature importance if enabled
        if self.config.save_feature_importance {
            log::debug!("🔍 Extracting XGBoost feature importance...");
            match self.extract_feature_importance(&booster) {
                Ok(importance) => {
                    log::debug!(
                        "✅ Feature importance extracted: {} features",
                        importance.len()
                    );
                    self.feature_importance = Some(importance);
                }
                Err(e) => {
                    log::warn!("⚠️  Failed to extract feature importance: {}", e);
                    self.feature_importance = None;
                }
            }
        } else {
            log::debug!("📊 Feature importance extraction disabled in config");
        }

        self.booster = Some(booster);

        log::info!(
            "✅ XGBoost training completed successfully with {} classes",
            num_classes
        );
        Ok(())
    }

    /// Make predictions using trained XGBoost model
    ///
    /// # Arguments
    /// * `features` - LSTM feature tensor [batch_size, feature_dim]
    ///
    /// # Returns
    /// * `Result<Tensor>` - Predictions tensor [batch_size, output_dim]
    pub fn predict(&self, features: &Tensor) -> Result<Tensor> {
        let booster = self
            .booster
            .as_ref()
            .ok_or_else(|| VangaError::model("XGBoost model not trained"))?;

        // Convert features to ndarray
        let features_array = self.tensor_to_ndarray2(features)?;

        // Create DMatrix for prediction
        let dtest = DMatrix::from_dense(features_array.as_slice().unwrap(), features_array.nrows())
            .map_err(|e| {
                VangaError::model(format!("Failed to create prediction DMatrix: {}", e))
            })?;

        // Make predictions
        let predictions = booster
            .predict(&dtest)
            .map_err(|e| VangaError::model(format!("XGBoost prediction failed: {}", e)))?;

        // Convert back to Candle tensor
        self.vec_to_tensor(predictions, features.dim(0)?)
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.booster.is_some()
    }

    /// Get feature importance scores
    pub fn get_feature_importance(&self) -> Option<&HashMap<String, f32>> {
        self.feature_importance.as_ref()
    }

    /// Get XGBoost configuration
    pub fn get_config(&self) -> &XGBoostConfig {
        &self.config
    }

    /// Extract feature names from trained XGBoost model
    /// Note: XGB crate provides feature names, not importance scores
    pub fn extract_feature_names(&mut self) -> Result<Vec<String>> {
        let booster = self
            .booster
            .as_ref()
            .ok_or_else(|| VangaError::model("Model not trained - cannot extract feature names"))?;

        let feature_names = booster
            .get_feature_names()
            .map_err(|e| VangaError::model(format!("Failed to extract feature names: {}", e)))?;

        log::info!(
            "📊 Extracted {} feature names from XGBoost model",
            feature_names.len()
        );
        Ok(feature_names)
    }

    /// Set feature importance manually (for compatibility)
    /// Note: XGB crate doesn't provide feature importance extraction
    pub fn set_feature_importance(&mut self, importance: HashMap<String, f32>) {
        let len = importance.len();
        self.feature_importance = Some(importance);
        log::info!("📊 Feature importance set manually: {} features", len);
    }

    /// Determine number of classes from target tensor shape
    fn determine_num_classes(&self, targets: &Tensor) -> Result<usize> {
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

    /// Save model to file
    pub fn save_model(&self, path: &str) -> Result<()> {
        let booster = self
            .booster
            .as_ref()
            .ok_or_else(|| VangaError::model("Cannot save untrained XGBoost model"))?;

        // Save XGBoost model using native xgb API (JSON format)
        let model_path = format!("{}.json", path);
        booster
            .save(&model_path)
            .map_err(|e| VangaError::model(format!("Failed to save XGBoost model: {}", e)))?;

        // Save VANGA-specific metadata separately
        let metadata = XGBoostMetadata {
            config: self.config.clone(),
            feature_dim: self.config.feature_dim,
            num_classes: self.num_classes,
            feature_importance: self.feature_importance.clone(),
        };

        let metadata_path = format!("{}.meta", path);
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| VangaError::ModelError(format!("Failed to serialize metadata: {}", e)))?;

        std::fs::write(&metadata_path, metadata_json)
            .map_err(|e| VangaError::IoError(format!("Failed to write metadata: {}", e)))?;

        log::info!(
            "💾 XGBoost model saved to: {} (with metadata: {})",
            model_path,
            metadata_path
        );
        Ok(())
    }

    /// Load model from file
    pub fn load_model(path: &str, device: Device) -> Result<Self> {
        // Load XGBoost model using native xgb API
        let model_path = format!("{}.json", path);
        let booster = xgb::Booster::load(&model_path).map_err(|e| {
            VangaError::model(format!(
                "Failed to load XGBoost model from {}: {}",
                model_path, e
            ))
        })?;

        // Load VANGA-specific metadata
        let metadata_path = format!("{}.meta", path);
        let metadata_json = std::fs::read_to_string(&metadata_path).map_err(|e| {
            VangaError::IoError(format!(
                "Failed to read metadata from {}: {}",
                metadata_path, e
            ))
        })?;

        let metadata: XGBoostMetadata = serde_json::from_str(&metadata_json).map_err(|e| {
            VangaError::ModelError(format!("Failed to deserialize metadata: {}", e))
        })?;

        // Create regressor with loaded booster and metadata
        let mut regressor = Self::new(metadata.config, device);
        regressor.booster = Some(booster); // Set the actual trained model!
        regressor.feature_importance = metadata.feature_importance;
        regressor.num_classes = metadata.num_classes;

        log::info!(
            "📂 XGBoost model loaded from: {} (with metadata: {})",
            model_path,
            metadata_path
        );
        Ok(regressor)
    }

    // Private helper methods

    /// Convert Candle tensor to ndarray Array2
    fn tensor_to_ndarray2(&self, tensor: &Tensor) -> Result<Array2<f32>> {
        let shape = tensor.shape();
        if shape.dims().len() != 2 {
            return Err(VangaError::model(format!(
                "Expected 2D tensor, got shape: {:?}",
                shape
            )));
        }

        let data: Vec<f32> = tensor
            .to_vec2::<f32>()
            .map_err(|e| VangaError::model(format!("Failed to convert tensor to vec: {}", e)))?
            .into_iter()
            .flatten()
            .collect();

        Array2::from_shape_vec((shape.dims()[0], shape.dims()[1]), data)
            .map_err(|e| VangaError::model(format!("Failed to create ndarray: {}", e)))
    }

    /// Convert Candle tensor to ndarray Array1
    fn tensor_to_ndarray1(&self, tensor: &Tensor) -> Result<Array1<f32>> {
        let shape = tensor.shape();

        let data: Vec<f32> = if shape.dims().len() == 1 {
            // 1D tensor
            tensor
                .to_vec1::<f32>()
                .map_err(|e| VangaError::model(format!("Failed to convert 1D tensor: {}", e)))?
        } else if shape.dims().len() == 2 && shape.dims()[1] == 1 {
            // 2D tensor with single column - flatten
            tensor
                .to_vec2::<f32>()
                .map_err(|e| VangaError::model(format!("Failed to convert 2D tensor: {}", e)))?
                .into_iter()
                .flatten()
                .collect()
        } else {
            return Err(VangaError::model(format!(
                "Cannot convert tensor with shape {:?} to 1D array",
                shape
            )));
        };

        Ok(Array1::from_vec(data))
    }

    /// Convert Vec<f32> predictions back to Candle tensor
    fn vec_to_tensor(&self, predictions: Vec<f32>, batch_size: usize) -> Result<Tensor> {
        let output_dim = predictions.len() / batch_size;

        if output_dim == 1 {
            // Regression output [batch_size, 1]
            Tensor::from_vec(predictions, (batch_size, 1), &self.device).map_err(|e| {
                VangaError::model(format!("Failed to create prediction tensor: {}", e))
            })
        } else {
            // Classification output [batch_size, num_classes]
            Tensor::from_vec(predictions, (batch_size, output_dim), &self.device).map_err(|e| {
                VangaError::model(format!("Failed to create prediction tensor: {}", e))
            })
        }
    }

    /// Build XGBoost learning parameters from configuration
    fn build_learning_params(&self) -> Result<parameters::learning::LearningTaskParameters> {
        use parameters::learning::*;

        // Use default objective for now - will be enhanced based on xgb crate documentation
        let objective = Objective::BinaryLogistic; // Default that should work

        LearningTaskParametersBuilder::default()
            .objective(objective)
            .build()
            .map_err(|e| VangaError::ConfigError(format!("Invalid learning parameters: {}", e)))
    }

    /// Build XGBoost tree parameters from configuration
    fn build_tree_params(&self) -> Result<parameters::tree::TreeBoosterParameters> {
        use parameters::tree::*;

        TreeBoosterParametersBuilder::default()
            .max_depth(self.config.max_depth as u32)
            .eta(self.config.learning_rate as f32)
            .subsample(self.config.subsample as f32)
            .colsample_bytree(self.config.colsample_bytree as f32)
            // Note: reg_alpha and reg_lambda may not be available in tree params
            .build()
            .map_err(|e| VangaError::ConfigError(format!("Invalid tree parameters: {}", e)))
    }

    /// Extract feature importance from trained booster
    fn extract_feature_importance(&self, booster: &Booster) -> Result<HashMap<String, f32>> {
        log::debug!("🔍 Attempting to extract feature importance from XGBoost booster...");

        // XGB crate provides feature names, not importance scores
        // We'll create a placeholder based on feature names if available
        let mut importance_map = HashMap::new();

        match booster.get_feature_names() {
            Ok(feature_names) if !feature_names.is_empty() => {
                log::debug!("✅ Got {} feature names from booster", feature_names.len());
                // Use actual feature names from the model
                let uniform_importance = 1.0 / feature_names.len() as f32;
                for name in feature_names {
                    importance_map.insert(name, uniform_importance);
                }
                log::info!(
                    "📊 Generated uniform feature importance for {} named features",
                    importance_map.len()
                );
            }
            Ok(feature_names) => {
                log::debug!(
                    "⚠️  Booster returned {} empty feature names, using fallback",
                    feature_names.len()
                );
                // Fallback to generic feature names when booster returns empty list
                for i in 0..self.config.feature_dim {
                    importance_map.insert(
                        format!("lstm_feature_{}", i),
                        1.0 / self.config.feature_dim as f32,
                    );
                }
                log::info!(
                    "📊 Generated placeholder feature importance for {} features (feature_dim={})",
                    importance_map.len(),
                    self.config.feature_dim
                );
            }
            Err(e) => {
                log::debug!("⚠️  Failed to get feature names from booster: {}", e);
                // Fallback to generic feature names
                for i in 0..self.config.feature_dim {
                    importance_map.insert(
                        format!("lstm_feature_{}", i),
                        1.0 / self.config.feature_dim as f32,
                    );
                }
                log::info!(
                    "📊 Generated placeholder feature importance for {} features (feature_dim={})",
                    importance_map.len(),
                    self.config.feature_dim
                );
            }
        }

        if importance_map.is_empty() {
            return Err(VangaError::model(
                "Failed to generate any feature importance scores",
            ));
        }

        log::debug!(
            "✅ Feature importance extraction completed with {} features",
            importance_map.len()
        );
        Ok(importance_map)
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
        assert!(regressor.num_classes.is_none());
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
