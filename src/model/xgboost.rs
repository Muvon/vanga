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

    /// Validation DMatrix for SHAP-based feature importance calculation
    validation_dmat: Option<DMatrix>,
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
            validation_dmat: None,
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

        // Validate and adjust feature dimension based on actual LSTM output
        let actual_feature_dim = features_array.ncols();
        if actual_feature_dim != self.config.feature_dim {
            log::warn!(
                "⚠️ Feature dimension mismatch: LSTM outputs {} features, but XGBoost config expects {}",
                actual_feature_dim, self.config.feature_dim
            );
            log::info!(
                "🔧 Auto-adjusting XGBoost feature_dim to match LSTM output: {}",
                actual_feature_dim
            );
            // Update the config to match actual features
            self.config.feature_dim = actual_feature_dim;
        }

        log::info!(
            "📊 XGBoost training data: features={:?}, targets={:?}, num_classes={}, feature_dim={}",
            features_array.dim(),
            targets_array.dim(),
            num_classes,
            actual_feature_dim
        );

        // Create XGBoost DMatrix - Debug feature data first
        let features_slice = features_array.as_slice().unwrap();
        let num_rows = features_array.nrows();
        let num_cols = features_array.ncols();

        // Debug: Check feature statistics
        let feature_sum: f64 = features_slice.iter().map(|&x| x as f64).sum();
        let non_zero_features = features_slice.iter().filter(|&&x| x != 0.0).count();

        log::info!(
            "🔍 DMatrix creation: {} rows × {} cols = {} total elements, data_len={}",
            num_rows,
            num_cols,
            num_rows * num_cols,
            features_slice.len()
        );

        log::info!(
            "🔍 Feature stats: sum={:.6}, non_zero={}/{}, range=[{:.6}, {:.6}]",
            feature_sum,
            non_zero_features,
            features_slice.len(),
            features_slice
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0),
            features_slice
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0)
        );

        if feature_sum.abs() < 1e-10 {
            log::warn!("⚠️ All LSTM features are near zero - this will cause XGBoost to fail");
            log::warn!("🔍 Check LSTM training and feature extraction process");
        }

        // Verify data format is correct for XGBoost
        if features_slice.len() != num_rows * num_cols {
            return Err(VangaError::model(format!(
                "Data size mismatch: expected {} elements ({}×{}), got {}",
                num_rows * num_cols,
                num_rows,
                num_cols,
                features_slice.len()
            )));
        }

        let mut dtrain = DMatrix::from_dense(features_slice, num_rows)
            .map_err(|e| VangaError::model(format!("Failed to create XGBoost DMatrix: {}", e)))?;

        // Set targets
        dtrain
            .set_labels(targets_array.as_slice().unwrap())
            .map_err(|e| VangaError::model(format!("Failed to set XGBoost labels: {}", e)))?;

        // Store validation data for SHAP-based feature importance calculation
        if self.config.save_feature_importance {
            log::debug!(
                "📊 Storing validation data for SHAP-based feature importance calculation..."
            );

            // Create validation subset using configured size
            let total_samples = features_array.nrows();
            let val_size = self
                .config
                .importance_validation_size
                .min(total_samples) // Don't exceed available samples
                .max(1); // At least 1 sample

            log::debug!(
                "🔍 Using {} samples (out of {}) for SHAP validation (configured: {})",
                val_size,
                total_samples,
                self.config.importance_validation_size
            );

            // Take first val_size samples for validation (could be randomized in future)
            let val_features_slice =
                &features_array.as_slice().unwrap()[..val_size * features_array.ncols()];

            match DMatrix::from_dense(val_features_slice, val_size) {
                Ok(val_dmat) => {
                    self.validation_dmat = Some(val_dmat);
                    log::debug!("✅ Validation DMatrix created for SHAP calculation");
                }
                Err(e) => {
                    log::warn!("⚠️ Failed to create validation DMatrix for SHAP: {}", e);
                    log::warn!("📊 Will fallback to placeholder feature importance");
                    self.validation_dmat = None;
                }
            }
        }

        // Configure XGBoost parameters using xgb crate API
        let learning_params = self.build_learning_params(num_classes)?;
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

        // Debug: Test if model can make predictions
        log::info!("🔍 Testing XGBoost model predictions...");
        match booster.predict(&dtrain) {
            Ok(predictions) => {
                let pred_sum: f32 = predictions.iter().sum();
                let pred_range = (
                    predictions
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap_or(&0.0),
                    predictions
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap_or(&0.0),
                );
                log::info!(
                    "✅ XGBoost predictions: {} values, sum={:.6}, range=[{:.6}, {:.6}]",
                    predictions.len(),
                    pred_sum,
                    pred_range.0,
                    pred_range.1
                );

                if pred_sum.abs() < 1e-10 {
                    log::warn!("⚠️ XGBoost predictions are all near zero - model may not be learning properly");
                }
            }
            Err(e) => {
                log::warn!("⚠️ XGBoost prediction test failed: {}", e);
            }
        }

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
    fn build_learning_params(
        &self,
        num_classes: usize,
    ) -> Result<parameters::learning::LearningTaskParameters> {
        use parameters::learning::*;

        // Choose appropriate objective based on number of classes
        let objective = if num_classes <= 2 {
            Objective::BinaryLogistic
        } else {
            // For multi-class classification - try different options available in xgb crate
            Objective::MultiSoftmax(num_classes as u32)
        };

        log::info!("🎯 XGBoost objective set for {} classes", num_classes);

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

    /// Calculate feature importance using SHAP values from predict_contributions()
    ///
    /// This method provides real feature importance based on actual model contributions,
    /// unlike the placeholder uniform importance used as fallback.
    fn calculate_shap_importance(&self, booster: &Booster) -> Result<HashMap<String, f32>> {
        log::debug!("🔍 Calculating SHAP-based feature importance...");

        // Check if we have validation data for SHAP calculation
        let validation_dmat = self.validation_dmat.as_ref().ok_or_else(|| {
            VangaError::model("No validation data available for SHAP calculation")
        })?;

        // Get SHAP contributions [samples, features+1] (includes bias term)
        let (contributions, (num_samples, num_features)) = booster
            .predict_contributions(validation_dmat)
            .map_err(|e| {
                VangaError::model(format!("Failed to calculate SHAP contributions: {}", e))
            })?;

        log::debug!(
            "📊 SHAP contributions shape: {} samples × {} features (including bias)",
            num_samples,
            num_features
        );

        // Debug: Check if contributions are all zeros
        let total_contribution: f32 = contributions.iter().sum();
        let non_zero_count = contributions.iter().filter(|&&x| x != 0.0).count();
        log::debug!(
            "🔍 SHAP Debug: total_contribution={:.6}, non_zero_values={}/{}, contribution_range=[{:.6}, {:.6}]",
            total_contribution,
            non_zero_count,
            contributions.len(),
            contributions.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0),
            contributions.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0)
        );

        if total_contribution.abs() < 1e-10 {
            log::warn!(
                "⚠️ All SHAP contributions are near zero - XGBoost model may not be learning"
            );
            log::warn!("🔍 This could indicate: 1) Model not trained properly, 2) Features are constant/zero, 3) Model convergence issues");
        }

        if num_features == 0 {
            return Err(VangaError::model("No features found in SHAP contributions"));
        }

        // Calculate mean absolute SHAP values per feature (exclude bias term)
        let feature_count = num_features - 1; // Exclude bias term
        let mut feature_importance = vec![0.0f32; feature_count];

        for sample in 0..num_samples {
            for (feature, importance) in feature_importance
                .iter_mut()
                .enumerate()
                .take(feature_count)
            {
                let idx = sample * num_features + feature;
                if idx < contributions.len() {
                    *importance += contributions[idx].abs();
                }
            }
        }

        // Normalize by number of samples to get mean absolute SHAP values
        for importance in &mut feature_importance {
            *importance /= num_samples as f32;
        }

        // Convert to HashMap with feature names
        let mut importance_map = HashMap::new();
        for (i, &importance) in feature_importance.iter().enumerate() {
            importance_map.insert(format!("lstm_feature_{}", i), importance);
        }

        // Normalize to sum to 1.0 for interpretability
        let total: f32 = importance_map.values().sum();
        if total > 0.0 {
            for value in importance_map.values_mut() {
                *value /= total;
            }
            log::info!(
                "✅ SHAP-based feature importance calculated for {} features (normalized sum=1.0)",
                importance_map.len()
            );
            log::info!("🎯 REAL FEATURE IMPORTANCE: Features show varied contributions based on model behavior");
        } else {
            log::warn!("⚠️ All SHAP importance values are zero - model may not be using features effectively");
        }

        let min_importance = importance_map
            .values()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);
        let max_importance = importance_map
            .values()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);

        log::info!(
            "📊 SHAP importance range: {:.6} to {:.6} (variance indicates real feature selection)",
            min_importance,
            max_importance
        );

        // Show top 5 most important features
        let mut sorted_features: Vec<_> = importance_map.iter().collect();
        sorted_features.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        log::info!("🏆 Top 5 most important features (SHAP-based):");
        for (i, (feature, importance)) in sorted_features.iter().take(5).enumerate() {
            log::info!("   {}. {}: {:.6}", i + 1, feature, importance);
        }

        Ok(importance_map)
    }

    /// Extract feature importance from trained booster
    fn extract_feature_importance(&self, booster: &Booster) -> Result<HashMap<String, f32>> {
        log::debug!("🔍 Attempting to extract feature importance from XGBoost booster...");

        // Try SHAP-based importance first (real feature contributions)
        if let Ok(shap_importance) = self.calculate_shap_importance(booster) {
            log::info!("✅ Using SHAP-based feature importance (real model contributions)");
            return Ok(shap_importance);
        }

        // Fallback to placeholder importance with clear warnings
        log::warn!("⚠️ SHAP calculation failed - using PLACEHOLDER feature importance");
        log::warn!(
            "🚨 WARNING: Feature importance scores are NOT REAL - all features show equal weight"
        );
        log::warn!(
            "📊 This indicates XGBoost is not providing meaningful feature selection insights"
        );

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
                log::warn!(
                    "📊 Generated PLACEHOLDER uniform importance for {} named features (all {:.6})",
                    importance_map.len(),
                    uniform_importance
                );
            }
            Ok(feature_names) => {
                log::debug!(
                    "⚠️  Booster returned {} empty feature names, using fallback",
                    feature_names.len()
                );
                // Fallback to generic feature names when booster returns empty list
                let uniform_importance = 1.0 / self.config.feature_dim as f32;
                for i in 0..self.config.feature_dim {
                    importance_map.insert(format!("lstm_feature_{}", i), uniform_importance);
                }
                log::warn!(
                    "📊 Generated PLACEHOLDER importance for {} features (feature_dim={}, all {:.6})",
                    importance_map.len(),
                    self.config.feature_dim,
                    uniform_importance
                );
            }
            Err(e) => {
                log::debug!("⚠️  Failed to get feature names from booster: {}", e);
                // Fallback to generic feature names
                let uniform_importance = 1.0 / self.config.feature_dim as f32;
                for i in 0..self.config.feature_dim {
                    importance_map.insert(format!("lstm_feature_{}", i), uniform_importance);
                }
                log::warn!(
                    "📊 Generated PLACEHOLDER importance for {} features (feature_dim={}, all {:.6})",
                    importance_map.len(),
                    self.config.feature_dim,
                    uniform_importance
                );
            }
        }

        if importance_map.is_empty() {
            return Err(VangaError::model(
                "Failed to generate any feature importance scores",
            ));
        }

        log::warn!(
            "🚨 PLACEHOLDER importance extraction completed with {} features - NOT REAL IMPORTANCE",
            importance_map.len()
        );
        log::warn!(
            "❌ All features have identical importance - XGBoost feature selection is ineffective"
        );
        log::warn!("💡 Consider investigating why SHAP calculation failed for real importance");
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
        assert!(regressor.validation_dmat.is_none());
    }

    #[test]
    fn test_xgboost_config_new_fields() {
        let config = XGBoostConfig::default();

        // Test new fields have correct default values
        assert_eq!(config.importance_method, "shap");
        assert_eq!(config.importance_validation_size, 50);
        assert_eq!(config.importance_type, "gain"); // Legacy field
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
