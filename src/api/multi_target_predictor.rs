// Multi-target prediction API
use crate::config::PredictionConfig;
use crate::data::DataPipeline;
use crate::model::multi_target::MultiTargetLSTMModel;
use crate::output::{OutputFormatter, PredictionResult};
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;

/// Multi-target predictor for cryptocurrency forecasting
pub struct MultiTargetPredictor {
    config: PredictionConfig,
}

impl MultiTargetPredictor {
    /// Create new multi-target predictor
    pub fn new(config: PredictionConfig) -> Self {
        Self { config }
    }

    /// Make predictions using trained multi-target model
    pub async fn predict(&self, model: &MultiTargetLSTMModel) -> Result<MultiTargetPredictions> {
        log::info!(
            "Starting multi-target prediction for symbol: {}",
            self.config.symbol
        );

        // Initialize data pipeline
        let data_pipeline = DataPipeline::new();

        // Load and prepare prediction data
        log::info!(
            "Loading prediction data from: {}",
            self.config.input_path.display()
        );
        let prepared_data = data_pipeline
            .prepare_prediction_data(&self.config.input_path, &self.config)
            .await?;

        // Extract current price from the input data (last close price)
        let current_price = self.extract_current_price(&self.config.input_path).await?;

        log::info!(
            "Prediction data prepared: {} sequences, {} features",
            prepared_data.sequences.shape()[0],
            prepared_data.sequences.shape()[2]
        );

        // Validate input compatibility with model
        if prepared_data.sequences.shape()[2] != model.get_input_size() {
            return Err(VangaError::ModelError(format!(
                "Input feature size mismatch: model expects {}, data has {}",
                model.get_input_size(),
                prepared_data.sequences.shape()[2]
            )));
        }

        // Make predictions using multi-target model
        log::info!(
            "Making predictions using multi-target model with {} targets",
            model.get_num_targets()
        );
        let raw_predictions = model.predict(&prepared_data.sequences).await?;

        // Format predictions with target names
        let predictions = MultiTargetPredictions::new(
            raw_predictions,
            model.get_target_names().to_vec(),
            self.config.symbol.clone(),
            current_price,
        );

        log::info!("✅ Multi-target predictions completed successfully");
        Ok(predictions)
    }

    /// Extract current price from input data (latest close price)
    async fn extract_current_price<P: AsRef<std::path::Path>>(&self, data_path: P) -> Result<f64> {
        use crate::data::DataLoader;

        let loader = DataLoader::new();
        let df = loader.load_csv(data_path).await?;

        // Get the last close price
        let close_series = df
            .column("close")
            .map_err(|e| VangaError::DataError(format!("Failed to get close column: {}", e)))?;

        let close_values = close_series
            .f64()
            .map_err(|e| VangaError::DataError(format!("Failed to convert close to f64: {}", e)))?;

        // Get the last non-null value using to_vec()
        let close_vec = close_values.to_vec();
        let current_price = close_vec
            .iter()
            .rev()
            .filter_map(|v| *v)
            .next()
            .ok_or_else(|| {
                VangaError::DataError("No valid close price found in data".to_string())
            })?;

        log::info!("Extracted current price: ${:.2}", current_price);
        Ok(current_price)
    }
}

/// Multi-target prediction results
#[derive(Debug, Clone)]
pub struct MultiTargetPredictions {
    /// Raw prediction values [samples, targets]
    pub predictions: Array2<f64>,
    /// Target names corresponding to prediction columns
    pub target_names: Vec<String>,
    /// Symbol these predictions are for
    pub symbol: String,
    /// Current price at prediction time
    pub current_price: f64,
}

impl MultiTargetPredictions {
    /// Create new multi-target predictions
    pub fn new(
        predictions: Array2<f64>,
        target_names: Vec<String>,
        symbol: String,
        current_price: f64,
    ) -> Self {
        Self {
            predictions,
            target_names,
            symbol,
            current_price,
        }
    }

    /// Get predictions for a specific target
    pub fn get_target_predictions(&self, target_name: &str) -> Option<Vec<f64>> {
        self.target_names
            .iter()
            .position(|name| name == target_name)
            .map(|target_idx| self.predictions.column(target_idx).to_vec())
    }

    /// Get all target names
    pub fn get_target_names(&self) -> &[String] {
        &self.target_names
    }

    /// Get number of samples
    pub fn num_samples(&self) -> usize {
        self.predictions.shape()[0]
    }

    /// Get number of targets
    pub fn num_targets(&self) -> usize {
        self.predictions.shape()[1]
    }

    /// Get prediction for specific sample and target
    pub fn get_prediction(&self, sample_idx: usize, target_name: &str) -> Option<f64> {
        if let Some(target_idx) = self
            .target_names
            .iter()
            .position(|name| name == target_name)
        {
            if sample_idx < self.predictions.shape()[0] {
                Some(self.predictions[[sample_idx, target_idx]])
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Convert raw predictions to structured format using OutputFormatter
    pub async fn to_structured_predictions(
        &self,
        config: &PredictionConfig,
    ) -> Result<Vec<PredictionResult>> {
        log::info!("Converting raw predictions to structured format");

        // Create output formatter with config
        let formatter = OutputFormatter::new(config.output_config.clone());

        // Extract horizon from config or use default
        let horizon = config.horizon.as_deref().unwrap_or("1h");

        // Format predictions using existing formatter
        let structured_predictions = formatter.format_predictions(
            &self.predictions,
            &self.symbol,
            horizon,
            self.current_price,
            None, // PreparedTargets not available in prediction context
        )?;

        log::info!(
            "✅ Successfully converted {} raw predictions to structured format",
            structured_predictions.len()
        );

        Ok(structured_predictions)
    }

    /// Get all predictions for a specific sample
    pub fn get_sample_predictions(&self, sample_idx: usize) -> Option<Vec<(String, f64)>> {
        if sample_idx < self.predictions.shape()[0] {
            Some(
                self.target_names
                    .iter()
                    .enumerate()
                    .map(|(target_idx, name)| {
                        (name.clone(), self.predictions[[sample_idx, target_idx]])
                    })
                    .collect(),
            )
        } else {
            None
        }
    }

    /// Format predictions as human-readable string
    pub fn format_predictions(&self, sample_idx: usize) -> String {
        if let Some(sample_preds) = self.get_sample_predictions(sample_idx) {
            let mut result = format!("Predictions for {} (sample {}):\n", self.symbol, sample_idx);
            for (target_name, value) in sample_preds {
                result.push_str(&format!("  {}: {:.4}\n", target_name, value));
            }
            result
        } else {
            format!("Invalid sample index: {}", sample_idx)
        }
    }
}

/// High-level prediction function for multi-target models
pub async fn predict_multi_target(
    config: PredictionConfig,
    model: &MultiTargetLSTMModel,
) -> Result<MultiTargetPredictions> {
    let predictor = MultiTargetPredictor::new(config);
    predictor.predict(model).await
}
