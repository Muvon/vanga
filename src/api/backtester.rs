//! Backtesting functionality for VANGA LSTM models
//!
//! Provides comprehensive backtesting capabilities by reusing existing
//! training, prediction, and metrics infrastructure with minimal code changes.

use crate::api::train_model;
use crate::config::{FeatureConfig, PredictionConfig, TrainingConfig};
use crate::data::loader::DataLoader;
use crate::targets::TargetGenerator;
use crate::utils::error::{Result, VangaError};
use crate::utils::metrics::RegressionMetrics;
use polars::prelude::*;
use std::path::Path;

/// Backtesting configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub symbol: String,
    pub train_split: f64,
    pub data_path: std::path::PathBuf,
}

/// Comprehensive backtesting results
#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub symbol: String,
    pub model_type: String,
    pub train_period: (String, String),
    pub test_period: (String, String),
    pub train_samples: usize,
    pub test_samples: usize,
    pub regression_metrics: RegressionMetrics,
    pub directional_accuracy: f64,
    pub prediction_count: usize,
}

/// Main backtesting orchestrator
pub struct Backtester {
    config: BacktestConfig,
}

impl Backtester {
    /// Create new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Execute comprehensive backtesting workflow
    pub async fn run_backtest(&self) -> Result<BacktestResults> {
        log::info!("🔄 Starting backtesting for symbol: {}", self.config.symbol);

        // Step 1: Load and validate data
        let data_loader = DataLoader::new();
        let full_df = data_loader.load_csv(&self.config.data_path).await?;

        log::info!("📊 Loaded {} samples for backtesting", full_df.height());

        if full_df.height() < 100 {
            return Err(VangaError::DataError(format!(
                "Insufficient data for backtesting (minimum 100 samples required, got {})",
                full_df.height()
            )));
        }

        // Step 2: Split data chronologically to prevent data leakage
        let (train_df, test_df) =
            data_loader.split_chronological(&full_df, self.config.train_split)?;

        // Extract time periods for reporting
        let train_period = self.extract_time_period(&train_df)?;
        let test_period = self.extract_time_period(&test_df)?;

        log::info!("📈 Train period: {} to {}", train_period.0, train_period.1);
        log::info!("📉 Test period: {} to {}", test_period.0, test_period.1);

        // Step 3: Create temporary files for training and prediction
        let temp_dir = std::env::temp_dir();
        let train_path = temp_dir.join(format!(
            "vanga_train_{}_{}.csv",
            self.config.symbol,
            std::process::id()
        ));
        let test_path = temp_dir.join(format!(
            "vanga_test_{}_{}.csv",
            self.config.symbol,
            std::process::id()
        ));

        // Write temporary CSV files
        self.write_dataframe_to_csv(&train_df, &train_path).await?;
        self.write_dataframe_to_csv(&test_df, &test_path).await?;

        // Step 4: Train model on training data
        log::info!("🚀 Training model on {} samples", train_df.height());
        let training_config = self.create_training_config(&train_path)?;
        let trained_model = train_model(training_config).await?;

        // Step 5: Generate predictions on test data
        log::info!("🔮 Generating predictions on {} samples", test_df.height());
        let prediction_config = self.create_prediction_config(&test_path)?;
        let predictions = {
            let predictor = crate::api::Predictor::new(prediction_config);
            predictor.predict(crate::api::ModelWrapper::MultiTarget(&trained_model)).await?
        };

        // Step 6: Generate actual targets for test data to calculate metrics
        log::info!("🎯 Generating targets for test data");
        let target_generator = TargetGenerator::with_defaults();
        let actual_targets = match target_generator.generate_all_targets(&test_df, None).await {
            Ok(targets) => targets,
            Err(e) => {
                log::error!("Failed to generate targets for test data: {}", e);
                return Err(VangaError::DataError(format!(
                    "Target generation failed: {}. This usually means insufficient data for sequence generation. Try using more data (minimum 500+ samples recommended).",
                    e
                )));
            }
        };

        // Step 7: Calculate comprehensive metrics
        let metrics = self
            .calculate_metrics(&predictions, &actual_targets)
            .await?;

        // Step 8: Clean up temporary files
        if let Err(e) = std::fs::remove_file(&train_path) {
            log::warn!(
                "Failed to remove temporary train file {}: {}",
                train_path.display(),
                e
            );
        }
        if let Err(e) = std::fs::remove_file(&test_path) {
            log::warn!(
                "Failed to remove temporary test file {}: {}",
                test_path.display(),
                e
            );
        }

        let results = BacktestResults {
            symbol: self.config.symbol.clone(),
            model_type: "MultiTargetLSTM".to_string(),
            train_period,
            test_period,
            train_samples: train_df.height(),
            test_samples: test_df.height(),
            regression_metrics: metrics,
            directional_accuracy: self
                .calculate_directional_accuracy(&predictions, &actual_targets)
                .await?,
            prediction_count: predictions.len(),
        };

        log::info!("✅ Backtesting completed for {}", self.config.symbol);
        Ok(results)
    }

    /// Extract time period from DataFrame for reporting
    fn extract_time_period(&self, df: &DataFrame) -> Result<(String, String)> {
        // Try to find timestamp column
        let timestamp_col = df
            .get_columns()
            .iter()
            .find(|col| {
                let name = col.name().to_lowercase();
                name.contains("timestamp") || name.contains("time") || name.contains("date")
            })
            .ok_or_else(|| VangaError::DataError("No timestamp column found".to_string()))?;

        // Get first and last timestamps
        let first_ts = timestamp_col
            .get(0)
            .map_err(|e| VangaError::DataError(format!("Failed to get first timestamp: {}", e)))?;
        let last_ts = timestamp_col
            .get(df.height() - 1)
            .map_err(|e| VangaError::DataError(format!("Failed to get last timestamp: {}", e)))?;

        // Convert to string representation
        let start_time = format!("{}", first_ts);
        let end_time = format!("{}", last_ts);

        Ok((start_time, end_time))
    }

    /// Write DataFrame to CSV file
    async fn write_dataframe_to_csv(&self, df: &DataFrame, path: &Path) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path).map_err(|e| {
            VangaError::DataError(format!("Failed to create file {}: {}", path.display(), e))
        })?;

        // Get column names for header
        let columns: Vec<&str> = df.get_column_names();
        writeln!(file, "{}", columns.join(","))
            .map_err(|e| VangaError::DataError(format!("Failed to write CSV header: {}", e)))?;

        // Write data rows
        for i in 0..df.height() {
            let row_values: Result<Vec<String>> = columns
                .iter()
                .map(|col_name| {
                    let column = df.column(col_name).map_err(|e| {
                        VangaError::DataError(format!("Failed to get column {}: {}", col_name, e))
                    })?;
                    let value = column.get(i).map_err(|e| {
                        VangaError::DataError(format!("Failed to get value at row {}: {}", i, e))
                    })?;
                    Ok(format!("{}", value))
                })
                .collect();

            let row_values = row_values?;
            writeln!(file, "{}", row_values.join(","))
                .map_err(|e| VangaError::DataError(format!("Failed to write CSV row: {}", e)))?;
        }

        file.flush()
            .map_err(|e| VangaError::DataError(format!("Failed to flush CSV file: {}", e)))?;

        Ok(())
    }

    /// Create training configuration for backtesting
    fn create_training_config(&self, train_path: &Path) -> Result<TrainingConfig> {
        use crate::config::ModelConfig;

        // Use default model configuration for backtesting - it's already optimized
        let model_config = ModelConfig::default();

        Ok(TrainingConfig {
            symbol: self.config.symbol.clone(),
            data_path: train_path.to_path_buf(),
            model: model_config,
            horizons: vec!["1h".to_string()], // Single horizon for backtesting
            fresh_training: true,             // Always start fresh for backtesting
            continue_training: false,
            features: FeatureConfig::default(),
            training: crate::config::training::TrainingParams {
                device: crate::config::training::DeviceConfig::Auto,
                epochs: crate::config::training::EpochConfig::Fixed(5), // Very short training for testing
                batch_size: crate::config::training::BatchSizeConfig::Fixed(16), // Smaller batch size
                learning_rate: crate::config::training::LearningRateConfig::Fixed(0.001),
                optimizer: crate::config::training::OptimizerType::AdamW {
                    weight_decay: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                },
                warmup_epochs: 0, // No warmup for short backtesting
                learning_schedule: None,
                validation_split: 0.2,
                test_split: 0.0, // No separate test split since we handle this in backtesting
                early_stopping: crate::config::training::EarlyStoppingConfig {
                    patience: 3, // Quick early stopping
                    min_delta: 0.0001,
                },
                gradient_clip: Some(1.0),
                print_every: 1, // Add missing print_every field
                class_weight_strategy: crate::config::training::ClassWeightStrategy::Global, // Use global for backtesting
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        })
    }

    /// Create prediction configuration for backtesting
    fn create_prediction_config(&self, test_path: &Path) -> Result<PredictionConfig> {
        Ok(PredictionConfig {
            symbols: vec![self.config.symbol.clone()],
            input_path: test_path.to_path_buf(),
            horizon: Some("1h".to_string()),
            ..Default::default()
        })
    }

    /// Calculate regression metrics from predictions vs actual targets
    async fn calculate_metrics(
        &self,
        predictions: &[crate::output::PredictionResult], // Updated to use unified predictor output
        actual_targets: &crate::targets::PreparedTargets,
    ) -> Result<RegressionMetrics> {
        // Calculate metrics from unified predictor output
        if predictions.is_empty() {
            return Err(VangaError::DataError(
                "No predictions available for metrics calculation".to_string(),
            ));
        }

        // Extract predicted values from the first prediction's price levels
        // For regression metrics, we'll use the most likely price level as the prediction
        let predicted_values: Vec<f64> = predictions
            .iter()
            .filter_map(|pred| {
                pred.price_levels.as_ref().and_then(|pl| {
                    // Find the bin with highest probability and use its midpoint
                    let mut max_prob = 0.0;
                    let mut best_price = 0.0;
                    
                    for bin in pl.bins.values() {
                        if bin.probability > max_prob {
                            max_prob = bin.probability;
                            // Use midpoint of price range
                            best_price = (bin.price[0] + bin.price[1]) / 2.0;
                        }
                    }
                    
                    if max_prob > 0.0 { Some(best_price) } else { None }
                })
            })
            .collect();

        if predicted_values.is_empty() {
            return Err(VangaError::DataError(
                "No valid price level predictions found for metrics calculation".to_string(),
            ));
        }

        // Get actual values for first target (price_level_1h)
        let actual_values = actual_targets.price_levels.get("1h").ok_or_else(|| {
            VangaError::DataError("No actual price_level targets found".to_string())
        })?;

        // Convert actual class indices to price values using current price
        let current_price = predictions[0].current_price;
        let actual_prices: Vec<f64> = actual_values
            .iter()
            .map(|&class_idx| {
                // Convert class index to approximate price change
                // Assuming 5-class system: 0=strong_down, 1=moderate_down, 2=neutral, 3=moderate_up, 4=strong_up
                let price_change_percent = match class_idx {
                    0 => -3.5, // strong_down
                    1 => -1.5, // moderate_down
                    2 => 0.0,  // neutral
                    3 => 1.5,  // moderate_up
                    4 => 3.5,  // strong_up
                    _ => 0.0,  // fallback
                };
                current_price * (1.0 + price_change_percent / 100.0)
            })
            .collect();

        // Align the data lengths (take minimum)
        let min_len = predicted_values.len().min(actual_prices.len());
        let predicted: Vec<f64> = predicted_values.into_iter().take(min_len).collect();
        let actual: Vec<f64> = actual_prices.into_iter().take(min_len).collect();

        if predicted.is_empty() || actual.is_empty() {
            return Err(VangaError::DataError(
                "No valid data for metrics calculation".to_string(),
            ));
        }

        // Calculate metrics
        let mse = predicted
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predicted.len() as f64;

        let rmse = mse.sqrt();

        let mae = predicted
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).abs())
            .sum::<f64>()
            / predicted.len() as f64;

        let actual_mean = actual.iter().sum::<f64>() / actual.len() as f64;
        let ss_tot = actual
            .iter()
            .map(|a| (a - actual_mean).powi(2))
            .sum::<f64>();
        let ss_res = predicted
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (a - p).powi(2))
            .sum::<f64>();
        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        let mape = predicted
            .iter()
            .zip(actual.iter())
            .filter(|(_, a)| **a != 0.0)
            .map(|(p, a)| ((a - p) / a).abs() * 100.0)
            .sum::<f64>()
            / predicted.len() as f64;

        Ok(RegressionMetrics {
            mse,
            rmse,
            mae,
            r_squared,
            mape,
        })
    }

    /// Calculate directional accuracy (percentage of correct direction predictions)
    async fn calculate_directional_accuracy(
        &self,
        predictions: &[crate::output::PredictionResult], // Updated to use unified predictor output
        actual_targets: &crate::targets::PreparedTargets,
    ) -> Result<f64> {
        // Extract direction predictions from unified predictor output
        if predictions.is_empty() {
            return Ok(0.0);
        }

        // Get predicted directions from the first prediction's direction field
        let predicted_directions: Vec<i32> = predictions
            .iter()
            .filter_map(|pred| {
                pred.direction.as_ref().map(|dir| {
                    // Convert direction prediction to class index
                    // Find the most likely direction
                    let directions = [
                        ("dump", dir.dump_probability),
                        ("down", dir.down_probability),
                        ("sideways", dir.sideways_probability),
                        ("up", dir.up_probability),
                        ("pump", dir.pump_probability),
                    ];
                    
                    let (max_idx, _) = directions
                        .iter()
                        .enumerate()
                        .max_by(|(_, (_, a)), (_, (_, b))| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or((2, &("sideways", 0.0))); // Default to sideways
                    
                    max_idx as i32
                })
            })
            .collect();

        if predicted_directions.is_empty() {
            return Ok(0.0);
        }

        // Get actual direction values
        let actual_directions = actual_targets.directions.get("1h").ok_or_else(|| {
            VangaError::DataError("No actual direction targets found".to_string())
        })?;

        // Align data lengths
        let min_len = predicted_directions.len().min(actual_directions.len());

        if min_len == 0 {
            return Ok(0.0);
        }

        // Calculate directional accuracy
        let correct_predictions = predicted_directions
            .iter()
            .take(min_len)
            .zip(actual_directions.iter().take(min_len))
            .filter(|(pred, actual)| pred == actual)
            .count();

        Ok(correct_predictions as f64 / min_len as f64)
    }
}

/// High-level backtesting function for easy integration
pub async fn run_backtest(
    symbol: &str,
    data_path: &Path,
    train_split: f64,
) -> Result<BacktestResults> {
    let config = BacktestConfig {
        symbol: symbol.to_string(),
        train_split,
        data_path: data_path.to_path_buf(),
    };

    let backtester = Backtester::new(config);
    backtester.run_backtest().await
}

/// Batch backtesting for multiple symbols
pub async fn run_batch_backtest(
    symbols: &[String],
    data_dir: &Path,
    train_split: f64,
) -> Result<Vec<BacktestResults>> {
    let mut results = Vec::new();

    for symbol in symbols {
        let data_path = data_dir.join(format!("{}.csv", symbol.to_lowercase()));

        if !data_path.exists() {
            log::warn!("Data file not found for {}: {:?}", symbol, data_path);
            continue;
        }

        match run_backtest(symbol, &data_path, train_split).await {
            Ok(result) => {
                log::info!("✅ Backtesting completed for {}", symbol);
                results.push(result);
            }
            Err(e) => {
                log::error!("❌ Backtesting failed for {}: {}", symbol, e);
            }
        }
    }

    Ok(results)
}
