//! Streaming prediction engine for real-time cryptocurrency forecasting
//!
//! This module provides real-time prediction capabilities for cryptocurrency markets,
//! integrating with the trained VANGA LSTM models for live forecasting.

use crate::api::multi_target_predictor::MultiTargetPredictor;
use crate::config::PredictionConfig;
use crate::data::structures::MarketDataRow;
use crate::model::multi_target::MultiTargetLSTMModel;
use crate::output::PredictionResult;
use crate::realtime::{CsvStreamer, FileWatcher, OutputFormat, RealtimeConfig};
use crate::utils::error::{Result, VangaError};
use crate::utils::model_path::get_model_path;
use std::collections::VecDeque;
use tokio::select;
use tokio::time::{interval, Duration, Instant};

/// Real-time streaming predictor with feature buffer management
///
/// This struct manages the complete real-time prediction pipeline including:
/// - File watching for new data
/// - Incremental CSV parsing
/// - Feature buffer management with sliding window
/// - Integration with existing ML prediction pipeline
/// - Configurable output formatting
pub struct StreamingPredictor {
    config: RealtimeConfig,
    model: MultiTargetLSTMModel,
    feature_buffer: VecDeque<MarketDataRow>,
    watcher: FileWatcher,
    streamer: CsvStreamer,
    last_prediction_time: Option<Instant>,
    prediction_count: u64,
    error_count: u64,
}

impl StreamingPredictor {
    /// Create a new streaming predictor
    ///
    /// # Arguments
    /// * `config` - Real-time configuration parameters
    ///
    /// # Returns
    /// * `Result<Self>` - New StreamingPredictor instance or error
    ///
    /// # Example
    /// ```rust,no_run
    /// use vanga::realtime::{StreamingPredictor, RealtimeConfig};
    /// use std::path::PathBuf;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = RealtimeConfig {
    ///     file_path: PathBuf::from("data/live.csv"),
    ///     symbol: "BTCUSDT".to_string(),
    ///     ..Default::default()
    /// };
    ///
    /// let predictor = StreamingPredictor::new(config).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(config: RealtimeConfig) -> Result<Self> {
        log::info!(
            "Initializing streaming predictor for symbol: {}",
            config.symbol
        );

        // Load the trained model for the symbol
        let model_path = get_model_path(&config.symbol);
        log::info!("Loading model from: {}", model_path.display());

        let model = MultiTargetLSTMModel::load(&model_path).map_err(|e| {
            VangaError::ModelError(format!(
                "Failed to load model for symbol {}: {}. Please train a model first using 'vanga train --symbol {}'",
                config.symbol, e, config.symbol
            ))
        })?;

        log::info!("✅ Model loaded successfully for symbol: {}", config.symbol);

        // Create predictor configuration for realtime use
        let _prediction_config = PredictionConfig {
            symbols: vec![config.symbol.clone()],
            input_path: config.file_path.clone(), // Will be updated per prediction
            horizon: Some("1h".to_string()),
            ..Default::default()
        };

        // Create file watcher
        let watcher = FileWatcher::new(&config.file_path).map_err(|e| {
            VangaError::IoError(format!(
                "Failed to create file watcher for {}: {}",
                config.file_path.display(),
                e
            ))
        })?;

        // Create CSV streamer
        let streamer = CsvStreamer::new(config.file_path.clone());

        // Initialize feature buffer with configured capacity
        let feature_buffer = VecDeque::with_capacity(config.buffer_size);

        log::info!("Streaming predictor initialized successfully");
        log::info!(
            "Buffer size: {}, Feature window: {}",
            config.buffer_size,
            config.feature_window
        );

        Ok(Self {
            config,
            model,
            feature_buffer,
            watcher,
            streamer,
            last_prediction_time: None,
            prediction_count: 0,
            error_count: 0,
        })
    }

    /// Run the main prediction loop
    ///
    /// This method starts the real-time prediction loop that:
    /// 1. Monitors file changes using the file watcher
    /// 2. Reads new CSV data when changes are detected
    /// 3. Maintains a sliding window buffer of market data
    /// 4. Generates predictions when sufficient data is available
    /// 5. Outputs predictions in the configured format
    ///
    /// The loop continues until interrupted (Ctrl+C) or an unrecoverable error occurs.
    ///
    /// # Returns
    /// * `Result<()>` - Success or error result
    pub async fn run(&mut self) -> Result<()> {
        log::info!("Starting real-time prediction loop");
        log::info!("Monitoring file: {}", self.config.file_path.display());
        log::info!("Output format: {:?}", self.config.output_format);

        // Initial file read to populate buffer
        self.initial_file_read().await?;

        // Set up periodic health checks
        let mut health_check_interval = interval(self.config.poll_interval);
        let mut stats_interval = interval(Duration::from_secs(60)); // Stats every minute

        loop {
            select! {
                // File change event detected
                event = self.watcher.next_event() => {
                    if let Some(event) = event {
                        if self.config.debug {
                            log::debug!("File event received: {:?}", event);
                        }

                        if let Err(e) = self.process_file_changes().await {
                            log::error!("Error processing file changes: {}", e);
                            self.error_count += 1;

                            // Continue processing unless too many errors
                        if self.error_count > 10 {
                            return Err(VangaError::TrainingError(
                                    "Too many consecutive errors, stopping prediction loop".to_string()
                                ));
                            }
                        }
                    }
                }

                // Periodic health check
                _ = health_check_interval.tick() => {
                    if let Err(e) = self.health_check().await {
                        log::warn!("Health check failed: {}", e);
                    }
                }

                // Periodic statistics logging
                _ = stats_interval.tick() => {
                    self.log_statistics();
                }

                // Graceful shutdown on Ctrl+C
                _ = tokio::signal::ctrl_c() => {
                    log::info!("Received shutdown signal, stopping real-time prediction");
                    self.log_final_statistics();
                    break;
                }
            }
        }

        Ok(())
    }

    /// Perform initial file read to populate the feature buffer
    async fn initial_file_read(&mut self) -> Result<()> {
        log::info!("Performing initial file read to populate buffer");

        let initial_rows = self.streamer.read_new_lines().await?;
        log::info!("Read {} initial rows from file", initial_rows.len());

        for row in initial_rows {
            self.add_data_point(row).await?;
        }

        log::info!(
            "Initial buffer populated with {} data points",
            self.feature_buffer.len()
        );

        // Generate initial prediction if we have enough data
        if self.has_enough_data() {
            log::info!("Sufficient data available, generating initial prediction");
            match self.generate_prediction().await {
                Ok(prediction) => {
                    self.output_prediction(prediction).await?;
                }
                Err(e) => {
                    log::warn!("Failed to generate initial prediction: {}", e);
                }
            }
        } else {
            log::info!(
                "Need {} more data points before predictions can be generated",
                self.config.feature_window - self.feature_buffer.len()
            );
        }

        Ok(())
    }

    /// Process file changes by reading new lines and generating predictions
    async fn process_file_changes(&mut self) -> Result<()> {
        let start_time = Instant::now();
        let new_rows = self.streamer.read_new_lines().await?;

        if new_rows.is_empty() {
            if self.config.debug {
                log::debug!("No new data found in file");
            }
            return Ok(());
        }

        log::info!("Processing {} new data points", new_rows.len());

        for row in new_rows {
            self.add_data_point(row).await?;

            // Generate prediction for each new data point if we have enough data
            if self.has_enough_data() {
                match self.generate_prediction().await {
                    Ok(prediction) => {
                        self.output_prediction(prediction).await?;
                        self.prediction_count += 1;
                        self.last_prediction_time = Some(Instant::now());
                    }
                    Err(e) => {
                        log::error!("Failed to generate prediction: {}", e);
                        self.error_count += 1;
                    }
                }
            }
        }

        let processing_time = start_time.elapsed();
        if self.config.debug {
            log::debug!("File processing completed in {:?}", processing_time);
        }

        Ok(())
    }

    /// Add a new data point to the feature buffer
    async fn add_data_point(&mut self, row: MarketDataRow) -> Result<()> {
        // Validate data point
        if row.open <= 0.0 || row.high <= 0.0 || row.low <= 0.0 || row.close <= 0.0 {
            return Err(VangaError::DataError(
                "Invalid market data: prices must be positive".to_string(),
            ));
        }

        if row.high < row.low {
            return Err(VangaError::DataError(
                "Invalid market data: high price less than low price".to_string(),
            ));
        }

        // Add to buffer
        self.feature_buffer.push_back(row);

        // Maintain buffer size by removing oldest data
        while self.feature_buffer.len() > self.config.buffer_size {
            self.feature_buffer.pop_front();
        }

        if self.config.debug {
            log::debug!(
                "Added data point to buffer (size: {}/{})",
                self.feature_buffer.len(),
                self.config.buffer_size
            );
        }

        Ok(())
    }

    /// Check if we have enough data points for prediction
    fn has_enough_data(&self) -> bool {
        self.feature_buffer.len() >= self.config.feature_window
    }

    /// Generate a prediction using the current feature buffer
    async fn generate_prediction(&mut self) -> Result<PredictionResult> {
        if !self.has_enough_data() {
            return Err(VangaError::TrainingError(format!(
                "Insufficient data for prediction: {} < {}",
                self.feature_buffer.len(),
                self.config.feature_window
            )));
        }

        // Convert buffer to the format expected by the predictor
        let data: Vec<MarketDataRow> = self
            .feature_buffer
            .iter()
            .rev() // Most recent data first
            .take(self.config.feature_window)
            .rev() // Restore chronological order
            .cloned()
            .collect();

        if self.config.debug {
            log::debug!("Generating prediction with {} data points", data.len());
        }

        // Use real ML prediction pipeline
        log::debug!("Converting buffer to CSV format for ML prediction");
        let temp_csv_path = self.create_temp_csv_from_buffer(&data).await?;

        // Create temporary prediction config with the CSV data
        let temp_config = PredictionConfig {
            symbols: vec![self.config.symbol.clone()],
            input_path: temp_csv_path.clone(),
            horizon: Some("1h".to_string()),
            ..Default::default()
        };
        let temp_predictor = MultiTargetPredictor::new(temp_config.clone());

        // Generate real ML predictions
        log::debug!("Generating ML prediction using trained model");
        let ml_predictions = temp_predictor.predict(&self.model).await?;
        let structured_predictions = ml_predictions
            .to_structured_predictions(&temp_config, &self.model)
            .await?;

        // Convert to realtime prediction format
        let prediction = self.convert_to_prediction_result(&structured_predictions)?;

        // Cleanup temporary file
        if let Err(e) = std::fs::remove_file(&temp_csv_path) {
            log::warn!("Failed to cleanup temporary CSV file: {}", e);
        }

        // Update statistics
        self.prediction_count += 1;

        Ok(prediction)
    }

    /// Output prediction in the configured format
    async fn output_prediction(&self, prediction: PredictionResult) -> Result<()> {
        match self.config.output_format {
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(&prediction).map_err(|e| {
                    VangaError::SerializationError(format!("JSON serialization failed: {}", e))
                })?;
                println!("{}", json);
            }
            OutputFormat::Csv => {
                // CSV format: timestamp,symbol,direction,probability,volatility,confidence
                println!(
                    "{},{},{},{:.4},{},{:.4}",
                    prediction.timestamp,
                    self.config.symbol,
                    prediction
                        .direction
                        .as_ref()
                        .map(|d| &d.prediction)
                        .unwrap_or(&"UNKNOWN".to_string()),
                    prediction
                        .direction
                        .as_ref()
                        .map(|d| d.up_probability.max(d.down_probability))
                        .unwrap_or(0.0),
                    prediction
                        .volatility
                        .as_ref()
                        .map(|v| &v.regime)
                        .unwrap_or(&"UNKNOWN".to_string()),
                    prediction
                        .direction
                        .as_ref()
                        .map(|d| d.confidence)
                        .unwrap_or(0.0)
                );
            }
            OutputFormat::Pretty => {
                let (direction_str, direction_emoji) = if prediction
                    .direction
                    .as_ref()
                    .map(|d| d.up_probability > 0.5)
                    .unwrap_or(false)
                {
                    ("LONG", "🚀")
                } else {
                    ("SHORT", "📉")
                };

                let (volatility_str, volatility_emoji) = if prediction
                    .volatility
                    .as_ref()
                    .map(|v| v.high_probability + v.very_high_probability > 0.5)
                    .unwrap_or(false)
                {
                    ("HIGH", "⚡")
                } else if prediction
                    .volatility
                    .as_ref()
                    .map(|v| {
                        v.medium_probability + v.high_probability + v.very_high_probability > 0.5
                    })
                    .unwrap_or(false)
                {
                    ("MEDIUM", "🌊")
                } else {
                    ("LOW", "😴")
                };

                println!(
                    "🔮 {} Prediction: {} {} ({:.1}%) | Volatility: {} {} | Orders: {} levels",
                    self.config.symbol,
                    direction_emoji,
                    direction_str,
                    prediction
                        .direction
                        .as_ref()
                        .map(|d| d.up_probability.max(d.down_probability))
                        .unwrap_or(0.0)
                        * 100.0,
                    volatility_emoji,
                    volatility_str,
                    prediction.orders.entry_levels.len()
                );
            }
        }

        // Flush stdout to ensure immediate output
        use std::io::{self, Write};
        io::stdout()
            .flush()
            .map_err(|e| VangaError::IoError(format!("Failed to flush output: {}", e)))?;

        Ok(())
    }

    /// Perform health check on the prediction system
    async fn health_check(&self) -> Result<()> {
        // Check if input file still exists and is readable
        if !self.config.file_path.exists() {
            return Err(VangaError::IoError(format!(
                "Input file no longer exists: {}",
                self.config.file_path.display()
            )));
        }

        // Check if file is readable
        if let Err(e) = std::fs::File::open(&self.config.file_path) {
            return Err(VangaError::IoError(format!(
                "Cannot read input file {}: {}",
                self.config.file_path.display(),
                e
            )));
        }

        // Check buffer health
        if self.feature_buffer.len() > self.config.buffer_size {
            log::warn!(
                "Feature buffer size exceeded: {} > {}",
                self.feature_buffer.len(),
                self.config.buffer_size
            );
        }

        // Check error rate
        if self.error_count > 0 && self.prediction_count > 0 {
            let error_rate =
                self.error_count as f64 / (self.prediction_count + self.error_count) as f64;
            if error_rate > 0.1 {
                log::warn!("High error rate detected: {:.1}%", error_rate * 100.0);
            }
        }

        if self.config.debug {
            log::debug!(
                "Health check passed - Buffer: {}/{}, Predictions: {}, Errors: {}",
                self.feature_buffer.len(),
                self.config.buffer_size,
                self.prediction_count,
                self.error_count
            );
        }

        Ok(())
    }

    /// Log current statistics
    fn log_statistics(&self) {
        let buffer_usage =
            (self.feature_buffer.len() as f64 / self.config.buffer_size as f64) * 100.0;
        let error_rate = if self.prediction_count + self.error_count > 0 {
            (self.error_count as f64 / (self.prediction_count + self.error_count) as f64) * 100.0
        } else {
            0.0
        };

        log::info!(
            "📊 Stats: Predictions: {}, Errors: {}, Error Rate: {:.1}%, Buffer: {:.1}%",
            self.prediction_count,
            self.error_count,
            error_rate,
            buffer_usage
        );

        if let Some(last_time) = self.last_prediction_time {
            let time_since_last = last_time.elapsed();
            if time_since_last > Duration::from_secs(300) {
                // 5 minutes
                log::warn!("No predictions generated in the last {:?}", time_since_last);
            }
        }
    }

    /// Log final statistics before shutdown
    fn log_final_statistics(&self) {
        log::info!("🏁 Final Statistics:");
        log::info!("   Total Predictions: {}", self.prediction_count);
        log::info!("   Total Errors: {}", self.error_count);
        log::info!(
            "   Final Buffer Size: {}/{}",
            self.feature_buffer.len(),
            self.config.buffer_size
        );
        log::info!("   CSV Lines Processed: {}", self.streamer.get_line_count());
        log::info!("   File Position: {} bytes", self.streamer.get_position());

        if self.prediction_count > 0 {
            let success_rate = (self.prediction_count as f64
                / (self.prediction_count + self.error_count) as f64)
                * 100.0;
            log::info!("   Success Rate: {:.1}%", success_rate);
        }
    }

    /// Get current buffer statistics
    pub fn get_buffer_stats(&self) -> (usize, usize, f64) {
        let current_size = self.feature_buffer.len();
        let max_size = self.config.buffer_size;
        let usage_percent = (current_size as f64 / max_size as f64) * 100.0;
        (current_size, max_size, usage_percent)
    }

    /// Get prediction statistics
    pub fn get_prediction_stats(&self) -> (u64, u64, f64) {
        let error_rate = if self.prediction_count + self.error_count > 0 {
            (self.error_count as f64 / (self.prediction_count + self.error_count) as f64) * 100.0
        } else {
            0.0
        };
        (self.prediction_count, self.error_count, error_rate)
    }

    /// Convert MarketDataRow buffer to temporary CSV file for ML prediction
    async fn create_temp_csv_from_buffer(
        &self,
        data: &[MarketDataRow],
    ) -> Result<std::path::PathBuf> {
        use std::io::Write;

        // Create temporary file
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!(
            "vanga_realtime_{}_{}.csv",
            self.config.symbol,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));

        let mut file = std::fs::File::create(&temp_file)
            .map_err(|e| VangaError::IoError(format!("Failed to create temp CSV: {}", e)))?;

        // Write CSV header
        writeln!(file, "timestamp,open,high,low,close,volume")
            .map_err(|e| VangaError::IoError(format!("Failed to write CSV header: {}", e)))?;

        // Write data rows
        for row in data {
            writeln!(
                file,
                "{},{},{},{},{},{}",
                row.timestamp, row.open, row.high, row.low, row.close, row.volume
            )
            .map_err(|e| VangaError::IoError(format!("Failed to write CSV row: {}", e)))?;
        }

        file.flush()
            .map_err(|e| VangaError::IoError(format!("Failed to flush CSV file: {}", e)))?;

        log::debug!(
            "Created temporary CSV with {} rows: {}",
            data.len(),
            temp_file.display()
        );
        Ok(temp_file)
    }

    /// Convert structured ML predictions to realtime PredictionResult format
    fn convert_to_prediction_result(
        &self,
        structured_predictions: &[crate::output::PredictionResult],
    ) -> Result<PredictionResult> {
        use crate::output::structures::{DirectionPrediction, VolatilityPrediction};

        // Use the first (most recent) prediction
        let ml_prediction = structured_predictions.first().ok_or_else(|| {
            VangaError::DataError("No predictions generated by ML model".to_string())
        })?;

        // Extract direction prediction
        let direction = if let Some(dir) = &ml_prediction.direction {
            DirectionPrediction {
                up_probability: dir.up_probability,
                down_probability: dir.down_probability,
                prediction: dir.prediction.clone(),
                confidence: dir.confidence,
            }
        } else {
            DirectionPrediction {
                up_probability: 0.5,
                down_probability: 0.5,
                prediction: "SIDEWAYS".to_string(),
                confidence: 0.5,
            }
        };

        // Extract volatility prediction
        let volatility = if let Some(vol) = &ml_prediction.volatility {
            VolatilityPrediction {
                very_low_probability: vol.very_low_probability,
                low_probability: vol.low_probability,
                medium_probability: vol.medium_probability,
                high_probability: vol.high_probability,
                very_high_probability: vol.very_high_probability,
                regime: vol.get_prediction(),
                confidence: vol.get_confidence(),
            }
        } else {
            VolatilityPrediction {
                very_low_probability: 0.2,
                low_probability: 0.2,
                medium_probability: 0.2,
                high_probability: 0.2,
                very_high_probability: 0.2,
                regime: "MEDIUM".to_string(),
                confidence: 0.2,
            }
        };

        // Use the trading orders from ML prediction
        let orders = ml_prediction.orders.clone();

        Ok(PredictionResult {
            symbol: self.config.symbol.clone(),
            timestamp: ml_prediction.timestamp.clone(),
            horizon: ml_prediction.horizon.clone(),
            current_price: ml_prediction.current_price,
            price_levels: ml_prediction.price_levels.clone(),
            direction: Some(direction),
            volatility: Some(volatility),
            orders,
            confidence: ml_prediction.confidence,
            metadata: ml_prediction.metadata.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    async fn create_test_config() -> (RealtimeConfig, NamedTempFile) {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "timestamp,open,high,low,close,volume").unwrap();
        writeln!(
            temp_file,
            "1640995200,47000.5,47100.0,46900.0,47050.0,1234.56"
        )
        .unwrap();
        temp_file.flush().unwrap();

        let config = RealtimeConfig {
            file_path: temp_file.path().to_path_buf(),
            symbol: "BTCUSDT".to_string(),
            buffer_size: 100,
            feature_window: 10,
            output_format: OutputFormat::Json,
            debug: true,
            ..Default::default()
        };

        (config, temp_file)
    }

    #[tokio::test]
    async fn test_streaming_predictor_creation() {
        let (config, _temp_file) = create_test_config().await;

        // Test successful creation with valid config and temp file
        let result = StreamingPredictor::new(config).await;
        // Should succeed with valid config and existing file
        assert!(
            result.is_ok(),
            "StreamingPredictor creation should succeed with valid config"
        );

        if let Ok(predictor) = result {
            assert_eq!(predictor.config.symbol, "BTCUSDT");
            assert_eq!(predictor.config.buffer_size, 100);
            assert_eq!(predictor.prediction_count, 0);
            assert_eq!(predictor.error_count, 0);
        }
    }

    #[tokio::test]
    async fn test_streaming_predictor_creation_with_invalid_file() {
        let config = RealtimeConfig {
            file_path: PathBuf::from("/nonexistent/path/data.csv"),
            symbol: "BTCUSDT".to_string(),
            buffer_size: 100,
            feature_window: 10,
            output_format: OutputFormat::Json,
            debug: true,
            ..Default::default()
        };

        // This should fail due to invalid file path
        let result = StreamingPredictor::new(config).await;
        assert!(
            result.is_err(),
            "StreamingPredictor should fail with invalid file path"
        );
    }

    #[tokio::test]
    async fn test_add_data_point() {
        let (config, _temp_file) = create_test_config().await;

        // Create a mock streaming predictor (without actual model loading)
        let mut feature_buffer = VecDeque::with_capacity(config.buffer_size);

        let test_row = MarketDataRow {
            timestamp: 1640995200,
            open: 47000.5,
            high: 47100.0,
            low: 46900.0,
            close: 47050.0,
            volume: 1234.56,
        };

        feature_buffer.push_back(test_row.clone());
        assert_eq!(feature_buffer.len(), 1);
        assert_eq!(feature_buffer[0].timestamp, test_row.timestamp);
    }

    #[tokio::test]
    async fn test_buffer_size_management() {
        let buffer_size = 5;
        let mut feature_buffer = VecDeque::with_capacity(buffer_size);

        // Add more items than buffer size
        for i in 0..10 {
            let row = MarketDataRow {
                timestamp: 1640995200 + i,
                open: 47000.0,
                high: 47100.0,
                low: 46900.0,
                close: 47050.0,
                volume: 1000.0,
            };

            feature_buffer.push_back(row);

            // Maintain buffer size
            while feature_buffer.len() > buffer_size {
                feature_buffer.pop_front();
            }
        }

        assert_eq!(feature_buffer.len(), buffer_size);
        // Should contain the last 5 items (timestamps 1640995205 to 1640995209)
        assert_eq!(feature_buffer[0].timestamp, 1640995205);
        assert_eq!(feature_buffer[4].timestamp, 1640995209);
    }

    #[test]
    fn test_has_enough_data() {
        let config = RealtimeConfig {
            feature_window: 10,
            ..Default::default()
        };

        let mut feature_buffer = VecDeque::new();

        // Add 5 items - not enough
        for i in 0..5 {
            feature_buffer.push_back(MarketDataRow {
                timestamp: 1640995200 + i,
                open: 47000.0,
                high: 47100.0,
                low: 46900.0,
                close: 47050.0,
                volume: 1000.0,
            });
        }

        assert!(!feature_buffer.len() >= config.feature_window);

        // Add 5 more items - now enough
        for i in 5..10 {
            feature_buffer.push_back(MarketDataRow {
                timestamp: 1640995200 + i,
                open: 47000.0,
                high: 47100.0,
                low: 46900.0,
                close: 47050.0,
                volume: 1000.0,
            });
        }

        assert!(feature_buffer.len() >= config.feature_window);
    }
}
