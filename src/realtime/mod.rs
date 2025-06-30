//! Real-time streaming prediction module
//!
//! This module provides real-time cryptocurrency prediction capabilities by monitoring
//! CSV files for updates and streaming predictions as new data arrives.
//!
//! Key features:
//! - Cross-platform file watching using notify crate
//! - Incremental CSV parsing with position tracking
//! - Streaming prediction pipeline integration
//! - Configurable output formats (JSON, CSV, Pretty)
//! - Memory-efficient sliding window buffer

pub mod predictor;
pub mod stream;
pub mod watcher;

pub use predictor::StreamingPredictor;
pub use stream::CsvStreamer;
pub use watcher::FileWatcher;

use crate::utils::error::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for real-time prediction streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConfig {
    /// Path to the CSV file being monitored
    pub file_path: PathBuf,
    /// Trading symbol (e.g., BTCUSDT)
    pub symbol: String,
    /// Polling interval for health checks
    pub poll_interval: Duration,
    /// Maximum size of the feature buffer
    pub buffer_size: usize,
    /// Minimum number of data points required for prediction
    pub feature_window: usize,
    /// Output format for predictions
    pub output_format: OutputFormat,
    /// Enable debug logging
    pub debug: bool,
}

/// Output format options for streaming predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    /// JSON format with full prediction details
    Json,
    /// CSV format for easy integration
    Csv,
    /// Human-readable pretty format
    Pretty,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            file_path: PathBuf::from("data/realtime.csv"),
            symbol: "BTCUSDT".to_string(),
            poll_interval: Duration::from_secs(1),
            buffer_size: 1000,
            feature_window: 100,
            output_format: OutputFormat::Json,
            debug: false,
        }
    }
}

/// Start real-time prediction streaming
///
/// This is the main entry point for real-time prediction. It creates a streaming
/// predictor and runs the prediction loop until interrupted.
///
/// # Arguments
/// * `config` - Real-time configuration parameters
///
/// # Returns
/// * `Result<()>` - Success or error result
///
/// # Example
/// ```rust,no_run
/// use vanga::realtime::{start_realtime_prediction, RealtimeConfig};
/// use std::path::PathBuf;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = RealtimeConfig {
///     file_path: PathBuf::from("data/live_btc.csv"),
///     symbol: "BTCUSDT".to_string(),
///     ..Default::default()
/// };
///
/// start_realtime_prediction(config).await?;
/// # Ok(())
/// # }
/// ```
pub async fn start_realtime_prediction(config: RealtimeConfig) -> Result<()> {
    log::info!("Initializing real-time prediction system");
    log::info!("Configuration: {:?}", config);

    let mut predictor = StreamingPredictor::new(config.clone()).await?;
    predictor.run().await
}
