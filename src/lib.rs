//! Core LSTM forecasting system for cryptocurrency markets
//!
//! This library provides a complete LSTM-based forecasting system specifically
//! designed for cryptocurrency markets with automatic feature engineering,
//! hyperparameter optimization, and multi-target prediction capabilities.
//!
//! # Example
//!
//! ```rust,no_run
//! use vanga::{train_model, predict, TrainingConfig, PredictionConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Train a multi-target model
//! let config = TrainingConfig {
//!     symbol: "BTCUSDT".to_string(),
//!     data_path: "./data/btc_ohlcv.csv".into(),
//!     horizons: vec!["1h".to_string(), "4h".to_string(), "1d".to_string()],
//!     ..TrainingConfig::default()
//! };
//!
//! let model = train_model(config).await?;
//!
//! // Make predictions using the trained multi-target model
//! let pred_config = PredictionConfig {
//!     symbol: "BTCUSDT".to_string(),
//!     input_path: "./data/recent_btc.csv".into(),
//!     ..PredictionConfig::default()
//! };
//! let predictions = model.predict_multi_target(&pred_config).await?;
//! # Ok(())
//! # }
//! ```

pub mod api;
pub mod config;
pub mod data;
pub mod features;
pub mod model;
pub mod optimization;
pub mod output;
pub mod realtime;
pub mod targets;
pub mod utils;

// External test modules
#[cfg(test)]
pub mod tests;

#[cfg(test)]
pub mod ta_tests;
