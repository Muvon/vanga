//! Core LSTM forecasting system for cryptocurrency markets
//!
//! This library provides a complete LSTM-based forecasting system specifically
//! designed for cryptocurrency markets with automatic feature engineering,
//! hyperparameter optimization, and multi-target prediction capabilities.
//!
//! # Example
//!
//! ```rust,no_run
//! use vanga::{train_model, predict, TrainingConfig};
//!
//! // Train a model
//! let config = TrainingConfig::default()
//!     .symbol("BTCUSDT")
//!     .data_path("./data/btc_ohlcv.csv")
//!     .horizons(vec!["1h", "4h", "1d"]);
//!
//! let model = train_model(config)?;
//!
//! // Make predictions
//! let predictions = predict(&model, "./data/recent_btc.csv")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod api;
pub mod config;
pub mod data;
pub mod features;
pub mod model;
pub mod targets;
pub mod utils;

// Re-export main API
pub use api::{predict, train_model, ModelTrainer, Predictor};
pub use config::{ModelConfig, PredictionConfig, TrainingConfig};
pub use data::{CryptoDataSchema, DataLoader};
pub use utils::error::{Result, VangaError};
