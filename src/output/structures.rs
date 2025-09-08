//! Backward compatibility re-exports for prediction output structures
//!
//! This module maintains backward compatibility by re-exporting all types
//! from the new modular structure. New code should import directly from
//! the specific modules (prediction_types, trading_orders, etc.).

// Re-export all types from the new modular structure
pub use super::metadata::{ConfidenceScore, DataQuality, PredictionMetadata};
pub use super::prediction_types::{
    DirectionPrediction, PredictionResult, PriceBin, PriceLevelPrediction, VolatilityPrediction,
};
pub use super::trading_orders::{OrderConfig, OrderLevel, SequenceAwareOrderConfig, TradingOrders};

// Include all tests from the original file to ensure they still work
