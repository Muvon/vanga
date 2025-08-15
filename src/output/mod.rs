//! Output formatting and serialization for predictions
//!
//! This module provides structured output formats for VANGA predictions,
//! converting raw LSTM outputs into user-friendly JSON and CSV formats.

pub mod confidence_calculator;
pub mod formatter;
pub mod metadata;
pub mod multi_target_parser;
pub mod post_processor;
pub mod prediction_types;
pub mod structures;
pub mod trading_orders;

// Re-export main types from new modular structure
pub use formatter::OutputFormatter;
pub use metadata::{ConfidenceScore, DataQuality, PredictionMetadata};
pub use multi_target_parser::{DirectionOutput, MultiTargetParser, ParsedOutput};
pub use post_processor::PostProcessor;
pub use prediction_types::{
    DirectionPrediction, PredictionResult, PriceBin, PriceLevelPrediction, VolatilityPrediction,
};
pub use trading_orders::{OrderConfig, OrderLevel, SequenceAwareOrderConfig, TradingOrders};
