//! Output formatting and serialization for predictions
//!
//! This module provides structured output formats for VANGA predictions,
//! converting raw LSTM outputs into user-friendly JSON and CSV formats.

pub mod formatter;
pub mod multi_target_parser;
pub mod post_processor;
pub mod structures;

// Re-export main types
pub use formatter::OutputFormatter;
pub use multi_target_parser::{DirectionOutput, MultiTargetParser, ParsedOutput};
pub use post_processor::PostProcessor;
pub use structures::{
    ConfidenceScore, DirectionPrediction, OrderConfig, OrderLevel, PredictionMetadata,
    PredictionResult, PriceBin, PriceLevelPrediction, TradingOrders, VolatilityPrediction,
};
