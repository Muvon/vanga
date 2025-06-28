//! Output formatting and serialization for predictions
//!
//! This module provides structured output formats for VANGA predictions,
//! converting raw LSTM outputs into user-friendly JSON and CSV formats.

pub mod formatter;
pub mod post_processor;
pub mod structures;

// Re-export main types
pub use formatter::OutputFormatter;
pub use post_processor::PostProcessor;
pub use structures::{
    ConfidenceScore, DirectionPrediction, PredictionMetadata, PredictionResult, PriceBin,
    PriceLevelPrediction, VolatilityPrediction,
};
