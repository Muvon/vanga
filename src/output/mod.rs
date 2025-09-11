//! Output formatting and serialization for predictions
//!
//! This module provides structured output formats for VANGA predictions,
//! converting raw LSTM outputs into user-friendly JSON and CSV formats.

pub mod adaptive_stop_calculator;
pub mod confidence_calculator;
#[cfg(test)]
pub mod confidence_calculator_test;
#[cfg(test)]
pub mod direct_exit_test;
#[cfg(test)]
pub mod exit_generation_bug_analysis_test;
pub mod formatter;
pub mod metadata;
pub mod model_boundaries;
pub mod multi_target_parser;
#[cfg(test)]
pub mod order_generation_test;
pub mod post_processor;
pub mod prediction_types;
pub mod sequence_statistics;
pub mod smart_order_generator;
pub mod structures;
pub mod trading_orders;

// Re-export main types from new modular structure
pub use formatter::OutputFormatter;
pub use metadata::{ConfidenceScore, DataQuality, PredictionMetadata};
pub use multi_target_parser::{DirectionOutput, MultiTargetParser, ParsedOutput};
pub use post_processor::PostProcessor;
pub use prediction_types::{
    DirectionPrediction, PredictionResult, PriceBin, PriceLevelPrediction, SentimentPrediction,
    VolatilityPrediction, VolumePrediction,
};
pub use sequence_statistics::SequenceStatistics;
pub use smart_order_generator::SmartConsensus;
pub use trading_orders::{OrderConfig, OrderLevel, SequenceAwareOrderConfig, TradingOrders};
