//! Metadata and configuration structures
//!
//! This module contains structures for prediction metadata, data quality assessment,
//! and confidence scoring used throughout the prediction system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Overall confidence scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScore {
    /// Overall prediction confidence (0.0 to 1.0)
    pub overall: f64,

    /// Price level confidence
    pub price_levels: f64,

    /// Direction confidence
    pub direction: f64,

    /// Volatility confidence
    pub volatility: f64,
}

/// Prediction metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetadata {
    /// Model version/identifier
    pub model_version: String,

    /// Prediction generation timestamp
    pub generated_at: DateTime<Utc>,

    /// Number of features used
    pub feature_count: usize,

    /// Sequence length used for prediction
    pub sequence_length: usize,

    /// Data quality indicators
    pub data_quality: DataQuality,
}

/// Data quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuality {
    /// Completeness score (0.0 to 1.0)
    pub completeness: f64,

    /// Data freshness (hours since last data point)
    pub freshness_hours: f64,

    /// Market condition assessment
    pub market_condition: String, // "NORMAL", "VOLATILE", "TRENDING"
}

impl Default for PredictionMetadata {
    fn default() -> Self {
        Self {
            model_version: "1.0.0".to_string(),
            generated_at: Utc::now(),
            feature_count: 0,
            sequence_length: 0,
            data_quality: DataQuality {
                completeness: 1.0,
                freshness_hours: 0.0,
                market_condition: "NORMAL".to_string(),
            },
        }
    }
}
