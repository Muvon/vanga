//! Prediction output data structures
//!
//! These structures define the JSON output format that matches ARCHITECTURE.md specifications
//! while reusing existing target generation logic from src/targets/

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main prediction result structure matching ARCHITECTURE.md JSON format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Trading symbol (e.g., "BTCUSDT")
    pub symbol: String,

    /// Prediction timestamp in ISO format
    pub timestamp: String,

    /// Prediction horizon (e.g., "4h", "1d")
    pub horizon: String,

    /// Current price at prediction time
    pub current_price: f64,

    /// Price level predictions (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price_levels: Option<PriceLevelPrediction>,

    /// Direction predictions (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<DirectionPrediction>,

    /// Volatility predictions (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volatility: Option<VolatilityPrediction>,

    /// Overall prediction confidence
    pub confidence: f64,

    /// Prediction metadata
    pub metadata: PredictionMetadata,
}

/// Price level prediction with probability distribution
/// Matches ARCHITECTURE.md bin structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevelPrediction {
    /// Probability distribution across price bins
    pub bins: HashMap<String, PriceBin>,

    /// Most likely price range as numeric array [min, max]
    pub most_likely_range: [f64; 2],

    /// Confidence in price level prediction
    pub confidence: f64,
}

/// Individual price bin with range and probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceBin {
    /// Price range as percentage array [min, max] (e.g., [-5.0, -3.0])
    pub range: [f64; 2],

    /// Price range in actual currency values [min, max]
    pub price: [f64; 2],

    /// Probability of price falling in this range
    pub probability: f64,
}

/// Direction prediction matching ARCHITECTURE.md format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionPrediction {
    /// Probability of upward movement
    pub up_probability: f64,

    /// Probability of downward movement
    pub down_probability: f64,

    /// Predicted direction ("UP", "DOWN", "SIDEWAYS")
    pub prediction: String,

    /// Confidence in direction prediction
    pub confidence: f64,
}

/// Multi-horizon volatility prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityPrediction {
    /// Expected 1-hour volatility
    pub expected_1h: f64,

    /// Expected 4-hour volatility
    pub expected_4h: f64,

    /// Expected 24-hour volatility
    pub expected_24h: f64,

    /// Volatility regime prediction ("LOW", "MEDIUM", "HIGH")
    pub regime: String,

    /// Confidence in volatility prediction
    pub confidence: f64,
}

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

impl PredictionResult {
    /// Create a new prediction result with basic info
    pub fn new(symbol: String, horizon: String, current_price: f64) -> Self {
        Self {
            symbol,
            timestamp: Utc::now().to_rfc3339(),
            horizon,
            current_price,
            price_levels: None,
            direction: None,
            volatility: None,
            confidence: 0.0,
            metadata: PredictionMetadata {
                model_version: "1.0.0".to_string(),
                generated_at: Utc::now(),
                feature_count: 0,   // Will be updated by formatter
                sequence_length: 0, // Will be updated by formatter
                data_quality: DataQuality {
                    completeness: 1.0,
                    freshness_hours: 0.0,
                    market_condition: "NORMAL".to_string(),
                },
            },
        }
    }

    /// Create a new prediction result with complete metadata
    pub fn new_with_metadata(
        symbol: String,
        horizon: String,
        current_price: f64,
        feature_count: usize,
        sequence_length: usize,
    ) -> Self {
        Self {
            symbol,
            timestamp: Utc::now().to_rfc3339(),
            horizon,
            current_price,
            price_levels: None,
            direction: None,
            volatility: None,
            confidence: 0.0,
            metadata: PredictionMetadata {
                model_version: "1.0.0".to_string(),
                generated_at: Utc::now(),
                feature_count,
                sequence_length,
                data_quality: DataQuality {
                    completeness: 1.0,
                    freshness_hours: 0.0,
                    market_condition: "NORMAL".to_string(),
                },
            },
        }
    }

    /// Set price level prediction
    pub fn with_price_levels(mut self, price_levels: PriceLevelPrediction) -> Self {
        self.price_levels = Some(price_levels);
        self
    }

    /// Set direction prediction
    pub fn with_direction(mut self, direction: DirectionPrediction) -> Self {
        self.direction = Some(direction);
        self
    }

    /// Set volatility prediction
    pub fn with_volatility(mut self, volatility: VolatilityPrediction) -> Self {
        self.volatility = Some(volatility);
        self
    }

    /// Set overall confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Update metadata
    pub fn with_metadata(mut self, metadata: PredictionMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}
