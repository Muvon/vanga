//! Multi-target output parser for LSTM predictions
//!
//! This module handles parsing raw LSTM outputs into structured prediction components
//! based on the configured output heads (price levels, direction, volatility).

use crate::config::model::NUM_CLASSES;
use crate::utils::error::{Result, VangaError};
use ndarray::{s, ArrayView1};

/// Output segments for multi-target parsing - always 5 targets with NUM_CLASSES each
#[derive(Debug, Clone)]
pub struct OutputSegments {
    /// Price levels segment: (start_idx, end_idx)
    pub price_levels: Option<(usize, usize)>,
    /// Direction segment: (start_idx, end_idx)
    pub direction: Option<(usize, usize)>,
    /// Volatility segment: (start_idx, end_idx)
    pub volatility: Option<(usize, usize)>,
    /// Sentiment segment: (start_idx, end_idx)
    pub sentiment: Option<(usize, usize)>,
    /// Volume segment: (start_idx, end_idx)
    pub volume: Option<(usize, usize)>,
}

impl Default for OutputSegments {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputSegments {
    /// Create output segments for always-enabled 5 targets with NUM_CLASSES=5 each
    pub fn new() -> Self {
        // All targets are always enabled with NUM_CLASSES=5 each
        // Total output size: 5 targets * 5 classes = 25
        Self {
            price_levels: Some((0, NUM_CLASSES)),                 // 0-4
            direction: Some((NUM_CLASSES, NUM_CLASSES * 2)),      // 5-9
            volatility: Some((NUM_CLASSES * 2, NUM_CLASSES * 3)), // 10-14
            sentiment: Some((NUM_CLASSES * 3, NUM_CLASSES * 4)),  // 15-19
            volume: Some((NUM_CLASSES * 4, NUM_CLASSES * 5)),     // 20-24
        }
    }
}

/// Multi-target output parser
pub struct MultiTargetParser {
    pub segments: OutputSegments, // Made public for debugging
}

impl Default for MultiTargetParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiTargetParser {
    /// Create new parser with output configuration
    pub fn new() -> Self {
        let segments = OutputSegments::new(); // Always use 3 targets with NUM_CLASSES=5 each
        Self { segments }
    }

    /// Parse raw LSTM output into structured components
    pub fn parse_output(&self, raw_output: ArrayView1<f64>) -> Result<ParsedOutput> {
        let mut parsed = ParsedOutput::new();

        // Parse price levels if enabled
        if let Some((start, end)) = self.segments.price_levels {
            if end <= raw_output.len() {
                let price_level_logits = raw_output.slice(s![start..end]);
                parsed.price_levels = Some(self.parse_price_levels(&price_level_logits)?);
            } else {
                log::warn!(
                    "Price levels segment out of bounds: {} > {}",
                    end,
                    raw_output.len()
                );
            }
        }

        // Parse direction if enabled
        if let Some((start, end)) = self.segments.direction {
            if end <= raw_output.len() {
                let direction_logits = raw_output.slice(s![start..end]);
                parsed.direction = Some(self.parse_direction(&direction_logits)?);
            } else {
                log::warn!(
                    "Direction segment out of bounds: {} > {}",
                    end,
                    raw_output.len()
                );
            }
        }

        // Parse volatility if enabled
        if let Some((start, end)) = self.segments.volatility {
            if end <= raw_output.len() {
                let volatility_values = raw_output.slice(s![start..end]);
                parsed.volatility = Some(self.parse_volatility(&volatility_values)?);
            } else {
                log::warn!(
                    "Volatility segment out of bounds: {} > {}",
                    end,
                    raw_output.len()
                );
            }
        }

        // Parse sentiment if enabled
        if let Some((start, end)) = self.segments.sentiment {
            if end <= raw_output.len() {
                let sentiment_values = raw_output.slice(s![start..end]);
                parsed.sentiment = Some(self.parse_sentiment(&sentiment_values)?);
            } else {
                log::warn!(
                    "Sentiment segment out of bounds: {} > {}",
                    end,
                    raw_output.len()
                );
            }
        }

        // Parse volume if enabled
        if let Some((start, end)) = self.segments.volume {
            if end <= raw_output.len() {
                let volume_values = raw_output.slice(s![start..end]);
                parsed.volume = Some(self.parse_volume(&volume_values)?);
            } else {
                log::warn!(
                    "Volume segment out of bounds: {} > {}",
                    end,
                    raw_output.len()
                );
            }
        }

        Ok(parsed)
    }

    /// Parse direction logits into class probabilities (5-class system)
    fn parse_direction(&self, logits: &ArrayView1<f64>) -> Result<DirectionOutput> {
        let expected_classes = NUM_CLASSES; // DUMP, DOWN, SIDEWAYS, UP, PUMP
        if logits.len() != expected_classes {
            return Err(VangaError::PredictionError(format!(
                "Expected {} direction classes, got {}",
                expected_classes,
                logits.len()
            )));
        }

        // Apply softmax to convert logits to probabilities
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();

        if sum_exp == 0.0 {
            return Err(VangaError::PredictionError(
                "Invalid direction logits: sum of exponentials is zero".to_string(),
            ));
        }

        let probabilities: Vec<f64> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        Ok(DirectionOutput {
            dump_probability: probabilities[0],     // New: Extreme down
            down_probability: probabilities[1],     // Shifted from index 0
            sideways_probability: probabilities[2], // Shifted from index 1
            up_probability: probabilities[3],       // Shifted from index 2
            pump_probability: probabilities[4],     // New: Extreme up
        })
    }

    /// Parse volatility values (5-class system)
    fn parse_volatility(&self, values: &ArrayView1<f64>) -> Result<VolatilityOutput> {
        let expected_total_classes = NUM_CLASSES;

        if values.len() != expected_total_classes {
            return Err(VangaError::PredictionError(format!(
                "Expected {} volatility classes, got {}",
                expected_total_classes,
                values.len()
            )));
        }

        // Apply softmax to convert logits to probabilities
        let probabilities = self.softmax(values);

        Ok(VolatilityOutput {
            very_low_probability: probabilities[0],
            low_probability: probabilities[1],
            medium_probability: probabilities[2],
            high_probability: probabilities[3],
            very_high_probability: probabilities[4],
        })
    }

    /// Parse sentiment output (5-class system)
    fn parse_sentiment(&self, values: &ArrayView1<f64>) -> Result<SentimentOutput> {
        let expected_total_classes = NUM_CLASSES;

        if values.len() != expected_total_classes {
            return Err(VangaError::PredictionError(format!(
                "Expected {} sentiment classes, got {}",
                expected_total_classes,
                values.len()
            )));
        }

        // Apply softmax to convert logits to probabilities
        let probabilities = self.softmax(values);

        Ok(SentimentOutput {
            very_bearish_probability: probabilities[0],
            bearish_probability: probabilities[1],
            neutral_probability: probabilities[2],
            bullish_probability: probabilities[3],
            very_bullish_probability: probabilities[4],
        })
    }

    /// Parse volume output (5-class system)
    fn parse_volume(&self, values: &ArrayView1<f64>) -> Result<VolumeOutput> {
        let expected_total_classes = NUM_CLASSES;

        if values.len() != expected_total_classes {
            return Err(VangaError::PredictionError(format!(
                "Expected {} volume classes, got {}",
                expected_total_classes,
                values.len()
            )));
        }

        // Apply softmax to convert logits to probabilities
        let probabilities = self.softmax(values);

        Ok(VolumeOutput {
            very_low_probability: probabilities[0],
            low_probability: probabilities[1],
            medium_probability: probabilities[2],
            high_probability: probabilities[3],
            very_high_probability: probabilities[4],
        })
    }

    /// Apply softmax to convert logits to probabilities
    fn softmax(&self, logits: &ArrayView1<f64>) -> Vec<f64> {
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();

        if sum_exp == 0.0 {
            // Return uniform distribution if sum is zero
            vec![1.0 / logits.len() as f64; logits.len()]
        } else {
            exp_logits.iter().map(|&x| x / sum_exp).collect()
        }
    }

    /// Parse price level logits with validation (5-class system)
    fn parse_price_levels(&self, logits: &ArrayView1<f64>) -> Result<Vec<f64>> {
        let expected_classes = NUM_CLASSES; // Unified 5-class system
        if logits.len() != expected_classes {
            return Err(VangaError::PredictionError(format!(
                "Expected {} price level classes, got {}",
                expected_classes,
                logits.len()
            )));
        }

        // Apply softmax to convert logits to probabilities
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();

        if sum_exp == 0.0 {
            return Err(VangaError::PredictionError(
                "Invalid logits: sum of exponentials is zero".to_string(),
            ));
        }

        let probabilities: Vec<f64> = exp_logits.iter().map(|&x| x / sum_exp).collect();
        Ok(probabilities)
    }
}

/// Parsed multi-target output
#[derive(Debug, Clone)]
pub struct ParsedOutput {
    /// Price level probabilities (if enabled)
    pub price_levels: Option<Vec<f64>>,

    /// Direction probabilities (if enabled) - 5-class system
    pub direction: Option<DirectionOutput>,

    /// Volatility values (if enabled) - 5-class system
    pub volatility: Option<VolatilityOutput>,

    /// Sentiment values (if enabled) - 5-class system
    pub sentiment: Option<SentimentOutput>,

    /// Volume values (if enabled) - 5-class system
    pub volume: Option<VolumeOutput>,
}

impl Default for ParsedOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl ParsedOutput {
    pub fn new() -> Self {
        Self {
            price_levels: None,
            direction: None,
            volatility: None,
            sentiment: None,
            volume: None,
        }
    }
}

/// Direction prediction output (5-class system)
#[derive(Debug, Clone)]
pub struct DirectionOutput {
    pub dump_probability: f64,     // Extreme down
    pub down_probability: f64,     // Moderate down
    pub sideways_probability: f64, // Minimal change
    pub up_probability: f64,       // Moderate up
    pub pump_probability: f64,     // Extreme up
}

/// Volatility prediction output for a single horizon (5-class system)
#[derive(Debug, Clone)]
pub struct VolatilityOutput {
    pub very_low_probability: f64,  // Class 0: Much below sequence baseline
    pub low_probability: f64,       // Class 1: Below sequence baseline
    pub medium_probability: f64,    // Class 2: Around sequence baseline
    pub high_probability: f64,      // Class 3: Above sequence baseline
    pub very_high_probability: f64, // Class 4: Much above sequence baseline
}

impl DirectionOutput {
    /// Get the most likely direction (5-class system)
    pub fn get_prediction(&self) -> String {
        let probabilities = [
            ("DUMP", self.dump_probability),
            ("DOWN", self.down_probability),
            ("SIDEWAYS", self.sideways_probability),
            ("UP", self.up_probability),
            ("PUMP", self.pump_probability),
        ];

        let (prediction, _) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        prediction.to_string()
    }

    /// Get confidence (highest probability across all 5 classes)
    pub fn get_confidence(&self) -> f64 {
        self.dump_probability
            .max(self.down_probability)
            .max(self.sideways_probability)
            .max(self.up_probability)
            .max(self.pump_probability)
    }
}

impl VolatilityOutput {
    /// Get the most likely volatility regime (5-class system)
    pub fn get_prediction(&self) -> String {
        let probabilities = [
            ("VERY_LOW", self.very_low_probability),
            ("LOW", self.low_probability),
            ("MEDIUM", self.medium_probability),
            ("HIGH", self.high_probability),
            ("VERY_HIGH", self.very_high_probability),
        ];

        let (prediction, _) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        prediction.to_string()
    }

    /// Get confidence (highest probability across all 5 classes)
    pub fn get_confidence(&self) -> f64 {
        self.very_low_probability
            .max(self.low_probability)
            .max(self.medium_probability)
            .max(self.high_probability)
            .max(self.very_high_probability)
    }
}

/// Sentiment prediction output (5-class system)
#[derive(Debug, Clone)]
pub struct SentimentOutput {
    pub very_bearish_probability: f64, // Class 0: Very bearish sentiment
    pub bearish_probability: f64,      // Class 1: Bearish sentiment
    pub neutral_probability: f64,      // Class 2: Neutral sentiment
    pub bullish_probability: f64,      // Class 3: Bullish sentiment
    pub very_bullish_probability: f64, // Class 4: Very bullish sentiment
}

impl SentimentOutput {
    /// Get the most likely sentiment (5-class system)
    pub fn get_prediction(&self) -> String {
        let probabilities = [
            ("VERY_BEARISH", self.very_bearish_probability),
            ("BEARISH", self.bearish_probability),
            ("NEUTRAL", self.neutral_probability),
            ("BULLISH", self.bullish_probability),
            ("VERY_BULLISH", self.very_bullish_probability),
        ];

        let (prediction, _) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        prediction.to_string()
    }

    /// Get confidence (highest probability across all 5 classes)
    pub fn get_confidence(&self) -> f64 {
        self.very_bearish_probability
            .max(self.bearish_probability)
            .max(self.neutral_probability)
            .max(self.bullish_probability)
            .max(self.very_bullish_probability)
    }
}

/// Volume prediction output (5-class system)
#[derive(Debug, Clone)]
pub struct VolumeOutput {
    pub very_low_probability: f64,  // Class 0: Very low volume
    pub low_probability: f64,       // Class 1: Low volume
    pub medium_probability: f64,    // Class 2: Medium volume
    pub high_probability: f64,      // Class 3: High volume
    pub very_high_probability: f64, // Class 4: Very high volume
}

impl VolumeOutput {
    /// Get the most likely volume regime (5-class system)
    pub fn get_prediction(&self) -> String {
        let probabilities = [
            ("VERY_LOW", self.very_low_probability),
            ("LOW", self.low_probability),
            ("MEDIUM", self.medium_probability),
            ("HIGH", self.high_probability),
            ("VERY_HIGH", self.very_high_probability),
        ];

        let (prediction, _) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        prediction.to_string()
    }

    /// Get confidence (highest probability across all 5 classes)
    pub fn get_confidence(&self) -> f64 {
        self.very_low_probability
            .max(self.low_probability)
            .max(self.medium_probability)
            .max(self.high_probability)
            .max(self.very_high_probability)
    }
}
