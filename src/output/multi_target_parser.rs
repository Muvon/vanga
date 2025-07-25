//! Multi-target output parser for LSTM predictions
//!
//! This module handles parsing raw LSTM outputs into structured prediction components
//! based on the configured output heads (price levels, direction, volatility).

use crate::config::model::{OutputHeadsConfig, OutputSegments, NUM_CLASSES};
use crate::utils::error::{Result, VangaError};
use ndarray::{s, ArrayView1};

/// Multi-target output parser
pub struct MultiTargetParser {
    output_heads: OutputHeadsConfig,
    pub segments: OutputSegments, // Made public for debugging
}

impl MultiTargetParser {
    /// Create new parser with output configuration
    pub fn new(output_heads: OutputHeadsConfig) -> Self {
        let segments = output_heads.get_output_segments();
        Self {
            output_heads,
            segments,
        }
    }

    /// Parse raw LSTM output into structured components
    pub fn parse_output(&self, raw_output: ArrayView1<f64>) -> Result<ParsedOutput> {
        let mut parsed = ParsedOutput::new();

        // Parse price levels if enabled
        if let Some((start, end)) = self.segments.price_levels {
            println!("DEBUG: Price levels segment found: ({}, {})", start, end);
            if end <= raw_output.len() {
                let price_level_logits = raw_output.slice(s![start..end]);
                println!(
                    "DEBUG: Price levels slice: start={}, end={}, slice_len={}, expected_classes={}",
                    start,
                    end,
                    price_level_logits.len(),
                    NUM_CLASSES // Unified 5-class system
                );
                println!(
                    "DEBUG: Price levels slice content: {:?}",
                    price_level_logits.to_vec()
                );
                parsed.price_levels = Some(self.parse_price_levels(&price_level_logits)?);
            } else {
                log::warn!(
                    "Price levels segment out of bounds: {} > {}",
                    end,
                    raw_output.len()
                );
            }
        } else {
            println!("DEBUG: No price levels segment found");
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

        // Validate that direction head is actually enabled
        if !self.output_heads.direction.enabled {
            return Err(VangaError::PredictionError(
                "Direction head is disabled but direction data provided".to_string(),
            ));
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
        // Validate that volatility head is enabled and class count matches
        if !self.output_heads.volatility.enabled {
            return Err(VangaError::PredictionError(
                "Volatility head is disabled but volatility data provided".to_string(),
            ));
        }

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
        // Validate that price levels head is enabled and class count matches
        if !self.output_heads.price_levels.enabled {
            return Err(VangaError::PredictionError(
                "Price levels head is disabled but price level data provided".to_string(),
            ));
        }

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
    pub very_low_probability: f64,  // <20th percentile
    pub low_probability: f64,       // 20th-40th percentile
    pub medium_probability: f64,    // 40th-60th percentile
    pub high_probability: f64,      // 60th-80th percentile
    pub very_high_probability: f64, // >80th percentile
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
