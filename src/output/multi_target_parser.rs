//! Multi-target output parser for LSTM predictions
//!
//! This module handles parsing raw LSTM outputs into structured prediction components
//! based on the configured output heads (price levels, direction, volatility).

use crate::config::model::{OutputHeadsConfig, OutputSegments};
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
                    "DEBUG: Price levels slice: start={}, end={}, slice_len={}, expected_bins={}",
                    start,
                    end,
                    price_level_logits.len(),
                    self.output_heads.price_levels.bins
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

    /// Parse direction logits into class probabilities
    fn parse_direction(&self, logits: &ArrayView1<f64>) -> Result<DirectionOutput> {
        let expected_classes = 3; // DOWN, SIDEWAYS, UP
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
            down_probability: probabilities[0],
            sideways_probability: probabilities[1],
            up_probability: probabilities[2],
        })
    }

    /// Parse volatility values for each horizon
    fn parse_volatility(&self, values: &ArrayView1<f64>) -> Result<Vec<f64>> {
        // Validate that volatility head is enabled and horizon count matches
        if !self.output_heads.volatility.enabled {
            return Err(VangaError::PredictionError(
                "Volatility head is disabled but volatility data provided".to_string(),
            ));
        }

        let expected_horizons = self.output_heads.volatility.horizons.len();
        if values.len() != expected_horizons {
            return Err(VangaError::PredictionError(format!(
                "Expected {} volatility horizons, got {}",
                expected_horizons,
                values.len()
            )));
        }

        // Apply sigmoid to ensure positive volatility values
        let volatilities: Vec<f64> = values
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp())) // Sigmoid function
            .map(|x| x * 0.1) // Scale to reasonable volatility range (0-10%)
            .collect();

        Ok(volatilities)
    }

    /// Parse price level logits with validation
    fn parse_price_levels(&self, logits: &ArrayView1<f64>) -> Result<Vec<f64>> {
        // Validate that price levels head is enabled and bin count matches
        if !self.output_heads.price_levels.enabled {
            return Err(VangaError::PredictionError(
                "Price levels head is disabled but price level data provided".to_string(),
            ));
        }

        let expected_bins = self.output_heads.price_levels.bins as usize;
        if logits.len() != expected_bins {
            return Err(VangaError::PredictionError(format!(
                "Expected {} price level bins, got {}",
                expected_bins,
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

    /// Direction probabilities (if enabled)
    pub direction: Option<DirectionOutput>,

    /// Volatility values for each horizon (if enabled)
    pub volatility: Option<Vec<f64>>,
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

/// Direction prediction output
#[derive(Debug, Clone)]
pub struct DirectionOutput {
    pub down_probability: f64,
    pub sideways_probability: f64,
    pub up_probability: f64,
}

impl DirectionOutput {
    /// Get the most likely direction
    pub fn get_prediction(&self) -> String {
        if self.up_probability > self.down_probability
            && self.up_probability > self.sideways_probability
        {
            "UP".to_string()
        } else if self.down_probability > self.sideways_probability {
            "DOWN".to_string()
        } else {
            "SIDEWAYS".to_string()
        }
    }

    /// Get confidence (highest probability)
    pub fn get_confidence(&self) -> f64 {
        self.up_probability
            .max(self.down_probability)
            .max(self.sideways_probability)
    }
}
