//! Output formatting logic for converting raw LSTM outputs to structured predictions
//!
//! This module bridges the gap between raw LSTM Array2<f64> outputs and the structured
//! JSON format specified in ARCHITECTURE.md, reusing existing target generation logic.

use crate::config::model::{OutputHeadsConfig, NUM_CLASSES};
use crate::config::prediction::{OutputConfig, OutputFormat};
use crate::output::multi_target_parser::{DirectionOutput, MultiTargetParser, VolatilityOutput};
use crate::output::structures::{
    DirectionPrediction, PredictionResult, PriceBin, PriceLevelPrediction, VolatilityPrediction,
};
use crate::targets::PreparedTargets;
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;
use std::collections::HashMap;

/// Output formatter that converts raw LSTM predictions to structured formats
pub struct OutputFormatter {
    config: OutputConfig,
    parser: Option<MultiTargetParser>,
    sequence_data: Option<Vec<f64>>,
    /// Model config bandwidth_size: sequence bandwidth multiplier (e.g., 1.0, 1.5, 2.0)
    /// Used in target generation as: sequence_bandwidth = (max - min) * bandwidth_size
    bandwidth_size: Option<f64>,
}

impl OutputFormatter {
    /// Create new formatter with configuration
    pub fn new(config: OutputConfig) -> Self {
        Self {
            config,
            parser: None,
            sequence_data: None,
            bandwidth_size: None,
        }
    }

    /// Set output heads configuration for proper 5-class parsing
    pub fn with_output_heads(mut self, output_heads: OutputHeadsConfig) -> Self {
        // Extract bandwidth_size from price levels config
        self.bandwidth_size = output_heads.price_levels.bandwidth_size;
        self.parser = Some(MultiTargetParser::new(output_heads));
        self
    }

    /// Set sequence data for sequence-aware price level calculations
    pub fn with_sequence_data(mut self, sequence_data: Vec<f64>) -> Self {
        self.sequence_data = Some(sequence_data);
        self
    }

    /// Check if multi-target parser is available
    pub fn has_parser(&self) -> bool {
        self.parser.is_some()
    }

    /// Get sequence data reference
    pub fn get_sequence_data(&self) -> Option<&[f64]> {
        self.sequence_data.as_deref()
    }

    /// Get bandwidth size
    pub fn get_bandwidth_size(&self) -> Option<f64> {
        self.bandwidth_size
    }

    /// Parse raw predictions using the internal MultiTargetParser
    pub fn parse_raw_predictions(
        &self,
        raw_output: ndarray::ArrayView1<f64>,
    ) -> Result<crate::output::multi_target_parser::ParsedOutput> {
        let parser = self.parser.as_ref().ok_or_else(|| {
            VangaError::PredictionError(
                "MultiTargetParser not configured. Use with_output_heads() to set up 5-class parsing.".to_string()
            )
        })?;
        parser.parse_output(raw_output)
    }

    /// Format raw LSTM predictions into structured output
    ///
    /// This is the main entry point that converts Array2<f64> to PredictionResult
    /// based on the configured output format and enabled prediction heads.
    pub fn format_predictions(
        &self,
        raw_predictions: &Array2<f64>,
        symbol: &str,
        horizon: &str,
        current_price: f64,
        targets_config: Option<&PreparedTargets>,
    ) -> Result<Vec<PredictionResult>> {
        match self.config.format {
            OutputFormat::ProbabilityDistribution => self.format_probability_distribution(
                raw_predictions,
                symbol,
                horizon,
                current_price,
                targets_config,
            ),
            OutputFormat::ConfidenceInterval => self.format_confidence_interval(
                raw_predictions,
                symbol,
                horizon,
                current_price,
                targets_config,
            ),
            OutputFormat::PointEstimate => self.format_point_estimate(
                raw_predictions,
                symbol,
                horizon,
                current_price,
                targets_config,
            ),
            OutputFormat::All => {
                // Return all formats (for now, just probability distribution)
                self.format_probability_distribution(
                    raw_predictions,
                    symbol,
                    horizon,
                    current_price,
                    targets_config,
                )
            }
        }
    }

    /// Format as probability distribution (main ARCHITECTURE.md format)
    fn format_probability_distribution(
        &self,
        raw_predictions: &Array2<f64>,
        symbol: &str,
        horizon: &str,
        current_price: f64,
        targets_config: Option<&PreparedTargets>,
    ) -> Result<Vec<PredictionResult>> {
        let mut results = Vec::new();

        // Calculate confidence based on target distribution balance if available
        let base_confidence = if let Some(targets_data) = targets_config {
            calculate_target_based_confidence(targets_data, horizon)
        } else {
            0.7 // Default confidence when no target statistics available
        };

        // Check if we have the multi-target parser configured
        let parser = self.parser.as_ref().ok_or_else(|| {
            VangaError::PredictionError(
                "MultiTargetParser not configured. Use with_output_heads() to set up 5-class parsing.".to_string()
            )
        })?;

        // For each batch in the predictions
        for batch_idx in 0..raw_predictions.nrows() {
            // Get actual feature count and sequence length from the prediction data
            let feature_count = raw_predictions.ncols();
            let sequence_length = 60; // Default LSTM sequence length since targets_config doesn't contain this info

            let mut result = PredictionResult::new_with_metadata(
                symbol.to_string(),
                horizon.to_string(),
                current_price,
                feature_count,
                sequence_length,
            );

            // Extract predictions for this batch
            let batch_predictions = raw_predictions.row(batch_idx);

            // Parse the raw predictions using the multi-target parser
            let parsed_output = parser.parse_output(batch_predictions)?;

            // Convert parsed output to structured predictions
            if let Some(price_level_probs) = parsed_output.price_levels {
                result = result.with_price_levels(self.create_price_level_prediction(
                    &price_level_probs,
                    current_price,
                    self.sequence_data.as_deref(),
                    self.bandwidth_size,
                )?);
            }

            if let Some(direction_output) = parsed_output.direction {
                // Calculate actual sequence bandwidth percentage from sequence data
                let sequence_bandwidth_percent = if let Some(sequence_prices) = &self.sequence_data
                {
                    if sequence_prices.len() >= 2 {
                        let min_price =
                            sequence_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max_price = sequence_prices
                            .iter()
                            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                        // Use model config bandwidth_size (sequence bandwidth multiplier)
                        let model_bandwidth_multiplier = self.bandwidth_size.ok_or_else(|| {
                            VangaError::PredictionError(
                                "Model bandwidth_size not configured. This is required for adaptive predictions.".to_string()
                            )
                        })?;

                        // Calculate bandwidth as percentage of current price
                        let sequence_price_range = max_price - min_price;
                        let model_bandwidth_absolute =
                            sequence_price_range * model_bandwidth_multiplier;
                        (model_bandwidth_absolute / current_price) * 100.0 // Convert to percentage
                    } else {
                        return Err(VangaError::PredictionError(format!(
                            "Insufficient sequence data for adaptive predictions: {} prices (need ≥2)",
                            sequence_prices.len()
                        )));
                    }
                } else {
                    return Err(VangaError::PredictionError(
                        "Sequence data not available for adaptive predictions. Use with_sequence_data() to provide it.".to_string()
                    ));
                };

                result = result.with_direction(self.create_direction_prediction(
                    &direction_output,
                    Some(horizon),
                    Some(sequence_length as u32),
                    Some(sequence_bandwidth_percent),
                )?);
            }

            if let Some(volatility_output) = parsed_output.volatility {
                // Calculate sequence bandwidth percentage (same logic as direction)
                let sequence_bandwidth_percent = if let Some(sequence_prices) = &self.sequence_data
                {
                    if sequence_prices.len() >= 2 {
                        let min_price =
                            sequence_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max_price = sequence_prices
                            .iter()
                            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                        // Use model config bandwidth_size (sequence bandwidth multiplier)
                        let model_bandwidth_multiplier = self.bandwidth_size.ok_or_else(|| {
                            VangaError::PredictionError(
                                "Model bandwidth_size not configured. This is required for adaptive predictions.".to_string()
                            )
                        })?;

                        let sequence_price_range = max_price - min_price;
                        let model_bandwidth_absolute =
                            sequence_price_range * model_bandwidth_multiplier;
                        (model_bandwidth_absolute / current_price) * 100.0 // Convert to percentage
                    } else {
                        return Err(VangaError::PredictionError(format!(
                            "Insufficient sequence data for adaptive predictions: {} prices (need ≥2)",
                            sequence_prices.len()
                        )));
                    }
                } else {
                    return Err(VangaError::PredictionError(
                        "Sequence data not available for adaptive predictions. Use with_sequence_data() to provide it.".to_string()
                    ));
                };

                // Calculate volatility percentile from sequence data
                let volatility_percentile = self.calculate_volatility_percentile(
                    self.sequence_data.as_ref().unwrap(), // Safe unwrap - already checked above
                );

                result = result.with_volatility(self.create_volatility_prediction(
                    &volatility_output,
                    Some(horizon),
                    Some(sequence_bandwidth_percent),
                    Some(volatility_percentile),
                )?);
            }

            // Apply the calculated confidence to the prediction result
            result = result.with_confidence(base_confidence);

            results.push(result);
        }

        Ok(results)
    }

    /// Create price level prediction from 5-class probabilities using sequence-aware ranges
    pub fn create_price_level_prediction(
        &self,
        probabilities: &[f64],
        current_price: f64,
        sequence_prices: Option<&[f64]>,
        bandwidth_size: Option<f64>,
    ) -> Result<PriceLevelPrediction> {
        if probabilities.len() != NUM_CLASSES {
            return Err(VangaError::PredictionError(format!(
                "Expected {} price level probabilities, got {}",
                NUM_CLASSES,
                probabilities.len()
            )));
        }

        let mut bins = HashMap::new();

        // Calculate sequence-aware ranges using the same logic as target generation
        let (bin_ranges, bin_names) = if let Some(prices) = sequence_prices {
            self.calculate_sequence_aware_ranges(
                prices,
                current_price,
                bandwidth_size.unwrap_or(1.0),
            )
        } else {
            // Fallback to reasonable default ranges if no sequence provided
            (
                vec![
                    [-5.0, -2.0], // Strong Down
                    [-2.0, -1.0], // Moderate Down
                    [-1.0, 1.0],  // Neutral
                    [1.0, 2.0],   // Moderate Up
                    [2.0, 5.0],   // Strong Up
                ],
                vec![
                    "strong_down",
                    "moderate_down",
                    "neutral",
                    "moderate_up",
                    "strong_up",
                ],
            )
        };

        for (i, (bin_name, range_pct)) in bin_names.iter().zip(bin_ranges.iter()).enumerate() {
            let price_min = current_price * (1.0 + range_pct[0] / 100.0);
            let price_max = current_price * (1.0 + range_pct[1] / 100.0);

            bins.insert(
                bin_name.to_string(),
                PriceBin {
                    range: *range_pct,
                    price: [price_min, price_max],
                    probability: probabilities[i],
                },
            );
        }

        // Find most likely range based on highest probability
        let (most_likely_idx, max_prob) = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let most_likely_range = bin_ranges[most_likely_idx];

        Ok(PriceLevelPrediction {
            bins,
            most_likely_range,
            confidence: *max_prob,
        })
    }

    /// Calculate sequence-aware ranges using the same logic as target generation
    fn calculate_sequence_aware_ranges(
        &self,
        sequence_prices: &[f64],
        current_price: f64,
        bandwidth_size: f64,
    ) -> (Vec<[f64; 2]>, Vec<&'static str>) {
        if sequence_prices.is_empty() {
            return (
                vec![
                    [-5.0, -2.0],
                    [-2.0, -1.0],
                    [-1.0, 1.0],
                    [1.0, 2.0],
                    [2.0, 5.0],
                ],
                vec![
                    "strong_down",
                    "moderate_down",
                    "neutral",
                    "moderate_up",
                    "strong_up",
                ],
            );
        }

        let sequence_min = sequence_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let sequence_max = sequence_prices
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let base_bandwidth = sequence_max - sequence_min;
        let bandwidth = base_bandwidth * bandwidth_size;

        // Handle edge case: flat sequence
        if bandwidth == 0.0 {
            return (
                vec![
                    [-1.0, -0.5],
                    [-0.5, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.5],
                    [0.5, 1.0],
                ],
                vec![
                    "strong_down",
                    "moderate_down",
                    "neutral",
                    "moderate_up",
                    "strong_up",
                ],
            );
        }

        // Calculate percentage ranges based on sequence analysis
        let min_pct = ((sequence_min - current_price) / current_price) * 100.0;
        let max_pct = ((sequence_max - current_price) / current_price) * 100.0;
        let bandwidth_pct = (bandwidth / current_price) * 100.0;

        let ranges = vec![
            [min_pct - bandwidth_pct, min_pct], // Strong Breakout Down
            [min_pct, min_pct + (max_pct - min_pct) * 0.3], // Moderate Down
            [min_pct + (max_pct - min_pct) * 0.3, max_pct], // Neutral (merged range)
            [max_pct, max_pct + bandwidth_pct * 0.5], // Moderate Up
            [max_pct + bandwidth_pct * 0.5, max_pct + bandwidth_pct], // Strong Breakout Up
        ];

        let names = vec![
            "strong_down",
            "moderate_down",
            "neutral",
            "moderate_up",
            "strong_up",
        ];

        (ranges, names)
    }

    /// Create direction prediction from DirectionOutput with 5-class system and adaptive metrics
    pub fn create_direction_prediction(
        &self,
        input: &DirectionOutput,
        training_horizon: Option<&str>,
        sequence_length: Option<u32>,
        sequence_bandwidth_percent: Option<f64>,
    ) -> Result<DirectionPrediction> {
        // Create prediction with 5-class probabilities
        let mut prediction = DirectionPrediction::from_probabilities(
            input.dump_probability,
            input.down_probability,
            input.sideways_probability,
            input.up_probability,
            input.pump_probability,
        );

        // Calculate adaptive metrics if we have the required information
        if let (Some(horizon), Some(seq_len), Some(bandwidth)) = (
            training_horizon,
            sequence_length,
            sequence_bandwidth_percent,
        ) {
            prediction.calculate_horizon_adaptive_metrics(bandwidth, horizon.to_string(), seq_len);
        } else {
            // Set default values for backward compatibility
            prediction.training_horizon = training_horizon.unwrap_or("unknown").to_string();
            prediction.sequence_length = sequence_length.unwrap_or(0);
            prediction.sequence_bandwidth_percent = sequence_bandwidth_percent.unwrap_or(0.0);
        }

        Ok(prediction)
    }

    /// Legacy method for single raw value direction prediction
    pub fn create_direction_prediction_legacy(
        &self,
        raw_output: f64,
    ) -> Result<DirectionPrediction> {
        let up_probability = (raw_output + 1.0) / 2.0;
        let down_probability = 1.0 - up_probability;

        // Convert 2-class probabilities to 5-class system based on actual calculated values
        let sideways_prob = 0.2; // Base sideways probability
        let remaining = 1.0 - sideways_prob;

        // Distribute remaining probability based on direction strength
        let dump_prob = if down_probability > 0.6 {
            (down_probability - 0.6) * remaining * 2.0 // Strong down becomes dump
        } else {
            0.0
        };
        let pump_prob = if up_probability > 0.6 {
            (up_probability - 0.6) * remaining * 2.0 // Strong up becomes pump
        } else {
            0.0
        };

        // Remaining moderate probabilities
        let down_moderate = (down_probability - dump_prob).max(0.0);
        let up_moderate = (up_probability - pump_prob).max(0.0);

        // Normalize to ensure probabilities sum to 1.0
        let total = dump_prob + down_moderate + sideways_prob + up_moderate + pump_prob;
        let norm_factor = if total > 0.0 { 1.0 / total } else { 1.0 };

        Ok(DirectionPrediction::from_probabilities(
            dump_prob * norm_factor,
            down_moderate * norm_factor,
            sideways_prob * norm_factor,
            up_moderate * norm_factor,
            pump_prob * norm_factor,
        ))
    }

    /// Create volatility prediction from VolatilityOutput with 5-class system and adaptive metrics
    pub fn create_volatility_prediction(
        &self,
        volatility_output: &VolatilityOutput,
        training_horizon: Option<&str>,
        sequence_bandwidth_percent: Option<f64>,
        current_volatility_percentile: Option<f64>,
    ) -> Result<VolatilityPrediction> {
        // Create prediction with 5-class probabilities
        let mut prediction = VolatilityPrediction::from_probabilities(
            volatility_output.very_low_probability,
            volatility_output.low_probability,
            volatility_output.medium_probability,
            volatility_output.high_probability,
            volatility_output.very_high_probability,
        );

        // Calculate adaptive metrics if we have the required information
        if let (Some(horizon), Some(bandwidth), Some(percentile)) = (
            training_horizon,
            sequence_bandwidth_percent,
            current_volatility_percentile,
        ) {
            prediction.calculate_horizon_adaptive_volatility(
                bandwidth,
                horizon.to_string(),
                percentile,
            );
        } else {
            // Set default values for backward compatibility
            prediction.training_horizon = training_horizon.unwrap_or("unknown").to_string();
            prediction.expected_range_percent = sequence_bandwidth_percent.unwrap_or(0.0);
            prediction.volatility_percentile = current_volatility_percentile.unwrap_or(50.0);
        }

        Ok(prediction)
    }

    /// Format as confidence intervals (placeholder)
    fn format_confidence_interval(
        &self,
        raw_predictions: &Array2<f64>,
        symbol: &str,
        horizon: &str,
        current_price: f64,
        targets_config: Option<&PreparedTargets>,
    ) -> Result<Vec<PredictionResult>> {
        // For now, delegate to probability distribution
        // Implement proper confidence intervals using targets_config and statistical analysis
        self.format_probability_distribution(
            raw_predictions,
            symbol,
            horizon,
            current_price,
            targets_config,
        )
    }

    /// Format as point estimates using target statistics for single-value outputs
    fn format_point_estimate(
        &self,
        raw_predictions: &Array2<f64>,
        symbol: &str,
        horizon: &str,
        current_price: f64,
        targets_config: Option<&PreparedTargets>,
    ) -> Result<Vec<PredictionResult>> {
        let mut results = Vec::new();

        for batch_idx in 0..raw_predictions.nrows() {
            let batch_predictions = raw_predictions.row(batch_idx);
            let mut result =
                PredictionResult::new(symbol.to_string(), horizon.to_string(), current_price);

            if !batch_predictions.is_empty() {
                // Calculate point estimate based on model outputs
                if batch_predictions.len() >= 3 {
                    // Multi-output model: price_level, direction, volatility
                    let price_output = batch_predictions[0];
                    let direction_output = batch_predictions[1];
                    let volatility_output = batch_predictions[2];

                    // Convert price level output to actual price estimate
                    let price_change_estimate =
                        self.convert_price_output_to_change(price_output, targets_config, horizon);

                    let estimated_price = current_price * (1.0 + price_change_estimate);

                    // Calculate confidence based on prediction certainty
                    let confidence_interval = (price_output - 0.5).abs() * 2.0; // Scale to 0-1

                    // Create single-bin price level with point estimate
                    let mut bins = std::collections::HashMap::new();
                    bins.insert(
                        "point_estimate".to_string(),
                        crate::output::structures::PriceBin {
                            range: [0.0, 0.0], // Point estimate has no range
                            price: [estimated_price, estimated_price],
                            probability: 1.0,
                        },
                    );

                    result =
                        result.with_price_levels(crate::output::structures::PriceLevelPrediction {
                            bins,
                            most_likely_range: [-1.0, 1.0], // Small range for point estimate
                            confidence: confidence_interval,
                        });

                    // Add direction prediction using new structure
                    let up_probability = self.sigmoid(direction_output);
                    let down_probability = 1.0 - up_probability;

                    // Convert to 5-class probabilities
                    let sideways_prob = 0.2;
                    let remaining = 1.0 - sideways_prob;
                    let dump_prob = if down_probability > 0.5 {
                        (down_probability - 0.5) * remaining
                    } else {
                        0.0
                    };
                    let pump_prob = if up_probability > 0.5 {
                        (up_probability - 0.5) * remaining
                    } else {
                        0.0
                    };
                    let down_moderate = down_probability - dump_prob;
                    let up_moderate = up_probability - pump_prob;

                    result = result.with_direction(DirectionPrediction::from_probabilities(
                        dump_prob,
                        down_moderate,
                        sideways_prob,
                        up_moderate,
                        pump_prob,
                    ));

                    // Add volatility prediction using new structure
                    let volatility_estimate = volatility_output.abs() * 0.1; // Scale to reasonable volatility

                    // Map volatility estimate to 5-class probabilities
                    let (very_low, low, medium, high, very_high) = if volatility_estimate < 0.02 {
                        (0.6, 0.3, 0.1, 0.0, 0.0) // Low volatility
                    } else if volatility_estimate < 0.05 {
                        (0.1, 0.2, 0.4, 0.2, 0.1) // Medium volatility
                    } else {
                        (0.0, 0.0, 0.2, 0.3, 0.5) // High volatility
                    };

                    result = result.with_volatility(VolatilityPrediction::from_probabilities(
                        very_low, low, medium, high, very_high,
                    ));
                } else {
                    // Single output - treat as price change
                    let price_change = (batch_predictions[0] - 0.5) * 0.1; // Normalize to ±5%
                    let estimated_price = current_price * (1.0 + price_change);

                    // Create simple point estimate prediction
                    let mut bins = std::collections::HashMap::new();
                    bins.insert(
                        "point_estimate".to_string(),
                        crate::output::structures::PriceBin {
                            range: [0.0, 0.0], // Point estimate has no range
                            price: [estimated_price, estimated_price],
                            probability: 1.0,
                        },
                    );

                    result =
                        result.with_price_levels(crate::output::structures::PriceLevelPrediction {
                            bins,
                            most_likely_range: [-1.0, 1.0], // Small range for point estimate
                            confidence: 0.5,                // Default confidence for single output
                        });
                };

                // Calculate confidence based on prediction certainty
                let confidence = self.calculate_point_estimate_confidence(
                    &batch_predictions,
                    targets_config,
                    horizon,
                );
                result = result.with_confidence(confidence);
            }

            results.push(result);
        }

        Ok(results)
    }

    /// Convert raw price output to price change percentage
    fn convert_price_output_to_change(
        &self,
        price_output: f64,
        targets_config: Option<&PreparedTargets>,
        horizon: &str,
    ) -> f64 {
        // Use target statistics to calibrate the output if available
        if let Some(targets) = targets_config {
            if let Some(price_targets) = targets.price_levels.get(horizon) {
                if !price_targets.is_empty() {
                    // Map model output to target distribution range
                    let min_target = *price_targets.iter().min().unwrap() as f64;
                    let max_target = *price_targets.iter().max().unwrap() as f64;

                    // Normalize price_output (assumed to be 0-1) to target range
                    let normalized_change = (price_output - 0.5) * 2.0; // Convert to -1 to 1
                    let target_range = (max_target - min_target) / 100.0; // Convert to decimal

                    return normalized_change * target_range;
                }
            }
        }

        // Default mapping: -5% to +5%
        (price_output - 0.5) * 0.1
    }

    /// Calculate confidence for point estimates
    fn calculate_point_estimate_confidence(
        &self,
        predictions: &ndarray::ArrayView1<f64>,
        targets_config: Option<&PreparedTargets>,
        horizon: &str,
    ) -> f64 {
        // Base confidence on prediction certainty and target quality
        let prediction_certainty = if predictions.len() > 1 {
            // Multi-output: average certainty across outputs
            let avg_distance_from_neutral =
                predictions.iter().map(|&x| (x - 0.5).abs()).sum::<f64>()
                    / predictions.len() as f64;
            avg_distance_from_neutral * 2.0 // Scale to 0-1
        } else if predictions.len() == 1 {
            (predictions[0] - 0.5).abs() * 2.0
        } else {
            0.0
        };

        // Adjust based on target quality if available
        let target_quality_factor = if let Some(targets) = targets_config {
            // Use horizon-specific target quality assessment
            let horizon_confidence = calculate_target_based_confidence(targets, horizon);
            let valid_ratio = targets.valid_indices.len() as f64 / targets.data_length as f64;
            (horizon_confidence + valid_ratio) / 2.0 // Combine both factors
        } else {
            0.7 // Default quality factor
        };

        (prediction_certainty * target_quality_factor).clamp(0.1, 0.95)
    }

    /// Calculate volatility percentile from sequence price data
    fn calculate_volatility_percentile(&self, sequence_prices: &[f64]) -> f64 {
        if sequence_prices.len() < 3 {
            // This should never happen if we validated properly above, but be explicit
            log::warn!(
                "Insufficient sequence data for volatility percentile calculation: {} prices",
                sequence_prices.len()
            );
            return 50.0; // Return median as last resort
        }

        // Calculate returns from the sequence
        let mut returns = Vec::new();
        for i in 1..sequence_prices.len() {
            let return_pct = (sequence_prices[i] - sequence_prices[i - 1]) / sequence_prices[i - 1];
            returns.push(return_pct.abs()); // Use absolute returns for volatility
        }

        if returns.is_empty() {
            log::warn!("No returns calculated from sequence data");
            return 50.0;
        }

        // Calculate current volatility (standard deviation of returns)
        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let current_volatility = variance.sqrt();

        // Sort returns to find percentile
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find where current volatility sits in the distribution
        let position = sorted_returns
            .iter()
            .position(|&r| r >= current_volatility)
            .unwrap_or(sorted_returns.len());

        (position as f64 / sorted_returns.len() as f64) * 100.0
    }

    /// Sigmoid activation function
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Helper function to convert raw LSTM outputs to JSON string
pub fn predictions_to_json(predictions: &[PredictionResult]) -> Result<String> {
    serde_json::to_string_pretty(predictions).map_err(|e| {
        VangaError::SerializationError(format!("Failed to serialize predictions: {}", e))
    })
}

/// Helper function to convert raw LSTM outputs to CSV string
pub fn predictions_to_csv(predictions: &[PredictionResult]) -> Result<String> {
    let mut csv = String::new();
    csv.push_str("symbol,timestamp,horizon,current_price,confidence,prediction_type,value\n");

    for pred in predictions {
        // Price levels
        if let Some(ref price_levels) = pred.price_levels {
            csv.push_str(&format!(
                "{},{},{},{:.6},{:.6},price_level_confidence,{:.6}\n",
                pred.symbol,
                pred.timestamp,
                pred.horizon,
                pred.current_price,
                pred.confidence,
                price_levels.confidence
            ));
        }

        // Direction
        if let Some(ref direction) = pred.direction {
            csv.push_str(&format!(
                "{},{},{},{:.6},{:.6},direction,{}\n",
                pred.symbol,
                pred.timestamp,
                pred.horizon,
                pred.current_price,
                pred.confidence,
                direction.prediction
            ));
            csv.push_str(&format!(
                "{},{},{},{:.6},{:.6},up_probability,{:.6}\n",
                pred.symbol,
                pred.timestamp,
                pred.horizon,
                pred.current_price,
                pred.confidence,
                direction.up_probability
            ));
        }

        // Volatility
        if let Some(ref volatility) = pred.volatility {
            csv.push_str(&format!(
                "{},{},{},{:.6},{:.6},volatility_1h,{:.6}\n",
                pred.symbol,
                pred.timestamp,
                pred.horizon,
                pred.current_price,
                pred.confidence,
                volatility.confidence
            ));
            csv.push_str(&format!(
                "{},{},{},{:.6},{:.6},volatility_regime,{}\n",
                pred.symbol,
                pred.timestamp,
                pred.horizon,
                pred.current_price,
                pred.confidence,
                volatility.regime
            ));
        }
    }

    Ok(csv)
}

/// Calculate confidence based on target distribution balance and data quality
/// Uses PreparedTargets statistics to assess prediction reliability
fn calculate_target_based_confidence(targets: &PreparedTargets, horizon: &str) -> f64 {
    let mut confidence_factors = Vec::new();

    // Check price level distribution balance
    if let Some(price_levels) = targets.price_levels.get(horizon) {
        let balance = calculate_class_balance(price_levels);
        confidence_factors.push(balance);
    }

    // Check direction distribution balance
    if let Some(directions) = targets.directions.get(horizon) {
        let balance = calculate_class_balance(directions);
        confidence_factors.push(balance);
    }

    // Check volatility regime distribution
    if let Some(volatility) = targets.volatility.get(horizon) {
        let balance = calculate_class_balance(volatility);
        confidence_factors.push(balance);
    }

    // Data quality factor based on valid indices ratio
    let data_quality = targets.valid_indices.len() as f64 / targets.data_length as f64;
    confidence_factors.push(data_quality);

    // Calculate overall confidence as weighted average
    if confidence_factors.is_empty() {
        0.5 // Neutral confidence when no data available
    } else {
        let sum: f64 = confidence_factors.iter().sum();
        (sum / confidence_factors.len() as f64).clamp(0.1, 0.95)
    }
}

/// Calculate class balance score for target distribution
/// Returns higher scores for more balanced distributions
fn calculate_class_balance(targets: &[i32]) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }

    // Count class frequencies
    let mut class_counts = std::collections::HashMap::new();
    for &target in targets {
        *class_counts.entry(target).or_insert(0) += 1;
    }

    if class_counts.len() <= 1 {
        return 0.1; // Low confidence for single class
    }

    // Calculate entropy-based balance score
    let total = targets.len() as f64;
    let mut entropy = 0.0;

    for count in class_counts.values() {
        let prob = *count as f64 / total;
        if prob > 0.0 {
            entropy -= prob * prob.log2();
        }
    }

    // Normalize entropy to [0, 1] range
    let max_entropy = (class_counts.len() as f64).log2();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}
