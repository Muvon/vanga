//! Output formatting logic for converting raw LSTM outputs to structured predictions
//!
//! This module bridges the gap between raw LSTM Array2<f64> outputs and the structured
//! JSON format specified in ARCHITECTURE.md, reusing existing target generation logic.

use crate::config::prediction::{OutputConfig, OutputFormat};
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
}

impl OutputFormatter {
    /// Create new formatter with configuration
    pub fn new(config: OutputConfig) -> Self {
        Self { config }
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
        let base_confidence = if let Some(targets) = targets_config {
            calculate_target_based_confidence(targets, horizon)
        } else {
            0.7 // Default confidence when no target statistics available
        };

        // For now, create one prediction result per batch
        for batch_idx in 0..raw_predictions.nrows() {
            let mut result =
                PredictionResult::new(symbol.to_string(), horizon.to_string(), current_price);

            // Extract predictions for this batch
            let batch_predictions = raw_predictions.row(batch_idx);

            // Convert raw outputs to structured predictions
            // Note: This is a simplified implementation - in production you'd need
            // to know which outputs correspond to which prediction heads

            if !batch_predictions.is_empty() {
                // Assume first output is price level probability
                let price_level_prob = batch_predictions[0];
                result = result.with_price_levels(
                    self.create_price_level_prediction(price_level_prob, current_price)?,
                );
            }

            if batch_predictions.len() >= 2 {
                // Assume second output is direction probability
                let direction_prob = batch_predictions[1];
                result = result.with_direction(self.create_direction_prediction(direction_prob)?);
            }

            if batch_predictions.len() >= 3 {
                // Assume third output is volatility
                let volatility_prob = batch_predictions[2];
                result =
                    result.with_volatility(self.create_volatility_prediction(volatility_prob)?);
            }

            // Apply the calculated confidence to the prediction result
            result = result.with_confidence(base_confidence);

            results.push(result);
        }

        Ok(results)
    }

    /// Create price level prediction from raw output
    /// Reuses the bin structure from ARCHITECTURE.md
    fn create_price_level_prediction(
        &self,
        raw_output: f64,
        current_price: f64,
    ) -> Result<PriceLevelPrediction> {
        // Convert single output to probability distribution across bins
        // This is a simplified implementation - in production you'd have
        // softmax outputs for each bin

        let mut bins = HashMap::new();

        // Create 7 bins as specified in ARCHITECTURE.md with actual price ranges
        let bin_configs = vec![
            ("bin_1", "< -5%", current_price * 0.95),
            ("bin_2", "-5% to -3%", current_price * 0.97),
            ("bin_3", "-3% to -1%", current_price * 0.99),
            ("bin_4", "-1% to 1%", current_price),
            ("bin_5", "1% to 3%", current_price * 1.01),
            ("bin_6", "3% to 5%", current_price * 1.03),
            ("bin_7", "> 5%", current_price * 1.05),
        ];

        // Simple distribution based on raw output
        // In production, this would be proper softmax probabilities
        let center_bin = ((raw_output + 1.0) / 2.0 * 7.0) as usize;
        let center_bin = center_bin.min(6);

        for (i, (bin_name, range, price_level)) in bin_configs.iter().enumerate() {
            let probability = if i == center_bin {
                0.4 // High probability for predicted bin
            } else if (i as i32 - center_bin as i32).abs() == 1 {
                0.2 // Medium probability for adjacent bins
            } else {
                0.1 / (bin_configs.len() - 3) as f64 // Low probability for others
            };

            bins.insert(
                bin_name.to_string(),
                PriceBin {
                    range: format!("{} (${:.2})", range, price_level), // Include actual price level
                    probability,
                },
            );
        }

        // Find most likely range using actual price calculation
        let most_likely_range = bin_configs[center_bin].1.to_string();
        let predicted_price = bin_configs[center_bin].2;
        let price_confidence = 1.0 - ((predicted_price - current_price).abs() / current_price);

        Ok(PriceLevelPrediction {
            bins,
            most_likely_range,
            confidence: price_confidence.clamp(0.0, 1.0),
        })
    }

    /// Create direction prediction from raw output
    fn create_direction_prediction(&self, raw_output: f64) -> Result<DirectionPrediction> {
        // Convert raw output to probability (assuming sigmoid-like output)
        let up_probability = (raw_output + 1.0) / 2.0; // Normalize -1,1 to 0,1
        let down_probability = 1.0 - up_probability;

        let prediction = if up_probability > 0.6 {
            "UP"
        } else if down_probability > 0.6 {
            "DOWN"
        } else {
            "SIDEWAYS"
        };

        let confidence = (up_probability - 0.5).abs() * 2.0; // Distance from neutral

        Ok(DirectionPrediction {
            up_probability,
            down_probability,
            prediction: prediction.to_string(),
            confidence,
        })
    }

    /// Create volatility prediction from raw output
    fn create_volatility_prediction(&self, raw_output: f64) -> Result<VolatilityPrediction> {
        // Convert raw output to volatility values
        let base_vol = (raw_output.abs() * 0.1).max(0.001); // Scale to reasonable volatility

        let regime = if base_vol < 0.02 {
            "LOW"
        } else if base_vol < 0.05 {
            "MEDIUM"
        } else {
            "HIGH"
        };

        Ok(VolatilityPrediction {
            expected_1h: base_vol * 0.5,
            expected_4h: base_vol,
            expected_24h: base_vol * 2.0,
            regime: regime.to_string(),
            confidence: raw_output.abs().min(1.0),
        })
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
                            range: format!("{:.2}", estimated_price),
                            probability: 1.0,
                        },
                    );

                    result =
                        result.with_price_levels(crate::output::structures::PriceLevelPrediction {
                            bins,
                            most_likely_range: format!("{:.2}", estimated_price),
                            confidence: confidence_interval,
                        });

                    // Add direction prediction
                    let up_probability = self.sigmoid(direction_output);
                    result =
                        result.with_direction(crate::output::structures::DirectionPrediction {
                            up_probability,
                            down_probability: 1.0 - up_probability,
                            prediction: if up_probability > 0.5 {
                                "UP".to_string()
                            } else {
                                "DOWN".to_string()
                            },
                            confidence: (up_probability - 0.5).abs() * 2.0,
                        });

                    // Add volatility prediction
                    let volatility_estimate = volatility_output.abs() * 0.1; // Scale to reasonable volatility
                    result =
                        result.with_volatility(crate::output::structures::VolatilityPrediction {
                            expected_1h: volatility_estimate,
                            expected_4h: volatility_estimate * 1.2,
                            expected_24h: volatility_estimate * 1.5,
                            regime: if volatility_estimate < 0.02 {
                                "LOW".to_string()
                            } else if volatility_estimate < 0.05 {
                                "MEDIUM".to_string()
                            } else {
                                "HIGH".to_string()
                            },
                            confidence: 1.0 - (volatility_estimate * 10.0).min(0.9),
                        });
                } else {
                    // Single output - treat as price change
                    let price_change = (batch_predictions[0] - 0.5) * 0.1; // Normalize to ±5%
                    let estimated_price = current_price * (1.0 + price_change);

                    // Create simple point estimate prediction
                    let mut bins = std::collections::HashMap::new();
                    bins.insert(
                        "point_estimate".to_string(),
                        crate::output::structures::PriceBin {
                            range: format!("{:.2}", estimated_price),
                            probability: 1.0,
                        },
                    );

                    result =
                        result.with_price_levels(crate::output::structures::PriceLevelPrediction {
                            bins,
                            most_likely_range: format!("{:.2}", estimated_price),
                            confidence: 0.5, // Default confidence for single output
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
                volatility.expected_1h
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
