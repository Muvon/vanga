//! Output formatting logic for converting raw LSTM outputs to structured predictions
//!
//! This module bridges the gap between raw LSTM Array2<f64> outputs and the structured
//! JSON format specified in ARCHITECTURE.md, reusing existing target generation logic.

use crate::config::model::NUM_CLASSES;
use crate::config::prediction::{OutputConfig, OutputFormat};
use crate::data::structures::MarketDataRow;
use crate::output::multi_target_parser::{DirectionOutput, MultiTargetParser, VolatilityOutput};
use crate::output::structures::{
    DirectionPrediction, PredictionResult, PriceBin, PriceLevelPrediction, VolatilityPrediction,
};
use crate::targets::PreparedTargets;
// Import reconstruction methods from target modules
use crate::targets::direction::reconstruct_direction;
use crate::targets::reconstruct_price_levels;
use crate::targets::sentiment::reconstruct_sentiment;
use crate::targets::volatility::reconstruct_volatility;
use crate::targets::volume::reconstruct_volume;
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;
use std::collections::HashMap;

/// Output formatter that converts raw LSTM predictions to structured formats
pub struct OutputFormatter {
    config: OutputConfig,
    parser: Option<MultiTargetParser>,
    sequence_ohlcv: Option<Vec<MarketDataRow>>,
    /// Model config bandwidth_size: sequence bandwidth multiplier (e.g., 1.0, 1.5, 2.0)
    /// Used in target generation as: sequence_bandwidth = (max - min) * bandwidth_size
    bandwidth_size: Option<f64>,
    /// Percentiles for price level calculation [lower, upper] (e.g., [0.1, 0.9])
    percentiles: Option<[f64; 2]>,
    /// Number of input features used for prediction
    feature_count: Option<usize>,
    /// Sequence length used for prediction
    sequence_length: Option<usize>,
    /// Calibrated target parameters for consistent reconstruction
    calibrated_parameters: Option<crate::targets::calibration::CalibratedParameters>,
}

impl OutputFormatter {
    /// Create new formatter with configuration
    pub fn new(config: OutputConfig) -> Self {
        Self {
            config,
            parser: Some(MultiTargetParser::new()), // Always initialize parser - all targets enabled with NUM_CLASSES=5
            sequence_ohlcv: None,
            bandwidth_size: None,
            percentiles: None,
            feature_count: None,
            sequence_length: None,
            calibrated_parameters: None,
        }
    }

    /// Set sequence data for sequence-aware price level calculations
    /// Set OHLCV sequence data for VWAP-based range calculation (matches training approach)
    pub fn with_sequence_ohlcv(mut self, sequence_ohlcv: Vec<MarketDataRow>) -> Self {
        self.sequence_ohlcv = Some(sequence_ohlcv);
        self
    }

    /// Set metadata for prediction results
    pub fn with_metadata(mut self, feature_count: usize, sequence_length: usize) -> Self {
        self.feature_count = Some(feature_count);
        self.sequence_length = Some(sequence_length);
        self
    }

    /// Set calibrated target parameters for consistent reconstruction
    pub fn with_calibrated_parameters(
        mut self,
        params: crate::targets::calibration::CalibratedParameters,
    ) -> Self {
        // Override training config with calibrated parameters for better accuracy
        self.bandwidth_size = Some(params.price_levels.bandwidth);
        self.percentiles = Some(params.price_levels.percentiles);
        self.calibrated_parameters = Some(params);
        self
    }

    /// Check if multi-target parser is available
    pub fn has_parser(&self) -> bool {
        self.parser.is_some()
    }

    /// Get sequence OHLCV data reference
    pub fn get_sequence_ohlcv(&self) -> Option<&[MarketDataRow]> {
        self.sequence_ohlcv.as_deref()
    }

    /// Get bandwidth size
    pub fn get_bandwidth_size(&self) -> Option<f64> {
        self.bandwidth_size
    }

    /// Get percentiles
    pub fn get_percentiles(&self) -> Option<[f64; 2]> {
        self.percentiles
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

    /// Extract horizon-specific predictions from multi-target model output
    ///
    /// Multi-target models output predictions for all targets and horizons:
    /// [price_level_16h(5), price_level_32h(5), price_level_3d(5), direction_16h(5), direction_32h(5), direction_3d(5), volatility_16h(5), volatility_32h(5), volatility_3d(5)]
    ///
    /// This method extracts only the predictions for the specified horizon:
    /// For "16h": [price_level_16h(5), direction_16h(5), volatility_16h(5)] = 15 outputs
    /// For "32h": [price_level_32h(5), direction_32h(5), volatility_32h(5)] = 15 outputs
    /// For "3d":  [price_level_3d(5), direction_3d(5), volatility_3d(5)] = 15 outputs
    fn extract_horizon_predictions(
        &self,
        raw_predictions: ndarray::ArrayView1<f64>,
        horizon: &str,
        target_names: &[String],
    ) -> Result<Vec<f64>> {
        // Find indices of models that match the requested horizon
        let mut horizon_indices = Vec::new();

        for (i, target_name) in target_names.iter().enumerate() {
            if target_name.ends_with(&format!("_{}", horizon)) {
                // Each model outputs NUM_CLASSES predictions
                let start_idx = i * NUM_CLASSES;
                let end_idx = start_idx + NUM_CLASSES;

                // Collect indices for this target's predictions
                for idx in start_idx..end_idx {
                    horizon_indices.push(idx);
                }

                log::debug!(
                    "Target '{}' matches horizon '{}': indices [{}, {})",
                    target_name,
                    horizon,
                    start_idx,
                    end_idx - 1
                );
            }
        }

        if horizon_indices.is_empty() {
            return Err(VangaError::PredictionError(format!(
                "No models found for horizon '{}'. Available targets: {:?}",
                horizon, target_names
            )));
        }

        // Validate indices are within bounds
        let max_idx = horizon_indices.iter().max().unwrap_or(&0);
        if *max_idx >= raw_predictions.len() {
            return Err(VangaError::PredictionError(format!(
                "Prediction index {} out of bounds for array of length {}. Target names: {:?}",
                max_idx,
                raw_predictions.len(),
                target_names
            )));
        }

        // Extract the predictions for this horizon
        let mut horizon_predictions = Vec::with_capacity(horizon_indices.len());
        for &idx in &horizon_indices {
            horizon_predictions.push(raw_predictions[idx]);
        }

        log::debug!(
            "Extracted {} predictions for horizon '{}' from {} total predictions",
            horizon_predictions.len(),
            horizon,
            raw_predictions.len()
        );

        // Verify we have exactly 25 predictions (5 targets × 5 classes)
        let expected_count = 5 * NUM_CLASSES; // 5 targets (price_level, direction, volatility, sentiment, volume) × 5 classes
        if horizon_predictions.len() != expected_count {
            return Err(VangaError::PredictionError(format!(
                "Expected {} predictions for horizon '{}' (5 targets × {} classes), but got {}. Indices: {:?}",
                expected_count, horizon, NUM_CLASSES, horizon_predictions.len(), horizon_indices
            )));
        }

        Ok(horizon_predictions)
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
        // For backward compatibility, call the new method with None for target_names
        self.format_predictions_with_targets(
            raw_predictions,
            symbol,
            horizon,
            current_price,
            targets_config,
            None,
        )
    }

    /// Format predictions with optional target names for horizon-specific extraction
    pub fn format_predictions_with_targets(
        &self,
        raw_predictions: &Array2<f64>,
        symbol: &str,
        horizon: &str,
        current_price: f64,
        targets_config: Option<&PreparedTargets>,
        target_names: Option<&[String]>,
    ) -> Result<Vec<PredictionResult>> {
        match self.config.format {
            OutputFormat::ProbabilityDistribution => self
                .format_probability_distribution_with_targets(
                    raw_predictions,
                    symbol,
                    horizon,
                    current_price,
                    targets_config,
                    target_names,
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
                self.format_probability_distribution_with_targets(
                    raw_predictions,
                    symbol,
                    horizon,
                    current_price,
                    targets_config,
                    target_names,
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
        // For backward compatibility, call the new method with None for target_names
        self.format_probability_distribution_with_targets(
            raw_predictions,
            symbol,
            horizon,
            current_price,
            targets_config,
            None,
        )
    }

    /// Format as probability distribution with optional target names for horizon-specific extraction
    fn format_probability_distribution_with_targets(
        &self,
        raw_predictions: &Array2<f64>,
        symbol: &str,
        horizon: &str,
        current_price: f64,
        targets_config: Option<&PreparedTargets>,
        target_names: Option<&[String]>,
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
                "MultiTargetParser not configured. This should not happen with unified targets."
                    .to_string(),
            )
        })?;

        // For each batch in the predictions
        for batch_idx in 0..raw_predictions.nrows() {
            // Get actual feature count and sequence length from the prediction data
            let feature_count = self.feature_count.unwrap_or(raw_predictions.ncols());
            let sequence_length = self.sequence_length.unwrap_or(60); // Default LSTM sequence length

            let mut result = PredictionResult::new_with_metadata(
                symbol.to_string(),
                horizon.to_string(),
                current_price,
                feature_count,
                sequence_length,
            );

            // Calculate sequence VWAP using the same method as training
            if let Some(ohlcv_data) = &self.sequence_ohlcv {
                use crate::targets::get_sequence_exponential_weighted_close;
                match get_sequence_exponential_weighted_close(ohlcv_data) {
                    Ok(sequence_vwap) => {
                        result.current_vwap_price = sequence_vwap;
                        log::debug!("Calculated sequence VWAP: {:.2}", sequence_vwap);
                    }
                    Err(e) => {
                        log::warn!("Failed to calculate sequence VWAP: {}", e);
                        result.current_vwap_price = current_price; // Fallback to current price
                    }
                }
            } else {
                log::warn!("No sequence OHLCV data available for VWAP calculation");
                result.current_vwap_price = current_price; // Fallback to current price
            }

            // Extract predictions for this batch
            let batch_predictions = raw_predictions.row(batch_idx);

            // Extract horizon-specific predictions if target names are provided (multi-target model)
            let horizon_predictions =
                if let Some(target_names) = target_names {
                    // Multi-target model: extract only predictions for this horizon
                    let extracted =
                        self.extract_horizon_predictions(batch_predictions, horizon, target_names)?;
                    log::debug!(
                    "Extracted {} horizon-specific predictions for '{}' from {} total predictions",
                    extracted.len(), horizon, batch_predictions.len()
                );
                    extracted
                } else {
                    // Single-target model or backward compatibility: use all predictions
                    batch_predictions.to_vec()
                };

            // Convert to ArrayView1 for parser
            let horizon_array = ndarray::Array1::from_vec(horizon_predictions);
            let horizon_view = horizon_array.view();

            // Parse the horizon-specific predictions using the multi-target parser
            let parsed_output = parser.parse_output(horizon_view)?;

            // Convert parsed output to structured predictions
            if let Some(price_level_probs) = parsed_output.price_levels {
                result = result.with_price_levels(self.create_price_level_prediction(
                    &price_level_probs,
                    current_price,
                    self.bandwidth_size,
                    self.percentiles,
                )?);
            }

            if let Some(direction_output) = parsed_output.direction {
                // Calculate sequence bandwidth percentage using OHLCV data with percentiles (matches training)
                let sequence_bandwidth_percent = if let Some(ohlcv_data) = &self.sequence_ohlcv {
                    // Calculate VWAP-weighted prices for sequence (matches training approach)
                    let mut sequence_vwap_prices = Vec::new();
                    for candle in ohlcv_data {
                        let vwap_price = if candle.volume > 0.0 {
                            // Use volume-weighted OHLC4 for this candle
                            (candle.open + candle.high + candle.low + candle.close) / 4.0
                        } else {
                            // Fallback to simple OHLC4 if no volume
                            (candle.open + candle.high + candle.low + candle.close) / 4.0
                        };
                        sequence_vwap_prices.push(vwap_price);
                    }

                    if sequence_vwap_prices.len() >= 2 {
                        // Use percentile boundaries (matches training approach)
                        let percentiles = self.percentiles.unwrap_or([0.1, 0.9]);
                        let mut sorted_prices = sequence_vwap_prices.clone();
                        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let n = sorted_prices.len();
                        let lower_idx = ((n as f64 * percentiles[0]) as usize).min(n - 1);
                        let upper_idx = ((n as f64 * percentiles[1]) as usize).min(n - 1);

                        let sequence_min = sorted_prices[lower_idx];
                        let sequence_max = sorted_prices[upper_idx];

                        // Use model config bandwidth_size (sequence bandwidth multiplier)
                        let model_bandwidth_multiplier = self.bandwidth_size.ok_or_else(|| {
                            VangaError::PredictionError(
                                "Model bandwidth_size not configured. This is required for adaptive predictions.".to_string()
                            )
                        })?;

                        // Calculate bandwidth as percentage of current price (using percentile range)
                        let sequence_price_range = sequence_max - sequence_min;
                        let sequence_range_percent = (sequence_price_range / current_price) * 100.0;

                        // Cap the sequence range to reasonable crypto values (max 50%)
                        let capped_sequence_range_percent = sequence_range_percent.min(50.0);

                        // Apply model bandwidth multiplier to the percentage range
                        let final_bandwidth_percent =
                            capped_sequence_range_percent * model_bandwidth_multiplier;

                        // Cap final result to prevent extreme values (max 100%)
                        final_bandwidth_percent.min(100.0)
                    } else {
                        return Err(VangaError::PredictionError(format!(
                            "Insufficient OHLCV data for adaptive predictions: {} rows (need ≥2)",
                            ohlcv_data.len()
                        )));
                    }
                } else {
                    return Err(VangaError::PredictionError(
                        "OHLCV sequence data not available. Use with_sequence_ohlcv() to provide it.".to_string()
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
                // Calculate sequence bandwidth percentage using OHLCV data with percentiles (matches training)
                let sequence_bandwidth_percent = if let Some(ohlcv_data) = &self.sequence_ohlcv {
                    // Calculate VWAP-weighted prices for sequence (matches training approach)
                    let mut sequence_vwap_prices = Vec::new();
                    for candle in ohlcv_data {
                        let vwap_price = if candle.volume > 0.0 {
                            // Use volume-weighted OHLC4 for this candle
                            (candle.open + candle.high + candle.low + candle.close) / 4.0
                        } else {
                            // Fallback to simple OHLC4 if no volume
                            (candle.open + candle.high + candle.low + candle.close) / 4.0
                        };
                        sequence_vwap_prices.push(vwap_price);
                    }

                    if sequence_vwap_prices.len() >= 2 {
                        // Use percentile boundaries (matches training approach)
                        let percentiles = self.percentiles.unwrap_or([0.1, 0.9]);
                        let mut sorted_prices = sequence_vwap_prices.clone();
                        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let n = sorted_prices.len();
                        let lower_idx = ((n as f64 * percentiles[0]) as usize).min(n - 1);
                        let upper_idx = ((n as f64 * percentiles[1]) as usize).min(n - 1);

                        let sequence_min = sorted_prices[lower_idx];
                        let sequence_max = sorted_prices[upper_idx];

                        // Use model config bandwidth_size (sequence bandwidth multiplier)
                        let model_bandwidth_multiplier = self.bandwidth_size.ok_or_else(|| {
                            VangaError::PredictionError(
                                "Model bandwidth_size not configured. This is required for adaptive predictions.".to_string()
                            )
                        })?;

                        // Calculate bandwidth as percentage of current price (using percentile range)
                        let sequence_price_range = sequence_max - sequence_min;
                        let sequence_range_percent = (sequence_price_range / current_price) * 100.0;

                        // Cap the sequence range to reasonable crypto values (max 50%)
                        let capped_sequence_range_percent = sequence_range_percent.min(50.0);

                        // Apply model bandwidth multiplier to the percentage range
                        let final_bandwidth_percent =
                            capped_sequence_range_percent * model_bandwidth_multiplier;

                        // Cap final result to prevent extreme values (max 100%)
                        final_bandwidth_percent.min(100.0)
                    } else {
                        return Err(VangaError::PredictionError(format!(
                            "Insufficient OHLCV data for adaptive predictions: {} rows (need ≥2)",
                            ohlcv_data.len()
                        )));
                    }
                } else {
                    return Err(VangaError::PredictionError(
                        "OHLCV sequence data not available. Use with_sequence_ohlcv() to provide it.".to_string()
                    ));
                };

                // Calculate volatility percentile from OHLCV close prices
                let sequence_prices_for_volatility: Vec<f64> = self
                    .sequence_ohlcv
                    .as_ref()
                    .unwrap() // Safe unwrap - already checked above
                    .iter()
                    .map(|row| row.close)
                    .collect();

                let volatility_percentile =
                    self.calculate_volatility_percentile(&sequence_prices_for_volatility);

                result = result.with_volatility(self.create_volatility_prediction(
                    &volatility_output,
                    Some(horizon),
                    Some(sequence_bandwidth_percent),
                    Some(volatility_percentile),
                )?);
            }

            if let Some(sentiment_output) = parsed_output.sentiment {
                result = result.with_sentiment(
                    self.create_sentiment_prediction(&sentiment_output, Some(horizon))?,
                );
            }

            if let Some(volume_output) = parsed_output.volume {
                result = result
                    .with_volume(self.create_volume_prediction(&volume_output, Some(horizon))?);
            }

            // Generate trading orders if we have all required predictions
            if result.price_levels.is_some()
                && result.direction.is_some()
                && result.volatility.is_some()
            {
                // Clone the predictions to avoid borrow checker issues
                let price_levels = result.price_levels.clone().unwrap();
                let direction = result.direction.clone().unwrap();
                let volatility = result.volatility.clone().unwrap();

                // Calculate ATR from OHLCV sequence data
                let atr_value = if let Some(ohlcv_data) = &self.sequence_ohlcv {
                    if ohlcv_data.len() >= 2 {
                        // Calculate true range for each period
                        let mut true_ranges = Vec::new();
                        for i in 1..ohlcv_data.len() {
                            let current = &ohlcv_data[i];
                            let previous = &ohlcv_data[i - 1];

                            // True Range = max(high - low, |high - prev_close|, |low - prev_close|)
                            let tr = (current.high - current.low)
                                .max((current.high - previous.close).abs())
                                .max((current.low - previous.close).abs());

                            // Convert to percentage
                            let tr_pct = (tr / current.close) * 100.0;
                            true_ranges.push(tr_pct);
                        }

                        // Average True Range as percentage
                        let atr_pct = if !true_ranges.is_empty() {
                            true_ranges.iter().sum::<f64>() / true_ranges.len() as f64
                        } else {
                            2.0
                        };

                        atr_pct.min(10.0) // Cap at 10% for sanity
                    } else {
                        2.0 // 2% fallback ATR
                    }
                } else {
                    2.0 // 2% fallback ATR
                };

                // Get bandwidth_size from training config
                let bandwidth_size = self.bandwidth_size.unwrap_or(1.0);

                // Generate sequence-aware orders - OHLCV data is REQUIRED!
                let ohlcv_data = self.sequence_ohlcv.as_ref()
                    .ok_or_else(|| VangaError::PredictionError(
                        "FATAL: No OHLCV sequence data available for order generation. This should have been set during formatter initialization.".to_string()
                    ))?;

                // Extract close prices for order generation (backward compatibility)
                let sequence_prices: Vec<f64> = ohlcv_data.iter().map(|row| row.close).collect();

                let order_config = crate::output::structures::OrderConfig::default();

                let config = crate::output::structures::SequenceAwareOrderConfig {
                    current_price,
                    direction_pred: &direction,
                    volatility_pred: &volatility,
                    price_levels: &price_levels,
                    atr_value,
                    config: &order_config,
                    sequence_prices: &sequence_prices,
                    bandwidth_size,
                };

                let orders = match crate::output::structures::TradingOrders::generate(config) {
                    Ok(orders) => {
                        log::info!(
                            "✅ Generated {} trading orders with {:.1}% directional edge",
                            orders.direction,
                            (direction.up_probability_aggregated
                                - direction.down_probability_aggregated)
                                * 100.0
                        );
                        orders
                    }
                    Err(e) => {
                        log::error!("❌ Failed to generate sequence-aware orders: {}", e);
                        return Err(e);
                    }
                };

                result = result.with_orders(orders);
            }

            // Apply the calculated confidence to the prediction result
            result = result.with_confidence(base_confidence);

            results.push(result);
        }

        Ok(results)
    }

    /// Create price level prediction from 5-class probabilities using enhanced reconstruction
    pub fn create_price_level_prediction(
        &self,
        probabilities: &[f64],
        current_price: f64,
        _bandwidth_size: Option<f64>, // Kept for API compatibility but unused (we use stored config)
        _percentiles: Option<[f64; 2]>, // Kept for API compatibility but unused (we use stored config)
    ) -> Result<PriceLevelPrediction> {
        if probabilities.len() != NUM_CLASSES {
            return Err(VangaError::PredictionError(format!(
                "Expected {} price level probabilities, got {}",
                NUM_CLASSES,
                probabilities.len()
            )));
        }

        // Get sequence OHLCV data (required for reconstruction)
        let sequence_ohlcv = self.sequence_ohlcv.as_ref().ok_or_else(|| {
            VangaError::PredictionError(
                "OHLCV sequence data not available. Use with_sequence_ohlcv() to provide it."
                    .to_string(),
            )
        })?;

        // Use calibrated parameters if available, otherwise return error
        let reconstruction = if let Some(ref calibrated_params) = self.calibrated_parameters {
            reconstruct_price_levels(
                probabilities,
                sequence_ohlcv,
                current_price,
                &calibrated_params.price_levels,
            )?
        } else {
            return Err(VangaError::ConfigError(
                "Adaptive parameters required for price level reconstruction".to_string(),
            ));
        };

        // Create bins using reconstruction results
        let mut bins = HashMap::new();
        let bin_names = [
            "strong_down",
            "moderate_down",
            "neutral",
            "moderate_up",
            "strong_up",
        ];

        for (i, bin_name) in bin_names.iter().enumerate() {
            bins.insert(
                bin_name.to_string(),
                PriceBin {
                    range: reconstruction.percentage_ranges[i],
                    vwap_range: reconstruction.exponential_weighted_percentage_ranges[i],
                    price: reconstruction.price_ranges[i],
                    probability: reconstruction.probabilities[i],
                },
            );
        }

        // Use reconstruction results for most likely range
        let most_likely_range = reconstruction.percentage_ranges[reconstruction.most_likely_class];

        Ok(PriceLevelPrediction {
            bins,
            most_likely_range,
            confidence: reconstruction.confidence,
        })
    }

    /// Create direction prediction from DirectionOutput with enhanced reconstruction
    /// Create direction prediction from DirectionOutput with enhanced reconstruction
    pub fn create_direction_prediction(
        &self,
        input: &DirectionOutput,
        training_horizon: Option<&str>,
        sequence_length: Option<u32>,
        sequence_bandwidth_percent: Option<f64>,
    ) -> Result<DirectionPrediction> {
        // Get sequence OHLCV data for reconstruction
        let sequence_ohlcv = self.sequence_ohlcv.as_ref();

        // Create base prediction with 5-class probabilities
        let mut prediction = DirectionPrediction::from_probabilities(
            input.dump_probability,
            input.down_probability,
            input.sideways_probability,
            input.up_probability,
            input.pump_probability,
        );

        // Enhance with reconstruction if sequence data is available
        if let Some(ohlcv_data) = sequence_ohlcv {
            let probabilities = vec![
                input.dump_probability,
                input.down_probability,
                input.sideways_probability,
                input.up_probability,
                input.pump_probability,
            ];

            // Use enhanced reconstruction from direction module with calibrated parameters
            let reconstruction_result =
                if let Some(ref calibrated_params) = self.calibrated_parameters {
                    reconstruct_direction(&probabilities, ohlcv_data, &calibrated_params.direction)
                } else {
                    Err(VangaError::ConfigError(
                        "Adaptive parameters required for direction reconstruction".to_string(),
                    ))
                };

            match reconstruction_result {
                Ok(reconstruction) => {
                    // Update existing fields with reconstruction results
                    prediction.breakout_probability = reconstruction.breakout_probability;

                    // Use reconstruction data to enhance existing calculations
                    let enhanced_upside = reconstruction.expected_trend_acceleration.max(0.0);
                    let enhanced_downside = (-reconstruction.expected_trend_acceleration).max(0.0);

                    if enhanced_downside > 0.0 {
                        prediction.risk_reward_ratio = enhanced_upside / enhanced_downside;
                    }

                    log::debug!(
                        "🎯 Direction reconstruction: momentum_change={:.4}, trend_accel={:.2}%, breakout_prob={:.3}",
                        reconstruction.expected_momentum_change,
                        reconstruction.expected_trend_acceleration,
                        reconstruction.breakout_probability
                    );
                }
                Err(e) => {
                    log::warn!("Direction reconstruction failed: {}, using fallback", e);
                }
            }
        }

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

    /// Create volatility prediction from VolatilityOutput with enhanced reconstruction
    pub fn create_volatility_prediction(
        &self,
        volatility_output: &VolatilityOutput,
        training_horizon: Option<&str>,
        sequence_bandwidth_percent: Option<f64>,
        current_volatility_percentile: Option<f64>,
    ) -> Result<VolatilityPrediction> {
        // Get sequence OHLCV data for reconstruction
        let sequence_ohlcv = self.sequence_ohlcv.as_ref();

        // Create base prediction with 5-class probabilities
        let mut prediction = VolatilityPrediction::from_probabilities(
            volatility_output.very_low_probability,
            volatility_output.low_probability,
            volatility_output.medium_probability,
            volatility_output.high_probability,
            volatility_output.very_high_probability,
        );

        // Enhance with reconstruction if sequence data is available
        if let Some(ohlcv_data) = sequence_ohlcv {
            let probabilities = vec![
                volatility_output.very_low_probability,
                volatility_output.low_probability,
                volatility_output.medium_probability,
                volatility_output.high_probability,
                volatility_output.very_high_probability,
            ];

            // Use enhanced reconstruction from volatility module with calibrated parameters
            let volatility_result = if let Some(ref calibrated_params) = self.calibrated_parameters
            {
                // Use calibrated parameters for volatility reconstruction
                reconstruct_volatility(&probabilities, ohlcv_data, &calibrated_params.volatility)
            } else {
                // Calibrated parameters are required for reconstruction
                Err(VangaError::ConfigError(
                    "Calibrated parameters required for volatility reconstruction - model needs recalibration".to_string()
                ))
            };

            match volatility_result {
                Ok(reconstruction) => {
                    // Update existing fields with reconstruction results
                    // Use ATR ratio to enhance expected range calculation
                    if let Some(bandwidth) = sequence_bandwidth_percent {
                        prediction.expected_range_percent =
                            bandwidth * reconstruction.expected_atr_ratio;
                    }

                    // Expected ATR ratio is already in reconstruction
                    // Clamp to reasonable range (0.1% to 100% expected range)
                    prediction.expected_range_percent =
                        prediction.expected_range_percent.clamp(0.001, 1.0);

                    log::debug!(
                        "🎯 Volatility reconstruction: atr_ratio={:.3}, vol_change={:.2}%, extreme_prob={:.3}",
                        reconstruction.expected_atr_ratio,
                        reconstruction.expected_volatility_change,
                        reconstruction.extreme_volatility_probability
                    );
                }
                Err(e) => {
                    log::warn!("Volatility reconstruction failed: {}, using fallback", e);
                }
            }
        }

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

    /// Create sentiment prediction from parsed output with reconstruction
    fn create_sentiment_prediction(
        &self,
        sentiment_output: &crate::output::multi_target_parser::SentimentOutput,
        training_horizon: Option<&str>,
    ) -> Result<crate::output::prediction_types::SentimentPrediction> {
        let mut prediction =
            crate::output::prediction_types::SentimentPrediction::from_probabilities(
                sentiment_output.very_bearish_probability,
                sentiment_output.bearish_probability,
                sentiment_output.neutral_probability,
                sentiment_output.bullish_probability,
                sentiment_output.very_bullish_probability,
            );

        // Set training horizon
        prediction.training_horizon = training_horizon.unwrap_or("unknown").to_string();

        // Enhanced reconstruction using calibrated parameters if available
        if let Some(sequence_ohlcv) = &self.sequence_ohlcv {
            if self.calibrated_parameters.is_some() {
                // Prepare probabilities array for reconstruction
                // Prepare probabilities array for reconstruction
                let probabilities = vec![
                    sentiment_output.very_bearish_probability,
                    sentiment_output.bearish_probability,
                    sentiment_output.neutral_probability,
                    sentiment_output.bullish_probability,
                    sentiment_output.very_bullish_probability,
                ];

                // Call reconstruction function with calibrated parameters
                match reconstruct_sentiment(
                    &probabilities,
                    sequence_ohlcv,
                    &self.calibrated_parameters.as_ref().unwrap().sentiment,
                ) {
                    Ok(reconstruction) => {
                        // Use reconstruction results to enhance prediction
                        // The reconstruction provides richer information than basic probabilities
                        log::debug!(
                        "🎯 Sentiment reconstruction: expected={:.4}, confidence={:.3}, interpretation={}",
                        reconstruction.expected_sentiment,
                        reconstruction.confidence,
                        reconstruction.sentiment_interpretation
                    );

                        // Update confidence with reconstruction confidence
                        prediction.confidence = reconstruction.confidence;
                    }
                    Err(e) => {
                        log::warn!("Failed to reconstruct sentiment: {}", e);
                        // Fall back to basic prediction without reconstruction
                    }
                }
            }
        } else {
            log::debug!(
                "Sentiment reconstruction skipped: calibrated_parameters={}, sequence_ohlcv={}",
                self.calibrated_parameters.is_some(),
                self.sequence_ohlcv.is_some()
            );
        }

        Ok(prediction)
    }

    /// Create volume prediction from parsed output with reconstruction
    fn create_volume_prediction(
        &self,
        volume_output: &crate::output::multi_target_parser::VolumeOutput,
        training_horizon: Option<&str>,
    ) -> Result<crate::output::prediction_types::VolumePrediction> {
        let mut prediction = crate::output::prediction_types::VolumePrediction::from_probabilities(
            volume_output.very_low_probability,
            volume_output.low_probability,
            volume_output.medium_probability,
            volume_output.high_probability,
            volume_output.very_high_probability,
        );

        // Set training horizon
        prediction.training_horizon = training_horizon.unwrap_or("unknown").to_string();

        // Enhanced reconstruction using calibrated parameters if available
        if let Some(sequence_ohlcv) = &self.sequence_ohlcv {
            if self.calibrated_parameters.is_some() {
                // Calculate sequence volume (average volume from OHLCV data)
                let sequence_volume = if !sequence_ohlcv.is_empty() {
                    sequence_ohlcv
                        .iter()
                        .map(|candle| candle.volume)
                        .sum::<f64>()
                        / sequence_ohlcv.len() as f64
                } else {
                    1000.0 // Default fallback
                };

                // Prepare probabilities array for reconstruction
                let probabilities = vec![
                    volume_output.very_low_probability,
                    volume_output.low_probability,
                    volume_output.medium_probability,
                    volume_output.high_probability,
                    volume_output.very_high_probability,
                ];

                // Call reconstruction function with calibrated parameters
                match reconstruct_volume(
                    &probabilities,
                    sequence_volume,
                    &self.calibrated_parameters.as_ref().unwrap().volume,
                ) {
                    Ok(reconstruction) => {
                        // Use reconstruction results to enhance prediction
                        // The reconstruction provides richer information than basic probabilities
                        log::debug!(
                        "🎯 Volume reconstruction: expected_ratio={:.4}, confidence={:.3}, interpretation={}",
                        reconstruction.expected_volume_ratio,
                        reconstruction.confidence,
                        reconstruction.volume_interpretation
                    );

                        // Update confidence with reconstruction confidence
                        prediction.confidence = reconstruction.confidence;
                    }
                    Err(e) => {
                        log::warn!("Failed to reconstruct volume: {}", e);
                        // Fall back to basic prediction without reconstruction
                    }
                }
            }
        } else {
            log::debug!(
                "Volume reconstruction skipped: adaptive_parameters={}, sequence_ohlcv={}",
                self.calibrated_parameters.is_some(),
                self.sequence_ohlcv.is_some()
            );
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
                            range: [0.0, 0.0],      // Point estimate has no range
                            vwap_range: [0.0, 0.0], // Point estimate has no VWAP range
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
                            range: [0.0, 0.0],      // Point estimate has no range
                            vwap_range: [0.0, 0.0], // Point estimate has no VWAP range
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
            0.9 // Increased from 0.7 to 0.9 for better model uncertainty confidence
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
    if let Some(directions) = targets.direction.get(horizon) {
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

impl OutputFormatter {
    /// Calculate percentage-based ATR from sequence OHLC data
    /// Uses the same sequence length as training for consistency
    /// Returns ATR as percentage of current price for crypto-appropriate scaling
    pub fn calculate_atr_from_sequence(
        &self,
        ohlc_data: &[crate::data::structures::MarketDataRow],
    ) -> Result<f64> {
        if ohlc_data.len() < 2 {
            return Err(VangaError::PredictionError(
                "Insufficient OHLC data for ATR calculation (need at least 2 periods)".to_string(),
            ));
        }

        let current_price = ohlc_data.last().unwrap().close;
        if current_price <= 0.0 {
            return Err(VangaError::PredictionError(
                "Invalid current price for ATR calculation".to_string(),
            ));
        }

        let mut true_ranges_pct = Vec::with_capacity(ohlc_data.len());

        // First period: TR = (high - low) / close as percentage
        let first_tr_pct = ((ohlc_data[0].high - ohlc_data[0].low) / ohlc_data[0].close) * 100.0;
        true_ranges_pct.push(first_tr_pct);

        // Calculate True Range as percentage for each subsequent period
        for i in 1..ohlc_data.len() {
            let current = &ohlc_data[i];
            let previous_close = ohlc_data[i - 1].close;

            let tr1 = (current.high - current.low) / current.close;
            let tr2 = ((current.high - previous_close).abs()) / current.close;
            let tr3 = ((current.low - previous_close).abs()) / current.close;

            let true_range_pct = (tr1.max(tr2).max(tr3)) * 100.0;
            true_ranges_pct.push(true_range_pct);
        }

        // Calculate ATR as percentage (simple moving average)
        let atr_pct = true_ranges_pct.iter().sum::<f64>() / true_ranges_pct.len() as f64;

        log::debug!(
            "Calculated percentage ATR from {} periods: {:.2}%",
            ohlc_data.len(),
            atr_pct
        );
        Ok(atr_pct)
    }

    /// Apply volatility-based ATR adjustment
    /// High volatility increases ATR for wider stops, low volatility decreases it
    pub fn adjust_atr_for_volatility(
        &self,
        base_atr_pct: f64,
        volatility_pred: &crate::output::structures::VolatilityPrediction,
    ) -> f64 {
        // Calculate volatility multiplier based on regime probabilities
        let volatility_multiplier = match volatility_pred.regime.as_str() {
            "VERY_LOW" => 0.7, // Tighter stops in low volatility
            "LOW" => 0.85,
            "MEDIUM" => 1.0, // Base case
            "HIGH" => 1.3,   // Wider stops in high volatility
            "VERY_HIGH" => 1.6,
            _ => 1.0,
        };

        let adjusted_atr = base_atr_pct * volatility_multiplier;

        log::debug!(
            "ATR adjustment: base={:.2}%, regime={}, multiplier={:.2}, adjusted={:.2}%",
            base_atr_pct,
            volatility_pred.regime,
            volatility_multiplier,
            adjusted_atr
        );

        adjusted_atr
    }
}
