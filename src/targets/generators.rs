//! Target generator implementations
//!
//! This module contains the actual target generator implementations
//! that implement the TargetGenerator trait for each target type.

use crate::config::model::TargetsConfig;
use crate::targets::adaptive_parameters::{
    DirectionAdaptiveParams, PriceLevelAdaptiveParams, SentimentAdaptiveParams,
    VolatilityAdaptiveParams, VolumeAdaptiveParams,
};
use crate::targets::interface::{AdaptiveParameters, TargetGenerator};
use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Price Level Target Generator
pub struct PriceLevelTargetGenerator;

impl TargetGenerator for PriceLevelTargetGenerator {
    fn target_type(&self) -> &'static str {
        "price_levels"
    }

    fn target_name(&self) -> &'static str {
        "Price Levels"
    }

    fn class_names(&self) -> Vec<&'static str> {
        vec![
            "Strong Down",
            "Moderate Down",
            "Neutral",
            "Moderate Up",
            "Strong Up",
        ]
    }

    fn generate_targets(
        &self,
        df: &DataFrame,
        horizons: &[String],
        targets_config: &TargetsConfig,
        sequence_indices: &[usize],
        sequence_length: usize,
        adaptive_params: Option<&dyn AdaptiveParameters>,
    ) -> Result<HashMap<String, Vec<i32>>> {
        let params =
            adaptive_params.and_then(|p| p.as_any().downcast_ref::<PriceLevelAdaptiveParams>());
        crate::targets::generate_price_level_targets_with_adaptive_params(
            df,
            horizons,
            targets_config,
            sequence_indices,
            sequence_length,
            params,
        )
    }

    fn calibrate_parameters(
        &self,
        df: &DataFrame,
        sequence_length: usize,
        horizon_steps: usize,
        targets_config: &TargetsConfig,
    ) -> Result<Box<dyn AdaptiveParameters>> {
        // Delegate to the existing calibration system to preserve original logic
        // This ensures the original grid search and MIN-CLASS optimization is used

        // Extract OHLCV data for calibration
        let ohlcv_data = crate::utils::market_data::extract_ohlcv_data(df)?;

        // Calculate sequence indices for calibration (same logic as original)
        let data_length = ohlcv_data.len();
        let max_horizon_steps = horizon_steps;
        let step_size = 1;

        let sequence_indices = crate::utils::sequence_utils::calculate_sequence_indices(
            data_length,
            sequence_length,
            step_size,
            max_horizon_steps,
        )?;

        // Create calibrator with same parameters as original
        let calibrator = crate::targets::adaptive_parameters::AdaptiveParameterCalibrator::new(
            targets_config.clone(),
        );

        // Use async runtime to call the calibration
        let runtime = tokio::runtime::Runtime::new()?;
        let params = runtime.block_on(calibrator.calibrate_price_levels(
            &ohlcv_data,
            sequence_length,
            horizon_steps,
            &sequence_indices,
        ))?;

        Ok(Box::new(params))
    }
}

/// Direction Target Generator
pub struct DirectionTargetGenerator;

impl TargetGenerator for DirectionTargetGenerator {
    fn target_type(&self) -> &'static str {
        "direction"
    }

    fn target_name(&self) -> &'static str {
        "Direction"
    }

    fn class_names(&self) -> Vec<&'static str> {
        vec!["DUMP", "DOWN", "SIDEWAYS", "UP", "PUMP"]
    }

    fn generate_targets(
        &self,
        df: &DataFrame,
        horizons: &[String],
        targets_config: &TargetsConfig,
        sequence_indices: &[usize],
        sequence_length: usize,
        adaptive_params: Option<&dyn AdaptiveParameters>,
    ) -> Result<HashMap<String, Vec<i32>>> {
        let params =
            adaptive_params.and_then(|p| p.as_any().downcast_ref::<DirectionAdaptiveParams>());
        crate::targets::generate_direction_targets_with_adaptive_params(
            df,
            horizons,
            targets_config,
            sequence_indices,
            sequence_length,
            params,
        )
    }

    fn calibrate_parameters(
        &self,
        df: &DataFrame,
        sequence_length: usize,
        horizon_steps: usize,
        targets_config: &TargetsConfig,
    ) -> Result<Box<dyn AdaptiveParameters>> {
        // Delegate to the existing calibration system to preserve original logic
        // This ensures the original grid search and MIN-CLASS optimization is used

        // Extract OHLCV data for calibration
        let ohlcv_data = crate::utils::market_data::extract_ohlcv_data(df)?;

        // Calculate sequence indices for calibration (same logic as original)
        let data_length = ohlcv_data.len();
        let max_horizon_steps = horizon_steps;
        let step_size = 1;

        let sequence_indices = crate::utils::sequence_utils::calculate_sequence_indices(
            data_length,
            sequence_length,
            step_size,
            max_horizon_steps,
        )?;

        // Create calibrator with same parameters as original
        let calibrator = crate::targets::adaptive_parameters::AdaptiveParameterCalibrator::new(
            targets_config.clone(),
        );

        // Use async runtime to call the calibration
        let runtime = tokio::runtime::Runtime::new()?;
        let all_params = runtime.block_on(calibrator.calibrate_all_targets(
            &ohlcv_data,
            sequence_length,
            horizon_steps,
            &sequence_indices,
        ))?;

        Ok(Box::new(all_params.direction))
    }
}

/// Volatility Target Generator
pub struct VolatilityTargetGenerator;

impl TargetGenerator for VolatilityTargetGenerator {
    fn target_type(&self) -> &'static str {
        "volatility"
    }

    fn target_name(&self) -> &'static str {
        "Volatility"
    }

    fn class_names(&self) -> Vec<&'static str> {
        vec!["VeryLow", "Low", "Medium", "High", "VeryHigh"]
    }

    fn generate_targets(
        &self,
        df: &DataFrame,
        horizons: &[String],
        targets_config: &TargetsConfig,
        sequence_indices: &[usize],
        sequence_length: usize,
        adaptive_params: Option<&dyn AdaptiveParameters>,
    ) -> Result<HashMap<String, Vec<i32>>> {
        let params =
            adaptive_params.and_then(|p| p.as_any().downcast_ref::<VolatilityAdaptiveParams>());
        crate::targets::generate_volatility_targets_with_adaptive_params(
            df,
            horizons,
            targets_config,
            sequence_indices,
            sequence_length,
            params,
        )
    }

    fn calibrate_parameters(
        &self,
        df: &DataFrame,
        sequence_length: usize,
        horizon_steps: usize,
        targets_config: &TargetsConfig,
    ) -> Result<Box<dyn AdaptiveParameters>> {
        // Delegate to the existing calibration system to preserve original logic
        // This ensures the original grid search and MIN-CLASS optimization is used

        // Extract OHLCV data for calibration
        let ohlcv_data = crate::utils::market_data::extract_ohlcv_data(df)?;

        // Calculate sequence indices for calibration (same logic as original)
        let data_length = ohlcv_data.len();
        let max_horizon_steps = horizon_steps;
        let step_size = 1;

        let sequence_indices = crate::utils::sequence_utils::calculate_sequence_indices(
            data_length,
            sequence_length,
            step_size,
            max_horizon_steps,
        )?;

        // Create calibrator with same parameters as original
        let calibrator = crate::targets::adaptive_parameters::AdaptiveParameterCalibrator::new(
            targets_config.clone(),
        );

        // Use async runtime to call the calibration
        let runtime = tokio::runtime::Runtime::new()?;
        let params = runtime.block_on(calibrator.calibrate_volatility(
            &ohlcv_data,
            sequence_length,
            horizon_steps,
            &sequence_indices,
        ))?;

        Ok(Box::new(params))
    }
}

/// Sentiment Target Generator
pub struct SentimentTargetGenerator;

impl TargetGenerator for SentimentTargetGenerator {
    fn target_type(&self) -> &'static str {
        "sentiment"
    }

    fn target_name(&self) -> &'static str {
        "Sentiment"
    }

    fn class_names(&self) -> Vec<&'static str> {
        vec![
            "Strong Panic",
            "Moderate Panic",
            "Neutral",
            "Moderate Greed",
            "Strong Greed",
        ]
    }

    fn generate_targets(
        &self,
        df: &DataFrame,
        horizons: &[String],
        targets_config: &TargetsConfig,
        sequence_indices: &[usize],
        sequence_length: usize,
        adaptive_params: Option<&dyn AdaptiveParameters>,
    ) -> Result<HashMap<String, Vec<i32>>> {
        let params =
            adaptive_params.and_then(|p| p.as_any().downcast_ref::<SentimentAdaptiveParams>());
        crate::targets::generate_sentiment_targets_with_adaptive_params(
            df,
            horizons,
            targets_config,
            sequence_indices,
            sequence_length,
            params,
        )
    }

    fn calibrate_parameters(
        &self,
        df: &DataFrame,
        sequence_length: usize,
        horizon_steps: usize,
        targets_config: &TargetsConfig,
    ) -> Result<Box<dyn AdaptiveParameters>> {
        // Delegate to the existing calibration system to preserve original logic
        // This ensures the original grid search and MIN-CLASS optimization is used

        // Extract OHLCV data for calibration
        let ohlcv_data = crate::utils::market_data::extract_ohlcv_data(df)?;

        // Calculate sequence indices for calibration (same logic as original)
        let data_length = ohlcv_data.len();
        let max_horizon_steps = horizon_steps;
        let step_size = 1;

        let sequence_indices = crate::utils::sequence_utils::calculate_sequence_indices(
            data_length,
            sequence_length,
            step_size,
            max_horizon_steps,
        )?;

        // Create calibrator with same parameters as original
        let calibrator = crate::targets::adaptive_parameters::AdaptiveParameterCalibrator::new(
            targets_config.clone(),
        );

        // Use async runtime to call the calibration
        let runtime = tokio::runtime::Runtime::new()?;
        let all_params = runtime.block_on(calibrator.calibrate_all_targets(
            &ohlcv_data,
            sequence_length,
            horizon_steps,
            &sequence_indices,
        ))?;

        Ok(Box::new(all_params.sentiment))
    }
}

/// Volume Target Generator
pub struct VolumeTargetGenerator;

impl TargetGenerator for VolumeTargetGenerator {
    fn target_type(&self) -> &'static str {
        "volume"
    }

    fn target_name(&self) -> &'static str {
        "Volume"
    }

    fn class_names(&self) -> Vec<&'static str> {
        vec!["Very Low", "Low", "Medium", "High", "Very High"]
    }

    fn generate_targets(
        &self,
        df: &DataFrame,
        horizons: &[String],
        targets_config: &TargetsConfig,
        sequence_indices: &[usize],
        sequence_length: usize,
        adaptive_params: Option<&dyn AdaptiveParameters>,
    ) -> Result<HashMap<String, Vec<i32>>> {
        let params =
            adaptive_params.and_then(|p| p.as_any().downcast_ref::<VolumeAdaptiveParams>());
        crate::targets::generate_volume_targets_with_adaptive_params(
            df,
            horizons,
            targets_config,
            sequence_indices,
            sequence_length,
            params,
        )
    }

    fn calibrate_parameters(
        &self,
        df: &DataFrame,
        sequence_length: usize,
        horizon_steps: usize,
        targets_config: &TargetsConfig,
    ) -> Result<Box<dyn AdaptiveParameters>> {
        // Delegate to the existing calibration system to preserve original logic
        // This ensures the original grid search and MIN-CLASS optimization is used

        // Extract OHLCV data for calibration
        let ohlcv_data = crate::utils::market_data::extract_ohlcv_data(df)?;

        // Calculate sequence indices for calibration (same logic as original)
        let data_length = ohlcv_data.len();
        let max_horizon_steps = horizon_steps;
        let step_size = 1;

        let sequence_indices = crate::utils::sequence_utils::calculate_sequence_indices(
            data_length,
            sequence_length,
            step_size,
            max_horizon_steps,
        )?;

        // Create calibrator with same parameters as original
        let calibrator = crate::targets::adaptive_parameters::AdaptiveParameterCalibrator::new(
            targets_config.clone(),
        );

        // Use async runtime to call the calibration
        let runtime = tokio::runtime::Runtime::new()?;
        let all_params = runtime.block_on(calibrator.calibrate_all_targets(
            &ohlcv_data,
            sequence_length,
            horizon_steps,
            &sequence_indices,
        ))?;

        Ok(Box::new(all_params.volume))
    }
}
