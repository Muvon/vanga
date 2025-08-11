//! Target generation interface for unified target system
//!
//! This module provides a clean trait-based interface for all target generators,
//! enabling consistent method signatures, dynamic registration, and extensibility.

use crate::config::model::TargetsConfig;
use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Common interface for all target generators
///
/// Each target type (price_levels, direction, volatility, sentiment, volume)
/// implements this trait to provide consistent generation and calibration methods.
pub trait TargetGenerator: Send + Sync {
    /// Target type identifier (e.g., "price_levels", "direction", "volatility")
    fn target_type(&self) -> &'static str;

    /// Human-readable target name (e.g., "Price Levels", "Direction", "Volatility")
    fn target_name(&self) -> &'static str;

    /// Class names for this target (all targets use 5-class system)
    fn class_names(&self) -> Vec<&'static str>;

    /// Generate targets with optional adaptive parameters
    ///
    /// This method wraps the existing target-specific generation functions
    /// to provide a unified interface while preserving all existing logic.
    fn generate_targets(
        &self,
        df: &DataFrame,
        horizons: &[String],
        targets_config: &TargetsConfig,
        sequence_indices: &[usize],
        sequence_length: usize,
        adaptive_params: Option<&dyn AdaptiveParameters>,
    ) -> Result<HashMap<String, Vec<i32>>>;

    /// Calibrate adaptive parameters for this target
    ///
    /// This method wraps the existing calibration logic to provide
    /// a unified interface for parameter optimization.
    fn calibrate_parameters(
        &self,
        df: &DataFrame,
        sequence_length: usize,
        horizon_steps: usize,
        targets_config: &TargetsConfig,
    ) -> Result<Box<dyn AdaptiveParameters>>;
}

/// Common interface for adaptive parameters
///
/// All target-specific adaptive parameter structs implement this trait
/// to enable type-safe parameter passing through the unified interface.
pub trait AdaptiveParameters: Send + Sync {
    /// Get reference to underlying type for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Clone the parameters (required for some use cases)
    fn clone_box(&self) -> Box<dyn AdaptiveParameters>;
}
