//! Calibration Module
//!
//! Refactored calibration system split into focused modules for maintainability.
//! This module provides the same public API as the original monolithic implementation.

pub mod bayesian;
#[cfg(test)]
mod bayesian_test;
pub mod core;
pub mod direction;
pub mod overlap_optimizer;
#[cfg(test)]
mod overlap_optimizer_test;
pub mod price_levels;
pub mod sentiment;
pub mod stop_levels;
pub mod types;
pub mod utils;
pub mod volatility;
pub mod volume;

// Re-export all public types and functions to maintain backward compatibility
pub use bayesian::{BayesianConfig, BayesianOptimizer};
pub use core::ParameterCalibrator;
pub use overlap_optimizer::{
    calculate_optimal_overlap, find_optimal_overlap_with_calibration, truncate_balanced_sequences,
};
pub use types::*;

// Legacy alias for backward compatibility
pub type TargetCalibrator = ParameterCalibrator;
