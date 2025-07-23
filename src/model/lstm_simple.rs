//! LSTM model implementation - now modularized
//!
//! This file maintains backward compatibility by re-exporting
//! all types and functions from the new modular structure.
//!
//! The implementation has been refactored into focused modules:
//! - `config`: Configuration structs, enums, and validation
//! - `core`: Model creation, initialization, and persistence
//! - `training`: Training pipeline, optimization, and batch management
//! - `inference`: Prediction pipeline and forward pass
//! - `loss`: Loss calculation, validation metrics, and gradient utilities
//!
//! All existing functionality is preserved exactly - this is purely
//! an organizational refactoring for better maintainability.

pub use crate::model::lstm::*;
