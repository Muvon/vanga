//! Fractional optimizers for VANGA LSTM
//!
//! This module provides fractional-order optimizers that use memory-based gradient updates
//! for improved convergence in cryptocurrency forecasting.
//!
//! Key Features:
//! - FracAdam: Fractional-order Adam optimizer
//! - FracNAdam: Fractional-order NAdam optimizer  
//! - Prodigy: Learning-rate-free optimizer with D-adaptation
//! - FracProdigy: Fractional-order Prodigy optimizer

pub mod frac_adam;
#[cfg(test)]
mod frac_adam_test;
#[cfg(test)]
mod frac_integration_test;
pub mod frac_nadam;
#[cfg(test)]
mod frac_nadam_test;
pub mod frac_prodigy;
pub mod fractional;
pub mod prodigy;

// Re-export main fractional optimizer components
pub use frac_adam::{FracAdam, ParamsFracAdam};
pub use frac_nadam::{FracNAdam, ParamsFracNAdam};
pub use frac_prodigy::{FracProdigy, ParamsFracProdigy};
pub use fractional::{FractionalConfig, FractionalDerivative};
pub use prodigy::{ParamsProdigy, Prodigy};
