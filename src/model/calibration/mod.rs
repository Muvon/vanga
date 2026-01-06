pub mod ece;
pub mod ensemble;
pub mod label_smoothing;
pub mod mixup;
pub mod temperature;

// Tests
#[cfg(test)]
mod ece_test;
#[cfg(test)]
mod ensemble_test;
#[cfg(test)]
mod performance_benchmark_test;
#[cfg(test)]
mod temperature_test;

pub use ece::{calculate_ece, calculate_per_class_ece, ReliabilityDiagram};
pub use ensemble::EnsembleCalibrator;
pub use label_smoothing::AdaptiveLabelSmoothing;
pub use mixup::AdaptiveMixup;
pub use temperature::AdaptiveTemperatureScaling;
