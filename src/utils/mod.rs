pub mod backtest_reporter;
pub mod device;
pub mod diagnostics;
pub mod error;
pub mod feature_window;
pub mod file_discovery;
pub mod market_data;
pub mod memory_manager;
pub mod metrics;
pub mod model_path;
pub mod parser;
pub mod sequence_utils;

pub use error::{Result, VangaError};
pub use feature_window::{
    calculate_max_feature_window, calculate_min_data_requirements, MinDataRequirements,
};
pub use memory_manager::{AdaptiveBatchSize, MemoryManager, MemoryOptimizationStrategy};

// Re-export diagnostics for easy access
pub use diagnostics::TrainingDiagnostics;
