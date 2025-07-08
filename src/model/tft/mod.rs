// Temporal Fusion Transformer (TFT) components for VANGA LSTM
// Building on existing attention and multi-target architecture

pub mod auto_optimizer;
pub mod quantile_regression;
pub mod variable_selection;

pub use auto_optimizer::{TFTAutoOptimizer, TFTAutoOptimizerConfig, TFTOptimizerFactory};
pub use quantile_regression::{
    QuantileMultiTargetModel, QuantileOutputConfig, QuantileRegressionHead,
};
pub use variable_selection::{
    VariableSelectionAttention, VariableSelectionConfig, VariableSelectionNetwork,
};
