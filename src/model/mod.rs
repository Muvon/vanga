pub mod attention;
pub mod attention_loss;
pub mod attention_optimizer;
pub mod attention_viz;
pub mod loss;
pub mod lstm;
pub mod lstm_simple;
pub mod multi_target;
pub mod tft;
pub mod xgboost;

#[cfg(test)]
pub mod loss_validation_tests;

pub use attention::{AttentionFactory, AttentionModule, MultiHeadAttention};
pub use attention_loss::{AttentionLossConfig, AttentionLossFactory, AttentionWeightedLoss};
pub use attention_optimizer::{
    OptimizedAttention, OptimizedAttentionConfig, OptimizedAttentionFactory,
};
pub use attention_viz::{AttentionAnalysis, AttentionVisualizationConfig, AttentionVisualizer};
pub use loss::{CryptoLossFunction, TensorCryptoLossFunction};
pub use lstm::{LSTMConfig, LSTMModel}; // Use new modular LSTM
pub use lstm_simple::*; // Backward compatibility
pub use multi_target::MultiTargetLSTMModel;
pub use tft::{
    QuantileMultiTargetModel, QuantileOutputConfig, QuantileRegressionHead,
    VariableSelectionAttention, VariableSelectionConfig, VariableSelectionNetwork,
};
pub use xgboost::{
    get_eval_metric_for_target, get_objective_for_target, XGBoostMetadata, XGBoostRegressor,
};
