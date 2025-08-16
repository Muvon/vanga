pub mod attention;
pub mod attention_loss;
pub mod attention_moh;
#[cfg(test)]
pub mod attention_moh_test;
pub mod attention_moh_wrapper;
pub mod attention_optimizer;
pub mod attention_viz;
pub mod bias_correction;
#[cfg(test)]
pub mod bias_correction_integration_test;

pub mod lstm;
pub mod lstm_simple;
pub mod multi_target;
pub mod ordinal_smartcore;
pub mod smartcore_backend;
pub mod tft;
pub mod xgboost;

#[cfg(test)]
pub mod smartcore_test;

pub use attention::{AttentionFactory, AttentionModule, MultiHeadAttention};
pub use attention_loss::{AttentionLossConfig, AttentionLossFactory, AttentionWeightedLoss};
pub use attention_moh::MixtureOfHeadAttention;
pub use attention_moh_wrapper::{
    EnhancedAttentionFactory, MoHAttentionWrapper, MoHMetrics, MoHTrainingLoss,
};
pub use attention_optimizer::{
    OptimizedAttention, OptimizedAttentionConfig, OptimizedAttentionFactory,
};
pub use attention_viz::{AttentionAnalysis, AttentionVisualizationConfig, AttentionVisualizer};
pub use bias_correction::{BiasCorrection, LinearBiasCorrector};
pub use lstm::{LSTMConfig, LSTMModel}; // Use new modular LSTM
pub use lstm_simple::*; // Backward compatibility
pub use multi_target::MultiTargetLSTMModel;
pub use smartcore_backend::{SmartCoreMetadata, SmartCoreRegressor};
pub use tft::{
    QuantileMultiTargetModel, QuantileOutputConfig, QuantileRegressionHead,
    VariableSelectionAttention, VariableSelectionConfig, VariableSelectionNetwork,
};
pub use xgboost::{
    get_eval_metric_for_target, get_objective_for_target, XGBoostMetadata, XGBoostRegressor,
};
