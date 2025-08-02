pub mod backtester;
pub mod predictor;
pub mod trainer;

pub use backtester::{run_backtest, run_batch_backtest, BacktestResults, Backtester};
pub use predictor::{predict, ModelWrapper, Predictor};
pub use trainer::ModelTrainer;

use crate::config::{PredictionConfig, TrainingConfig};
use crate::utils::error::Result;

/// High-level training function
pub async fn train_model(
    config: TrainingConfig,
) -> Result<crate::model::multi_target::MultiTargetLSTMModel> {
    let mut trainer = ModelTrainer::new(config);
    trainer.train().await
}

/// High-level prediction function for single-target models
pub async fn predict_single(
    config: PredictionConfig,
    model: &crate::model::lstm_simple::LSTMModel,
) -> Result<Vec<crate::output::PredictionResult>> {
    predict(config, model).await
}
