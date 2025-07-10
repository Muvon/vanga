pub mod backtester;
pub mod multi_target_predictor;
pub mod predictor;
pub mod trainer;

pub use backtester::{run_backtest, run_batch_backtest, BacktestResults, Backtester};
pub use multi_target_predictor::{
    predict_multi_target, MultiTargetPredictions, MultiTargetPredictor,
};
pub use predictor::Predictor;
pub use trainer::ModelTrainer;

use crate::config::{PredictionConfig, TrainingConfig};
use crate::utils::error::Result;

/// High-level training function
pub async fn train_model(
    config: TrainingConfig,
) -> Result<crate::model::multi_target::MultiTargetLSTMModel> {
    let trainer = ModelTrainer::new(config);
    trainer.train().await
}

/// High-level prediction function
pub async fn predict(
    config: PredictionConfig,
    model: &crate::model::lstm_simple::LSTMModel,
) -> Result<Vec<crate::output::PredictionResult>> {
    let predictor = Predictor::new(config);
    predictor.predict(model).await
}
