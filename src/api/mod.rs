pub mod predictor;
pub mod trainer;

pub use predictor::Predictor;
pub use trainer::ModelTrainer;

use crate::config::{PredictionConfig, TrainingConfig};
use crate::utils::error::Result;

/// High-level training function
pub async fn train_model(config: TrainingConfig) -> Result<crate::model::lstm_simple::LSTMModel> {
    let trainer = ModelTrainer::new(config);
    trainer.train().await
}

/// High-level prediction function
pub async fn predict(
    config: PredictionConfig,
    model: &crate::model::lstm_simple::LSTMModel,
) -> Result<ndarray::Array2<f64>> {
    let predictor = Predictor::new(config);
    predictor.predict(model).await
}
