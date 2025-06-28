use thiserror::Error;

pub type Result<T> = std::result::Result<T, VangaError>;
#[derive(Error, Debug)]
pub enum VangaError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Data error: {0}")]
    DataError(String),

    #[error("Data validation error: {0}")]
    DataValidation(#[from] crate::data::DataValidationError),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Prediction error: {0}")]
    PredictionError(String),

    #[error("Feature engineering error: {0}")]
    FeatureError(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Polars error: {0}")]
    PolarsError(#[from] polars::error::PolarsError),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Invalid parameter: {parameter} = {value}, reason: {reason}")]
    InvalidParameter {
        parameter: String,
        value: String,
        reason: String,
    },
}

// Add From implementations for common error types
impl From<std::io::Error> for VangaError {
    fn from(err: std::io::Error) -> Self {
        VangaError::IoError(err.to_string())
    }
}

impl VangaError {
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::ConfigError(msg.into())
    }

    pub fn data<S: Into<String>>(msg: S) -> Self {
        Self::DataError(msg.into())
    }

    pub fn model<S: Into<String>>(msg: S) -> Self {
        Self::ModelError(msg.into())
    }

    pub fn training<S: Into<String>>(msg: S) -> Self {
        Self::TrainingError(msg.into())
    }

    pub fn prediction<S: Into<String>>(msg: S) -> Self {
        Self::PredictionError(msg.into())
    }

    pub fn feature<S: Into<String>>(msg: S) -> Self {
        Self::FeatureError(msg.into())
    }

    pub fn invalid_param<S: Into<String>>(parameter: S, value: S, reason: S) -> Self {
        Self::InvalidParameter {
            parameter: parameter.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }
}
