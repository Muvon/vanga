use std::fmt;

pub type Result<T> = std::result::Result<T, VangaError>;
#[derive(Debug)]
pub enum VangaError {
    ConfigError(String),
    DataError(String),
    DataValidation(crate::data::DataValidationError),
    ModelError(String),
    TrainingError(String),
    PredictionError(String),
    FeatureError(String),
    IoError(String),
    SerializationError(String),
    PolarsError(polars::error::PolarsError),
    OptimizationError(String),
    InvalidParameter {
        parameter: String,
        value: String,
        reason: String,
    },
}

// Custom Display implementation to handle DataValidation errors specially
impl fmt::Display for VangaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VangaError::DataValidation(crate::data::DataValidationError::InvalidData {
                issue,
                ..
            }) => {
                // For InvalidData errors, show just the formatted issue message
                write!(f, "Error: {}", issue)
            }
            _ => {
                // For all other errors, use standard formatting
                match self {
                    VangaError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
                    VangaError::DataError(msg) => write!(f, "Data error: {}", msg),
                    VangaError::ModelError(msg) => write!(f, "Model error: {}", msg),
                    VangaError::TrainingError(msg) => write!(f, "Training error: {}", msg),
                    VangaError::PredictionError(msg) => write!(f, "Prediction error: {}", msg),
                    VangaError::FeatureError(msg) => {
                        write!(f, "Feature engineering error: {}", msg)
                    }
                    VangaError::IoError(msg) => write!(f, "IO error: {}", msg),
                    VangaError::SerializationError(msg) => {
                        write!(f, "Serialization error: {}", msg)
                    }
                    VangaError::PolarsError(err) => write!(f, "Polars error: {}", err),
                    VangaError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
                    VangaError::InvalidParameter {
                        parameter,
                        value,
                        reason,
                    } => {
                        write!(
                            f,
                            "Invalid parameter: {} = {}, reason: {}",
                            parameter, value, reason
                        )
                    }
                    VangaError::DataValidation(err) => {
                        // For other DataValidation errors, use their Display implementation
                        write!(f, "Data validation error: {}", err)
                    }
                }
            }
        }
    }
}

// Implement std::error::Error trait manually
impl std::error::Error for VangaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VangaError::DataValidation(err) => Some(err),
            VangaError::PolarsError(err) => Some(err),
            _ => None,
        }
    }
}

// Add From implementations for common error types
impl From<std::io::Error> for VangaError {
    fn from(err: std::io::Error) -> Self {
        VangaError::IoError(err.to_string())
    }
}

impl From<crate::data::DataValidationError> for VangaError {
    fn from(err: crate::data::DataValidationError) -> Self {
        VangaError::DataValidation(err)
    }
}

impl From<polars::error::PolarsError> for VangaError {
    fn from(err: polars::error::PolarsError) -> Self {
        VangaError::PolarsError(err)
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
