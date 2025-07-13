use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    /// Trading symbols (e.g., ["BTCUSDT"] or ["BTCUSDT", "ETHUSDT"])
    pub symbols: Vec<String>,

    /// Path to input data CSV file or directory containing symbol files
    pub input_path: PathBuf,

    /// Specific prediction horizon (if not all)
    pub horizon: Option<String>,

    /// Whether to predict all available horizons
    pub all_horizons: bool,

    /// Output path for predictions
    pub output_path: Option<PathBuf>,

    /// Minimum confidence threshold for predictions
    pub min_confidence: f64,

    /// Prediction output configuration
    pub output_config: OutputConfig,

    /// Ensemble configuration
    pub ensemble_config: EnsembleConfig,

    /// Post-processing configuration
    pub post_processing: PostProcessingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output format for predictions
    pub format: OutputFormat,

    /// Confidence levels for intervals
    pub confidence_levels: Vec<f64>,

    /// Whether to include feature importance
    pub include_feature_importance: bool,

    /// Whether to include model metadata
    pub include_metadata: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Ensemble method
    pub method: EnsembleMethod,

    /// Weights for weighted ensemble (if applicable)
    pub weights: Option<Vec<f64>>,

    /// Whether to use dynamic weighting based on recent performance
    pub dynamic_weighting: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostProcessingConfig {
    /// Post-processing steps to apply
    pub steps: Vec<PostProcessingStep>,

    /// Volatility adjustment settings
    pub volatility_adjustment: VolatilityAdjustment,

    /// Market regime detection settings
    pub regime_detection: RegimeDetection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    PointEstimate,
    ConfidenceInterval,
    ProbabilityDistribution,
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    SimpleAverage,
    WeightedAverage,
    Stacking,
    Voting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PostProcessingStep {
    VolatilityAdjustment,
    TrendSmoothing,
    OutlierFiltering,
    RegimeAdjustment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityAdjustment {
    pub enabled: bool,
    pub method: VolatilityMethod,
    pub lookback_periods: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityMethod {
    RealizedVolatility,
    GARCH,
    EWMA,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeDetection {
    pub enabled: bool,
    pub method: RegimeMethod,
    pub sensitivity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegimeMethod {
    HiddenMarkov,
    ChangePoint,
    VolatilityBased,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            symbols: vec![],
            input_path: PathBuf::new(),
            horizon: None,
            all_horizons: false,
            output_path: None,
            min_confidence: 0.7,
            output_config: OutputConfig::default(),
            ensemble_config: EnsembleConfig::default(),
            post_processing: PostProcessingConfig::default(),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::ProbabilityDistribution,
            confidence_levels: vec![0.8, 0.9, 0.95],
            include_feature_importance: false,
            include_metadata: true,
        }
    }
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            method: EnsembleMethod::WeightedAverage,
            weights: None,
            dynamic_weighting: true,
        }
    }
}

impl Default for PostProcessingConfig {
    fn default() -> Self {
        Self {
            steps: vec![
                PostProcessingStep::VolatilityAdjustment,
                PostProcessingStep::TrendSmoothing,
            ],
            volatility_adjustment: VolatilityAdjustment {
                enabled: true,
                method: VolatilityMethod::RealizedVolatility,
                lookback_periods: 24,
            },
            regime_detection: RegimeDetection {
                enabled: true,
                method: RegimeMethod::VolatilityBased,
                sensitivity: 0.5,
            },
        }
    }
}

// Builder pattern implementation
impl PredictionConfig {
    /// Set single symbol (for backward compatibility)
    pub fn symbol<S: Into<String>>(mut self, symbol: S) -> Self {
        self.symbols = vec![symbol.into()];
        self
    }

    /// Set multiple symbols
    pub fn symbols<S: Into<String>>(mut self, symbols: Vec<S>) -> Self {
        self.symbols = symbols.into_iter().map(|s| s.into()).collect();
        self
    }

    pub fn input_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.input_path = path.into();
        self
    }

    pub fn horizon<S: Into<String>>(mut self, horizon: S) -> Self {
        self.horizon = Some(horizon.into());
        self
    }

    pub fn all_horizons(mut self, all: bool) -> Self {
        self.all_horizons = all;
        self
    }

    pub fn output_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.output_path = Some(path.into());
        self
    }

    pub fn min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Check if this is multi-symbol prediction
    pub fn is_multi_symbol(&self) -> bool {
        self.symbols.len() > 1
    }

    /// Validate horizon against model's trained horizons
    pub fn validate_horizon_against_model(
        &self,
        model: &crate::model::multi_target::MultiTargetLSTMModel,
    ) -> crate::utils::error::Result<()> {
        if let Some(requested_horizon) = &self.horizon {
            let trained_horizons = model.get_trained_horizons();
            if !trained_horizons.contains(requested_horizon) {
                return Err(crate::utils::error::VangaError::ConfigError(format!(
                    "Requested horizon '{}' was not trained. Available horizons: {:?}. Use one of the available horizons or --all-horizons to predict all.",
                    requested_horizon, trained_horizons
                )));
            }
        }
        Ok(())
    }

    /// Get effective horizon for prediction (validates and returns appropriate horizon)
    pub fn get_effective_horizon(
        &self,
        model: &crate::model::multi_target::MultiTargetLSTMModel,
    ) -> crate::utils::error::Result<String> {
        let trained_horizons = model.get_trained_horizons();

        if let Some(requested_horizon) = &self.horizon {
            // Validate requested horizon
            if !trained_horizons.contains(requested_horizon) {
                return Err(crate::utils::error::VangaError::ConfigError(format!(
                    "Requested horizon '{}' was not trained. Available horizons: {:?}",
                    requested_horizon, trained_horizons
                )));
            }
            Ok(requested_horizon.clone())
        } else if self.all_horizons {
            // For all_horizons, return the first one (caller should handle multiple)
            if trained_horizons.is_empty() {
                Ok("1h".to_string()) // Fallback for backward compatibility
            } else {
                Ok(trained_horizons[0].clone())
            }
        } else {
            // No specific horizon requested - use first trained horizon
            if trained_horizons.is_empty() {
                Ok("1h".to_string()) // Fallback for backward compatibility
            } else {
                Ok(trained_horizons[0].clone())
            }
        }
    }
}
