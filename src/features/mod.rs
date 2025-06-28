pub mod custom;
pub mod engineering;
pub mod microstructure;
pub mod technical;
pub mod volatility;

use crate::config::FeatureConfig;
use crate::utils::error::Result;
use polars::prelude::*;

/// Main feature engineering pipeline
pub struct FeatureEngineer {
    config: FeatureConfig,
}

impl FeatureEngineer {
    pub fn new(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Generate all features for the given DataFrame
    pub async fn generate_features(&self, df: DataFrame) -> Result<DataFrame> {
        let mut result_df = df;

        // Generate technical indicators
        if self.config.technical_indicators.enabled {
            result_df = technical::generate_technical_indicators(
                result_df,
                &self.config.technical_indicators,
            )
            .await?;
        }

        // Generate market microstructure features
        if self.config.market_microstructure.enabled {
            result_df = microstructure::generate_microstructure_features(
                result_df,
                &self.config.market_microstructure,
            )
            .await?;
        }

        // Generate volatility features
        if self.config.volatility_features.enabled {
            result_df = volatility::generate_volatility_features(
                result_df,
                &self.config.volatility_features,
            )
            .await?;
        }

        // Process custom features
        result_df =
            custom::process_custom_features(result_df, &self.config.custom_features).await?;

        // Apply feature engineering
        result_df =
            engineering::apply_feature_engineering(result_df, &self.config.engineering).await?;

        Ok(result_df)
    }
}
