// Feature window calculation utility
use crate::config::FeatureConfig;
use crate::utils::error::Result;

/// Calculate the maximum feature window required for all configured features
/// This determines how many periods of historical data are needed before
/// all technical indicators and engineered features become valid (non-NaN)
pub fn calculate_max_feature_window(config: &FeatureConfig) -> usize {
    let mut max_window = 0;

    // Technical indicators windows
    if config.technical_indicators.enabled {
        // Moving averages
        max_window = max_window.max(
            config
                .technical_indicators
                .moving_averages
                .sma_periods
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );
        max_window = max_window.max(
            config
                .technical_indicators
                .moving_averages
                .ema_periods
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );
        max_window = max_window.max(
            config
                .technical_indicators
                .moving_averages
                .wma_periods
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );
        max_window = max_window.max(
            config
                .technical_indicators
                .moving_averages
                .hull_periods
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );

        // Momentum indicators
        max_window = max_window.max(
            config
                .technical_indicators
                .momentum
                .rsi_periods
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );
        max_window = max_window.max(
            config
                .technical_indicators
                .momentum
                .cci_periods
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );
        max_window = max_window.max(
            config
                .technical_indicators
                .momentum
                .momentum_periods
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );

        // Volatility indicators
        max_window = max_window.max(
            config
                .technical_indicators
                .volatility
                .bollinger_bands
                .period as usize,
        );
        max_window = max_window.max(
            config
                .technical_indicators
                .volatility
                .atr_periods
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );

        // Volume indicators
        max_window = max_window.max(
            config
                .technical_indicators
                .volume
                .volume_sma_periods
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );
        max_window = max_window.max(
            config
                .technical_indicators
                .volume
                .mfi_periods
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );

        // Trend indicators (MACD typically uses 26 period slow EMA)
        if config.technical_indicators.trend.macd.enabled {
            max_window =
                max_window.max(config.technical_indicators.trend.macd.slow_period as usize);
        }
    }

    // Feature engineering windows
    // Lag features
    if config.engineering.lag_features.enabled {
        max_window = max_window.max(
            config
                .engineering
                .lag_features
                .lag_periods
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );
    }

    // Rolling features
    if config.engineering.rolling_features.enabled {
        max_window = max_window.max(
            config
                .engineering
                .rolling_features
                .window_sizes
                .iter()
                .max()
                .copied()
                .unwrap_or(0) as usize,
        );
    }

    // Volatility features windows
    if config.volatility_features.enabled {
        // GARCH features typically need more historical data
        if config.volatility_features.garch_features.enabled {
            // GARCH models typically need at least 100 observations
            max_window = max_window.max(100);
        }
    }

    // Cross-asset features may need additional lookback
    if config.cross_asset.enabled {
        if config.cross_asset.sentiment_analysis.enabled {
            max_window = max_window.max(config.cross_asset.sentiment_analysis.lookback_periods);
        }
        if config.cross_asset.correlation_analysis.enabled {
            max_window = max_window.max(config.cross_asset.correlation_analysis.min_periods);
            max_window = max_window.max(config.cross_asset.correlation_analysis.correlation_window);
        }
    }

    // Ensure minimum window for basic calculations
    max_window.max(1)
}

/// Calculate minimum data requirements for training/prediction
pub fn calculate_min_data_requirements(
    config: &FeatureConfig,
    sequence_length: usize,
    horizons: &[String],
) -> Result<MinDataRequirements> {
    let max_feature_window = calculate_max_feature_window(config);

    // Calculate maximum horizon steps
    let max_horizon_steps = horizons
        .iter()
        .map(|h| crate::utils::parser::parse_horizon_to_steps(h).unwrap_or(1))
        .max()
        .unwrap_or(1);

    let min_total_rows = max_feature_window + sequence_length + max_horizon_steps;
    let effective_training_rows = min_total_rows - max_feature_window;

    Ok(MinDataRequirements {
        max_feature_window,
        sequence_length,
        max_horizon_steps,
        min_total_rows,
        effective_training_rows,
    })
}

/// Data requirements breakdown for validation and logging
#[derive(Debug, Clone)]
pub struct MinDataRequirements {
    /// Maximum window needed for feature calculation
    pub max_feature_window: usize,
    /// LSTM sequence length
    pub sequence_length: usize,
    /// Maximum prediction horizon steps
    pub max_horizon_steps: usize,
    /// Minimum total rows needed in dataset
    pub min_total_rows: usize,
    /// Effective rows available for training after feature window
    pub effective_training_rows: usize,
}

impl MinDataRequirements {
    /// Validate that dataset has sufficient data
    pub fn validate(&self, actual_rows: usize) -> Result<()> {
        if actual_rows < self.min_total_rows {
            return Err(crate::utils::error::VangaError::DataError(format!(
                "Insufficient data: {} rows available, {} required\n\
                 Breakdown:\n\
                 • Feature window: {} periods (for technical indicators)\n\
                 • Sequence length: {} periods (for LSTM input)\n\
                 • Horizon buffer: {} periods (for target calculation)\n\
                 • Total required: {} periods\n\
                 \n\
                 Solution: Provide at least {} rows of historical data",
                actual_rows,
                self.min_total_rows,
                self.max_feature_window,
                self.sequence_length,
                self.max_horizon_steps,
                self.min_total_rows,
                self.min_total_rows
            )));
        }
        Ok(())
    }

    /// Log data requirements summary
    pub fn log_summary(&self, actual_rows: usize, context: &str) {
        log::info!("📊 DATA REQUIREMENTS SUMMARY ({})", context);
        log::info!("   • Dataset size: {} rows", actual_rows);
        log::info!(
            "   • Feature window: {} periods (technical indicators)",
            self.max_feature_window
        );
        log::info!(
            "   • Sequence length: {} periods (LSTM input)",
            self.sequence_length
        );
        log::info!(
            "   • Horizon buffer: {} periods (target calculation)",
            self.max_horizon_steps
        );
        log::info!("   • Total required: {} periods", self.min_total_rows);
        log::info!(
            "   • Effective training data: {} rows",
            actual_rows.saturating_sub(self.max_feature_window)
        );

        if actual_rows >= self.min_total_rows {
            log::info!("   ✅ Sufficient data available");
        } else {
            log::warn!(
                "   ❌ Insufficient data - need {} more rows",
                self.min_total_rows - actual_rows
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::features::*;

    #[test]
    fn test_calculate_max_feature_window() {
        let mut config = FeatureConfig::default();

        // Test with SMA periods
        config.technical_indicators.enabled = true;
        config.technical_indicators.moving_averages.sma_periods = vec![5, 10, 20, 50, 200];

        let max_window = calculate_max_feature_window(&config);
        assert_eq!(max_window, 200);
    }

    #[test]
    fn test_calculate_min_data_requirements() {
        // Create minimal config with only 20-period SMA
        let config = FeatureConfig {
            technical_indicators: crate::config::features::TechnicalIndicatorsConfig {
                enabled: true,
                moving_averages: crate::config::features::MovingAveragesConfig {
                    sma_periods: vec![20],
                    ema_periods: vec![], // Empty to avoid larger windows
                    wma_periods: vec![],
                    hull_periods: vec![],
                },
                momentum: crate::config::features::MomentumConfig {
                    rsi_periods: vec![],
                    stochastic: false,
                    williams_r: false,
                    cci_periods: vec![],
                    momentum_periods: vec![],
                },
                volatility: crate::config::features::VolatilityIndicatorsConfig {
                    bollinger_bands: crate::config::features::BollingerBandsConfig {
                        enabled: false,
                        period: 20,
                        std_dev: 2.0,
                    },
                    atr_periods: vec![],
                    keltner_channels: false,
                    donchian_channels: false,
                },
                volume: crate::config::features::VolumeIndicatorsConfig {
                    obv: false,
                    volume_sma_periods: vec![],
                    mfi_periods: vec![],
                    ad_line: false,
                    chaikin_oscillator: false,
                },
                trend: crate::config::features::TrendIndicatorsConfig {
                    macd: crate::config::features::MACDConfig {
                        enabled: false,
                        fast_period: 12,
                        slow_period: 26,
                        signal_period: 9,
                    },
                    adx_periods: vec![],
                    parabolic_sar: false,
                    aroon: false,
                    advanced: crate::config::features::AdvancedIndicatorsConfig {
                        enabled: false,
                        hurst_window: 100,
                        fractal_window: 50,
                        regime_window: 50,
                        clustering_window: 50,
                        reversion_window: 50,
                    },
                },
            },
            cross_asset: crate::config::features::CrossAssetConfig {
                enabled: false,
                ..Default::default()
            },
            volatility_features: crate::config::features::VolatilityFeaturesConfig {
                enabled: false,
                ..Default::default()
            },
            engineering: crate::config::features::FeatureEngineeringConfig {
                rolling_features: crate::config::features::RollingFeaturesConfig {
                    enabled: false,
                    window_sizes: vec![],
                    statistics: vec![],
                },
                ..Default::default()
            },
            ..Default::default()
        };

        let horizons = vec!["1h".to_string()];
        let sequence_length = 60;

        let requirements =
            calculate_min_data_requirements(&config, sequence_length, &horizons).unwrap();

        assert_eq!(requirements.max_feature_window, 20);
        assert_eq!(requirements.sequence_length, 60);
        assert!(requirements.min_total_rows > 20 + 60); // feature_window + sequence + horizon
    }

    #[test]
    fn test_validation_insufficient_data() {
        let requirements = MinDataRequirements {
            max_feature_window: 200,
            sequence_length: 60,
            max_horizon_steps: 1,
            min_total_rows: 261,
            effective_training_rows: 61,
        };

        // Should fail with insufficient data
        assert!(requirements.validate(100).is_err());

        // Should pass with sufficient data
        assert!(requirements.validate(300).is_ok());
    }
}
