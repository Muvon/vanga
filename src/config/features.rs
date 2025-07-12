use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FeatureConfig {
    /// Technical indicators configuration
    pub technical_indicators: TechnicalIndicatorsConfig,

    /// Market microstructure features
    pub market_microstructure: MarketMicrostructureConfig,

    /// Volatility features
    pub volatility_features: VolatilityFeaturesConfig,

    /// Custom features (any additional columns in CSV)
    pub custom_features: CustomFeaturesConfig,

    /// Feature engineering settings
    pub engineering: FeatureEngineeringConfig,

    /// Feature selection settings
    pub selection: FeatureSelectionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicatorsConfig {
    pub enabled: bool,

    /// Moving averages
    pub moving_averages: MovingAveragesConfig,

    /// Momentum indicators
    pub momentum: MomentumConfig,

    /// Volatility indicators
    pub volatility: VolatilityIndicatorsConfig,

    /// Volume indicators
    pub volume: VolumeIndicatorsConfig,

    /// Trend indicators
    pub trend: TrendIndicatorsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovingAveragesConfig {
    pub sma_periods: Vec<u32>,
    pub ema_periods: Vec<u32>,
    pub wma_periods: Vec<u32>,
    pub hull_periods: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumConfig {
    pub rsi_periods: Vec<u32>,
    pub stochastic: bool,
    pub williams_r: bool,
    pub cci_periods: Vec<u32>,
    pub momentum_periods: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityIndicatorsConfig {
    pub bollinger_bands: BollingerBandsConfig,
    pub atr_periods: Vec<u32>,
    pub keltner_channels: bool,
    pub donchian_channels: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BollingerBandsConfig {
    pub enabled: bool,
    pub period: u32,
    pub std_dev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeIndicatorsConfig {
    pub obv: bool,
    pub volume_sma_periods: Vec<u32>,
    pub mfi_periods: Vec<u32>,
    pub ad_line: bool,
    pub chaikin_oscillator: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendIndicatorsConfig {
    pub macd: MACDConfig,
    pub adx_periods: Vec<u32>,
    pub parabolic_sar: bool,
    pub aroon: bool,
    /// Advanced mathematical indicators
    pub advanced: AdvancedIndicatorsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MACDConfig {
    pub enabled: bool,
    pub fast_period: u32,
    pub slow_period: u32,
    pub signal_period: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedIndicatorsConfig {
    pub enabled: bool,
    pub hurst_window: usize,
    pub fractal_window: usize,
    pub regime_window: usize,
    pub clustering_window: usize,
    pub reversion_window: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMicrostructureConfig {
    pub enabled: bool,

    /// Price velocity and acceleration
    pub price_dynamics: PriceDynamicsConfig,

    /// Volume-price relationship
    pub volume_price: VolumePriceConfig,

    /// Trade intensity metrics
    pub trade_intensity: TradeIntensityConfig,

    /// Market depth proxies
    pub market_depth: MarketDepthConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceDynamicsConfig {
    pub velocity_periods: Vec<u32>,
    pub acceleration_periods: Vec<u32>,
    pub jerk_periods: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumePriceConfig {
    pub vwap_periods: Vec<u32>,
    pub volume_price_trend: bool,
    pub ease_of_movement: bool,
    pub negative_volume_index: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeIntensityConfig {
    pub enabled: bool,
    pub lookback_periods: Vec<u32>,
    pub intensity_metrics: Vec<IntensityMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntensityMetric {
    TradesPerMinute,
    VolumePerTrade,
    PriceImpact,
    OrderFlow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDepthConfig {
    pub bid_ask_spread_proxy: bool,
    pub order_book_imbalance_proxy: bool,
    pub market_impact_proxy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityFeaturesConfig {
    pub enabled: bool,

    /// Realized volatility
    pub realized_volatility: RealizedVolatilityConfig,

    /// GARCH-based features
    pub garch_features: GARCHFeaturesConfig,

    /// Volatility clustering
    pub clustering: VolatilityClusteringConfig,

    /// Regime detection
    pub regime_detection: RegimeDetectionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealizedVolatilityConfig {
    pub periods: Vec<String>, // e.g., ["1h", "4h", "24h"]
    pub estimators: Vec<VolatilityEstimator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityEstimator {
    Standard,
    RangeBasedYangZhang,
    RangeBasedGarmanKlass,
    RangeBasedRogers,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GARCHFeaturesConfig {
    pub enabled: bool,
    pub model_orders: Vec<(u32, u32)>, // (p, q) orders
    pub conditional_volatility: bool,
    pub volatility_forecasts: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityClusteringConfig {
    pub enabled: bool,
    pub clustering_strength: bool,
    pub persistence_measures: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeDetectionConfig {
    pub enabled: bool,
    pub methods: Vec<RegimeMethod>,
    pub sensitivity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegimeMethod {
    VolatilityBased,
    TrendBased,
    HiddenMarkov,
    ChangePoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomFeaturesConfig {
    /// Whether to automatically include all additional CSV columns
    pub auto_include_all: bool,

    /// Specific custom features to include
    pub include_features: Vec<String>,

    /// Features to exclude
    pub exclude_features: Vec<String>,

    /// Custom feature transformations
    pub transformations: HashMap<String, FeatureTransformation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureTransformation {
    Log,
    Sqrt,
    Square,
    Diff,
    PercentChange,
    ZScore,
    MinMaxScale,
    RobustScale,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    /// Polynomial features
    pub polynomial_features: PolynomialFeaturesConfig,

    /// Interaction features
    pub interaction_features: InteractionFeaturesConfig,

    /// Lag features
    pub lag_features: LagFeaturesConfig,

    /// Rolling window features
    pub rolling_features: RollingFeaturesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialFeaturesConfig {
    pub enabled: bool,
    pub degree: u32,
    pub include_bias: bool,
    pub interaction_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionFeaturesConfig {
    pub enabled: bool,
    pub max_interactions: u32,
    pub feature_pairs: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagFeaturesConfig {
    pub enabled: bool,
    pub lag_periods: Vec<u32>,
    pub features_to_lag: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingFeaturesConfig {
    pub enabled: bool,
    pub window_sizes: Vec<u32>,
    pub statistics: Vec<RollingStatistic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollingStatistic {
    Mean,
    Std,
    Min,
    Max,
    Median,
    Skew,
    Kurtosis,
    Quantile(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionConfig {
    pub enabled: bool,

    /// Maximum number of features to select
    pub max_features: Option<usize>,

    /// Correlation threshold for removing highly correlated features
    pub correlation_threshold: f64,

    /// Minimum importance threshold
    pub importance_threshold: f64,

    /// Feature selection methods
    pub methods: Vec<FeatureSelectionMethod>,

    /// Whether to keep crypto-specific features regardless of selection
    pub keep_crypto_features: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    CorrelationFilter,
    VarianceThreshold,
    UnivariateSelection,
    RecursiveElimination,
    ImportanceBased,
    LassoBased,
}

impl Default for TechnicalIndicatorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            moving_averages: MovingAveragesConfig {
                sma_periods: vec![5, 10, 20, 50, 200],
                ema_periods: vec![5, 10, 20, 50, 200],
                wma_periods: vec![10, 20],
                hull_periods: vec![9, 21],
            },
            momentum: MomentumConfig {
                rsi_periods: vec![14, 21],
                stochastic: true,
                williams_r: true,
                cci_periods: vec![14, 20],
                momentum_periods: vec![10, 20],
            },
            volatility: VolatilityIndicatorsConfig {
                bollinger_bands: BollingerBandsConfig {
                    enabled: true,
                    period: 20,
                    std_dev: 2.0,
                },
                atr_periods: vec![14, 21],
                keltner_channels: true,
                donchian_channels: true,
            },
            volume: VolumeIndicatorsConfig {
                obv: true,
                volume_sma_periods: vec![10, 20],
                mfi_periods: vec![14],
                ad_line: true,
                chaikin_oscillator: true,
            },
            trend: TrendIndicatorsConfig {
                macd: MACDConfig {
                    enabled: true,
                    fast_period: 12,
                    slow_period: 26,
                    signal_period: 9,
                },
                adx_periods: vec![14],
                parabolic_sar: true,
                aroon: true,
                advanced: AdvancedIndicatorsConfig::default(),
            },
        }
    }
}

impl Default for MarketMicrostructureConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            price_dynamics: PriceDynamicsConfig {
                velocity_periods: vec![5, 10, 20],
                acceleration_periods: vec![5, 10],
                jerk_periods: vec![5],
            },
            volume_price: VolumePriceConfig {
                vwap_periods: vec![20, 50],
                volume_price_trend: true,
                ease_of_movement: true,
                negative_volume_index: true,
            },
            trade_intensity: TradeIntensityConfig {
                enabled: true,
                lookback_periods: vec![10, 20, 50],
                intensity_metrics: vec![
                    IntensityMetric::TradesPerMinute,
                    IntensityMetric::VolumePerTrade,
                ],
            },
            market_depth: MarketDepthConfig {
                bid_ask_spread_proxy: true,
                order_book_imbalance_proxy: true,
                market_impact_proxy: true,
            },
        }
    }
}

impl Default for VolatilityFeaturesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            realized_volatility: RealizedVolatilityConfig {
                periods: vec!["1h".to_string(), "4h".to_string(), "24h".to_string()],
                estimators: vec![
                    VolatilityEstimator::Standard,
                    VolatilityEstimator::RangeBasedYangZhang,
                ],
            },
            garch_features: GARCHFeaturesConfig {
                enabled: true,
                model_orders: vec![(1, 1), (1, 2), (2, 1)],
                conditional_volatility: true,
                volatility_forecasts: true,
            },
            clustering: VolatilityClusteringConfig {
                enabled: true,
                clustering_strength: true,
                persistence_measures: true,
            },
            regime_detection: RegimeDetectionConfig {
                enabled: true,
                methods: vec![RegimeMethod::VolatilityBased, RegimeMethod::TrendBased],
                sensitivity: 0.5,
            },
        }
    }
}

impl Default for CustomFeaturesConfig {
    fn default() -> Self {
        Self {
            auto_include_all: true,
            include_features: vec![],
            exclude_features: vec![],
            transformations: HashMap::new(),
        }
    }
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            polynomial_features: PolynomialFeaturesConfig {
                enabled: false,
                degree: 2,
                include_bias: false,
                interaction_only: true,
            },
            interaction_features: InteractionFeaturesConfig {
                enabled: true,
                max_interactions: 10,
                feature_pairs: vec![],
            },
            lag_features: LagFeaturesConfig {
                enabled: true,
                lag_periods: vec![1, 2, 3, 5, 10],
                features_to_lag: vec!["close".to_string(), "volume".to_string()],
            },
            rolling_features: RollingFeaturesConfig {
                enabled: true,
                window_sizes: vec![5, 10, 20],
                statistics: vec![
                    RollingStatistic::Mean,
                    RollingStatistic::Std,
                    RollingStatistic::Min,
                    RollingStatistic::Max,
                ],
            },
        }
    }
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_features: None,
            correlation_threshold: 0.95,
            importance_threshold: 0.001,
            methods: vec![
                FeatureSelectionMethod::CorrelationFilter,
                FeatureSelectionMethod::ImportanceBased,
            ],
            keep_crypto_features: true,
        }
    }
}

impl Default for AdvancedIndicatorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hurst_window: 100,
            fractal_window: 50,
            regime_window: 50,
            clustering_window: 50,
            reversion_window: 50,
        }
    }
}
