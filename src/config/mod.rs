pub mod features;
pub mod model;
pub mod prediction;
pub mod training;

pub use features::FeatureConfig;
pub use model::ModelConfig;
pub use prediction::PredictionConfig;
pub use training::TrainingConfig;

/// Global configuration defaults for the LSTM forecasting system
pub struct GlobalConfig;

impl GlobalConfig {
    /// Default model directory
    pub const MODEL_DIR: &'static str = "./models";

    /// Default data directory
    pub const DATA_DIR: &'static str = "./data";

    /// Default configuration directory
    pub const CONFIG_DIR: &'static str = "./configs";

    /// Default prediction horizons
    pub const DEFAULT_HORIZONS: &'static [&'static str] = &["1h", "4h", "1d", "7d"];

    /// Required CSV columns (OHLCV)
    pub const REQUIRED_COLUMNS: &'static [&'static str] =
        &["timestamp", "open", "high", "low", "close", "volume"];

    /// Auto-generated technical indicators
    pub const AUTO_INDICATORS: &'static [&'static str] = &[
        // Trend indicators
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_50",
        "ema_200",
        "macd",
        "macd_signal",
        "macd_histogram",
        // Momentum indicators
        "rsi_14",
        "rsi_21",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "cci_14",
        "momentum_10",
        "momentum_20",
        // Volatility indicators
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "bb_percent",
        "atr_14",
        "atr_21",
        "keltner_upper",
        "keltner_lower",
        // Volume indicators
        "obv",
        "volume_sma_10",
        "volume_sma_20",
        "volume_ratio",
        "mfi_14",
        "ad_line",
        // Crypto-specific indicators
        "price_velocity",
        "price_acceleration",
        "volume_price_trend",
        "ease_of_movement",
        "chaikin_oscillator",
        // Market microstructure
        "bid_ask_spread_proxy",
        "trade_intensity",
        "vwap_deviation",
        "realized_volatility_1h",
        "realized_volatility_4h",
        "realized_volatility_24h",
        // Advanced patterns
        "fractal_dimension",
        "hurst_exponent",
        "regime_indicator",
        "volatility_clustering",
        "mean_reversion_strength",
    ];
}
