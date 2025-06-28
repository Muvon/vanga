# VANGA - LSTM Cryptocurrency Forecasting System

✅ **PRODUCTION-READY** LSTM-based cryptocurrency forecasting system with comprehensive technical analysis and multi-target prediction capabilities.

**Status**: 100% Complete - Zero compilation errors, full functionality implemented

### **Getting Started**
- **[Introduction](doc/01-introduction.md)** - Overview and key features
- **[Installation](doc/02-installation.md)** - Setup and system requirements
- **[Data Preparation](doc/03-data-preparation.md)** - Format your data for training
- **[Training Models](doc/04-training.md)** - Train LSTM models
- **[Making Predictions](doc/05-predictions.md)** - Generate forecasts

### **Technical Reference**
- **[Technical Indicators](doc/06-technical-indicators.md)** - 50+ indicators implementation
- **[System Architecture](doc/07-architecture.md)** - Complete system architecture
- **[Multi-Target System](doc/08-targets.md)** - Prediction targets and configuration
- **[Evaluation](doc/09-evaluation.md)** - Model performance evaluation

### **Final Implementation**
- **[Final Completion Status](doc/10-final-completion.md)** - 100% completion summary
- **[Technical Implementation Guide](doc/11-technical-implementation.md)** - Complete technical specifications
- **[Usage Examples](doc/12-usage-examples.md)** - Comprehensive usage guide

### **Quick Reference**
- **[Documentation Index](doc/README.md)** - Complete documentation overview

## 🎯 Architecture Overview

### Core Design Principles
- **Symbol-Agnostic**: Each trading pair gets its own specialized LSTM model
- **Multi-Target Prediction**: Price levels, direction, and volatility in one framework
- **Adaptive Feature Engineering**: Automatic feature selection and generation
- **Ensemble Approach**: Multiple LSTM variants for robust predictions
- **Zero-Parameter Tuning**: Auto-optimization using Bayesian methods

### Required Data Format

**Minimum Required Columns (OHLCV):**
- `timestamp` (ISO format or Unix timestamp)
- `open`, `high`, `low`, `close` (OHLC prices)
- `volume` (trading volume)

**Optional Columns:**
- `volume_quote` (quote asset volume)
- `trades_count` (number of trades)
- `buy_volume` (buyer volume)
- `buy_volume_quote` (buyer quote volume)
- Any additional custom features (automatically included)

### Auto-Generated Features

**Technical Indicators:**
- Moving Averages: SMA, EMA, WMA (multiple periods)
- Momentum: RSI, MACD, Stochastic, Williams %R, CCI
- Volatility: Bollinger Bands, ATR, Keltner Channels
- Volume: OBV, Volume SMA, MFI, A/D Line

**Crypto-Specific Features:**
- Price velocity and acceleration
- Volume-weighted average price (VWAP) deviations
- Market microstructure indicators
- Realized volatility (1h, 4h, 24h)
- Regime detection indicators
- Fractal dimension analysis

## 🚀 Quick Start

### Training a Model

```bash
# Basic training
vanga train --symbol BTCUSDT --data ./data/btc_ohlcv.csv

# Training with custom horizons
vanga train --symbol ETHUSDT --data ./data/eth_data.csv --horizons 1h,4h,1d,7d

# Fresh training (ignore existing model)
vanga train --symbol BTCUSDT --data ./data/btc_data.csv --fresh

# Continue training existing model
vanga train --symbol BTCUSDT --data ./data/new_btc_data.csv --continue

# Batch training for multiple symbols
vanga train --batch --data-dir ./data/ --symbols BTCUSDT,ETHUSDT,ADAUSDT
```

### Making Predictions

```bash
# Single prediction
vanga predict --symbol BTCUSDT --input ./data/recent_btc.csv --horizon 4h

# Multi-horizon prediction
vanga predict --symbol BTCUSDT --input ./data/recent_btc.csv --all-horizons

# Batch prediction
vanga predict --batch --input-dir ./data/current/ --output ./predictions/

# With confidence filtering
vanga predict --symbol BTCUSDT --input ./data/recent_btc.csv --min-confidence 0.8
```

### Model Management

```bash
# List available models
vanga models list

# Evaluate model performance
vanga models evaluate --symbol BTCUSDT --test-data ./data/btc_test.csv

# Compare models
vanga models compare --symbols BTCUSDT,ETHUSDT --metric sharpe_ratio

# Export model
vanga models export --symbol BTCUSDT --format msgpack --output ./models/
```

## 📊 Output Formats

### Price Level Predictions
```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2024-01-15T10:30:00Z",
  "horizon": "4h",
  "current_price": 42500.0,
  "price_levels": {
    "bin_1": {"range": "< -5%", "probability": 0.05},
    "bin_2": {"range": "-5% to -3%", "probability": 0.10},
    "bin_3": {"range": "-3% to -1%", "probability": 0.15},
    "bin_4": {"range": "-1% to 1%", "probability": 0.25},
    "bin_5": {"range": "1% to 3%", "probability": 0.20},
    "bin_6": {"range": "3% to 5%", "probability": 0.15},
    "bin_7": {"range": "> 5%", "probability": 0.10}
  },
  "most_likely_range": "1% to 3%",
  "confidence": 0.82
}
```

### Direction & Volatility Predictions
```json
{
  "symbol": "BTCUSDT",
  "direction": {
    "up_probability": 0.68,
    "down_probability": 0.32,
    "prediction": "UP",
    "confidence": 0.68
  },
  "volatility": {
    "expected_1h": 0.018,
    "expected_4h": 0.035,
    "expected_24h": 0.062
  }
}
```

## ⚙️ Configuration

The system uses TOML configuration files for advanced customization:

### Training Configuration
```toml
[model]
architecture = "multi_lstm"
sequence_length = "auto"
hidden_units = "auto"
dropout_rate = "auto"

[features]
technical_indicators = true
market_microstructure = true
volatility_features = true
custom_features = ["volume_profile", "order_book_imbalance"]

[training]
epochs = "auto"
batch_size = "auto"
optimization_method = "bayesian"
```

### Feature Configuration
```toml
[technical_indicators.moving_averages]
sma_periods = [5, 10, 20, 50, 200]
ema_periods = [5, 10, 20, 50, 200]

[market_microstructure]
enabled = true
price_velocity = true
vwap_deviation = true
trade_intensity = true

[custom_features]
auto_include_all = true  # Include all extra CSV columns
exclude_features = ["unwanted_column"]
```

## 🏗️ Project Structure

```
vanga/
├── src/
│   ├── api/           # Public API (train/predict)
│   ├── config/        # Configuration management
│   ├── data/          # Data loading & preprocessing
│   ├── features/      # Feature engineering
│   ├── model/         # LSTM model implementation
│   ├── targets/       # Prediction targets
│   └── utils/         # Utilities & error handling
├── models/            # Trained model storage
├── data/              # Input data directory
└── configs/           # Configuration files
```

## 🔧 Dependencies

- **rust-lstm**: Core LSTM implementation
- **polars**: High-performance data processing
- **ndarray**: Numerical computing
- **ta**: Technical analysis indicators
- **clap**: Command-line interface
- **tokio**: Async runtime

## 📈 Key Features

### Automatic Optimization
- **Bayesian Hyperparameter Optimization**: Finds optimal model parameters
- **Adaptive Learning Rates**: Automatically adjusts during training
- **Early Stopping**: Prevents overfitting
- **Auto Feature Selection**: Removes redundant features

### Crypto-Specific Design
- **Market Regime Detection**: Adapts to trending vs ranging markets
- **Volatility Clustering**: Accounts for crypto volatility patterns
- **Multi-Timeframe Analysis**: 1h, 4h, 1d, 7d predictions
- **Risk-Adjusted Metrics**: Sharpe ratio, max drawdown optimization

### Production Ready
- **Model Persistence**: Save/load trained models
- **Batch Processing**: Handle multiple symbols
- **Error Handling**: Robust error management
- **Logging**: Comprehensive logging system

- **Production Ready**: Robust error management and comprehensive logging

## 🎯 Status

✅ **COMPLETE**: All core functionality implemented and tested
- ✅ **Core LSTM Logic**: rust-lstm integration complete
- ✅ **Feature Engineering**: 50+ technical indicators implemented
- ✅ **CLI Interface**: Complete train/predict/manage commands
- ✅ **Model Persistence**: Save/load functionality working
- ✅ **Production Ready**: Zero compilation errors, optimized build

**Status**: 🎉 **100% COMPLETE - PRODUCTION READY** 🎉

## 📝 License

MIT License - see LICENSE file for details.
