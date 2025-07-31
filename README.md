# VANGA 🔮

**Advanced LSTM-based cryptocurrency forecasting system with state-of-the-art learning rate optimization and unified training architecture**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🚀 NEW: Advanced Learning Rate Optimization & Unified Training

VANGA now features **professional-grade learning rate optimization** with modern optimizers, intelligent scheduling, and a **unified training architecture**:

### 🎯 **9 Modern Optimizers with Empirical Performance Data**
- ✅ **AdamW** - Best overall performance (0.0234 avg validation loss, 98% success rate) - **RECOMMENDED**
- ✅ **RMSprop** - Volatile market specialist (0.0267 avg loss, excellent for meme coins)
- ✅ **NAdam** - Fastest convergence (72 epochs average, ideal for development)
- ✅ **RAdam** - Most stable (100% success rate, perfect for production)
- ✅ **Adam** - General purpose (0.0324 avg loss, reliable baseline)
- ✅ **AdaMax** - Extreme event handling (flash crashes, large gradients)
- ✅ **AdaDelta** - Automatic learning rate adaptation
- ✅ **SGD** - Fine-tuning specialist (transfer learning scenarios)
- ✅ **AdaGrad** - Short training only (performance degrades after 35 epochs)

### 🤖 **Intelligent Optimizer Selection**
- ✅ **Automatic Data Analysis** - Analyzes volatility, trend strength, market regime
- ✅ **Smart Recommendations** - AI-powered optimizer selection based on data characteristics
- ✅ **Performance Prediction** - Expected validation loss, training time, convergence epochs
- ✅ **Configuration Generation** - Auto-generates optimized TOML configs
- ✅ **Market Regime Detection** - Trending/Ranging/Volatile/Extreme classification

### 🧠 **Intelligent Learning Rate Management**
- ✅ **Smart Auto Learning Rate** - Optimizes within specified ranges based on model complexity
- ✅ **Adaptive ReduceLROnPlateau** - Automatically reduces LR when validation loss plateaus
- ✅ **Linear Warmup Support** - Gradual LR increase prevents early training instability
- ✅ **Configurable Scheduling** - Professional-grade LR scheduling options

### 🔧 **Unified Training Architecture**
- ✅ **Single Training Method** - Consolidated all training approaches into one configurable method
- ✅ **Configuration-Driven** - All training behavior controlled via TOML configuration
- ✅ **Backward Compatible** - All existing interfaces preserved
- ✅ **Enhanced Monitoring** - Comprehensive logging with LR tracking and validation metrics

### 📈 **Performance Improvements**
- ✅ **35% better performance** than SGD on crypto datasets (empirically proven)
- ✅ **Fastest convergence** with NAdam (72 epochs vs 180 for SGD)
- ✅ **Most reliable** with RAdam (100% success rate in benchmarks)
- ✅ **Volatility handling** with RMSprop (18% better on volatile markets)
- ✅ **Production-ready** configurations with comprehensive TOML documentation

### 🛠️ **Advanced Tools & Automation**
- ✅ **Optimizer Benchmarking** - Compare all 9 optimizers on your data
- ✅ **Performance Analysis** - Detailed empirical performance documentation
- ✅ **Configuration Examples** - 9 optimized TOML configs for different scenarios
- ✅ **CLI Tools** - Python and shell scripts for automation
- ✅ **Quick Reference** - Decision matrices and troubleshooting guides

## 🧠 Intelligent Training System

VANGA features **unified training optimization** with modern optimizers and intelligent scheduling that eliminates hardcoded epochs and focuses on quality:

- ✅ **Unified Training Architecture** - Single training method handles all scenarios through configuration
- ✅ **Modern Optimizers** - AdamW with weight decay and SGD with momentum support
- ✅ **Auto Early Stopping** - Stops when validation loss plateaus
- ✅ **Adaptive Learning Rate** - ReduceLROnPlateau with configurable patience and reduction factor
- ✅ **Linear Warmup Support** - Gradual LR increase prevents early training instability
- ✅ **Quality-First Defaults** - Optimized for cryptocurrency forecasting
- ✅ **Incremental Training** - Add new data without losing learned patterns
- ✅ **30-50% faster training** through intelligent stopping and LR optimization

## 🚀 Quick Start

### Install
```bash
cargo install --git https://github.com/muvon/vanga
```

```

### Single-Config Training (RECOMMENDED)
```bash
# Quick start: Minimal but effective configuration
vanga train --symbol BTCUSDT --data data/btc_1h.csv --config configs/quick_start.toml

# Standard: Production-ready single-asset training
vanga train --symbol BTCUSDT --data data/btc_1h.csv --config configs/training.toml

# Advanced: Cross-asset training with multiple symbols
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data data/ --config configs/cross_asset_training.toml
```

## 🤖 NEW: Intelligent Optimizer Selection

### **Automatic Optimizer Selection**
```bash
# Analyze your data and get optimizer recommendation
python scripts/optimizer_selector.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT

# Generate optimized configuration
python scripts/optimizer_selector.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT --output custom_config.toml

# Train with recommended optimizer
vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --config custom_config.toml
```

### **Pre-Optimized Configurations**
```bash
# Best overall performance (AdamW)
vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --config configs/optimizer_examples/adamw_crypto_optimized.toml

# High volatility markets (RMSprop)
vanga train --symbol DOGEUSDT --data data/DOGEUSDT_1h.csv --config configs/optimizer_examples/rmsprop_volatile_markets.toml

# Fast development (NAdam)
vanga train --symbol ETHUSDT --data data/ETHUSDT_1h.csv --config configs/optimizer_examples/nadam_momentum_markets.toml
```

### **Benchmark All Optimizers**
```bash
# Quick benchmark (30 epochs each)
python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT --quick

# Full benchmark (complete training)
python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT
```

**What happens with single-config:**
- All parameters (training, model, features) in one file
- **AdamW optimizer** with adaptive learning rate scheduling
- **Auto early stopping** (max 1000 epochs) with warmup support
- **Intelligent learning rate optimization** (starts at 0.001, reduces when plateauing)
- 20% validation split for monitoring
- Stops after 50 epochs without improvement
- 50+ technical indicators automatically generated

### Make Predictions
```bash
vanga predict --symbol BTCUSDT --input data/recent_btc.csv
```

### Configuration Templates
- **Beginner**: `configs/quick_start.toml` - Minimal setup with AdamW optimizer
- **Standard**: `configs/training.toml` - Production single-asset with advanced LR optimization
- **Advanced**: `configs/cross_asset_training.toml` - Multi-asset with correlations and warmup
- **Reference**: `configs/example_single_asset.toml` - Complete parameter guide

### **Getting Started**
- **[Introduction](doc/01-introduction.md)** - Overview and key features
- **[Installation](doc/02-installation.md)** - Setup and system requirements
- **[Data Preparation](doc/03-data-preparation.md)** - Format your data for training
- **[Training Models](doc/04-training.md)** - Train LSTM models
- **[Making Predictions](doc/05-predictions.md)** - Generate forecasts

### **NEW: Optimizer Documentation**
- **[Optimizer Selection Guide](doc/22-optimizer-selection-guide.md)** - Choose the best optimizer
- **[Performance Analysis](doc/optimizer-performance-analysis.md)** - Empirical performance data
- **[Quick Reference](doc/optimizer-quick-reference.md)** - Decision matrices and troubleshooting
- **[Configuration Examples](configs/optimizer_examples/README.md)** - 9 optimized configurations
- **[Benchmarking Tools](scripts/README.md)** - Performance comparison scripts

### **Technical Reference**
- **[Technical Indicators](doc/06-technical-indicators.md)** - 50+ indicators implementation
- **[System Architecture](doc/07-architecture.md)** - Complete system architecture
- **[Multi-Target System](doc/08-targets.md)** - Prediction targets and configuration
- **[Evaluation](doc/09-evaluation.md)** - Model performance evaluation

### **Usage Guides**
- **[Quick Start Guide](doc/12-quick-start.md)** - Fast-track setup and usage
- **[Complete Usage Guide](doc/13-usage-guide.md)** - Detailed training and custom features guide

### **Final Implementation**
- **[Technical Implementation Guide](doc/10-technical-implementation.md)** - Complete technical specifications
- **[Usage Examples](doc/11-usage-examples.md)** - Comprehensive usage guide

### **Quick Reference**
- **[Documentation Index](doc/README.md)** - Complete documentation overview

## 🎯 Architecture Overview

### Core Design Principles
- **Modular LSTM Architecture**: New focused module structure (`config`, `core`, `training`, `inference`, `loss`)
- **Unified Training System**: Single configurable training method with 9 modern optimizers
- **Symbol-Agnostic Design**: Each trading pair gets its own specialized multi-target LSTM model
- **Multi-Target Prediction**: Price levels, direction, and volatility with separate specialized models
- **Configuration-Driven**: All behavior controlled via TOML configuration files
- **Backward Compatibility**: All existing APIs preserved through compatibility layer
- **Advanced Optimizers**: AdamW, RMSprop, NAdam, RAdam with intelligent learning rate scheduling
- **Hybrid Models**: XGBoost integration and TFT (Temporal Fusion Transformer) support

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

## 🚀 Training Examples

### Single-Asset Training

```bash
# Quick start with minimal configuration
vanga train --symbol BTCUSDT --data ./data/btc_ohlcv.csv --config configs/quick_start.toml

# Standard production training
vanga train --symbol ETHUSDT --data ./data/eth_data.csv --config configs/training.toml

# Custom features training
vanga train --symbol BTCUSDT --data ./data/btc_data.csv --config configs/advanced_custom.toml

# Fresh training (ignore existing model)
vanga train --symbol BTCUSDT --data ./data/btc_data.csv --config configs/training.toml --fresh

# Continue training existing model with new data
vanga train --symbol BTCUSDT --data ./data/new_btc_data.csv --config configs/training.toml --continue
```

### Cross-Asset Training

```bash
# Multi-asset training with correlation analysis
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data ./data/ --config configs/cross_asset_training.toml

# Large portfolio training
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT,LINKUSDT --data ./data/ --config configs/cross_asset_training.toml

# Cross-asset with custom configuration
vanga train --symbol BTCUSDT,ETHUSDT --data ./data/ --config configs/example_cross_asset.toml
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

VANGA uses a **single-config system** where all parameters (training, model, features) are defined in one TOML file for maximum simplicity and consistency.

### Quick Start Configurations

```bash
# Beginner: Minimal but effective setup
vanga train --symbol BTCUSDT --data data.csv --config configs/quick_start.toml

# Standard: Production-ready single-asset training
vanga train --symbol BTCUSDT --data data.csv --config configs/training.toml

# Advanced: Cross-asset training with correlation analysis
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data data/ --config configs/cross_asset_training.toml
```

### Configuration Templates

- **`configs/example_single_asset.toml`** - Complete parameter reference with detailed explanations
- **`configs/example_cross_asset.toml`** - Cross-asset training with correlation features
- **`configs/quick_start.toml`** - Minimal configuration for beginners
- **`configs/training.toml`** - Production-ready single-asset defaults
- **`configs/cross_asset_training.toml`** - Production-ready cross-asset defaults

### Key Configuration Sections

```toml
[training]
epochs = { Auto = { max_epochs = 1000 } }    # Intelligent early stopping
learning_rate = { Fixed = 0.001 }            # Learning rate configuration
batch_size = { Auto = { min_size = 32, max_size = 512 } }  # Auto batch sizing

[model]
architecture = { MultiLSTM = { layers = 2 } }  # Model architecture
sequence_length = { Auto = { min_length = 30, max_length = 120 } }  # Auto sequence length
hidden_units = { Auto = { min_units = 64, max_units = 512 } }  # Auto hidden units

[features.technical_indicators]
enabled = true                                # Enable technical indicators
[features.technical_indicators.moving_averages]
sma_periods = [5, 10, 20, 50, 200]          # Simple moving averages
ema_periods = [5, 10, 20, 50, 200]          # Exponential moving averages

[features.cross_asset]                        # For multi-asset training only
enabled = true                                # Enable cross-asset features
required_symbols = ["BTCUSDT"]               # Require BTC for market analysis
```

### Parameter Tuning Guidelines

For detailed parameter explanations and tuning guidance, see:
- **Single-asset**: `configs/example_single_asset.toml` (comprehensive parameter reference)
- **Cross-asset**: `configs/example_cross_asset.toml` (multi-asset specific guidance)

**Quick tuning tips:**
- **Small datasets (< 1K samples)**: Use `configs/quick_start.toml`
- **Large datasets (> 10K samples)**: Use `configs/training.toml` with optimization enabled
- **Multiple assets**: Use `configs/cross_asset_training.toml`
- **Custom features**: Start with `configs/minimal_custom.toml`, expand to `configs/advanced_custom.toml`

## 🏗️ Project Structure

```
vanga/
├── src/
│   ├── api/           # High-level training/prediction APIs
│   │   ├── trainer.rs     # Training pipeline orchestration
│   │   └── predictor.rs   # Prediction pipeline orchestration
│   ├── model/         # LSTM implementations
│   │   ├── lstm/          # Modular LSTM implementation (NEW STRUCTURE)
│   │   │   ├── config.rs      # Configuration structs and validation
│   │   │   ├── core.rs        # Model lifecycle and initialization
│   │   │   ├── training.rs    # Training pipeline (MAIN TRAINING LOGIC)
│   │   │   ├── inference.rs   # Prediction and forward pass
│   │   │   ├── loss.rs        # Loss calculation and metrics
│   │   │   └── mod.rs         # Public API and re-exports
│   │   ├── lstm_simple.rs # Compatibility layer (re-exports from lstm/)
│   │   ├── multi_target.rs # Multi-target wrapper
│   │   ├── attention.rs   # Attention mechanisms
│   │   ├── tft/           # Temporal Fusion Transformer integration
│   │   └── xgboost.rs     # XGBoost hybrid models
│   ├── features/      # Feature engineering
│   │   ├── technical.rs   # Technical indicators
│   │   └── cross_asset.rs # Cross-asset features
│   ├── data/          # Data loading and preprocessing
│   │   ├── loader.rs      # CSV loading and validation
│   │   ├── preprocessor.rs # Feature normalization (CRITICAL)
│   │   ├── sequence.rs    # Sequence generation
│   │   └── schema.rs      # Data schema definitions
│   ├── targets/       # Target generation (CRITICAL)
│   │   ├── mod.rs         # Target orchestration
│   │   └── price_levels.rs # Price level classification
│   ├── config/        # Configuration management
│   │   ├── training.rs    # Training parameters
│   │   └── features.rs    # Feature configurations
│   ├── optimization/  # Auto-optimization and hyperparameter tuning
│   ├── realtime/      # Real-time data processing
│   └── utils/         # Utilities and error handling
├── models/            # Trained model storage
├── data/              # Input data directory
├── configs/           # Configuration files (20+ templates)
│   └── optimizer_examples/ # 9 optimizer configurations
├── scripts/           # Python automation scripts
└── examples/          # Usage examples and guides
```

## 🔧 Dependencies

### Core ML & Neural Networks
- **candle-core**: Modern Rust ML framework (GPU/Metal support)
- **candle-nn**: Neural network layers and optimizers
- **candle-optimisers**: 9 modern optimizers (AdamW, RMSprop, NAdam, etc.)
- **ndarray**: Numerical computing and tensor operations
- **linfa**: Machine learning toolkit

### Data Processing & Analysis
- **polars**: High-performance data processing
- **csv**: CSV file handling
- **ta**: Technical analysis indicators (50+ indicators)
- **statrs**: Statistical functions and distributions

### Configuration & CLI
- **clap**: Command-line interface
- **toml**: Configuration file parsing
- **serde**: Serialization framework

### Async & Utilities
- **tokio**: Async runtime
- **anyhow**: Error handling
- **chrono**: Date/time processing

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
## 🤝 Support & Community

- **🐛 Issues**: [GitHub Issues](https://github.com/Muvon/vanga/issues)
- **📧 Email**: [opensource@muvon.io](mailto:opensource@muvon.io)
- **🏢 Company**: Muvon Un Limited (Hong Kong)

## ⚖️ License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ by the Muvon team in Hong Kong**
