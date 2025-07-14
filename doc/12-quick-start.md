# VANGA Single-Config Quick Start Guide

## 🚀 **Single-Config Training Commands**

### **Quick Start Training (RECOMMENDED)**
```bash
# Beginner-friendly minimal configuration
vanga train --symbol BTCUSDT --data btc_data.csv --config configs/quick_start.toml
# Result: 2-layer MultiLSTM with essential technical indicators
```

### **Production Training**
```bash
# Full-featured production configuration
vanga train --symbol BTCUSDT --data btc_data.csv --config configs/training.toml
# Result: Optimized architecture with 50+ technical indicators and advanced features
```

### **Cross-Asset Training**
```bash
# Multi-asset training with correlation analysis
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data data/ --config configs/cross_asset_training.toml
# Result: Cross-asset LSTM with correlation features and market sentiment analysis
```

### **Custom Configuration Training**
```bash
# Use comprehensive example as starting point
vanga train --symbol BTCUSDT --data btc_data.csv --config configs/example_single_asset.toml

# Minimal custom features
vanga train --symbol BTCUSDT --data btc_data.csv --config configs/minimal_custom.toml

# Advanced custom features
vanga train --symbol BTCUSDT --data btc_data.csv --config configs/advanced_custom.toml
```

### **Training Mode Options**
```bash
# Fresh training: Always create new model
vanga train --symbol BTCUSDT --data btc_data.csv --config configs/training.toml --fresh

# Continue training: Add new data to existing model (uses unified training method)
vanga train --symbol BTCUSDT --data btc_data.csv --config configs/training.toml --continue-training
```

## 🔧 **Single-Config System**

### **Unified Configuration Loading**
All configuration parameters are loaded from a single TOML file:

```bash
# Single config file contains all parameters
vanga train --symbol BTCUSDT --data btc_data.csv --config configs/training.toml
# ✅ Configuration loaded and validated from: configs/training.toml
```

### **Available Configuration Templates**

| Template | Use Case | Features |
|----------|----------|----------|
| `quick_start.toml` | Beginners | Minimal but effective setup |
| `training.toml` | Production | Full-featured single-asset |
| `cross_asset_training.toml` | Multi-asset | Cross-asset correlations |
| `minimal_custom.toml` | Simple custom | Basic custom features |
| `advanced_custom.toml` | Complex custom | Advanced feature engineering |
| `example_single_asset.toml` | Reference | Complete parameter guide |
| `example_cross_asset.toml` | Reference | Cross-asset parameter guide |

### **Key Configuration Sections**

#### **Training Parameters**
```toml
[training]
epochs = { Auto = { max_epochs = 1000 } }        # Auto early stopping (RECOMMENDED)
learning_rate = { Fixed = 0.001 }                # Fixed learning rate
batch_size = { Auto = { min_size = 32, max_size = 512 } }  # Auto batch sizing
validation_split = 0.2                           # 20% validation split
early_stopping_patience = 50                     # Stop after 50 epochs without improvement
gradient_clip = 1.0                              # Gradient clipping (prevents exploding gradients)
```

**Parameter Tuning:**
- **epochs**: Use `Auto` for production, `Fixed` for reproducible experiments
- **learning_rate**: Start with 0.001, reduce to 0.0001 for fine-tuning
- **batch_size**: `Auto` optimizes memory usage, `Fixed` for consistent behavior
- **early_stopping_patience**: Increase (75-100) for complex models, decrease (25-40) for simple models

#### **Model Architecture**
```toml
[model]
architecture = { MultiLSTM = { layers = 2 } }    # Multi-layer LSTM (RECOMMENDED)
sequence_length = { Auto = { min_length = 30, max_length = 120 } }  # Auto sequence length
hidden_units = { Auto = { min_units = 64, max_units = 512 } }  # Auto hidden units

[model.dropout]
enabled = true                                    # Enable dropout regularization
rate = { Fixed = 0.2 }                          # 20% dropout rate
variational = true                               # Variational dropout (RECOMMENDED for LSTM)
recurrent = true                                 # Recurrent dropout
```

**Parameter Tuning:**
- **architecture**: MultiLSTM (general), StackedLSTM (sequential), BidirectionalLSTM (patterns)
- **layers**: 1-2 for small data, 2-3 for medium data, 3-4 for large datasets
- **sequence_length**: Auto optimizes for crypto patterns (typically 30-120)
- **dropout rate**: Increase (0.3-0.5) if overfitting, decrease (0.1-0.2) if underfitting

#### **Feature Configuration**
```toml
[features.technical_indicators]
enabled = true                                   # Enable technical indicators

[features.technical_indicators.moving_averages]
sma_periods = [5, 10, 20, 50, 200]             # Simple moving averages
ema_periods = [5, 10, 20, 50, 200]             # Exponential moving averages

[features.technical_indicators.momentum]
rsi_periods = [14, 21]                          # RSI periods
stochastic = true                               # Stochastic oscillator
williams_r = true                               # Williams %R

[features.custom_features]
enabled = true                                  # Enable custom features
include_all_numeric = true                      # Include all CSV numeric columns
exclude_features = ["unwanted_column"]          # Exclude specific features
```

**Parameter Tuning:**
- **periods**: Shorter (3, 5, 7) for scalping, longer (100, 200, 300) for position trading
- **indicators**: Enable more for complex patterns, disable for minimal setups
- **custom_features**: Include domain-specific features (sentiment, on-chain, macro)

#### **Cross-Asset Features (Multi-Asset Only)**
```toml
[features.cross_asset]
enabled = true                                  # Enable cross-asset features
min_symbols_required = 2                        # Minimum symbols for cross-asset
required_symbols = ["BTCUSDT"]                  # Require BTC for market analysis
btc_dominance_enabled = true                    # BTC dominance calculation
eth_btc_ratio_enabled = true                    # ETH/BTC ratio

[features.cross_asset.correlation_analysis]
enabled = true                                  # Enable correlation analysis
correlation_window = 50                         # Rolling correlation window
```

**Parameter Tuning:**
- **min_symbols_required**: 2-3 for basic cross-asset, 5+ for comprehensive analysis
- **correlation_window**: Shorter (20-30) for dynamic correlations, longer (100+) for stable
- **required_symbols**: Always include BTCUSDT for market context
validation_split = 0.2                # Data split validation (0.0 < x < 1.0)
early_stopping_patience = 10          # Early stopping (validated > 0)

[optimization]
method = "Bayesian"                   # Optimization method (validated)
n_trials = 50                        # Trial count (validated > 0)
timeout_seconds = 3600               # Timeout (validated > 0)
```

### **Configuration Error Handling**
Invalid configurations are caught early with detailed error messages:
```bash
# Example validation error
Error: validation_split + test_split must be < 1.0, got: 0.8 + 0.3 = 1.1
Error: gradient_clip must be positive, got: -1.0
Error: n_trials must be greater than 0
```

## 🏗️ **Multi-Layer Architecture Configuration**

### **Quick Architecture Templates**

#### **Production Quality (3-Layer MultiLSTM)**
```toml
# configs/production_multi_lstm.toml
[model]
architecture = "MultiLSTM"

[model.architecture_config.MultiLSTM]
layers = 3

[model.lstm]
hidden_size = 128
sequence_length = 60

[training]
[training.epochs]
type = "Auto"
max_epochs = 1000

[training.learning_rate]
type = "Adaptive"
initial_lr = 0.001
```

#### **Fast Training (2-Layer)**
```toml
# configs/fast_training.toml
[training]
epochs = { Fixed = 100 }
learning_rate = { Fixed = 0.001 }
batch_size = { Fixed = 64 }
validation_split = 0.2
early_stopping_patience = 20

[model]
architecture = { MultiLSTM = { layers = 2 } }
hidden_units = { Fixed = 64 }
sequence_length = { Fixed = 30 }
```

#### **Advanced Quality (4-Layer StackedLSTM)**
```toml
# configs/stacked_lstm.toml
[training]
epochs = { Auto = { max_epochs = 1500 } }
learning_rate = { Adaptive = { initial_lr = 0.0005 } }
batch_size = { Auto = { min_size = 16, max_size = 256 } }
validation_split = 0.2
test_split = 0.1
early_stopping_patience = 75
gradient_clip = 1.0

[model]
architecture = { StackedLSTM = { layers = 4 } }
hidden_units = { Fixed = 256 }
sequence_length = { Fixed = 120 }

[model.dropout]
enabled = true
rate = { Fixed = 0.3 }
variational = true
recurrent = true
```

### 📊 **Custom Features + Multi-Layer Workflow**

#### 1. **Prepare Enhanced CSV**
```csv
timestamp,open,high,low,close,volume,social_sentiment,funding_rate,whale_activity
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,0.75,-0.01,1250000
```

#### 2. **Configure Multi-Layer + Custom Features**
```toml
# configs/custom_multi_layer.toml - Training Configuration
[training]
epochs = { Auto = { max_epochs = 800 } }
learning_rate = { Adaptive = { initial_lr = 0.001 } }
batch_size = { Auto = { min_size = 32, max_size = 256 } }
validation_split = 0.2
early_stopping_patience = 40

[model]
architecture = { MultiLSTM = { layers = 3 } }  # Optimal for custom features
hidden_units = { Auto = { min_units = 128, max_units = 512 } }
sequence_length = { Auto = { min_length = 60, max_length = 120 } }

[model.attention]
enabled = true
mechanism = "SelfAttention"
heads = 8
```

Create separate features config `configs/custom_features.toml`:
```toml
[custom_features]
auto_include_all = true

[custom_features.transformations]
social_sentiment = "ZScore"
funding_rate = "PercentChange"
whale_activity = "Log"

[features.technical_indicators]
enabled = true  # 50+ indicators + custom features
```

#### 3. **Train Multi-Layer Model**
```bash
vanga train --symbol BTCUSDT --data enhanced_data.csv --config configs/custom_multi_layer.toml
# Result: 3-layer LSTM with 50+ technical indicators + custom features
```

## Training Behavior Explained

| Command | Existing Model | Behavior |
|---------|----------------|----------|
| `vanga train --symbol X --data Y` | ❌ Not found | ✅ Create new model |
| `vanga train --symbol X --data Y` | ✅ Found | ✅ Continue training |
| `vanga train --symbol X --data Y --fresh` | ✅ Found | ✅ Create new model (ignore existing) |
| `vanga train --symbol X --data Y --continue-training` | ❌ Not found | ❌ Error: No existing model |
| `vanga train --symbol X --data Y --continue-training` | ✅ Found | ✅ Continue training |

## Custom Features Examples

### 📈 **Sentiment Data**
```csv
timestamp,open,high,low,close,volume,social_sentiment,fear_greed_index,google_trends
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,0.75,25,67
```

**Configuration:**
```toml
[custom_features.transformations]
social_sentiment = "ZScore"
fear_greed_index = "MinMaxScale"
google_trends = "MinMaxScale"
```

### 🔗 **On-Chain Metrics**
```csv
timestamp,open,high,low,close,volume,active_addresses,nvt_ratio,mvrv_ratio
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,950000,45.2,1.8
```

**Configuration:**
```toml
[custom_features.transformations]
active_addresses = "Log"
nvt_ratio = "ZScore"
mvrv_ratio = "ZScore"
```

### 💰 **Market Microstructure**
```csv
timestamp,open,high,low,close,volume,funding_rate,open_interest,liquidations
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,-0.01,1500000000,25000000
```

**Configuration:**
```toml
[custom_features.transformations]
funding_rate = "PercentChange"
open_interest = "Log"
liquidations = "RobustScale"
```

## Transformation Types Guide

| Transformation | Use Case | Example Features |
|----------------|----------|------------------|
| `"ZScore"` | Normalize to mean=0, std=1 | sentiment scores, ratios, indices |
| `"MinMaxScale"` | Scale to [0,1] | percentages, bounded indices |
| `"PercentChange"` | Convert to % change | rates, price changes |
| `"Log"` | Handle skewed data | volumes, counts, addresses |
| `"RobustScale"` | Handle outliers | liquidations, whale activity |

## Configuration Templates

### 🎯 **Minimal Setup** (`configs/minimal_custom.toml`)
- Basic technical indicators
- Simple custom feature inclusion
- Lightweight feature engineering

### 🚀 **Standard Setup** (`configs/crypto_features.toml`)
- Comprehensive technical indicators
- Market microstructure features
- Balanced feature engineering

### 🔥 **Advanced Setup** (`configs/advanced_custom.toml`)
- All available indicators
- Complex feature interactions
- Maximum feature engineering

## Validation & Testing

### 🔍 **Data Validation**
```bash
# Validate CSV structure and features
python3 scripts/validate_features.py your_data.csv

# Get detailed analysis and recommendations
python3 scripts/validate_features.py your_data.csv --verbose

# Generate custom configuration
python3 scripts/validate_features.py your_data.csv --generate-config my_config.toml
```

### 🧪 **Test Complete Workflow**
```bash
# Run all examples and tests
./examples/run_examples.sh

# Test validation tools
./scripts/test_validation.sh
```

## Common Issues & Solutions

### ❌ **Model Input Size Mismatch**
```
Error: Model input size mismatch: existing model expects 45 features, but data has 52 features
```
**Solution:** Use `--fresh` to retrain with new feature structure
```bash
vanga train --symbol BTCUSDT --data new_data.csv --fresh
```

### ❌ **Continue Training Failed**
```
Error: Continue training requested but no existing model found
```
**Solution:** Train without `--continue-training` first
```bash
vanga train --symbol BTCUSDT --data data.csv
```

### ❌ **Missing Custom Features**
**Check:** Enable verbose logging to see feature detection
```bash
vanga train --symbol BTCUSDT --data data.csv --verbose
```

## Best Practices

### ✅ **Data Quality**
- Ensure minimal missing values in custom features
- Use consistent data frequency (match OHLCV intervals)
- Validate feature ranges and distributions

### ✅ **Feature Engineering**
- Start with `auto_include_all = true` for simplicity
- Use appropriate transformations for each feature type
- Monitor correlation and importance after training

### ✅ **Model Management**
- Use descriptive configuration file names
- Keep track of features used for each model
- Use `--fresh` when adding new features

### ✅ **Performance**
- Limit total features to avoid overfitting
- Remove highly correlated features
- Filter low-importance features

## File Structure

```
├── configs/
│   ├── crypto_features.toml      # Standard configuration
│   ├── minimal_custom.toml       # Minimal setup
│   └── advanced_custom.toml      # Advanced setup
├── examples/
│   ├── btc_with_sentiment.csv    # Sample sentiment data
│   ├── btc_with_onchain.csv      # Sample on-chain data
│   ├── sentiment_features.toml   # Sentiment config
│   ├── onchain_features.toml     # On-chain config
│   └── run_examples.sh           # Complete workflow demo
├── scripts/
│   ├── validate_features.py      # Data validation tool
│   └── test_validation.sh        # Validation testing
├── models/                       # Trained models (auto-created)
└── predictions/                  # Prediction outputs (auto-created)
```

---

**Ready to get started?**

1. 📊 **Validate your data:** `python3 scripts/validate_features.py your_data.csv`
2. 🔧 **Generate config:** `python3 scripts/validate_features.py your_data.csv --generate-config config.toml`
3. 🚀 **Train model:** `vanga train --symbol SYMBOL --data your_data.csv --features-config config.toml`
4. 🎯 **Make predictions:** `vanga predict --symbol SYMBOL --input new_data.csv`
vanga predict --symbol SYMBOL --input new_data.csv`
  # Sample on-chain data
│   ├── sentiment_features.toml   # Sentiment config
│   ├── onchain_features.toml     # On-chain config
│   └── run_examples.sh           # Complete workflow demo
├── scripts/
│   ├── validate_features.py      # Data validation tool
│   └── test_validation.sh        # Validation testing
├── models/                       # Trained models (auto-created)
└── predictions/                  # Prediction outputs (auto-created)
```

---

**Ready to get started?**

1. 📊 **Validate your data:** `python3 scripts/validate_features.py your_data.csv`
2. 🔧 **Generate config:** `python3 scripts/validate_features.py your_data.csv --generate-config config.toml`
3. 🚀 **Train model:** `vanga train --symbol SYMBOL --data your_data.csv --features-config config.toml`
4. 🎯 **Make predictions:** `vanga predict --symbol SYMBOL --input new_data.csv`
vanga predict --symbol SYMBOL --input new_data.csv`
