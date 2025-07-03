# VANGA Multi-Layer LSTM Quick Start Guide

## 🚀 **Multi-Layer LSTM Training Commands**

### **Auto-Optimized Training (RECOMMENDED)**
```bash
# Automatically selects optimal 2-3 layer architecture
vanga train --symbol BTCUSDT --data btc_data.csv
# Result: 3-layer MultiLSTM with 50+ technical indicators
```

### **Custom Architecture Training**
```bash
# Specify exact layer count and architecture
vanga train --symbol BTCUSDT --data btc_data.csv --config configs/multi_lstm_3layer.toml

# Fast 2-layer training for development
vanga train --symbol BTCUSDT --data btc_data.csv --config configs/fast_training.toml

# Advanced 4-layer training for maximum quality
vanga train --symbol BTCUSDT --data btc_data.csv --config configs/stacked_lstm.toml
```

### **Training Mode Options**
```bash
# Fresh training: Always create new multi-layer model
vanga train --symbol BTCUSDT --data btc_data.csv --fresh

# Continue training: Preserve existing layer architecture
vanga train --symbol BTCUSDT --data btc_data.csv --continue-training

# With custom features + multi-layer architecture
vanga train --symbol BTCUSDT --data btc_data.csv --features-config configs/custom.toml --config configs/multi_lstm.toml
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
[model]
architecture = "MultiLSTM"

[model.architecture_config.MultiLSTM]
layers = 2

[model.lstm]
hidden_size = 64
sequence_length = 30
```

#### **Advanced Quality (4-Layer StackedLSTM)**
```toml
# configs/stacked_lstm.toml
[model]
architecture = "StackedLSTM"

[model.architecture_config.StackedLSTM]
layers = 4

[model.lstm]
hidden_size = 256
sequence_length = 120
```

### 📊 **Custom Features + Multi-Layer Workflow**

#### 1. **Prepare Enhanced CSV**
```csv
timestamp,open,high,low,close,volume,social_sentiment,funding_rate,whale_activity
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,0.75,-0.01,1250000
```

#### 2. **Configure Multi-Layer + Custom Features**
```toml
# configs/custom_multi_layer.toml
[model]
architecture = "MultiLSTM"

[model.architecture_config.MultiLSTM]
layers = 3  # Optimal for custom features

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
