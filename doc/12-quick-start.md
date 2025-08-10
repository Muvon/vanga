# VANGA Quick Start Guide

Get up and running with VANGA's cryptocurrency forecasting system in minutes.

## 🚀 **Installation & Setup**

### **Prerequisites**
```bash
# Ensure you have Rust installed (1.87.0 or later)
rustc --version

# Clone the repository
git clone https://github.com/muvon/vanga.git
cd vanga

# Build the project (development mode)
cargo check --message-format=short  # Fast compilation check
```

### **Data Preparation**
```bash
# Your CSV data should have these columns:
# timestamp,open,high,low,close,volume
# 2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42200.0,1500.5

# Example data structure:
head -5 data/BTCUSDT_1h.csv
```

## 🎯 **Quick Training (5 Minutes)**

### **Step 1: Basic Training**
```bash
# Train a Bitcoin model with intelligent defaults
cargo run -- train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --config configs/quick_start.toml

# The system will automatically:
# - Generate 50+ technical indicators
# - Create 5 targets (price_levels, direction, volatility, sentiment, volume)
# - Use adaptive parameters for balanced class distribution
# - Apply early stopping when validation loss plateaus
# - Save the trained model to models/BTCUSDT/
```

### **Step 2: Make Predictions**
```bash
# Generate predictions with the trained model
cargo run -- predict --symbol BTCUSDT --input data/BTCUSDT_recent.csv

# Output: predictions.json with structured results
# - Price level predictions (5 classes)
# - Direction predictions (5 classes)
# - Volatility predictions (5 classes)
# - Sentiment predictions (5 classes)
# - Volume predictions (5 classes)
```

### **Step 3: Evaluate Performance**
```bash
# Run comprehensive backtesting
cargo run -- backtest --symbol BTCUSDT --data data/BTCUSDT_1h.csv

# Results include:
# - Classification accuracy for all 5 targets
# - Trading metrics (Sharpe ratio, max drawdown)
# - Directional accuracy
# - Confusion matrices
```

## 🔧 **Current System Architecture**

### **Multi-Target Prediction System**
VANGA predicts 5 different aspects of cryptocurrency markets:

1. **Price Levels** (5 classes): Strong Down, Moderate Down, Neutral, Moderate Up, Strong Up
2. **Direction** (5 classes): DUMP, DOWN, SIDEWAYS, UP, PUMP
3. **Volatility** (5 classes): Very Low, Low, Medium, High, Very High
4. **Sentiment** (5 classes): Strong Panic, Moderate Panic, Neutral, Moderate Greed, Strong Greed
5. **Volume** (5 classes): Very Low, Low, Medium, High, Very High

### **Key Features**
- **Modular LSTM Architecture**: Separate modules for training, inference, loss calculation
- **9 Modern Optimizers**: AdamW, RMSprop, NAdam, RAdam, Adam, AdaMax, AdaDelta, SGD, AdaGrad
- **Adaptive Parameters**: Automatic parameter calibration for balanced predictions
- **Real-time Streaming**: Live prediction capabilities with file watching
- **Comprehensive Evaluation**: Classification, regression, and trading metrics

## 📋 **Configuration Examples**

### **Quick Start Configuration (configs/quick_start.toml)**
```toml
[training]
# Auto early stopping with intelligent defaults
epochs = { Auto = { max_epochs = 1000 } }
learning_rate = 0.001
batch_size = { Auto = { min_size = 32, max_size = 512 } }
validation_split = 0.2
validation_gap = "1h"
test_split = 0.1
early_stopping = { patience = 50, min_delta = 0.00005 }
gradient_clip = 1.0
seed = 42  # Fixed seed for reproducible results

[model]
# Simple but effective architecture
architecture = { MultiLSTM = { layers = 2 } }
sequence_length = { Auto = { min_length = 30, max_length = 120 } }
hidden_units = { Auto = { min_units = 64, max_units = 512 } }

[model.dropout]
enabled = true
rate = { Fixed = 0.2 }
variational = true
recurrent = true

[targets]
# All 5 targets enabled with adaptive parameters
price_levels = { enabled = true, adaptive = true }
direction = { enabled = true, adaptive = true }
volatility = { enabled = true, adaptive = true }
sentiment = { enabled = true, adaptive = true }
volume = { enabled = true, adaptive = true }

[features]
# Comprehensive feature engineering
technical_indicators = { enabled = true }
cross_asset = { enabled = false }  # Disabled for single symbol
```

### **Advanced Configuration (configs/training.toml)**
```toml
[training]
# AdamW optimizer (recommended for crypto)
optimizer = { AdamW = { weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } }
epochs = { Auto = { max_epochs = 1000 } }
learning_rate = 0.001
batch_size = { Auto = { min_size = 16, max_size = 128 } }

[model]
# Multi-layer LSTM with attention
architecture = { MultiLSTM = { layers = 3 } }
sequence_length = { Auto = { min_length = 60, max_length = 240 } }
hidden_units = { Auto = { min_units = 128, max_units = 1024 } }

[model.attention]
enabled = true
heads = 8
dropout = 0.1

[features.technical_indicators]
enabled = true
sma_periods = [5, 10, 20, 50, 200]
ema_periods = [5, 10, 20, 50, 200]
rsi_periods = [14, 21]
```

## 🎯 **Step-by-Step Tutorial**

### **Step 1: Prepare Your Data**
```bash
# Ensure your CSV has the required format
head -5 data/BTCUSDT_1h.csv
# timestamp,open,high,low,close,volume
# 2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42200.0,1500.5
# 2024-01-01T01:00:00Z,42200.0,42800.0,42000.0,42600.0,1200.3
```

### **Step 2: Train Your First Model**
```bash
# Basic training with quick start configuration
cargo run -- train \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --config configs/quick_start.toml

# Training output:
# 🚀 Starting VANGA LSTM Training...
# 📊 Data loaded: 10,000 samples
# 🔧 Features generated: 127 technical indicators
# 🎯 Targets created: 5 targets × 5 classes each
# 📈 Training progress: [████████████████████] 100%
# ✅ Model saved to: models/BTCUSDT/
```

### **Step 3: Make Predictions**
```bash
# Generate predictions on new data
cargo run -- predict \
    --symbol BTCUSDT \
    --input data/BTCUSDT_recent.csv \
    --output predictions.json

# Prediction output structure:
# {
#   "symbol": "BTCUSDT",
#   "timestamp": "2024-08-10T16:17:40Z",
#   "horizon": "1h",
#   "price_levels": { "bins": {...}, "prediction": "MODERATE_UP" },
#   "direction": { "prediction": "UP", "confidence": 0.75 },
#   "volatility": { "regime": "MEDIUM", "confidence": 0.68 },
#   "sentiment": { "regime": "MODERATE_GREED", "confidence": 0.72 },
#   "volume": { "regime": "HIGH", "confidence": 0.65 }
# }
```

### **Step 4: Evaluate Performance**
```bash
# Run comprehensive backtesting
cargo run -- backtest \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --train-split 0.8

# Backtest results:
# 📊 Backtest Results for BTCUSDT:
#   Training Period: 2024-01-01 to 2024-06-30
#   Test Period: 2024-07-01 to 2024-08-10
#   Overall Accuracy: 69.4%
#   Price Levels F1: 0.65
#   Direction F1: 0.70
#   Volatility F1: 0.73
#   Sharpe Ratio: 1.85
#   Max Drawdown: 12.3%
```

## 🔍 **Real-Time Streaming**

### **Live Prediction Setup**
```bash
# Start real-time predictions
cargo run -- stream \
    --symbol BTCUSDT \
    --data-path data/live_btc.csv \
    --output-path live_predictions.json \
    --interval 300  # 5 minutes

# The system will:
# - Monitor the CSV file for new rows
# - Generate predictions every 5 minutes
# - Output structured JSON predictions
# - Handle errors gracefully with recovery
```

## 🛠 **Development Workflow**

### **Fast Development Cycle**
```bash
# Use these commands during development (much faster than --release)
cargo check --message-format=short  # Fast compilation check
cargo clippy --all-features --all-targets -- -D warnings  # Code quality
cargo test  # Run tests

# Only use --release for production (extremely slow during development)
```

### **Configuration Testing**
```bash
# Test different configurations quickly
cargo run -- train --symbol BTCUSDT --data data/small_sample.csv --config configs/debug.toml

# Available configurations:
ls configs/
# quick_start.toml          - Beginner-friendly
# training.toml             - Full-featured
# crypto_hybrid.toml        - Multi-asset
# attention_moh.toml        - With attention mechanism
# gradient_fix.toml         - Gradient debugging
```

## 📊 **Understanding Output**

### **Training Output**
```
🚀 Starting VANGA LSTM Training for BTCUSDT...
📊 Data loaded: 8,760 samples (1 year of hourly data)
🔧 Features: 127 technical indicators generated
🎯 Targets: 5 targets with adaptive parameters
   - Price Levels: [20%, 20%, 20%, 20%, 20%] (balanced)
   - Direction: [18%, 22%, 20%, 22%, 18%] (balanced)
   - Volatility: [15%, 25%, 30%, 25%, 5%] (crypto-adjusted)

📈 Training Progress:
Epoch 1/1000: Loss=1.234, Val_Loss=1.456, Accuracy=0.456
Epoch 50/1000: Loss=0.789, Val_Loss=0.823, Accuracy=0.678
...
Early stopping at epoch 234: Val_Loss=0.654, Accuracy=0.694

✅ Model saved to: models/BTCUSDT/
📊 Final Metrics:
   - Overall Accuracy: 69.4%
   - Training Time: 12m 34s
   - Model Size: 15.2 MB
```

### **Prediction Output**
```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2024-08-10T16:17:40Z",
  "horizon": "1h",
  "current_price": 42500.0,
  "price_levels": {
    "prediction": "MODERATE_UP",
    "confidence": 0.68,
    "bins": {
      "strong_down": {"range": [-15.0, -8.0], "probability": 0.15},
      "moderate_down": {"range": [-8.0, -3.0], "probability": 0.25},
      "neutral": {"range": [-3.0, 3.0], "probability": 0.20},
      "moderate_up": {"range": [3.0, 8.0], "probability": 0.25},
      "strong_up": {"range": [8.0, 15.0], "probability": 0.15}
    }
  },
  "direction": {
    "prediction": "UP",
    "confidence": 0.75,
    "probabilities": {
      "dump": 0.05, "down": 0.15, "sideways": 0.20, "up": 0.45, "pump": 0.15
    }
  },
  "volatility": {
    "regime": "MEDIUM",
    "confidence": 0.68,
    "probabilities": {
      "very_low": 0.10, "low": 0.20, "medium": 0.40, "high": 0.25, "very_high": 0.05
    }
  }
}
```

## 🚨 **Common Issues & Solutions**

### **Build Issues**
```bash
# If compilation fails:
cargo clean
cargo check --message-format=short

# If you see "linker not found":
# Install build tools for your platform
# Windows: Install Visual Studio Build Tools
# macOS: xcode-select --install
# Linux: sudo apt install build-essential
```

### **Data Issues**
```bash
# If training fails with "Invalid CSV format":
# Ensure your CSV has exactly these columns:
# timestamp,open,high,low,close,volume

# If you get "Insufficient data":
# Ensure you have at least 1000 rows for meaningful training
wc -l data/your_data.csv
```

### **Memory Issues**
```bash
# If you run out of memory during training:
# Reduce batch size in your config:
batch_size = { Fixed = 16 }  # Instead of Auto

# Or reduce sequence length:
sequence_length = { Fixed = 60 }  # Instead of Auto
```

## 🎉 **Next Steps**

Once you have a working model:

1. **Experiment with Configurations**: Try different optimizers and architectures
2. **Multi-Symbol Training**: Train models for multiple cryptocurrencies
3. **Real-Time Integration**: Set up live prediction streaming
4. **Performance Optimization**: Fine-tune hyperparameters for your specific use case
5. **Custom Features**: Add domain-specific technical indicators

**Ready to start forecasting cryptocurrency prices with VANGA!** 🚀

### **1. Train Your First Model**
```bash
# Basic training (uses default configuration)
vanga train --symbol BTCUSDT --data data/btc_historical.csv

# Expected output:
# [INFO] Starting model training for symbol: BTCUSDT
# [INFO] Loading training data from: data/btc_historical.csv
# [INFO] Training data prepared: 1000 sequences, 55 features
# [INFO] Model training completed successfully
```

### **2. Make Predictions**
```bash
# Basic prediction
vanga predict --symbol BTCUSDT --input data/btc_recent.csv

# Predict all horizons
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --all-horizons

# Save predictions to file
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --output predictions.csv
```

### **3. Check Your Models**
```bash
# List available models
vanga models list

# Evaluate model performance
vanga models evaluate --symbol BTCUSDT --test-data data/btc_test.csv
```

## 🔧 **Advanced Options**

### **Training with Custom Configuration**
```bash
# Use custom configuration file
vanga train --symbol BTCUSDT --data data/btc_data.csv --config configs/custom.toml

# Fresh training (ignore existing model)
vanga train --symbol BTCUSDT --data data/btc_data.csv --fresh

# Continue training existing model
vanga train --symbol BTCUSDT --data data/btc_data.csv --continue-training
```

### **Multi-Symbol Training**
```bash
# Train multiple symbols
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data data/

# Cross-asset training
vanga train --symbol BTCUSDT,ETHUSDT --data data/ --config configs/cross_asset.toml
```

### **Advanced Features**
```bash
# Enable attention mechanism
vanga train --symbol BTCUSDT --data data/btc_data.csv --attention

# Enable TFT (Temporal Fusion Transformer)
vanga train --symbol BTCUSDT --data data/btc_data.csv --tft

# Custom learning rate
vanga train --symbol BTCUSDT --data data/btc_data.csv --lr 0.001
```

### **Prediction Options**
```bash
# Specific horizon
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --horizon 4h

# Confidence filtering
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --min-confidence 0.8

# Batch predictions
vanga predict --batch --input-dir data/current/ --output predictions/
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Model Not Found**
```bash
# Error: Model file not found for symbol: BTCUSDT
# Solution: Train the model first
vanga train --symbol BTCUSDT --data data/btc_data.csv
```

#### **Data Format Error**
```bash
# Error: Missing required columns
# Solution: Ensure CSV has required OHLCV columns
# Required: timestamp,open,high,low,close,volume
```

#### **Configuration Error**
```bash
# Error: Configuration validation failed
# Solution: Check your TOML file syntax and parameter values
vanga train --symbol BTCUSDT --data data/btc_data.csv --config configs/training.toml
```

### **Getting Help**
```bash
# Show help for main command
vanga --help

# Show help for training
vanga train --help

# Show help for prediction
vanga predict --help

# Show help for model management
vanga models --help
```

## 📚 **Next Steps**

1. **Start Simple**: Begin with basic training using default settings
2. **Explore Configurations**: Try different TOML configuration files
3. **Multi-Symbol**: Experiment with cross-asset training
4. **Advanced Features**: Enable attention mechanism or TFT
5. **Model Management**: Use evaluation and comparison tools

### **Configuration Files**
Check the `configs/` directory for example configurations:
- `training.toml` - Production configuration
- `quick_start.toml` - Beginner-friendly setup
- `cross_asset_training.toml` - Multi-asset configuration
- `adamw_crypto_optimized.toml` - Crypto-optimized settings

### **File Structure**
```
vanga/
├── configs/           # Configuration templates
├── data/             # Your CSV data files
├── models/           # Trained models (auto-created)
├── predictions/      # Prediction outputs (auto-created)
└── target/release/   # Compiled binary
    └── vanga         # Main executable
```

**Ready to start forecasting cryptocurrency prices with VANGA!** 🚀
