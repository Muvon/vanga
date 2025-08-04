# VANGA Quick Start Guide

## 🚀 **Getting Started with Current System**

### **Basic Training**
```bash
# Train a Bitcoin model with default settings
vanga train --symbol BTCUSDT --data data/btc_data.csv

# Train with custom configuration
vanga train --symbol BTCUSDT --data data/btc_data.csv --config configs/training.toml
```

### **Basic Prediction**
```bash
# Make predictions with trained model
vanga predict --symbol BTCUSDT --input data/btc_recent.csv

# Predict all available horizons
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --all-horizons
```

### **Model Management**
```bash
# List available models
vanga models list

# Evaluate model performance
vanga models evaluate --symbol BTCUSDT --test-data data/btc_test.csv
```

## 🔧 **Current System Overview**

### **CLI Commands**
VANGA uses a command-line interface with the following main commands:

```bash
# Training
vanga train --symbol SYMBOL --data PATH [OPTIONS]

# Prediction
vanga predict --symbol SYMBOL --input PATH [OPTIONS]

# Model management
vanga models SUBCOMMAND [OPTIONS]
```

### **Key Features**
- **Multi-target system**: 3 targets (PriceLevel, Direction, Volatility) × 5 classes each
- **9 modern optimizers**: AdamW, RMSprop, NAdam, RAdam, Adam, AdaMax, AdaDelta, SGD, AdaGrad
- **Flexible configuration**: TOML files with comprehensive validation
- **Cross-asset support**: Train and predict multiple symbols simultaneously
- **Real-time predictions**: Live prediction capabilities
- **Model management**: List, evaluate, compare, and export models

## 📋 **Prerequisites**

### **System Requirements**
- Rust 1.87.0+ (install from https://rustup.rs/)
- Git for cloning the repository
- CSV data files with OHLCV format

### **Installation**
```bash
# Clone the repository
git clone https://github.com/your-org/vanga.git
cd vanga

# Build the project
cargo build --release

# The binary will be available at target/release/vanga
```

### **Data Format**
Your CSV files should contain these columns:
```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,50000.0,51000.0,49500.0,50500.0,1000.0
2024-01-01T01:00:00Z,50500.0,51200.0,50000.0,51000.0,1200.0
```

## 🎯 **Quick Start Examples**

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
