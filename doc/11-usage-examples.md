# VANGA Multi-Layer LSTM Usage Examples

## 🚀 **Complete Multi-Layer Usage Guide**

This document provides comprehensive usage examples for the VANGA multi-layer LSTM cryptocurrency forecasting system with intelligent architecture optimization.

---

## 📋 **Prerequisites**

### **System Requirements**
- Rust 1.87.0 or later
- Compiled VANGA binary (`./target/release/vanga`)
- CSV data files with OHLCV format

### **Data Format**
Your CSV files should contain the following columns:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,50000.0,51000.0,49500.0,50500.0,1000.0
2024-01-01 01:00:00,50500.0,51200.0,50000.0,51000.0,1200.0
...
```

---

## 🔧 **Configuration Parameters**

### **Training Parameters**
All training parameters can be configured via TOML files and are automatically validated:

```toml
[training]
# Gradient clipping (prevents exploding gradients in volatile crypto markets)
gradient_clip = 1.0                    # Threshold for gradient clipping (must be > 0)

# Data splits (automatically validated)
validation_split = 0.2                 # Must be between 0.0 and 1.0
test_split = 0.1                      # Must be between 0.0 and 1.0
                                      # Combined splits must be < 1.0

# Early stopping
early_stopping_patience = 10          # Must be > 0
```

### **Optimization Configuration**
Hyperparameter optimization is now fully configurable:

```toml
[optimization]
method = "Bayesian"                    # Bayesian, Grid, Random, or None
n_trials = 50                         # Number of optimization trials (must be > 0)
timeout_seconds = 3600                # Optimization timeout in seconds (must be > 0)
metric = "ValidationLoss"             # Optimization target metric
```

### **Configuration Validation**
All parameters are automatically validated when loading configuration files:
- **Range validation**: Ensures splits, timeouts, and trials are within valid ranges
- **Consistency checks**: Validates that validation_split + test_split < 1.0
- **Type safety**: Prevents invalid optimization methods or metrics
- **Error reporting**: Provides detailed error messages with actual values

## 🎯 **Multi-Layer LSTM Usage Examples**

### **1. Auto-Optimized Multi-Layer Training**

#### **Intelligent Training (RECOMMENDED)**
```bash
# Automatically selects optimal 2-3 layer architecture
./target/release/vanga train --symbol BTCUSDT --data data/btc_historical.csv
# Result: 3-layer MultiLSTM with 50+ technical indicators
```

#### **Expected Output:**
```
[INFO] Initializing multi-layer LSTM network with config: LSTMConfig { input_size: 52, hidden_size: 128, num_layers: 3, ... }
[INFO] ✅ LSTM layer 0 initialized: input_size=52, hidden_size=128
[INFO] ✅ LSTM layer 1 initialized: input_size=128, hidden_size=128
[INFO] ✅ LSTM layer 2 initialized: input_size=128, hidden_size=128
[INFO] Training multi-layer LSTM with 3 layers, input_size: 52
[INFO] Multi-layer LSTM training completed successfully
```

### **2. Custom Architecture Training**

#### **Fast 2-Layer Training**
```bash
# Fast training for development and testing
./target/release/vanga train \
    --symbol BTCUSDT \
    --data data/btc_historical.csv \
    --config configs/fast_training.toml
# Result: 2-layer MultiLSTM, ~5 minute training
```

#### **Advanced 4-Layer Training**
```bash
# Maximum quality training for production
./target/release/vanga train \
    --symbol BTCUSDT \
    --data data/btc_historical.csv \
    --config configs/stacked_lstm.toml
# Result: 4-layer StackedLSTM, ~20 minute training
```

#### **Bidirectional LSTM Training**
```bash
# Bidirectional processing for time series with future context
./target/release/vanga train \
    --symbol BTCUSDT \
    --data data/btc_historical.csv \
    --config configs/bidirectional_lstm.toml
# Result: 2-layer BidirectionalLSTM
```

### **3. Batch Multi-Layer Training**
```bash
# Train multi-layer models for multiple cryptocurrencies
./target/release/vanga train \
    --batch \
    --symbols BTCUSDT,ETHUSDT,ADAUSDT \
    --data-dir data/crypto/ \
    --fresh
```

### **2. Making Predictions**

#### **Simple Prediction**
```bash
# Make predictions for Bitcoin
./target/release/vanga predict \
    --symbol BTCUSDT \
    --input data/btc_recent.csv \
    --output predictions/btc_predictions.csv
```

#### **Multi-Horizon Predictions**
```bash
# Predict all available horizons
./target/release/vanga predict \
    --symbol BTCUSDT \
    --input data/btc_recent.csv \
    --all-horizons \
    --output predictions/btc_all_horizons.csv
```

#### **Predictions with Confidence Filtering**
```bash
# Only output predictions above 70% confidence
./target/release/vanga predict \
    --symbol BTCUSDT \
    --input data/btc_recent.csv \
    --min-confidence 0.7 \
    --output predictions/btc_high_confidence.csv
```

### **3. Model Management**

#### **List Available Models**
```bash
# Show all trained models
./target/release/vanga models list
```

#### **Model Evaluation** (Future Feature)
```bash
# Evaluate model performance
./target/release/vanga models evaluate \
    --symbol BTCUSDT \
    --test-data data/btc_test.csv
```

#### **Model Comparison** (Future Feature)
```bash
# Compare multiple models
./target/release/vanga models compare \
    --symbols BTCUSDT,ETHUSDT \
    --metric accuracy
```

---

## **Complete Workflow Examples**

### **End-to-End Bitcoin Forecasting**

#### **Step 1: Prepare Data**
```bash
# Ensure your data directory exists
mkdir -p data models predictions

# Place your Bitcoin historical data
# File: data/btc_historical.csv (training data)
# File: data/btc_recent.csv (prediction data)
```

#### **Step 2: Train Model**
```bash
# Train Bitcoin model with verbose logging
RUST_LOG=info ./target/release/vanga train \
    --symbol BTCUSDT \
    --data data/btc_historical.csv \
    --fresh
```

Expected output:
```
[INFO] Starting model training for symbol: BTCUSDT
[INFO] Loading training data from: data/btc_historical.csv
[INFO] Training data prepared: 1000 sequences, 55 features
[INFO] Starting LSTM training...
[INFO] Model training completed successfully
[INFO] Model saved to: ./models/BTCUSDT_model.bin
[INFO] Training completed successfully
```

#### **Step 3: Make Predictions**
```bash
# Generate predictions
./target/release/vanga predict \
    --symbol BTCUSDT \
    --input data/btc_recent.csv \
    --output predictions/btc_predictions.csv
```

Expected output:
```
[INFO] Starting prediction for symbol: BTCUSDT
[INFO] Loading prediction data from: data/btc_recent.csv
[INFO] Prediction data prepared: 100 sequences, 55 features
[INFO] Generating predictions...
[INFO] Generated 100 predictions
[INFO] Predictions saved to: predictions/btc_predictions.csv
[INFO] Prediction completed successfully
```

#### **Step 4: Review Results**
```bash
# Check the saved model
./target/release/vanga models list

# View prediction results
head predictions/btc_predictions.csv
```

### **Multi-Asset Portfolio Forecasting**

#### **Setup Multiple Assets**
```bash
# Create data structure
mkdir -p data/{btc,eth,ada} predictions models

# Place data files
# data/btc/historical.csv, data/btc/recent.csv
# data/eth/historical.csv, data/eth/recent.csv
# data/ada/historical.csv, data/ada/recent.csv
```

#### **Train All Models**
```bash
# Train Bitcoin model
./target/release/vanga train --symbol BTCUSDT --data data/btc/historical.csv

# Train Ethereum model
./target/release/vanga train --symbol ETHUSDT --data data/eth/historical.csv

# Train Cardano model
./target/release/vanga train --symbol ADAUSDT --data data/ada/historical.csv
```

#### **Generate Portfolio Predictions**
```bash
# Generate predictions for all assets
./target/release/vanga predict --symbol BTCUSDT --input data/btc/recent.csv --output predictions/btc.csv
./target/release/vanga predict --symbol ETHUSDT --input data/eth/recent.csv --output predictions/eth.csv
./target/release/vanga predict --symbol ADAUSDT --input data/ada/recent.csv --output predictions/ada.csv

# List all trained models
./target/release/vanga models list
```

---

## ⚙️ **Configuration Examples**

### **Custom Features Configuration**

Create `config/custom_features.toml`:
```toml
[technical_indicators]
enabled = true

[technical_indicators.moving_averages]
sma_periods = [5, 10, 20, 50]
ema_periods = [5, 10, 20, 50]

[technical_indicators.momentum]
rsi_periods = [14, 21]
stochastic_k_period = 14
stochastic_d_period = 3
williams_r_period = 14

[technical_indicators.volatility]
bollinger_period = 20
bollinger_std_dev = 2.0
atr_periods = [14, 21]

[technical_indicators.volume]
obv_enabled = true
volume_sma_periods = [10, 20]
mfi_period = 14
```

Use with training:
```bash
./target/release/vanga train \
    --symbol BTCUSDT \
    --data data/btc_historical.csv \
    --features-config config/custom_features.toml
```

### **Training Configuration**

Create `config/training.toml`:
```toml
[training]
epochs = { Fixed = 100 }
learning_rate = { Fixed = 0.001 }
batch_size = { Fixed = 32 }
validation_split = 0.2
test_split = 0.1
early_stopping_patience = 10
gradient_clip = 1.0

[model]
architecture = { MultiLSTM = { layers = 3 } }
hidden_units = { Fixed = 256 }
sequence_length = { Auto = { min_length = 30, max_length = 120 } }

[model.dropout]
enabled = true
rate = { Fixed = 0.2 }
variational = true
recurrent = true

[data]
normalization = "Robust"
sequence_overlap = 0.8
missing_data_strategy = "Interpolate"

[optimization]
enabled = false
trials = 50
metric = "ValidationLoss"
```

---

## 📊 **Output Format Examples**

### **Prediction Output**

The prediction CSV file will contain:
```csv
prediction
0.750123
0.823456
0.691234
0.789012
...
```

For multi-horizon predictions:
```csv
horizon_1h,horizon_4h,horizon_1d
0.750123,0.823456,0.691234
0.789012,0.856789,0.723456
0.712345,0.798012,0.667890
...
```

### **Model List Output**

Running `vanga models list` shows:
```
[INFO] Available models:
  - BTCUSDT (BTCUSDT_model.bin)
  - ETHUSDT (ETHUSDT_model.bin)
  - ADAUSDT (ADAUSDT_model.bin)
```

---

## 🔧 **Troubleshooting Examples**

### **Common Issues and Solutions**

#### **Model Not Found Error**
```bash
# Error: Model file not found
# Solution: Train the model first
./target/release/vanga train --symbol BTCUSDT --data data/btc_historical.csv
```

#### **Data Format Error**
```bash
# Error: Invalid CSV format
# Solution: Ensure CSV has required columns: timestamp,open,high,low,close,volume
head -n 5 data/btc_historical.csv
```

#### **Memory Issues with Large Files**
```bash
# For very large datasets, use chunked processing
# The system automatically handles this, but you can monitor with:
RUST_LOG=debug ./target/release/vanga train --symbol BTCUSDT --data large_data.csv
```

### **Verbose Logging for Debugging**
```bash
# Enable detailed logging
export RUST_LOG=debug
./target/release/vanga train --symbol BTCUSDT --data data/btc_historical.csv

# Or for specific modules
export RUST_LOG=vanga::model=debug,vanga::data=info
```

---

## 🚀 **Advanced Usage Patterns**

### **Automated Trading Pipeline**

Create a shell script `trading_pipeline.sh`:
```bash
#!/bin/bash

# Set up environment
export RUST_LOG=info
VANGA_BIN="./target/release/vanga"
DATA_DIR="data"
MODELS_DIR="models"
PREDICTIONS_DIR="predictions"

# Symbols to trade
SYMBOLS=("BTCUSDT" "ETHUSDT" "ADAUSDT")

# Train models if they don't exist
for symbol in "${SYMBOLS[@]}"; do
    if [ ! -f "$MODELS_DIR/${symbol}_model.bin" ]; then
        echo "Training model for $symbol..."
        $VANGA_BIN train --symbol $symbol --data "$DATA_DIR/${symbol}_historical.csv"
    fi
done

# Generate predictions
for symbol in "${SYMBOLS[@]}"; do
    echo "Generating predictions for $symbol..."
    $VANGA_BIN predict \
        --symbol $symbol \
        --input "$DATA_DIR/${symbol}_recent.csv" \
        --output "$PREDICTIONS_DIR/${symbol}_predictions.csv" \
        --min-confidence 0.7
done

# List all models
echo "Available models:"
$VANGA_BIN models list
```

Make executable and run:
```bash
chmod +x trading_pipeline.sh
./trading_pipeline.sh
```

### **Backtesting Workflow**

Create `backtest.sh`:
```bash
#!/bin/bash

SYMBOL="BTCUSDT"
TRAIN_DATA="data/btc_train.csv"
TEST_DATA="data/btc_test.csv"
RESULTS_DIR="backtest_results"

mkdir -p $RESULTS_DIR

# Train model on training data
echo "Training model..."
./target/release/vanga train --symbol $SYMBOL --data $TRAIN_DATA --fresh

# Generate predictions on test data
echo "Generating predictions..."
./target/release/vanga predict \
    --symbol $SYMBOL \
    --input $TEST_DATA \
    --output "$RESULTS_DIR/${SYMBOL}_backtest.csv"

echo "Backtesting complete. Results in $RESULTS_DIR/"
```

---

## 📈 **Performance Monitoring**

### **Timing Your Operations**
```bash
# Time training process
time ./target/release/vanga train --symbol BTCUSDT --data data/btc_historical.csv

# Time prediction process
time ./target/release/vanga predict --symbol BTCUSDT --input data/btc_recent.csv --output predictions.csv
```

### **Memory Usage Monitoring**
```bash
# Monitor memory usage during training
/usr/bin/time -v ./target/release/vanga train --symbol BTCUSDT --data data/btc_historical.csv
```

---

## 🎯 **Best Practices**

### **Data Management**
1. **Organize by symbol**: Keep data for each cryptocurrency in separate directories
2. **Regular updates**: Update historical data regularly for better model performance
3. **Data validation**: Ensure CSV files have proper OHLCV format
4. **Backup models**: Regularly backup trained models

### **Model Training**
1. **Fresh training**: Use `--fresh` flag when you have significantly new data
2. **Feature configuration**: Customize indicators based on your trading strategy
3. **Multiple horizons**: Train for multiple time horizons for comprehensive analysis
4. **Validation**: Always validate predictions on out-of-sample data

### **Production Usage**
1. **Automated pipelines**: Create scripts for regular model updates and predictions
2. **Monitoring**: Use logging to monitor system performance
3. **Error handling**: Implement proper error handling in your scripts
4. **Resource management**: Monitor memory and CPU usage for large datasets

---

## ✅ **Summary**

The VANGA LSTM system provides a complete cryptocurrency forecasting solution with:

- **Simple CLI interface** for easy operation
- **Flexible configuration** for customization
- **Production-ready performance** for real-world applications
- **Comprehensive error handling** for robust operation
- **Extensible architecture** for future enhancements

**Ready for professional cryptocurrency trading and analysis applications!** 🚀
