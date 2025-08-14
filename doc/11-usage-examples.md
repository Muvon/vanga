# VANGA Usage Examples

Comprehensive usage examples for VANGA's **unified training pipeline** cryptocurrency forecasting system with current CLI interface and configuration system.

## 🚀 **Current System Overview**

### **CLI Interface**
All operations use the `cargo run --` command structure:

```bash
# Training with unified pipeline
cargo run -- train --symbol SYMBOL --data PATH [OPTIONS]

# Prediction with 5-class output
cargo run -- predict --symbol SYMBOL --input PATH [OPTIONS]

# Backtesting with performance metrics
cargo run -- backtest --symbol SYMBOL --data PATH [OPTIONS]

# Real-time streaming with predictions
cargo run -- stream --symbol SYMBOL --data-path PATH [OPTIONS]
```

### **System Architecture**
- **5-Target Ordinal Prediction**: Price levels, direction, volatility, volume, sentiment (5 ordinal classes each)
- **Trading-Aware Ordinal Loss**: Optimized for trading profitability, not just accuracy
- **Adaptive Target Calibration**: Dynamic parameter optimization for balanced 20% per class
- **Modular LSTM**: Separate modules for training, inference, ordinal loss calculation
- **11 Advanced Optimizers**: AdamW, FracAdam, FracNAdam, RMSprop, NAdam, RAdam, Adam, AdaMax, AdaDelta, SGD, AdaGrad
- **Fractional Memory**: FracAdam and FracNAdam for volatile market conditions
- **Real-time Streaming**: Live ordinal predictions with file watching

## 📋 **Prerequisites**

### **System Requirements**
- Rust 1.87.0 or later
- CSV data files with OHLCV format
- Minimum 1000 rows for meaningful training

### **Data Format**
```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56
2024-01-01T01:00:00Z,42300.0,42800.0,42100.0,42600.0,1567.89
```

| Command | Purpose | Key Features |
|---------|---------|--------------|
| `vanga train` | Ordinal loss training | Single/multi-symbol, fresh/continue modes, 11 optimizers |
| `vanga predict` | Make predictions | Single/batch, all horizons, confidence filtering |
| `vanga models list` | List trained models | Shows available models and horizons |
| `vanga models evaluate` | Model evaluation | Backtesting, performance metrics |
| `vanga models compare` | Compare models | Multi-model performance comparison |
| `vanga models export` | Export models | Various formats for deployment |

### **CLI Command Structure**
```bash
# Training command structure
vanga train --symbol SYMBOL --data PATH [OPTIONS]

# Prediction command structure
vanga predict --symbol SYMBOL --input PATH [OPTIONS]

# Model management
vanga models SUBCOMMAND [OPTIONS]
```

---

## 🎯 **Training Examples**

### **1. Basic Training**

#### **Single Symbol Training**
```bash
# Basic training with default configuration
vanga train --symbol BTCUSDT --data data/btc_historical.csv

# Training with custom configuration
vanga train --symbol BTCUSDT --data data/btc_historical.csv --config configs/training.toml

# Fresh training (ignore existing model)
vanga train --symbol BTCUSDT --data data/btc_historical.csv --fresh
```

#### **Multi-Symbol Training**
```bash
# Train multiple symbols
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data data/

# Cross-asset training with correlation features
vanga train --symbol BTCUSDT,ETHUSDT --data data/ --config configs/cross_asset_training.toml
```

### **2. Advanced Training Options**

#### **Custom Horizons**
```bash
# Train for specific horizons
vanga train --symbol BTCUSDT --data data/btc_data.csv --horizons 1h,4h,1d

# Train for all available horizons
vanga train --symbol BTCUSDT --data data/btc_data.csv --horizons 1h,4h,1d,7d,30d
```

#### **Device Selection**
```bash
# Auto device selection (recommended)
vanga train --symbol BTCUSDT --data data/btc_data.csv --device auto

# Force CPU training
vanga train --symbol BTCUSDT --data data/btc_data.csv --device cpu

# Use specific GPU
vanga train --symbol BTCUSDT --data data/btc_data.csv --device gpu:0
```

#### **Advanced Features**
```bash
# Enable attention mechanism
vanga train --symbol BTCUSDT --data data/btc_data.csv --attention

# Enable TFT (Temporal Fusion Transformer)
vanga train --symbol BTCUSDT --data data/btc_data.csv --tft

# Override learning rate
vanga train --symbol BTCUSDT --data data/btc_data.csv --lr 0.001
```

### **3. Batch Training**
```bash
# Batch training (auto-detected when data is directory)
vanga train --symbol BTCUSDT,ETHUSDT --data data/

# Force batch mode
vanga train --symbol BTCUSDT,ETHUSDT --data data/ --batch
```

---

## 🔮 **Prediction Examples**

### **1. Basic Predictions**

#### **Single Symbol Prediction**
```bash
# Basic prediction with default horizon
vanga predict --symbol BTCUSDT --input data/btc_recent.csv

# Prediction with specific horizon
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --horizon 4h

# Prediction with output file
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --output predictions/btc.csv
```

#### **Multi-Horizon Predictions**
```bash
# Predict all available horizons
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --all-horizons

# Predict all horizons with output
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --all-horizons --output predictions/btc_all.csv
```

### **2. Batch Predictions**

#### **Multiple Symbols**
```bash
# Batch predictions for multiple symbols
vanga predict --symbol BTCUSDT,ETHUSDT,ADAUSDT --input-dir data/current/ --output predictions/

# Batch mode with confidence filtering
vanga predict --batch --input-dir data/current/ --output predictions/ --min-confidence 0.7
```

#### **Directory-Based Batch**
```bash
# Predict from directory (auto-detects symbol files)
vanga predict --symbol BTCUSDT --input-dir data/current/ --output predictions/

# Cross-asset predictions
vanga predict --symbol BTCUSDT,ETHUSDT --input data/cross_asset_data.csv --output predictions/cross_asset.csv
```

### **3. Advanced Prediction Options**

#### **Confidence Filtering**
```bash
# Only output high-confidence predictions
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --min-confidence 0.8

# Medium confidence threshold
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --min-confidence 0.6 --output predictions/medium_conf.csv
```

#### **Real-time Predictions**
```bash
# Real-time prediction mode
vanga predict --symbol BTCUSDT --realtime --source binance --interval 1m

# Real-time with custom interval
vanga predict --symbol BTCUSDT --realtime --source binance --interval 5m --output predictions/realtime.csv
```

#### **Device Selection**
```bash
# Use specific device for prediction
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --device gpu:0

# Force CPU prediction
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --device cpu
```

---

## 📊 **Model Management Examples**

### **1. List Available Models**
```bash
# List all trained models
vanga models list

# Expected output:
# Available models:
#   - BTCUSDT (horizons: 1h, 4h, 1d)
#   - ETHUSDT (horizons: 4h, 1d)
#   - ADAUSDT (horizons: 1h, 4h)
```

### **2. Model Evaluation**
```bash
# Evaluate single model
vanga models evaluate --symbol BTCUSDT --test-data data/btc_test.csv

# Evaluate with backtesting
vanga models evaluate --symbol BTCUSDT --test-data data/btc_test.csv --backtest

# Custom train/test split
vanga models evaluate --symbol BTCUSDT --test-data data/btc_test.csv --backtest --train-split 0.7
```

### **3. Model Comparison**
```bash
# Compare multiple models by accuracy
vanga models compare --symbols BTCUSDT,ETHUSDT --metric accuracy

# Compare by Sharpe ratio
vanga models compare --symbols BTCUSDT,ETHUSDT,ADAUSDT --metric sharpe_ratio

# Compare by directional accuracy
vanga models compare --symbols BTCUSDT,ETHUSDT --metric directional_accuracy
```

### **4. Model Export**
```bash
# Export model in MessagePack format
vanga models export --symbol BTCUSDT --format msgpack --output production/

# Export multiple models
vanga models export --symbols BTCUSDT,ETHUSDT --format msgpack --output production/

# Export with metadata
vanga models export --symbol BTCUSDT --format msgpack --output production/ --include-metadata
```

### **5. Model Ensemble**
```bash
# Create ensemble from multiple models
vanga models ensemble --symbols BTCUSDT,ETHUSDT,ADAUSDT --method weighted --output ensemble_model

# Ensemble with equal weights
vanga models ensemble --symbols BTCUSDT,ETHUSDT --method average --output equal_ensemble

# Performance-weighted ensemble
vanga models ensemble --symbols BTCUSDT,ETHUSDT,ADAUSDT --method performance --metric sharpe_ratio --output perf_ensemble
```

---

## 🚀 **Advanced Usage Patterns**

### **1. Model Comparison Workflow**

```bash
# Train multiple models with different configurations
vanga train --symbol BTCUSDT --data data/btc.csv --config configs/quick_start.toml --fresh
vanga train --symbol BTCUSDT --data data/btc.csv --config configs/training.toml --fresh
vanga train --symbol BTCUSDT --data data/btc.csv --config configs/advanced_custom.toml --fresh

# Compare model performance
vanga models compare --symbol BTCUSDT --metric sharpe_ratio
vanga models compare --symbol BTCUSDT --metric accuracy
```

### **2. Production Deployment Workflow**

```bash
# 1. Train production model
vanga train --symbol BTCUSDT --data data/btc_historical.csv --config configs/training.toml

# 2. Evaluate on test data
vanga models evaluate --symbol BTCUSDT --test-data data/btc_test.csv

# 3. Export for deployment
vanga models export --symbol BTCUSDT --format msgpack --output production/

# 4. Make real-time predictions
vanga predict --symbol BTCUSDT --input data/btc_current.csv --output predictions/
```

### **3. Research and Development Workflow**

```bash
# 1. Quick prototype with minimal config
vanga train --symbol BTCUSDT --data data/btc_sample.csv --config configs/quick_start.toml --fresh

# 2. Experiment with custom features
vanga train --symbol BTCUSDT --data data/btc_with_features.csv --config configs/minimal_custom.toml --fresh

# 3. Scale up with full features
vanga train --symbol BTCUSDT --data data/btc_full.csv --config configs/advanced_custom.toml --fresh

# 4. Cross-asset analysis
vanga train --symbol BTCUSDT,ETHUSDT --data data/ --config configs/cross_asset_training.toml --fresh
```

---

## 📈 **Performance Optimization Tips**

### **1. Training Speed Optimization**
- Use `configs/quick_start.toml` for development
- Enable fewer technical indicators for faster feature engineering
- Use smaller batch sizes for memory-constrained systems
- Disable attention mechanism for faster training

### **2. Accuracy Optimization**
- Use `configs/training.toml` or `configs/advanced_custom.toml`
- Enable all technical indicators and volatility features
- Use cross-asset training for portfolio models
- Enable hyperparameter optimization for production models

### **3. Memory Optimization**
- Use smaller sequence lengths (30-60)
- Reduce number of LSTM layers (1-2)
- Use smaller hidden units (64-128)
- Enable feature selection to reduce dimensionality

---

## 🔍 **Troubleshooting Common Issues**

### **Configuration Errors**
```bash
# Error: Invalid configuration parameter
[ERROR] Configuration validation failed: validation_split (0.5) + test_split (0.6) = 1.1 > 1.0

# Solution: Adjust splits in config file
validation_split = 0.2
test_split = 0.1
```

### **Data Issues**
```bash
# Error: Missing required columns
[ERROR] Missing required columns: ['close', 'volume']

# Solution: Ensure CSV has required OHLCV columns
timestamp,open,high,low,close,volume
```

### **Memory Issues**
```bash
# Error: Out of memory during training
[ERROR] CUDA out of memory

# Solution: Reduce batch size or model complexity
[model]
hidden_units = { Fixed = 64 }  # Reduce from default
[training]
batch_size = { Fixed = 32 }    # Reduce batch size
```

---

## 📚 **Next Steps**

1. **Start with Quick Start**: Use `configs/quick_start.toml` for your first model
2. **Explore Templates**: Review `configs/example_single_asset.toml` for parameter details
3. **Scale Up**: Move to `configs/training.toml` for production models
4. **Cross-Asset**: Try `configs/cross_asset_training.toml` for portfolio analysis
5. **Customize**: Create your own config based on the examples
    --data data/btc_historical.csv \
    --config configs/bidirectional_lstm.toml
# Result: 2-layer BidirectionalLSTM
```

### **3. Batch Multi-Layer Training**
```bash
# Train multi-layer models for multiple cryptocurrencies
vanga train \
    --batch \
    --symbols BTCUSDT,ETHUSDT,ADAUSDT \
    --data-dir data/crypto/ \
    --fresh
```

### **2. Making Predictions**

#### **Simple Prediction**
```bash
# Make predictions for Bitcoin
vanga predict \
    --symbol BTCUSDT \
    --input data/btc_recent.csv \
    --output predictions/btc_predictions.csv
```

#### **Multi-Horizon Predictions**
```bash
# Predict all available horizons
vanga predict \
    --symbol BTCUSDT \
    --input data/btc_recent.csv \
    --all-horizons \
    --output predictions/btc_all_horizons.csv
```

#### **Predictions with Confidence Filtering**
```bash
# Only output predictions above 70% confidence
vanga predict \
    --symbol BTCUSDT \
    --input data/btc_recent.csv \
    --min-confidence 0.7 \
    --output predictions/btc_high_confidence.csv
```

### **3. Model Management**

#### **List Available Models**
```bash
# Show all trained models
vanga models list
```

#### **Model Evaluation** (Future Feature)
```bash
# Evaluate model performance
vanga models evaluate \
    --symbol BTCUSDT \
    --test-data data/btc_test.csv
```

#### **Model Comparison** (Future Feature)
```bash
# Compare multiple models
vanga models compare \
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
RUST_LOG=info vanga train \
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
vanga predict \
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
vanga models list

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
vanga train --symbol BTCUSDT --data data/btc/historical.csv

# Train Ethereum model
vanga train --symbol ETHUSDT --data data/eth/historical.csv

# Train Cardano model
vanga train --symbol ADAUSDT --data data/ada/historical.csv
```

#### **Generate Portfolio Predictions**
```bash
# Generate predictions for all assets
vanga predict --symbol BTCUSDT --input data/btc/recent.csv --output predictions/btc.csv
vanga predict --symbol ETHUSDT --input data/eth/recent.csv --output predictions/eth.csv
vanga predict --symbol ADAUSDT --input data/ada/recent.csv --output predictions/ada.csv

# List all trained models
vanga models list
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
vanga train \
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
vanga train --symbol BTCUSDT --data data/btc_historical.csv
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
RUST_LOG=debug vanga train --symbol BTCUSDT --data large_data.csv
```

### **Verbose Logging for Debugging**
```bash
# Enable detailed logging
export RUST_LOG=debug
vanga train --symbol BTCUSDT --data data/btc_historical.csv

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
VANGA_BIN="vanga"
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
vanga train --symbol $SYMBOL --data $TRAIN_DATA --fresh

# Generate predictions on test data
echo "Generating predictions..."
vanga predict \
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
time vanga train --symbol BTCUSDT --data data/btc_historical.csv

# Time prediction process
time vanga predict --symbol BTCUSDT --input data/btc_recent.csv --output predictions.csv
```

### **Memory Usage Monitoring**
```bash
# Monitor memory usage during training
/usr/bin/time -v vanga train --symbol BTCUSDT --data data/btc_historical.csv
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
