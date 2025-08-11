# VANGA LSTM Usage Guide

Comprehensive guide for using VANGA's cryptocurrency forecasting system with current modular LSTM architecture and CLI interface.

## 🚀 **Current CLI Interface**

### **Main Commands**
```bash
# Training
cargo run -- train --symbol SYMBOL --data PATH [OPTIONS]

# Prediction
cargo run -- predict --symbol SYMBOL --input PATH [OPTIONS]

# Backtesting
cargo run -- backtest --symbol SYMBOL --data PATH [OPTIONS]

# Real-time streaming
cargo run -- stream --symbol SYMBOL --data-path PATH [OPTIONS]
```

## 🎯 **Training Workflows**

### **Basic Training**
```bash
# Train a new model with default configuration
cargo run -- train --symbol BTCUSDT --data data/BTCUSDT_1h.csv

# Train with custom configuration
cargo run -- train \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --config configs/training.toml

# Train with specific horizons
cargo run -- train \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --horizons 1h,4h,1d
```

### **Advanced Training Options**
```bash
# Fresh training (ignore existing model)
cargo run -- train \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --fresh

# Continue training existing model
cargo run -- train \
    --symbol BTCUSDT \
    --data data/new_btc_data.csv \
    --continue-training

# Training with attention mechanism
cargo run -- train \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --attention

# GPU training
cargo run -- train \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --device cuda:0
```

### **Training Configuration Examples**

#### **Quick Start Configuration**
```toml
# configs/quick_start.toml
[training]
epochs = { Auto = { max_epochs = 1000 } }
learning_rate = 0.001
batch_size = { Auto = { min_size = 32, max_size = 512 } }
validation_split = 0.2
early_stopping = { patience = 50, min_delta = 0.00005 }
seed = 42

[model]
architecture = { MultiLSTM = { layers = 2 } }
sequence_length = { Auto = { min_length = 30, max_length = 120 } }
hidden_units = { Auto = { min_units = 64, max_units = 512 } }

[targets]
price_levels = { enabled = true, adaptive = true }
direction = { enabled = true, adaptive = true }
volatility = { enabled = true, adaptive = true }
sentiment = { enabled = true, adaptive = true }
volume = { enabled = true, adaptive = true }
```

#### **Advanced Configuration**
```toml
# configs/advanced.toml
[training]
optimizer = { AdamW = { weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } }
epochs = { Auto = { max_epochs = 1000 } }
learning_rate = 0.001
batch_size = { Auto = { min_size = 16, max_size = 128 } }

[model]
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

## 🔮 **Prediction Workflows**

### **Basic Predictions**
```bash
# Generate predictions for a symbol
cargo run -- predict \
    --symbol BTCUSDT \
    --input data/BTCUSDT_recent.csv

# Predict with specific output file
cargo run -- predict \
    --symbol BTCUSDT \
    --input data/BTCUSDT_recent.csv \
    --output predictions.json

# Predict specific horizon
cargo run -- predict \
    --symbol BTCUSDT \
    --input data/BTCUSDT_recent.csv \
    --horizon 4h
```

### **Batch Predictions**
```bash
# Predict multiple files in a directory
cargo run -- predict \
    --symbol BTCUSDT \
    --input-dir data/batch_predictions/ \
    --batch

# Predict all available horizons
cargo run -- predict \
    --symbol BTCUSDT \
    --input data/BTCUSDT_recent.csv \
    --all-horizons
```

### **Advanced Prediction Options**
```bash
# Predictions with confidence filtering
cargo run -- predict \
    --symbol BTCUSDT \
    --input data/BTCUSDT_recent.csv \
    --min-confidence 0.7

# GPU predictions
cargo run -- predict \
    --symbol BTCUSDT \
    --input data/BTCUSDT_recent.csv \
    --device cuda:0
```

## 🔄 **Real-Time Streaming**

### **Live Prediction Setup**
```bash
# Start real-time predictions
cargo run -- stream \
    --symbol BTCUSDT \
    --data-path data/live_btc.csv \
    --output-path live_predictions.json \
    --interval 5m

# Real-time with custom configuration
cargo run -- stream \
    --symbol BTCUSDT \
    --data-path data/live_btc.csv \
    --output-path live_predictions.json \
    --interval 1m \
    --min-confidence 0.6
```

### **Streaming Configuration**
```toml
# configs/streaming.toml
[realtime]
symbol = "BTCUSDT"
data_path = "data/live_btc.csv"
output_path = "live_predictions.json"
prediction_interval = "5m"
buffer_size = 1000
min_confidence = 0.7
output_format = "JSON"

[realtime.file_watcher]
enabled = true
poll_interval = "1s"
```

## 📊 **Backtesting**

### **Basic Backtesting**
```bash
# Run comprehensive backtest
cargo run -- backtest \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv

# Backtest with custom train/test split
cargo run -- backtest \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --train-split 0.8
```

### **Walk-Forward Analysis**
```bash
# Advanced backtesting with walk-forward
cargo run -- backtest \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --walk-forward \
    --train-window 1000 \
    --test-window 100 \
    --step-size 50
```

## 🛠 **API Usage Examples**

### **Training API**
```rust
use vanga::api::train_model;
use vanga::config::TrainingConfig;

// Load configuration
let config = TrainingConfig::from_file("configs/training.toml")?;

// Train model
let model = train_model(config).await?;

// Model is automatically saved to models/{symbol}/
println!("Model trained and saved successfully!");
```

### **Prediction API**
```rust
use vanga::api::{Predictor, ModelWrapper};
use vanga::config::PredictionConfig;
use vanga::model::multi_target::MultiTargetLSTMModel;

// Load trained model
let model = MultiTargetLSTMModel::load("models/BTCUSDT")?;

// Configure prediction
let config = PredictionConfig {
    symbols: vec!["BTCUSDT".to_string()],
    input_path: "data/recent_data.csv".into(),
    output_path: Some("predictions.json".into()),
    horizons: vec!["1h".to_string(), "4h".to_string()],
    device: DeviceConfig::Auto,
    min_confidence: 0.0,
    output: OutputConfig {
        format: OutputFormat::JSON,
        include_metadata: true,
        include_orders: true,
        include_adaptive_signals: true,
    },
    batch_size: None,
};

// Make predictions
let predictor = Predictor::new(config);
let results = predictor.predict(ModelWrapper::MultiTarget(&model)).await?;

// Process results
for result in results {
    println!("Symbol: {}", result.symbol);
    println!("Timestamp: {}", result.timestamp);
    println!("Overall Confidence: {:.3}", result.confidence);

    if let Some(price_levels) = result.price_levels {
        println!("Price Level Prediction: {:?}", price_levels.prediction);
    }

    if let Some(direction) = result.direction {
        println!("Direction: {} (confidence: {:.3})", direction.prediction, direction.confidence);
    }
}
```

### **Backtesting API**
```rust
use vanga::api::backtester::{Backtester, BacktestConfig};

// Configure backtesting
let config = BacktestConfig {
    symbol: "BTCUSDT".to_string(),
    train_split: 0.8,
    data_path: "data/BTCUSDT_1h.csv".into(),
};

// Run backtest
let backtester = Backtester::new(config);
let results = backtester.run_backtest().await?;

println!("Backtest Results:");
println!("  Training Period: {} to {}", results.train_period.0, results.train_period.1);
println!("  Test Period: {} to {}", results.test_period.0, results.test_period.1);
println!("  Directional Accuracy: {:.3}%", results.directional_accuracy * 100.0);
println!("  RMSE: {:.4}", results.regression_metrics.rmse);
```

## 📁 **Data Management**

### **CSV Data Requirements**
```csv
# Required columns (OHLCV format)
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56
2024-01-01T01:00:00Z,42300.0,42800.0,42100.0,42600.0,1567.89
```

### **Data Validation**
```bash
# Validate data format before training
cargo run -- validate-data --file data/BTCUSDT_1h.csv

# Check data statistics
cargo run -- data-info --file data/BTCUSDT_1h.csv
```

### **Custom Features**
```csv
# Optional: Add custom features as additional columns
timestamp,open,high,low,close,volume,social_sentiment,funding_rate
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,0.75,-0.01
2024-01-01T01:00:00Z,42300.0,42800.0,42100.0,42600.0,1567.89,0.82,0.02
```

## 🔧 **Configuration Management**

### **Available Configurations**
```bash
# List all available configurations
ls configs/
# quick_start.toml          - Beginner-friendly
# training.toml             - Full-featured
# crypto_hybrid.toml        - Multi-asset
# attention_moh.toml        - With attention mechanism
# gradient_fix.toml         - Gradient debugging
```

### **Configuration Validation**
```bash
# Validate configuration file
cargo run -- validate-config --config configs/training.toml

# Show configuration details
cargo run -- config-info --config configs/training.toml
```

### **Custom Configuration Creation**
```toml
# configs/my_custom.toml
[training]
epochs = { Auto = { max_epochs = 500 } }
learning_rate = 0.0005
batch_size = { Fixed = 64 }
validation_split = 0.15
early_stopping = { patience = 30, min_delta = 0.0001 }

[model]
architecture = { MultiLSTM = { layers = 2 } }
sequence_length = { Fixed = 90 }
hidden_units = { Fixed = 256 }

[model.dropout]
enabled = true
rate = { Fixed = 0.3 }

[targets]
# Enable only specific targets
price_levels = { enabled = true, adaptive = true }
direction = { enabled = true, adaptive = true }
volatility = { enabled = false }
sentiment = { enabled = false }
volume = { enabled = false }
```

## 🚀 **Performance Optimization**

### **Development vs Production**
```bash
# Development (fast compilation)
cargo check --message-format=short
cargo run -- train --symbol BTCUSDT --data data/small_sample.csv

# Production (optimized performance)
cargo build --release
./target/release/vanga train --symbol BTCUSDT --data data/full_dataset.csv
```

### **Memory Optimization**
```toml
# For large datasets, reduce memory usage
[training]
batch_size = { Fixed = 16 }  # Smaller batches
sequence_length = { Fixed = 60 }  # Shorter sequences

[model]
hidden_units = { Fixed = 128 }  # Smaller model
```

### **GPU Acceleration**
```bash
# Use GPU for training (if available)
cargo run -- train \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --device cuda:0

# Check GPU availability
cargo run -- device-info
```

## 🔍 **Debugging & Troubleshooting**

### **Verbose Logging**
```bash
# Enable detailed logging
RUST_LOG=debug cargo run -- train \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --verbose

# Log to file
RUST_LOG=info cargo run -- train \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv 2> training.log
```

### **Common Issues**
```bash
# If training fails with "Insufficient data"
wc -l data/your_data.csv  # Ensure at least 1000 rows

# If prediction fails with "Model not found"
ls models/BTCUSDT/  # Check if model exists

# If memory issues occur
# Reduce batch_size and sequence_length in config
```

### **Model Inspection**
```bash
# Check model details
cargo run -- model-info --symbol BTCUSDT

# List all trained models
cargo run -- models list

# Compare model performance
cargo run -- models compare --symbols BTCUSDT,ETHUSDT
```

This comprehensive usage guide covers all aspects of using VANGA's current API and CLI interface for cryptocurrency forecasting.
]

# Optional: Apply transformations
[custom_features.transformations]
social_sentiment = "ZScore"      # Normalize to z-score
funding_rate = "PercentChange"   # Convert to percentage change
whale_activity = "Log"           # Apply log transformation
```

#### Selective Custom Features
```toml
# configs/selective_features.toml
[custom_features]
auto_include_all = false  # Only include specified features

# Explicitly include specific features
include_features = [
    "social_sentiment",
    "funding_rate",
    "whale_activity"
]

[custom_features.transformations]
social_sentiment = "ZScore"
funding_rate = "PercentChange"
```

### Training with Custom Features

#### Step 1: Prepare Your Data
```csv
# btc_enhanced.csv
timestamp,open,high,low,close,volume,social_sentiment,funding_rate,whale_activity
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,0.75,-0.01,1250000
2024-01-01T01:00:00Z,42300.0,42800.0,42100.0,42600.0,1567.89,0.82,0.02,1180000
# ... more data
```

#### Step 2: Configure Features
```toml
# configs/btc_custom.toml
[technical_indicators]
enabled = true
# ... standard technical indicators

[custom_features]
auto_include_all = true

[custom_features.transformations]
social_sentiment = "ZScore"
funding_rate = "PercentChange"
whale_activity = "Log"
```

#### Step 3: Train Model
```bash
vanga train --symbol BTCUSDT --data btc_enhanced.csv --features-config configs/btc_custom.toml
```

### Advanced Custom Features Examples

#### 1. Cryptocurrency-Specific Features
```csv
timestamp,open,high,low,close,volume,funding_rate,open_interest,liquidations,fear_greed_index
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,-0.01,1500000000,25000000,25
```

```toml
[custom_features]
auto_include_all = true

[custom_features.transformations]
funding_rate = "PercentChange"
open_interest = "Log"
liquidations = "ZScore"
fear_greed_index = "MinMaxScale"
```

#### 2. On-Chain Metrics
```csv
timestamp,open,high,low,close,volume,active_addresses,transaction_count,nvt_ratio,mvrv_ratio
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,950000,280000,45.2,1.8
```

```toml
[custom_features]
auto_include_all = true

[custom_features.transformations]
active_addresses = "Log"
transaction_count = "Log"
nvt_ratio = "ZScore"
mvrv_ratio = "ZScore"
```

#### 3. Social & Sentiment Data
```csv
timestamp,open,high,low,close,volume,twitter_sentiment,reddit_mentions,google_trends,news_sentiment
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,0.75,1250,67,0.82
```

```toml
[custom_features]
auto_include_all = true

[custom_features.transformations]
twitter_sentiment = "ZScore"
reddit_mentions = "Log"
google_trends = "MinMaxScale"
news_sentiment = "ZScore"
```

### Feature Transformation Types

#### Available Transformations
- `"ZScore"` - Normalize to z-score (mean=0, std=1)
- `"MinMaxScale"` - Scale to range [0, 1]
- `"PercentChange"` - Convert to percentage change
- `"Log"` - Apply logarithmic transformation
- `"RobustScale"` - Robust scaling using median and IQR

#### When to Use Each Transformation
- **ZScore**: For normally distributed features (sentiment scores, ratios)
- **MinMaxScale**: For bounded features (percentages, indices)
- **PercentChange**: For rates and changes (funding rates, growth rates)
- **Log**: For skewed features (volumes, counts, addresses)
- **RobustScale**: For features with outliers (liquidations, whale activity)

### Common Use Cases

#### 1. Adding Market Sentiment
```bash
# Data with sentiment columns
vanga train --symbol BTCUSDT --data btc_with_sentiment.csv

# The system automatically detects and includes sentiment columns
# No configuration needed if using auto_include_all = true
```

#### 2. Incorporating On-Chain Data
```bash
# Data with on-chain metrics
vanga train --symbol BTCUSDT --data btc_onchain.csv --features-config configs/onchain_features.toml
```

#### 3. Multi-Exchange Data
```csv
timestamp,open,high,low,close,volume,binance_volume,coinbase_volume,kraken_volume,volume_ratio
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,800.0,300.0,134.56,0.65
```

### Troubleshooting Custom Features

#### Feature Compatibility Issues
```bash
# Error: Model input size mismatch
# Solution: Use --fresh to retrain with new feature structure
vanga train --symbol BTCUSDT --data new_features.csv --fresh
```

#### Missing Custom Features
```bash
# Check if features are being included
vanga train --symbol BTCUSDT --data data.csv --verbose

# Look for log messages about feature engineering
# Should show: "Custom features detected: [feature1, feature2, ...]"
```

#### Feature Validation
```toml
# Ensure features are properly configured
[custom_features]
auto_include_all = true

# Verify exclude list doesn't remove wanted features
exclude_features = []  # Empty list to include all
```

### Best Practices

#### 1. Data Quality
- Ensure custom features have minimal missing values
- Use consistent data frequency (same as OHLCV)
- Validate feature ranges and distributions

#### 2. Feature Engineering
- Start with `auto_include_all = true` for simplicity
- Use appropriate transformations for each feature type
- Monitor feature importance after training

#### 3. Model Management
- Use descriptive feature configuration file names
- Keep track of which features were used for each model
- Use `--fresh` when adding new features to existing models

#### 4. Performance Optimization
- Limit total features to avoid overfitting (max_features = 100)
- Remove highly correlated features (correlation_threshold = 0.95)
- Filter low-importance features (importance_threshold = 0.001)

### Complete Workflow Example

```bash
# 1. Prepare data with custom features
# btc_complete.csv contains OHLCV + sentiment + on-chain + funding data

# 2. Create custom configuration
# configs/btc_complete.toml with appropriate transformations

# 3. Train model
vanga train --symbol BTCUSDT --data btc_complete.csv --features-config configs/btc_complete.toml

# 4. Make predictions
vanga predict --symbol BTCUSDT --input recent_data.csv

# 5. Continue training with new data
vanga train --symbol BTCUSDT --data additional_data.csv --features-config configs/btc_complete.toml
```

This workflow ensures consistent feature engineering across training and prediction phases.
