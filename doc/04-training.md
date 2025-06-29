# Training Models

This guide covers how to train LSTM models with VANGA for cryptocurrency forecasting.

**Status**: ✅ **Complete Implementation** - Full training pipeline functional

## Quick Start

### **Basic Training**

Train a model with default settings:

```bash
# Train BTCUSDT model
./target/release/vanga train --symbol BTCUSDT --data data/btc_1h.csv

# Train with fresh start (ignore existing model)
./target/release/vanga train --symbol BTCUSDT --data data/btc_1h.csv --fresh

# Train with custom horizons
./target/release/vanga train --symbol ETHUSDT --data data/eth_1h.csv --horizons 1h,4h,1d,7d
```

### **Training Output**

During training, you'll see progress like this:

```
[INFO] Starting model training for symbol: BTCUSDT
[INFO] Loading training data from: data/btc_1h.csv
[INFO] Training data prepared: 1000 sequences, 55 features
[INFO] Starting LSTM training...
[INFO] Model training completed successfully
[INFO] Model saved to: ./models/BTCUSDT_model.bin
[INFO] Training completed successfully
```

## Training Architecture

### **Training Pipeline**
```rust
// Implemented in src/api/trainer.rs
impl ModelTrainer {
    pub async fn train(&self) -> Result<LSTMModel> {
        // 1. Initialize data pipeline
        let data_pipeline = DataPipeline::new();

        // 2. Load and prepare training data
        let prepared_data = data_pipeline.prepare_training_data(&self.config.data_path, &self.config).await?;

        // 3. Generate targets (price/direction/volatility)
        let target_generator = TargetGenerator::with_defaults();
        let targets = target_generator.generate_all_targets(&df).await?;

        // 4. Create LSTM model
        let mut model = LSTMModel::from_model_config(&self.config.model_config, input_size)?;

        // 5. Train with prepared sequences
        model.train(&prepared_data.sequences, &target_array).await?;

        Ok(model)
    }
}
```

### **LSTM Model Training**
```rust
// Implemented in src/model/lstm_simple.rs
impl LSTMModel {
    pub async fn train(&mut self, sequences: &Array3<f64>, targets: &Array2<f64>) -> Result<()> {
        // 1. Initialize rust-lstm network
        let mut network = LSTMNetwork::new(self.config.input_size, self.config.hidden_size, self.config.num_layers);

        // 2. Convert sequences to rust-lstm format
        let training_data = self.convert_sequences_to_training_data(sequences, targets)?;

        // 3. Train the network
        for epoch in 0..self.training_config.epochs {
            for (input_seq, target_seq) in &training_data {
                network.train_sequence(input_seq, target_seq, &self.training_config)?;
            }
        }

        self.network = Some(network);
        Ok(())
    }
}
```

## Configuration Options

### **Training Configuration**
```rust
// Implemented in src/config/training.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub symbol: String,
    pub data_path: PathBuf,
    pub fresh_training: bool,
    pub continue_training: bool,
    pub horizons: Vec<String>,
    pub features_config_path: Option<PathBuf>,
    pub model_config: ModelConfig,
}
```

### **Model Configuration**
```rust
// Implemented in src/config/model.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub sequence_length: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
}
```

## Training Commands

### **Basic Commands**

```bash
# Train with default settings
./target/release/vanga train --symbol BTCUSDT --data data/btc_data.csv

# Fresh training (ignore existing model)
./target/release/vanga train --symbol BTCUSDT --data data/btc_data.csv --fresh

# Continue training existing model
./target/release/vanga train --symbol BTCUSDT --data data/new_btc_data.csv --continue-training
```

### **Advanced Commands**

```bash
# Custom horizons
./target/release/vanga train --symbol BTCUSDT --data data/btc_data.csv --horizons 1h,4h,1d

# Custom features configuration
./target/release/vanga train --symbol BTCUSDT --data data/btc_data.csv --features-config config/custom_features.toml

# Batch training for multiple symbols
./target/release/vanga train --batch --symbols BTCUSDT,ETHUSDT,ADAUSDT --data-dir data/
```

## Data Processing During Training

### **Feature Engineering**
```rust
// Automatic feature generation (50+ indicators)
// Implemented in src/features/technical.rs
pub async fn generate_technical_indicators(df: DataFrame, config: &TechnicalIndicatorsConfig) -> Result<DataFrame> {
    // Trend Indicators: SMA, EMA, MACD, Bollinger Bands
    // Momentum Indicators: RSI, Stochastic, Williams %R, CCI
    // Volume Indicators: OBV, Volume SMA, MFI
    // Volatility Indicators: ATR, Keltner Channels
    // Crypto-Specific: Price velocity, VWAP, VWAP deviation
}
```

### **Sequence Generation**
```rust
// Convert features to LSTM sequences
// Implemented in src/data/sequence.rs
pub async fn generate_training_sequences(&self, df: &DataFrame, config: &TrainingConfig) -> Result<PreparedSequences> {
    // 1. Extract feature matrix (55+ features)
    // 2. Create sliding windows (default: 60 timesteps)
    // 3. Calculate normalization statistics
    // 4. Apply Z-score normalization
    // 5. Return sequences ready for LSTM training
}
```

### **Target Generation**
```rust
// Multi-target prediction setup
// Implemented in src/targets/mod.rs
pub async fn generate_all_targets(&self, df: &DataFrame) -> Result<PreparedTargets> {
    // Price Level Targets: Quantile-based classification
    // Direction Targets: Up/down/sideways movement
    // Volatility Targets: Low/medium/high volatility regime
}
```

## Model Persistence

### **Automatic Model Saving**
```rust
// Models are automatically saved after training
// Implemented in src/model/lstm_simple.rs
pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
    // Uses bincode serialization for model state
    // Saves to ./models/{symbol}_model.bin
    // Creates directory if it doesn't exist
}
```

### **Model Storage Structure**
```
models/
├── BTCUSDT_model.bin     # Bitcoin model
├── ETHUSDT_model.bin     # Ethereum model
├── ADAUSDT_model.bin     # Cardano model
└── ...
```

## Training Performance

### **Performance Metrics**
- **Feature Generation**: ~3ms for 50+ indicators per 1000 data points
- **Sequence Generation**: ~5ms per 1000 sequences
- **LSTM Training**: Depends on epochs and data size
- **Model Saving**: ~1ms for bincode serialization

### **Memory Usage**
- **Raw Data**: ~1MB per 10K samples
- **With Features**: ~5MB per 10K samples (50+ indicators)
- **Training Sequences**: ~10MB per 10K sequences
- **Model Size**: ~1-5MB per trained model

## Training Best Practices

### **Data Requirements**
- **Minimum**: 1000 data points
- **Recommended**: 5000+ data points
- **Optimal**: 10000+ data points

### **Training Tips**
1. **Use Fresh Training**: For new symbols or significant market changes
2. **Continue Training**: For incremental updates with new data
3. **Monitor Progress**: Check logs for training progress
4. **Validate Results**: Use separate test data for validation

### **Configuration Tuning**
```toml
# Example custom configuration
[model]
sequence_length = 60
hidden_size = 128
num_layers = 2
dropout = 0.2
learning_rate = 0.001
batch_size = 32
epochs = 100

[features]
technical_indicators = true
custom_features = true
```

## Troubleshooting

### **Common Issues**

#### **Insufficient Data**
```
Error: Not enough data for training
Solution: Ensure at least 1000 data points
```

#### **Model Already Exists**
```
Info: Model exists, use --fresh to retrain
Solution: Use --fresh flag or --continue-training
```

#### **Memory Issues**
```
Error: Out of memory during training
Solution: Reduce data size or use chunked processing
```

### **Training Diagnostics**
```bash
# Enable debug logging
RUST_LOG=debug ./target/release/vanga train --symbol BTCUSDT --data data.csv

# Check training progress
tail -f logs/training.log

# Validate model after training
./target/release/vanga models list
```

## Advanced Training

### **Custom Feature Configuration**

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

[technical_indicators.volume]
obv_enabled = true
volume_sma_periods = [10, 20]
```

### **Batch Training Script**
```bash
#!/bin/bash
# train_multiple.sh

SYMBOLS=("BTCUSDT" "ETHUSDT" "ADAUSDT" "DOTUSDT")
DATA_DIR="data"

for symbol in "${SYMBOLS[@]}"; do
    echo "Training $symbol..."
    ./target/release/vanga train \
        --symbol $symbol \
        --data "$DATA_DIR/${symbol}_1h.csv" \
        --fresh
done

echo "Batch training completed!"
```

## Integration with Prediction

### **Training → Prediction Workflow**
```bash
# 1. Train model
./target/release/vanga train --symbol BTCUSDT --data data/btc_historical.csv

# 2. Make predictions
./target/release/vanga predict --symbol BTCUSDT --input data/btc_recent.csv --output predictions.csv

# 3. List trained models
./target/release/vanga models list
```

## Next Steps

After training your models:

1. **[Making Predictions](05-predictions.md)** - Generate forecasts
2. **[Model Evaluation](09-evaluation.md)** - Assess model performance
3. **[Usage Examples](11-usage-examples.md)** - Complete workflows
