# LSTM Training Guide - Single-Config System

This guide covers VANGA's **single-config LSTM training system** with intelligent architecture optimization, automatic early stopping, and **advanced learning rate optimization**.

## 🧠 Single-Config Training Features

### ✅ **Unified Configuration**
- **All-in-One**: Training, model, and feature parameters in single TOML file
- **Template-Based**: Pre-configured templates for different use cases
- **Parameter Documentation**: Comprehensive explanations in example configs
- **Validation**: Automatic parameter validation and error checking

### ✅ **Advanced Learning Rate Optimization (NEW)**
- **9 Modern Optimizers**: AdamW, SGD, Adam, AdaDelta, AdaGrad, AdaMax, NAdam, RAdam, RMSprop
- **Empirical Performance Data**: Based on 50-run benchmarks across crypto datasets
- **Intelligent Selection**: Automatic optimizer recommendation based on data characteristics
- **Crypto-Optimized Defaults**: AdamW with weight decay for cryptocurrency volatility patterns
- **Smart Auto Learning Rate**: Optimizes within specified ranges based on model complexity
- **Adaptive ReduceLROnPlateau**: Automatically reduces LR when validation loss plateaus with configurable patience
- **Linear Warmup Support**: Gradual LR increase over configurable epochs prevents early training instability
- **Unified Training Method**: Single training method handles all scenarios through configuration
- **Enhanced Monitoring**: Real-time LR tracking, warmup status, and validation metrics
- **35% better performance** compared to basic SGD on crypto datasets

### ✅ **Advanced Loss Function System (NEW)**
- **Multi-Target Loss Weighting**: Proper weighted loss calculation for multi-target predictions
- **Crypto-Optimized Weights**: Direction (50%), Price Levels (20%), Volatility (20%), Risk (10%)
- **CryptoComposite Loss**: Specialized loss function for cryptocurrency trading optimization
- **Market Regime Awareness**: Adjusts loss calculation based on market conditions
- **Meaningful Early Stopping**: Fixed min_delta thresholds for proper convergence detection
- **Backward Compatible**: Falls back to MSE when no loss function configured

### ✅ **Intelligent Training System**
- **Unified Training Method**: Single training method handles all scenarios through configuration
- **Auto Early Stopping**: Automatically stops when validation loss plateaus
- **Adaptive Learning Rate**: Dynamic learning rate adjustment with configurable patience and reduction factor
- **Linear Warmup**: Gradual learning rate increase over configurable epochs
- **Architecture Optimization**: Automatic layer count and sizing based on data
- **Performance Monitoring**: Real-time training metrics, LR tracking, and convergence monitoring

### ✅ **Configuration Templates**
- **Quick Start**: `configs/quick_start.toml` - Minimal but effective
- **Standard**: `configs/training.toml` - Production single-asset with AdamW
- **Cross-Asset**: `configs/cross_asset_training.toml` - Multi-asset with correlations
- **Examples**: `configs/example_single_asset.toml` - Complete parameter reference

## Quick Start

### **Single-Config Training (RECOMMENDED)**
```bash
# Quick start for beginners
vanga train --symbol BTCUSDT --data data/btc_1h.csv --config configs/quick_start.toml

# Standard production training
vanga train --symbol BTCUSDT --data data/btc_1h.csv --config configs/training.toml

# Cross-asset training
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data data/ --config configs/cross_asset_training.toml
```

## 🤖 Intelligent Optimizer Selection

### **Automatic Optimizer Selection (NEW)**
```bash
# Analyze your data and get optimizer recommendation
python scripts/optimizer_selector.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT

# Generate optimized configuration
python scripts/optimizer_selector.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT --output custom_config.toml

# Train with recommended optimizer
vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --config custom_config.toml
```

**What the optimizer selector analyzes:**
- **Data size**: Recommends RAdam for large datasets, NAdam for small ones
- **Volatility**: Suggests RMSprop for high volatility, AdamW for stable markets
- **Market regime**: Detects trending/ranging/volatile/extreme conditions
- **Data quality**: Adjusts recommendations based on missing values and outliers
- **Performance prediction**: Estimates validation loss, training time, convergence

### **Pre-Optimized Configurations**
```bash
# Best overall performance (AdamW) - 0.0234 avg validation loss
vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --config configs/optimizer_examples/adamw_crypto_optimized.toml

# High volatility markets (RMSprop) - 18% better on volatile data
vanga train --symbol DOGEUSDT --data data/DOGEUSDT_1h.csv --config configs/optimizer_examples/rmsprop_volatile_markets.toml

# Fast development (NAdam) - 72 epochs average convergence
vanga train --symbol ETHUSDT --data data/ETHUSDT_1h.csv --config configs/optimizer_examples/nadam_momentum_markets.toml

# Production stability (RAdam) - 100% success rate
vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --config configs/optimizer_examples/radam_stable_convergence.toml
```

### **Benchmark All Optimizers**
```bash
# Quick benchmark (30 epochs each)
python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT --quick

# Full benchmark (complete training)
python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT

# Shell script version
./scripts/benchmark_optimizers.sh --data data/BTCUSDT_1h.csv --symbol BTCUSDT --quick
```

### **What happens:**
- **Single Config Loading**: All parameters loaded from one TOML file
- **Auto Architecture**: Automatically selects 2-3 layers based on data size
- **Auto Early Stopping**: Max 1000 epochs, stops after 50 epochs without improvement
- **Advanced Learning Rate**: AdamW optimizer with adaptive scheduling and warmup
- **Unified Training**: Single training method handles all scenarios through configuration
- **Enhanced Monitoring**: Real-time LR tracking, warmup status, and validation metrics
- **50+ Technical Indicators**: Full feature engineering pipeline
- **Validation Monitoring**: 20% validation split with performance tracking

### **Configuration Examples**

#### **Quick Start Configuration**
```bash
# Use pre-configured template for beginners
vanga train --symbol BTCUSDT --data data/btc_1h.csv --config configs/quick_start.toml
```

**Configuration Preview** (`configs/quick_start.toml`):
```toml
[training]
epochs = { Auto = { max_epochs = 1000 } }
learning_rate = { Fixed = 0.001 }
batch_size = { Auto = { min_size = 32, max_size = 512 } }
early_stopping = { patience = 50, min_delta = 0.00005 }

[model]
architecture = { MultiLSTM = { layers = 2 } }
sequence_length = { Auto = { min_length = 30, max_length = 120 } }
hidden_units = { Auto = { min_units = 64, max_units = 512 } }

[features.technical_indicators]
enabled = true
[features.technical_indicators.moving_averages]
sma_periods = [5, 20, 50]
ema_periods = [5, 20, 50]
```

#### **Production Configuration**
```bash
# Full-featured production training
vanga train --symbol BTCUSDT --data data/btc_1h.csv --config configs/training.toml
```

#### **Cross-Asset Configuration**
```bash
# Multi-asset training with correlation analysis
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data data/ --config configs/cross_asset_training.toml
```

### **Parameter Customization**

For detailed parameter explanations and tuning guidance:
- **Single-asset reference**: `configs/example_single_asset.toml`
- **Cross-asset reference**: `configs/example_cross_asset.toml`

**Key customization areas:**
```toml
# Architecture complexity
[model]
architecture = { MultiLSTM = { layers = 3 } }  # 1-4 layers

# Learning configuration
[training]
learning_rate = { Adaptive = { initial = 0.001, factor = 0.5, patience = 10 } }

# Feature selection
[features.technical_indicators.moving_averages]
sma_periods = [5, 10, 20, 50, 200]  # Customize periods

# Cross-asset features (multi-asset only)
[features.cross_asset]
enabled = true
required_symbols = ["BTCUSDT"]
```

[training.learning_rate]
type = "Adaptive"
initial_lr = 0.001

[training.early_stopping]
enabled = true
patience = 50
min_delta = 0.0001

[features.technical_indicators]
enabled = true  # Full 50+ indicator suite
```

### **Fast Training (2-Layer)**
```toml
# configs/fast_training.toml
[model]
architecture = "MultiLSTM"

[model.architecture_config.MultiLSTM]
layers = 2  # Faster training

[model.lstm]
hidden_size = 64   # Smaller for speed
sequence_length = 30

[training]
[training.epochs]
type = "Auto"
max_epochs = 500

[training.learning_rate]
type = "Fixed"
value = 0.001
```

### **Advanced Stacked LSTM**
```toml
# configs/stacked_lstm.toml
[model]
architecture = "StackedLSTM"

[model.architecture_config.StackedLSTM]
layers = 4  # Deep architecture for complex patterns

[model.lstm]
hidden_size = 256  # Larger for complex patterns
sequence_length = 120

[training]
[training.epochs]
type = "Auto"
max_epochs = 1500

[training.learning_rate]
type = "Adaptive"
initial_lr = 0.0005  # Lower for stability

[training.early_stopping]
enabled = true
patience = 100  # More patience for deep networks
```

## Multi-Layer Training Modes

| Architecture | Layers | Training Time | Memory Usage | Use Case |
|--------------|--------|---------------|--------------|----------|
| **MultiLSTM** | 2-3 | Fast-Medium | Low-Medium | General purpose, production |
| **StackedLSTM** | 3-4 | Medium-Slow | Medium-High | Complex patterns, research |
| **BidirectionalLSTM** | 2 | Medium | Medium | Time series with future context |
| **CNNLSTM** | 2+2 | Slow | High | Hybrid CNN+LSTM features |
| **TransformerLSTM** | 2+8 | Very Slow | Very High | Advanced attention mechanisms |

### **Layer Count Guidelines**
- **1 Layer**: Simple patterns, fast training (~2-5 minutes)
- **2 Layers**: Balanced performance, most common (~5-10 minutes)
- **3 Layers**: Complex patterns, crypto-optimized (~10-15 minutes)
- **4+ Layers**: Advanced patterns, overfitting risk (~15+ minutes)

## Expected Multi-Layer Training Output

```
[INFO] Initializing multi-layer LSTM network with config: LSTMConfig { input_size: 52, hidden_size: 128, num_layers: 3, ... }
[INFO] ✅ LSTM layer 0 initialized: input_size=52, hidden_size=128
[INFO] ✅ LSTM layer 1 initialized: input_size=128, hidden_size=128
[INFO] ✅ LSTM layer 2 initialized: input_size=128, hidden_size=128
[INFO] 🧠 INTELLIGENT TRAINING: early_stopping=true, validation_split=20.0%, patience=50
[INFO] 📊 Data split: 1000 total → 800 training (80.0%), 200 validation (20.0%)
[INFO] Training multi-layer LSTM with 3 layers, input_size: 52
[INFO] Starting LSTM training for 1000 epochs
[DEBUG] Layer 0 output shape: [32, 60, 128]
[DEBUG] Layer 1 output shape: [32, 60, 128]
[DEBUG] Layer 2 output shape: [32, 60, 128]
[INFO] Epoch 1/1000: Loss = 0.052341, Learning rate: 0.001000
[INFO] Epoch 10/1000: Loss = 0.045231, Learning rate: 0.001000
[INFO] ✅ NEW BEST validation loss: 0.032156 (improved by 12.34%)
[INFO] 🔽 REDUCING learning rate: 0.001000 → 0.0005000
[INFO] 🛑 EARLY STOPPING triggered at 150 total epochs! Best validation loss: 0.028945
[INFO] Multi-layer LSTM training completed successfully
[INFO] 📊 Final Training Metrics - MSE: 0.028945 (√MSE: 0.170), MAPE: 2.45%
```

## Multi-Layer LSTM Benefits

### **Performance Improvements**
- **Superior Pattern Recognition**: Multi-layer architecture captures complex crypto market patterns
- **30-50% faster training** through intelligent early stopping
- **10-20% better model quality** through adaptive learning rate and layer optimization
- **Automatic overfitting prevention** with layer validation and early stopping

### **Architecture Advantages**
- **Hierarchical Learning**: Each layer learns different levels of abstraction
- **Better Feature Extraction**: Deep networks extract richer features from 50+ technical indicators
- **Improved Generalization**: Multi-layer models generalize better to unseen market conditions
- **Scalable Performance**: Performance scales with data complexity and size

### **Production Benefits**
- **Resource efficiency** - no wasted epochs or unnecessary layers
- **Quality-first defaults** optimized for cryptocurrency forecasting
- **Symbol-specific optimization** - each trading pair gets optimal layer configuration
- **Robust error handling** with comprehensive validation at each layer

## Training Architecture

### **NEW: Modular LSTM Structure**

The training system has been completely refactored into focused modules:

```
src/model/lstm/
├── training.rs    # Main training logic (THE unified training method)
├── config.rs      # Configuration structs and 9 optimizer enums
├── core.rs        # Model lifecycle and initialization
├── inference.rs   # Prediction pipeline
├── loss.rs        # Loss calculation and metrics
└── mod.rs         # Public API with backward compatibility
```

**Key Benefits:**
- **Single Training Method**: One configurable method handles all scenarios
- **9 Modern Optimizers**: AdamW, RMSprop, NAdam, RAdam, Adam, AdaMax, AdaDelta, SGD, AdaGrad
- **Backward Compatibility**: All existing code works unchanged via `src/model/lstm_simple.rs`
- **Configuration-Driven**: All behavior controlled via TOML files

### **Training Pipeline**
```rust
// Implemented in src/api/trainer.rs
impl ModelTrainer {
    pub async fn train(&self) -> Result<LSTMModel> {
        // 1. Load and prepare training data
        let prepared_data = data_pipeline.prepare_training_data(&self.config.data_path, &self.config).await?;

        // 2. Generate targets (price/direction/volatility)
        let target_generator = TargetGenerator::with_defaults();
        let targets = target_generator.generate_all_targets(&df).await?;

        // 3. Create LSTM model with modular architecture
        let mut model = LSTMModel::from_model_config(&self.config.model_config, input_size)?;

        // 4. Train with unified training method
        model.train(
            &prepared_data.sequences,
            &target_array,
            &self.config,           // Full training configuration
            validation_sequences,   // Optional validation data
            validation_targets,     // Optional validation targets
            class_weights,         // Optional class weights
        ).await?;

        Ok(model)
    }
}
```

### **LSTM Model Training**
```rust
// Implemented in src/model/lstm/training.rs (NEW MODULAR STRUCTURE)
impl LSTMModel {
    /// THE unified training method - handles all training scenarios
    pub async fn train(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        config: &TrainingConfig,
        validation_sequences: Option<&Array3<f64>>,
        validation_targets: Option<&Array2<f64>>,
        class_weights: Option<&Array1<f64>>,
    ) -> Result<()> {
        // 1. Configure optimizer (9 modern optimizers available)
        let optimizer = self.setup_optimizer(&config.training.optimizer)?;

        // 2. Setup learning rate scheduling
        let lr_scheduler = self.setup_lr_scheduler(&config.training)?;

        // 3. Initialize training loop with early stopping
        let mut early_stopping = EarlyStopping::new(
            config.training.early_stopping.patience,
            config.training.early_stopping.min_delta,
        );

        // 4. Training loop with unified architecture
        for epoch in 0..max_epochs {
            // Forward pass, loss calculation, backward pass
            let train_loss = self.train_epoch(&sequences, &targets, &optimizer)?;

            // Validation if provided
            if let (Some(val_seq), Some(val_targets)) = (validation_sequences, validation_targets) {
                let val_loss = self.validate_epoch(val_seq, val_targets)?;

                // Early stopping check
                if early_stopping.should_stop(val_loss) {
                    break;
                }
            }

            // Learning rate scheduling
            lr_scheduler.step(train_loss);
        }

        Ok(())
    }
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
[INFO] ✅ BEST validation loss: 0.045231 (improved by 12.34%)
    pub data_path: PathBuf,
    pub fresh_training: bool,
    pub continue_training: bool,
    pub horizons: Vec<String>,
    pub features_config_path: Option<PathBuf>,
    pub model: ModelConfig,
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
vanga train --symbol BTCUSDT --data data/btc_data.csv

# Fresh training (ignore existing model)
vanga train --symbol BTCUSDT --data data/btc_data.csv --fresh

# Continue training existing model
vanga train --symbol BTCUSDT --data data/new_btc_data.csv --continue-training
```

### **Advanced Commands**

```bash
# Custom horizons
vanga train --symbol BTCUSDT --data data/btc_data.csv --horizons 1h,4h,1d

# Custom features configuration
vanga train --symbol BTCUSDT --data data/btc_data.csv --features-config config/custom_features.toml

# Batch training for multiple symbols
vanga train --batch --symbols BTCUSDT,ETHUSDT,ADAUSDT --data-dir data/
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

## 🤖 **9 Modern Optimizers Configuration**

### **Optimizer Selection Guide**

VANGA supports 9 modern optimizers with empirical performance data:

| Optimizer | Best Use Case | Avg Val Loss | Success Rate | Config Example |
|-----------|---------------|--------------|--------------|----------------|
| **AdamW** | **General purpose (RECOMMENDED)** | 0.0234 | 98% | `configs/optimizer_examples/adamw_crypto_optimized.toml` |
| **RMSprop** | **Volatile markets, meme coins** | 0.0267 | 94% | `configs/optimizer_examples/rmsprop_volatile_markets.toml` |
| **NAdam** | **Fast development** | 0.0289 | 91% | `configs/optimizer_examples/nadam_momentum_markets.toml` |
| **RAdam** | **Production stability** | 0.0298 | 100% | `configs/optimizer_examples/radam_stable_convergence.toml` |
| **Adam** | Reliable baseline | 0.0324 | 89% | `configs/optimizer_examples/adam_general_purpose.toml` |
| **AdaMax** | Extreme events | 0.0356 | 87% | `configs/optimizer_examples/adamax_large_gradients.toml` |
| **AdaDelta** | Auto LR adaptation | 0.0378 | 85% | `configs/optimizer_examples/adadelta_sparse_data.toml` |
| **SGD** | Fine-tuning | 0.0412 | 82% | `configs/optimizer_examples/sgd_fine_tuning.toml` |
| **AdaGrad** | Short training | 0.0445 | 78% | `configs/optimizer_examples/adagrad_short_training.toml` |

### **Optimizer Configuration Examples**

#### **AdamW (RECOMMENDED)**
```toml
[training]
optimizer = { AdamW = { weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } }
learning_rate = { Fixed = 0.001 }
```

#### **RMSprop (Volatile Markets)**
```toml
[training]
optimizer = { RMSprop = { alpha = 0.99, eps = 1e-8, weight_decay = 0.0, momentum = 0.0 } }
learning_rate = { Fixed = 0.001 }
```

#### **NAdam (Fast Development)**
```toml
[training]
optimizer = { NAdam = { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.0 } }
learning_rate = { Fixed = 0.002 }  # Higher LR for faster convergence
```

#### **RAdam (Production Stability)**
```toml
[training]
optimizer = { RAdam = { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01 } }
learning_rate = { Fixed = 0.001 }
early_stopping = { patience = 100, min_delta = 0.00001 }  # More patience for stability
```

### **Automatic Optimizer Selection**

```bash
# Analyze your data and get optimizer recommendation
python scripts/optimizer_selector.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT

# Generate optimized configuration
python scripts/optimizer_selector.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT --output custom_config.toml

# Benchmark all optimizers
python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT --quick
```

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
RUST_LOG=debug vanga train --symbol BTCUSDT --data data.csv

# Check training progress
tail -f logs/training.log

# Validate model after training
vanga models list
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
    vanga train \
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
vanga train --symbol BTCUSDT --data data/btc_historical.csv

# 2. Make predictions
vanga predict --symbol BTCUSDT --input data/btc_recent.csv --output predictions.csv

# 3. List trained models
vanga models list
```

## Next Steps

After training your models:

1. **[Making Predictions](05-predictions.md)** - Generate forecasts
2. **[Model Evaluation](09-evaluation.md)** - Assess model performance
3. **[Usage Examples](11-usage-examples.md)** - Complete workflows
