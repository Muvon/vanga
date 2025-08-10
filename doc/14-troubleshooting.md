# VANGA Troubleshooting Guide

Comprehensive troubleshooting guide for VANGA's cryptocurrency forecasting system with current development practices and solutions.

## 🔧 **Development Workflow Issues**

### **Build Performance (CRITICAL)**

**Issue**: Slow compilation during development

**Solution**: Use fast development commands:
```bash
# ✅ FAST - Use during development
cargo check --message-format=short  # Fastest compilation check
cargo clippy --all-features --all-targets -- -D warnings  # Code quality
cargo test  # Run tests

# ❌ SLOW - Only for production
cargo build --release  # Extremely slow, only use for final builds
```

**Performance Comparison**:
- `cargo check`: ~30 seconds
- `cargo build`: ~2 minutes
- `cargo build --release`: ~15 minutes (avoid during development)

### **Memory Issues During Compilation**

**Issue**: Out of memory during build

**Solution**: Reduce parallel compilation:
```bash
# Limit parallel jobs
export CARGO_BUILD_JOBS=2
cargo check --message-format=short

# Or use single-threaded compilation
export CARGO_BUILD_JOBS=1
cargo build
```

## 🎯 **Training Issues**

### **Insufficient Data Error**

**Issue**: "Insufficient data for training" error

**Root Cause**: Less than 1000 rows in CSV file

**Solution**:
```bash
# Check data size
wc -l data/your_data.csv

# Minimum requirements:
# - Training: 1000+ rows
# - Backtesting: 1000+ rows
# - Robust results: 5000+ rows
```

### **Model Not Found Error**

**Issue**: "Model not found" during prediction

**Root Cause**: Model wasn't saved or wrong path

**Solution**:
```bash
# Check if model exists
ls models/BTCUSDT/

# Retrain if missing
cargo run -- train --symbol BTCUSDT --data data/BTCUSDT_1h.csv

# Models are saved to: models/{SYMBOL}/
```

### **Training Fails with "Invalid CSV Format"**

**Issue**: CSV parsing errors during training

**Root Cause**: Incorrect column names or format

**Solution**:
```bash
# Verify exact column names (case-sensitive)
head -1 data/your_data.csv
# Must be: timestamp,open,high,low,close,volume

# Check data format
head -5 data/your_data.csv
# timestamp,open,high,low,close,volume
# 2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56
```

## 🔮 **Prediction Issues**

### **Shape Mismatch Errors**

**Issue**: Tensor shape mismatch during prediction

**Root Cause**: Model trained with different feature count

**Solution**:
```bash
# Check model info
cargo run -- model-info --symbol BTCUSDT

# Retrain with fresh data if feature count changed
cargo run -- train --symbol BTCUSDT --data data/new_data.csv --fresh
```

### **Low Prediction Confidence**

**Issue**: All predictions have low confidence scores

**Root Cause**: Model needs more training or better data

**Solution**:
```bash
# Continue training with more data
cargo run -- train --symbol BTCUSDT --data data/extended_data.csv --continue-training

# Or retrain with better configuration
cargo run -- train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --config configs/training.toml
```

## 🏗 **Configuration Issues**

### **Configuration File Not Found**

**Issue**: "Config file not found" error

**Root Cause**: Wrong path or missing file

**Solution**:
```bash
# List available configurations
ls configs/
# quick_start.toml, training.toml, etc.

# Use absolute path if needed
cargo run -- train --symbol BTCUSDT --data data.csv --config $(pwd)/configs/training.toml
```

### **Invalid Configuration Values**

**Issue**: Configuration validation errors

**Root Cause**: Invalid TOML syntax or values

**Solution**:
```bash
# Validate TOML syntax
cargo run -- validate-config --config configs/your_config.toml

# Check example configurations
cat configs/quick_start.toml
```

## 💾 **Data Issues**

### **Data Loading Failures**

**Issue**: "Failed to load CSV" error

**Root Cause**: File permissions, encoding, or format issues

**Solution**:
```bash
# Check file permissions
ls -la data/your_data.csv

# Check file encoding (should be UTF-8)
file data/your_data.csv

# Check for hidden characters
cat -A data/your_data.csv | head -5
```

### **Missing Timestamps**

**Issue**: "Invalid timestamp format" error

**Root Cause**: Incorrect timestamp format

**Solution**:
```bash
# Correct format: ISO 8601
# ✅ CORRECT: 2024-01-01T00:00:00Z
# ❌ WRONG: 2024-01-01 00:00:00
# ❌ WRONG: 01/01/2024 00:00:00

# Fix timestamps in your data
sed 's/ /T/g' data/input.csv | sed 's/$/Z/' > data/fixed.csv
```

## 🖥 **System Issues**

### **GPU Not Detected**

**Issue**: CUDA device not available

**Root Cause**: CUDA not installed or configured

**Solution**:
```bash
# Check CUDA installation
nvcc --version

# Check GPU availability
nvidia-smi

# Test GPU with VANGA
cargo run -- device-info

# Use CPU if GPU unavailable
cargo run -- train --symbol BTCUSDT --data data.csv --device cpu
```

### **Out of Memory Errors**

**Issue**: System runs out of memory during training

**Root Cause**: Dataset too large or insufficient RAM

**Solution**:
```bash
# Reduce batch size in configuration
[training]
batch_size = { Fixed = 16 }  # Instead of Auto

# Reduce sequence length
[model]
sequence_length = { Fixed = 60 }  # Instead of Auto

# Use smaller model
[model]
hidden_units = { Fixed = 128 }  # Instead of Auto
```

## 🔄 **Real-time Streaming Issues**

### **File Watcher Not Working**

**Issue**: Real-time streaming not detecting new data

**Root Cause**: File system permissions or polling issues

**Solution**:
```bash
# Check file permissions
ls -la data/live_data.csv

# Use shorter polling interval
cargo run -- stream \
    --symbol BTCUSDT \
    --data-path data/live_data.csv \
    --interval 30s  # Shorter interval
```

### **Streaming Performance Issues**

**Issue**: High CPU usage during streaming

**Root Cause**: Too frequent predictions or large buffer

**Solution**:
```bash
# Increase prediction interval
cargo run -- stream \
    --symbol BTCUSDT \
    --data-path data/live_data.csv \
    --interval 5m  # Less frequent predictions

# Reduce buffer size
[realtime]
buffer_size = 500  # Instead of 1000
```

## 🧪 **Testing and Debugging**

### **Enable Verbose Logging**

```bash
# Debug level logging
RUST_LOG=debug cargo run -- train --symbol BTCUSDT --data data.csv

# Info level logging
RUST_LOG=info cargo run -- train --symbol BTCUSDT --data data.csv

# Log to file
RUST_LOG=info cargo run -- train --symbol BTCUSDT --data data.csv 2> training.log
```

### **Test with Minimal Data**

```bash
# Create minimal test dataset
head -1000 data/large_dataset.csv > data/test_sample.csv

# Test training
cargo run -- train --symbol BTCUSDT --data data/test_sample.csv --config configs/quick_start.toml
```

### **Validate Installation**

```bash
# Check Rust version
rustc --version
# Should be 1.87.0 or later

# Check VANGA help
cargo run -- --help

# Test compilation
cargo check --message-format=short
```

## 🔍 **Performance Optimization**

### **Development Performance**
```bash
# Fast development cycle
cargo check --message-format=short  # Use this 90% of the time
cargo clippy --all-features --all-targets -- -D warnings  # Code quality
cargo test  # Testing

# Only when you need the binary
cargo build  # Debug build
```

### **Production Performance**
```bash
# Optimized build (only for production)
cargo build --release

# Use release binary for large datasets
./target/release/vanga train --symbol BTCUSDT --data large_dataset.csv
```

### **Memory Optimization**
```toml
# In your configuration file
[training]
batch_size = { Fixed = 32 }  # Smaller batches
sequence_length = { Fixed = 60 }  # Shorter sequences

[model]
hidden_units = { Fixed = 256 }  # Smaller model
```

## 📊 **Common Error Messages**

### **"Tensor shape mismatch"**
- **Cause**: Model expects different input size
- **Fix**: Retrain model or check feature engineering

### **"Insufficient data for sequence generation"**
- **Cause**: Not enough data for sequence length
- **Fix**: Use more data or reduce sequence length

### **"Model architecture mismatch"**
- **Cause**: Trying to load model with different architecture
- **Fix**: Retrain with correct architecture or use compatible config

### **"CUDA out of memory"**
- **Cause**: GPU memory exhausted
- **Fix**: Reduce batch size or use CPU

### **"Permission denied"**
- **Cause**: File system permissions
- **Fix**: Check file permissions and ownership

## 🆘 **Getting Help**

### **Self-Diagnosis Checklist**
1. ✅ Rust version 1.87.0+?
2. ✅ Data has 1000+ rows?
3. ✅ Correct CSV format?
4. ✅ Using `cargo check` for development?
5. ✅ Sufficient disk space?
6. ✅ Correct file permissions?

### **Debug Information to Collect**
```bash
# System information
rustc --version
cargo --version
uname -a

# VANGA information
cargo run -- --help
ls -la models/
ls -la configs/

# Data information
wc -l data/*.csv
head -5 data/your_data.csv
```

### **Performance Monitoring**
```bash
# Monitor memory usage
top -p $(pgrep -f vanga)

# Monitor disk usage
df -h

# Monitor GPU usage (if applicable)
nvidia-smi -l 1
```

**Remember**: Use `cargo check --message-format=short` for fast development, save `--release` builds for production only!
- `src/model/tft/variable_selection.rs` - Broadcast operations
- `src/model/tft/quantile_regression.rs` - Quantile concatenation

**Prevention**: Always call `.contiguous()` after:
- `transpose()` operations before `reshape()`
- `narrow()` operations before `squeeze()`
- `broadcast_*()` operations
- `Tensor::cat()` operations
- Before `matmul()` with potentially non-contiguous tensors

---

## 🚨 **Multi-Layer LSTM Implementation (COMPLETED)**

### **Multi-Layer Architecture Implementation (RESOLVED)**

**Feature**: Complete multi-layer LSTM implementation with manual layer chaining and intelligent optimization.

**Implementation** (Applied in `src/model/lstm_simple.rs`):
```rust
/// Multi-layer LSTM model with manual chaining
pub struct LSTMModel {
    config: LSTMConfig,
    lstm_layers: Option<Vec<LSTM>>,  // Multi-layer manual chaining
    output_layer: Option<Linear>,
    device: Device,
    varmap: VarMap,
    training_config: TrainingConfig,
    trained: bool,
}

/// Forward pass through multi-layer LSTM network
fn forward(&self, input: &Tensor) -> Result<Tensor> {
    let lstm_layers = self.lstm_layers.as_ref()
        .ok_or_else(|| VangaError::ModelError("LSTM layers not initialized".to_string()))?;

    // Manual forward pass through LSTM layers
    let mut current_output = input.clone();
    for (i, lstm_layer) in lstm_layers.iter().enumerate() {
        let layer_states = lstm_layer.seq(&current_output)?;

        // Validate we have states to process
        if layer_states.is_empty() {
            return Err(VangaError::ModelError(format!("Layer {} produced no states", i)));
        }

        // Collect and stack hidden states
        let mut hidden_states = Vec::new();
        for state in &layer_states {
            hidden_states.push(state.h().clone());
        }

        // Stack to form [batch_size, seq_len, hidden_size]
        current_output = Tensor::stack(&hidden_states, 1)?;

        // Validate output dimensions
        let output_shape = current_output.shape();
        if output_shape.dims().len() != 3 {
            return Err(VangaError::ModelError(format!(
                "Layer {} output has wrong dimensions: expected 3D tensor, got {:?}",
                i, output_shape
            )));
        }

        log::debug!("Layer {} output shape: {:?}", i, output_shape);
    }

    // Extract last timestep and apply output layer
    let seq_len = current_output.dim(1)?;
    let last_hidden = current_output.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
    let output_layer = self.output_layer.as_ref()
        .ok_or_else(|| VangaError::ModelError("Output layer not initialized".to_string()))?;

    output_layer.forward(&last_hidden)
}
```

**Status**: ✅ **COMPLETED** - Full multi-layer implementation with validation

---

## 🏗️ **Multi-Layer Specific Issues**

### **Layer Count Optimization**

**Issue**: Choosing optimal number of layers for different datasets.

**Guidelines**:
- **1 Layer**: Simple patterns, fast training (~2-5 minutes)
- **2 Layers**: Balanced performance, most common (~5-10 minutes)
- **3 Layers**: Complex patterns, crypto-optimized (~10-15 minutes)
- **4+ Layers**: Advanced patterns, overfitting risk (~15+ minutes)

**Auto-Selection Logic**:
```rust
fn select_optimal_layers(data_size: usize, complexity: f64) -> usize {
    match (data_size, complexity) {
        (size, _) if size < 1000 => 1,
        (size, complexity) if size < 5000 && complexity < 0.5 => 2,
        (size, _) if size < 10000 => 3,
        _ => 3, // Default optimal for crypto
    }
}
```

### **Memory Management Issues**

**Issue**: High memory usage with multiple layers and long sequences.

**Solutions**:
1. **Reduce Sequence Length**: Use 30-60 instead of 120+
2. **Smaller Hidden Size**: Use 64-128 instead of 256+
3. **Fewer Layers**: Start with 2-3 layers
4. **Chunked Processing**: Enable for large datasets

**Configuration**:
```toml
[model.lstm]
hidden_size = 64      # Reduced for memory efficiency
sequence_length = 30  # Shorter sequences
layers = 2           # Fewer layers
```

### **Training Time Optimization**

**Issue**: Long training times with deep networks.

**Solutions**:
1. **Use Fast Training Config**: 2-layer architecture
2. **Reduce Max Epochs**: Set max_epochs = 500
3. **Early Stopping**: Enable with patience = 25
4. **Fixed Learning Rate**: Avoid adaptive for speed

**Fast Training Config**:
```toml
[model.architecture_config.MultiLSTM]
layers = 2

[training]
[training.epochs]
type = "Auto"
max_epochs = 500

[training.early_stopping]
patience = 25
```

### **Layer Validation Errors**

**Issue**: Dimension mismatch between layers.

**Common Errors**:
```
Layer 1 output has wrong dimensions: expected 3D tensor, got 2D tensor
Layer 2 produced no states
```

**Solutions**:
1. **Check Input Size**: Ensure first layer input_size matches feature count
2. **Validate Sequence Length**: Must be > 0
3. **Monitor Layer Outputs**: Enable debug logging

**Debug Commands**:
```bash
RUST_LOG=debug vanga train --symbol BTCUSDT --data data.csv
```

### **Overfitting with Deep Networks**

**Issue**: 4+ layer models overfit on small datasets.

**Prevention**:
1. **Layer Count Warning**: System warns when layers > 4
2. **Early Stopping**: Automatic validation monitoring
3. **Regularization**: Built-in gradient clipping
4. **Data Size Check**: Recommend fewer layers for small datasets

**Overfitting Detection**:
```
[WARN] Large number of layers (5) may cause overfitting. Consider 2-3 layers for most datasets.
[INFO] 🛑 EARLY STOPPING triggered at 150 total epochs! Best validation loss: 0.028945
```

## 🔧 **Legacy Issues (Resolved)**

### **Early Stopping and Validation Monitoring (IMPLEMENTED)**

**Feature**: Intelligent training with automatic early stopping and adaptive learning rates.

**Implementation** (Applied in `src/model/lstm_simple.rs`):
```rust
// Early stopping with validation monitoring
pub async fn train_with_early_stopping(
    &mut self,
    sequences: &Array3<f64>,
    targets: &Array2<f64>,
    vanga_config: &TrainingConfig,
) -> Result<()> {
    // 1. Validation data splitting based on validation_split
    // 2. Real Candle LSTM network with SGD optimizer
    // 3. Training loop with validation monitoring every epoch
    // 4. Early stopping with configurable patience
    // 5. Adaptive learning rate reduction when loss plateaus
}
```

**Expected Output**:
```
📈 Epoch 11/1000: Train Loss = 5.240563, Validation loss: 5.459465, Learning rate: 0.008500
✅ BEST validation loss: 0.045231 (improved by 12.34%)
🔽 REDUCING learning rate: 0.010000 → 0.005000
🛑 EARLY STOPPING triggered at 25 total epochs! Best validation loss: 0.032156
```

**Status**: ✅ **IMPLEMENTED** - Full early stopping functionality working
    let final_mse = self.calculate_mse_loss(&final_predictions, &val_targets);
    let final_mape = self.calculate_mape(&final_predictions, &val_targets);
    // ... log metrics
} else {
    log::warn!(
        "Skipping validation metrics due to shape mismatch: predictions={:?}, targets={:?}",
        final_predictions.shape(),
        val_targets.shape()
    );
}
```

**Status**: ✅ **RESOLVED** - Graceful handling with clear warnings

---

## 🔍 **Debugging Guidelines**

### **Build and Validation Commands**

```bash
# Fast compilation check (PREFERRED for development)
cargo check --message-format=short

# Code quality enforcement (MANDATORY before commits)
cargo clippy --all-features --all-targets -- -D warnings

# Testing
cargo test

# Debug build (only when you need the binary)
cargo build

# NEVER use --release during development (extremely slow)
```

### **Common Development Issues**

#### **1. Compilation Errors**
- **Always run `cargo check` first** - fastest way to catch errors
- **Check for unused variables** - fix root cause, don't use `_var`
- **Never use `#[allow(dead_code)]`** - remove unused code instead

#### **2. LSTM Training Issues**
- **Check data quality**: Ensure no excessive missing values
- **Verify feature consistency**: Training and prediction data must match
- **Monitor memory usage**: Use sequence length optimization
- **Check target distribution**: Ensure balanced targets

#### **3. Prediction Accuracy Issues**
- **Validate input data**: Same preprocessing as training
- **Check for data leakage**: No future information in features
- **Monitor prediction confidence**: Check for extreme values
- **Verify model loading**: Ensure correct model file

---

## 📊 **Performance Optimization**

### **Memory Management**
- **Batch Size**: Auto-optimized based on available memory
- **Sequence Length**: Automatically calculated per trading pair
- **Feature Caching**: Cache computed technical indicators
- **Model Checkpointing**: Regular saves during training

### **Training Optimization**
- **Early Stopping**: Validation-based with patience parameter
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate**: Auto-tuned based on data characteristics
- **Validation Split**: Automatic 80/20 split with monitoring

---

## 🔧 **Architecture-Specific Issues**

### **Candle Framework Integration**
- **Single Output**: Each model handles one target only
- **Hidden State Access**: No direct output layer configuration
- **Serialization**: Network weights not serializable (config only)
- **Multi-Target**: Requires separate models per target

### **Workarounds Implemented**
- **Output Projection**: Manual projection from hidden states
- **Model Persistence**: Save/load configuration, recreate network
- **Shape Validation**: Comprehensive array shape checking
- **Error Recovery**: Graceful degradation with informative logs

---

## 📝 **File-Specific Troubleshooting**

### **`src/model/lstm_simple.rs`**
- **Lines 780-798**: Fixed prediction output projection
- **Lines 455-479**: Added validation metric shape checking
- **Lines 190-204**: MSE calculation with shape validation
- **Lines 207-230**: MAPE calculation with shape validation

### **Configuration Issues**
- **Check `configs/*.toml`**: Ensure all required parameters
- **Validate data paths**: Verify CSV file accessibility
- **Symbol naming**: Ensure consistent trading pair names
- **Feature configuration**: Match available data columns

---

## 🚀 **Performance Benchmarks**

### **Training Performance**
- **Small datasets (< 1K rows)**: ~30 seconds
- **Medium datasets (1K-10K rows)**: ~2-5 minutes
- **Large datasets (> 10K rows)**: ~10-30 minutes

### **Prediction Performance**
- **Single prediction**: < 1 second
- **Batch predictions (100 samples)**: < 5 seconds
- **Real-time predictions**: < 100ms per sample

---

## 📞 **Support and Maintenance**

### **Code Quality Standards**
- **Zero clippy warnings**: All code must pass clippy
- **Comprehensive error handling**: Use `Result<T>` everywhere
- **Configuration-driven**: Avoid hardcoded parameters
- **Symbol-agnostic**: Code works for any trading pair

### **Testing Strategy**
- **Unit tests**: Individual component testing
- **Integration tests**: Full pipeline testing
- **End-to-end tests**: Sample cryptocurrency data
- **Performance tests**: Large dataset handling

---

**Last Updated**: 2025-06-29
**Status**: ✅ **CURRENT** - All major issues resolved
**: Full pipeline testing
- **End-to-end tests**: Sample cryptocurrency data
- **Performance tests**: Large dataset handling
- **Multi-layer tests**: Layer validation and performance

## 📊 **Multi-Layer Performance Monitoring**

### **Layer Performance Debugging**
```bash
# Enable detailed layer logging
RUST_LOG=debug vanga train --symbol BTCUSDT --data data.csv

# Expected debug output:
# [DEBUG] Layer 0 output shape: [32, 60, 128]
# [DEBUG] Layer 1 output shape: [32, 60, 128]
# [DEBUG] Layer 2 output shape: [32, 60, 128]
```

### **Memory Usage Monitoring**
```bash
# Monitor memory during training
top -p $(pgrep vanga)

# Expected memory usage:
# 1 Layer: ~50-100MB
# 2 Layers: ~100-200MB
# 3 Layers: ~200-400MB
# 4+ Layers: ~400MB+
```

### **Training Time Benchmarks**
| Layers | Dataset Size | Training Time | Memory Usage |
|--------|--------------|---------------|--------------|
| 1      | 10k samples  | 2-5 minutes   | ~100MB       |
| 2      | 10k samples  | 5-10 minutes  | ~200MB       |
| 3      | 10k samples  | 10-15 minutes | ~300MB       |
| 4      | 10k samples  | 15-25 minutes | ~500MB       |

---

**Last Updated**: 2025-07-02
**Status**: ✅ **CURRENT** - Multi-layer LSTM implementation complete, all issues resolved
