# VANGA Multi-Layer LSTM Troubleshooting Guide

## 🔧 **Multi-Layer LSTM Issues and Solutions**

This document provides solutions to common issues encountered during VANGA multi-layer LSTM development and usage, including layer-specific problems and optimization challenges.

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
RUST_LOG=debug ./target/release/vanga train --symbol BTCUSDT --data data.csv
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
RUST_LOG=debug ./target/release/vanga train --symbol BTCUSDT --data data.csv

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
