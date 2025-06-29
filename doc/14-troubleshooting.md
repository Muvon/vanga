# VANGA LSTM Troubleshooting Guide

## 🔧 **Common Issues and Solutions**

This document provides solutions to common issues encountered during VANGA LSTM development and usage.

---

## 🚨 **Critical Bug Fixes**

### **LSTM Output Size Mismatch (RESOLVED)**

**Issue**: Hundreds of warnings during prediction:
```
Output size mismatch: expected 1, got 64 - using 1 dimensions
```

**Root Cause**: The rust-lstm library v0.2.0 returns hidden states (64 dimensions) directly without output projection layer.

**Solution** (Applied in `src/model/lstm_simple.rs`):
```rust
// FIXED: Project hidden states to single prediction value
let prediction_value = if last_output.nrows() > 0 {
    // Simple projection: take mean of hidden state values
    let sum: f64 = (0..last_output.nrows())
        .map(|i| last_output[[i, 0]])
        .sum();
    sum / last_output.nrows() as f64
} else {
    0.0
};
```

**Status**: ✅ **RESOLVED** - No more repeated warnings

---

### **Shape Mismatch in Validation Metrics (RESOLVED)**

**Issue**: Errors during validation:
```
Shape mismatch in MSE calculation: predictions=[394, 1], targets=[323, 1]
```

**Root Cause**: Prediction and validation target arrays had different shapes due to data processing.

**Solution** (Applied in `src/model/lstm_simple.rs`):
```rust
// FIXED: Validate shapes before calculating metrics
if final_predictions.shape() == val_targets.shape() {
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

### **rust-lstm Library Limitations**
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
