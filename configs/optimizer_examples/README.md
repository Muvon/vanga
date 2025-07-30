# VANGA Optimizer Configuration Examples

This directory contains optimized TOML configuration files for each of the 9 available optimizers in VANGA, specifically tuned for cryptocurrency forecasting scenarios.

## 🎯 **Quick Selection Guide**

### **🥇 RECOMMENDED DEFAULT**
- **`adamw_crypto_optimized.toml`** - Use for 99% of crypto training scenarios
- Best overall performance, handles volatility well, built-in regularization

### **🚀 Specialized Scenarios**

| Optimizer | Configuration File | Best For | Key Benefits |
|-----------|-------------------|----------|--------------|
| **AdamW** | `adamw_crypto_optimized.toml` | General crypto training | Volatility handling, weight decay, robust |
| **RMSprop** | `rmsprop_volatile_markets.toml` | Highly volatile markets | Non-stationary objectives, regime changes |
| **NAdam** | `nadam_momentum_markets.toml` | Trending/momentum markets | Nesterov acceleration, faster convergence |
| **RAdam** | `radam_stable_convergence.toml` | Large datasets, stability | Variance rectification, stable convergence |
| **AdaMax** | `adamax_large_gradients.toml` | Extreme market movements | Large gradient handling, flash crashes |
| **Adam** | `adam_general_purpose.toml` | Standard scenarios | General purpose, reliable |
| **AdaDelta** | `adadelta_sparse_data.toml` | Sparse features, auto-LR | Automatic learning rate adaptation |
| **SGD** | `sgd_fine_tuning.toml` | Fine-tuning pre-trained models | Stability, transfer learning |
| **AdaGrad** | `adagrad_short_training.toml` | ⚠️ Short training only | Sparse features, early exploration |

## 📋 **Usage Instructions**

### **Basic Usage**
```bash
# Use the recommended default
vanga train --config configs/optimizer_examples/adamw_crypto_optimized.toml --symbol BTCUSDT --data data.csv

# For volatile markets (meme coins, new tokens)
vanga train --config configs/optimizer_examples/rmsprop_volatile_markets.toml --symbol DOGEUSDT --data data.csv

# For trending bull/bear markets
vanga train --config configs/optimizer_examples/nadam_momentum_markets.toml --symbol ETHUSDT --data data.csv
```

### **Customization**
Each configuration file can be customized by:
1. Copy the relevant file to your project directory
2. Modify parameters as needed for your specific use case
3. Use the modified configuration with `--config your_custom_config.toml`

## 🔧 **Configuration Details**

### **Key Differences Between Configurations**

| Aspect | AdamW | RMSprop | NAdam | RAdam | AdaMax |
|--------|-------|---------|-------|-------|--------|
| **Learning Rate** | 0.001 | 0.002 | 0.0015 | 0.0008 | 0.001 |
| **Batch Size** | 32 | 24 | 40 | 48 | 32 |
| **Epochs** | 100 | 120 | 90 | 150 | 100 |
| **Sequence Length** | 60 | 90 | 72 | 60 | 56 |
| **Hidden Units** | 128 | 160 | 144 | 128 | 112 |
| **Dropout Rate** | 0.2 | 0.3 | 0.2 | 0.18 | 0.25 |
| **Warmup Epochs** | 5 | 8 | 4 | 10 | 6 |

### **Optimizer-Specific Parameters**

#### **AdamW (Recommended)**
```toml
optimizer = { AdamW = { weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } }
```
- `weight_decay`: Built-in regularization (0.01 = 1% weight decay)
- `beta1`: Momentum parameter for first moment (0.9 standard)
- `beta2`: Momentum parameter for second moment (0.999 standard)
- `eps`: Numerical stability parameter (1e-8 standard)

#### **RMSprop (Volatile Markets)**
```toml
optimizer = { RMSprop = { alpha = 0.99, eps = 1e-8, weight_decay = 0.01, momentum = 0.0, centered = false } }
```
- `alpha`: Smoothing constant for squared gradients (0.99 for crypto volatility)
- `eps`: Small constant for numerical stability
- `momentum`: Momentum factor (0.0 = no momentum)
- `centered`: Whether to center the second moment

#### **NAdam (Momentum Markets)**
```toml
optimizer = { NAdam = { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01, momentum_decay = 0.004 } }
```
- `momentum_decay`: Nesterov momentum decay rate (0.004 for crypto trends)

## ⚠️ **Important Warnings**

### **AdaGrad Limitations**
- **USE ONLY FOR SHORT TRAINING** (< 50 epochs)
- Learning rate decays over time and can become too small
- Good for initial exploration but not production training
- Consider switching to AdamW for longer training runs

### **Learning Rate Considerations**
- **AdaDelta**: Uses internal learning rate adaptation, external LR set to 1.0
- **AdaGrad**: Uses higher initial LR (0.01) as it decays automatically
- **Others**: Use adaptive learning rate schedules for best results

## 🎯 **Performance Expectations**

### **Convergence Speed** (Typical crypto datasets)
1. **NAdam**: Fastest convergence (60-80 epochs)
2. **AdamW**: Fast convergence (80-100 epochs)
3. **Adam**: Moderate convergence (80-120 epochs)
4. **RMSprop**: Variable (depends on volatility)
5. **RAdam**: Slower but stable (120-150 epochs)

### **Final Performance** (Typical ranking)
1. **AdamW**: Best overall performance
2. **RMSprop**: Best for volatile markets
3. **NAdam**: Best for trending markets
4. **RAdam**: Most stable results
5. **AdaMax**: Best for extreme events

## 📊 **Monitoring Training**

### **Key Metrics to Watch**
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should track training loss without large gaps
- **Learning Rate**: Should adapt based on validation performance
- **Gradient Norm**: Should remain stable (watch for explosion)

### **Early Stopping Triggers**
- **AdamW/Adam**: 15 epochs patience
- **RMSprop**: 20 epochs patience (volatile markets need more time)
- **RAdam**: 25 epochs patience (stable convergence takes time)
- **AdaGrad**: 8 epochs patience (short training only)

## 🔄 **Migration Guide**

### **From SGD to AdamW**
1. Reduce learning rate by 10x (0.01 → 0.001)
2. Add weight decay (0.01)
3. Increase batch size if possible
4. Add warmup epochs (5)

### **From Adam to AdamW**
1. Add weight_decay parameter (0.01)
2. Reduce manual L2 regularization
3. Slightly increase dropout if needed

### **From AdaGrad to AdamW**
1. Reduce learning rate significantly (0.01 → 0.001)
2. Switch to adaptive learning rate schedule
3. Increase training epochs (30 → 100)
4. Add warmup epochs

## 📚 **Additional Resources**

- **Main Documentation**: `doc/optimizer-selection-guide.md`
- **Training Guide**: `doc/04-training.md`
- **Configuration Reference**: `doc/20-configuration.md`
- **Troubleshooting**: `doc/14-troubleshooting.md`

---

**💡 Pro Tip**: Start with `adamw_crypto_optimized.toml` and only switch to specialized configurations if you have specific requirements or performance issues.
