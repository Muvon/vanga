# Reproducible Training with Seed Management

## Overview

VANGA now supports fully reproducible training using Candle's native `Device::set_seed()` functionality. This ensures that training runs with the same seed will produce identical results, making experiments reproducible and debugging easier.

## How It Works

### Seed Configuration

The seed is configured in the `[training]` section of your configuration file:

```toml
[training]
# Random seed for reproducible training
# 0 = random initialization (different weights each run) - DEFAULT
# >0 = reproducible initialization (same weights each run)
seed = 42
```

### Seed Behavior

- **`seed = 0`**: Random initialization - each training run will produce different results
- **`seed > 0`**: Reproducible initialization - same seed always produces identical results
- **No seed specified**: Defaults to `seed = 0` (random)

### Implementation Details

1. **Device-Level Seeding**: Uses Candle's native `Device::set_seed()` for all tensor operations
2. **Multi-Target Consistency**: Each target model gets an incremental seed (`seed + target_index`)
3. **Weight Initialization**: All `Tensor::randn()` calls use the device seed automatically
4. **Pipeline Integration**: Seed propagates from config → device → model → weights

## Usage Examples

### Development/Research (Reproducible)

```toml
[training]
seed = 42  # Fixed seed for consistent results
```

Use this when:
- Debugging training issues
- Comparing different hyperparameters
- Research experiments requiring reproducibility
- Validating model changes

### Production/Ensemble (Random)

```toml
[training]
seed = 0  # Random initialization
```

Use this when:
- Training ensemble models (want diversity)
- Production training (avoid overfitting to specific initialization)
- Creating multiple models for averaging

## Configuration Examples

### Quick Start (Reproducible)

```bash
# configs/quick_start.toml already includes seed = 42
vanga train --symbol BTCUSDT --data data.csv --config configs/quick_start.toml
```

### Custom Seed

```toml
[training]
seed = 12345  # Your custom seed
learning_rate = 0.001
epochs = { Auto = { max_epochs = 1000 } }
# ... other parameters
```

### Random Training

```toml
[training]
seed = 0  # Explicitly random
# ... other parameters
```

## Multi-Target Model Seeding

For multi-target models, each target gets a unique but deterministic seed:

- Target 0: `seed`
- Target 1: `seed + 1`
- Target 2: `seed + 2`
- etc.

This ensures:
- **Consistency**: Same base seed always produces same results
- **Uniqueness**: Each target has different initialization
- **Reproducibility**: Results are deterministic across runs

## Verification

### Test Reproducibility

```bash
# Run the same training twice with fixed seed
vanga train --symbol BTCUSDT --data data.csv --config configs/quick_start.toml
# Results should be identical

# Run with random seed
vanga train --symbol BTCUSDT --data data.csv --config configs/training.toml
# Results should be different each time
```

### Automated Tests

```bash
# Run reproducibility tests
cargo test test_reproducible_training
```

## Troubleshooting

### Different Results with Same Seed

If you get different results with the same seed, check:

1. **Configuration consistency**: Ensure all parameters are identical
2. **Data consistency**: Same input data and preprocessing
3. **Version consistency**: Same VANGA and Candle versions
4. **Device consistency**: Same device type (CPU/GPU)
5. **Weight initialization**: Ensure proper LSTM weight initialization (see below)

### LSTM Weight Initialization Issues

**Problem**: Non-deterministic training results even with same seed and configuration.

**Root Cause**: Missing orthogonal initialization for LSTM recurrent weights.

**Solution**: VANGA now automatically applies proper LSTM weight initialization:

- **Input-to-hidden weights**: Xavier/Glorot initialization
- **Hidden-to-hidden (recurrent) weights**: Orthogonal initialization
- **Biases**: Zero initialization (except forget gate bias = 1.0)

**Why This Matters**:
- Orthogonal initialization prevents vanishing/exploding gradients in RNNs
- Ensures stable gradient flow during backpropagation
- Critical for reproducible LSTM training
- Based on research by Saxe et al. (2013) and Smerity (2016)

**Verification**: Look for these log messages during training:
```
🔧 Applying proper LSTM weight initialization (Xavier + Orthogonal)...
🔄 Applying orthogonal initialization to recurrent weight: shape=[64, 64]
📥 Applying Xavier initialization to input weight: shape=[10, 64]
✅ LSTM weight initialization complete: 8 weight matrices (4 recurrent), 8 biases
```

**Testing Deterministic Behavior**:
```rust
// Create identical models with same seed
let seed = 42;
let mut model1 = LSTMModel::new_with_seed(config.clone(), Some(seed))?;
let mut model2 = LSTMModel::new_with_seed(config.clone(), Some(seed))?;

// Train with identical data and config
model1.train(&sequences, &targets, &training_config, None, None).await?;
model2.train(&sequences, &targets, &training_config, None, None).await?;

// Predictions should be identical (difference < 1e-6)
let pred1 = model1.predict(&sequences).await?;
let pred2 = model2.predict(&sequences).await?;
```

### CPU Seeding Limitations

**Issue**: Candle may not support CPU seeding on all platforms.

**Symptoms**: Warning message about CPU RNG seeding not supported.

**Workaround**: Use GPU devices (CUDA/Metal) for fully reproducible training:
```toml
[training]
device = "cuda"  # or "metal" on macOS
seed = 42
```

**CPU Fallback**: On CPU, the system gracefully falls back to random initialization with a warning.

### Performance Impact

- **Minimal overhead**: Device seeding has negligible performance impact
- **Memory usage**: No additional memory required
- **Training speed**: No measurable difference in training time

## Best Practices

### For Development

```toml
[training]
seed = 42  # Use a fixed seed
epochs = { Fixed = 100 }  # Fixed epochs for consistency
```

### For Production

```toml
[training]
seed = 0  # Random initialization
epochs = { Auto = { max_epochs = 1000 } }  # Auto early stopping
```

### For Research

```toml
[training]
seed = 42  # Fixed seed for reproducibility
# Document the seed in your research notes
```

## Technical Details

### Device Seed Setting

The seed is set at the device level before weight initialization:

```rust
// Automatically handled by VANGA
device.set_seed(seed)?;
```

### LSTM Weight Initialization

VANGA implements proper LSTM weight initialization for stable and reproducible training:

#### **Orthogonal Initialization for Recurrent Weights**
```rust
// Recurrent (hidden-to-hidden) weights use orthogonal matrices
pub fn orthogonal_tensor(shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
    // Uses Gram-Schmidt orthogonalization process
    // Prevents vanishing/exploding gradients in RNNs
}
```

#### **Xavier Initialization for Input Weights**
```rust
// Input-to-hidden weights use Xavier/Glorot initialization
pub fn xavier_tensor(shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
    let fan_in = shape[0];
    let fan_out = shape[1];
    let std_dev = (2.0 / (fan_in + fan_out) as f64).sqrt();
    Tensor::randn(0.0, std_dev as f32, shape, device)
}
```

#### **Comprehensive LSTM Initialization**
```rust
// Applied automatically during network creation
SeededTensorUtils::apply_lstm_weight_initialization(
    &varmap,
    &device,
    seed
)?;
```

### Weight Initialization Strategy

The system automatically determines the correct initialization for each weight tensor:

1. **2D Tensors (Weight Matrices)**:
   - Recurrent weights (hidden-to-hidden): **Orthogonal initialization**
   - Input weights (input-to-hidden): **Xavier initialization**

2. **1D Tensors (Biases)**:
   - Forget gate bias: **1.0** (helps with gradient flow)
   - All other biases: **0.0**

### Scientific Basis

This approach is based on established research:
- **Saxe et al. (2013)**: "Exact solutions to the nonlinear dynamics of learning in deep linear networks"
- **Smerity (2016)**: "Explaining and illustrating orthogonal initialization for recurrent neural networks"
- **Bengio et al.**: Deep Learning book recommendations for RNN initialization

### Gram-Schmidt Orthogonalization

For creating orthogonal matrices, VANGA implements the Gram-Schmidt process:

```rust
fn gram_schmidt_orthogonalization(matrix: &Tensor) -> Result<Tensor> {
    // 1. Convert to f64 for numerical stability
    // 2. For each column vector:
    //    - Subtract projections onto previous orthogonal vectors
    //    - Normalize the resulting vector
    // 3. Concatenate all orthogonal vectors
    // 4. Convert back to f32
}
```

This ensures that recurrent weight matrices are orthogonal, which:
- Preserves gradient magnitudes during backpropagation
- Prevents vanishing gradients in long sequences
- Prevents exploding gradients
- Enables stable and reproducible training

### Seed Propagation Flow

```
TrainingConfig.seed → DeviceManager::create_device_with_seed() → Device::set_seed() → Tensor::randn()
```

## Migration from Custom RNG

If you were using custom seeding approaches:

### Before (Custom RNG)
```rust
let mut rng = StdRng::seed_from_u64(seed);
// Manual tensor creation with custom RNG
```

### After (Device Seeding)
```rust
device.set_seed(seed)?;
let tensor = Tensor::randn(0.0, std_dev, shape, &device)?;
```

The new approach is:
- **Simpler**: No manual RNG management
- **More reliable**: Uses Candle's native seeding
- **Better integrated**: Works with all Candle operations
- **More efficient**: No custom tensor creation overhead

## Logging

VANGA provides detailed logging for seed usage:

```
🎲 Created LSTMModel with seed: 42
🎲 Seed = 42: Reproducible weight initialization will be used
🎲 Setting device seed to 42 for reproducible training
🎲 Multi-target model using seed: 42
🎲 Target 'price_level_1h' using seed: 42
🎲 Target 'direction_1h' using seed: 43
🎲 Target 'volatility_1h' using seed: 44
```

This helps verify that seeding is working correctly and troubleshoot any issues.
