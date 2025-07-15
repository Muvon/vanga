# VANGA Training Configuration Guide

## Overview

VANGA supports flexible training configuration through TOML files. You can control epochs, learning rates, batch sizes, and early stopping behavior.

## Configuration Types

### 1. Auto Mode (Recommended)
Uses intelligent early stopping and automatic parameter optimization.

```toml
[training]
epochs = { Auto = { max_epochs = 1000 } }
learning_rate = { Adaptive = { initial_lr = 0.01 } }
validation_split = 0.2  # Required for early stopping
early_stopping_patience = 50
```

### 2. Fixed Mode (Research)
Fixed parameters for reproducible experiments.

```toml
[training]
epochs = { Fixed = 200 }
learning_rate = { Fixed = 0.001 }
validation_split = 0.0  # Optional since no early stopping
```

## Complete Configuration Examples

### Production Training (Recommended)
```bash
vanga train --symbol BTCUSDT --data data.csv --config examples/production_training.toml
```

### Fixed Epochs Training
```bash
vanga train --symbol BTCUSDT --data data.csv --config examples/fixed_epochs_training.toml
```

### Custom Auto Training
```bash
vanga train --symbol BTCUSDT --data data.csv --config examples/custom_auto_training.toml
```

### High Performance Training
```bash
vanga train --symbol BTCUSDT --data data.csv --config examples/high_performance_training.toml
```

### Research Training
```bash
vanga train --symbol BTCUSDT --data data.csv --config examples/research_training.toml
```

## Configuration Parameters

### Epochs Configuration
- `Auto { max_epochs = N }` - Early stopping enabled, max N epochs
- `Fixed(N)` - Exactly N epochs, no early stopping

### Learning Rate Configuration
- `Fixed(rate)` - Fixed learning rate
- `Adaptive { initial_lr = rate }` - Adaptive learning rate starting at rate
- `Auto { min_lr = min, max_lr = max }` - Auto-optimized learning rate range

### Batch Size Configuration
- `Fixed(size)` - Fixed batch size
- `Auto { min_size = min, max_size = max }` - Auto-optimized batch size range

### Validation & Early Stopping
- `validation_split` - Fraction of data for validation (0.0-1.0)
- `early_stopping` - Early stopping configuration with patience and min_delta threshold
- `gradient_clip` - Gradient clipping threshold (optional)

## Key Differences

| Mode | Early Stopping | Validation Required | Use Case |
|------|---------------|-------------------|----------|
| Auto | ✅ Yes | ✅ Yes | Production, optimal performance |
| Fixed | ❌ No | ❌ No | Research, reproducible experiments |

## Examples by Use Case

### Quick Experiment (Fast)
```toml
epochs = { Fixed = 50 }
learning_rate = { Fixed = 0.001 }
batch_size = { Fixed = 32 }
```

### Production Deployment (Optimal)
```toml
epochs = { Auto = { max_epochs = 1000 } }
learning_rate = { Adaptive = { initial_lr = 0.01 } }
batch_size = { Auto = { min_size = 32, max_size = 512 } }
validation_split = 0.2
early_stopping = { patience = 50, min_delta = 0.00005 }
```

### Research Paper (Reproducible)
```toml
epochs = { Fixed = 500 }
learning_rate = { Fixed = 0.001 }
batch_size = { Fixed = 64 }
validation_split = 0.2
```

### High Performance (Maximum Quality)
```toml
epochs = { Auto = { max_epochs = 2000 } }
learning_rate = { Auto = { min_lr = 0.0001, max_lr = 0.01 } }
batch_size = { Auto = { min_size = 64, max_size = 1024 } }
validation_split = 0.2
early_stopping_patience = 100
```

## Command Line Usage

```bash
# Use default configuration
vanga train --symbol BTCUSDT --data data.csv

# Use custom configuration file
vanga train --symbol BTCUSDT --data data.csv --config my_config.toml

# Override specific parameters
vanga train --symbol BTCUSDT --data data.csv --config my_config.toml --epochs 100
```

## Validation

The system automatically validates your configuration and provides helpful error messages:

- ✅ Auto mode requires `validation_split > 0`
- ✅ Fixed mode can use `validation_split = 0`
- ✅ All numeric parameters are validated for reasonable ranges
- ✅ Configuration files are validated on load

## No Hardcoding

VANGA respects your configuration completely:
- ❌ No hardcoded epoch limits
- ❌ No hardcoded learning rates
- ❌ No hardcoded batch sizes
- ✅ All parameters come from your config file
- ✅ Sensible defaults only when no config provided
