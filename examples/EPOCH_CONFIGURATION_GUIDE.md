# VANGA Epoch Configuration - Complete Guide

## ✅ NO HARDCODING - Configuration Driven

VANGA's epoch configuration is completely driven by your config files with **zero hardcoding**:

- ❌ No hardcoded epoch limits in training logic
- ❌ No hardcoded learning rates
- ❌ No hardcoded batch sizes
- ✅ All parameters come from your TOML configuration
- ✅ Sensible defaults only when no config provided

## Configuration Options

### 1. Auto Mode (Intelligent Early Stopping)
```toml
[training]
epochs = { Auto = { max_epochs = 500 } }  # Your custom limit
validation_split = 0.2                    # Required for early stopping
early_stopping = { patience = 25, min_delta = 0.00005 }  # Your custom patience and improvement threshold
```

### 2. Fixed Mode (Exact Control)
```toml
[training]
epochs = { Fixed = 200 }                  # Exactly 200 epochs
validation_split = 0.0                    # Optional
```

## Complete Working Examples

### Example 1: Quick Training (50 epochs max)
```bash
# Create config file
cat > quick_training.toml << EOF
[training]
epochs = { Auto = { max_epochs = 50 } }
learning_rate = { Fixed = 0.001 }
validation_split = 0.2
early_stopping = { patience = 10, min_delta = 0.0001 }
batch_size = { Fixed = 32 }
EOF

# Use it
vanga train --symbol BTCUSDT --data data.csv --config quick_training.toml
```

### Example 2: Exact Control (100 epochs, no early stopping)
```bash
# Create config file
cat > exact_training.toml << EOF
[training]
epochs = { Fixed = 100 }
learning_rate = { Fixed = 0.0005 }
batch_size = { Fixed = 64 }
validation_split = 0.0
EOF

# Use it
vanga train --symbol BTCUSDT --data data.csv --config exact_training.toml
```

### Example 3: High Performance (2000 epochs max)
```bash
# Create config file
cat > high_perf_training.toml << EOF
[training]
epochs = { Auto = { max_epochs = 2000 } }
learning_rate = { Adaptive = { initial_lr = 0.01 } }
validation_split = 0.2
early_stopping = { patience = 100, min_delta = 0.00005 }
batch_size = { Auto = { min_size = 64, max_size = 512 } }
EOF

# Use it
vanga train --symbol BTCUSDT --data data.csv --config high_perf_training.toml
```

## Pre-made Configuration Files

### Available Configs:
```bash
# Production (recommended)
vanga train --symbol BTCUSDT --data data.csv --config examples/production_training.toml

# Fixed epochs (no early stopping)
vanga train --symbol BTCUSDT --data data.csv --config examples/fixed_epochs_training.toml

# Custom auto (custom early stopping)
vanga train --symbol BTCUSDT --data data.csv --config examples/custom_auto_training.toml

# High performance (extended training)
vanga train --symbol BTCUSDT --data data.csv --config examples/high_performance_training.toml

# Research (reproducible)
vanga train --symbol BTCUSDT --data data.csv --config examples/research_training.toml
```

## How It Works Internally

### 1. Configuration Loading
```rust
// Your config file is parsed
epochs = { Auto = { max_epochs = 500 } }

// Becomes this internally
EpochConfig::Auto { max_epochs: 500 }
```

### 2. Training Configuration
```rust
// Called automatically before training
model.configure_training(&config);

// Sets epochs based on your config
match config.training.epochs {
    EpochConfig::Auto { max_epochs } => {
        self.training_config.epochs = max_epochs;
        // Enable early stopping
    }
    EpochConfig::Fixed(epochs) => {
        self.training_config.epochs = epochs;
        // No early stopping
    }
}
```

### 3. Training Execution
```rust
// Auto mode: Uses rust-lstm built-in validation
trainer.train(&training_data, Some(&validation_data));

// Fixed mode: Uses training data only
trainer.train(&training_data, None);
```

## Verification

### Check Your Config Works:
```bash
# Test config loading (will show your epochs setting)
vanga train --symbol TEST --data small_data.csv --config your_config.toml
```

### Expected Output:
```
[INFO] 🔧 Loading training config from: "your_config.toml"
[INFO] ✅ Training configured: epochs=500, lr=0.001000, early_stopping=true
```

## Key Benefits

- 🎯 **Complete Control**: Set any epoch limit you want
- 🧠 **Intelligent Training**: Auto mode with early stopping
- 🔬 **Reproducible**: Fixed mode for research
- 📊 **Flexible**: Mix and match with other parameters
- ⚡ **No Surprises**: Configuration is exactly what you specify

## Common Patterns

### Development/Testing
```toml
epochs = { Fixed = 10 }        # Quick test
```

### Production Deployment
```toml
epochs = { Auto = { max_epochs = 1000 } }  # Optimal with early stopping
```

### Research Paper
```toml
epochs = { Fixed = 500 }       # Reproducible results
```

### Maximum Performance
```toml
epochs = { Auto = { max_epochs = 5000 } }  # Extended training
```
