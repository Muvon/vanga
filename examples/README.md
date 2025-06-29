# VANGA Training Configuration Examples

This directory contains simple configuration examples for different training scenarios.

## Available Configurations

### 🏆 production_training.toml (RECOMMENDED)
- **Auto early stopping** - stops when validation loss plateaus
- **Adaptive learning rate** - reduces when needed for optimal convergence
- **Quality-first approach** - optimized for best model performance
- **Use for**: Live trading, production deployments

### ⚡ dev_training.toml
- **Fixed 100 epochs** - predictable training time
- **Fixed learning rate** - consistent behavior
- **No validation split** - uses all data for speed
- **Use for**: Development, quick testing, debugging

### 🔬 research_training.toml
- **Fixed 500 epochs** - reproducible results
- **Fixed learning rate** - consistent experiments
- **20% validation split** - proper academic evaluation
- **Use for**: Research, academic studies, benchmarking

## Usage Examples

### Default Intelligent Training
```bash
# Uses built-in intelligent defaults (same as production_training.toml)
vanga train --symbol BTCUSDT --data data.csv
```

### Custom Configuration
```bash
# Use specific config file
vanga train --symbol BTCUSDT --data data.csv --config examples/dev_training.toml
```

### Incremental Training
```bash
# First training
vanga train --symbol BTCUSDT --data historical_data.csv

# Later: add new data (automatically continues training)
vanga train --symbol BTCUSDT --data new_data.csv
```

## Configuration Format

All configs follow this structure:
```toml
[training_params]
epochs = { Auto = { max_epochs = 1000 } }  # or { Fixed = 500 }
learning_rate = { Adaptive = { initial_lr = 0.01 } }  # or { Fixed = 0.001 }
validation_split = 0.2  # 0.0 to disable validation
early_stopping_patience = 50  # epochs to wait for improvement
gradient_clip = 1.0  # prevent exploding gradients
```

## Expected Performance

| Config | Training Time | Model Quality | Use Case |
|--------|---------------|---------------|----------|
| Production | Variable (auto-stops) | Highest | Live trading |
| Development | ~10 minutes | Good | Testing |
| Research | ~50 minutes | High | Academic |

## Tips

- **Start with production_training.toml** for best results
- **Use dev_training.toml** for quick iteration during development
- **Use research_training.toml** for reproducible academic work
- **Validation split of 0.2 (20%)** is optimal for early stopping
- **Patience of 50** balances quality vs training time
