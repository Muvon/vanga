# Multi-Symbol Training and Prediction Guide

## Overview

VANGA supports both single-symbol and multi-symbol training with a unified interface. Multi-symbol training enables cross-asset learning and portfolio-level insights.

## Unified Symbol Interface

### Single Symbol (Traditional)
```bash
# Single symbol training
vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --config configs/tft_enhanced.toml

# Single symbol prediction
vanga predict --symbol BTCUSDT --input data/BTCUSDT_recent.csv --model models/BTCUSDT_tft.model
```

### Multiple Symbols (Cross-Asset Learning)
```bash
# Multi-symbol training with comma separation
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data-dir data/multi_asset/ --config configs/tft_gnn_multi_asset.toml

# Multi-symbol prediction
vanga predict --symbol BTCUSDT,ETHUSDT,ADAUSDT --input-dir data/recent/ --model models/multi_asset_gnn.model
```

## Data Organization for Multi-Symbol

### File Structure for Multi-Symbol Training

```
data/multi_asset/
├── BTCUSDT_1h.csv      # Individual symbol data
├── ETHUSDT_1h.csv
├── ADAUSDT_1h.csv
├── DOTUSDT_1h.csv
└── correlation_matrix.csv  # Optional: pre-computed correlations
```

### File Structure for Multi-Symbol Prediction

```
data/recent/
├── BTCUSDT_recent.csv   # Recent data for each symbol
├── ETHUSDT_recent.csv
├── ADAUSDT_recent.csv
└── DOTUSDT_recent.csv
```

### CSV Format (Same for All Symbols)

```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56
2024-01-01T01:00:00Z,42300.0,42800.0,42100.0,42600.0,1456.78
```

## Training Workflows

### 1. Single Symbol TFT Training

```bash
# Prepare single symbol data
vanga data prepare --symbol BTCUSDT --timeframe 1h --days 365 --output data/BTCUSDT_1h.csv

# Train TFT-enhanced model
vanga train \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --config configs/tft_enhanced.toml \
    --output models/BTCUSDT_tft.model

# Predict with uncertainty
vanga predict \
    --symbol BTCUSDT \
    --input data/BTCUSDT_recent.csv \
    --model models/BTCUSDT_tft.model \
    --quantiles 0.05,0.95
```

### 2. Multi-Symbol GNN Training

```bash
# Prepare multi-symbol data
vanga data prepare \
    --symbol BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT \
    --timeframe 1h \
    --days 365 \
    --output-dir data/multi_asset/

# Train multi-symbol GNN model
vanga train \
    --symbol BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT \
    --data-dir data/multi_asset/ \
    --config configs/tft_gnn_multi_asset.toml \
    --output models/multi_asset_gnn.model

# Multi-symbol prediction
vanga predict \
    --symbol BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT \
    --input-dir data/recent/ \
    --model models/multi_asset_gnn.model \
    --output predictions/portfolio_predictions.json
```

### 3. Auto-Optimized Training

```bash
# Single symbol with auto-optimization
vanga train \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --config configs/tft_enhanced.toml \
    --auto-optimize \
    --strategy crypto_optimized

# Multi-symbol with auto-optimization
vanga train \
    --symbol BTCUSDT,ETHUSDT,ADAUSDT \
    --data-dir data/multi_asset/ \
    --config configs/tft_gnn_multi_asset.toml \
    --auto-optimize \
    --strategy portfolio_optimized
```

## Prediction Workflows

### Single Symbol Prediction Output

```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2024-01-01T12:00:00Z",
  "predictions": {
    "price_levels": {
      "point_prediction": 42500.0,
      "quantiles": {
        "0.05": 41200.0,
        "0.25": 41800.0,
        "0.5": 42500.0,
        "0.75": 43200.0,
        "0.95": 43800.0
      }
    },
    "direction": {
      "prediction": "up",
      "confidence": 0.78
    },
    "volatility": {
      "1h": 0.023,
      "4h": 0.045,
      "24h": 0.089
    }
  },
  "feature_importance": {
    "close_price": 0.25,
    "volume": 0.18,
    "rsi": 0.15,
    "macd": 0.12
  },
  "uncertainty_score": 0.876
}
```

### Multi-Symbol Prediction Output

```json
{
  "portfolio": {
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    "timestamp": "2024-01-01T12:00:00Z",
    "market_regime": {
      "current": "Bull",
      "confidence": 0.82,
      "transition_probability": 0.15
    },
    "cross_asset_correlations": {
      "BTCUSDT_ETHUSDT": 0.78,
      "BTCUSDT_ADAUSDT": 0.65,
      "ETHUSDT_ADAUSDT": 0.71
    }
  },
  "individual_predictions": {
    "BTCUSDT": {
      "price_levels": {
        "point_prediction": 42500.0,
        "quantiles": {
          "0.05": 41200.0,
          "0.95": 43800.0
        }
      },
      "cross_asset_influence": {
        "from_ETHUSDT": 0.23,
        "from_ADAUSDT": 0.15
      }
    },
    "ETHUSDT": {
      "price_levels": {
        "point_prediction": 2850.0,
        "quantiles": {
          "0.05": 2720.0,
          "0.95": 2980.0
        }
      },
      "cross_asset_influence": {
        "from_BTCUSDT": 0.31,
        "from_ADAUSDT": 0.18
      }
    },
    "ADAUSDT": {
      "price_levels": {
        "point_prediction": 0.485,
        "quantiles": {
          "0.05": 0.462,
          "0.95": 0.508
        }
      },
      "cross_asset_influence": {
        "from_BTCUSDT": 0.28,
        "from_ETHUSDT": 0.22
      }
    }
  },
  "portfolio_metrics": {
    "total_portfolio_risk": 0.067,
    "diversification_benefit": 0.23,
    "regime_adjusted_allocation": {
      "BTCUSDT": 0.45,
      "ETHUSDT": 0.35,
      "ADAUSDT": 0.20
    }
  }
}
```

## CLI Interface Implementation

### Symbol Parsing Logic

```rust
// Parse single or multiple symbols
fn parse_symbols(symbol_arg: &str) -> Vec<String> {
    symbol_arg
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .collect()
}

// Usage examples:
// --symbol BTCUSDT          → ["BTCUSDT"]
// --symbol BTCUSDT,ETHUSDT  → ["BTCUSDT", "ETHUSDT"]
// --symbol "BTC,ETH,ADA"    → ["BTCUSDT", "ETHUSDT", "ADAUSDT"] (with normalization)
```

### Data Path Resolution

```rust
// Automatic data path resolution
fn resolve_data_paths(symbols: &[String], data_arg: Option<&str>, data_dir_arg: Option<&str>) -> Result<DataPaths> {
    match symbols.len() {
        1 => {
            // Single symbol: use --data file
            if let Some(data_file) = data_arg {
                Ok(DataPaths::Single(data_file.to_string()))
            } else {
                Err("Single symbol requires --data file")
            }
        }
        _ => {
            // Multi-symbol: use --data-dir directory
            if let Some(data_dir) = data_dir_arg {
                let mut paths = HashMap::new();
                for symbol in symbols {
                    let file_path = format!("{}/{}_1h.csv", data_dir, symbol);
                    paths.insert(symbol.clone(), file_path);
                }
                Ok(DataPaths::Multi(paths))
            } else {
                Err("Multi-symbol requires --data-dir directory")
            }
        }
    }
}
```

## Configuration Adaptation

### Automatic Config Selection

```rust
// Auto-select appropriate config based on symbol count
fn select_config(symbols: &[String], config_arg: Option<&str>) -> String {
    if let Some(config) = config_arg {
        return config.to_string();
    }
    
    match symbols.len() {
        1 => "configs/tft_enhanced.toml".to_string(),
        2..=4 => "configs/tft_gnn_small_portfolio.toml".to_string(),
        5..=8 => "configs/tft_gnn_multi_asset.toml".to_string(),
        _ => "configs/tft_gnn_large_portfolio.toml".to_string(),
    }
}
```

### Dynamic Config Adjustment

```toml
# configs/tft_gnn_multi_asset.toml - Auto-adjusts based on symbol count

[model.gnn.cross_asset]
enabled = true
# Assets list populated automatically from --symbol argument
correlation_threshold = 0.2
max_connections = { type = "Auto", base = 3, per_asset = 1.5 }  # 3 + 1.5 * num_assets

[model.gnn.graph_attention]
num_heads = { type = "Auto", min = 8, max = 16, per_asset = 2 }  # Scale with portfolio size
hidden_dim = { type = "Auto", base = 256, per_asset = 32 }       # 256 + 32 * num_assets
```

## Model Architecture for Multi-Symbol

### Single Symbol Model
```
Input: [batch, sequence, features]
  ↓
LSTM + TFT Variable Selection
  ↓
Quantile Regression Heads
  ↓
Output: [batch, targets, quantiles]
```

### Multi-Symbol Model
```
Input: {
  "BTCUSDT": [batch, sequence, features],
  "ETHUSDT": [batch, sequence, features],
  "ADAUSDT": [batch, sequence, features]
}
  ↓
Individual LSTM + TFT Processing
  ↓
Graph Neural Network (Cross-Asset Learning)
  ↓
Market Regime Detection
  ↓
Portfolio-Level Quantile Regression
  ↓
Output: {
  "individual": {...},
  "portfolio": {...},
  "regime": {...}
}
```

## Training Data Requirements

### Single Symbol
- **Minimum**: 1,000 samples
- **Recommended**: 5,000+ samples
- **File**: Single CSV with OHLCV data

### Multi-Symbol
- **Minimum**: 2,000 samples per symbol
- **Recommended**: 10,000+ samples per symbol
- **Files**: Separate CSV per symbol, aligned timestamps
- **Correlation**: Minimum 0.2 correlation between assets for effective cross-learning

## Performance Expectations

### Single Symbol TFT
- **Training Time**: 5-15 minutes (depending on data size)
- **Memory Usage**: 2-4 GB
- **Accuracy Improvement**: +5-8% over baseline LSTM

### Multi-Symbol GNN
- **Training Time**: 15-45 minutes (depending on portfolio size)
- **Memory Usage**: 4-12 GB
- **Accuracy Improvement**: +8-15% over baseline LSTM
- **Additional Benefits**: Portfolio risk metrics, regime detection, cross-asset insights

## Troubleshooting Multi-Symbol Issues

### Common Problems

1. **Misaligned Timestamps**
   ```bash
   # Solution: Use data alignment tool
   vanga data align --symbols BTCUSDT,ETHUSDT --input-dir data/raw/ --output-dir data/aligned/
   ```

2. **Missing Symbol Data**
   ```bash
   # Solution: Check data availability
   vanga data validate --symbols BTCUSDT,ETHUSDT --data-dir data/multi_asset/
   ```

3. **Low Cross-Asset Correlation**
   ```bash
   # Solution: Analyze correlations first
   vanga analyze correlations --symbols BTCUSDT,ETHUSDT,ADAUSDT --data-dir data/multi_asset/
   ```

4. **Memory Issues with Large Portfolios**
   ```toml
   # Solution: Reduce model complexity
   [model.gnn.graph_attention]
   hidden_dim = 128  # Reduce from 256
   num_heads = 8     # Reduce from 16
   
   [training.training_params]
   batch_size = { type = "Fixed", value = 16 }  # Smaller batches
   ```

The unified symbol interface provides a consistent experience whether working with single assets or complex multi-asset portfolios, while automatically adapting the underlying architecture and configuration for optimal performance.