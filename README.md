# VANGA 🔮

> Multi-target LSTM forecasting for cryptocurrency — trading-aware, calibrated, and real-time.

[![CI](https://github.com/Muvon/vanga/actions/workflows/ci.yml/badge.svg)](https://github.com/Muvon/vanga/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmuvon%2Fvanga%2Fbadges%2Fcoverage.json&style=flat-square)](https://github.com/Muvon/vanga/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.87%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

🌐 [vanga.muvon.io](https://vanga.muvon.io)

## Why VANGA?

Most price predictors give you a single number. VANGA trains **five coordinated LSTM targets** — price levels, direction, volatility, volume, and sentiment — each as a calibrated 5-class classifier, so you get a structured market view with honest confidence scores instead of a point estimate. A trading-aware ordinal loss penalizes wrong directional calls harder than magnitude errors, and adaptive calibration keeps class distributions balanced (20% per class) so models don't cheat toward dominant outcomes.

Built in Rust on [Candle](https://github.com/huggingface/candle) with Metal/CUDA support, per-sequence normalization (symbol-agnostic — works on any pair), and real-time streaming prediction.

## Quick Start

### Install

```bash
cargo install --git https://github.com/Muvon/vanga
```

Or build from source:

```bash
git clone https://github.com/Muvon/vanga && cd vanga
cargo build
```

### Train

```bash
# Minimal effective setup
vanga train --symbol BTCUSDT --data data/btc_1h.csv --config configs/quick_start.toml

# Production-ready single-asset training
vanga train --symbol BTCUSDT --data data/btc_1h.csv --config configs/training.toml

# Cross-asset training with correlation features
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data data/ --config configs/cross_asset_training.toml
```

Training auto-generates 50+ technical indicators, calibrates target thresholds, and stops early when validation loss plateaus.

### Predict

```bash
# Single prediction
vanga predict --symbol BTCUSDT --input data/recent_btc.csv --horizon 4h

# All horizons at once
vanga predict --symbol BTCUSDT --input data/recent_btc.csv --all-horizons

# Filter low-confidence predictions
vanga predict --symbol BTCUSDT --input data/recent_btc.csv --min-confidence 0.7

# Real-time streaming from a live CSV feed
vanga predict --symbol BTCUSDT --input data/stream.csv --realtime --interval 1m
```

## Features

- **5-target prediction** — price levels, direction, volatility, volume, sentiment; separate LSTM per target×horizon
- **Trading-aware ordinal loss** — directional mistakes cost more than magnitude errors (SOFL + CDW-CE)
- **Adaptive calibration** — thresholds optimized per dataset for exactly 20% per class; same parameters at training and prediction time
- **13 optimizers** — AdamW, Adam, SGD, AdaDelta, AdaGrad, AdaMax, NAdam, RAdam, RMSprop, Prodigy, plus fractional-memory variants (FracAdam, FracNAdam, FracProdigy) for volatile markets
- **Per-sequence normalization** — each input window normalized independently; works on any trading pair without reconfiguration
- **50+ technical indicators** — auto-generated from raw OHLCV (SMA, EMA, RSI, MACD, Bollinger, ATR, Ichimoku, Supertrend, …) plus cross-asset and microstructure features
- **Real-time streaming** — file-watch based live prediction with sliding-window feature buffers
- **Ensemble calibration** — post-hoc temperature scaling, label smoothing, mixup, and ECE measurement
- **Backtesting** — walk-forward validation with performance reporting
- **Reproducible training** — seeded weight initialization, deterministic shuffling, chronological splits

## Data Format

**Required columns (OHLCV):**

| Column | Description |
|--------|-------------|
| `timestamp` | ISO 8601 or Unix timestamp |
| `open`, `high`, `low`, `close` | OHLC prices |
| `volume` | Trading volume |

**Optional columns:** `volume_quote`, `trades_count`, `buy_volume`, `buy_volume_quote`, plus any custom feature columns (auto-detected and included).

## Output

Predictions are structured JSON with per-target class probabilities and calibrated confidence:

```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2024-01-15T10:30:00Z",
  "horizon": "4h",
  "current_price": 42500.0,
  "price_levels": {
    "class_0": {"range": "Strong Down", "probability": 0.05},
    "class_1": {"range": "Moderate Down", "probability": 0.15},
    "class_2": {"range": "Neutral", "probability": 0.35},
    "class_3": {"range": "Moderate Up", "probability": 0.30},
    "class_4": {"range": "Strong Up", "probability": 0.15}
  },
  "direction": {
    "up_probability": 0.68,
    "down_probability": 0.32,
    "prediction": "UP",
    "confidence": 0.68
  },
  "volatility": {
    "class": "Medium",
    "atr_ratio": 1.25,
    "expected_change": 0.035,
    "regime": "moderate_volatility"
  },
  "sentiment": {
    "score": 0.72,
    "class": "Moderate Greed",
    "body_ratio": 0.65,
    "volume_confirmation": 0.84,
    "consistency": 0.78
  },
  "volume": {
    "class": "High",
    "log_ratio": 0.372,
    "regime": "volume_surge",
    "classification": "above_average"
  }
}
```

Confidence scores are conservatively mapped from 5-class probabilities via entropy and deviation-from-uniform analysis — a 0.30 max probability in a balanced 5-class system is genuinely good, and the mapping reflects that rather than overpromising.

## Configuration

Everything lives in a single TOML file — training, model, and features:

```toml
[training]
epochs = { Auto = { max_epochs = 1000 } }
learning_rate = 0.001
batch_size = { Auto = { min_size = 32, max_size = 512 } }
optimizer = { AdamW = { weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } }
validation_split = 0.2
early_stopping = { patience = 50, min_delta = 0.0001 }
gradient_clip = 1.0
seed = 42

[model]
architecture = { MultiLSTM = { layers = 2 } }
sequence_length = { Auto = { min_length = 30, max_length = 120 } }
hidden_units = { Auto = { min_units = 64, max_units = 512 } }
dropout = { enabled = true, rate = { Fixed = 0.2 } }
attention = { enabled = true, mechanism = "MultiHeadAttention", heads = 8 }

[features.technical_indicators]
enabled = true
```

**Starting points:**

| Scenario | Config |
|----------|--------|
| Beginner / small dataset | `configs/quick_start.toml` |
| Production single-asset | `configs/training.toml` |
| Cross-asset / multi-symbol | `configs/cross_asset_training.toml` |
| Full parameter reference | `configs/example_single_asset.toml` |
| Optimizer-specific presets | `configs/optimizer_examples/` |

## Model Management

```bash
# List available models
vanga models list

# Evaluate against test data
vanga models evaluate --symbol BTCUSDT --test-data data/btc_test.csv

# Evaluate with backtesting
vanga models evaluate --symbol BTCUSDT --test-data data/btc_test.csv --backtest --train-split 0.8

# Compare models across symbols
vanga models compare --symbols BTCUSDT,ETHUSDT --metric accuracy

# Export for deployment
vanga models export --symbol BTCUSDT --format msgpack --output ./models/
```

## Architecture

```
src/
├── api/               # High-level orchestration (trainer, predictor, backtester)
├── config/            # TrainingConfig, ModelConfig, FeatureConfig, PredictionConfig
├── data/              # CSV loading, preprocessing, sequence generation, balancing
├── features/          # 50+ technical indicators, cross-asset, microstructure, regime
├── model/
│   ├── lstm/          # Core LSTM: config, training, inference, loss, scheduling
│   ├── calibration/   # Ensemble calibration (temperature, label smoothing, mixup, ECE)
│   ├── tft/           # Temporal Fusion Transformer (variable selection, quantile regression)
│   ├── multi_target.rs    # MultiTargetLSTMModel wrapper (one model per target×horizon)
│   ├── attention.rs       # Multi-head attention
│   ├── attention_moh.rs   # Mixture-of-Head attention
│   ├── bias_correction.rs # Linear/ensemble bias correction
│   └── smartcore_backend.rs # SmartCore ML hybrid backend
├── targets/
│   ├── calibration/   # Adaptive parameter optimization (Bayesian, per-target)
│   ├── price_levels.rs, direction.rs, volatility.rs, volume.rs, sentiment.rs
│   └── stop_levels.rs # Stop-level target generation
├── output/            # Prediction parsing, formatting, confidence calculation, post-processing
├── optimization/      # Fractional optimizers, Prodigy, hyperparameter search, feature selection
├── realtime/          # Streaming prediction engine, file watcher, CSV stream
└── utils/             # Error types, metrics, diagnostics, device management
```

**Pipeline flow:**

```
Raw CSV → Feature Engineering (50+ indicators) → NaN/Outlier Removal
→ Sequence Generation (sliding window) → Per-Sequence Normalization
→ Adaptive Calibration → Target Generation (5 targets × 5 classes)
→ Multi-Model Training → Ensemble Calibration → Model Persistence
```

## Optimizer Benchmarks

Empirical results across BTCUSDT/ETHUSDT datasets ([full analysis](doc/23-optimizer-performance-analysis.md)):

| Optimizer | Avg Val Loss | Convergence | Best For |
|-----------|-------------|-------------|----------|
| **AdamW** | 0.0234 | ~85 epochs | General crypto training |
| RMSprop | 0.0267 | ~110 epochs | Volatile markets |
| NAdam | 0.0289 | ~72 epochs | Fast iteration |
| RAdam | 0.0301 | ~145 epochs | Large datasets, stability |
| Prodigy | — | adaptive | Learning-rate-free training |

Benchmark all optimizers on your data:

```bash
python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT --quick
```

## Documentation

- **[Documentation Index](doc/README.md)** — complete guide listing (30+ docs)
- **[Introduction](doc/01-introduction.md)** — system overview
- **[Installation](doc/02-installation.md)** — setup and requirements
- **[Data Preparation](doc/03-data-preparation.md)** — format your data
- **[Training](doc/04-training.md)** — training pipeline deep-dive
- **[Predictions](doc/05-predictions.md)** — prediction pipeline
- **[Multi-Target System](doc/08-targets.md)** — target generation and calibration
- **[Backtesting](doc/18-backtesting-guide.md)** — walk-forward validation
- **[Configuration Reference](doc/20-configuration.md)** — all parameters explained
- **[Optimizer Selection Guide](doc/22-optimizer-selection-guide.md)** — choosing the right optimizer

## Development

```bash
cargo check --message-format=short   # Fast iteration
cargo clippy --all-features --all-targets -- -D warnings  # Must pass before commits
cargo test                            # All tests in separate *_test.rs files
```

Requires Rust 1.87+. Metal acceleration is automatic on macOS; enable CUDA with `--features cuda`.

## Disclaimer

VANGA is a research and educational tool. Cryptocurrency markets are highly volatile and unpredictable. Nothing produced by this system constitutes financial advice. Past model performance does not guarantee future results. Use at your own risk.

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/Muvon/vanga/issues)
- **Email**: [opensource@muvon.io](mailto:opensource@muvon.io)
- **Company**: Muvon Un Limited (Hong Kong)

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
