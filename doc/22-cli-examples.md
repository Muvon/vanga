# Multi-Symbol CLI Examples

## Unified Symbol Interface Examples

### Single Symbol Training
```bash
# Basic TFT training
vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --output models/BTCUSDT_tft.model

# With auto-optimization
vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --auto-optimize --strategy crypto_optimized --output models/BTCUSDT_optimized.model

# Custom configuration
vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --config configs/tft_enhanced.toml --output models/BTCUSDT_custom.model
```

### Multi-Symbol Training
```bash
# Small portfolio (2-4 assets)
vanga train --symbol BTCUSDT,ETHUSDT --data-dir data/multi_asset/

# Medium portfolio (5-8 assets)
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT,LINKUSDT --data-dir data/multi_asset/

# Large portfolio (9+ assets)
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT,LINKUSDT,UNIUSDT,AAVEUSDT,COMPUSDT,SOLUSDT --data-dir data/multi_asset/
```

### Single Symbol Prediction
```bash
# Basic prediction
vanga predict --symbol BTCUSDT --input data/BTCUSDT_recent.csv --model models/BTCUSDT_tft.model

# With custom quantiles
vanga predict --symbol BTCUSDT --input data/BTCUSDT_recent.csv --model models/BTCUSDT_tft.model --quantiles 0.1,0.9

# With output file
vanga predict --symbol BTCUSDT --input data/BTCUSDT_recent.csv --model models/BTCUSDT_tft.model --output predictions/BTCUSDT_pred.json
```

### Multi-Symbol Prediction
```bash
# Portfolio prediction
vanga predict --symbol BTCUSDT,ETHUSDT,ADAUSDT --input-dir data/recent/ --output predictions/

# With batch mode
vanga predict --symbol BTCUSDT,ETHUSDT,ADAUSDT --batch --input-dir data/recent/ --output predictions/

# With correlations
vanga predict --symbol BTCUSDT,ETHUSDT,ADAUSDT --input-dir data/recent/ --model models/portfolio.model --include-correlations --output predictions/portfolio_with_corr.json

# Full analysis
vanga predict --symbol BTCUSDT,ETHUSDT,ADAUSDT --input-dir data/recent/ --model models/portfolio.model --include-regime --include-correlations --quantiles 0.05,0.95 --output predictions/full_analysis.json
```

## Data Preparation Examples

### Single Symbol Data Preparation
```bash
# Download and prepare single symbol data
vanga data prepare --symbol BTCUSDT --timeframe 1h --days 365 --output data/BTCUSDT_1h.csv

# Validate data quality
vanga data validate --symbol BTCUSDT --data data/BTCUSDT_1h.csv
```

### Multi-Symbol Data Preparation
```bash
# Download and prepare multiple symbols
vanga data prepare --symbol BTCUSDT,ETHUSDT,ADAUSDT --timeframe 1h --days 365 --output-dir data/multi_asset/

# Align timestamps across symbols
vanga data align --symbol BTCUSDT,ETHUSDT,ADAUSDT --input-dir data/raw/ --output-dir data/aligned/

# Validate multi-symbol data
vanga data validate --symbol BTCUSDT,ETHUSDT,ADAUSDT --data-dir data/multi_asset/

# Analyze correlations
vanga analyze correlations --symbol BTCUSDT,ETHUSDT,ADAUSDT --data-dir data/multi_asset/ --output correlations.json
```

## Model Management Examples

### Model Information
```bash
# Show model info (works for both single and multi-symbol)
vanga model info --model models/BTCUSDT_tft.model
vanga model info --model models/portfolio.model

# List all models
vanga model list --directory models/

# Compare models
vanga model compare --model-a models/BTCUSDT_basic.model --model-b models/BTCUSDT_tft.model --test-data data/BTCUSDT_test.csv
```

### Model Export
```bash
# Export for production (single symbol)
vanga export --model models/BTCUSDT_tft.model --format onnx --output production/BTCUSDT_tft.onnx

# Export portfolio model
vanga export --model models/portfolio.model --format onnx --output production/portfolio.onnx
```

## Advanced Usage Examples

### Curriculum Learning
```bash
# Train with curriculum learning (start simple, add complexity)
vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data-dir data/multi_asset/ --curriculum --stages basic,technical,cross_asset --output models/curriculum_trained.model
```

### Transfer Learning
```bash
# Transfer from single to multi-symbol
vanga transfer --source models/BTCUSDT_tft.model --target-symbols BTCUSDT,ETHUSDT --data-dir data/multi_asset/ --output models/transferred_portfolio.model

# Transfer between portfolios
vanga transfer --source models/small_portfolio.model --target-symbols BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT --data-dir data/multi_asset/ --output models/expanded_portfolio.model
```

### Hyperparameter Tuning
```bash
# Auto-tune for single symbol
vanga tune --symbol BTCUSDT --data data/BTCUSDT_1h.csv --trials 50 --output models/BTCUSDT_tuned.model

# Auto-tune for portfolio
vanga tune --symbol BTCUSDT,ETHUSDT,ADAUSDT --data-dir data/multi_asset/ --trials 30 --strategy portfolio_optimized --output models/portfolio_tuned.model
```

### Backtesting
```bash
# Backtest single symbol
vanga backtest --symbol BTCUSDT --model models/BTCUSDT_tft.model --test-data data/BTCUSDT_test.csv --output backtest/BTCUSDT_results.json

# Backtest portfolio
vanga backtest --symbol BTCUSDT,ETHUSDT,ADAUSDT --model models/portfolio.model --test-dir data/test/ --output backtest/portfolio_results.json
```

### Real-time Monitoring
```bash
# Monitor single symbol model
vanga monitor --symbol BTCUSDT --model models/BTCUSDT_tft.model --data-stream ws://api.binance.com/ws/btcusdt@kline_1h

# Monitor portfolio
vanga monitor --symbol BTCUSDT,ETHUSDT,ADAUSDT --model models/portfolio.model --data-streams ws://api.binance.com/ws/btcusdt@kline_1h,ws://api.binance.com/ws/ethusdt@kline_1h,ws://api.binance.com/ws/adausdt@kline_1h
```

## Error Handling Examples

### Common Error Scenarios
```bash
# Wrong symbol for single-symbol model
vanga predict --symbol ETHUSDT --input data/ETHUSDT_recent.csv --model models/BTCUSDT_tft.model
# Error: Symbol mismatch: model trained on BTCUSDT, prediction requested for ETHUSDT

# Missing symbol in multi-symbol model
vanga predict --symbol BTCUSDT,ETHUSDT,DOTUSDT --input-dir data/recent/ --model models/portfolio.model
# Error: Symbol DOTUSDT not found in training symbols: [BTCUSDT, ETHUSDT, ADAUSDT]

# Wrong data structure
vanga train --symbol BTCUSDT,ETHUSDT --data data/BTCUSDT_1h.csv --output models/portfolio.model
# Error: Multi-symbol requires --data-dir <directory> argument

# Missing data files
vanga train --symbol BTCUSDT,ETHUSDT --data-dir data/incomplete/ --output models/portfolio.model
# Error: Missing data file for symbol: ETHUSDT
```

### Validation Commands
```bash
# Validate symbol compatibility
vanga validate compatibility --model models/portfolio.model --symbols BTCUSDT,ETHUSDT

# Validate data structure
vanga validate data --symbol BTCUSDT,ETHUSDT --data-dir data/multi_asset/

# Validate model performance
vanga validate performance --model models/portfolio.model --test-dir data/test/ --min-accuracy 0.85
```

The unified symbol interface provides a consistent and intuitive experience for both single-symbol and multi-symbol workflows, with automatic configuration adaptation and comprehensive error handling.
