# VANGA Optimizer Benchmarking Scripts

This directory contains scripts to benchmark and compare the performance of all 9 VANGA optimizers on cryptocurrency data.

## 🚀 Quick Start

### Python Script (Recommended)
```bash
# Install dependencies
pip install -r scripts/requirements.txt

# Run full benchmark
python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT

# Quick benchmark (reduced epochs)
python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT --quick

# Test specific optimizers
python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT --optimizers AdamW RMSprop NAdam
```

### Shell Script (Simple)
```bash
# Make executable
chmod +x scripts/benchmark_optimizers.sh

# Run full benchmark
./scripts/benchmark_optimizers.sh --data data/BTCUSDT_1h.csv --symbol BTCUSDT

# Quick benchmark
./scripts/benchmark_optimizers.sh --data data/BTCUSDT_1h.csv --symbol BTCUSDT --quick
```

## 📊 Output Files

Both scripts generate:

1. **JSON Results** (`benchmark_results_SYMBOL_TIMESTAMP.json`)
   - Detailed metrics for each optimizer
   - Training times, convergence data, performance metrics

2. **Markdown Report** (`benchmark_report_SYMBOL_TIMESTAMP.md`)
   - Human-readable summary with rankings
   - Performance comparison tables
   - Recommendations

3. **CSV Results** (`benchmark_results_SYMBOL_TIMESTAMP.csv`) - Shell script only
   - Simple tabular format for spreadsheet analysis

4. **Visualization Charts** (`benchmark_plots_SYMBOL_TIMESTAMP.png`) - Python script only
   - Performance comparison charts
   - Training time vs accuracy plots

5. **Individual Training Logs** (`OPTIMIZER_TIMESTAMP.log`)
   - Detailed training output for each optimizer

## 🎯 Benchmark Metrics

### Performance Metrics
- **Validation Loss**: Primary performance indicator
- **MAE/MSE/RMSE**: Prediction accuracy metrics
- **MAPE**: Mean Absolute Percentage Error

### Training Metrics
- **Training Time**: Total time to complete training
- **Epochs Trained**: Number of epochs completed
- **Epochs to Convergence**: When validation loss stabilized
- **Convergence Stability**: Standard deviation of final epochs

### Status Indicators
- **Converged**: Whether training completed successfully
- **Early Stopped**: Whether early stopping was triggered
- **Error Message**: Details of any failures

## 📈 Interpreting Results

### Performance Ranking
Optimizers are ranked by **best validation loss** (lower is better):

1. **Rank 1**: Best overall performance
2. **Rank 2-3**: Good alternatives
3. **Rank 4+**: Consider only for specific use cases

### Key Insights to Look For

#### 🥇 **Best Overall Performance**
- Lowest validation loss
- Reasonable training time
- Stable convergence

#### ⚡ **Fastest Training**
- Shortest training time
- Fewest epochs to convergence
- Good for rapid iteration

#### 🎯 **Most Stable**
- Low convergence stability score
- Consistent performance across runs
- Good for production use

#### ⚠️ **Problematic Optimizers**
- Failed to converge
- Extremely long training times
- High convergence instability

## 🔧 Customization

### Testing Specific Optimizers
```bash
# Python
python scripts/benchmark_optimizers.py --data data.csv --symbol BTCUSDT --optimizers AdamW RMSprop

# Shell
./scripts/benchmark_optimizers.sh --data data.csv --symbol BTCUSDT --optimizers "AdamW RMSprop"
```

### Quick vs Full Benchmarking

#### Quick Mode (`--quick`)
- Reduced epochs (30 max)
- Shorter early stopping patience
- 10-minute timeout per optimizer
- Good for initial exploration

#### Full Mode (default)
- Full epoch counts from configs
- Standard early stopping
- 1-hour timeout per optimizer
- Comprehensive evaluation

### Custom Output Directory
```bash
python scripts/benchmark_optimizers.py --data data.csv --symbol BTCUSDT --output my_benchmark_results
```

## 📋 Example Workflow

### 1. Initial Exploration
```bash
# Quick benchmark to identify promising optimizers
python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT --quick
```

### 2. Detailed Evaluation
```bash
# Full benchmark on top 3 optimizers from quick run
python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT --optimizers AdamW RMSprop NAdam
```

### 3. Production Selection
```bash
# Use the best performing optimizer configuration
vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --config configs/optimizer_examples/adamw_crypto_optimized.toml
```

## 🚨 Troubleshooting

### Common Issues

#### "Config file not found"
- Ensure you're running from the VANGA root directory
- Check that `configs/optimizer_examples/` directory exists

#### "Command not found: cargo"
- Install Rust and Cargo: https://rustup.rs/
- Ensure VANGA project compiles: `cargo check`

#### Python dependencies missing
```bash
pip install -r scripts/requirements.txt
```

#### Timeout errors
- Increase timeout in script for large datasets
- Use `--quick` mode for initial testing
- Check system resources (memory, CPU)

#### Memory issues
- Reduce batch size in optimizer configs
- Use smaller datasets for benchmarking
- Monitor system memory usage

### Performance Tips

#### For Large Datasets
- Use `--quick` mode first
- Test 2-3 best optimizers in full mode
- Consider data sampling for initial benchmarks

#### For Multiple Symbols
- Run benchmarks separately for each symbol
- Different symbols may prefer different optimizers
- Save results for comparison

#### For Production Use
- Run full benchmarks on representative data
- Test on multiple market conditions (bull/bear/sideways)
- Validate results with out-of-sample data

## 📚 Related Documentation

- **Optimizer Selection Guide**: `doc/optimizer-selection-guide.md`
- **Configuration Examples**: `configs/optimizer_examples/README.md`
- **Training Documentation**: `doc/04-training.md`
- **Troubleshooting**: `doc/14-troubleshooting.md`

## 🎯 Expected Results

### Typical Performance Ranking (Crypto Data)
1. **AdamW**: Best overall performance, handles volatility well
2. **RMSprop**: Good for volatile markets, stable convergence
3. **NAdam**: Fast convergence, good for trending markets
4. **RAdam**: Stable but slower, good for large datasets
5. **Adam**: Reliable general-purpose performance
6. **AdaMax**: Good for extreme market movements
7. **AdaDelta**: Automatic LR adaptation, moderate performance
8. **SGD**: Simple but requires careful tuning
9. **AdaGrad**: Only for short training runs

### Typical Training Times (1000 samples)
- **Quick Mode**: 2-10 minutes per optimizer
- **Full Mode**: 10-60 minutes per optimizer
- **Total Benchmark**: 1-8 hours depending on mode and optimizers

---

**💡 Pro Tip**: Start with a quick benchmark to identify the top 3 optimizers, then run a full benchmark on just those for detailed comparison.
