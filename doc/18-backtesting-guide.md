# Backtesting Guide

Comprehensive backtesting functionality for VANGA LSTM models with real data pipeline integration.

## Overview

VANGA's backtesting system provides professional-grade performance evaluation by training models on historical data and testing on future data with proper chronological splitting to prevent data leakage.

## Features

### Core Capabilities
- **Real Data Pipeline**: Uses actual CSV data loading and processing
- **Chronological Splitting**: Prevents data leakage with time-based train/test splits
- **Multi-Target Evaluation**: Tests price levels, direction, and volatility predictions
- **Comprehensive Metrics**: Regression and classification metrics from real predictions
- **Batch Processing**: Support for multiple symbols simultaneously

### Integration Architecture
- Reuses existing `train_model()` and `predict_multi_target()` functions
- Leverages `DataLoader` for CSV processing
- Uses `TargetGenerator` for label creation
- Implements temporary file management for train/test data
- Automatic cleanup of resources

## Usage

### Single Symbol Backtesting
```bash
# Basic backtesting with 80% train, 20% test split
vanga models evaluate --symbol BTCUSDT --test-data data.csv --backtest --train-split 0.8

# Custom train/test ratio
vanga models evaluate --symbol BTCUSDT --test-data data.csv --backtest --train-split 0.7

# With specific output format
vanga models evaluate --symbol BTCUSDT --test-data data.csv --backtest --output results.json
```

### Batch Backtesting
```bash
# Multiple symbols from directory
vanga models evaluate --symbols BTCUSDT,ETHUSDT --test-data data/ --backtest --batch

# All CSV files in directory
vanga models evaluate --test-data data/ --backtest --batch --train-split 0.8
```

## Data Requirements

### Minimum Data Format
```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56
2024-01-01T01:00:00Z,42300.0,42800.0,42100.0,42600.0,1456.78
...
```

### Data Quality Requirements
- **Minimum Samples**: 500+ rows recommended for reliable training
- **Required Columns**: timestamp, open, high, low, close, volume
- **Time Ordering**: Data must be chronologically ordered
- **No Missing Values**: Critical OHLCV data should be complete

## Backtesting Workflow

### 1. Data Loading and Validation
```rust
// Load and validate CSV data
let data_loader = DataLoader::new();
let full_df = data_loader.load_csv(&self.config.data_path).await?;

// Validate minimum data requirements
if full_df.height() < 100 {
    return Err(VangaError::DataError(
        format!("Insufficient data for backtesting (minimum 100 samples required, got {})", full_df.height()),
    ));
}
```

### 2. Chronological Data Splitting
```rust
// Split data chronologically to prevent data leakage
let (train_df, test_df) = data_loader.split_chronological(&full_df, self.config.train_split)?;

// Extract time periods for reporting
let train_period = self.extract_time_period(&train_df)?;
let test_period = self.extract_time_period(&test_df)?;
```

### 3. Model Training
```rust
// Create training configuration
let training_config = self.create_training_config(&train_path)?;

// Train model on training data
let trained_model = train_model(training_config).await?;
```

### 4. Prediction Generation
```rust
// Create prediction configuration
let prediction_config = self.create_prediction_config(&test_path)?;

// Generate predictions on test data
let predictions = predict_multi_target(prediction_config, &trained_model).await?;
```

### 5. Metrics Calculation
```rust
// Generate actual targets for test data
let target_generator = TargetGenerator::with_defaults();
let actual_targets = target_generator.generate_all_targets(&test_df).await?;

// Calculate comprehensive metrics
let metrics = self.calculate_metrics(&predictions, &actual_targets).await?;
let directional_accuracy = self.calculate_directional_accuracy(&predictions, &actual_targets).await?;
```

## Results and Reporting

### Console Output
```
🔄 Starting backtesting for symbol: BTCUSDT
📊 Loaded 1000 samples for backtesting
📈 Train period: 2024-01-01T00:00:00Z to 2024-01-27T04:00:00Z
📉 Test period: 2024-01-27T04:00:00Z to 2024-02-05T20:00:00Z
🚀 Training model on 800 samples
🔮 Generating predictions on 200 samples
🎯 Generating targets for test data
✅ Backtesting completed for BTCUSDT

📊 Backtesting Results
Symbol: BTCUSDT
Model: MultiTargetLSTM
Train Period: 2024-01-01T00:00:00Z to 2024-01-27T04:00:00Z (800 samples)
Test Period: 2024-01-27T04:00:00Z to 2024-02-05T20:00:00Z (200 samples)

Regression Metrics:
├── RMSE: 0.032
├── MAE: 0.025
├── R²: 0.75
└── MAPE: 2.5%

Classification Metrics:
├── Directional Accuracy: 68%
└── Prediction Count: 200
```

### Export Formats
- **JSON**: Structured data for programmatic analysis
- **CSV**: Tabular format for spreadsheet analysis
- **Markdown**: Human-readable reports

## Configuration

### Training Configuration
```rust
// Optimized configuration for backtesting
TrainingConfig {
    symbol: symbol.to_string(),
    data_path: train_path.to_path_buf(),
    model: ModelConfig::default(),
    horizons: vec!["1h".to_string()],
    fresh_training: true,
    continue_training: false,
    training_params: TrainingParams {
        epochs: EpochConfig::Fixed(5),
        batch_size: BatchSizeConfig::Fixed(16),
        learning_rate: LearningRateConfig::Fixed(0.001),
        validation_split: 0.2,
        test_split: 0.0,
        early_stopping_patience: 3,
        gradient_clip: Some(1.0),
    },
    // ... other defaults
}
```

### Prediction Configuration
```rust
PredictionConfig {
    symbol: symbol.to_string(),
    input_path: test_path.to_path_buf(),
    horizon: Some("1h".to_string()),
    // ... other defaults
}
```

## Error Handling

### Common Issues and Solutions

#### Insufficient Data
```
Error: Insufficient data for backtesting (minimum 100 samples required, got 50)
```
**Solution**: Provide more historical data (500+ samples recommended)

#### Target Generation Failure
```
Error: Target generation failed: Insufficient data for price level target generation
```
**Solution**: Ensure data has enough samples for sequence generation (1000+ recommended)

#### Model Training Issues
```
Error: Model error: Candle error: shape mismatch in div
```
**Solution**: This is a known model architecture issue, not a backtesting problem

## Implementation Details

### Key Files
- `src/api/backtester.rs` - Main backtesting implementation
- `src/utils/backtest_reporter.rs` - Results formatting and reporting
- `tests/backtesting_integration.rs` - Integration tests
- `configs/backtest.toml` - Configuration template

### Key Functions
- `run_backtest()` - High-level backtesting function
- `run_batch_backtest()` - Batch processing for multiple symbols
- `Backtester::run_backtest()` - Core backtesting workflow
- `calculate_metrics()` - Performance metrics calculation
- `calculate_directional_accuracy()` - Direction prediction accuracy

### Integration Points
- Uses existing `api::train_model()` for model training
- Uses existing `api::predict_multi_target()` for predictions
- Uses existing `DataLoader::split_chronological()` for data splitting
- Uses existing `TargetGenerator` for label creation

## Best Practices

### Data Preparation
1. Ensure chronological ordering of data
2. Validate data quality before backtesting
3. Use sufficient historical data (1000+ samples)
4. Include all required OHLCV columns

### Configuration
1. Use appropriate train/test split ratios (70-80% train)
2. Configure short training epochs for testing
3. Enable early stopping for efficiency
4. Use default model configurations for stability

### Validation
1. Verify time periods in results
2. Check sample counts match expectations
3. Validate metrics are reasonable for the asset
4. Compare results across different time periods

## Future Enhancements

### Planned Features
- Walk-forward analysis
- Rolling window backtesting
- Model comparison across architectures
- Advanced performance metrics (Sharpe ratio, drawdown)
- Real-time backtesting with live data feeds

### Extension Points
- Custom metrics calculation
- Alternative data splitting strategies
- Model ensemble backtesting
- Cross-asset correlation analysis
