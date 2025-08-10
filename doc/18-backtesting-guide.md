# Backtesting Guide

Comprehensive backtesting system for VANGA LSTM models with professional-grade time-series validation and performance evaluation.

## 🎯 **Overview**

VANGA's backtesting system provides rigorous model evaluation by training on historical data and testing on future data with proper chronological splitting to prevent data leakage. The system integrates seamlessly with the existing training and prediction pipeline.

## 🏗 **Architecture**

### **Core Components**
- **Chronological Data Splitting**: Time-based train/test splits prevent data leakage
- **Real Pipeline Integration**: Uses actual `train_model()` and prediction APIs
- **Multi-Target Evaluation**: Tests all 5 targets (price levels, direction, volatility, sentiment, volume)
- **Comprehensive Metrics**: Regression, classification, and trading performance metrics
- **Walk-Forward Analysis**: Advanced time-series validation with rolling windows

### **Integration Points**
```rust
// Implemented in src/api/backtester.rs
pub struct Backtester {
    config: BacktestConfig,
}

#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub symbol: String,
    pub train_split: f64,           // Training data percentage (e.g., 0.8 for 80%)
    pub data_path: std::path::PathBuf,
}

#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub symbol: String,
    pub model_type: String,
    pub train_period: (String, String),    // (start_date, end_date)
    pub test_period: (String, String),     // (start_date, end_date)
    pub train_samples: usize,
    pub test_samples: usize,
    pub regression_metrics: RegressionMetrics,
    pub directional_accuracy: f64,
    pub prediction_count: usize,
}
```

## 🚀 **Usage**

### **Basic Backtesting**
```bash
# Simple backtest with 80% train, 20% test split
cargo run -- backtest --symbol BTCUSDT --data data/BTCUSDT_1h.csv

# Custom train/test ratio
cargo run -- backtest \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --train-split 0.7

# With specific output file
cargo run -- backtest \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --output backtest_results.json
```

### **Advanced Backtesting Options**
```bash
# Walk-forward analysis
cargo run -- backtest \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --walk-forward \
    --train-window 1000 \
    --test-window 100 \
    --step-size 50

# Multiple symbols batch processing
cargo run -- backtest \
    --symbols BTCUSDT,ETHUSDT,ADAUSDT \
    --data-dir data/ \
    --batch \
    --train-split 0.8

# With custom configuration
cargo run -- backtest \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --config configs/backtest.toml
```

## 🔄 **Backtesting Workflow**

### **Step-by-Step Process**
```rust
impl Backtester {
    pub async fn run_backtest(&self) -> Result<BacktestResults> {
        // 1. Load and validate data
        let data_loader = DataLoader::new();
        let full_df = data_loader.load_csv(&self.config.data_path).await?;

        // 2. Split data chronologically (prevents data leakage)
        let (train_df, test_df) = data_loader.split_chronological(&full_df, self.config.train_split)?;

        // 3. Train model on training data
        let training_config = self.create_training_config(&train_path)?;
        let trained_model = train_model(training_config).await?;

        // 4. Generate predictions on test data
        let prediction_config = self.create_prediction_config(&test_path)?;
        let predictions = predictor.predict(ModelWrapper::MultiTarget(&trained_model)).await?;

        // 5. Generate actual targets for test data
        let target_generator = TargetGenerator::with_defaults();
        let actual_targets = target_generator.generate_all_targets(&test_df, None, &sequence_indices, sequence_length).await?;

        // 6. Calculate comprehensive metrics
        let metrics = self.calculate_backtest_metrics(&predictions, &actual_targets)?;

        // 7. Return structured results
        Ok(BacktestResults { /* ... */ })
    }
}
```

### **Chronological Data Splitting**
```rust
// Prevents data leakage by ensuring temporal order
pub fn split_chronological(
    &self,
    df: &DataFrame,
    train_split: f64
) -> Result<(DataFrame, DataFrame)> {
    let total_rows = df.height();
    let train_rows = (total_rows as f64 * train_split) as usize;

    // Split at specific row index (not random)
    let train_df = df.slice(0, train_rows as i64);
    let test_df = df.slice(train_rows as i64, (total_rows - train_rows) as i64);

    Ok((train_df, test_df))
}
```

## 📊 **Walk-Forward Analysis**

### **Advanced Time-Series Validation**
```rust
// Implemented for robust model validation
pub struct WalkForwardBacktester {
    config: WalkForwardConfig,
}

#[derive(Debug, Clone)]
pub struct WalkForwardConfig {
    pub symbol: String,
    pub data_path: std::path::PathBuf,
    pub train_window: usize,        // Training window size (e.g., 1000 samples)
    pub test_window: usize,         // Test window size (e.g., 100 samples)
    pub step_size: usize,           // Step size for rolling window (e.g., 50 samples)
}

impl WalkForwardBacktester {
    pub async fn run_walk_forward_analysis(&self) -> Result<Vec<BacktestResults>> {
        let mut results = Vec::new();
        let data_loader = DataLoader::new();
        let df = data_loader.load_csv(&self.config.data_path).await?;

        let total_samples = df.height();
        let mut start_idx = 0;

        // Rolling window analysis
        while start_idx + self.config.train_window + self.config.test_window <= total_samples {
            // Extract training and test windows
            let train_end = start_idx + self.config.train_window;
            let test_end = train_end + self.config.test_window;

            let train_df = df.slice(start_idx as i64, self.config.train_window);
            let test_df = df.slice(train_end as i64, self.config.test_window);

            // Run backtest on this window
            let window_config = BacktestConfig {
                symbol: self.config.symbol.clone(),
                train_split: 1.0, // Use entire window for training
                data_path: self.create_temp_file(&train_df)?,
            };

            let backtester = Backtester::new(window_config);
            let result = backtester.run_backtest().await?;
            results.push(result);

            // Move to next window
            start_idx += self.config.step_size;
        }

        Ok(results)
    }
}
```

### **Walk-Forward Usage**
```bash
# Run walk-forward analysis
cargo run -- backtest \
    --symbol BTCUSDT \
    --data data/BTCUSDT_1h.csv \
    --walk-forward \
    --train-window 2000 \
    --test-window 200 \
    --step-size 100

# Results show performance across multiple time periods
# Window 1: 2024-01-01 to 2024-03-01 (train) → 2024-03-01 to 2024-03-15 (test)
# Window 2: 2024-01-15 to 2024-03-15 (train) → 2024-03-15 to 2024-03-30 (test)
# ...
```

## 📈 **Evaluation Metrics**

### **Comprehensive Performance Assessment**
```rust
// Multi-target evaluation metrics
pub struct BacktestMetrics {
    // Regression metrics
    pub rmse: f64,
    pub mae: f64,
    pub r_squared: f64,
    pub mape: f64,

    // Classification metrics (per target)
    pub price_levels_accuracy: f64,
    pub direction_accuracy: f64,
    pub volatility_accuracy: f64,
    pub sentiment_accuracy: f64,
    pub volume_accuracy: f64,

    // Trading metrics
    pub directional_accuracy: f64,
    pub sharpe_ratio: Option<f64>,
    pub max_drawdown: Option<f64>,

    // Overall performance
    pub overall_accuracy: f64,
    pub prediction_confidence: f64,
}
```

### **Metric Calculation**
```rust
// Calculate comprehensive backtest metrics
fn calculate_backtest_metrics(
    &self,
    predictions: &[PredictionResult],
    actual_targets: &PreparedTargets,
) -> Result<BacktestMetrics> {
    // Extract predictions and targets for each target type
    let price_level_predictions = self.extract_target_predictions(predictions, "price_levels")?;
    let direction_predictions = self.extract_target_predictions(predictions, "direction")?;
    let volatility_predictions = self.extract_target_predictions(predictions, "volatility")?;
    let sentiment_predictions = self.extract_target_predictions(predictions, "sentiment")?;
    let volume_predictions = self.extract_target_predictions(predictions, "volume")?;

    // Calculate accuracy for each target
    let price_levels_accuracy = calculate_classification_accuracy(&price_level_predictions, &actual_targets.price_levels)?;
    let direction_accuracy = calculate_classification_accuracy(&direction_predictions, &actual_targets.direction)?;
    let volatility_accuracy = calculate_classification_accuracy(&volatility_predictions, &actual_targets.volatility)?;
    let sentiment_accuracy = calculate_classification_accuracy(&sentiment_predictions, &actual_targets.sentiment)?;
    let volume_accuracy = calculate_classification_accuracy(&volume_predictions, &actual_targets.volume)?;

    // Calculate overall metrics
    let overall_accuracy = (
        price_levels_accuracy * 0.3 +      // Price levels most important
        direction_accuracy * 0.25 +        // Direction second most important
        volatility_accuracy * 0.2 +        // Volatility third
        sentiment_accuracy * 0.15 +        // Sentiment fourth
        volume_accuracy * 0.1              // Volume least important
    );

    Ok(BacktestMetrics {
        price_levels_accuracy,
        direction_accuracy,
        volatility_accuracy,
        sentiment_accuracy,
        volume_accuracy,
        overall_accuracy,
        // ... other metrics
    })
}
```

## ⚙️ **Configuration**

### **Backtesting Configuration (configs/backtest.toml)**
```toml
[backtest]
# Default train/test split ratio
default_train_split = 0.8

# Minimum samples required for backtesting
min_samples = 1000

# Output configuration
output_dir = "backtest_results"
save_predictions = true
save_detailed_metrics = true

# Report formats to generate
report_formats = ["console", "json", "csv"]

[evaluation_metrics]
# Regression metrics
calculate_rmse = true
calculate_mae = true
calculate_r_squared = true
calculate_mape = true

# Trading metrics
calculate_directional_accuracy = true
calculate_sharpe_ratio = false  # Requires return data
calculate_max_drawdown = false  # Requires return data

# Classification metrics (for all 5 targets)
calculate_classification_metrics = true

[time_series]
# Ensure chronological splitting (no data leakage)
chronological_split = true

# Minimum test period (as fraction of total data)
min_test_ratio = 0.1

# Maximum test period (as fraction of total data)
max_test_ratio = 0.4

[walk_forward]
# Walk-forward analysis settings
enabled = false
default_train_window = 1000
default_test_window = 100
default_step_size = 50

[batch_processing]
# Parallel processing for multiple symbols
parallel_execution = true

# Maximum concurrent backtests
max_concurrent = 4
```

## 📋 **Data Requirements**

### **Minimum Data Requirements**
```bash
# Check data size before backtesting
wc -l data/BTCUSDT_1h.csv
# Minimum: 1000 samples (recommended: 5000+ for robust results)

# Data format validation
head -5 data/BTCUSDT_1h.csv
# timestamp,open,high,low,close,volume
# 2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56
```

### **Data Quality Checks**
```rust
// Automatic data validation during backtesting
fn validate_backtest_data(&self, df: &DataFrame) -> Result<()> {
    // Check minimum samples
    if df.height() < 100 {
        return Err(VangaError::DataError(format!(
            "Insufficient data for backtesting (minimum 100 samples required, got {})",
            df.height()
        )));
    }

    // Check for required columns
    let required_columns = ["timestamp", "open", "high", "low", "close", "volume"];
    for col in required_columns {
        if !df.get_column_names().contains(&col) {
            return Err(VangaError::DataError(format!(
                "Missing required column: {}", col
            )));
        }
    }

    // Check for data gaps
    self.validate_timestamp_continuity(df)?;

    Ok(())
}
```

## 📊 **Results Interpretation**

### **Console Output Example**
```
🔄 Starting backtesting for symbol: BTCUSDT
📊 Loaded 8760 samples for backtesting
📈 Train period: 2024-01-01T00:00:00Z to 2024-09-01T00:00:00Z
📉 Test period: 2024-09-01T00:00:00Z to 2024-11-01T00:00:00Z
🚀 Training model on 7008 samples
🔮 Generating predictions on 1752 samples
🎯 Generating targets for test data

📊 Backtesting Results for BTCUSDT:
═══════════════════════════════════════════
Model Type: MultiTargetLSTM
Training Period: 2024-01-01T00:00:00Z to 2024-09-01T00:00:00Z
Test Period: 2024-09-01T00:00:00Z to 2024-11-01T00:00:00Z
Training Samples: 7008
Test Samples: 1752

📈 Multi-Target Performance:
  Overall Accuracy: 69.4%
  Price Levels Accuracy: 68.2%
  Direction Accuracy: 72.1%
  Volatility Accuracy: 74.8%
  Sentiment Accuracy: 63.5%
  Volume Accuracy: 68.9%

📊 Regression Metrics:
  RMSE: 0.0234
  MAE: 0.0187
  R²: 0.7456
  MAPE: 3.21%

🎯 Trading Metrics:
  Directional Accuracy: 72.1%
  Prediction Count: 1752
  Average Confidence: 0.678

✅ Backtest completed successfully!
```

### **JSON Output Structure**
```json
{
  "symbol": "BTCUSDT",
  "model_type": "MultiTargetLSTM",
  "train_period": ["2024-01-01T00:00:00Z", "2024-09-01T00:00:00Z"],
  "test_period": ["2024-09-01T00:00:00Z", "2024-11-01T00:00:00Z"],
  "train_samples": 7008,
  "test_samples": 1752,
  "metrics": {
    "overall_accuracy": 0.694,
    "target_accuracies": {
      "price_levels": 0.682,
      "direction": 0.721,
      "volatility": 0.748,
      "sentiment": 0.635,
      "volume": 0.689
    },
    "regression_metrics": {
      "rmse": 0.0234,
      "mae": 0.0187,
      "r_squared": 0.7456,
      "mape": 3.21
    },
    "trading_metrics": {
      "directional_accuracy": 0.721,
      "prediction_count": 1752,
      "average_confidence": 0.678
    }
  },
  "timestamp": "2024-08-10T16:17:40Z"
}
```

## 🔧 **API Usage**

### **Programmatic Backtesting**
```rust
use vanga::api::backtester::{Backtester, BacktestConfig};

// Configure backtesting
let config = BacktestConfig {
    symbol: "BTCUSDT".to_string(),
    train_split: 0.8,
    data_path: "data/BTCUSDT_1h.csv".into(),
};

// Run backtest
let backtester = Backtester::new(config);
let results = backtester.run_backtest().await?;

// Process results
println!("Backtest Results for {}:", results.symbol);
println!("  Overall Accuracy: {:.3}%", results.overall_accuracy * 100.0);
println!("  Directional Accuracy: {:.3}%", results.directional_accuracy * 100.0);
println!("  RMSE: {:.4}", results.regression_metrics.rmse);
```

### **Walk-Forward API**
```rust
use vanga::api::backtester::{WalkForwardBacktester, WalkForwardConfig};

// Configure walk-forward analysis
let config = WalkForwardConfig {
    symbol: "BTCUSDT".to_string(),
    data_path: "data/BTCUSDT_1h.csv".into(),
    train_window: 2000,
    test_window: 200,
    step_size: 100,
};

// Run walk-forward analysis
let backtester = WalkForwardBacktester::new(config);
let results = backtester.run_walk_forward_analysis().await?;

// Analyze results across windows
for (i, result) in results.iter().enumerate() {
    println!("Window {}: Accuracy = {:.3}%", i + 1, result.overall_accuracy * 100.0);
}

let avg_accuracy = results.iter().map(|r| r.overall_accuracy).sum::<f64>() / results.len() as f64;
println!("Average Accuracy Across Windows: {:.3}%", avg_accuracy * 100.0);
```

## 🚨 **Best Practices**

### **Data Preparation**
- **Minimum 1000 samples**: Ensure sufficient data for meaningful results
- **Quality validation**: Check for missing values, outliers, and data gaps
- **Chronological order**: Maintain temporal sequence in your data
- **Consistent timeframes**: Use consistent intervals (1h, 4h, 1d)

### **Backtesting Setup**
- **Appropriate splits**: Use 70-80% for training, 20-30% for testing
- **No data leakage**: Always use chronological splitting
- **Multiple horizons**: Test different prediction horizons
- **Walk-forward validation**: Use for robust performance assessment

### **Results Interpretation**
- **Overall accuracy > 60%**: Good performance for 5-class classification
- **Directional accuracy > 55%**: Better than random for trading
- **Consistent performance**: Look for stable results across time periods
- **Target-specific analysis**: Some targets may perform better than others

### **Common Pitfalls**
- **Insufficient data**: Less than 1000 samples leads to unreliable results
- **Data leakage**: Random splitting instead of chronological
- **Overfitting**: Perfect training performance but poor test performance
- **Ignoring confidence**: Low-confidence predictions should be filtered

This comprehensive backtesting guide provides everything needed to evaluate VANGA LSTM models with professional-grade time-series validation.
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
