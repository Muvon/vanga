# VANGA LSTM Usage Guide

## Training Continuation & Custom Features

### Training Command Behavior

#### Default Training (Recommended)
```bash
# Continue training if model exists, create new if not
vanga train --symbol BTCUSDT --data btc_data.csv
```

#### Fresh Training
```bash
# Always start fresh training (ignore existing model)
vanga train --symbol BTCUSDT --data btc_data.csv --fresh
```

#### Force Continue Training
```bash
# Force continuation (error if no existing model)
vanga train --symbol BTCUSDT --data btc_data.csv --continue-training
```

### Training Workflow Examples

#### 1. First Time Training
```bash
# No existing model - creates new model
vanga train --symbol BTCUSDT --data btc_data.csv
# Result: Creates ./models/BTCUSDT_model.bin
```

#### 2. Continuing Training with More Data
```bash
# Existing model found - continues training
vanga train --symbol BTCUSDT --data btc_new_data.csv
# Result: Updates existing ./models/BTCUSDT_model.bin
```

#### 3. Retraining with Different Features
```bash
# Model exists but data structure changed - use fresh
vanga train --symbol BTCUSDT --data btc_enhanced_data.csv --fresh
# Result: Creates new model with new feature structure
```

## Custom Features Usage

### CSV Data Structure

#### Required Columns (OHLCV)
```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56
2024-01-01T01:00:00Z,42300.0,42800.0,42100.0,42600.0,1567.89
```

#### With Custom Features
```csv
timestamp,open,high,low,close,volume,social_sentiment,funding_rate,whale_activity,on_chain_volume
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,0.75,-0.01,1250000,89.5
2024-01-01T01:00:00Z,42300.0,42800.0,42100.0,42600.0,1567.89,0.82,0.02,1180000,92.1
```

### Custom Features Configuration

#### Auto-Include All Custom Columns (Recommended)
```toml
# configs/my_custom_features.toml
[custom_features]
auto_include_all = true  # Automatically include all non-OHLCV columns

# Optional: Exclude specific columns
exclude_features = [
    "irrelevant_column",
    "noisy_indicator"
]

# Optional: Apply transformations
[custom_features.transformations]
social_sentiment = "ZScore"      # Normalize to z-score
funding_rate = "PercentChange"   # Convert to percentage change
whale_activity = "Log"           # Apply log transformation
```

#### Selective Custom Features
```toml
# configs/selective_features.toml
[custom_features]
auto_include_all = false  # Only include specified features

# Explicitly include specific features
include_features = [
    "social_sentiment",
    "funding_rate",
    "whale_activity"
]

[custom_features.transformations]
social_sentiment = "ZScore"
funding_rate = "PercentChange"
```

### Training with Custom Features

#### Step 1: Prepare Your Data
```csv
# btc_enhanced.csv
timestamp,open,high,low,close,volume,social_sentiment,funding_rate,whale_activity
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,0.75,-0.01,1250000
2024-01-01T01:00:00Z,42300.0,42800.0,42100.0,42600.0,1567.89,0.82,0.02,1180000
# ... more data
```

#### Step 2: Configure Features
```toml
# configs/btc_custom.toml
[technical_indicators]
enabled = true
# ... standard technical indicators

[custom_features]
auto_include_all = true

[custom_features.transformations]
social_sentiment = "ZScore"
funding_rate = "PercentChange"
whale_activity = "Log"
```

#### Step 3: Train Model
```bash
vanga train --symbol BTCUSDT --data btc_enhanced.csv --features-config configs/btc_custom.toml
```

### Advanced Custom Features Examples

#### 1. Cryptocurrency-Specific Features
```csv
timestamp,open,high,low,close,volume,funding_rate,open_interest,liquidations,fear_greed_index
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,-0.01,1500000000,25000000,25
```

```toml
[custom_features]
auto_include_all = true

[custom_features.transformations]
funding_rate = "PercentChange"
open_interest = "Log"
liquidations = "ZScore"
fear_greed_index = "MinMaxScale"
```

#### 2. On-Chain Metrics
```csv
timestamp,open,high,low,close,volume,active_addresses,transaction_count,nvt_ratio,mvrv_ratio
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,950000,280000,45.2,1.8
```

```toml
[custom_features]
auto_include_all = true

[custom_features.transformations]
active_addresses = "Log"
transaction_count = "Log"
nvt_ratio = "ZScore"
mvrv_ratio = "ZScore"
```

#### 3. Social & Sentiment Data
```csv
timestamp,open,high,low,close,volume,twitter_sentiment,reddit_mentions,google_trends,news_sentiment
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,0.75,1250,67,0.82
```

```toml
[custom_features]
auto_include_all = true

[custom_features.transformations]
twitter_sentiment = "ZScore"
reddit_mentions = "Log"
google_trends = "MinMaxScale"
news_sentiment = "ZScore"
```

### Feature Transformation Types

#### Available Transformations
- `"ZScore"` - Normalize to z-score (mean=0, std=1)
- `"MinMaxScale"` - Scale to range [0, 1]
- `"PercentChange"` - Convert to percentage change
- `"Log"` - Apply logarithmic transformation
- `"RobustScale"` - Robust scaling using median and IQR

#### When to Use Each Transformation
- **ZScore**: For normally distributed features (sentiment scores, ratios)
- **MinMaxScale**: For bounded features (percentages, indices)
- **PercentChange**: For rates and changes (funding rates, growth rates)
- **Log**: For skewed features (volumes, counts, addresses)
- **RobustScale**: For features with outliers (liquidations, whale activity)

### Common Use Cases

#### 1. Adding Market Sentiment
```bash
# Data with sentiment columns
vanga train --symbol BTCUSDT --data btc_with_sentiment.csv

# The system automatically detects and includes sentiment columns
# No configuration needed if using auto_include_all = true
```

#### 2. Incorporating On-Chain Data
```bash
# Data with on-chain metrics
vanga train --symbol BTCUSDT --data btc_onchain.csv --features-config configs/onchain_features.toml
```

#### 3. Multi-Exchange Data
```csv
timestamp,open,high,low,close,volume,binance_volume,coinbase_volume,kraken_volume,volume_ratio
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56,800.0,300.0,134.56,0.65
```

### Troubleshooting Custom Features

#### Feature Compatibility Issues
```bash
# Error: Model input size mismatch
# Solution: Use --fresh to retrain with new feature structure
vanga train --symbol BTCUSDT --data new_features.csv --fresh
```

#### Missing Custom Features
```bash
# Check if features are being included
vanga train --symbol BTCUSDT --data data.csv --verbose

# Look for log messages about feature engineering
# Should show: "Custom features detected: [feature1, feature2, ...]"
```

#### Feature Validation
```toml
# Ensure features are properly configured
[custom_features]
auto_include_all = true

# Verify exclude list doesn't remove wanted features
exclude_features = []  # Empty list to include all
```

### Best Practices

#### 1. Data Quality
- Ensure custom features have minimal missing values
- Use consistent data frequency (same as OHLCV)
- Validate feature ranges and distributions

#### 2. Feature Engineering
- Start with `auto_include_all = true` for simplicity
- Use appropriate transformations for each feature type
- Monitor feature importance after training

#### 3. Model Management
- Use descriptive feature configuration file names
- Keep track of which features were used for each model
- Use `--fresh` when adding new features to existing models

#### 4. Performance Optimization
- Limit total features to avoid overfitting (max_features = 100)
- Remove highly correlated features (correlation_threshold = 0.95)
- Filter low-importance features (importance_threshold = 0.001)

### Complete Workflow Example

```bash
# 1. Prepare data with custom features
# btc_complete.csv contains OHLCV + sentiment + on-chain + funding data

# 2. Create custom configuration
# configs/btc_complete.toml with appropriate transformations

# 3. Train model
vanga train --symbol BTCUSDT --data btc_complete.csv --features-config configs/btc_complete.toml

# 4. Make predictions
vanga predict --symbol BTCUSDT --input recent_data.csv

# 5. Continue training with new data
vanga train --symbol BTCUSDT --data additional_data.csv --features-config configs/btc_complete.toml
```

This workflow ensures consistent feature engineering across training and prediction phases.
