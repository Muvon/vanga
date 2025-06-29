# VANGA LSTM Development Instructions

! THE PROJECT in ACTIVE DEVELOPMENT - DO NOT keep legacy or fallbacks to old code. KEEP CLEAN

## Core Principles

### Strict Configuration Management
- **TOML-First**: All configuration must be explicitly defined in `configs/` directory
- **Template-Based**: Use configuration templates for different scenarios (training, prediction, features)
- **Symbol-Specific**: Each trading pair gets its own specialized configuration and model
- **Auto-Optimization**: Prefer automatic parameter tuning over manual configuration

### Code Reuse & Architecture

#### LSTM Training Core Pattern
```rust
// Always use this pattern for model training
let trainer = Trainer::new(config);
let model = trainer.train_model().await?;
model.save(&model_path)?;
```

#### Feature Engineering Pipeline
```rust
// Standard feature engineering workflow
let features = FeatureEngineer::new(config.features)
    .add_technical_indicators()
    .add_market_microstructure()
    .add_custom_features()
    .process(df).await?;
```

#### Prediction Pattern
```rust
// Consistent prediction interface
let predictor = Predictor::new(config);
let predictions = predictor.predict(&model).await?;
let formatted = PredictionFormatter::new().format(predictions)?;
```

### LSTM Performance & Model Guidelines

#### Intelligent Model Architecture
VANGA uses intelligent LSTM architecture optimization that automatically tunes model parameters based on data characteristics and crypto market patterns. **No manual tuning required** - all optimizations are automatic.

#### Key Performance Features
- **Smart Architecture Selection**: Automatically chooses between multi-LSTM, stacked, or bidirectional based on data
- **Optimal Sequence Length**: Calculates ideal lookback period for each trading pair
- **Growth-Aware Training**: Adjusts model complexity as training data grows
- **Crypto-Specific Loss**: Uses specialized loss functions for cryptocurrency forecasting

#### Best Practices for Model Usage

```rust
// ✅ GOOD: Use the optimized training pipeline
let model = train_model(config).await?;

// ✅ GOOD: Leverage automatic feature engineering
let features = process_features(df, &config.features).await?;

// ✅ GOOD: Use multi-target prediction
let predictions = predict_multi_target(config, &model).await?;

// ❌ AVOID: Manual hyperparameter tuning (optimizer handles this)
// config.hidden_units = 128; // Don't hardcode

// ❌ AVOID: Fixed sequence lengths (optimizer calculates optimal values)
// config.sequence_length = 60; // Don't hardcode
```

#### Performance Characteristics
- **Small datasets (< 1K rows)**: Simple LSTM with basic features
- **Medium datasets (1K-10K rows)**: Multi-layer LSTM with technical indicators
- **Large datasets (> 10K rows)**: Advanced ensemble with all features
- **Training**: Automatic early stopping with validation monitoring

## Project Structure

### Core Modules
- `src/api/` - High-level training and prediction APIs
- `src/model/` - LSTM model implementations and architectures
- `src/features/` - Feature engineering and technical indicators
- `src/data/` - Data loading, preprocessing, and sequence generation
- `src/targets/` - Multi-target prediction system (price levels, direction, volatility)
- `src/config/` - Configuration management and validation
- `src/utils/` - Error handling, logging, and utilities

### CLI Command Structure
- **Training Mode**: `vanga train --symbol BTCUSDT --data path.csv`
- **Prediction Mode**: `vanga predict --symbol BTCUSDT --input recent.csv`
- **Model Management**: `vanga models list|evaluate|export`
- **Batch Processing**: `vanga train --batch --data-dir ./data/`

### Key Files
- `configs/*.toml` - Configuration templates for different scenarios
- `src/main.rs` - CLI entry point and command routing
- `src/api/trainer.rs` - Training pipeline orchestration
- `src/api/predictor.rs` - Prediction pipeline orchestration
- `src/model/lstm_simple.rs` - Core LSTM implementation

## Development Patterns

!!! NEVER SILENCE WARNINGS with undefined variables like _var INSTEAD Find the REAL reaon why worning appears and FIX it
!!! NEVER hiding unused functions/methods with #[allow(dead_code)]

### Adding New Technical Indicators
1. Add indicator logic to `src/features/technical.rs`
2. Update `TechnicalConfig` struct in `src/config/features.rs`
3. Add configuration to `configs/crypto_features.toml`
4. Update feature processing pipeline

### Adding New Prediction Targets
1. Create target module in `src/targets/{target}.rs`
2. Implement `TargetGenerator` trait
3. Add to `MultiTargetGenerator` in `src/targets/mod.rs`
4. Update target configuration schema

### Adding New Model Architectures
1. Implement model in `src/model/{architecture}.rs`
2. Add to model factory in `src/model/mod.rs`
3. Update `ModelConfig` enum
4. Add architecture-specific configuration

### Configuration Management
1. Update struct in relevant `src/config/*.rs`
2. Add defaults in `Default` impl
3. **MANDATORY**: Update corresponding `configs/*.toml` template
4. Add validation logic if needed

### Data Pipeline Integration
1. **CSV Loading**: Use `DataLoader` with automatic schema detection
2. **Feature Engineering**: Apply `FeatureEngineer` pipeline
3. **Sequence Generation**: Use `SequenceGenerator` for LSTM input
4. **Target Creation**: Apply `MultiTargetGenerator` for labels

## Performance Guidelines

### Training Optimization
- **Batch Size**: Auto-optimized based on available memory and data size
- **Sequence Length**: Automatically calculated per trading pair
- **Feature Selection**: Automatic correlation analysis and importance scoring
- **Early Stopping**: Validation-based with patience parameter

### Memory Management
- **Lazy Loading**: Progressive data loading during training
- **Feature Caching**: Cache computed technical indicators
- **Model Checkpointing**: Regular model saves during training
- **Sequence Batching**: Efficient batch processing for large datasets

### Model Efficiency
- **Symbol-Specific Models**: One model per trading pair for optimal performance
- **Incremental Training**: Support for continuing training with new data
- **Model Compression**: Automatic pruning of less important weights
- **Prediction Caching**: Cache recent predictions for repeated queries

## Error Handling & Logging

### Error Handling Pattern
```rust
// Always use the Result type alias
use crate::utils::error::Result;

// Return descriptive errors with context
Err(VangaError::DataError("Missing required column 'close' in CSV".to_string()))

// Use ? operator for error propagation
let processed_data = data_loader.load(path)?.validate()?;

// Handle specific error types
match result {
    Err(VangaError::ModelNotFound { symbol }) => {
        log::warn!("No trained model found for {}, starting fresh training", symbol);
        train_new_model(symbol).await?
    }
    Err(e) => return Err(e),
    Ok(model) => model,
}
```

### Logging Guidelines
```rust
// Use structured logging with context
log::info!("Starting training for symbol: {} with {} samples", symbol, data_len);
log::warn!("Low data quality detected: {}% missing values", missing_pct);
log::error!("Training failed for {}: {}", symbol, error);

// Performance logging
let start = std::time::Instant::now();
let result = expensive_operation().await?;
log::debug!("Operation completed in {:?}", start.elapsed());
```

## Quick Start Checklist

1. **Config First**: Always update relevant `configs/*.toml` template
2. **Symbol-Specific**: Each trading pair needs its own model and configuration
3. **Feature Pipeline**: Use established feature engineering patterns
4. **Auto-Optimization**: Leverage automatic parameter tuning
5. **Error Handling**: Comprehensive error management with descriptive messages
6. **Testing**: Verify with end-to-end train→predict→evaluate workflow

## Advanced Topics

### LSTM Model Performance Troubleshooting

#### Training Issues
- Check logs for "Training started" and convergence messages
- Verify data quality: no excessive missing values or outliers
- Monitor validation loss for overfitting signs
- Ensure sufficient training data (minimum 1000 samples recommended)

#### Prediction Accuracy Issues
- Verify feature consistency between training and prediction data
- Check for data leakage in feature engineering
- Validate target distribution balance
- Monitor prediction confidence scores

#### Memory and Performance Issues
- Enable batch processing for large datasets
- Use sequence length optimization for memory efficiency
- Monitor GPU/CPU utilization during training
- Check for memory leaks in long-running processes

### Cryptocurrency-Specific Optimizations

#### Market Regime Detection
- Automatic detection of trending vs ranging markets
- Regime-specific model parameters
- Volatility clustering adaptation
- Cross-asset correlation analysis

#### Feature Engineering Best Practices
- **Required Columns**: timestamp, open, high, low, close, volume
- **Optional Columns**: All additional numeric columns automatically included
- **Technical Indicators**: 50+ crypto-optimized indicators
- **Market Microstructure**: Price velocity, VWAP deviations, trade intensity

### Development Performance Tips

#### MANDATORY BUILD COMMANDS:
```bash
# Fast compilation check (PREFERRED for development)
cargo check --message-format=short

# Code quality enforcement (MANDATORY before commits)
cargo clippy --all-features --all-targets -- -D warnings

# Testing (run frequently during development)
cargo test

# Debug build (only when you need the binary)
cargo build

# NEVER use --release during development (extremely slow)
```

#### Code Quality Standards
- **Zero clippy warnings** - All code must pass clippy without warnings
- **Comprehensive error handling** - Use `Result<T>` everywhere
- **Configuration-driven** - Avoid hardcoded parameters
- **Symbol-agnostic** - Code should work for any trading pair
- **Async-first** - Use tokio throughout for non-blocking operations

#### Testing Approach
- **Unit tests** for individual components (feature engineering, targets)
- **Integration tests** for full training/prediction pipelines
- **End-to-end tests** with sample cryptocurrency data
- **Performance tests** for large dataset handling

#### Architecture Decisions
- **Configuration-first** - All features configurable via TOML files
- **Modular design** - Independent feature engineering, model training, and prediction
- **Symbol-specific models** - One model per trading pair for optimal performance
- **Multi-target prediction** - Price levels, direction, and volatility in one framework
- **Auto-optimization** - Minimal manual parameter tuning required

### Cryptocurrency Data Requirements

#### Required Data Format
```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42300.0,1234.56
```

#### Optional Enhancements
- `volume_quote` - Quote asset volume
- `trades_count` - Number of trades
- `buy_volume` - Buyer volume
- Custom indicators - Any additional numeric columns

#### Data Quality Guidelines
- **Minimum frequency**: 1-minute to 1-day intervals supported
- **Minimum history**: 1000+ samples for reliable training
- **Missing data**: Automatic interpolation and forward-fill
- **Outlier handling**: Automatic detection and treatment

ANTI-PATTERNS

NEVER DO THIS:
- Hardcode model parameters instead of using auto-optimization
- Create symbol-agnostic models (each pair needs its own model)
- Skip feature engineering pipeline
- Use manual hyperparameter tuning
- Ignore error handling in async functions
- Hardcode file paths or configuration values
- Skip validation of input data quality
- Use synchronous operations for I/O-heavy tasks
- Assume data format without validation
- Create models without proper target generation

⚡ MANDATORY PRE-DEVELOPMENT CHECK:
□ Using auto-optimization instead of manual tuning?
□ Symbol-specific model architecture?
□ Comprehensive error handling with Result<T>?
□ Configuration-driven instead of hardcoded?
□ Following established feature engineering patterns?

UNCHECKED = STOP & FIX
