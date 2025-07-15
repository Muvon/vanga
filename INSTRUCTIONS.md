# VANGA LSTM Development Instructions & Onboarding

## 🎯 Core Philosophy

### Single Source of Truth
- **One method, one purpose**: Don't create `train_a`, `train_b`, `train_with_xyz` - enhance the existing method
- **Configuration-driven**: Use TOML configs and conditional logic, not method proliferation
- **Unified interfaces**: Prefer single methods with optional parameters over multiple specialized methods

### Code Quality Standards
- **Zero warnings**: All code must pass `cargo clippy --all-features --all-targets -- -D warnings`
- **No hidden variables**: Never use `_variable` to silence warnings - fix the root cause
- **No dead code**: Don't use `#[allow(dead_code)]` - remove unused code or fix the issue
- **DRY principle**: Don't repeat yourself - extract common logic into shared functions

## 🚀 Quick Start Checklist

When you receive a task, follow this sequence:

### 1. Context Discovery (Parallel Execution)
```bash
# Run these simultaneously for maximum efficiency
semantic_search(["task keywords", "related functionality"])
graphrag(operation="search", query="find files related to task")
list_files(directory="relevant_dir", pattern="*.rs")
view_signatures(files=["key_files"])
remember(["task context", "similar issues"])
```

### 2. Code Analysis
- **Find existing patterns**: Look for similar implementations before creating new ones
- **Check configuration**: Verify TOML configs in `configs/` directory
- **Understand data flow**: Trace from input → processing → output
- **Identify integration points**: How does this fit with existing code?

### 3. Implementation Strategy
- **Enhance, don't duplicate**: Modify existing methods with conditional logic
- **Configuration first**: Add new parameters to TOML configs
- **Test-driven**: Ensure changes work with existing tests
- **Error handling**: Use `Result<T>` and proper error propagation

## 📁 Project Structure Deep Dive

### Core Architecture
```
src/
├── api/           # High-level training/prediction APIs
│   ├── trainer.rs     # Training pipeline orchestration
│   └── predictor.rs   # Prediction pipeline orchestration
├── model/         # LSTM implementations
│   ├── lstm_simple.rs # Core LSTM (MAIN TRAINING LOGIC)
│   └── multi_target.rs # Multi-target wrapper
├── features/      # Feature engineering
│   ├── technical.rs   # Technical indicators
│   └── cross_asset.rs # Cross-asset features
├── data/          # Data loading and preprocessing
│   ├── loader.rs      # CSV loading and validation
│   └── schema.rs      # Data schema definitions
├── config/        # Configuration management
│   ├── training.rs    # Training parameters
│   └── features.rs    # Feature configurations
└── utils/         # Utilities and error handling
    └── error.rs       # Error types and handling
```

### Key Files to Know

#### `src/model/lstm_simple.rs` - CRITICAL
- **Main training method**: `pub async fn train()` - THE method to enhance
- **Never create**: `train_with_xyz()` methods - use conditional logic inside `train()`
- **Validation logic**: Handles both internal splits and pre-split chronological data
- **Configuration-driven**: All behavior controlled by `TrainingConfig`

#### `src/model/multi_target.rs`
- **Wrapper around lstm_simple**: Trains multiple models for different targets
- **Chronological validation**: `train_with_chronological_validation()` for time-series data
- **Symbol-specific**: Each trading pair gets its own model

#### `src/config/training.rs`
- **Training parameters**: Epochs, learning rate, batch size, validation splits
- **Validation methods**: `validate()` and `validate_for_symbols()`
- **Auto-optimization**: Intelligent parameter tuning configurations

#### `configs/*.toml`
- **Configuration templates**: Different scenarios (training, prediction, features)
- **Symbol-specific**: Each trading pair can have specialized configs
- **Feature flags**: Enable/disable functionality via configuration

## 🔧 Common Task Patterns

### Adding New Features

#### ❌ WRONG Approach
```rust
// DON'T create new methods
pub async fn train_with_new_feature() { ... }
pub async fn train_without_new_feature() { ... }
```

#### ✅ CORRECT Approach
```rust
// DO enhance existing method with conditional logic
pub async fn train(
    &mut self,
    sequences: &Array3<f64>,
    targets: &Array2<f64>,
    config: &TrainingConfig,
    // Add optional parameters as needed
    new_feature_data: Option<&SomeType>,
) -> Result<()> {
    // Conditional logic based on config or parameters
    if config.new_feature.enabled || new_feature_data.is_some() {
        // Use new feature
    } else {
        // Standard behavior
    }
}
```

### Fixing Validation Issues

#### Common Problem: Unused Variables
```rust
// ❌ WRONG - hiding the issue
let _unused_var = some_computation();

// ✅ CORRECT - fix the root cause
let validation_data = some_computation();
// Actually use validation_data in the logic
```

#### Validation Data Flow
1. **Chronological split**: `src/data/loader.rs::split_chronological()`
2. **Multi-target training**: `src/model/multi_target.rs::train_with_chronological_validation()`
3. **Single model training**: `src/model/lstm_simple.rs::train()` with validation parameters
4. **Early stopping**: Uses validation loss for stopping decisions

### Configuration Management

#### Adding New Config Parameters
1. **Update struct**: Add field to relevant config struct in `src/config/`
2. **Add default**: Implement in `Default` trait
3. **Update TOML**: Add to corresponding `configs/*.toml` file
4. **Add validation**: Include in validation methods if needed

#### Example Flow
```rust
// 1. src/config/training.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    pub new_parameter: bool,
    // ... existing fields
}

// 2. Default implementation
impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            new_parameter: false,
            // ... existing defaults
        }
    }
}

// 3. configs/training.toml
new_parameter = true  # Enable new functionality

// 4. Usage in code
if config.training.new_parameter {
    // New behavior
}
```

## 🚨 Critical Anti-Patterns

### Method Proliferation
```rust
// ❌ NEVER DO THIS
pub async fn train() { ... }
pub async fn train_with_validation() { ... }
pub async fn train_without_validation() { ... }
pub async fn train_with_early_stopping() { ... }
pub async fn train_with_custom_lr() { ... }
```

### Hidden Variables
```rust
// ❌ NEVER DO THIS
let _validation_data = extract_validation();  // Unused!

// ✅ DO THIS INSTEAD
let validation_data = extract_validation();
if let Some(val_data) = validation_data {
    // Actually use the validation data
    model.validate_with(val_data)?;
}
```

### Dead Code Allowance
```rust
// ❌ NEVER DO THIS
#[allow(dead_code)]
fn unused_function() { ... }

// ✅ DO THIS INSTEAD
// Remove the function or fix why it's not being used
```

### Hardcoded Parameters
```rust
// ❌ NEVER DO THIS
let batch_size = 32;  // Hardcoded!

// ✅ DO THIS INSTEAD
let batch_size = config.training.batch_size;
```

## 🔍 Debugging & Investigation

### Finding Issues
1. **Compiler errors**: Start with `cargo check --message-format=short`
2. **Code quality**: Run `cargo clippy --all-features --all-targets -- -D warnings`
3. **Semantic search**: Use descriptive queries, not symbol names
4. **AST patterns**: Use `ast_grep` for structural code searches
5. **Memory search**: Check `remember()` for similar past issues

### Common Issue Locations

#### Training Problems
- **Main logic**: `src/model/lstm_simple.rs::train()`
- **Multi-target**: `src/model/multi_target.rs`
- **Configuration**: `src/config/training.rs`
- **Data loading**: `src/data/loader.rs`

#### Feature Engineering Issues
- **Technical indicators**: `src/features/technical.rs`
- **Cross-asset features**: `src/features/cross_asset.rs`
- **Configuration**: `src/config/features.rs`

#### Validation & Early Stopping
- **Early stopping logic**: `src/model/lstm_simple.rs` (lines ~1020-1040)
- **Validation splits**: `src/model/lstm_simple.rs` (lines ~790-830)
- **Chronological validation**: `src/model/multi_target.rs::train_with_chronological_validation()`

## 📋 Development Workflow

### Before Making Changes
1. **Understand the task**: What exactly needs to be fixed/added?
2. **Find existing patterns**: How is similar functionality implemented?
3. **Check configuration**: Is this configurable? Should it be?
4. **Plan the approach**: Enhance existing code or truly need new code?

### During Development
1. **Follow DRY**: Don't repeat existing logic
2. **Use configuration**: Make behavior configurable when possible
3. **Handle errors**: Use `Result<T>` and proper error propagation
4. **Add logging**: Use structured logging with context
5. **Test incrementally**: `cargo check` frequently

### After Implementation
1. **Code quality**: `cargo clippy --all-features --all-targets -- -D warnings`
2. **Test functionality**: Verify the fix works as expected
3. **Update configs**: Add new parameters to TOML files if needed
4. **Document changes**: Update relevant documentation

## 🎯 Performance Guidelines

### Memory Management
- **Lazy loading**: Load data progressively during training
- **Batch processing**: Process data in configurable batch sizes
- **Avoid cloning**: Use references when possible
- **Clean up**: Drop large data structures when done

### Training Optimization
- **Auto-optimization**: Prefer automatic parameter tuning
- **Symbol-specific**: One model per trading pair
- **Chronological validation**: Prevent data leakage
- **Early stopping**: Stop training when validation loss plateaus

### Build Performance
```bash
# Fast development cycle
cargo check --message-format=short  # PREFERRED for development

# Code quality (mandatory before commits)
cargo clippy --all-features --all-targets -- -D warnings

# Testing
cargo test

# Debug build (only when needed)
cargo build

# NEVER use --release during development (extremely slow)
```

## 🔧 Tool Usage

### Semantic Search
```rust
// ✅ GOOD - descriptive, multi-term queries
semantic_search(["user authentication flow", "login validation", "jwt token handling"])

// ❌ BAD - single terms or symbol names
semantic_search(["train"])  // Too generic
semantic_search(["train_with_early_stopping"])  // Symbol name, not concept
```

### AST Grep
```rust
// Find patterns, not exact matches
ast_grep(pattern="fn $NAME($ARGS) -> Result<()>", language="rust")

// Find method calls
ast_grep(pattern="$OBJ.train($$$)", language="rust")
```

### GraphRAG
```rust
// Find files by purpose/description
graphrag(operation="search", query="training pipeline orchestration")

// Understand relationships
graphrag(operation="get-relationships", node_id="src/model/lstm_simple.rs")
```

## 📚 Key Concepts

### LSTM Architecture
- **Intelligent optimization**: Automatic parameter tuning based on data
- **Multi-layer support**: Stacked LSTM layers for complex patterns
- **Attention mechanism**: Optional attention layers for better performance
- **Sequence-to-one**: Predicts single output from sequence input

### Cryptocurrency-Specific Features
- **Technical indicators**: 50+ crypto-optimized indicators
- **Market microstructure**: Price velocity, VWAP deviations, trade intensity
- **Cross-asset analysis**: BTC dominance, ETH/BTC ratio, market sentiment
- **Volatility clustering**: Crypto-specific volatility patterns

### Data Pipeline
1. **CSV loading**: Automatic schema detection and validation
2. **Feature engineering**: Technical indicators and custom features
3. **Sequence generation**: Convert time series to LSTM input format
4. **Target creation**: Multi-target labels (price, direction, volatility)
5. **Chronological splitting**: Prevent data leakage in validation

## 🚀 Quick Reference

### Essential Commands
```bash
# Development cycle
cargo check --message-format=short
cargo clippy --all-features --all-targets -- -D warnings
cargo test

# Find code patterns
rg "pattern" --type rust
ast-grep --pattern "pattern" --lang rust

# Training example
cargo run -- train --symbol BTCUSDT --data data.csv --config configs/training.toml
```

### Key Files for Common Tasks
- **Training issues**: `src/model/lstm_simple.rs`
- **Multi-target problems**: `src/model/multi_target.rs`
- **Feature engineering**: `src/features/technical.rs`, `src/features/cross_asset.rs`
- **Configuration**: `src/config/training.rs`, `configs/*.toml`
- **Data loading**: `src/data/loader.rs`
- **Error handling**: `src/utils/error.rs`

### Configuration Hierarchy
1. **Default values**: In `Default` trait implementations
2. **TOML files**: `configs/*.toml` templates
3. **Command line**: Override via CLI arguments
4. **Runtime**: Dynamic adjustments during training

---

## 🎯 Remember: Quality Over Speed

- **Think before coding**: Understand the problem fully
- **Reuse before creating**: Look for existing solutions
- **Configure before hardcoding**: Make behavior configurable
- **Test before committing**: Ensure code quality standards
- **Document before forgetting**: Update relevant documentation

**The goal is maintainable, high-quality code that follows VANGA's architectural principles.**

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
