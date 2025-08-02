# VANGA LSTM Development Instructions & Onboarding

## 🎯 Core Philosophy

### Single Source of Truth
- **One method, one purpose**: Don't create `train_a`, `train_b`, `train_with_xyz` - enhance the existing method
- **Configuration-driven**: Use TOML configs and conditional logic, not method proliferation
- **Unified interfaces**: Prefer single methods with optional parameters over multiple specialized methods
- **Tensor-first architecture**: All operations use Candle tensors with proper broadcasting and gradient flow

### CRITICAL: Global vs Per-Batch Calculations
- **Global calculations**: Class weights, quantiles, normalization parameters must be calculated ONCE from entire training dataset
- **Per-batch calculations**: Only use for gradient updates, loss accumulation, and batch-specific operations
- **Loss consistency**: Training and validation must use SAME global parameters for comparable losses
- **Chronological integrity**: Preserve time-series order - no shuffling in crypto/financial data
- **Target consistency**: Use percentage-based quantiles for symbol-agnostic classification difficulty
- **Normalization consistency**: Prediction must use training normalization statistics

### Code Quality Standards
- **Zero warnings**: All code must pass `cargo clippy --all-features --all-targets -- -D warnings`
- **No hidden variables**: Never use `_variable` to silence warnings - fix the root cause
- **No dead code**: Don't use `#[allow(dead_code)]` - remove unused code or fix the issue
- **DRY principle**: Don't repeat yourself - extract common logic into shared functions
- **Tensor safety**: Always use `broadcast_as()` for shape matching, ensure `.contiguous()` for operations
- **Test organization**: ALL tests must be in separate `*_test.rs` files - NEVER inline `#[cfg(test)]` modules

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
- **Check configuration**: Verify TOML configs in `configs/` directory (20+ available configs)
- **Understand data flow**: Trace from input → processing → output
- **Identify integration points**: How does this fit with existing code?
- **Tensor operations**: Check for proper broadcasting and gradient flow

### 3. Implementation Strategy
- **Enhance, don't duplicate**: Modify existing methods with conditional logic
- **Configuration first**: Add new parameters to TOML configs
- **Test-driven**: Ensure changes work with existing tests
- **Error handling**: Use `Result<T>` and proper error propagation
- **Tensor safety**: Use `broadcast_as()` for shape matching, `.contiguous()` for operations

## 🧪 Test Organization & Standards

### MANDATORY Test Structure
- **Separate test files ONLY**: All tests must be in dedicated `*_test.rs` files (singular)
- **NO inline tests**: Never use `#[cfg(test)]` modules within source files
- **Consistent naming**: Test files follow `{module_name}_test.rs` pattern (singular, not `_tests.rs`)
- **Proper imports**: Use `use crate::module::*;` for accessing tested code
- **Test discovery**: Cargo automatically discovers `*_test.rs` files

### Test File Organization
```
src/
├── data/
│   ├── structures.rs          # Implementation
│   ├── structures_test.rs     # Tests for structures.rs
│   ├── target_converter.rs    # Implementation
│   └── target_converter_test.rs # Tests for target_converter.rs
├── model/
│   ├── multi_target.rs        # Implementation
│   ├── multi_target_test.rs   # Tests for multi_target.rs
│   ├── attention_optimizer.rs # Implementation
│   └── attention_optimizer_test.rs # Tests for attention_optimizer.rs
└── utils/
    ├── sequence_utils.rs      # Implementation
    ├── sequence_utils_test.rs # Tests for sequence_utils.rs
    ├── file_discovery.rs      # Implementation
    ├── file_discovery_test.rs # Tests for file_discovery.rs
    ├── device.rs              # Implementation
    └── device_test.rs         # Tests for device.rs
```

### Test File Template
```rust
// src/module/feature_test.rs (singular _test.rs, not _tests.rs)
use crate::module::feature::*;
use crate::other::dependencies::*;

#[test]
fn test_basic_functionality() {
    // Test implementation
    assert_eq!(expected, actual);
}

#[tokio::test]  // For async tests
async fn test_async_functionality() {
    // Async test implementation
    let result = async_function().await;
    assert!(result.is_ok());
}
```

### Benefits of Separate Test Files
- ✅ **Cleaner source code**: Implementation files focus purely on logic
- ✅ **Better organization**: Tests are easily discoverable and maintainable
- ✅ **Rust best practices**: Follows community standards and conventions
- ✅ **IDE support**: Better test runner integration and debugging
- ✅ **Parallel compilation**: Tests can be compiled independently
- ✅ **Reduced cognitive load**: Developers can focus on implementation or tests separately

### Test Development Rules
- **Test-first approach**: Write tests in separate files from the start
- **Comprehensive coverage**: Test both success and error cases
- **Descriptive names**: Use clear, descriptive test function names
- **Isolated tests**: Each test should be independent and repeatable
- **Mock dependencies**: Use proper mocking for external dependencies

### FORBIDDEN Test Patterns
```rust
// ❌ NEVER DO THIS - Inline test modules
impl SomeStruct {
    fn some_method(&self) -> Result<()> {
        // implementation
    }
}

#[cfg(test)]
mod tests {  // ❌ FORBIDDEN - inline test module
    use super::*;

    #[test]
    fn test_some_method() {
        // test code
    }
}
```

```rust
// ✅ DO THIS INSTEAD - Separate test file
// In src/module/some_struct_test.rs
use crate::module::some_struct::*;

#[test]
fn test_some_method() {
    let instance = SomeStruct::new();
    let result = instance.some_method();
    assert!(result.is_ok());
}
```

## 📁 Project Structure Deep Dive

### Core Architecture
```
src/
├── api/           # High-level training/prediction APIs
│   ├── trainer.rs     # Training pipeline orchestration
│   ├── predictor.rs   # Prediction pipeline orchestration
│   └── backtester.rs  # Backtesting framework
├── model/         # LSTM implementations
│   ├── lstm/          # Modular LSTM implementation (CURRENT STRUCTURE)
│   │   ├── config.rs      # LSTMConfig, OptimizerWrapper, TargetFormat
│   │   ├── core.rs        # Model lifecycle and initialization
│   │   ├── training.rs    # THE unified training method (MAIN LOGIC)
│   │   ├── inference.rs   # Prediction and forward pass
│   │   ├── loss.rs        # Loss calculation and metrics
│   │   ├── manual_lstm.rs # Manual LSTM cell implementation
│   │   └── mod.rs         # Public API and re-exports
│   ├── lstm_simple.rs # Compatibility layer: `pub use crate::model::lstm::*;`
│   ├── multi_target.rs # Multi-target wrapper
│   ├── attention.rs   # Multi-head attention mechanisms
│   └── loss.rs        # Composite loss functions
├── features/      # Feature engineering
│   ├── technical.rs   # 50+ technical indicators
│   ├── cross_asset.rs # Cross-asset features
│   └── engineering.rs # Feature engineering pipeline
├── data/          # Data loading and preprocessing
│   ├── loader.rs      # CSV loading and validation
│   ├── preprocessor.rs # Feature normalization (CRITICAL)
│   ├── sequence.rs    # Sequence generation
│   ├── schema.rs      # Data schema definitions
│   ├── structures.rs  # Data structures
│   └── target_converter.rs # Target conversion utilities
├── targets/       # Target generation (CRITICAL)
│   ├── mod.rs         # Target orchestration
│   └── price_levels.rs # VWAP-weighted range analysis (5-class system)
├── config/        # Configuration management
│   ├── training.rs    # TrainingConfig, TrainingParams, 9 optimizers
│   ├── features.rs    # Feature configurations
│   ├── model.rs       # Model architecture configurations
│   └── mod.rs         # Configuration coordination
├── optimization/  # Optimization and feature selection
│   └── feature_selection.rs # Feature selection algorithms
└── utils/         # Utilities and error handling
    ├── error.rs       # VangaError types and handling
    └── metrics.rs     # Evaluation metrics
```

## 🔄 CRITICAL: Training vs Prediction Data Flow

### Training Pipeline Architecture
```
Raw CSV → Feature Engineering → NaN Removal → Outlier Handling → Target Generation → Sequence Creation → Multi-Model Training
    ↓           ↓                    ↓             ↓                ↓                  ↓                ↓
OHLCV Data  Technical Indicators  Clean Data   Processed Data   3×5 Targets      Sequences      N×LSTMModel
```

### Prediction Pipeline Architecture
```
Raw CSV → Feature Engineering → NaN Removal → Outlier Handling → Sequence Creation → Multi-Model Prediction
    ↓           ↓                    ↓             ↓                ↓                  ↓
OHLCV Data  Technical Indicators  Clean Data   Processed Data   Sequences         N×Predictions
```

### Key Data Flow Details
- **No Global Normalization**: Uses per-sequence processing approach
- **Feature Engineering**: Applied before any other processing
- **NaN Removal**: Critical step to remove lag feature warmup period
- **Target Independence**: Each target type calculated independently from sequences
- **Multi-Model Coordination**: MultiTargetLSTMModel manages separate models per target×horizon

### ⚠️ CRITICAL ARCHITECTURE REQUIREMENTS

#### Multi-Model Architecture
- **Single LSTMModel Limitation**: Each `LSTMModel` handles only ONE target (5 categorical outputs)
- **MultiTargetLSTMModel Solution**: Wraps multiple `LSTMModel` instances (one per target×horizon)
- **Example**: 3 targets × 2 horizons = 6 separate `LSTMModel` instances
- **Training Coordination**: `TrainingContext` manages training across all models
- **Prediction Aggregation**: Combines predictions from all individual models

#### Target Generation (Training Only)
- **3 Target Types**: Price levels, direction, volatility - each independent
- **5-Class System**: Each target outputs 5 categorical classes (`NUM_CLASSES = 5`)
- **Sequence-Based**: All targets calculated from sequence data, not market regimes
- **VWAP-Weighted**: Price levels use volume-weighted analysis for accuracy
- **Total Output**: 3 targets × 5 classes = 15 outputs per prediction

#### Data Processing Consistency
- **No Global Normalization**: Uses per-sequence processing approach
- **Feature Engineering First**: Technical indicators applied before any processing
- **NaN Removal Critical**: Must remove lag feature warmup period
- **Outlier Handling**: Applied after feature engineering, before target generation
- **Sequence Alignment**: Targets must align with sequence indices, not raw data indices

### Key Files to Know

#### `src/model/lstm/` - SINGLE LSTM MODEL (Core Implementation)
- **training.rs**: `pub async fn train(&mut self, sequences: &Array3<f64>, targets: &Array2<f64>, config: &TrainingConfig, val_sequences: Option<&Array3<f64>>, val_targets: Option<&Array2<f64>>, class_weights: Option<&Vec<f32>>) -> Result<()>` - THE unified training method
- **config.rs**: `LSTMConfig`, `OptimizerWrapper` (9 optimizers), `TargetFormat` - Single model configuration
- **core.rs**: Model lifecycle, initialization, persistence, Xavier initialization
- **inference.rs**: `predict()` method - Single model prediction
- **loss.rs**: Loss calculation with weighted cross-entropy for single target
- **Limitation**: Can only handle ONE target at a time (hence the wrapper)

#### `src/model/multi_target.rs` - MULTI-LSTM WRAPPER
- **Purpose**: Wraps multiple `LSTMModel` instances to overcome single-target limitation
- **Architecture**: Creates separate `LSTMModel` for each target×horizon combination
- **Example**: 3 targets × 2 horizons = 6 separate `LSTMModel` instances
- **Training**: `TrainingContext` coordinates training across all models
- **Prediction**: Aggregates predictions from all individual models

#### `src/model/lstm_simple.rs` - COMPATIBILITY LAYER
- **Implementation**: `pub use crate::model::lstm::*;` - Pure re-export
- **Purpose**: Maintains backward compatibility for existing code

#### `src/targets/` - TARGET GENERATION (3 Targets × 5 Classes Each)
- **Architecture**: 3 independent target types, each with 5 categorical outputs
- **price_levels.rs**: VWAP-weighted range analysis (5-class: Strong Down, Moderate Down, Neutral, Moderate Up, Strong Up)
- **direction.rs**: Directional movement classification (5-class categorical)
- **volatility.rs**: Volatility regime classification (5-class categorical)
- **mod.rs**: `TargetGenerator` orchestrates all 3 target types
- **Output Structure**: `NUM_CLASSES = 5` for each target type
- **Sequence-based**: All targets calculated from sequence data, independent of market/regime
- **Total Output**: 3 targets × 5 classes = 15 total outputs per prediction

#### `src/config/training.rs` - TRAINING CONFIGURATION
- **TrainingConfig**: Complete pipeline configuration coordinator
- **TrainingParams**: 9 optimizers (AdamW, SGD, Adam, AdaDelta, AdaGrad, AdaMax, NAdam, RAdam, RMSprop)
- **DataConfig**: Outlier handling, feature processing configuration
- **OptimizationConfig**: Hyperparameter optimization settings
- **DeviceConfig**: CPU/GPU device selection
- **EpochConfig**: Auto early stopping or fixed epochs

#### `src/data/preprocessor.rs` - DATA PROCESSING
- **process_features_only()**: Feature engineering without global normalization
- **remove_nan_rows()**: Critical NaN removal for lag features
- **Outlier handling**: IQR and Z-score methods
- **No global normalization**: Uses per-sequence approach
- **Feature engineering integration**: Applies technical indicators first

#### `src/data/preprocessor.rs` - CRITICAL NORMALIZATION
- **Feature normalization**: Z-score normalization with saved statistics
- **Training mode**: Calculates and saves normalization parameters
- **Prediction mode**: Loads and applies SAME normalization parameters
- **Consistency rule**: Prediction MUST use training normalization stats
- **Storage**: Normalization stats saved with model for inference consistency

#### `configs/*.toml` - 30+ CONFIGURATIONS
- **training.toml**: Main configuration with all 9 optimizers documented
- **optimizer_examples/**: 9 optimizer-specific configurations (adamw_crypto_optimized.toml, etc.)
- **quick_start.toml**: Beginner-friendly minimal setup
- **cross_asset_training.toml**: Multi-asset correlation analysis
- **hybrid_training.toml**: XGBoost + TFT integration examples
- **backtest.toml**: Backtesting framework configuration

#### Configuration Structure Example:
```toml
[training]
optimizer = { AdamW = { weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } }
epochs = { Auto = { max_epochs = 1000 } }
batch_size = { Auto = { min_size = 16, max_size = 128 } }

[model]
architecture = { MultiLSTM = { layers = 2 } }
sequence_length = { Auto = { min_length = 30, max_length = 120 } }

[data]
outlier_handling = { enabled = true, method = "IQR", threshold = 3.0 }
```

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

### Fixing Tensor Broadcasting Issues (CRITICAL PATTERN)

#### Common Problem: Shape Mismatch
```rust
// ❌ WRONG - Direct multiplication without broadcasting
let result = tensor_a.mul(&tensor_b)?; // Fails with [16,6] × [1,6]

// ✅ CORRECT - Use broadcast_as for explicit shape matching
let tensor_b_broadcast = tensor_b.broadcast_as(tensor_a.shape())?;
let result = tensor_a.mul(&tensor_b_broadcast)?.contiguous()?;
```

#### Tensor Safety Pattern (MANDATORY)
```rust
// Always follow this pattern for tensor operations:
let tensor_contiguous = input_tensor.contiguous()?;
let broadcast_tensor = weight_tensor.broadcast_as(tensor_contiguous.shape())?;
let result = tensor_contiguous.mul(&broadcast_tensor)?.contiguous()?;
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

### Inline Test Modules
```rust
// ❌ NEVER DO THIS - Inline test modules in source files
impl SomeStruct {
    fn implementation(&self) -> Result<()> {
        // implementation code
    }
}

#[cfg(test)]  // ❌ FORBIDDEN - inline tests
mod tests {
    use super::*;

    #[test]
    fn test_implementation() {
        // test code mixed with implementation
    }
}

// ✅ DO THIS INSTEAD - Separate test file (src/module/some_struct_test.rs)
use crate::module::some_struct::*;

#[test]
fn test_implementation() {
    let instance = SomeStruct::new();
    let result = instance.implementation();
    assert!(result.is_ok());
}
```

### Tensor Broadcasting Errors (CRITICAL)
```rust
// ❌ NEVER DO THIS - Direct multiplication without shape checking
let result = targets.mul(&weights)?; // FAILS with shape mismatch

// ✅ DO THIS INSTEAD - Always use broadcast_as
let weights_broadcast = weights.broadcast_as(targets.shape())?;
let result = targets.mul(&weights_broadcast)?.contiguous()?;
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
let num_classes = 6;  // Hardcoded!

// ✅ DO THIS INSTEAD
let batch_size = config.training.batch_size;
let num_classes = self.get_num_classes_for_target(target_type);
```

### Gradient Flow Breaking
```rust
// ❌ NEVER DO THIS - Manual loss calculation breaks gradients
let loss_value = manual_calculation(predictions, targets);
let loss_tensor = Tensor::new(loss_value, device)?;

// ✅ DO THIS INSTEAD - Use tensor operations throughout
let loss_tensor = predictions.sub(targets)?.sqr()?.mean_all()?;
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
- **Main logic**: `src/model/lstm/training.rs::train()` (NEW LOCATION)
- **Loss functions**: `src/model/lstm/loss.rs` (tensor broadcasting, class weights)
- **Configuration**: `src/model/lstm/config.rs` + `src/config/training.rs`
- **Multi-target**: `src/model/multi_target.rs`
- **Data loading**: `src/data/loader.rs`

#### Tensor Broadcasting Issues (CRITICAL)
- **LSTM loss functions**: `src/model/lstm/loss.rs::calculate_weighted_soft_crossentropy_loss()` (NEW LOCATION)
- **Composite loss**: `src/model/loss.rs::calculate_weighted_soft_crossentropy_loss()` (matches LSTM implementation)
- **Attention mechanism**: `src/model/attention.rs` (uses broadcast_div pattern)
- **Loss functions**: `src/model/loss.rs` (uses broadcast_as pattern)
- **Pattern**: Always use `broadcast_as()` before tensor operations

#### Loss Function Consistency (CRITICAL)
- **Class weighting**: Both MSE and Composite must use identical class weights for categorical targets
- **MSE path**: Uses LSTM's `calculate_weighted_soft_crossentropy_loss()` with global class weights
- **Composite path**: Uses `TensorCryptoLossFunction::calculate_weighted_soft_crossentropy_loss()` (same logic)
- **Critical**: Extreme class imbalance (248x weight for rare classes) requires sophisticated weighting
- **Debug logs**: Look for `🌍 Applying global class weights` and `⚖️ Composite Weighted Soft CrossEntropy`

#### Feature Engineering Issues
- **Technical indicators**: `src/features/technical.rs`
- **Cross-asset features**: `src/features/cross_asset.rs`
- **Configuration**: `src/config/features.rs`

#### Validation & Early Stopping
- **Early stopping logic**: `src/model/lstm/training.rs` (NEW LOCATION)
- **Validation splits**: `src/model/lstm/training.rs` (NEW LOCATION)
- **Chronological validation**: `src/model/multi_target.rs::train_with_chronological_validation()`

#### Price Level Target Issues (CRITICAL ARCHITECTURE)
- **Target generation**: `src/targets/price_levels.rs::calculate_price_level_targets()`
- **Percentage-based quantiles**: Ensures symbol-agnostic classification difficulty
- **Class imbalance**: `src/targets/price_levels.rs::analyze_class_distribution()`
- **Class weighting**: `src/model/lstm_simple.rs::calculate_class_weights()`
- **Label smoothing**: `src/model/lstm_simple.rs::apply_label_smoothing()`
- **Architecture rule**: Never use raw price quantiles (creates symbol-specific difficulty)

#### Normalization Consistency Issues (CRITICAL)
- **Training normalization**: `src/data/preprocessor.rs` (calculates and saves stats)
- **Prediction normalization**: `src/data/preprocessor.rs` (loads and applies same stats)
- **Consistency rule**: Prediction MUST use training normalization parameters
- **Common error**: Calculating new normalization stats during prediction
- **Fix pattern**: Always load saved normalization stats for inference

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

### Data Pipeline Architecture
1. **CSV loading**: Automatic schema detection and validation (`src/data/loader.rs`)
2. **Target generation**: Percentage-based price level classification (`src/targets/price_levels.rs`)
3. **Feature engineering**: Technical indicators and custom features (`src/features/`)
4. **Normalization**: Z-score normalization with saved statistics (`src/data/preprocessor.rs`)
5. **Sequence generation**: Convert time series to LSTM input format (`src/data/sequence.rs`)
6. **Chronological splitting**: Prevent data leakage in validation (`src/data/loader.rs`)

### ⚠️ CRITICAL ARCHITECTURE RULES

#### Target Generation Consistency
- **Training**: Uses percentage-based quantiles for symbol-agnostic classification
- **Validation**: Uses SAME percentage boundaries for comparable losses
- **Never**: Use raw price quantiles (creates symbol-specific difficulty)
- **Result**: All symbols have comparable validation losses (~0.8-1.2 range)

#### Normalization Consistency
- **Training**: Calculate normalization stats from training data, save with model
- **Prediction**: Load and apply SAME normalization stats from training
- **Never**: Calculate new normalization stats during prediction
- **Result**: Consistent feature scaling between training and inference

#### Loss Function Architecture
- **Categorical targets**: Cross-entropy loss with class weighting and label smoothing
- **Regression targets**: MSE/MAE loss with normalized values
- **Multi-target**: Weighted combination of target-specific losses
- **Validation**: Uses SAME loss calculation as training for comparability

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

# Available configurations (20+ configs)
ls configs/
ls configs/optimizer_examples/
```

### Key Files for Common Tasks
- **Training issues**: `src/model/lstm/training.rs` (NEW: main training logic)
- **Loss functions**: `src/model/lstm/loss.rs` (NEW: tensor broadcasting, class weights)
- **LSTM configuration**: `src/model/lstm/config.rs` (NEW: LSTMConfig, optimizers)
- **Model lifecycle**: `src/model/lstm/core.rs` (NEW: initialization, persistence)
- **Prediction**: `src/model/lstm/inference.rs` (NEW: predict method)
- **Multi-target problems**: `src/model/multi_target.rs`
- **Target generation**: `src/targets/price_levels.rs` (percentage-based quantiles)
- **Feature normalization**: `src/data/preprocessor.rs` (training/prediction consistency)
- **Feature engineering**: `src/features/technical.rs`, `src/features/cross_asset.rs`
- **Configuration**: `src/config/training.rs`, `configs/*.toml`
- **Data loading**: `src/data/loader.rs`
- **Sequence generation**: `src/data/sequence.rs`
- **Error handling**: `src/utils/error.rs`
- **Tensor operations**: `src/model/loss.rs`, `src/model/attention.rs`

### Critical Architecture Understanding
1. **Symbol-Agnostic Design**: All code must work for any trading pair
2. **Percentage-Based Targets**: Price levels use percentage changes, not raw prices
3. **Normalization Consistency**: Prediction uses training normalization stats
4. **Chronological Integrity**: Time-series order preserved, no shuffling
5. **Configuration-Driven**: All behavior controlled via TOML configs
6. **Tensor Safety**: Always use `broadcast_as()` for shape matching

### Recent Critical Fixes (Architecture Impact)
- **Composite Loss Class Weighting**: Fixed missing class weights in `TensorCryptoLossFunction` causing severe overfitting
- **Target Generation**: Fixed symbol-specific loss scaling using percentage-based quantiles
- **Normalization Consistency**: Ensured prediction uses training normalization parameters
- **Tensor Broadcasting**: Fixed `[16,6] × [1,6]` multiplication using `broadcast_as()`
- **Loss Comparability**: All loss functions now use identical class weighting for categorical targets
- **Categorical Metrics**: Added accuracy, precision, recall, F1 scores for classification

### Build Performance Guidelines
```bash
# MANDATORY: Fast development cycle
cargo check --message-format=short  # PREFERRED for development

# MANDATORY: Code quality (before any commits)
cargo clippy --all-features --all-targets -- -D warnings

# Testing
cargo test

# Debug build (only when needed)
cargo build

# NEVER use --release during development (extremely slow)
```

---

## 🎯 Remember: Quality Over Speed

- **Think before coding**: Understand the problem fully
- **Reuse before creating**: Look for existing solutions
- **Configure before hardcoding**: Make behavior configurable
- **Test before committing**: Ensure code quality standards
- **Document before forgetting**: Update relevant documentation
- **Broadcast before multiplying**: Always use `broadcast_as()` for tensor operations
- **Weight before loss**: Ensure all loss functions use identical class weighting for categorical targets

**The goal is maintainable, high-quality code that follows VANGA's architectural principles.**

## 🔧 Development Performance Tips

### MANDATORY BUILD COMMANDS:
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

### Code Quality Standards
- **Zero clippy warnings** - All code must pass clippy without warnings
- **Comprehensive error handling** - Use `Result<T>` everywhere
- **Configuration-driven** - Avoid hardcoded parameters
- **Symbol-agnostic** - Code should work for any trading pair
- **Async-first** - Use tokio throughout for non-blocking operations
- **Tensor safety** - Always use `broadcast_as()` for shape matching

## ✅ PRE-RESPONSE CHECK
- □ Maximum parallel tools in one block?
- □ Using plan() for implementations?
- □ Batch file operations when possible?
- □ Only doing what was asked?
- □ Need explicit confirmation for execution?
- □ Tensor operations use broadcast_as()?
- □ All tensors are .contiguous()?

UNCHECKED = STOP & FIX
