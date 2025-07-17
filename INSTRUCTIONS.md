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

### Code Quality Standards
- **Zero warnings**: All code must pass `cargo clippy --all-features --all-targets -- -D warnings`
- **No hidden variables**: Never use `_variable` to silence warnings - fix the root cause
- **No dead code**: Don't use `#[allow(dead_code)]` - remove unused code or fix the issue
- **DRY principle**: Don't repeat yourself - extract common logic into shared functions
- **Tensor safety**: Always use `broadcast_as()` for shape matching, ensure `.contiguous()` for operations

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
- **Tensor operations**: Contains critical broadcasting fixes (lines 2240-2250)
- **Loss calculation**: Multi-target aware with class weighting and label smoothing

#### `src/model/multi_target.rs`
- **Wrapper around lstm_simple**: Trains multiple models for different targets
- **Chronological validation**: `train_with_chronological_validation()` for time-series data
- **Symbol-specific**: Each trading pair gets its own model

#### `src/config/training.rs`
- **Training parameters**: Epochs, learning rate, batch size, validation splits
- **Optimizer configuration**: 9 available optimizers (AdamW, SGD, Adam, AdaDelta, AdaGrad, AdaMax, NAdam, RAdam, RMSprop)
- **Validation methods**: `validate()` and `validate_for_symbols()` with optimizer parameter validation
- **Auto-optimization**: Intelligent parameter tuning configurations

#### `src/targets/price_levels.rs` - RECENTLY UPDATED
- **Price level classification**: Quantile-based categorical targets (4-6 bins)
- **Class distribution analysis**: `analyze_class_distribution()` for imbalance detection
- **Global quantiles**: Fixed quantile calculation for consistency
- **Integration**: Works with class weighting and label smoothing

#### `configs/*.toml` - 20+ CONFIGURATIONS
- **Configuration templates**: Different scenarios (training, prediction, features)
- **Symbol-specific**: Each trading pair can have specialized configs
- **Feature flags**: Enable/disable functionality via configuration
- **Optimizer examples**: All 9 optimizers with crypto-specific recommendations
- **Recent additions**: Cross-asset training, TFT enhanced, backtest configs

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
- **Main logic**: `src/model/lstm_simple.rs::train()`
- **Multi-target**: `src/model/multi_target.rs`
- **Configuration**: `src/config/training.rs`
- **Data loading**: `src/data/loader.rs`

#### Tensor Broadcasting Issues (CRITICAL)
- **Price level loss**: `src/model/lstm_simple.rs::calculate_weighted_soft_crossentropy_loss()` (lines 2240-2250)
- **Attention mechanism**: `src/model/attention.rs` (uses broadcast_div pattern)
- **Loss functions**: `src/model/loss.rs` (uses broadcast_as pattern)
- **Pattern**: Always use `broadcast_as()` before tensor operations

#### Feature Engineering Issues
- **Technical indicators**: `src/features/technical.rs`
- **Cross-asset features**: `src/features/cross_asset.rs`
- **Configuration**: `src/config/features.rs`

#### Validation & Early Stopping
- **Early stopping logic**: `src/model/lstm_simple.rs` (lines ~1020-1040)
- **Validation splits**: `src/model/lstm_simple.rs` (lines ~790-830)
- **Chronological validation**: `src/model/multi_target.rs::train_with_chronological_validation()`

#### Price Level Target Issues (RECENTLY FIXED)
- **Class imbalance**: `src/targets/price_levels.rs::analyze_class_distribution()`
- **Quantile consistency**: `src/targets/price_levels.rs::calculate_price_level_targets()`
- **Class weighting**: `src/model/lstm_simple.rs::calculate_class_weights()`
- **Label smoothing**: `src/model/lstm_simple.rs::apply_label_smoothing()`

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

# Available configurations (20+ configs)
ls configs/
ls configs/optimizer_examples/
```

### Key Files for Common Tasks
- **Training issues**: `src/model/lstm_simple.rs`
- **Multi-target problems**: `src/model/multi_target.rs`
- **Feature engineering**: `src/features/technical.rs`, `src/features/cross_asset.rs`
- **Configuration**: `src/config/training.rs`, `configs/*.toml`
- **Data loading**: `src/data/loader.rs`
- **Error handling**: `src/utils/error.rs`
- **Tensor operations**: `src/model/loss.rs`, `src/model/attention.rs`
- **Price level targets**: `src/targets/price_levels.rs`

### Configuration Hierarchy
1. **Default values**: In `Default` trait implementations
2. **TOML files**: `configs/*.toml` templates (20+ available)
3. **Command line**: Override via CLI arguments
4. **Runtime**: Dynamic adjustments during training

### Recent Critical Fixes (Memorized)
- **Tensor Broadcasting**: Fixed `[16,6] × [1,6]` multiplication using `broadcast_as()`
- **Price Level Targets**: Implemented class weighting and label smoothing
- **Global Quantiles**: Fixed quantile calculation consistency
- **Categorical Metrics**: Added accuracy, precision, recall, F1 scores

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
