# VANGA LSTM Development Instructions & Onboarding

## 🎯 System Overview

**VANGA** is a cryptocurrency forecasting system using LSTM neural networks with:
- **5-Target Classification**: Price Levels, Direction, Volatility, Volume, Sentiment
- **Per-Sequence Processing**: Each sequence normalized independently
- **Adaptive Calibration**: Optimal parameters for balanced 20% per class distribution
- **Multi-Model Architecture**: Separate LSTM per target×horizon combination
- **Real-time Prediction**: Streaming WebSocket integration with live market data

## 🏗️ Core Architecture Principles

### Sequence-First Design
- **No Global Normalization**: Each sequence (e.g., 60 timesteps) normalized using only its own data
- **Sequence-Based Targets**: Each sequence generates ONE target based on horizon analysis
- **Chronological Integrity**: Time-series preserved, no shuffling
- **Symbol-Agnostic**: Percentage-based calculations work for any trading pair

### Calibration-Driven Classification
- **Adaptive Parameters**: System finds optimal thresholds for balanced class distribution
- **Training-Prediction Consistency**: Same calibrated parameters used in both phases
- **5-Class System**: All targets use `NUM_CLASSES = 5` (0-4) for LSTM compatibility
- **FULLY BALANCED TRAINING**: All datasets are PERFECTLY balanced (exactly 20% per class) for optimal training

### Multi-Model Coordination
- **Individual Models**: Each target×horizon = separate LSTM model
- **Wrapper Architecture**: `MultiTargetLSTMModel` coordinates multiple individual models
- **Shared Input**: Same normalized sequences fed to all models
- **Different Targets**: Each model trained on different target classification

### Code Quality Standards
- **Zero warnings**: All code must pass `cargo clippy --all-features --all-targets -- -D warnings`
- **No hidden variables**: Never use `_variable` to silence warnings - fix the root cause
- **No dead code**: Don't use `#[allow(dead_code)]` - remove unused code or fix the issue
- **DRY principle**: Don't repeat yourself - extract common logic into shared functions
- **Tensor safety**: Always use `broadcast_as()` for shape matching, ensure `.contiguous()` for operations
- **Test organization**: ALL tests must be in separate `*_test.rs` files - NEVER inline `#[cfg(test)]` modules

## 📊 Complete System Architecture

### A) Training Pipeline Flow
```
Raw CSV Data (OHLCV + Volume)
    ↓
Feature Engineering (50+ Technical Indicators) → src/features/
    ↓
NaN Removal & Outlier Handling → src/data/preprocessor.rs
    ↓
Sequence Generation (Sliding Windows: 30-120 timesteps) → src/data/sequence.rs
    ↓
Per-Sequence Normalization (Each sequence: mean=0, std=1)
    ↓
Calibration System (Find optimal adaptive parameters) → src/targets/calibration.rs
    ↓
Target Generation (5-class classification per sequence) → src/targets/
    ↓
Chronological Splitting (Train/Validation/Test by time) → src/data/loader.rs
    ↓
Multi-Model Training (Separate LSTM per target×horizon) → src/model/multi_target.rs
    ↓
Model Persistence (Save trained models + calibration params)
```

### B) Prediction Pipeline Flow
```
Raw CSV Data (Recent OHLCV)
    ↓
Feature Engineering (Same 50+ indicators) → src/features/
    ↓
NaN Removal & Outlier Handling → src/data/preprocessor.rs
    ↓
Sequence Generation (Latest sequences) → src/data/sequence.rs
    ↓
Per-Sequence Normalization (Using saved training stats)
    ↓
Multi-Model Prediction → src/model/multi_target.rs
    ↓
Raw LSTM Outputs (Array2<f64>)
    ↓
Output Parsing & Reconstruction → src/output/multi_target_parser.rs
    ↓
Structured Predictions → src/output/formatter.rs
    ↓
Post-Processing & Confidence Filtering → src/output/post_processor.rs
    ↓
Final Predictions with Confidence Scores
```

### C) Real-time Streaming Flow
```
Live Market Data (WebSocket/CSV Stream)
    ↓
Feature Buffer Management → src/realtime/predictor.rs
    ↓
Sliding Window Updates (Maintain sequence length)
    ↓
Continuous Prediction Pipeline (B above)
    ↓
Real-time Predictions
    ↓
Output Streaming (JSON/CSV formats)
```

### Key Processing Steps

#### 1. Feature Engineering (`src/features/`)
- **Technical Indicators**: 50+ crypto-optimized indicators (SMA, EMA, RSI, MACD, etc.)
- **Cross-Asset Features**: BTC dominance, ETH/BTC ratio, market correlation
- **Market Microstructure**: Price velocity, VWAP deviations, trade intensity
- **Output**: Enhanced DataFrame with engineered features

#### 2. Data Preprocessing (`src/data/preprocessor.rs`)
- **NaN Removal**: Critical step to remove lag feature warmup period
- **Outlier Handling**: IQR and Z-score methods for data cleaning
- **Feature Processing**: Apply engineering without global normalization
- **Output**: Clean DataFrame ready for sequence generation

#### 3. Sequence Generation (`src/data/sequence.rs`)
- **Sliding Windows**: Extract sequences of configurable length (30-120 timesteps)
- **Per-Sequence Normalization**: Each sequence normalized independently
- **No Global Stats**: Each sequence self-contained for symbol-agnostic operation
- **Output**: `Array3<f64>` [batch_size, sequence_length, features]

#### 4. Calibration System (`src/targets/calibration.rs`)
- **Purpose**: Find optimal adaptive parameters for balanced class distribution
- **Process**: Analyze entire dataset to find "sweet spot" thresholds
- **Target**: 20% per class distribution across all 5 targets
- **Output**: `CalibratedParameters` with optimized thresholds
- **CRITICAL**: All training datasets are PERFECTLY balanced (exactly 20% per class, validated by `validate_perfect_balance()`)

#### 5. Target Generation (`src/targets/`)
- **5 Target Types**: Price Levels, Direction, Volatility, Volume, Sentiment
- **5-Class System**: Each target classified into classes 0-4
- **Sequence-Based**: Each sequence generates one target based on horizon analysis
- **Adaptive Parameters**: Use calibrated thresholds for consistent classification
- **Perfect Balance**: Training data is perfectly balanced across all classes for optimal learning

#### 6. Multi-Model Training (`src/model/multi_target.rs`)
- **Individual Models**: Each target×horizon = separate LSTM model
- **Coordination**: `MultiTargetLSTMModel` wrapper manages all models
- **Shared Input**: Same normalized sequences for all models
- **Different Targets**: Each model trained on specific target classification

#### 7. Prediction Processing (`src/output/`)
- **Raw Output Parsing**: Convert LSTM Array2<f64> to structured predictions
- **Reconstruction**: Apply target-specific reconstruction functions
- **Confidence Calculation**: Multi-target agreement and uncertainty quantification
- **Post-Processing**: Filtering, smoothing, and regime adjustments
- **CRITICAL CONFIDENCE MAPPING**: 5-class predictions mapped to real-world confidence using `calibrate_5_class_confidence()`
  - **Baseline**: 0.2 (20% = random guessing for 5 classes)
  - **Good Model**: 0.25-0.35 max probability → 0.42-0.68 confidence
  - **Excellent Model**: 0.35-0.42 max probability → 0.68-0.78 confidence
  - **Exceptional**: 0.42+ max probability → 0.78+ confidence (rare in well-calibrated models)

#### 8. Prediction Processing (`src/output/`)
- **Raw Output Parsing**: Convert LSTM Array2<f64> to structured predictions
- **Reconstruction**: Apply target-specific reconstruction functions
- **Confidence Calculation**: Multi-target agreement and uncertainty quantification
- **Post-Processing**: Filtering, smoothing, and regime adjustments

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
│   │   └── mod.rs         # Public API and re-exports
│   ├── lstm_simple.rs # Compatibility layer: `pub use crate::model::lstm::*;`
│   ├── multi_target.rs # Multi-target wrapper
│   ├── attention.rs   # Multi-head attention mechanisms
│   ├── xgboost.rs     # XGBoost integration for hybrid models
│   └── bias_correction.rs # Bias correction and calibration
├── features/      # Feature engineering
│   ├── technical.rs   # 50+ technical indicators
│   ├── cross_asset.rs # Cross-asset features
│   ├── microstructure.rs # Market microstructure features
│   └── engineering.rs # Feature engineering pipeline
├── data/          # Data loading and preprocessing
│   ├── loader.rs      # CSV loading and validation
│   ├── preprocessor.rs # Feature normalization and cleaning
│   ├── sequence.rs    # Sequence generation with per-sequence normalization
│   ├── schema.rs      # Data schema definitions
│   ├── structures.rs  # Data structures
│   └── target_converter.rs # Target conversion utilities
├── targets/       # Target generation (5 targets × 5 classes)
│   ├── mod.rs         # Target orchestration
│   ├── calibration.rs # Adaptive parameter optimization
│   ├── price_levels.rs # VWAP-weighted range analysis
│   ├── direction.rs   # Directional movement classification
│   ├── volatility.rs  # Volatility regime classification
│   ├── volume.rs      # Volume activity classification
│   ├── sentiment.rs   # Market sentiment classification
│   └── sequence_reconstruction.rs # Unified reconstruction logic
├── output/        # Prediction processing & formatting
│   ├── multi_target_parser.rs # Raw LSTM output parsing
│   ├── formatter.rs   # Structured prediction formatting
│   ├── post_processor.rs # Confidence filtering and smoothing
│   ├── confidence_calculator.rs # Multi-target agreement
│   └── structures.rs  # Prediction result types
├── realtime/      # Real-time streaming prediction
│   ├── predictor.rs   # Streaming prediction engine
│   ├── stream.rs      # CSV streaming and parsing
│   └── watcher.rs     # File watching for new data
├── config/        # Configuration management
│   ├── training.rs    # TrainingConfig, TrainingParams, 9 optimizers
│   ├── prediction.rs  # PredictionConfig, OutputConfig
│   ├── features.rs    # Feature configurations
│   ├── model.rs       # Model architecture configurations
│   └── mod.rs         # Configuration coordination
├── optimization/  # Optimization and feature selection
│   ├── feature_selection.rs # Feature selection algorithms
│   ├── hyperparameter.rs # Hyperparameter optimization
│   └── fractional.rs # Fractional optimization methods
└── utils/         # Utilities and error handling
    ├── error.rs       # VangaError types and handling
    ├── metrics.rs     # Evaluation metrics
    └── device.rs      # Device management (CPU/GPU/Metal)
```

## 🔄 CRITICAL: Training vs Prediction Data Flow

### Training Pipeline Architecture
```
Raw CSV → Feature Engineering → NaN Removal → Outlier Handling → Sequence Creation → Target Generation → Calibration → Multi-Model Training → Model Persistence
    ↓           ↓                    ↓             ↓                ↓                  ↓                ↓             ↓                    ↓
OHLCV Data  Technical Indicators  Clean Data   Processed Data   Sequences         5×Targets        Parameters   N×LSTMModel         Saved Models
```

### Prediction Pipeline Architecture
```
Raw CSV → Feature Engineering → NaN Removal → Outlier Handling → Sequence Creation → Multi-Model Prediction → Output Parsing → Reconstruction → Post-Processing → Final Predictions
    ↓           ↓                    ↓             ↓                ↓                  ↓                    ↓             ↓               ↓                ↓
OHLCV Data  Technical Indicators  Clean Data   Processed Data   Sequences         Raw Predictions      Structured    Formatted       Filtered         Confidence Scores
```

### Real-time Pipeline Architecture
```
Live Data Stream → Feature Buffer → Sliding Window → Prediction Pipeline → Prediction Stream → Output Formats
       ↓               ↓              ↓                    ↓                  ↓                 ↓
   WebSocket/CSV   Circular Buffer  Latest Sequence    Same as Above      Live Predictions  JSON/CSV/API
```

### Key Data Flow Details
- **No Global Normalization**: Uses per-sequence processing approach
- **Feature Engineering**: Applied before any other processing
- **NaN Removal**: Critical step to remove lag feature warmup period
- **Target Independence**: Each target type calculated independently from sequences
- **Multi-Model Coordination**: MultiTargetLSTMModel manages separate models per target×horizon
- **Calibration Consistency**: Same adaptive parameters used in training and prediction
- **Output Processing**: Raw LSTM outputs converted to structured predictions with confidence scores

### ⚠️ CRITICAL ARCHITECTURE REQUIREMENTS

#### Perfect Balance Training System
- **MANDATORY BALANCE**: All training datasets are PERFECTLY balanced (exactly 20% per class)
- **Balance Validation**: `validate_perfect_balance()` enforces strict 20% distribution
- **Global Balance**: `src/data/balance.rs` provides sophisticated balanced sequence selection
- **No Imbalance**: System rejects any training data that isn't perfectly balanced
- **Why Critical**: Prevents model bias toward dominant classes, ensures fair learning

#### Multi-Model Architecture
- **Single LSTMModel Limitation**: Each `LSTMModel` handles only ONE target (5 categorical outputs)
- **MultiTargetLSTMModel Solution**: Wraps multiple `LSTMModel` instances (one per target×horizon)
- **Example**: 5 targets × 2 horizons = 10 separate `LSTMModel` instances
- **Training Coordination**: `TrainingContext` manages training across all models
- **Prediction Aggregation**: Combines predictions from all individual models

#### Real-World Confidence Mapping (CRITICAL)
- **5-Class Challenge**: Raw LSTM probabilities don't directly translate to trading confidence
- **Confidence Calibration**: `calibrate_5_class_confidence()` maps model outputs to real-world confidence
- **Conservative Approach**: Recognizes that 0.3 max probability in 5-class is actually very good
- **Trading Reality**: Prevents overconfidence that leads to excessive risk-taking
- **Mathematical Basis**: Uses entropy, deviation from uniform, and Gini coefficient

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

#### `src/api/` - HIGH-LEVEL ORCHESTRATION
- **trainer.rs**: `pub async fn train_multi_target()` - Complete training pipeline orchestration
- **predictor.rs**: `pub async fn predict()` - Unified prediction pipeline for single/multi-target models
- **backtester.rs**: Backtesting framework with walk-forward validation

#### `src/model/lstm/` - SINGLE LSTM MODEL (Core Implementation)
- **training.rs**: `pub async fn train(&mut self, sequences: &Array3<f64>, targets: &Array2<f64>, config: &TrainingConfig, val_sequences: Option<&Array3<f64>>, val_targets: Option<&Array2<f64>>) -> Result<()>` - THE unified training method
- **config.rs**: `LSTMConfig`, `OptimizerWrapper` (9 optimizers), `TargetFormat` - Single model configuration
- **core.rs**: Model lifecycle, initialization, persistence, Xavier initialization
- **inference.rs**: `predict()` method - Single model prediction
- **loss.rs**: Loss calculation with weighted cross-entropy for single target
- **Limitation**: Can only handle ONE target at a time (hence the wrapper)

#### `src/model/multi_target.rs` - MULTI-LSTM WRAPPER
- **Purpose**: Wraps multiple `LSTMModel` instances to overcome single-target limitation
- **Architecture**: Creates separate `LSTMModel` for each target×horizon combination
- **Example**: 5 targets × 2 horizons = 10 separate `LSTMModel` instances
- **Training**: `TrainingContext` coordinates training across all models
- **Prediction**: Aggregates predictions from all individual models

#### `src/model/lstm_simple.rs` - COMPATIBILITY LAYER
- **Implementation**: `pub use crate::model::lstm::*;` - Pure re-export
- **Purpose**: Maintains backward compatibility for existing code

#### `src/targets/` - TARGET GENERATION (5 Targets × 5 Classes Each)
- **Architecture**: 5 independent target types, each with 5 categorical outputs
- **price_levels.rs**: VWAP-weighted range analysis (5-class: Strong Down, Moderate Down, Neutral, Moderate Up, Strong Up)
- **direction.rs**: Directional movement classification (5-class categorical)
- **volatility.rs**: Volatility regime classification (5-class categorical)
- **volume.rs**: Volume activity classification (5-class categorical)
- **sentiment.rs**: Market sentiment classification (5-class categorical)
- **calibration.rs**: Adaptive parameter optimization for balanced class distribution
- **mod.rs**: `TargetGenerator` orchestrates all 5 target types
- **Output Structure**: `NUM_CLASSES = 5` for each target type
- **Sequence-based**: All targets calculated from sequence data, independent of market/regime
- **Total Output**: 5 targets × 5 classes = 25 total outputs per prediction

#### `src/output/` - PREDICTION PROCESSING & FORMATTING
- **multi_target_parser.rs**: Parse raw LSTM Array2<f64> outputs into structured predictions
- **formatter.rs**: `OutputFormatter` - Convert raw predictions to structured formats using reconstruction functions
- **post_processor.rs**: `PostProcessor` - Apply confidence filtering, smoothing, outlier removal
- **confidence_calculator.rs**: Multi-target agreement and uncertainty quantification
- **structures.rs**: Prediction result types and data structures

#### `src/realtime/` - STREAMING PREDICTION
- **predictor.rs**: `StreamingPredictor` - Real-time prediction with feature buffer management
- **stream.rs**: CSV streaming and incremental parsing
- **watcher.rs**: File watching for new data detection

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

#### `src/data/sequence.rs` - SEQUENCE GENERATION
- **generate_training_sequences()**: Create training sequences with targets
- **generate_prediction_sequences()**: Create prediction sequences without targets
- **Per-sequence normalization**: Each sequence normalized independently
- **Sliding window**: Configurable overlap and sequence length

#### `src/output/confidence_calculator.rs` - CONFIDENCE MAPPING
- **calibrate_5_class_confidence()**: Maps raw LSTM probabilities to real-world confidence
- **Conservative Mapping**: Prevents overconfidence in crypto trading
- **Mathematical Foundation**: Entropy, deviation from uniform, Gini coefficient
- **Trading-Optimized**: Recognizes that 0.3 max probability is actually very good in 5-class system

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

## 🏗️ Project Structure

```
src/
├── api/                    # High-level training/prediction APIs
│   ├── trainer.rs             # Training pipeline orchestration
│   ├── predictor.rs           # Prediction pipeline orchestration
│   └── backtester.rs          # Backtesting framework
├── config/                 # Configuration management
│   ├── training.rs            # TrainingConfig, 9 optimizers
│   ├── model.rs               # ModelConfig, NUM_CLASSES = 5
│   ├── features.rs            # FeatureConfig
│   └── prediction.rs          # PredictionConfig
├── data/                   # Data processing pipeline
│   ├── loader.rs              # CSV loading + chronological splitting
│   ├── sequence.rs            # Sequence generation + per-sequence normalization
│   ├── preprocessor.rs        # Feature engineering pipeline
│   ├── structures.rs          # Data structures (MarketDataRow, etc.)
│   └── target_converter.rs    # Target format conversion
├── features/               # Feature engineering
│   ├── technical.rs           # 50+ technical indicators
│   └── cross_asset.rs         # Cross-asset correlation features
├── model/                  # LSTM implementations
│   ├── lstm/                  # Modular LSTM implementation
│   │   ├── config.rs             # LSTMConfig, OptimizerWrapper
│   │   ├── core.rs               # Model lifecycle, initialization
│   │   ├── training.rs           # Unified training method
│   │   ├── inference.rs          # Prediction and forward pass
│   │   └── loss.rs               # Loss calculation, tensor broadcasting
│   ├── lstm_simple.rs         # Compatibility layer (re-exports)
│   ├── multi_target.rs        # Multi-model wrapper coordination
│   ├── attention.rs           # Multi-head attention mechanisms
│   └── tft/                   # Temporal Fusion Transformer
├── targets/                # Target generation (5-class system)
│   ├── calibration.rs         # Adaptive parameter calibration
│   ├── price_levels.rs        # Price movement classification
│   ├── direction.rs           # Trend direction detection
│   ├── volatility.rs          # Volatility regime classification
│   ├── volume.rs              # Volume activity classification
│   ├── sentiment.rs           # Market sentiment classification
│   └── sequence_reconstruction.rs # Unified reconstruction logic
├── output/                 # Output processing & formatting
│   ├── multi_target_parser.rs # Prediction array segmentation
│   ├── formatter.rs           # Reconstruction function application
│   └── structures.rs          # Prediction result types
├── optimization/           # Auto-optimization system
│   └── feature_selection.rs  # Feature selection algorithms
├── realtime/               # Real-time streaming prediction
│   └── websocket.rs           # WebSocket integration
└── utils/                  # Utilities and error handling
    ├── error.rs               # VangaError types
    └── metrics.rs             # Evaluation metrics
```

### Configuration Files (`configs/`)
- **`training.toml`**: Main training configuration with all 9 optimizers
- **`prediction.toml`**: Prediction pipeline configuration
- **`realtime.toml`**: Real-time streaming settings
- **`optimizer_examples/`**: 9 optimizer-specific configurations
- **`quick_start.toml`**: Beginner-friendly minimal setup
- **30+ specialized configs**: Various training scenarios and optimizations

## 🚀 Entry Points & Command Interface

### Main Entry Points (`src/main.rs`)

#### Training Commands
```bash
# Single symbol training
cargo run -- train --symbol BTCUSDT --data data.csv --config configs/training.toml

# Multi-symbol batch training
cargo run -- train --symbol BTCUSDT,ETHUSDT,DOTUSDT --data data_directory/ --batch

# Continue training existing model
cargo run -- train --symbol BTCUSDT --data new_data.csv --continue-training

# XGBoost-only training (requires existing LSTM model)
cargo run -- train --symbol BTCUSDT --data data.csv --xgboost-only
```

#### Prediction Commands
```bash
# Single prediction
cargo run -- predict --symbol BTCUSDT --input recent_data.csv --horizon 1h

# All horizons prediction
cargo run -- predict --symbol BTCUSDT --input data.csv --all-horizons

# Real-time streaming prediction
cargo run -- predict --symbol BTCUSDT --input data.csv --realtime --interval 1m

# Batch prediction with confidence filtering
cargo run -- predict --symbol BTCUSDT --input data_dir/ --batch --min-confidence 0.7
```

#### Model Management Commands
```bash
# List available models and their horizons
cargo run -- models list

# Model information and statistics
cargo run -- models info --symbol BTCUSDT

# Validate model integrity
cargo run -- models validate --symbol BTCUSDT
```

### API Entry Points (`src/api/`)

#### High-Level Training API
- **`api::trainer::train_multi_target()`**: Complete training pipeline orchestration
- **`api::trainer::continue_training()`**: Continue training existing models
- **`api::trainer::train_xgboost_only()`**: XGBoost-only training phase

#### High-Level Prediction API
- **`api::predictor::predict()`**: Unified prediction pipeline
- **`api::predictor::predict_single_target()`**: Single target prediction
- **`api::predictor::predict_multi_target()`**: Multi-target prediction

#### Backtesting API
- **`api::backtester::run_backtest()`**: Walk-forward backtesting
- **`api::backtester::evaluate_performance()`**: Performance evaluation

## 🔧 Development Workflow
```bash
# Fast development cycle (PREFERRED)
cargo check --message-format=short

# Code quality enforcement (MANDATORY before commits)
cargo clippy --all-features --all-targets -- -D warnings

# Run tests
cargo test

# Training example
cargo run -- train --symbol BTCUSDT --data data.csv --config configs/training.toml

# Prediction example
cargo run -- predict --symbol BTCUSDT --input recent_data.csv --output predictions.json

# Real-time streaming
cargo run -- predict --symbol BTCUSDT --input data.csv --realtime --interval 1m
```

### Understanding Problems (Debugging Checklist)
1. **Training Issues**: Check `src/api/trainer.rs` → `src/model/multi_target.rs` → `src/model/lstm/training.rs`
2. **Prediction Issues**: Check `src/api/predictor.rs` → `src/output/formatter.rs` → `src/output/post_processor.rs`
3. **Real-time Issues**: Check `src/realtime/predictor.rs` → `src/realtime/stream.rs`
5. **Sequence Context**: What sequence length/horizon is involved?
6. **Normalization**: Is per-sequence normalization working correctly?
7. **Calibration**: Are adaptive parameters being used consistently?
8. **Target Classification**: Are all 5 targets using same classification logic?
9. **Multi-Model**: Is each target×horizon handled by separate LSTM?
10. **Tensor Broadcasting**: Are shapes compatible with `broadcast_as()`?

### Making Changes (Development Rules)
1. **Sequence-First Thinking**: How does this affect sequence processing?
2. **Calibration Impact**: Do adaptive parameters need recalibration?
3. **Training-Prediction Consistency**: Same logic in both phases?
4. **Multi-Model Coordination**: Does wrapper handle all combinations?
5. **Output Processing**: How does this affect prediction formatting?
6. **Real-time Impact**: Does this affect streaming predictions?
7. **Test with Real Data**: Verify with actual sequences, not synthetic data

### Critical Implementation Patterns

#### Per-Sequence Normalization
```rust
// Each sequence normalized independently
fn normalize_sequence_window(&self, window_df: &DataFrame) -> Result<DataFrame> {
    // Calculate mean/std from THIS WINDOW ONLY
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();
    // Normalize: (x - mean) / std using only window data
}
```

#### Calibration Usage
```rust
// Training: Find optimal parameters
let calibrated_params = calibrator.calibrate(ohlcv_data, sequence_length, horizon_steps).await?;

// Apply consistently in training and prediction
let target_class = classify_with_params(sequence, &calibrated_params);
```

#### Multi-Model Coordination
```rust
// Each target×horizon = separate LSTM model
for target_type in ["price_levels", "direction", "volatility", "volume", "sentiment"] {
    for horizon in ["1h", "4h", "1d"] {
        let model = LSTMModel::new(config);
        multi_target_model.add_model(target_type, horizon, model);
    }
}
```

#### Prediction Processing
```rust
// Raw LSTM outputs → Structured predictions
let raw_predictions = multi_target_model.predict(&sequences).await?;
let parsed_predictions = parser.parse_multi_target_output(&raw_predictions)?;
let formatted_predictions = formatter.format_predictions(parsed_predictions)?;
let filtered_predictions = post_processor.process(formatted_predictions)?;
```

### ⚠️ Critical Rules (NEVER BREAK)

1. **Sequence-Based Processing**: All operations work on sequences, not individual data points
2. **Per-Sequence Normalization**: Each sequence normalized using only its own data
3. **PERFECT BALANCE REQUIREMENT**: All training datasets must be exactly 20% per class (validated by `validate_perfect_balance()`)
4. **Adaptive Parameter Consistency**: Same calibrated parameters in training and prediction
5. **Chronological Integrity**: Never shuffle time-series data
6. **Multi-Model Architecture**: Each target×horizon requires separate LSTM model
7. **Tensor Broadcasting**: Always use `broadcast_as()` for shape compatibility
8. **Test Separation**: ALL tests in separate `*_test.rs` files
9. **Output Processing**: Raw LSTM outputs must be parsed, formatted, and post-processed
10. **Confidence Mapping**: Use `calibrate_5_class_confidence()` for real-world confidence translation
10. **Real-time Consistency**: Streaming predictions use same pipeline as batch predictions

### 🚫 Anti-Patterns (AVOID)

- **Global Normalization**: Using dataset-wide statistics for normalization
- **Imbalanced Training**: Using datasets that aren't perfectly balanced (20% per class)
- **Raw Confidence Usage**: Using LSTM probabilities directly without confidence calibration
- **Method Proliferation**: Creating `train_a`, `train_b` instead of enhancing existing methods
- **Hardcoded Parameters**: Not using adaptive/calibrated parameters
- **Data Shuffling**: Breaking chronological sequence in time-series data
- **Inline Tests**: Using `#[cfg(test)]` modules within source files
- **Hidden Variables**: Using `_variable` to silence warnings instead of fixing issues
- **Raw Output Usage**: Using LSTM outputs directly without parsing/formatting
- **Single-Target Thinking**: Ignoring multi-target consensus in decision making
- **Batch-Only Design**: Not considering real-time streaming requirements
- **Overconfident Trading**: Not using conservative confidence mapping for crypto markets

## 🎯 Architecture Mastery Checklist

After reading this document, you should understand:

✅ **Complete Data Flow**: Raw CSV → Features → Sequences → Normalization → Calibration → Targets → Training → Prediction → Confidence
✅ **Sequence Processing**: Each sequence self-contained, normalized independently
✅ **PERFECT BALANCE TRAINING**: All datasets exactly 20% per class, validated by `validate_perfect_balance()`
✅ **Calibration System**: How adaptive parameters achieve balanced classification
✅ **5-Target Architecture**: All targets use unified 5-class classification (0-4)
✅ **Multi-Model Reality**: Each target×horizon = separate LSTM model
✅ **Output Processing**: Raw predictions → Parsing → Formatting → Post-processing → Confidence Scores
✅ **CONFIDENCE MAPPING**: How 5-class probabilities map to real-world trading confidence
✅ **Real-time Integration**: Streaming predictions with feature buffer management
✅ **Real-time Integration**: Streaming predictions with feature buffer management
✅ **Entry Points**: CLI commands and API entry points for all operations
✅ **File Organization**: Where to find specific functionality
✅ **Development Rules**: Critical patterns and anti-patterns

**Ready to develop? Start with `cargo check` and explore the codebase!**

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
