# VANGA Configuration Reference

## 📋 **Current Configuration System Overview**

VANGA uses a **unified configuration system** where all parameters (training, model, features, data processing) are defined in TOML files and CLI arguments. This approach ensures consistency, simplifies management, and provides comprehensive parameter documentation.

### **Configuration Philosophy**
- **All-in-One**: Single file contains all parameters
- **Template-Based**: Pre-configured templates for common use cases
- **Self-Documenting**: Comprehensive parameter explanations in example files
- **Validated**: Automatic parameter validation with clear error messages

## 🚀 **NEW: Advanced Learning Rate Optimization**

VANGA now features **professional-grade learning rate optimization** with modern optimizers and intelligent scheduling:

### **Modern Optimizers**
- **AdamW**: Modern optimizer with weight decay (RECOMMENDED for crypto - handles volatility well)
- **Adam**: Classic adaptive optimizer, good general-purpose choice
- **SGD**: Traditional optimizer with optional momentum, good for fine-tuning
- **AdaDelta**: Adaptive without manual tuning, excellent for sparse crypto data
- **AdaGrad**: Accumulates gradients, good for sparse features but can slow down
- **AdaMax**: More stable than Adam for problems with large gradients (crypto spikes)
- **NAdam**: Nesterov-accelerated Adam, often converges faster than standard Adam
- **RAdam**: Rectified Adam with variance correction, more stable early training
- **RMSprop**: Excellent for RNNs and non-stationary objectives (perfect for crypto markets)

### **Intelligent Learning Rate Modes**
- **Auto**: Optimizes learning rate within specified ranges based on model complexity
- **Adaptive**: ReduceLROnPlateau with configurable patience and reduction factor
- **Fixed**: Constant learning rate for fine-tuning and controlled training

### **Warmup Support**
- **Linear warmup** from 0 to target learning rate over specified epochs
- **Prevents early training instability** with large models
- **Configurable warmup duration** (0-20 epochs recommended)

### **Enhanced Training Configuration**
```toml
[training]
# Modern optimizer with adaptive learning (RECOMMENDED for crypto)
optimizer = { AdamW = { weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999 } }

# Alternative optimizers for different crypto scenarios:
# optimizer = { Adam = { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01, amsgrad = false } }
# optimizer = { NAdam = { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01, momentum_decay = 0.004 } }
# optimizer = { RAdam = { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01 } }
# optimizer = { RMSprop = { alpha = 0.99, eps = 1e-8, weight_decay = 0.01, momentum = 0.0, centered = false } }

# Traditional SGD for fine-tuning
# optimizer = { SGD = {} }  # Basic SGD
# optimizer = { SGD = { momentum = 0.9 } }  # SGD with momentum
# learning_rate = { Fixed = 0.001 }
# warmup_epochs = 0

# Unified training handles all scenarios through configuration
# - Validation split automatically enables early stopping
# - Warmup epochs provide gradual LR increase
# - Adaptive LR reduces when validation loss plateaus
```

---

## 🗂️ **Configuration Templates**

### **Available Templates**

| Template | Purpose | Use Case | Features |
|----------|---------|----------|----------|
| `configs/quick_start.toml` | Beginner setup | Learning, small datasets | Minimal but effective |
| `configs/training.toml` | Production single-asset | Standard crypto trading | Full feature set |
| `configs/cross_asset_training.toml` | Multi-asset production | Portfolio management | Cross-asset correlations |
| `configs/minimal_custom.toml` | Simple customization | Basic custom features | Lightweight custom setup |
| `configs/advanced_custom.toml` | Complex customization | Advanced feature engineering | Comprehensive features |
| `configs/example_single_asset.toml` | Complete reference | Parameter documentation | All parameters explained |
| `configs/example_cross_asset.toml` | Cross-asset reference | Multi-asset documentation | Cross-asset specific guidance |

### **Template Selection Guide**

```bash
# Beginner: Start here
--config configs/quick_start.toml

# Standard: Production single-asset
--config configs/training.toml

# Advanced: Multi-asset with correlations
--config configs/cross_asset_training.toml

# Custom: Add your own features
--config configs/minimal_custom.toml  # Simple
--config configs/advanced_custom.toml  # Complex

# Reference: Complete parameter documentation
--config configs/example_single_asset.toml
--config configs/example_cross_asset.toml
```

---

## ⚙️ **Configuration Structure**

### **Main Sections**

```toml
[training]          # Training process control
[model]             # Neural network architecture
[features]          # Feature engineering pipeline
[data]              # Data processing and normalization
[optimization]      # Hyperparameter optimization (optional)
```

### **Section Dependencies**
- `[training]` → Core training parameters (required)
- `[model]` → Architecture definition (required)
- `[features]` → Feature engineering (required)
- `[data]` → Data preprocessing (optional, uses defaults)
- `[optimization]` → Hyperparameter tuning (optional)

---

## 🎯 **Training Configuration**

### **Core Training Parameters**

```toml
[training]
# Device configuration - hardware acceleration settings
device = "Auto"                               # Auto-detect best device (RECOMMENDED)
# device = "CPU"                             # Force CPU usage
# device = "GPU:0"                           # Use first NVIDIA CUDA GPU
# device = "Metal:0"                         # Use first Apple Silicon GPU (macOS)

# Epoch configuration - controls training duration
epochs = { Auto = { max_epochs = 1000 } }     # Auto early stopping (RECOMMENDED)
# epochs = { Fixed = 200 }                    # Fixed epoch count

# Learning rate - base learning rate for optimization
learning_rate = 0.001                         # Base learning rate (0.0001-0.01 range)

# Modern optimizer configuration (9 optimizers available)
optimizer = { AdamW = { weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } }
# Alternative optimizers:
# optimizer = { RMSprop = { alpha = 0.99, eps = 1e-8, weight_decay = 0.01, momentum = 0.0 } }
# optimizer = { NAdam = { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01 } }
# optimizer = { RAdam = { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01 } }
# optimizer = { SGD = { momentum = 0.9 } }

# Learning rate warmup - gradual LR increase
warmup_epochs = 5                             # Warmup epochs (0-20 range)

# Learning rate scheduling - adaptive LR reduction
learning_schedule = { ReduceLROnPlateau = { factor = 0.5, patience = 10, min_lr = 1e-6 } }

# Batch size - controls memory usage and gradient stability
batch_size = { Auto = { min_size = 32, max_size = 512 } }  # Auto sizing (RECOMMENDED)
# batch_size = { Fixed = 64 }                 # Fixed batch size

# Data splits - controls validation and testing
validation_split = 0.2                        # 20% for validation (0.1-0.3 range)
validation_gap = "1h"                         # Gap to prevent data leakage
test_split = 0.1                              # 10% for testing (0.05-0.2 range)

# Early stopping - prevents overfitting
early_stopping = { patience = 50, min_delta = 0.0001 }   # Stop after 50 epochs without improvement

# Gradient clipping - prevents exploding gradients
gradient_clip = 1.0                           # Clipping threshold (0.5-2.0 range)

# Progress monitoring
print_every = 1                               # Print progress every N epochs

# Class weighting strategy for imbalanced datasets
class_weight_strategy = "Global"              # Global class weighting
# class_weight_strategy = "PerTarget"         # Per-target class weighting

# Window-based training parameters (for walk-forward training)
window_decay = 1.0                            # Learning rate decay per window (0.8-1.0 range)
min_train_ratio = 0.4                         # Minimum training data ratio (0.3-0.6 range)
min_increment_ratio = 0.3                     # Minimum increment ratio (0.2-0.5 range)

# Reproducible training
seed = 42                                     # Fixed seed for reproducibility
```

### **Parameter Tuning Guidelines**

#### **Device Configuration**
```toml
# Auto device selection (RECOMMENDED)
device = "Auto"
# EFFECT: Automatically selects best available device (CUDA > Metal > CPU)
# TUNING: Use for optimal performance across different hardware

# CPU device (compatibility)
device = "CPU"
# EFFECT: Forces CPU usage, slower but always available
# TUNING: Use for development, testing, or when GPU drivers unavailable

# NVIDIA CUDA GPU (high performance)
device = "GPU:0"
# EFFECT: Uses first NVIDIA CUDA GPU for acceleration
# TUNING: Requires CUDA drivers, best performance for NVIDIA GPUs

# Apple Silicon GPU (macOS)
device = "Metal:0"
# EFFECT: Uses first Apple Silicon GPU for acceleration
# TUNING: macOS only, best performance for M1/M2/M3 Macs
```

#### **Epochs Configuration**
```toml
# Auto early stopping (RECOMMENDED for production)
epochs = { Auto = { max_epochs = 1000 } }
# EFFECT: Stops when validation loss plateaus, prevents overfitting
# TUNING: Increase max_epochs (1500-2000) for complex models

# Fixed epochs (for reproducible experiments)
epochs = { Fixed = 200 }
# EFFECT: Runs exact number of epochs regardless of performance
# TUNING: Use for research, debugging, or when you know optimal epoch count
```

#### **Learning Rate Configuration**
```toml
# Base learning rate (simple and direct)
learning_rate = 0.001
# TUNING: 0.0001 (conservative), 0.001 (standard), 0.01 (aggressive)
# EFFECT: Higher = faster learning but risk instability

# Learning rate warmup (gradual increase)
warmup_epochs = 5
# TUNING: 0 (no warmup), 5-10 (standard), 15-20 (large models)
# EFFECT: Prevents early training instability with large models

# Learning rate scheduling (adaptive reduction)
learning_schedule = { ReduceLROnPlateau = { factor = 0.5, patience = 10, min_lr = 1e-6 } }
# TUNING: Reduce factor (0.2-0.3) for aggressive reduction, increase patience (15-20) for stability
# EFFECT: Automatically reduces when loss plateaus

# Window-aware learning rate decay (for walk-forward training)
window_decay = 0.95
# TUNING: 0.9 (aggressive decay), 0.95 (moderate), 1.0 (no decay)
# EFFECT: Reduces learning rate progressively across training windows
```

#### **Batch Size Configuration**
```toml
# Auto batch sizing (RECOMMENDED)
batch_size = { Auto = { min_size = 32, max_size = 512 } }
# TUNING: Increase min_size (64-128) for stable gradients, reduce max_size for memory constraints
# EFFECT: Automatically optimizes based on data size and available memory

# Fixed batch size (for consistent behavior)
batch_size = { Fixed = 64 }
# TUNING: 32 (small data/memory), 64 (standard), 128+ (large data/memory)
# EFFECT: Larger batches = more stable gradients but higher memory usage
```

#### **🆕 Advanced Training Parameters**

```toml
# Validation gap - prevents data leakage from features with lookback periods
validation_gap = "1h"
# TUNING: "0" (no gap), "1h" (standard), "2h" (conservative for features with long lookback)
# EFFECT: Creates temporal gap between training and validation to prevent information leakage

# Class weighting strategy - handles imbalanced datasets
class_weight_strategy = "Global"
# OPTIONS: "Global" (global class weights), "PerTarget" (per-target weights), "None" (no weighting)
# EFFECT: Balances training for imbalanced target classes

# Window-based training parameters (for walk-forward training)
min_train_ratio = 0.4
# TUNING: 0.3 (aggressive), 0.4 (standard), 0.6 (conservative)
# EFFECT: Minimum percentage of data to use for initial training window

min_increment_ratio = 0.3
# TUNING: 0.2 (small increments), 0.3 (standard), 0.5 (large increments)
# EFFECT: Minimum percentage increase per training window

# Reproducible training
seed = 42
# TUNING: Any integer (42, 123, 2024, etc.)
# EFFECT: Ensures deterministic training results for research and debugging

# Progress monitoring
print_every = 1
# TUNING: 1 (every epoch), 5 (every 5 epochs), 10 (every 10 epochs)
# EFFECT: Controls frequency of training progress output
```

#### **🤖 Optimizer Configuration**

```toml
# AdamW - Best overall performance (RECOMMENDED)
optimizer = { AdamW = { weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } }
# TUNING: weight_decay (0.001-0.1), beta1 (0.8-0.95), beta2 (0.99-0.999)
# BEST FOR: General cryptocurrency forecasting, handles volatility well

# RMSprop - Volatile markets specialist
optimizer = { RMSprop = { alpha = 0.99, eps = 1e-8, weight_decay = 0.01, momentum = 0.0 } }
# TUNING: alpha (0.9-0.99), momentum (0.0-0.9)
# BEST FOR: High volatility markets, meme coins, rapid price movements

# NAdam - Fastest convergence
optimizer = { NAdam = { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01 } }
# TUNING: Similar to Adam parameters
# BEST FOR: Development, quick experiments, fast convergence needed

# RAdam - Most stable
optimizer = { RAdam = { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01 } }
# TUNING: Conservative parameter changes recommended
# BEST FOR: Production environments, stable training required

# SGD - Fine-tuning specialist
optimizer = { SGD = { momentum = 0.9 } }
# TUNING: momentum (0.0-0.95)
# BEST FOR: Fine-tuning pre-trained models, transfer learning
```

---

## 🧠 **Model Configuration**

### **Architecture Definition**

```toml
[model]
# Architecture type - defines LSTM structure
architecture = { MultiLSTM = { layers = 2 } }  # Multi-layer with residual connections (RECOMMENDED)
# architecture = { StackedLSTM = { layers = 3 } }  # Traditional stacked layers
# architecture = { BidirectionalLSTM = { layers = 2 } }  # Bidirectional processing

# Sequence length - historical context window
sequence_length = { Auto = { min_length = 30, max_length = 120 } }  # Auto optimization (RECOMMENDED)
# sequence_length = { Fixed = 60 }  # Fixed sequence length

# Hidden units - model capacity
hidden_units = { Auto = { min_units = 64, max_units = 512 } }  # Auto sizing (RECOMMENDED)
# hidden_units = { Fixed = 128 }  # Fixed hidden units
```

### **Architecture Types**

#### **MultiLSTM (RECOMMENDED)**
```toml
architecture = { MultiLSTM = { layers = 2 } }
# FEATURES: Residual connections, layer normalization, efficient training
# BEST FOR: General cryptocurrency forecasting, most use cases
# TUNING: 1-2 layers (small data), 2-3 layers (medium data), 3-4 layers (large data)
```

#### **StackedLSTM**
```toml
architecture = { StackedLSTM = { layers = 3 } }
# FEATURES: Traditional stacked architecture, sequential processing
# BEST FOR: Sequential pattern recognition, time series with clear temporal structure
# TUNING: 2-3 layers typical, avoid >4 layers (vanishing gradients)
```

#### **BidirectionalLSTM**
```toml
architecture = { BidirectionalLSTM = { layers = 2 } }
# FEATURES: Processes sequences in both directions, captures future context
# BEST FOR: Pattern recognition, when future context is available
# TUNING: Use fewer layers (1-2) due to doubled parameters
```

### **Dropout Configuration**

```toml
[model.dropout]
enabled = true                                 # Enable dropout regularization (RECOMMENDED)
rate = { Fixed = 0.2 }                        # Dropout rate (0.1-0.5 range)
variational = true                             # Variational dropout (RECOMMENDED for LSTM)
recurrent = true                               # Recurrent dropout (prevents overfitting)
```

#### **Dropout Tuning**
- **Low dropout (0.1-0.2)**: Large datasets, stable training
- **Medium dropout (0.2-0.3)**: Standard setup, balanced regularization
- **High dropout (0.3-0.5)**: Small datasets, overfitting prevention

### **Attention Mechanism**

```toml
[model.attention]
enabled = true                                 # Enable attention (RECOMMENDED for complex patterns)
mechanism = "MultiHeadAttention"               # Attention type
# mechanism = "SelfAttention"                  # Simple self-attention
# mechanism = "MixtureOfHeads"                 # Advanced MoH attention

heads = 8                                      # Number of attention heads (4-16 range)
head_dim = 64                                  # Dimension per head (32-128 range)

# Enhanced dropout configurations
dropout_rate = 0.1                            # Base attention dropout
dropout_weights = true                         # Dropout on attention weights
dropout_output = true                          # Dropout on attention output
dropout_projections = true                     # Dropout on projections
dropout_scores = true                          # Dropout on attention scores

temperature_scaling = 1.0                     # Attention sharpness (0.5-2.0 range)
use_relative_position = true                   # Include position encoding

# Visualization settings (for debugging and analysis)
[model.attention.visualization]
enabled = false                                # Enable attention visualization
save_path = "attention_maps/"                  # Directory for attention maps
save_frequency = 10                            # Save every N epochs

# Mixture-of-Head specific configuration (when mechanism = "MixtureOfHeads")
[model.attention.moh]
enabled = false                                # Enable MoH attention
num_mixtures = 4                              # Number of attention mixtures (2-8 range)
mixture_dropout = 0.1                         # Dropout for mixture weights
```

#### **Attention Tuning**
- **Few heads (4-6)**: Simple patterns, faster training
- **Many heads (12-16)**: Complex patterns, cross-asset relationships
- **Small head_dim (32-48)**: Memory efficient, faster training
- **Large head_dim (64-128)**: More expressive, better for complex data

#### **🆕 Mixture-of-Head Attention**
- **Purpose**: Advanced attention mechanism with multiple attention mixtures
- **Benefits**: Better pattern recognition, improved model capacity
- **Use Cases**: Complex market patterns, multi-asset relationships
- **Tuning**: Start with 4 mixtures, increase for more complex patterns
- **Temperature scaling**: Lower (0.5-0.8) for sharper attention, higher (1.2-2.0) for softer

### **🤖 NEW: Hybrid Model Integration**

#### **XGBoost Integration**
```toml
[model.xgboost]
enabled = true                                 # Enable XGBoost hybrid model
phase = "Second"                               # Use XGBoost in second phase (after LSTM)
n_estimators = 100                             # Number of boosting rounds
max_depth = 6                                  # Maximum tree depth
learning_rate = 0.1                           # XGBoost learning rate
subsample = 0.8                               # Subsample ratio
colsample_bytree = 0.8                        # Feature subsample ratio

# Target-specific XGBoost configuration
[model.xgboost.targets]
price_levels = { objective = "multi:softprob", eval_metric = "mlogloss" }
direction = { objective = "binary:logistic", eval_metric = "logloss" }
volatility = { objective = "reg:squarederror", eval_metric = "rmse" }
```

#### **TFT (Temporal Fusion Transformer) Integration**
```toml
[model.tft]
enabled = true                                 # Enable TFT integration
variable_selection = true                      # Enable variable selection network
attention_heads = 4                           # Number of attention heads
hidden_size = 128                             # Hidden layer size
dropout = 0.1                                 # TFT dropout rate

# Quantile outputs for uncertainty estimation
[model.tft.quantile_outputs]
enabled = true                                 # Enable quantile regression
quantiles = [0.1, 0.5, 0.9]                  # Prediction quantiles
loss_weights = [0.3, 0.4, 0.3]               # Loss weights per quantile

# Variable selection configuration
[model.tft.variable_selection]
attention_mechanism = "MultiHeadAttention"     # Attention type
selection_threshold = 0.1                     # Variable importance threshold
max_variables = 50                            # Maximum selected variables
```

### **Output Heads Configuration**

```toml
[model.output_heads]

# Price level prediction (probability distribution)
[model.output_heads.price_levels]
enabled = true                                 # Enable price level prediction
bins = 10                                      # Number of price bins (5-20 range)
range_percent = 0.05                          # Price range (±5%)

# Direction prediction (up/down)
[model.output_heads.direction]
enabled = true                                 # Enable direction prediction
threshold = 0.01                              # Significance threshold (1%)
confidence_calibration = true                  # Calibrate confidence scores

# Volatility prediction (risk estimation)
[model.output_heads.volatility]
enabled = true                                 # Enable volatility prediction
method = "Direct"                             # Prediction method
horizons = ["1h", "4h", "1d"]                # Time horizons
```

---

## 🔧 **Features Configuration**

### **Technical Indicators**

```toml
[features.technical_indicators]
enabled = true                                 # Enable technical indicators (HIGHLY RECOMMENDED)

# Moving averages - trend following
[features.technical_indicators.moving_averages]
sma_periods = [5, 10, 20, 50, 200]           # Simple moving average periods
ema_periods = [5, 10, 20, 50, 200]           # Exponential moving average periods
wma_periods = [10, 20]                        # Weighted moving average periods
hull_periods = [9, 21]                        # Hull moving average periods

# Momentum indicators - trend strength
[features.technical_indicators.momentum]
rsi_periods = [14, 21]                        # RSI periods
stochastic = true                             # Stochastic oscillator
williams_r = true                             # Williams %R
cci_periods = [14, 20]                        # Commodity Channel Index periods
momentum_periods = [10, 20]                   # Price momentum periods

# Volatility indicators - market uncertainty
[features.technical_indicators.volatility]
bollinger_bands = { enabled = true, period = 20, std_dev = 2.0 }  # Bollinger Bands
atr_periods = [14, 21]                        # Average True Range periods
keltner_channels = true                       # Keltner Channels
donchian_channels = true                      # Donchian Channels

# Volume indicators - market participation
[features.technical_indicators.volume]
obv = true                                    # On-Balance Volume
volume_sma_periods = [10, 20]                # Volume moving averages
mfi_periods = [14]                           # Money Flow Index periods
volume_profile = true                         # Volume profile analysis
```

### **Market Microstructure Features**

```toml
[features.market_microstructure]
enabled = true                                # Enable microstructure features

# Spread analysis (requires order book data)
[features.market_microstructure.spread_analysis]
bid_ask_spread = true                         # Bid-ask spread analysis
effective_spread = true                       # Effective spread calculation
price_impact = true                           # Price impact measurement

# Order flow analysis (works with OHLCV)
[features.market_microstructure.order_flow]
volume_imbalance = true                       # Buy/sell volume imbalance
trade_intensity = true                        # Trading intensity measurement
arrival_rate = true                           # Order arrival rate analysis
```

### **Volatility Features**

```toml
[features.volatility_features]
enabled = true                                # Enable volatility features

# Realized volatility calculation
[features.volatility_features.realized_volatility]
periods = ["1h", "4h", "24h"]                # Volatility calculation periods
estimators = ["Standard", "RangeBasedYangZhang"]  # Volatility estimators

# GARCH features (advanced volatility modeling)
[features.volatility_features.garch_features]
enabled = true                                # Enable GARCH features
model_orders = [[1, 1], [1, 2], [2, 1]]     # GARCH model orders (p, q)
conditional_volatility = true                 # Include conditional volatility
volatility_forecasts = true                   # Include volatility forecasts
```

### **Custom Features**

```toml
[features.custom_features]
enabled = true                                # Enable custom features
include_all_numeric = true                    # Include all numeric CSV columns
exclude_features = ["unwanted_column"]        # Exclude specific features

# Feature transformations
[features.custom_features.transformations]
# sentiment = "ZScore"                        # Z-score normalization
# funding_rate = "PercentChange"              # Percent change transformation
# on_chain_volume = "Log"                     # Log transformation
```

### **Feature Engineering**

```toml
[features.engineering]
enabled = true                                # Enable feature engineering

# Lag features - historical values
[features.engineering.lag_features]
enabled = true                                # Enable lag features (HIGHLY RECOMMENDED)
lag_periods = [1, 2, 3, 5, 10]              # Lag periods
features_to_lag = ["close", "volume", "rsi_14"]  # Features to lag

# Rolling features - windowed statistics
[features.engineering.rolling_features]
enabled = true                                # Enable rolling features
window_sizes = [5, 10, 20]                   # Rolling window sizes
statistics = ["Mean", "Std", "Min", "Max"]   # Statistics to calculate

# Interaction features - feature combinations
[features.engineering.interaction_features]
enabled = true                                # Enable interaction features
max_interactions = 10                         # Maximum interactions to create
feature_pairs = []                            # Specific pairs (empty = auto)

# Polynomial features - non-linear combinations
[features.engineering.polynomial_features]
enabled = false                               # Disabled by default (can cause overfitting)
degree = 2                                    # Polynomial degree
include_bias = false                          # Include bias term
interaction_only = true                       # Only interaction terms
```

### **Feature Selection**

```toml
[features.selection]
enabled = true                                # Enable feature selection (HIGHLY RECOMMENDED)
max_features = 100                            # Maximum features to keep
correlation_threshold = 0.95                  # Remove highly correlated features
importance_threshold = 0.001                  # Remove low-importance features
methods = ["CorrelationFilter", "ImportanceBased"]  # Selection methods
keep_crypto_features = true                   # Always keep crypto-specific features
```

### **Cross-Asset Features (Multi-Asset Only)**

```toml
[features.cross_asset]
enabled = true                                # Enable cross-asset features
min_symbols_required = 2                      # Minimum symbols for cross-asset
required_symbols = ["BTCUSDT"]               # Required symbols (BTC for market context)
btc_dominance_enabled = true                  # Calculate BTC dominance
eth_btc_ratio_enabled = true                  # Calculate ETH/BTC ratio

# Cross-asset sentiment analysis
[features.cross_asset.sentiment_analysis]
enabled = true                                # Enable sentiment analysis
lookback_periods = 24                        # Lookback periods for sentiment
price_velocity_weight = 0.3                  # Price velocity weight
volume_spike_weight = 0.3                    # Volume spike weight
volatility_weight = 0.4                      # Volatility weight

# Cross-asset correlation analysis
[features.cross_asset.correlation_analysis]
enabled = true                                # Enable correlation analysis
min_periods = 50                             # Minimum periods for correlation
correlation_window = 20                       # Rolling correlation window
```

---

## 📊 **Data Processing Configuration**

### **Data Preprocessing**

```toml
[data]
normalization = "Robust"                      # Normalization method
# Options: "StandardScaler", "MinMaxScaler", "Robust", "None"
sequence_overlap = 0.8                        # Sequence overlap (0.0-1.0)

# Outlier handling
[data.outlier_handling]
enabled = true                                # Enable outlier detection
method = "ModifiedZScore"                     # Outlier detection method
threshold = 3.5                               # Outlier threshold
```

### **Normalization Methods**

#### **Robust (RECOMMENDED for crypto)**
```toml
normalization = "Robust"
# FEATURES: Handles outliers well, uses median and IQR
# BEST FOR: Cryptocurrency data with extreme price movements
# EFFECT: Stable normalization despite outliers
```

#### **StandardScaler**
```toml
normalization = "StandardScaler"
# FEATURES: Mean=0, std=1 normalization
# BEST FOR: Normal distributions, stable data
# EFFECT: Standard z-score normalization
```

#### **MinMaxScaler**
```toml
normalization = "MinMaxScaler"
# FEATURES: Scales to [0, 1] range
# BEST FOR: When you need bounded outputs
# EFFECT: Preserves relationships, bounded range
```

---

## 🎯 **Optimization Configuration**

### **Hyperparameter Optimization**

```toml
[optimization]
enabled = false                               # Disable for quick training
method = "None"                              # No optimization

# For production models:
# enabled = true
# method = "Bayesian"                         # Bayesian optimization (RECOMMENDED)
# n_trials = 100                             # Number of trials
# timeout_seconds = 3600                     # Timeout (1 hour)
# metric = "MAE"                             # Optimization metric
```

### **Optimization Methods**

#### **Bayesian Optimization (RECOMMENDED)**
```toml
method = "Bayesian"
n_trials = 100
# FEATURES: Efficient search, learns from previous trials
# BEST FOR: Production models, limited time budget
# EFFECT: Finds good parameters with fewer trials
```

#### **Grid Search**
```toml
method = "Grid"
# FEATURES: Exhaustive search over parameter grid
# BEST FOR: Small parameter spaces, research
# EFFECT: Guaranteed to find best combination in grid
```

#### **Random Search**
```toml
method = "Random"
n_trials = 200
# FEATURES: Random sampling of parameter space
# BEST FOR: Large parameter spaces, baseline comparison
# EFFECT: Good coverage with enough trials
```

---

## 📝 **Configuration Best Practices**

### **1. Template Selection**
- **Beginners**: Start with `configs/quick_start.toml`
- **Production**: Use `configs/training.toml` for single-asset
- **Portfolio**: Use `configs/cross_asset_training.toml` for multi-asset
- **Custom**: Modify `configs/minimal_custom.toml` or `configs/advanced_custom.toml`

### **2. Parameter Tuning Order**
1. **Data size**: Adjust architecture complexity based on dataset size
2. **Training stability**: Set appropriate learning rate and dropout
3. **Feature selection**: Enable relevant features for your use case
4. **Optimization**: Enable hyperparameter optimization for production

### **3. Common Configurations**

#### **Small Dataset (< 1000 samples)**
```toml
[model]
architecture = { MultiLSTM = { layers = 1 } }
hidden_units = { Fixed = 64 }
[model.dropout]
rate = { Fixed = 0.3 }
[features.selection]
max_features = 30
```

#### **Large Dataset (> 10000 samples)**
```toml
[model]
architecture = { MultiLSTM = { layers = 3 } }
hidden_units = { Auto = { min_units = 128, max_units = 768 } }
[optimization]
enabled = true
method = "Bayesian"
```

#### **High-Frequency Data**
```toml
[model]
sequence_length = { Auto = { min_length = 60, max_length = 300 } }
[features.technical_indicators.moving_averages]
sma_periods = [3, 5, 10, 20]  # Shorter periods
[features.engineering.lag_features]
lag_periods = [1, 2, 3, 5, 10, 15, 30]  # More lags
```

### **4. Validation and Testing**
- Always validate configuration files before training
- Use `--config` parameter to specify configuration file
- Monitor training logs for parameter validation messages
- Test with small datasets before scaling up

---

## 🏗️ **NEW: Modular Architecture Configuration**

### **Architecture Overview**

VANGA now uses a **modular LSTM architecture** with focused modules:

```
src/model/lstm/
├── config.rs      # LSTMConfig, OptimizerWrapper, TargetFormat
├── core.rs        # Model lifecycle and initialization
├── training.rs    # THE unified training method (main training logic)
├── inference.rs   # Prediction pipeline and forward pass
├── loss.rs        # Loss calculation and metrics
├── window_aware_lr.rs # Window-aware learning rate scheduling
└── mod.rs         # Public API and re-exports

src/config/
├── training.rs    # TrainingConfig, TrainingParams, 9 optimizers
├── features.rs    # FeatureConfig and feature engineering
├── model.rs       # ModelConfig and architecture configurations
├── prediction.rs  # PredictionConfig for inference
└── mod.rs         # Configuration coordination
```

### **Configuration Module Structure**

```rust
// src/config/mod.rs
pub use features::FeatureConfig;
pub use model::ModelConfig;
pub use prediction::PredictionConfig;
pub use training::TrainingConfig;

// Global configuration constants
pub struct GlobalConfig {
    pub const MODEL_DIR: &'static str = "./models";
    pub const DATA_DIR: &'static str = "./data";
    pub const CONFIG_DIR: &'static str = "./configs";
    pub const REQUIRED_COLUMNS: &'static [&'static str] =
        &["timestamp", "open", "high", "low", "close", "volume"];
}
```
└── mod.rs         # Public API with backward compatibility
```

### **Configuration Impact**

The modular architecture provides:

- **Unified Training**: Single training method handles all scenarios via configuration
- **9 Modern Optimizers**: Full optimizer support with proper configuration validation
- **Backward Compatibility**: All existing configurations work unchanged
- **Enhanced Validation**: Better error messages and parameter validation

### **Migration Notes**

- **No changes required**: Existing TOML files work as-is
- **Enhanced features**: New optimizers and hybrid models available
- **Better performance**: Improved training efficiency and memory usage
- **Cleaner code**: Modular structure improves maintainability

---

## 🔍 **Configuration Troubleshooting**

### **Common Errors**

#### **Validation Split Error**
```
[ERROR] Configuration validation failed: validation_split (0.5) + test_split (0.6) = 1.1 > 1.0
```
**Solution**: Ensure validation_split + test_split < 1.0

#### **Invalid Architecture**
```
[ERROR] Unknown architecture type: "InvalidLSTM"
```
**Solution**: Use valid architecture types (MultiLSTM, StackedLSTM, BidirectionalLSTM)

#### **Feature Configuration Error**
```
[ERROR] Cross-asset features enabled but min_symbols_required = 1
```
**Solution**: Set min_symbols_required ≥ 2 for cross-asset features

### **Performance Issues**

#### **Out of Memory**
```
[ERROR] CUDA out of memory
```
**Solutions**:
- Reduce batch_size: `batch_size = { Fixed = 32 }`
- Reduce hidden_units: `hidden_units = { Fixed = 64 }`
- Reduce sequence_length: `sequence_length = { Fixed = 30 }`

#### **Slow Training**
**Solutions**:
- Disable attention: `[model.attention] enabled = false`
- Reduce features: `[features.selection] max_features = 50`
- Use simpler architecture: `architecture = { MultiLSTM = { layers = 1 } }`

---

## 📚 **Further Reading**

- **Quick Start**: `doc/12-quick-start.md` - Getting started guide
- **Training Guide**: `doc/04-training.md` - Detailed training instructions
- **Usage Examples**: `doc/11-usage-examples.md` - Comprehensive usage patterns
- **Example Configs**: `configs/example_single_asset.toml` and `configs/example_cross_asset.toml`
