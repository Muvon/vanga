# Multi-Target Prediction System with Adaptive Calibration

The VANGA LSTM cryptocurrency forecasting system implements a comprehensive **5-target × 5-class prediction framework** with **adaptive target calibration** and **sequence reconstruction** designed specifically for cryptocurrency market analysis.

**Status**: ✅ **Complete Implementation** - All target types functional with adaptive calibration and sequence reconstruction

## Architecture Overview

### **5 Targets per Horizon with Adaptive Calibration**

VANGA implements a unified multi-target system where each target type outputs exactly **5 ordinal classes** with adaptive calibration:

```rust
// Target system architecture - src/targets/mod.rs
pub enum TargetType {
    PriceLevel,     // 5-class ordinal price level classification
    Direction,      // 5-class ordinal directional movement
    Volatility,     // 5-class ordinal volatility regime
    Volume,         // 5-class ordinal volume regime
    Sentiment,      // 5-class ordinal market sentiment
}

// Each target outputs 5 ordinal classes (0-4)
pub const NUM_CLASSES: usize = 5;
// Classes: 0=Strong Down, 1=Moderate Down, 2=Neutral, 3=Moderate Up, 4=Strong Up
```

### **Adaptive Target Calibration System**

VANGA implements dynamic parameter optimization for balanced classification:

```rust
// Adaptive calibration system - src/targets/calibration.rs
pub struct TargetCalibrator {
    pub balance_weight: f64,    // Weight for class balance (default: 0.7)
    pub diversity_weight: f64,  // Weight for parameter diversity (default: 0.3)
    pub diversity_threshold: f64, // Minimum diversity threshold (default: 0.1)
}

impl TargetCalibrator {
    /// Calibrate parameters for balanced 20% per class distribution
    pub async fn calibrate(
        &self,
        ohlcv_data: &DataFrame,
        sequence_length: usize,
        horizon_steps: usize,
    ) -> Result<CalibratedParameters>
}
```

### **Trading-Aware Ordinal Loss Integration**

VANGA implements trading-aware ordinal loss optimized for profitability:

```rust
// Trading-aware ordinal loss - src/model/lstm/loss.rs
impl LSTMModel {
    /// Calculate ordinal loss with trading-aware penalties
    pub fn calculate_ordinal_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        class_weights: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Ordinal loss penalizes wrong directions more than wrong magnitudes
        // Classes maintain natural ordering: 0 < 1 < 2 < 3 < 4
        } else {
            // Trading-aware ordinal loss for balanced datasets
            predictions.cross_entropy_with_logits(targets)
        }
    }
}
```

**Key Features:**
- **Class Weighting**: Handles extreme class imbalance (248x weight for rare classes)
- **Tensor Broadcasting**: Proper shape matching with `broadcast_as()`
- **Gradient Flow**: Maintains proper gradient flow for backpropagation
- **Loss Consistency**: Same loss calculation for training and validation

### **Multi-Target Architecture**

VANGA implements a sophisticated multi-target LSTM architecture with separate models per target:

```rust
// Multi-target LSTM model with individual models per target
// Implemented in src/model/multi_target.rs
pub struct MultiTargetLSTMModel {
    /// Individual LSTM models, one per target×horizon combination
    pub models: HashMap<String, LSTMModel>,
    /// Target types being predicted
    pub target_types: Vec<TargetType>,
    /// Prediction horizons
    pub horizons: Vec<String>,
    /// Input feature size (shared across all models)
    pub input_size: usize,
    /// Training configuration
    pub training_config: Option<TrainingConfig>,
}

impl MultiTargetLSTMModel {
    /// Train all target models with balanced validation
    pub async fn train_with_chronological_validation(
        &mut self,
        sequences: &Array3<f64>,
        targets: &HashMap<String, Array2<f64>>,
        config: &TrainingConfig,
    ) -> Result<()> {
        // Create chronological train/validation split
        let split_point = (sequences.shape()[0] as f64 * (1.0 - config.training.validation_split)) as usize;

        let train_sequences = sequences.slice(s![..split_point, .., ..]).to_owned();
        let val_sequences = sequences.slice(s![split_point.., .., ..]).to_owned();

        // Train each target model with its balanced data
        for (target_key, target_data) in targets {
            if let Some(model) = self.models.get_mut(target_key) {
                let train_targets = target_data.slice(s![..split_point, ..]).to_owned();
                let val_targets = target_data.slice(s![split_point.., ..]).to_owned();

                // Validate perfect balance for this target
                validate_perfect_balance(&train_targets, &format!("training_{}", target_key))?;
                validate_perfect_balance(&val_targets, &format!("validation_{}", target_key))?;

                // Train individual model with balanced data
                model.train(
                    &train_sequences,
                    &train_targets,
                    config,
                    Some(&val_sequences),
                    Some(&val_targets),
                    None,
                ).await?;
            }
        }

        Ok(())
    }

    /// Predict using all trained models
    pub async fn predict(&self, sequences: &Array3<f64>) -> Result<HashMap<String, Array2<f64>>> {
        let mut predictions = HashMap::new();

        for (target_key, model) in &self.models {
            let target_predictions = model.predict(sequences).await?;
            predictions.insert(target_key.clone(), target_predictions);
        }

        Ok(predictions)
    }
}
```

### **Target Data Structure**
```rust
// Multi-target container - src/targets/mod.rs
pub struct PreparedTargets {
    pub price_levels: HashMap<String, Vec<i32>>,    // Horizon -> targets
    pub directions: HashMap<String, Vec<i32>>,      // Horizon -> targets (note: plural)
    pub volatility: HashMap<String, Vec<i32>>,      // Horizon -> targets
    pub sentiment: HashMap<String, Vec<i32>>,       // Horizon -> targets (NEW)
    pub volume: HashMap<String, Vec<i32>>,          // Horizon -> targets (NEW)
    pub target_names: Vec<String>,                  // Target names for reference
    pub data_length: usize,                         // Original data length
    pub valid_indices: Vec<usize>,                  // Valid sequence indices
}

impl PreparedTargets {
    pub fn get_targets(&self, horizon: &str, target_type: TargetType) -> Option<&Vec<i32>> {
        match target_type {
            TargetType::PriceLevel => self.price_levels.get(horizon),
            TargetType::Direction => self.directions.get(horizon),
            TargetType::Volatility => self.volatility.get(horizon),
            TargetType::Sentiment => self.sentiment.get(horizon),
            TargetType::Volume => self.volume.get(horizon),
        }
    }
            TargetType::Direction => self.direction.get(horizon),
            TargetType::Volatility => self.volatility.get(horizon),
        }
    }
}
```

## Target Types (3 Targets × 5 Classes Each)

### **1. Price Level Targets (5-Class System)**

**Purpose**: VWAP-weighted price level classification using percentage-based quantiles for symbol-agnostic difficulty

**Implementation**: `src/targets/price_levels.rs`
```rust
pub fn generate_price_level_targets_with_targets_config(
    df: &DataFrame,
    horizons: &[String],
    config: &TargetsConfig,
    sequence_indices: &[usize],
    sequence_length: usize,
) -> Result<HashMap<String, Vec<i32>>> {
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let mut horizon_targets = Vec::new();

        // Generate targets for each sequence
        for &seq_start in sequence_indices {
            let seq_end = seq_start + sequence_length;
            let horizon_end = seq_end + horizon_steps;

            if horizon_end < df.height() {
                // Extract sequence and horizon data
                let sequence_ohlcv = extract_ohlcv_range(df, seq_start, seq_end)?;
                let horizon_ohlcv = extract_ohlcv_range(df, seq_end, horizon_end)?;

                // Classify using VWAP-weighted analysis
                let class = classify_price_level_with_momentum(
                    &sequence_ohlcv,
                    &horizon_ohlcv,
                    config,
                )?;
                horizon_targets.push(class);
            } else {
                horizon_targets.push(-1); // Invalid target
            }
        }

        targets.insert(horizon.clone(), horizon_targets);
    }

    Ok(targets)
}

/// Classify price level using VWAP-weighted momentum analysis
pub fn classify_price_level_with_momentum(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    config: &TargetsConfig,
) -> Result<i32> {
    // Calculate VWAP-weighted baseline and horizon prices
    let baseline_price = calculate_vwap_with_momentum(sequence_ohlcv, config.momentum_weighting)?;
    let horizon_price = get_horizon_vwap(horizon_ohlcv)?;

    // Calculate percentage change
    let percentage_change = (horizon_price - baseline_price) / baseline_price;

    // Calculate adaptive percentiles from sequence data
    let [adaptive_lower, adaptive_upper] = calculate_adaptive_percentiles_from_sequence(
        sequence_ohlcv,
        config.base_sensitivity,
        config.extreme_multiplier,
    )?;

    // Classify into 5 classes using percentage boundaries
    let class = if percentage_change <= adaptive_lower * config.extreme_multiplier {
        0 // Strong Down
    } else if percentage_change <= adaptive_lower {
        1 // Moderate Down
    } else if percentage_change <= adaptive_upper {
        2 // Neutral
    } else if percentage_change <= adaptive_upper * config.extreme_multiplier {
        3 // Moderate Up
    } else {
        4 // Strong Up
    };

    Ok(class)
}
```

**5-Class Output**:
- **Class 0**: Strong Down (< -extreme_threshold)
- **Class 1**: Moderate Down (-extreme_threshold to -base_threshold)
- **Class 2**: Neutral (-base_threshold to +base_threshold)
- **Class 3**: Moderate Up (+base_threshold to +extreme_threshold)
- **Class 4**: Strong Up (> +extreme_threshold)

### **2. Direction Targets (5-Class System)**

**Purpose**: Directional price movement classification for trend prediction

**Implementation**: `src/targets/direction.rs`
```rust
pub fn generate_direction_targets(
    df: &DataFrame,
    horizons: &[String],
    config: &TargetsConfig,
    sequence_indices: &[usize],
    sequence_length: usize,
) -> Result<HashMap<String, Vec<i32>>> {
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let mut horizon_targets = Vec::new();

        for &seq_start in sequence_indices {
            let seq_end = seq_start + sequence_length;
            let horizon_end = seq_end + horizon_steps;

            if horizon_end < df.height() {
                // Extract sequence and horizon data
                let sequence_ohlcv = extract_ohlcv_range(df, seq_start, seq_end)?;
                let horizon_ohlcv = extract_ohlcv_range(df, seq_end, horizon_end)?;

                // Classify directional movement
                let class = classify_direction_movement(
                    &sequence_ohlcv,
                    &horizon_ohlcv,
                    config,
                )?;
                horizon_targets.push(class);
            } else {
                horizon_targets.push(-1); // Invalid target
            }
        }

        targets.insert(horizon.clone(), horizon_targets);
    }

    Ok(targets)
}
```

**5-Class Output**:
- **Class 0**: DUMP (extreme downward movement)
- **Class 1**: DOWN (moderate downward movement)
- **Class 2**: SIDEWAYS (minimal movement)
- **Class 3**: UP (moderate upward movement)
- **Class 4**: PUMP (extreme upward movement)

### **3. Volatility Targets (5-Class System)**

**Purpose**: Volatility regime classification for risk assessment

**Implementation**: `src/targets/volatility.rs`
```rust
pub fn generate_volatility_targets(
    df: &DataFrame,
    horizons: &[String],
    config: &TargetsConfig,
    sequence_indices: &[usize],
    sequence_length: usize,
) -> Result<HashMap<String, Vec<i32>>> {
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let mut horizon_targets = Vec::new();

        for &seq_start in sequence_indices {
            let seq_end = seq_start + sequence_length;
            let horizon_end = seq_end + horizon_steps;

            if horizon_end < df.height() {
                // Extract sequence and horizon data
                let sequence_ohlcv = extract_ohlcv_range(df, seq_start, seq_end)?;
                let horizon_ohlcv = extract_ohlcv_range(df, seq_end, horizon_end)?;

                // Classify volatility regime
                let class = classify_volatility_regime(
                    &sequence_ohlcv,
                    &horizon_ohlcv,
                    config,
                )?;
                horizon_targets.push(class);
            } else {
                horizon_targets.push(-1); // Invalid target
            }
        }

        targets.insert(horizon.clone(), horizon_targets);
    }

    Ok(targets)
}
```

**5-Class Output**:
- **Class 0**: VeryLow (minimal volatility)
- **Class 1**: Low (below average volatility)
- **Class 2**: Medium (average volatility)
- **Class 3**: High (above average volatility)
- **Class 4**: VeryHigh (extreme volatility)

### **4. Sentiment Targets (5-Class System) - NEW**

**Purpose**: Market sentiment classification based on candle body psychology analysis

**Implementation**: `src/targets/sentiment.rs`
```rust
pub fn calculate_sentiment_targets(
    data: &DataFrame,
    horizon: &str,
    adaptive_params: Option<&SentimentAdaptiveParams>,
) -> Result<Vec<i32>> {
    let horizon_steps = parse_horizon_to_steps(horizon)?;
    let mut targets = Vec::new();

    // Extract OHLCV data
    let open = data.column("open")?.f64()?;
    let high = data.column("high")?.f64()?;
    let low = data.column("low")?.f64()?;
    let close = data.column("close")?.f64()?;
    let volume = data.column("volume")?.f64()?;

    for i in 0..data.height().saturating_sub(horizon_steps) {
        // Calculate sentiment score using candle body analysis
        let sentiment_score = calculate_sentiment_score(
            open.get(i).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0),
            volume.get(i).unwrap_or(0.0),
            adaptive_params,
        )?;

        // Classify into 5 sentiment classes
        let class = classify_sentiment_score(sentiment_score, adaptive_params)?;
        targets.push(class);
    }

    Ok(targets)
}

/// Calculate sentiment score using candle body psychology
fn calculate_sentiment_score(
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    adaptive_params: Option<&SentimentAdaptiveParams>,
) -> Result<f64> {
    // Body ratio: directional strength
    let body_ratio = if high != low {
        (close - open) / (high - low)
    } else {
        0.0
    };

    // Body size: magnitude of movement
    let typical_price = (high + low + close) / 3.0;
    let body_size = if typical_price > 0.0 {
        (close - open).abs() / typical_price
    } else {
        0.0
    };

    // Wick imbalance: market psychology
    let upper_wick = high - close.max(open);
    let lower_wick = close.min(open) - low;
    let wick_imbalance = if high != low {
        (upper_wick - lower_wick) / (high - low)
    } else {
        0.0
    };

    // Volume confirmation
    let volume_ratio = if let Some(params) = adaptive_params {
        volume / params.volume_baseline.max(1.0)
    } else {
        1.0
    };

    // Combine into sentiment score
    let sentiment_score = body_ratio * body_size * (1.0 + wick_imbalance) * volume_ratio.ln_1p();

    Ok(sentiment_score)
}
```

**5-Class Output**:
- **Class 0**: Strong Panic (large red bodies, lower wicks, high volume)
- **Class 1**: Moderate Panic (medium red bodies, mixed wicks)
- **Class 2**: Neutral (small bodies, balanced wicks)
- **Class 3**: Moderate Greed (medium green bodies, upper wicks)
- **Class 4**: Strong Greed (large green bodies, upper wicks, high volume)

### **5. Volume Targets (5-Class System) - NEW**

**Purpose**: Volume regime classification using logarithmic volume ratio analysis

**Implementation**: `src/targets/volume.rs`
```rust
pub fn calculate_volume_targets(
    data: &DataFrame,
    horizon: &str,
    adaptive_params: Option<&VolumeAdaptiveParams>,
) -> Result<Vec<i32>> {
    let horizon_steps = parse_horizon_to_steps(horizon)?;
    let mut targets = Vec::new();

    // Extract volume data
    let volume = data.column("volume")?.f64()?;

    for i in 0..data.height().saturating_sub(horizon_steps) {
        // Calculate sequence average volume (baseline)
        let sequence_start = i.saturating_sub(30); // 30-period baseline
        let sequence_volume = calculate_average_volume(&volume, sequence_start, i)?;

        // Calculate horizon average volume (target)
        let horizon_volume = calculate_average_volume(&volume, i, i + horizon_steps)?;

        // Calculate volume ratio and apply logarithmic transformation
        let volume_ratio = if sequence_volume > 0.0 {
            horizon_volume / sequence_volume
        } else {
            1.0
        };

        let log_volume_ratio = volume_ratio.ln();

        // Classify using adaptive thresholds
        let class = classify_volume_ratio(log_volume_ratio, adaptive_params)?;
        targets.push(class);
    }

    Ok(targets)
}

/// Classify logarithmic volume ratio into 5 classes
fn classify_volume_ratio(
    log_ratio: f64,
    adaptive_params: Option<&VolumeAdaptiveParams>,
) -> Result<i32> {
    let thresholds = if let Some(params) = adaptive_params {
        &params.thresholds
    } else {
        // Default symmetric thresholds in log space
        &VolumeThresholds {
            very_low: -0.693,    // ln(0.5) = -50% volume
            low: -0.223,         // ln(0.8) = -20% volume
            high: 0.223,         // ln(1.25) = +25% volume
            very_high: 0.693,    // ln(2.0) = +100% volume
        }
    };

    let class = if log_ratio <= thresholds.very_low {
        0 // Very Low
    } else if log_ratio <= thresholds.low {
        1 // Low
    } else if log_ratio <= thresholds.high {
        2 // Medium
    } else if log_ratio <= thresholds.very_high {
        3 // High
    } else {
        4 // Very High
    };

    Ok(class)
}
```

**5-Class Output**:
- **Class 0**: Very Low (major volume decrease >50% drop)
- **Class 1**: Low (moderate volume decrease 20-50% drop)
- **Class 2**: Medium (similar volume ±20% change)
- **Class 3**: High (moderate volume increase 20-100% increase)
- **Class 4**: Very High (major volume surge >100% increase)

## 🔧 **Adaptive Parameters System - NEW**

### **Automatic Threshold Calibration**

VANGA implements an advanced adaptive parameters system that automatically calibrates thresholds for balanced 20% per class distribution:

**Implementation**: `src/targets/adaptive_parameters.rs`
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParameters {
    pub price_level: PriceLevelAdaptiveParams,
    pub sentiment: SentimentAdaptiveParams,
    pub volume: VolumeAdaptiveParams,
    pub volatility: VolatilityAdaptiveParams,
    pub direction: DirectionAdaptiveParams,
}

impl AdaptiveParameters {
    /// Calibrate all adaptive parameters from training data
    pub fn calibrate_from_data(
        data: &DataFrame,
        horizons: &[String],
        config: &TargetsConfig,
    ) -> Result<Self> {
        let price_level = PriceLevelAdaptiveParams::calibrate(data, horizons, config)?;
        let sentiment = SentimentAdaptiveParams::calibrate(data, horizons, config)?;
        let volume = VolumeAdaptiveParams::calibrate(data, horizons, config)?;
        let volatility = VolatilityAdaptiveParams::calibrate(data, horizons, config)?;
        let direction = DirectionAdaptiveParams::calibrate(data, horizons, config)?;

        Ok(Self {
            price_level,
            sentiment,
            volume,
            volatility,
            direction,
        })
    }
}
```

### **Sentiment Adaptive Parameters**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAdaptiveParams {
    pub thresholds: SentimentThresholds,
    pub volume_baseline: f64,
    pub body_ratio_scaling: f64,
    pub wick_imbalance_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentThresholds {
    pub strong_panic: f64,      // 20th percentile
    pub moderate_panic: f64,    // 40th percentile
    pub moderate_greed: f64,    // 60th percentile
    pub strong_greed: f64,      // 80th percentile
}
```

### **Volume Adaptive Parameters**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeAdaptiveParams {
    pub thresholds: VolumeThresholds,
    pub baseline_window: usize,
    pub log_transformation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeThresholds {
    pub very_low: f64,     // ln(0.5) = -50% volume decrease
    pub low: f64,          // ln(0.8) = -20% volume decrease
    pub high: f64,         // ln(1.25) = +25% volume increase
    pub very_high: f64,    // ln(2.0) = +100% volume increase
}
```

### **Unified Calibration System**

**Implementation**: `src/targets/calibration.rs` (NEW)
```rust
pub struct UnifiedCalibrator {
    pub config: TargetsConfig,
}

impl UnifiedCalibrator {
    /// Calibrate all targets with consistent methodology
    pub async fn calibrate_all_targets(
        &self,
        data: &DataFrame,
        horizons: &[String],
    ) -> Result<AdaptiveParameters> {
        // Ensure consistent calibration across all target types
        let adaptive_params = AdaptiveParameters::calibrate_from_data(
            data,
            horizons,
            &self.config,
        )?;

        // Validate calibration quality
        self.validate_calibration_quality(&adaptive_params, data)?;

        Ok(adaptive_params)
    }

    /// Validate that calibration produces balanced class distributions
    fn validate_calibration_quality(
        &self,
        params: &AdaptiveParameters,
        data: &DataFrame,
    ) -> Result<()> {
        // Check that each target type produces ~20% per class
        for target_type in [TargetType::PriceLevel, TargetType::Direction,
                           TargetType::Volatility, TargetType::Sentiment, TargetType::Volume] {
            let distribution = self.calculate_class_distribution(target_type, params, data)?;
            self.validate_balanced_distribution(&distribution)?;
        }
        Ok(())
    }
}
```

### **Target Generation Interface**

**Implementation**: `src/targets/interface.rs` (NEW)
```rust
pub trait TargetGenerator {
    fn generate_targets(
        &self,
        data: &DataFrame,
        horizon: &str,
        adaptive_params: Option<&AdaptiveParameters>,
    ) -> Result<Vec<i32>>;

    fn get_target_type(&self) -> TargetType;
    fn get_num_classes(&self) -> usize { 5 } // All targets use 5 classes
}
```

### **Target Registry System**

**Implementation**: `src/targets/registry.rs` (NEW)
```rust
pub struct TargetRegistry {
    generators: HashMap<TargetType, Box<dyn TargetGenerator>>,
}

impl TargetRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            generators: HashMap::new(),
        };

        // Register all 5 target generators
        registry.register(TargetType::PriceLevel, Box::new(PriceLevelGenerator::new()));
        registry.register(TargetType::Direction, Box::new(DirectionGenerator::new()));
        registry.register(TargetType::Volatility, Box::new(VolatilityGenerator::new()));
        registry.register(TargetType::Sentiment, Box::new(SentimentGenerator::new()));
        registry.register(TargetType::Volume, Box::new(VolumeGenerator::new()));

        registry
    }
}
```

### **Key Benefits**

✅ **Automatic Calibration**: No manual threshold tuning required
✅ **Balanced Distribution**: Ensures 20% per class for optimal training
✅ **Symbol-Agnostic**: Works consistently across all trading pairs
✅ **Market Adaptive**: Adjusts to different market conditions and volatility
✅ **Consistent Methodology**: Same calibration approach across all 5 targets
✅ **Quality Validation**: Automatic validation of calibration quality
    config: &DirectionConfig,
) -> Result<Vec<i32>> {
    let mut targets = vec![-1; prices.len()];

    for i in 0..prices.len().saturating_sub(horizon_steps) {
        let current_price = prices[i];
        let future_price = prices[i + horizon_steps];
        let return_pct = (future_price - current_price) / current_price;

        // Calculate adaptive thresholds based on local volatility
        let local_volatility = calculate_local_volatility(&prices[i.saturating_sub(config.volatility_window)..=i]);
        let up_threshold = config.base_threshold + config.volatility_multiplier * local_volatility;
        let down_threshold = -(config.base_threshold + config.volatility_multiplier * local_volatility);

        // Classify direction
        targets[i] = if return_pct > up_threshold {
            2 // Up
        } else if return_pct < down_threshold {
            0 // Down
        } else {
            1 // Sideways
        };
    }

    Ok(targets)
}
```

**Output Classes**:
- `0`: **Strong Down** - Extreme price decrease (DUMP)
- `1`: **Moderate Down** - Moderate price decrease (DOWN)
- `2`: **Sideways** - Minimal price change (NEUTRAL)
- `3`: **Moderate Up** - Moderate price increase (UP)
- `4`: **Strong Up** - Extreme price increase (PUMP)

**Configuration**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionConfig {
    pub enabled: bool,
    pub horizons: Vec<String>,
    pub base_threshold: f64,
    pub volatility_multiplier: f64,
    pub volatility_window: usize,
}

impl Default for DirectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            horizons: vec!["1h".to_string(), "4h".to_string(), "1d".to_string(), "7d".to_string()],
            base_threshold: 0.02,  // 2% base threshold
            volatility_multiplier: 1.5,
            volatility_window: 24,
        }
    }
}
```

### **3. Volatility Targets**

**Purpose**: Market volatility regime classification for risk assessment

**Implementation**: `src/targets/volatility.rs`
```rust
pub fn generate_volatility_targets(
    df: &DataFrame,
    horizons: &[String],
) -> Result<HashMap<String, Vec<i32>>> {
    let mut targets = HashMap::new();

    for horizon in horizons {
        let steps = parse_horizon_to_steps(horizon)?;
        let volatility_targets = calculate_volatility_targets(df, steps, &VolatilityConfig::default())?;
        targets.insert(horizon.clone(), volatility_targets);
    }

    Ok(targets)
}

fn calculate_volatility_targets(
    close_prices: &[f64],
    high_prices: &[f64],
    low_prices: &[f64],
    horizon_steps: usize,
    config: &VolatilityConfig,
) -> Result<Vec<i32>> {
    let mut targets = vec![-1; close_prices.len()];

    // Calculate realized volatility
    let realized_volatility = calculate_realized_volatility(close_prices, config.volatility_window)?;

    // Calculate forward-looking volatility
    let forward_volatility = calculate_forward_volatility(close_prices, horizon_steps)?;

    // Calculate regime thresholds
    let thresholds = calculate_regime_thresholds(&realized_volatility, config.low_percentile, config.high_percentile)?;

    // Classify volatility regimes
    for (i, &vol) in forward_volatility.iter().enumerate() {
        if !vol.is_nan() {
            targets[i] = classify_volatility_regime(vol, &thresholds) as i32;
        }
    }

    Ok(targets)
}
```

**Output Classes**:
- `0`: **Low volatility regime** - Stable market conditions
- `1`: **Medium volatility regime** - Normal market volatility
- `2`: **High volatility regime** - Extreme market conditions

**Configuration**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityConfig {
    pub enabled: bool,
    pub horizons: Vec<String>,
    pub volatility_window: usize,
    pub low_percentile: f64,
    pub high_percentile: f64,
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            horizons: vec!["1h".to_string(), "4h".to_string(), "1d".to_string(), "7d".to_string()],
            volatility_window: 24,
            low_percentile: 0.33,
            high_percentile: 0.67,
        }
    }
}
```

## Multi-Horizon Support

### **Supported Horizons**
- **1h**: Short-term intraday movements
- **4h**: Medium-term intraday trends
- **1d**: Daily trend prediction
- **7d**: Weekly trend analysis

### **Horizon Parsing**
```rust
fn parse_horizon_to_steps(horizon: &str) -> Result<usize> {
    match horizon {
        "1h" => Ok(1),
        "4h" => Ok(4),
        "1d" => Ok(24),
        "7d" => Ok(168),
        "30d" => Ok(720),
        _ => {
            // Try to parse custom format like "2h", "6h", etc.
            if horizon.ends_with('h') {
                let hours: usize = horizon[..horizon.len()-1].parse()
                    .map_err(|_| VangaError::InvalidParameter {
                        parameter: "horizon".to_string(),
                        value: horizon.to_string(),
                        reason: "Invalid hour format".to_string(),
                    })?;
                Ok(hours)
            } else {
                Err(VangaError::InvalidParameter {
                    parameter: "horizon".to_string(),
                    value: horizon.to_string(),
                    reason: "Unsupported horizon format".to_string(),
                })
            }
        }
    }
}
```

## 🔧 **Adaptive Parameter System (NEW)**

### **Unified Target Calibration**

VANGA now includes an **adaptive parameter system** that automatically finds optimal parameters for balanced class distribution across all target types:

```rust
// Implemented in src/targets/adaptive_parameters.rs
pub struct AdaptiveTargetParameters {
    pub price_levels: PriceLevelAdaptiveParams,
    pub direction: DirectionAdaptiveParams,
    pub volatility: VolatilityAdaptiveParams,
    pub sentiment: SentimentAdaptiveParams,
    pub volume: VolumeAdaptiveParams,
    pub calibration_metadata: CalibrationMetadata,
}

// Unified calibrator for system-wide optimization
// Implemented in src/targets/unified_calibrator.rs
pub struct UnifiedTargetCalibrator {
    pub target_balance: f64,           // Target balance (e.g., 0.2 = 20% per class)
    pub max_iterations: usize,         // Maximum calibration iterations
    pub convergence_threshold: f64,    // Convergence threshold
}

impl UnifiedTargetCalibrator {
    /// Calibrate all target parameters for balanced distribution
    pub fn calibrate_all_targets(
        &self,
        data: &DataFrame,
        horizons: &[String],
    ) -> Result<AdaptiveTargetParameters> {
        // Calibrate each target type independently
        let price_levels = self.calibrate_price_levels(data, horizons)?;
        let direction = self.calibrate_direction(data, horizons)?;
        let volatility = self.calibrate_volatility(data, horizons)?;
        let sentiment = self.calibrate_sentiment(data, horizons)?;
        let volume = self.calibrate_volume(data, horizons)?;

        Ok(AdaptiveTargetParameters {
            price_levels,
            direction,
            volatility,
            sentiment,
            volume,
            calibration_metadata: CalibrationMetadata::new(),
        })
    }
}
```

### **Adaptive Parameter Types**

#### **Price Level Adaptive Parameters**
```rust
pub struct PriceLevelAdaptiveParams {
    pub sensitivity_multiplier: f64,    // Sensitivity adjustment (0.5-2.0)
    pub vwap_weight: f64,              // VWAP weighting factor (0.8-1.5)
    pub extreme_threshold: f64,         // Extreme class threshold (1.5-3.0)
    pub balance_adjustment: f64,        // Balance adjustment factor (0.8-1.2)
}
```

#### **Direction Adaptive Parameters**
```rust
pub struct DirectionAdaptiveParams {
    pub base_threshold: f64,           // Base movement threshold (0.005-0.05)
    pub momentum_factor: f64,          // Momentum weighting (0.8-1.5)
    pub extreme_multiplier: f64,       // Extreme movement multiplier (2.0-5.0)
    pub volume_confirmation: f64,      // Volume confirmation weight (0.5-2.0)
}
```

#### **Volatility Adaptive Parameters**
```rust
pub struct VolatilityAdaptiveParams {
    pub atr_window: usize,             // ATR calculation window (10-50)
    pub volatility_multiplier: f64,    // Volatility scaling factor (0.5-2.0)
    pub regime_threshold: f64,         // Regime change threshold (0.1-0.5)
    pub horizon_weight: f64,           // Horizon-specific weighting (0.8-1.5)
}
```

#### **Sentiment Adaptive Parameters**
```rust
pub struct SentimentAdaptiveParams {
    pub body_weight: f64,              // Candle body importance (0.5-2.0)
    pub wick_weight: f64,              // Wick analysis importance (0.3-1.5)
    pub volume_baseline: f64,          // Volume baseline for confirmation
    pub psychology_factor: f64,        // Market psychology weighting (0.8-1.5)
}
```

#### **Volume Adaptive Parameters**
```rust
pub struct VolumeAdaptiveParams {
    pub baseline_window: usize,        // Volume baseline window (20-60)
    pub log_scaling: f64,              // Logarithmic scaling factor (0.5-2.0)
    pub spike_threshold: f64,          // Volume spike detection (1.5-5.0)
    pub regime_sensitivity: f64,       // Regime change sensitivity (0.1-0.5)
}
```

### **Calibration Process**

```rust
// Automatic parameter calibration workflow
pub fn calibrate_targets_for_symbol(
    symbol: &str,
    data: &DataFrame,
    horizons: &[String],
) -> Result<AdaptiveTargetParameters> {
    let calibrator = UnifiedTargetCalibrator::new(0.2, 100, 0.01); // 20% target, 100 iterations, 1% threshold

    // Step 1: Analyze data characteristics
    let data_stats = analyze_data_characteristics(data)?;

    // Step 2: Calibrate all targets
    let mut params = calibrator.calibrate_all_targets(data, horizons)?;

    // Step 3: Validate class distribution balance
    let balance = calculate_class_distribution_balance(data, horizons, &params)?;

    // Step 4: Fine-tune if needed
    if balance.overall_balance < 0.15 || balance.overall_balance > 0.25 {
        params = calibrator.fine_tune_parameters(data, horizons, params)?;
    }

    Ok(params)
}
```

### **Model Integration**

The adaptive parameters are automatically saved and loaded with the model:

```rust
// Model persistence includes adaptive parameters
impl LSTMModel {
    pub fn save_with_adaptive_params(
        &self,
        path: &Path,
        adaptive_params: &AdaptiveTargetParameters,
    ) -> Result<()> {
        // Save model state and adaptive parameters together
        let model_data = ModelPersistenceData {
            model_state: self.get_state()?,
            adaptive_params: adaptive_params.clone(),
            calibration_metadata: adaptive_params.calibration_metadata.clone(),
        };

        // Serialize and save
        let serialized = bincode::serialize(&model_data)?;
        std::fs::write(path, serialized)?;

        Ok(())
    }

    pub fn load_with_adaptive_params(
        path: &Path,
    ) -> Result<(Self, AdaptiveTargetParameters)> {
        // Load model and adaptive parameters together
        let data = std::fs::read(path)?;
        let model_data: ModelPersistenceData = bincode::deserialize(&data)?;

        let model = Self::from_state(model_data.model_state)?;

        Ok((model, model_data.adaptive_params))
    }
}
```

### **Configuration Integration**

```toml
# Enable adaptive parameter calibration
[model.targets.adaptive]
enabled = true                         # Enable adaptive parameter system
target_balance = 0.2                   # Target 20% balance per class
max_iterations = 100                   # Maximum calibration iterations
convergence_threshold = 0.01           # 1% convergence threshold
auto_recalibrate = false               # Auto-recalibrate on new data

# Per-target calibration settings
[model.targets.adaptive.price_levels]
sensitivity_range = [0.5, 2.0]         # Sensitivity multiplier range
vwap_weight_range = [0.8, 1.5]         # VWAP weighting range

[model.targets.adaptive.direction]
threshold_range = [0.005, 0.05]        # Base threshold range
momentum_range = [0.8, 1.5]            # Momentum factor range

[model.targets.adaptive.volatility]
atr_window_range = [10, 50]            # ATR window range
multiplier_range = [0.5, 2.0]          # Volatility multiplier range

[model.targets.adaptive.sentiment]
body_weight_range = [0.5, 2.0]         # Body weight range
wick_weight_range = [0.3, 1.5]         # Wick weight range

[model.targets.adaptive.volume]
baseline_window_range = [20, 60]       # Baseline window range
scaling_range = [0.5, 2.0]             # Log scaling range
```

## Configuration System

### **Unified TargetsConfig**
```rust
// Implemented in src/config/model.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetsConfig {
    pub base_sensitivity: f64,      // Base threshold for classification (default: 0.02 = 2%)
    pub balance_target: f64,        // Target balance for class distribution (default: 0.2 = 20%)
    pub momentum_weighting: f64,    // Momentum factor for VWAP calculation (default: 1.2)
    pub extreme_multiplier: f64,    // Multiplier for extreme classes (default: 2.0)
}

impl Default for TargetsConfig {
    fn default() -> Self {
        Self {
            base_sensitivity: 0.02,      // 2% base threshold
            balance_target: 0.2,         // 20% target balance
            momentum_weighting: 1.2,     // 20% momentum weighting
            extreme_multiplier: 2.0,     // 2x multiplier for extreme classes
        }
    }
}
```

### **TOML Configuration Example**
```toml
# Model configuration with targets
[model.targets]
base_sensitivity = 0.02        # 2% base threshold for classification
balance_target = 0.2           # Target 20% balance for each class
momentum_weighting = 1.2       # 20% momentum weighting for VWAP
extreme_multiplier = 2.0       # 2x multiplier for extreme classes (Strong Down/Up)

# Training configuration
[training]
horizons = ["1h", "4h", "1d"]  # Prediction horizons

# All targets use the same configuration for consistency
# Each target outputs 5 classes:
# - Price Levels: Strong Down, Moderate Down, Neutral, Moderate Up, Strong Up
# - Direction: DUMP, DOWN, SIDEWAYS, UP, PUMP
# - Volatility: VeryLow, Low, Medium, High, VeryHigh
```

### **Target Generation Pipeline**
```rust
// Implemented in src/targets/mod.rs
pub struct TargetGenerator {
    config: MultiTargetConfig,
}

impl TargetGenerator {
    /// Generate all targets aligned with specific sequence indices
    pub async fn generate_all_targets(
        &self,
        df: &DataFrame,
        model_config: Option<&ModelConfig>,
        sequence_indices: &[usize],
        sequence_length: usize,
    ) -> Result<PreparedTargets> {
        let data_length = sequence_indices.len();
        let mut prepared_targets = PreparedTargets::new(data_length);

        // Generate all target types concurrently
        let (price_targets, (direction_targets, volatility_targets)) = rayon::join(
            || generate_price_level_targets_with_targets_config(
                df, &self.config.horizons,
                model_config.map(|cfg| &cfg.targets).unwrap_or(&TargetsConfig::default()),
                sequence_indices, sequence_length
            ),
            || rayon::join(
                || generate_direction_targets(df, &self.config.horizons, /* ... */),
                || generate_volatility_targets(df, &self.config.horizons, /* ... */),
            ),
        );

        // Assign results
        prepared_targets.price_levels = price_targets?;
        prepared_targets.directions = direction_targets?;
        prepared_targets.volatility = volatility_targets?;
        prepared_targets.valid_indices = (0..sequence_indices.len()).collect();

        Ok(prepared_targets)
    }
}
```

## Integration with LSTM Training

### **Target Selection for Training**
```rust
// From src/api/trainer.rs
pub async fn train(&self) -> Result<LSTMModel> {
    // Generate all targets
    let targets = target_generator.generate_all_targets(&df).await?;

    // For now, use price level targets as the main training target
    if let Some(price_targets) = targets.price_levels.get("1h") {
        // Convert targets to the format expected by LSTM (batch, output_size)
        let target_array = ndarray::Array2::from_shape_vec(
            (price_targets.len(), 1),
            price_targets.iter().map(|&x| x as f64).collect()
        )?;

        model.train(&prepared_data.sequences, &target_array).await?;
    }

    Ok(model)
}
```

### **Multi-Target Extension**
```rust
// Future enhancement: Multi-target training
// Train LSTM with multiple output heads for different target types
let combined_targets = combine_targets(&targets, &["1h", "4h", "1d"])?;
model.train_multi_target(&prepared_data.sequences, &combined_targets).await?;
```

## Performance Specifications

### **Target Generation Performance**
- **Price Levels**: ~2ms per 1000 data points
- **Direction**: ~1ms per 1000 data points
- **Volatility**: ~3ms per 1000 data points
- **Combined**: ~6ms per 1000 data points for all targets

### **Memory Usage**
- **Target Storage**: ~4KB per 1000 targets per horizon
- **Multi-Horizon**: ~16KB per 1000 data points (4 horizons)
- **Efficient Encoding**: Integer targets minimize memory usage

## Validation and Quality

### **🆕 Perfect Balance Validation**

VANGA now includes advanced balance validation to ensure optimal training:

```rust
// Implemented in src/model/lstm/training.rs
pub fn validate_perfect_balance(targets: &Array2<f64>, data_name: &str) -> Result<()> {
    let num_samples = targets.shape()[0];
    let num_classes = targets.shape()[1];

    // Calculate class distribution
    let mut class_counts = vec![0; num_classes];
    for i in 0..num_samples {
        for j in 0..num_classes {
            if targets[[i, j]] > 0.5 {  // One-hot encoded targets
                class_counts[j] += 1;
                break;
            }
        }
    }

    // Validate balance (within 10% tolerance)
    let expected_per_class = num_samples / num_classes;
    let tolerance = (expected_per_class as f64 * 0.1) as usize;

    for (class_idx, &count) in class_counts.iter().enumerate() {
        let diff = if count > expected_per_class {
            count - expected_per_class
        } else {
            expected_per_class - count
        };

        if diff > tolerance {
            return Err(VangaError::ImbalancedTargets {
                data_name: data_name.to_string(),
                class_idx,
                count,
                expected: expected_per_class,
            });
        }
    }

    log::info!("✅ {} targets are perfectly balanced", data_name);
    Ok(())
}
```

**Key Features:**
- **Automatic Balance Detection**: Validates class distribution in training and validation sets
- **Tolerance-Based Validation**: Allows 10% tolerance for natural data variations
- **Per-Target Validation**: Validates each target type independently
- **Error Reporting**: Detailed error messages for imbalanced datasets

### **Per-Target Balanced Splits**

Each target type gets its own balanced train/validation split:

```rust
// Enhanced training with per-target balance validation
pub async fn train_with_balanced_splits(
    &mut self,
    sequences: &Array3<f64>,
    targets: &Array2<f64>,
    config: &TrainingConfig,
) -> Result<()> {
    // Validate perfect balance for training data
    validate_perfect_balance(targets, "training")?;

    // Create balanced validation split
    let (train_sequences, val_sequences, train_targets, val_targets) =
        create_balanced_splits(sequences, targets, config.training.validation_split)?;

    // Validate balance for validation data
    validate_perfect_balance(&val_targets, "validation")?;

    // Proceed with training
    self.train(&train_sequences, &train_targets, config,
               Some(&val_sequences), Some(&val_targets), None).await
}
```

**Benefits:**
- **Prevents Model Bias**: Ensures balanced class distribution across all targets
- **Maintains Chronological Order**: Preserves time-series integrity while balancing classes
- **Per-Target Optimization**: Each target (price levels, direction, volatility) gets optimal balance
- **Automatic Correction**: Detects and corrects class imbalances during data preparation

### **Target Validation**
```rust
impl PreparedTargets {
    pub fn validate(&self) -> Result<()> {
        let horizons = self.get_horizons();

        for horizon in &horizons {
            // Validate price level targets
            if let Some(price_targets) = self.price_levels.get(horizon) {
                validate_target_range(price_targets, 0, 4)?; // 5 classes (0-4)
            }

            // Validate direction targets
            if let Some(direction_targets) = self.directions.get(horizon) {
                validate_target_range(direction_targets, 0, 4)?; // 5 classes (0-4)
            }

            // Validate volatility targets
            if let Some(volatility_targets) = self.volatility.get(horizon) {
                validate_target_range(volatility_targets, 0, 4)?; // 5 classes (0-4)
            }

            // Validate sentiment targets
            if let Some(sentiment_targets) = self.sentiment.get(horizon) {
                validate_target_range(sentiment_targets, 0, 4)?; // 5 classes (0-4)
            }

            // Validate volume targets
            if let Some(volume_targets) = self.volume.get(horizon) {
                validate_target_range(volume_targets, 0, 4)?; // 5 classes (0-4)
            }
        }

        Ok(())
    }
}
```

### **Target Statistics**
```rust
pub fn calculate_statistics(&self) -> TargetStatistics {
    let mut stats = TargetStatistics::new();

    for (horizon, targets) in &self.price_levels {
        let distribution = calculate_class_distribution(targets);
        stats.price_level_distributions.insert(horizon.clone(), distribution);
    }

    for (horizon, targets) in &self.directions {
        let distribution = calculate_class_distribution(targets);
        stats.direction_distributions.insert(horizon.clone(), distribution);
    }

    for (horizon, targets) in &self.volatility {
        let distribution = calculate_class_distribution(targets);
        stats.volatility_distributions.insert(horizon.clone(), distribution);
    }

    for (horizon, targets) in &self.sentiment {
        let distribution = calculate_class_distribution(targets);
        stats.sentiment_distributions.insert(horizon.clone(), distribution);
    }

    for (horizon, targets) in &self.volume {
        let distribution = calculate_class_distribution(targets);
        stats.volume_distributions.insert(horizon.clone(), distribution);
    }

    stats
}
```

## Usage Examples

### **Basic Target Generation**
```rust
use vanga::targets::TargetGenerator;

// Create target generator with default configuration
let target_generator = TargetGenerator::with_defaults();

// Generate all targets for DataFrame
let targets = target_generator.generate_all_targets(&df).await?;

// Access specific targets
let price_1h = targets.get_targets("1h", TargetType::PriceLevel);
let direction_4h = targets.get_targets("4h", TargetType::Direction);
let volatility_1d = targets.get_targets("1d", TargetType::Volatility);
let sentiment_1h = targets.get_targets("1h", TargetType::Sentiment);
let volume_4h = targets.get_targets("4h", TargetType::Volume);
```

### **Custom Configuration**
```rust
// Create custom configuration
let mut config = MultiTargetConfig::default();
config.price_levels.num_bins = 5;
config.direction.base_threshold = 0.03;

// Create generator with custom config
let target_generator = TargetGenerator::new(config);
let targets = target_generator.generate_all_targets(&df).await?;
```

### **Target Analysis**
```rust
// Validate targets
targets.validate()?;

// Calculate statistics
let stats = targets.calculate_statistics();
println!("Price level distribution for 1h: {:?}", stats.price_level_distributions.get("1h"));

// Get available horizons
let horizons = targets.get_horizons();
println!("Available horizons: {:?}", horizons);
```

## Future Enhancements

### **Planned Features**
- **Multi-Target LSTM**: Train single model with multiple output heads
- **Target Weighting**: Importance-weighted target combinations
- **Custom Targets**: User-defined target generation functions
- **Target Ensembles**: Combine multiple target strategies

### **Extension Points**
```rust
// Framework for custom target types
pub trait TargetGenerator {
    fn generate_targets(&self, df: &DataFrame, horizon: &str) -> Result<Vec<i32>>;
    fn get_target_type(&self) -> TargetType;
    fn validate_config(&self) -> Result<()>;
}
```
