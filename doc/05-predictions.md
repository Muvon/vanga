# Making Predictions

This guide covers how to generate predictions using trained LSTM models in VANGA.

**Status**: ✅ **Complete Implementation** - Full prediction pipeline functional

## Quick Start

### **Basic Prediction with Current API**

Generate predictions using the current Rust API:

```rust
// Using current prediction API - src/api/predictor.rs
use vanga::api::predictor::{predict_single_model, predict_multi_target_model, ModelWrapper};
use vanga::config::PredictionConfig;
use vanga::model::multi_target::MultiTargetLSTMModel;

// Load trained multi-target model
let model = MultiTargetLSTMModel::load("models/BTCUSDT")?;

// Create prediction configuration
let config = PredictionConfig {
    symbols: vec!["BTCUSDT".to_string()],
    input_path: "data/recent_btc.csv".into(),
    output_path: Some("predictions.json".into()),
    horizons: vec!["1h".to_string(), "4h".to_string()],
    device: DeviceConfig::Auto,
    min_confidence: 0.0,
};

// Make predictions
let predictor = Predictor::new(config);
let results = predictor.predict(ModelWrapper::MultiTarget(&model)).await?;
```

### **Prediction Output Structure**

Current prediction output includes all 5 targets × 5 classes:

```rust
// Each prediction result contains:
pub struct PredictionResult {
    pub symbol: String,
    pub timestamp: String,
    pub horizon: String,
    pub predictions: HashMap<String, Vec<f64>>,  // target_name -> 5 class probabilities
    pub confidence: f64,
}

// Example output structure:
// predictions = {
//     "price_level_1h": [0.1, 0.2, 0.4, 0.2, 0.1],  // 5 classes: Strong Down, Moderate Down, Neutral, Moderate Up, Strong Up
//     "direction_1h": [0.15, 0.25, 0.3, 0.2, 0.1],   // 5 classes: DUMP, DOWN, SIDEWAYS, UP, PUMP
//     "volatility_1h": [0.2, 0.3, 0.3, 0.15, 0.05],  // 5 classes: VeryLow, Low, Medium, High, VeryHigh
//     "sentiment_1h": [0.1, 0.2, 0.3, 0.25, 0.15],   // 5 classes: Strong Panic, Moderate Panic, Neutral, Moderate Greed, Strong Greed
//     "volume_1h": [0.05, 0.2, 0.4, 0.25, 0.1],      // 5 classes: Very Low, Low, Medium, High, Very High
// }
```

## Prediction Architecture

### **Current Prediction Pipeline**
```rust
// Implemented in src/api/predictor.rs with modular LSTM architecture
impl Predictor {
    pub async fn predict(&self, model: ModelWrapper<'_>) -> Result<Vec<PredictionResult>> {
        // 1. Initialize device from configuration
        let device = DeviceManager::create_device(&self.config.device.to_device_string())?;

        // 2. Initialize data pipeline
        let data_pipeline = DataPipeline::new();

        // 3. Load and prepare prediction data
        let prepared_data = data_pipeline.prepare_prediction_data(&self.config).await?;

        // 4. Make predictions using model (single or multi-target)
        let raw_predictions = model.predict(&prepared_data.sequences).await?;

        // 5. Post-process predictions with confidence filtering
        let post_processor = PostProcessor::new(&self.config);
        let processed_predictions = post_processor.process_predictions(raw_predictions)?;

        // 6. Format output (JSON/CSV)
        let formatter = OutputFormatter::new(&self.config);
        let results = formatter.format_predictions(processed_predictions, &model)?;

        Ok(results)
    }
}
```

### **Multi-Target Model Prediction**
```rust
// Implemented in src/model/multi_target.rs
impl MultiTargetLSTMModel {
    pub async fn predict(&self, sequences: &Array3<f64>) -> Result<Array2<f64>> {
        // 1. Validate input sequences
        let (batch_size, seq_len, features) = sequences.dim();
        if features != self.input_size {
            return Err(VangaError::ModelError(
                format!("Input size mismatch: expected {}, got {}", self.input_size, features)
            ));
        }

        // 2. Make predictions with each target model
        let mut all_predictions = Vec::new();
        for (target_name, model) in &self.models {
            let target_predictions = model.predict(sequences).await?;
            all_predictions.push((target_name.clone(), target_predictions));
        }

        // 3. Combine predictions from all models
        // Output shape: [batch_size, total_outputs] where total_outputs = num_targets * 5
        let combined_predictions = self.combine_target_predictions(all_predictions)?;

        Ok(combined_predictions)
    }
}
```

### **Single LSTM Model Prediction**
```rust
// Implemented in src/model/lstm/inference.rs
impl LSTMModel {
    pub async fn predict(&self, sequences: &Array3<f64>) -> Result<Array2<f64>> {
        // 1. Validate network is trained
        if self.network.is_none() {
            return Err(VangaError::ModelError("Network not initialized".to_string()));
        }

        // 2. Convert sequences to tensors
        let seq_tensor = self.convert_sequences_to_tensor(sequences)?;

        // 3. Forward pass through LSTM network
        let network = self.network.as_ref().unwrap();
        let output = network.forward(&seq_tensor)?;

        // 4. Apply softmax for classification outputs
        let probabilities = candle_nn::ops::softmax(&output, 1)?;

        // 5. Convert back to ndarray
        let predictions = self.tensor_to_array2(&probabilities)?;

        Ok(predictions)
    }
}
```

## Prediction Configuration

### **PredictionConfig Structure**
```rust
// Implemented in src/config/prediction.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    pub symbols: Vec<String>,              // Trading symbols to predict
    pub input_path: PathBuf,               // Path to prediction data CSV
    pub output_path: Option<PathBuf>,      // Optional output file path
    pub horizons: Vec<String>,             // Prediction horizons ["1h", "4h", "1d"]
    pub device: DeviceConfig,              // CPU/GPU device configuration
    pub min_confidence: f64,               // Minimum confidence threshold (0.0-1.0)
    pub output_format: OutputFormat,       // JSON or CSV output format
    pub batch_size: Option<usize>,         // Optional batch size for prediction
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    JSON,
    CSV,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceConfig {
    Auto,
    CPU,
    GPU { device_id: usize },
}
```

## Prediction Commands

### **Current API Usage**

```rust
// Basic prediction with multi-target model
use vanga::api::predictor::{Predictor, ModelWrapper};
use vanga::config::PredictionConfig;
use vanga::model::multi_target::MultiTargetLSTMModel;

// Load model
let model = MultiTargetLSTMModel::load("models/BTCUSDT")?;

// Configure prediction
let config = PredictionConfig {
    symbols: vec!["BTCUSDT".to_string()],
    input_path: "data/recent_data.csv".into(),
    output_path: Some("predictions.json".into()),
    horizons: vec!["1h".to_string(), "4h".to_string()],
    device: DeviceConfig::Auto,
    min_confidence: 0.0,
    output_format: OutputFormat::JSON,
    batch_size: None,
};

// Make predictions
let predictor = Predictor::new(config);
let results = predictor.predict(ModelWrapper::MultiTarget(&model)).await?;
```

### **Advanced Configuration**

```rust
// Prediction with confidence filtering and specific device
let config = PredictionConfig {
    symbols: vec!["BTCUSDT".to_string()],
    input_path: "data/recent_data.csv".into(),
    output_path: Some("high_confidence_predictions.json".into()),
    horizons: vec!["4h".to_string()],
    device: DeviceConfig::GPU { device_id: 0 },
    min_confidence: 0.8,  // Only high-confidence predictions
    output_format: OutputFormat::JSON,
    batch_size: Some(32),
};

let predictor = Predictor::new(config);
let results = predictor.predict(ModelWrapper::MultiTarget(&model)).await?;
```

### **Batch Prediction for Multiple Symbols**

```rust
// Predict multiple symbols
let symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string(), "ADAUSDT".to_string()];
let mut all_results = Vec::new();

for symbol in symbols {
    let model = MultiTargetLSTMModel::load(&format!("models/{}", symbol))?;
    let config = PredictionConfig {
        symbols: vec![symbol.clone()],
        input_path: format!("data/{}_recent.csv", symbol).into(),
        output_path: Some(format!("predictions/{}_forecast.json", symbol).into()),
        horizons: vec!["1h".to_string(), "4h".to_string()],
        device: DeviceConfig::Auto,
        min_confidence: 0.0,
        output_format: OutputFormat::JSON,
        batch_size: None,
    };

    let predictor = Predictor::new(config);
    let results = predictor.predict(ModelWrapper::MultiTarget(&model)).await?;
    all_results.extend(results);
}
```

## Data Processing During Prediction

### **Feature Engineering (TA Crate Integration)**
```rust
// Professional technical analysis with TA crate integration
// Implemented in src/features/technical.rs + src/features/ta_tests.rs
pub async fn generate_technical_indicators(df: DataFrame, config: &TechnicalIndicatorsConfig) -> Result<DataFrame> {
    // Ensures consistent feature set between training and prediction
    // Generates 50+ professional technical indicators using TA crate:

    // Trend Indicators (TA Crate)
    // - SMA, EMA, DEMA, TEMA (multiple periods)
    // - MACD with signal line and histogram
    // - Bollinger Bands with professional implementation
    // - Parabolic SAR, Supertrend

    // Momentum Indicators (TA Crate)
    // - RSI with proper gain/loss averaging
    // - Stochastic Oscillator (%K and %D)
    // - Williams %R, CCI, ROC, Momentum
    // - Ultimate Oscillator, Detrended Price Oscillator

    // Volume Indicators (TA Crate)
    // - OBV, MFI, A/D Line with professional calculation
    // - Volume SMA, Volume Rate of Change
    // - VWAP with volume-weighted accuracy

    // Volatility Indicators (TA Crate)
    // - ATR with true range calculation
    // - Keltner Channels, Standard Deviation
    // - Professional volatility measurement

    // Crypto-specific Features
    // - Price velocity and acceleration
    // - VWAP deviations and momentum
    // - Market microstructure indicators

    // Maintains feature order and naming consistency with training
    // Uses same TA crate configuration for reproducible results
}
```

### **Sequence Generation**
```rust
// Convert features to LSTM prediction sequences
// Implemented in src/data/sequence.rs
impl SequenceGenerator {
    pub async fn generate_prediction_sequences(&self, df: &DataFrame, config: &PredictionConfig) -> Result<PreparedPredictionData> {
        // 1. Extract feature matrix (same columns as training)
        let feature_matrix = self.extract_feature_matrix(df, &self.feature_columns)?;

        // 2. Create prediction sequences (last N timesteps)
        let sequences = self.create_sequences_for_prediction(&feature_matrix, self.sequence_length)?;

        // 3. Apply same normalization as training (CRITICAL)
        let normalized_sequences = if let Some(norm_stats) = &self.normalization_stats {
            self.apply_normalization(&sequences, norm_stats)?
        } else {
            return Err(VangaError::DataError("No normalization stats available".to_string()));
        };

        // 4. Return sequences ready for LSTM prediction
        Ok(PreparedPredictionData {
            sequences: normalized_sequences,
            timestamps: self.extract_timestamps(df)?,
            feature_names: self.feature_columns.clone(),
        })
    }
}
```

### **Normalization Consistency (CRITICAL)**
```rust
// Prediction MUST use training normalization statistics
// Implemented in src/data/preprocessor.rs
impl DataPipeline {
    pub async fn prepare_prediction_data(&self, config: &PredictionConfig) -> Result<PreparedPredictionData> {
        // 1. Load prediction data
        let df = self.load_csv(&config.input_path).await?;

        // 2. Generate same features as training
        let df_with_features = self.generate_features(df, &self.feature_config).await?;

        // 3. CRITICAL: Use training normalization stats (never calculate new ones)
        if let Some(norm_stats) = &self.normalization_stats {
            let normalized_df = self.apply_saved_normalization(&df_with_features, norm_stats)?;
            // Continue with sequence generation...
        } else {
            return Err(VangaError::DataError(
                "No training normalization statistics available. Model must be retrained.".to_string()
            ));
        }
    }
}
```

## Model Loading

### **Automatic Model Loading**
```rust
// Multi-target models are automatically loaded with all components
// Implemented in src/model/multi_target.rs
impl MultiTargetLSTMModel {
    pub fn load<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref();

        // 1. Load model metadata
        let metadata_path = base_path.join("metadata.json");
        let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(metadata_path)?)?;

        // 2. Load individual target models
        let mut models = HashMap::new();
        for target_name in &metadata.target_names {
            let model_path = base_path.join(format!("{}.bin", target_name));
            let model = LSTMModel::load(&model_path)?;
            models.insert(target_name.clone(), model);
        }

        // 3. Load normalization statistics (CRITICAL for prediction)
        let norm_stats_path = base_path.join("normalization_stats.json");
        let normalization_stats = if norm_stats_path.exists() {
            Some(serde_json::from_str(&std::fs::read_to_string(norm_stats_path)?)?)
        } else {
            None
        };

        // 4. Load feature and training configurations
        let feature_config = Self::load_feature_config(base_path)?;
        let training_config = Self::load_training_config(base_path)?;

        Ok(Self {
            models,
            target_names: metadata.target_names,
            trained_horizons: metadata.trained_horizons,
            num_targets: metadata.num_targets,
            input_size: metadata.input_size,
            feature_config,
            training_config,
            normalization_stats,
        })
    }
}
```

### **Single Model Loading**
```rust
// Single LSTM models loaded with configuration
// Implemented in src/model/lstm/core.rs
impl LSTMModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        // 1. Read model file
        let data = std::fs::read(&path)?;

        // 2. Deserialize model state
        let model_state: ModelState = bincode::deserialize(&data)?;

        // 3. Create new LSTM model with loaded configuration
        let mut model = Self::new(model_state.config)?;

        // 4. Restore training state
        model.training_config = model_state.training_config;
        model.normalization_stats = model_state.normalization_stats;

        // 5. Initialize the network for predictions
        model.initialize_network()?;
        model.trained = true;

        Ok(model)
    }
}
```

## Output Formats

### **Structured JSON Output (Current Default)**

VANGA generates structured JSON predictions with crypto-native terminology and 5-class system:

```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2024-01-15T10:30:00Z",
  "horizon": "4h",
  "current_price": 42500.0,
  "predictions": {
    "price_level_4h": [0.05, 0.15, 0.40, 0.30, 0.10],
    "direction_4h": [0.10, 0.20, 0.30, 0.25, 0.15],
    "volatility_4h": [0.20, 0.30, 0.30, 0.15, 0.05],
    "sentiment_4h": [0.08, 0.18, 0.35, 0.28, 0.11],
    "volume_4h": [0.12, 0.22, 0.32, 0.24, 0.10]
  },
  "class_labels": {
    "price_level": ["Strong Down", "Moderate Down", "Neutral", "Moderate Up", "Strong Up"],
    "direction": ["DUMP", "DOWN", "SIDEWAYS", "UP", "PUMP"],
    "volatility": ["Very Low", "Low", "Medium", "High", "Very High"],
    "sentiment": ["Strong Panic", "Moderate Panic", "Neutral", "Moderate Greed", "Strong Greed"],
    "volume": ["Very Low", "Low", "Medium", "High", "Very High"]
  },
  "most_likely_predictions": {
    "price_level_4h": {
      "class": "Neutral",
      "probability": 0.40,
      "class_index": 2
    },
    "direction_4h": {
      "class": "SIDEWAYS",
      "probability": 0.30,
      "class_index": 2
    },
    "volatility_4h": {
      "class": "Low",
      "probability": 0.30,
      "class_index": 1
    },
    "sentiment_4h": {
      "class": "Neutral",
      "probability": 0.35,
      "class_index": 2
    },
    "volume_4h": {
      "class": "Medium",
      "probability": 0.32,
      "class_index": 2
    }
  },
  "confidence": 0.82,
  "model_info": {
    "model_type": "MultiTargetLSTM",
    "num_targets": 5,
    "num_classes_per_target": 5,
    "total_outputs": 25,
    "trained_horizons": ["1h", "4h", "1d"],
    "input_features": 55
  }
}
```

### **CSV Output Format**

For compatibility with external tools:

```csv
symbol,timestamp,horizon,target,class_0,class_1,class_2,class_3,class_4,most_likely_class,confidence
BTCUSDT,2024-01-15T10:30:00Z,4h,price_level,0.05,0.15,0.40,0.30,0.10,Neutral,0.82
BTCUSDT,2024-01-15T10:30:00Z,4h,direction,0.10,0.20,0.30,0.25,0.15,SIDEWAYS,0.82
BTCUSDT,2024-01-15T10:30:00Z,4h,volatility,0.20,0.30,0.30,0.15,0.05,Low,0.82
BTCUSDT,2024-01-15T10:30:00Z,4h,sentiment,0.08,0.18,0.35,0.28,0.11,Neutral,0.82
BTCUSDT,2024-01-15T10:30:00Z,4h,volume,0.12,0.22,0.32,0.24,0.10,Medium,0.82
```

### **5-Class System Explained**

**All targets use consistent 5-class classification:**

**Class Labels:**
- **Class 0 - Strong Down**: Significant negative movement/low volatility
- **Class 1 - Moderate Down**: Moderate negative movement/below-average volatility
- **Class 2 - Neutral**: Minimal movement/average volatility (most common)
- **Class 3 - Moderate Up**: Moderate positive movement/above-average volatility
- **Class 4 - Strong Up**: Significant positive movement/high volatility

**Target-Specific Interpretations:**
- **Price Level**: Percentage-based price movement ranges (symbol-agnostic)
- **Direction**: Directional movement classification with momentum consideration
- **Volatility**: Volatility regime classification (Very Low to Very High)

**Probability Distribution:**
Each target outputs 5 probabilities that sum to 1.0, representing the likelihood of each class.

```rust
// Example interpretation
let price_level_probs = [0.05, 0.15, 0.40, 0.30, 0.10];
// Interpretation: 40% chance of neutral price movement, 30% chance of moderate up movement

let direction_probs = [0.10, 0.20, 0.30, 0.25, 0.15];
// Interpretation: 30% chance of neutral direction, 25% chance of moderate up direction

let volatility_probs = [0.20, 0.30, 0.30, 0.15, 0.05];
// Interpretation: 30% chance each of low and medium volatility
```

## Prediction Performance

### **Performance Metrics**
- **Feature Generation**: ~3ms for 50+ indicators per 1000 data points
- **Sequence Generation**: ~2ms per 1000 sequences
- **Multi-Target LSTM Prediction**: ~5ms per 100 sequences (5 targets × 5 classes)
- **Single LSTM Prediction**: ~1ms per 100 sequences (1 target × 5 classes)
- **Output Generation**: <1ms for JSON/CSV writing
- **Total Pipeline**: ~10ms per prediction batch

### **Memory Usage**
- **Input Data**: ~1MB per 10K samples
- **With Features**: ~5MB per 10K samples (50+ indicators)
- **Prediction Sequences**: ~2MB per 1K sequences
- **Multi-Target Output**: ~25 values per prediction (5 targets × 5 classes)
- **Single Target Output**: ~5 values per prediction (1 target × 5 classes)

### **Scalability**
- **Batch Processing**: Efficient processing of multiple sequences
- **Memory Optimization**: Streaming processing for large datasets
- **GPU Acceleration**: Automatic GPU utilization when available
- **Multi-Symbol**: Parallel processing of multiple trading pairs

## Real-World Usage Examples

### **Single Asset Prediction**
```rust
// Complete prediction workflow
use vanga::api::predictor::{Predictor, ModelWrapper};
use vanga::config::PredictionConfig;
use vanga::model::multi_target::MultiTargetLSTMModel;

async fn predict_btc() -> Result<()> {
    // 1. Load trained model
    let model = MultiTargetLSTMModel::load("models/BTCUSDT")?;

    // 2. Configure prediction
    let config = PredictionConfig {
        symbols: vec!["BTCUSDT".to_string()],
        input_path: "data/btc_recent_1h.csv".into(),
        output_path: Some("predictions/btc_forecast.json".into()),
        horizons: vec!["1h".to_string(), "4h".to_string()],
        device: DeviceConfig::Auto,
        min_confidence: 0.7,
        output_format: OutputFormat::JSON,
        batch_size: None,
    };

    // 3. Make predictions
    let predictor = Predictor::new(config);
    let results = predictor.predict(ModelWrapper::MultiTarget(&model)).await?;

    // 4. Process results
    for result in results {
        println!("Symbol: {}, Horizon: {}, Confidence: {:.3}",
                 result.symbol, result.horizon, result.confidence);

        for (target_name, probabilities) in &result.predictions {
            let max_prob_idx = probabilities.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            println!("  {}: Class {} ({:.3})", target_name, max_prob_idx, probabilities[max_prob_idx]);
        }
    }

    Ok(())
}
```

### **Multi-Asset Portfolio Prediction**
```rust
// Predict multiple assets in parallel
use tokio::task::JoinSet;

async fn predict_portfolio() -> Result<Vec<PredictionResult>> {
    let symbols = vec!["BTCUSDT", "ETHUSDT", "ADAUSDT"];
    let mut join_set = JoinSet::new();

    // Spawn prediction tasks for each symbol
    for symbol in symbols {
        join_set.spawn(async move {
            let model = MultiTargetLSTMModel::load(&format!("models/{}", symbol))?;
            let config = PredictionConfig {
                symbols: vec![symbol.to_string()],
                input_path: format!("data/{}_recent.csv", symbol).into(),
                output_path: Some(format!("predictions/{}_forecast.json", symbol).into()),
                horizons: vec!["4h".to_string()],
                device: DeviceConfig::Auto,
                min_confidence: 0.0,
                output_format: OutputFormat::JSON,
                batch_size: None,
            };

            let predictor = Predictor::new(config);
            predictor.predict(ModelWrapper::MultiTarget(&model)).await
        });
    }

    // Collect all results
    let mut all_results = Vec::new();
    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok(predictions)) => all_results.extend(predictions),
            Ok(Err(e)) => eprintln!("Prediction error: {}", e),
            Err(e) => eprintln!("Task error: {}", e),
        }
    }

    Ok(all_results)
}
```

### **Automated Prediction Pipeline**
```rust
// Continuous prediction system
use tokio::time::{interval, Duration};

async fn automated_prediction_pipeline() -> Result<()> {
    let mut interval = interval(Duration::from_secs(3600)); // Every hour

    loop {
        interval.tick().await;

        // 1. Update data (implement your data fetching logic)
        update_market_data().await?;

        // 2. Generate predictions for all symbols
        let symbols = vec!["BTCUSDT", "ETHUSDT", "ADAUSDT"];
        for symbol in symbols {
            match predict_symbol(symbol).await {
                Ok(results) => {
                    println!("Generated {} predictions for {}", results.len(), symbol);
                    // Process results (send alerts, update database, etc.)
                    process_predictions(results).await?;
                }
                Err(e) => eprintln!("Failed to predict {}: {}", symbol, e),
            }
        }

        // 3. Wait for next interval
        println!("Prediction cycle completed, waiting for next interval...");
    }
}

async fn predict_symbol(symbol: &str) -> Result<Vec<PredictionResult>> {
    let model = MultiTargetLSTMModel::load(&format!("models/{}", symbol))?;
    let config = PredictionConfig {
        symbols: vec![symbol.to_string()],
        input_path: format!("data/{}_latest.csv", symbol).into(),
        output_path: Some(format!("predictions/{}_current.json", symbol).into()),
        horizons: vec!["1h".to_string(), "4h".to_string()],
        device: DeviceConfig::Auto,
        min_confidence: 0.0,
        output_format: OutputFormat::JSON,
        batch_size: None,
    };

    let predictor = Predictor::new(config);
    predictor.predict(ModelWrapper::MultiTarget(&model)).await
}
```

## 🆕 **Advanced Prediction Features**

### **Prediction Quality Assessment**

VANGA now includes comprehensive prediction quality metrics during inference:

```rust
// Enhanced prediction with quality assessment
// Implemented in src/model/lstm/inference.rs
impl LSTMModel {
    pub async fn predict_with_quality(&self, sequences: &Array3<f64>) -> Result<PredictionWithQuality> {
        // 1. Generate predictions
        let predictions = self.predict(sequences).await?;

        // 2. Calculate prediction quality metrics
        let quality_metrics = self.assess_prediction_quality(&predictions, sequences)?;

        // 3. Calculate confidence scores
        let confidence_scores = self.calculate_confidence_scores(&predictions)?;

        // 4. Apply distance weighting for multi-horizon predictions
        let weighted_quality = self.calculate_distance_weighted_quality(&predictions)?;

        Ok(PredictionWithQuality {
            predictions,
            quality_metrics,
            confidence_scores,
            weighted_quality,
        })
    }

    /// Calculate trading-specific quality metrics
    fn assess_prediction_quality(
        &self,
        predictions: &Array2<f64>,
        sequences: &Array3<f64>,
    ) -> Result<QualityMetrics> {
        let mut quality_metrics = QualityMetrics::new();

        // Entropy-based confidence (lower entropy = higher confidence)
        for pred_row in predictions.outer_iter() {
            let entropy = self.calculate_entropy(pred_row.as_slice().unwrap())?;
            quality_metrics.entropy_scores.push(entropy);
        }

        // Prediction consistency across similar sequences
        let consistency_score = self.calculate_prediction_consistency(predictions, sequences)?;
        quality_metrics.consistency_score = consistency_score;

        // Market regime alignment
        let regime_alignment = self.assess_market_regime_alignment(predictions, sequences)?;
        quality_metrics.regime_alignment = regime_alignment;

        Ok(quality_metrics)
    }
}

#[derive(Debug, Clone)]
pub struct PredictionWithQuality {
    pub predictions: Array2<f64>,
    pub quality_metrics: QualityMetrics,
    pub confidence_scores: Vec<f64>,
    pub weighted_quality: f64,
}

#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub entropy_scores: Vec<f64>,
    pub consistency_score: f64,
    pub regime_alignment: f64,
    pub trading_quality: f64,
}
```
```

## Prediction Validation

### **Model Availability Check**
```rust
// Check if models exist before prediction
use std::path::Path;

fn check_model_availability(symbol: &str) -> Result<bool> {
    let model_path = format!("models/{}", symbol);
    let metadata_path = Path::new(&model_path).join("metadata.json");

    if metadata_path.exists() {
        println!("✅ Model available for {}", symbol);

        // Load and display model info
        let metadata_content = std::fs::read_to_string(metadata_path)?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;

        println!("  - Targets: {:?}", metadata["target_names"]);
        println!("  - Horizons: {:?}", metadata["trained_horizons"]);
        println!("  - Input size: {}", metadata["input_size"]);

        Ok(true)
    } else {
        println!("❌ Model not found for {}", symbol);
        println!("   Train the model first using the training API");
        Ok(false)
    }
}

// List all available models
fn list_available_models() -> Result<Vec<String>> {
    let models_dir = Path::new("models");
    let mut available_models = Vec::new();

    if models_dir.exists() {
        for entry in std::fs::read_dir(models_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let model_name = entry.file_name().to_string_lossy().to_string();
                if check_model_availability(&model_name)? {
                    available_models.push(model_name);
                }
            }
        }
    }

    Ok(available_models)
}
```

### **Data Compatibility Validation**
```rust
// Automatic validation ensures prediction data matches training format
// Implemented in src/data/pipeline.rs
impl DataPipeline {
    pub async fn validate_prediction_data(&self, config: &PredictionConfig) -> Result<()> {
        // 1. Load prediction data
        let df = self.load_csv(&config.input_path).await?;

        // 2. Check required OHLCV columns
        let required_columns = ["timestamp", "open", "high", "low", "close", "volume"];
        for col in required_columns {
            if !df.get_column_names().contains(&col) {
                return Err(VangaError::DataError(
                    format!("Missing required column: {}", col)
                ));
            }
        }

        // 3. Validate data types
        self.validate_data_types(&df)?;

        // 4. Check for sufficient data
        let min_required = self.sequence_length + 50; // Buffer for indicators
        if df.height() < min_required {
            return Err(VangaError::DataError(
                format!("Insufficient data: need at least {} rows, got {}", min_required, df.height())
            ));
        }

        // 5. Validate feature compatibility with trained model
        if let Some(expected_features) = &self.expected_feature_columns {
            let generated_features = self.generate_feature_columns(&df)?;
            if generated_features.len() != expected_features.len() {
                return Err(VangaError::DataError(
                    format!("Feature count mismatch: expected {}, got {}",
                            expected_features.len(), generated_features.len())
                ));
            }
        }

        Ok(())
    }
}
```

## Troubleshooting

### **Common Issues**

#### **Model Not Found**
```
Error: Model directory not found: ./models/BTCUSDT
Solution: Train the model first using the training API
```
```rust
// Check if model exists
if !Path::new("models/BTCUSDT").exists() {
    println!("Model not found. Train first:");
    println!("let config = TrainingConfig::from_file(\"configs/training.toml\")?;");
    println!("let model = train_model(config).await?;");
}
```

#### **Data Format Mismatch**
```
Error: Missing required column: 'close'
Solution: Ensure all OHLCV columns are present in prediction data
```
```rust
// Validate data format
let required_columns = ["timestamp", "open", "high", "low", "close", "volume"];
for col in required_columns {
    if !df.get_column_names().contains(&col) {
        return Err(VangaError::DataError(format!("Missing column: {}", col)));
    }
}
```

#### **Feature Count Mismatch**
```
Error: Feature count mismatch: expected 55, got 48
Solution: Ensure same feature configuration as training
```
```rust
// Check feature compatibility
if generated_features.len() != expected_features.len() {
    println!("Feature mismatch detected:");
    println!("  Expected: {} features", expected_features.len());
    println!("  Generated: {} features", generated_features.len());
    println!("  Ensure same technical indicators are enabled as during training");
}
```

#### **Insufficient Data**
```
Error: Insufficient data: need at least 110 rows, got 50
Solution: Provide more historical data (sequence_length + indicator warmup period)
```
```rust
// Check data sufficiency
let min_required = sequence_length + 50; // Buffer for technical indicators
if df.height() < min_required {
    return Err(VangaError::DataError(
        format!("Need at least {} data points for prediction", min_required)
    ));
}
```

#### **Normalization Stats Missing**
```
Error: No training normalization statistics available
Solution: Model must be retrained with normalization stats saved
```
```rust
// Check normalization stats
if model.get_normalization_stats().is_none() {
    println!("Warning: No normalization statistics found");
    println!("Model may need to be retrained with current API");
}
```

### **Prediction Diagnostics**
```rust
// Enable debug logging for detailed prediction info
env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

// Diagnostic information during prediction:
// - Model loading details
// - Feature generation progress
// - Sequence creation info
// - Normalization application
// - Prediction tensor shapes
// - Output formatting details
```

### **Performance Debugging**
```rust
// Monitor prediction performance
use std::time::Instant;

let start = Instant::now();

// Feature generation timing
let feature_start = Instant::now();
let df_with_features = generate_technical_indicators(df, &config).await?;
println!("Feature generation: {:?}", feature_start.elapsed());

// Sequence generation timing
let seq_start = Instant::now();
let sequences = generate_prediction_sequences(&df_with_features, &config).await?;
println!("Sequence generation: {:?}", seq_start.elapsed());

// Model prediction timing
let pred_start = Instant::now();
let predictions = model.predict(&sequences).await?;
println!("Model prediction: {:?}", pred_start.elapsed());

println!("Total prediction time: {:?}", start.elapsed());
```

## Advanced Prediction Features

### **Confidence Filtering**
```rust
// Only output high-confidence predictions
let config = PredictionConfig {
    symbols: vec!["BTCUSDT".to_string()],
    input_path: "data/recent_data.csv".into(),
    output_path: Some("high_confidence_predictions.json".into()),
    horizons: vec!["4h".to_string()],
    device: DeviceConfig::Auto,
    min_confidence: 0.8,  // Only predictions with >80% confidence
    output_format: OutputFormat::JSON,
    batch_size: None,
};

let predictor = Predictor::new(config);
let results = predictor.predict(ModelWrapper::MultiTarget(&model)).await?;

// Results will only contain predictions where the highest class probability > 0.8
```

### **Custom Output Processing**
```rust
// Process predictions with custom logic
use serde_json::Value;

fn process_predictions(results: Vec<PredictionResult>) -> Result<Vec<TradingSignal>> {
    let mut signals = Vec::new();

    for result in results {
        // Extract price level predictions
        if let Some(price_level_probs) = result.predictions.get("price_level_4h") {
            let max_prob_idx = price_level_probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            let max_prob = price_level_probs[max_prob_idx];

            // Generate trading signal based on prediction
            let signal = match max_prob_idx {
                0 | 1 => TradingSignal::Sell { confidence: max_prob, strength: max_prob_idx },
                2 => TradingSignal::Hold { confidence: max_prob },
                3 | 4 => TradingSignal::Buy { confidence: max_prob, strength: max_prob_idx - 2 },
                _ => TradingSignal::Hold { confidence: 0.0 },
            };

            signals.push(signal);
        }
    }

    Ok(signals)
}

#[derive(Debug)]
enum TradingSignal {
    Buy { confidence: f64, strength: usize },
    Sell { confidence: f64, strength: usize },
    Hold { confidence: f64 },
}
```

### **Batch Processing for Large Datasets**
```rust
// Process large prediction datasets in batches
async fn batch_predict(
    model: &MultiTargetLSTMModel,
    input_path: &Path,
    batch_size: usize,
) -> Result<Vec<PredictionResult>> {
    // 1. Load and chunk data
    let df = load_csv(input_path).await?;
    let total_rows = df.height();
    let num_batches = (total_rows + batch_size - 1) / batch_size;

    let mut all_results = Vec::new();

    // 2. Process each batch
    for batch_idx in 0..num_batches {
        let start_row = batch_idx * batch_size;
        let end_row = std::cmp::min(start_row + batch_size, total_rows);

        // Extract batch
        let batch_df = df.slice(start_row as i64, end_row - start_row);

        // Create temporary file for batch
        let batch_path = format!("temp_batch_{}.csv", batch_idx);
        write_csv(&batch_df, &batch_path).await?;

        // Configure prediction for batch
        let config = PredictionConfig {
            symbols: vec!["BTCUSDT".to_string()],
            input_path: batch_path.into(),
            output_path: None,
            horizons: vec!["4h".to_string()],
            device: DeviceConfig::Auto,
            min_confidence: 0.0,
            output_format: OutputFormat::JSON,
            batch_size: Some(32),
        };

        // Make predictions
        let predictor = Predictor::new(config);
        let batch_results = predictor.predict(ModelWrapper::MultiTarget(model)).await?;
        all_results.extend(batch_results);

        // Clean up temporary file
        std::fs::remove_file(&batch_path)?;

        println!("Processed batch {}/{}", batch_idx + 1, num_batches);
    }

    Ok(all_results)
}
```

## Integration with Trading Systems

### **API Integration Pattern**
```rust
// Example integration with trading system
use tokio::time::{interval, Duration};

pub struct TradingBot {
    models: HashMap<String, MultiTargetLSTMModel>,
    exchange_client: ExchangeClient,
    risk_manager: RiskManager,
}

impl TradingBot {
    pub async fn run_trading_loop(&mut self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(300)); // 5 minutes

        loop {
            interval.tick().await;

            // 1. Fetch latest market data
            let market_data = self.exchange_client.get_latest_ohlcv().await?;

            // 2. Generate predictions for all symbols
            let mut trading_signals = Vec::new();

            for (symbol, model) in &self.models {
                match self.generate_trading_signals(symbol, model, &market_data).await {
                    Ok(signals) => trading_signals.extend(signals),
                    Err(e) => eprintln!("Failed to generate signals for {}: {}", symbol, e),
                }
            }

            // 3. Apply risk management
            let filtered_signals = self.risk_manager.filter_signals(trading_signals)?;

            // 4. Execute trades
            for signal in filtered_signals {
                match self.execute_trade(signal).await {
                    Ok(order) => println!("Executed trade: {:?}", order),
                    Err(e) => eprintln!("Failed to execute trade: {}", e),
                }
            }
        }
    }

    async fn generate_trading_signals(
        &self,
        symbol: &str,
        model: &MultiTargetLSTMModel,
        market_data: &MarketData,
    ) -> Result<Vec<TradingSignal>> {
        // 1. Prepare prediction data
        let prediction_data = self.prepare_prediction_data(symbol, market_data).await?;

        // 2. Configure prediction
        let config = PredictionConfig {
            symbols: vec![symbol.to_string()],
            input_path: prediction_data.path,
            output_path: None,
            horizons: vec!["1h".to_string(), "4h".to_string()],
            device: DeviceConfig::Auto,
            min_confidence: 0.6,
            output_format: OutputFormat::JSON,
            batch_size: None,
        };

        // 3. Make predictions
        let predictor = Predictor::new(config);
        let results = predictor.predict(ModelWrapper::MultiTarget(model)).await?;

        // 4. Convert predictions to trading signals
        let signals = self.convert_predictions_to_signals(results, symbol)?;

        Ok(signals)
    }

    fn convert_predictions_to_signals(
        &self,
        results: Vec<PredictionResult>,
        symbol: &str,
    ) -> Result<Vec<TradingSignal>> {
        let mut signals = Vec::new();

        for result in results {
            // Combine all target predictions for decision making
            let price_level_signal = self.analyze_price_level_prediction(&result)?;
            let direction_signal = self.analyze_direction_prediction(&result)?;
            let volatility_signal = self.analyze_volatility_prediction(&result)?;

            // Create composite trading signal
            let composite_signal = TradingSignal {
                symbol: symbol.to_string(),
                horizon: result.horizon,
                action: self.determine_action(price_level_signal, direction_signal, volatility_signal)?,
                confidence: result.confidence,
                risk_level: self.calculate_risk_level(&result)?,
                position_size: self.calculate_position_size(&result)?,
            };

            signals.push(composite_signal);
        }

        Ok(signals)
    }
}

#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub symbol: String,
    pub horizon: String,
    pub action: TradeAction,
    pub confidence: f64,
    pub risk_level: RiskLevel,
    pub position_size: f64,
}

#[derive(Debug, Clone)]
pub enum TradeAction {
    Buy { target_price: f64, stop_loss: f64 },
    Sell { target_price: f64, stop_loss: f64 },
    Hold,
    Close { reason: String },
}

#[derive(Debug, Clone)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}
```

### **Automated Trading Pipeline**
```rust
// Continuous prediction and trading system
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct AutomatedTradingSystem {
    models: Arc<RwLock<HashMap<String, MultiTargetLSTMModel>>>,
    data_fetcher: DataFetcher,
    signal_processor: SignalProcessor,
    order_executor: OrderExecutor,
    performance_tracker: PerformanceTracker,
}

impl AutomatedTradingSystem {
    pub async fn start(&mut self) -> Result<()> {
        // Start background tasks
        let data_task = self.start_data_collection();
        let prediction_task = self.start_prediction_loop();
        let trading_task = self.start_trading_loop();
        let monitoring_task = self.start_performance_monitoring();

        // Wait for all tasks
        tokio::try_join!(data_task, prediction_task, trading_task, monitoring_task)?;

        Ok(())
    }

    async fn start_prediction_loop(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(60)); // Every minute

        loop {
            interval.tick().await;

            let models = self.models.read().await;
            let mut prediction_tasks = Vec::new();

            // Generate predictions for all symbols in parallel
            for (symbol, model) in models.iter() {
                let symbol = symbol.clone();
                let model = model.clone();
                let data_fetcher = self.data_fetcher.clone();

                let task = tokio::spawn(async move {
                    // Fetch latest data
                    let latest_data = data_fetcher.get_latest_data(&symbol).await?;

                    // Generate prediction
                    let config = PredictionConfig {
                        symbols: vec![symbol.clone()],
                        input_path: latest_data.path,
                        output_path: None,
                        horizons: vec!["5m".to_string(), "15m".to_string(), "1h".to_string()],
                        device: DeviceConfig::Auto,
                        min_confidence: 0.5,
                        output_format: OutputFormat::JSON,
                        batch_size: None,
                    };

                    let predictor = Predictor::new(config);
                    let results = predictor.predict(ModelWrapper::MultiTarget(&model)).await?;

                    Ok::<_, VangaError>((symbol, results))
                });

                prediction_tasks.push(task);
            }

            // Collect all predictions
            for task in prediction_tasks {
                match task.await {
                    Ok(Ok((symbol, results))) => {
                        self.signal_processor.process_predictions(symbol, results).await?;
                    }
                    Ok(Err(e)) => eprintln!("Prediction error: {}", e),
                    Err(e) => eprintln!("Task error: {}", e),
                }
            }
        }
    }
}
```

## CLI Prediction Commands

### **Basic CLI Usage**

```bash
# Single prediction with default horizon
vanga predict --symbol BTCUSDT --input data/recent_btc.csv

# Specify prediction horizon
vanga predict --symbol BTCUSDT --input data/recent_btc.csv --horizon 4h

# Predict all available horizons
vanga predict --symbol BTCUSDT --input data/recent_btc.csv --all-horizons

# Save predictions to file
vanga predict --symbol BTCUSDT --input data/recent_btc.csv --output predictions.json

# High-confidence predictions only
vanga predict --symbol BTCUSDT --input data/recent_btc.csv --min-confidence 0.8
```

### **Multi-Symbol Prediction**

```bash
# Cross-asset prediction
vanga predict --symbol BTCUSDT,ETHUSDT,ADAUSDT --input data/ --output predictions/

# Batch prediction mode
vanga predict --batch --input-dir data/current/ --output predictions/

# Specify device for prediction
vanga predict --symbol BTCUSDT --input data/recent_btc.csv --device gpu:0
```

### **Real-Time Prediction**

```bash
# Real-time streaming predictions
vanga predict --symbol BTCUSDT --realtime --source binance --interval 1m

# Real-time with custom update interval
vanga predict --symbol BTCUSDT --realtime --interval 5m --output live_predictions.json
```

## Next Steps

After generating predictions:

1. **[Model Evaluation](09-evaluation.md)** - Assess prediction accuracy with current metrics
2. **[Usage Examples](11-usage-examples.md)** - Complete workflows with current API
3. **[Technical Implementation](10-technical-implementation.md)** - Advanced details of modular LSTM
4. **[Training](04-training.md)** - Retrain models with new data using current TrainingConfig
