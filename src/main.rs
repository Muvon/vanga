use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Duration;
use vanga::api;
use vanga::config::{PredictionConfig, TrainingConfig};
use vanga::realtime::{start_realtime_prediction, OutputFormat, RealtimeConfig};
use vanga::utils::error::{Result, VangaError};
use vanga::utils::file_discovery;

/// Training command parameters
struct TrainParams {
    symbol: String,
    data: PathBuf,
    fresh: bool,
    continue_training: bool,
    horizons: Option<Vec<String>>,
    config: Option<PathBuf>,
    device: String,
    attention: bool,
    tft: bool,
    batch: bool,
}

/// Prediction command parameters
struct PredictParams {
    symbol: String,
    input: PathBuf,
    horizon: Option<String>,
    all_horizons: bool,
    batch: bool,
    input_dir: Option<PathBuf>,
    output: Option<PathBuf>,
    min_confidence: Option<f64>,
    realtime: bool,
    source: Option<String>,
    interval: Option<String>,
    device: String,
}

#[derive(Parser)]
#[command(name = "vanga")]
#[command(about = "LSTM-based cryptocurrency forecasting system")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Train LSTM model for cryptocurrency forecasting
    Train {
        /// Trading symbol(s) - single: BTCUSDT or multiple: BTCUSDT,ETHUSDT,DOTUSDT
        #[arg(short, long)]
        symbol: String,

        /// Path to CSV data file or directory (auto-detects batch mode for directories)
        #[arg(short, long)]
        data: PathBuf,

        /// Start fresh training (ignore existing model)
        #[arg(long)]
        fresh: bool,

        /// Continue training existing model
        #[arg(long)]
        continue_training: bool,

        /// Prediction horizons (comma-separated: 1h,4h,1d,7d)
        #[arg(long, value_delimiter = ',')]
        horizons: Option<Vec<String>>,

        /// Training configuration file (includes all features and settings)
        #[arg(long)]
        config: Option<PathBuf>,

        /// Device to use for training (auto, cpu, gpu:0, metal:0)
        #[arg(long, default_value = "auto")]
        device: String,

        /// Enable attention mechanism for enhanced accuracy
        #[arg(long)]
        attention: bool,

        /// Enable TFT (Temporal Fusion Transformer) with Variable Selection and Quantile Regression
        #[arg(long)]
        tft: bool,

        /// Force batch training mode (optional - auto-detected when --data is directory)
        #[arg(long)]
        batch: bool,
    },

    /// Make predictions using trained model
    Predict {
        /// Trading symbol(s) - single: BTCUSDT or multiple: BTCUSDT,ETHUSDT for cross-asset prediction
        #[arg(
            short,
            long,
            help = "Trading symbol(s): single (BTCUSDT) or comma-separated for cross-asset (BTCUSDT,ETHUSDT)"
        )]
        symbol: String,

        /// Path to input CSV data file or directory containing symbol files
        #[arg(
            short,
            long,
            help = "Input CSV file or directory with {SYMBOL}.csv files for multi-symbol prediction"
        )]
        input: PathBuf,

        /// Prediction horizon (must match one used during training: e.g., 1h, 4h, 1d, 7d)
        #[arg(
            long,
            help = "Prediction horizon (must match one used during training). Use 'vanga models list' to see available horizons for each model."
        )]
        horizon: Option<String>,

        /// Predict all available horizons (shows predictions for all horizons the model was trained on)
        #[arg(
            long,
            help = "Predict all available horizons that the model was trained on"
        )]
        all_horizons: bool,

        /// Batch prediction mode
        #[arg(long)]
        batch: bool,

        /// Input directory for batch prediction
        #[arg(long)]
        input_dir: Option<PathBuf>,

        /// Output directory/file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Minimum confidence threshold
        #[arg(long)]
        min_confidence: Option<f64>,

        /// Real-time prediction mode
        #[arg(long)]
        realtime: bool,

        /// Data source for real-time mode
        #[arg(long)]
        source: Option<String>,

        /// Update interval for real-time mode
        #[arg(long)]
        interval: Option<String>,

        /// Device to use for prediction (auto, cpu, gpu:0, metal:0)
        #[arg(long, default_value = "auto")]
        device: String,
    },

    /// Model management commands
    Models {
        #[command(subcommand)]
        action: ModelCommands,
    },
}

#[derive(Subcommand)]
enum ModelCommands {
    /// List available models
    List,

    /// Evaluate model performance with optional backtesting
    Evaluate {
        /// Trading symbol (required for single symbol evaluation)
        #[arg(short, long)]
        symbol: Option<String>,

        /// Test data path
        #[arg(long)]
        test_data: PathBuf,

        /// Enable backtesting mode (train/test split)
        #[arg(long)]
        backtest: bool,

        /// Training data split ratio (0.0-1.0, default: 0.8)
        #[arg(long, default_value = "0.8")]
        train_split: f64,

        /// Batch evaluation for multiple symbols
        #[arg(long)]
        batch: bool,

        /// Symbols for batch evaluation (comma-separated)
        #[arg(long, value_delimiter = ',')]
        symbols: Option<Vec<String>>,
    },

    /// Compare multiple models
    Compare {
        /// Symbols to compare (comma-separated)
        #[arg(long, value_delimiter = ',')]
        symbols: Vec<String>,

        /// Evaluation metric
        #[arg(long, default_value = "accuracy")]
        metric: String,
    },

    /// Export model for deployment
    Export {
        /// Trading symbol
        #[arg(short, long)]
        symbol: String,

        /// Export format (onnx, msgpack, json)
        #[arg(long, default_value = "msgpack")]
        format: String,

        /// Output path
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Create model ensemble
    Ensemble {
        /// Symbols to ensemble (comma-separated)
        #[arg(long, value_delimiter = ',')]
        symbols: Vec<String>,

        /// Ensemble strategies (comma-separated)
        #[arg(long, value_delimiter = ',')]
        strategies: Vec<String>,

        /// Output ensemble name
        #[arg(short, long)]
        output: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    if cli.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    // Configure rayon for optimal CPU utilization
    configure_rayon_threads();

    let _monitor = PerformanceMonitor::new("VANGA Application");

    match cli.command {
        Commands::Train {
            symbol,
            data,
            fresh,
            continue_training,
            horizons,
            config,
            device,
            attention,
            tft,
            batch,
        } => {
            let params = TrainParams {
                symbol,
                data,
                fresh,
                continue_training,
                horizons,
                config,
                device,
                attention,
                tft,
                batch,
            };
            handle_train_command(params).await
        }

        Commands::Predict {
            symbol,
            input,
            horizon,
            all_horizons,
            batch,
            input_dir,
            output,
            min_confidence,
            realtime,
            source,
            interval,
            device,
        } => {
            let params = PredictParams {
                symbol,
                input,
                horizon,
                all_horizons,
                batch,
                input_dir,
                output,
                min_confidence,
                realtime,
                source,
                interval,
                device,
            };
            handle_predict_command(params).await
        }

        Commands::Models { action } => handle_model_commands(action).await,
    }
}

async fn handle_train_command(params: TrainParams) -> Result<()> {
    let monitor = PerformanceMonitor::new(&format!("Training {}", params.symbol));
    log::info!("Starting training for symbol: {}", params.symbol);

    // Parse symbols from --symbol parameter (supports comma-separated)
    let symbols: Vec<String> = params
        .symbol
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    log::info!("📋 Parsed {} symbol(s): {:?}", symbols.len(), symbols);

    // Determine if this is batch mode (multiple symbols OR directory data path OR explicit --batch)
    let is_batch_mode = symbols.len() > 1 || params.data.is_dir() || params.batch;

    if is_batch_mode {
        log::info!("🔄 Batch training mode detected");

        // Validate data path for batch mode
        file_discovery::validate_data_path_for_symbols(&params.data, &symbols)?;

        // Store symbol count for memory cleanup logic
        let symbol_count = symbols.len();

        // Process each symbol
        for symbol in symbols {
            log::info!("🚀 Training model for symbol: {}", symbol);

            // Resolve data file path using reusable utility
            let data_file_path =
                match file_discovery::resolve_symbol_data_path(&params.data, &symbol) {
                    Ok(path) => path,
                    Err(e) => {
                        log::error!("❌ Failed to resolve data file for {}: {}", symbol, e);
                        continue; // Skip this symbol and continue with others
                    }
                };

            log::info!("📂 Using data file: {}", data_file_path.display());

            // Create training config for this symbol
            let mut symbol_config = if let Some(config_path) = &params.config {
                log::info!("🔧 Loading training config from: {:?}", config_path);
                TrainingConfig::default()
                    .symbol(symbol.clone())
                    .data_path(data_file_path)
                    .with_config_from_file(config_path)?
            } else {
                TrainingConfig::default()
                    .symbol(symbol.clone())
                    .data_path(data_file_path)
            };

            // Apply other parameters
            if params.fresh {
                symbol_config = symbol_config.fresh_training(true);
            }
            if params.continue_training {
                symbol_config = symbol_config.continue_training(true);
            }
            if let Some(ref horizons) = params.horizons {
                symbol_config = symbol_config.horizons(horizons.clone());
            }

            // Apply device configuration
            symbol_config = symbol_config.with_device_config(&params.device)?;

            if params.attention {
                log::info!("🎯 Attention mechanism enabled for {}", symbol);
                symbol_config = symbol_config.with_attention_enabled(true);
            }
            if params.tft {
                log::info!(
                    "🔮 TFT (Temporal Fusion Transformer) enabled for {}",
                    symbol
                );
                symbol_config = symbol_config.with_tft_enabled(true);
            }

            monitor.checkpoint(&format!("Config prepared for {}", symbol));

            // Train the model
            match api::train_model(symbol_config.clone()).await {
                Ok(model) => {
                    monitor.checkpoint(&format!("Model trained for {}", symbol));
                    log::info!("✅ Successfully trained model for {}", symbol);

                    // Save model with proper error handling
                    let model_path = vanga::utils::model_path::get_model_path(&symbol);
                    let _ = vanga::utils::model_path::ensure_models_dir_exists();

                    match model.save(&model_path) {
                        Ok(()) => {
                            monitor.checkpoint(&format!("Model saved for {}", symbol));
                            log::info!("💾 Model saved to: {}", model_path.display());

                            // CRITICAL: Explicit memory cleanup to prevent accumulation
                            drop(model);

                            // Force immediate cleanup hint for batch training
                            std::hint::black_box(());
                        }
                        Err(e) => {
                            log::error!("❌ CRITICAL: Failed to save model for {}: {}", symbol, e);
                            log::error!("❌ Stopping batch training due to save failure");
                            return Err(e);
                        }
                    }
                }
                Err(e) => {
                    log::error!("❌ Failed to train model for {}: {}", symbol, e);
                    // Continue with next symbol instead of failing completely
                }
            }

            // Simple memory cleanup between symbols in batch mode
            if symbol_count > 1 {
                log::debug!("🧹 Memory cleanup between symbols in batch training");
                std::hint::black_box(());
            }
        }
    } else {
        // Single symbol training
        let symbol = &symbols[0];
        log::info!("🎯 Single symbol training mode: {}", symbol);

        // Validate data file exists
        if !params.data.exists() {
            return Err(VangaError::DataError(format!(
                "❌ Data file not found: {}\n💡 Expected file: {}/{}.csv",
                params.data.display(),
                params
                    .data
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .display(),
                symbol
            )));
        }

        if params.data.is_dir() {
            return Err(VangaError::DataError(format!(
                "❌ Directory passed for single symbol training: {}\n💡 For single symbol: use --data {}/{}.csv\n💡 For batch mode: use multiple symbols --symbol BTCUSDT,ETHUSDT",
                params.data.display(),
                params.data.display(),
                symbol
            )));
        }

        log::info!("📂 Using data file: {}", params.data.display());

        // Create training config
        let mut config = if let Some(config_path) = params.config {
            log::info!("🔧 Loading training config from: {:?}", config_path);
            match TrainingConfig::default()
                .symbol(symbol.clone())
                .data_path(params.data.clone())
                .with_config_from_file(&config_path)
            {
                Ok(file_config) => file_config,
                Err(e) => {
                    log::error!("Failed to load config file: {}", e);
                    log::info!("Falling back to default configuration");
                    TrainingConfig::default()
                        .symbol(symbol.clone())
                        .data_path(params.data)
                }
            }
        } else {
            log::info!("🔧 Using default intelligent training configuration");
            TrainingConfig::default()
                .symbol(symbol.clone())
                .data_path(params.data)
        };

        // Apply parameters
        if params.fresh {
            config = config.fresh_training(true);
        }
        if params.continue_training {
            config = config.continue_training(true);
        }
        if let Some(horizons) = params.horizons {
            config = config.horizons(horizons);
        }

        // Apply device configuration
        config = config.with_device_config(&params.device)?;
        // Features are now part of the main config, no separate handling needed
        if params.attention {
            log::info!("🎯 Attention mechanism enabled for enhanced accuracy");
            config = config.with_attention_enabled(true);
        }
        if params.tft {
            log::info!("🔮 TFT (Temporal Fusion Transformer) enabled for enhanced accuracy");
            config = config.with_tft_enabled(true);
        }

        monitor.checkpoint("Configuration prepared");

        // Train the model
        let mut model = api::train_model(config.clone()).await?;
        monitor.checkpoint("Model training completed");

        // Set complete training configuration in model metadata for prediction use
        model.set_training_config(config.clone());

        // Save model
        let model_path = vanga::utils::model_path::get_model_path(&config.symbol);
        let _ = vanga::utils::model_path::ensure_models_dir_exists();
        model.save(&model_path)?;
        monitor.checkpoint("Model saved to disk");

        log::info!("💾 Model saved to: {}", model_path.display());
        log::info!("✅ Training completed successfully");
    }

    Ok(())
}

async fn handle_predict_command(params: PredictParams) -> Result<()> {
    let _monitor = PerformanceMonitor::new(&format!("Prediction {}", params.symbol));
    log::info!("Starting prediction for symbol: {}", params.symbol);

    if params.realtime {
        log::info!("Real-time prediction mode");

        let mut config = RealtimeConfig {
            file_path: params.input,
            symbol: params.symbol,
            poll_interval: Duration::from_secs(1), // Default 1 second
            buffer_size: 1000,
            feature_window: 100,
            output_format: OutputFormat::Json,
            debug: false,
        };

        // Parse interval if provided
        if let Some(interval_str) = params.interval {
            let seconds = interval_str
                .trim_end_matches('s')
                .parse::<u64>()
                .unwrap_or(1);
            config.poll_interval = Duration::from_secs(seconds);
            log::info!("Using custom poll interval: {}s", seconds);
        }

        // Set output format based on source parameter
        if let Some(source) = params.source {
            match source.as_str() {
                "json" => config.output_format = OutputFormat::Json,
                "csv" => config.output_format = OutputFormat::Csv,
                "pretty" => config.output_format = OutputFormat::Pretty,
                _ => {
                    log::warn!("Unknown output format: {}, using JSON", source);
                    config.output_format = OutputFormat::Json;
                }
            }
            log::info!("Using output format: {:?}", config.output_format);
        }

        // Start real-time prediction
        start_realtime_prediction(config).await?;
    } else if params.batch {
        log::info!("Batch prediction mode");
        if let Some(input_dir) = params.input_dir {
            log::info!("Processing batch from directory: {}", input_dir.display());
            // Batch prediction from directory
            log::warn!("Batch directory prediction not yet implemented - use single file prediction instead");
        } else {
            log::warn!("Batch mode enabled but no input directory specified");
        }
    } else {
        // Parse symbols (single or comma-separated)
        let symbols: Vec<String> = params
            .symbol
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        log::info!("📋 Parsed {} symbol(s): {:?}", symbols.len(), symbols);

        if symbols.is_empty() {
            return Err(vanga::utils::error::VangaError::ConfigError(
                "No valid symbols provided".to_string(),
            ));
        }

        if symbols.len() == 1 {
            // Single symbol prediction - works exactly as before
            log::info!("🎯 Single symbol prediction mode: {}", symbols[0]);

            let mut config = PredictionConfig::default()
                .symbol(symbols[0].clone())
                .input_path(params.input);

            if let Some(horizon) = params.horizon {
                config = config.horizon(horizon);
            }

            if params.all_horizons {
                config = config.all_horizons(true);
            }

            if let Some(output) = params.output {
                config = config.output_path(output);
            }

            if let Some(min_confidence) = params.min_confidence {
                config = config.min_confidence(min_confidence);
            }

            // Apply device configuration
            config = config.with_device_config(&params.device)?;

            // Resolve data file path
            let data_file_path =
                file_discovery::resolve_symbol_data_path(&config.input_path, &symbols[0])?;

            log::info!("📂 Using data file: {}", data_file_path.display());

            // Load the trained model
            let model_path = vanga::utils::model_path::get_model_path(&symbols[0]);
            let model = vanga::model::multi_target::MultiTargetLSTMModel::load(&model_path)?;

            // Validate horizon configuration against model
            config.validate_horizon_against_model(&model)?;

            // Log available horizons
            let trained_horizons = model.get_trained_horizons();
            log::info!("Model trained horizons: {:?}", trained_horizons);
            if let Some(requested_horizon) = &config.horizon {
                log::info!("Using requested horizon: {}", requested_horizon);
            } else if config.all_horizons {
                log::info!("Predicting all available horizons: {:?}", trained_horizons);
            } else {
                log::info!(
                    "No horizon specified, using primary horizon: {}",
                    trained_horizons.first().unwrap_or(&"1h".to_string())
                );
            }

            // Make predictions using the unified predictor API
            let structured_predictions = {
                let predictor = vanga::api::Predictor::new(config.clone());
                predictor.predict(vanga::api::ModelWrapper::MultiTarget(&model)).await?
            };

            // Save predictions if output path specified
            if let Some(ref output_path) = config.output_path {
                log::info!("Saving structured predictions to file...");
                let output_content =
                    vanga::output::formatter::predictions_to_json(&structured_predictions)?;

                std::fs::write(output_path.as_path(), output_content)?;
                log::info!("Structured predictions saved to: {}", output_path.display());
            } else {
                // If no output path, print structured predictions to console
                log::info!("Outputting structured predictions to console...");
                let output_content =
                    vanga::output::formatter::predictions_to_json(&structured_predictions)?;
                println!("{}", output_content);
            }

            log::info!("Prediction configuration: {:?}", config);
            log::info!("Prediction completed successfully");
        } else {
            // Multi-symbol prediction with cross-asset features
            log::info!(
                "🔄 Multi-symbol prediction mode with {} symbols",
                symbols.len()
            );

            // Resolve data file paths for all symbols
            let mut symbol_paths = std::collections::HashMap::new();
            for symbol in &symbols {
                match file_discovery::resolve_symbol_data_path(&params.input, symbol) {
                    Ok(path) => {
                        symbol_paths.insert(symbol.clone(), path);
                    }
                    Err(e) => {
                        log::error!("❌ Failed to resolve data file for {}: {}", symbol, e);
                        continue;
                    }
                }
            }

            if symbol_paths.is_empty() {
                return Err(vanga::utils::error::VangaError::DataError(
                    "No valid data files found for any symbols".to_string(),
                ));
            }

            log::info!("📂 Found data files for {} symbols", symbol_paths.len());

            // Create prediction config for multi-symbol mode
            let mut config = PredictionConfig::default()
                .symbols(symbols.clone())
                .input_path(params.input.clone());

            if let Some(horizon) = params.horizon.clone() {
                config = config.horizon(horizon);
            }

            if params.all_horizons {
                config = config.all_horizons(true);
            }

            if let Some(min_confidence) = params.min_confidence {
                config = config.min_confidence(min_confidence);
            }

            // Get feature configuration from first symbol's model metadata
            let features_config = {
                let first_symbol = &symbols[0];
                let model_path = vanga::utils::model_path::get_model_path(first_symbol);
                match vanga::model::multi_target::MultiTargetLSTMModel::load(&model_path) {
                    Ok(model) => {
                        if let Some(config) = model.get_feature_config() {
                            log::info!("✅ Loaded feature configuration from model metadata");
                            config.clone()
                        } else {
                            log::warn!("Model metadata missing feature config, using default");
                            vanga::config::FeatureConfig::default()
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "Could not load model for {}: {}, using default feature config",
                            first_symbol,
                            e
                        );
                        vanga::config::FeatureConfig::default()
                    }
                }
            };

            // Check if cross-asset features should be enabled
            let cross_asset_enabled = features_config.cross_asset.enabled && symbols.len() > 1;

            if cross_asset_enabled {
                log::info!("🔗 Cross-asset features enabled for multi-symbol prediction");

                // Use cross-asset prediction pipeline
                let data_pipeline = vanga::data::DataPipeline::new();
                let prepared_data = data_pipeline
                    .prepare_cross_asset_prediction_data(&symbol_paths, &config, &features_config)
                    .await?;

                // Make predictions for each symbol with cross-asset features
                for symbol in &symbols {
                    if let Some(_symbol_data) = prepared_data.get(symbol) {
                        log::info!("🎯 Making cross-asset prediction for {}", symbol);

                        // Load the trained model for this symbol
                        let model_path = vanga::utils::model_path::get_model_path(symbol);
                        let model = match vanga::model::multi_target::MultiTargetLSTMModel::load(
                            &model_path,
                        ) {
                            Ok(model) => model,
                            Err(e) => {
                                log::error!("❌ Failed to load model for {}: {}", symbol, e);
                                continue;
                            }
                        };

                        // Create single-symbol config for this prediction
                        let symbol_config = config.clone().symbol(symbol.clone());

                        // Validate horizon configuration against model
                        if let Err(e) = symbol_config.validate_horizon_against_model(&model) {
                            log::error!("❌ Horizon validation failed for {}: {}", symbol, e);
                            continue;
                        }

                        // Make prediction using multi-target API
                        let predictions = {
                            let predictor = vanga::api::Predictor::new(symbol_config.clone());
                            match predictor.predict(vanga::api::ModelWrapper::MultiTarget(&model)).await {
                                Ok(predictions) => predictions,
                                Err(e) => {
                                    log::error!(
                                        "❌ Cross-asset prediction failed for {}: {}",
                                        symbol,
                                        e
                                    );
                                    continue;
                                }
                            }
                        };

                        // Save predictions for this symbol
                        let output_path = if let Some(output) = &params.output {
                            if output.is_dir() {
                                Some(
                                    output.join(format!("{}_cross_asset_predictions.json", symbol)),
                                )
                            } else {
                                let stem = output.file_stem().unwrap_or_default().to_string_lossy();
                                let ext = output.extension().unwrap_or_default().to_string_lossy();
                                Some(output.with_file_name(format!(
                                    "{}_{}_cross_asset.{}",
                                    stem, symbol, ext
                                )))
                            }
                        } else {
                            None
                        };

                        if let Some(output_path) = output_path {
                            log::info!("💾 Saving cross-asset predictions to file...");
                            let output_content =
                                vanga::output::formatter::predictions_to_json(&predictions)?;
                            std::fs::write(&output_path, output_content)?;
                            log::info!(
                                "✅ Cross-asset predictions for {} saved to: {}",
                                symbol,
                                output_path.display()
                            );
                        } else {
                            let output_content =
                                vanga::output::formatter::predictions_to_json(&predictions)?;
                            println!("=== Cross-Asset Predictions for {} ===", symbol);
                            println!("{}", output_content);
                        }
                    } else {
                        log::error!("❌ No prepared data found for symbol: {}", symbol);
                    }
                }
            } else {
                log::info!("🔄 Cross-asset features disabled, using individual predictions");

                // Fall back to individual symbol predictions
                for symbol in &symbols {
                    log::info!("🎯 Predicting symbol: {}", symbol);

                    let mut symbol_config = PredictionConfig::default()
                        .symbol(symbol.clone())
                        .input_path(params.input.clone());

                    if let Some(horizon) = &params.horizon {
                        symbol_config = symbol_config.horizon(horizon.clone());
                    }

                    if params.all_horizons {
                        symbol_config = symbol_config.all_horizons(true);
                    }

                    if let Some(min_confidence) = params.min_confidence {
                        symbol_config = symbol_config.min_confidence(min_confidence);
                    }

                    // Resolve data file path for this symbol
                    let data_file_path = match file_discovery::resolve_symbol_data_path(
                        &symbol_config.input_path,
                        symbol,
                    ) {
                        Ok(path) => path,
                        Err(e) => {
                            log::error!("❌ Failed to resolve data file for {}: {}", symbol, e);
                            continue;
                        }
                    };

                    log::info!(
                        "📂 Using data file for {}: {}",
                        symbol,
                        data_file_path.display()
                    );

                    // Load the trained model for this symbol
                    let model_path = vanga::utils::model_path::get_model_path(symbol);
                    let model =
                        match vanga::model::multi_target::MultiTargetLSTMModel::load(&model_path) {
                            Ok(model) => model,
                            Err(e) => {
                                log::error!("❌ Failed to load model for {}: {}", symbol, e);
                                continue;
                            }
                        };

                    // Validate horizon configuration against model
                    if let Err(e) = symbol_config.validate_horizon_against_model(&model) {
                        log::error!("❌ Horizon validation failed for {}: {}", symbol, e);
                        continue;
                    }

                    // Make prediction for this symbol
                    log::info!("🔮 Making prediction for {}", symbol);

                    let predictions = {
                        let predictor = vanga::api::Predictor::new(symbol_config.clone());
                        match predictor.predict(vanga::api::ModelWrapper::MultiTarget(&model)).await {
                            Ok(predictions) => predictions,
                            Err(e) => {
                                log::error!("❌ Prediction failed for {}: {}", symbol, e);
                                continue;
                            }
                        }
                    };

                    // Save predictions for this symbol
                    let output_path = if let Some(output) = &params.output {
                        if output.is_dir() {
                            Some(output.join(format!("{}_predictions.json", symbol)))
                        } else {
                            let stem = output.file_stem().unwrap_or_default().to_string_lossy();
                            let ext = output.extension().unwrap_or_default().to_string_lossy();
                            Some(output.with_file_name(format!("{}_{}.{}", stem, symbol, ext)))
                        }
                    } else {
                        None
                    };

                    if let Some(output_path) = output_path {
                        log::info!("💾 Saving predictions to file...");
                        let output_content =
                            vanga::output::formatter::predictions_to_json(&predictions)?;
                        std::fs::write(&output_path, output_content)?;
                        log::info!(
                            "✅ Predictions for {} saved to: {}",
                            symbol,
                            output_path.display()
                        );
                    } else {
                        let output_content =
                            vanga::output::formatter::predictions_to_json(&predictions)?;
                        println!("=== Predictions for {} ===", symbol);
                        println!("{}", output_content);
                    }
                }
            }

            log::info!("✅ Multi-symbol prediction completed");
        }
    }

    Ok(())
}

async fn handle_model_commands(action: ModelCommands) -> Result<()> {
    match action {
        ModelCommands::List => {
            log::info!("Listing available models");

            let models = vanga::utils::model_path::list_available_models()?;

            if models.is_empty() {
                log::info!("No models found in ./models directory");
                println!("No trained models available. Train a model first with: vanga train --symbol <SYMBOL> --data <DATA_FILE>");
            } else {
                log::info!("Available models:");
                println!("\n📊 Available Trained Models:");
                println!(
                    "{:<15} {:<20} {:<30}",
                    "Symbol", "Status", "Trained Horizons"
                );
                println!("{}", "-".repeat(70));

                for model_name in &models {
                    let model_path = vanga::utils::model_path::get_model_path(model_name);

                    // Try to load model to get horizon information
                    match vanga::model::multi_target::MultiTargetLSTMModel::load(&model_path) {
                        Ok(model) => {
                            let horizons = model.get_trained_horizons();
                            let horizons_str = if horizons.is_empty() {
                                "[legacy - no horizon info]".to_string()
                            } else {
                                format!("{:?}", horizons)
                            };
                            println!("{:<15} {:<20} {:<30}", model_name, "✅ Ready", horizons_str);
                        }
                        Err(_) => {
                            println!(
                                "{:<15} {:<20} {:<30}",
                                model_name, "❌ Error", "Unable to load"
                            );
                        }
                    }
                }

                println!("\n💡 Usage:");
                println!("  • Predict with specific horizon: vanga predict --symbol <SYMBOL> --horizon <HORIZON> --input <DATA>");
                println!("  • Predict all horizons: vanga predict --symbol <SYMBOL> --all-horizons --input <DATA>");
                println!("  • Auto-select horizon: vanga predict --symbol <SYMBOL> --input <DATA>");
            }
        }

        ModelCommands::Evaluate {
            symbol,
            test_data,
            backtest,
            train_split,
            batch,
            symbols,
        } => {
            if backtest {
                if batch && symbols.is_some() {
                    // Batch backtesting for multiple symbols
                    let symbols = symbols.unwrap();
                    log::info!("📊 Running batch backtesting for {} symbols", symbols.len());

                    match vanga::api::run_batch_backtest(&symbols, &test_data, train_split).await {
                        Ok(results) => {
                            vanga::utils::backtest_reporter::print_backtest_results(&results);

                            // Save results to file
                            let output_dir = std::path::Path::new("backtest_results");
                            if let Err(e) = vanga::utils::backtest_reporter::save_backtest_report(
                                &results, output_dir, "json",
                            ) {
                                log::warn!("Failed to save backtest report: {}", e);
                            }
                        }
                        Err(e) => {
                            log::error!("❌ Batch backtesting failed: {}", e);
                            return Err(e);
                        }
                    }
                } else if symbol.is_some() {
                    // Single symbol backtesting
                    let symbol = symbol.unwrap();
                    match vanga::api::run_backtest(&symbol, &test_data, train_split).await {
                        Ok(result) => {
                            vanga::utils::backtest_reporter::print_backtest_results(&[result]);
                        }
                        Err(e) => {
                            log::error!("❌ Backtesting failed for {}: {}", symbol, e);
                            return Err(e);
                        }
                    }
                } else {
                    log::error!("❌ Symbol is required for single symbol backtesting");
                    return Err(vanga::utils::error::VangaError::DataError(
                        "Symbol is required when not using batch mode".to_string(),
                    ));
                }
            } else if symbol.is_some() {
                // Traditional evaluation (existing placeholder)
                let symbol = symbol.unwrap();
                log::info!(
                    "Evaluating model for symbol: {} with test data: {:?}",
                    symbol,
                    test_data
                );
                log::warn!("Traditional model evaluation not yet implemented - use --backtest flag for comprehensive evaluation");
            } else {
                log::error!("❌ Symbol is required for evaluation");
                return Err(vanga::utils::error::VangaError::DataError(
                    "Symbol is required for evaluation".to_string(),
                ));
            }
        }

        ModelCommands::Compare { symbols, metric } => {
            log::info!(
                "Comparing models for symbols: {:?} using metric: {}",
                symbols,
                metric
            );
            // Model comparison implementation
            log::warn!("Model comparison not yet implemented - feature planned for future release");
        }

        ModelCommands::Export {
            symbol,
            format,
            output,
        } => {
            log::info!(
                "Exporting model for symbol: {} in format: {} to: {:?}",
                symbol,
                format,
                output
            );
            // Model export implementation
            log::warn!("Model export not yet implemented - feature planned for future release");
        }

        ModelCommands::Ensemble {
            symbols,
            strategies,
            output,
        } => {
            log::info!(
                "Creating ensemble for symbols: {:?} with strategies: {:?} as: {}",
                symbols,
                strategies,
                output
            );
            // Ensemble creation implementation
            log::warn!(
                "Ensemble creation not yet implemented - feature planned for future release"
            );
        }
    }

    Ok(())
}

/// Configure rayon thread pool for optimal CPU utilization
fn configure_rayon_threads() {
    let num_cpus = num_cpus::get();
    let optimal_threads = std::cmp::max(1, num_cpus - 1); // Leave one core for system

    rayon::ThreadPoolBuilder::new()
        .num_threads(optimal_threads)
        .build_global()
        .expect("Failed to configure rayon thread pool");

    log::info!(
        "🚀 Configured rayon with {} threads for {} CPU cores",
        optimal_threads,
        num_cpus
    );
}

/// Performance monitoring helper
struct PerformanceMonitor {
    start_time: std::time::Instant,
    task_name: String,
}

impl PerformanceMonitor {
    fn new(task_name: &str) -> Self {
        log::info!("⏱️  Starting: {}", task_name);
        Self {
            start_time: std::time::Instant::now(),
            task_name: task_name.to_string(),
        }
    }

    fn checkpoint(&self, checkpoint_name: &str) {
        let elapsed = self.start_time.elapsed();
        log::info!(
            "⏱️  {} - {}: {:.2}s",
            self.task_name,
            checkpoint_name,
            elapsed.as_secs_f64()
        );
    }
}

impl Drop for PerformanceMonitor {
    fn drop(&mut self) {
        let elapsed = self.start_time.elapsed();
        log::info!(
            "✅ Completed: {} in {:.2}s",
            self.task_name,
            elapsed.as_secs_f64()
        );
    }
}
