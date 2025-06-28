use clap::{Parser, Subcommand};
use std::path::PathBuf;
use vanga::api;
use vanga::model::lstm_simple::LSTMModel;
use vanga::{PredictionConfig, Result, TrainingConfig, VangaError};

/// Training command parameters
struct TrainParams {
    symbol: String,
    data: PathBuf,
    fresh: bool,
    continue_training: bool,
    horizons: Option<Vec<String>>,
    features_config: Option<PathBuf>,
    batch: bool,
    data_dir: Option<PathBuf>,
    symbols: Option<Vec<String>>,
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
        /// Trading symbol (e.g., BTCUSDT)
        #[arg(short, long)]
        symbol: String,

        /// Path to CSV data file
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

        /// Custom features configuration file
        #[arg(long)]
        features_config: Option<PathBuf>,

        /// Batch training for multiple symbols
        #[arg(long)]
        batch: bool,

        /// Data directory for batch training
        #[arg(long)]
        data_dir: Option<PathBuf>,

        /// Symbols for batch training (comma-separated)
        #[arg(long, value_delimiter = ',')]
        symbols: Option<Vec<String>>,
    },

    /// Make predictions using trained model
    Predict {
        /// Trading symbol (e.g., BTCUSDT)
        #[arg(short, long)]
        symbol: String,

        /// Path to input CSV data
        #[arg(short, long)]
        input: PathBuf,

        /// Prediction horizon (1h, 4h, 1d, 7d)
        #[arg(long)]
        horizon: Option<String>,

        /// Predict all available horizons
        #[arg(long)]
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

    /// Evaluate model performance
    Evaluate {
        /// Trading symbol
        #[arg(short, long)]
        symbol: String,

        /// Test data path
        #[arg(long)]
        test_data: PathBuf,
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

    match cli.command {
        Commands::Train {
            symbol,
            data,
            fresh,
            continue_training,
            horizons,
            features_config,
            batch,
            data_dir,
            symbols,
        } => {
            let params = TrainParams {
                symbol,
                data,
                fresh,
                continue_training,
                horizons,
                features_config,
                batch,
                data_dir,
                symbols,
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
            };
            handle_predict_command(params).await
        }

        Commands::Models { action } => handle_model_commands(action).await,
    }
}

async fn handle_train_command(params: TrainParams) -> Result<()> {
    log::info!("Starting training for symbol: {}", params.symbol);

    if params.batch {
        // Batch training logic
        log::info!("Batch training mode enabled");
        if let (Some(data_dir), Some(symbols)) = (params.data_dir, params.symbols) {
            for sym in symbols {
                log::info!(
                    "Training model for symbol: {} using data from: {}",
                    sym,
                    data_dir.display()
                );

                // Create training config for this symbol
                let symbol_config = TrainingConfig::default()
                    .symbol(sym.clone())
                    .data_path(data_dir.join(format!("{}.csv", sym)));

                // Train the model
                match api::train_model(symbol_config.clone()).await {
                    Ok(model) => {
                        log::info!("Successfully trained model for {}", sym);
                        // Save model with symbol-specific name
                        let model_path = format!("./models/{}_model.bin", sym);
                        if let Err(e) = model.save(&model_path) {
                            log::error!("Failed to save model for {}: {}", sym, e);
                        } else {
                            log::info!("Model saved to: {}", model_path);
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to train model for {}: {}", sym, e);
                    }
                }
            }
        } else {
            return Err(VangaError::ConfigError(
                "Batch mode requires --data-dir and --symbols".to_string(),
            ));
        }
    } else {
        // Single symbol training
        let mut config = TrainingConfig::default()
            .symbol(params.symbol)
            .data_path(params.data);

        if params.fresh {
            config = config.fresh_training(true);
        }

        if params.continue_training {
            config = config.continue_training(true);
        }

        if let Some(horizons) = params.horizons {
            config = config.horizons(horizons);
        }

        if let Some(features_config) = params.features_config {
            config = config.features_config_path(features_config);
        }

        // Train the model using the API
        let model = crate::api::train_model(config.clone()).await?;

        // Save the trained model
        let model_path = format!("./models/{}_model.bin", config.symbol);
        std::fs::create_dir_all("./models")?;
        model.save(&model_path)?;

        log::info!("Model saved to: {}", model_path);
        log::info!("Training completed successfully");
    }

    Ok(())
}

async fn handle_predict_command(params: PredictParams) -> Result<()> {
    log::info!("Starting prediction for symbol: {}", params.symbol);

    if params.realtime {
        log::info!("Real-time prediction mode");
        if let Some(source) = params.source {
            log::info!("Using data source: {}", source);
        }
        if let Some(interval) = params.interval {
            log::info!("Using interval: {}", interval);
        }
        // Real-time prediction implementation
        log::warn!("Real-time prediction not yet implemented - use batch mode instead");
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
        // Single prediction
        let mut config = PredictionConfig::default()
            .symbol(params.symbol)
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

        // Load the trained model
        let model_path = format!("./models/{}_model.bin", config.symbol);
        let model = LSTMModel::load(&model_path)?;

        // Make predictions using the API
        let predictions = crate::api::predict(config.clone(), &model).await?;

        // Save predictions if output path specified
        if let Some(ref output_path) = config.output_path {
            use vanga::output::formatter::{predictions_to_csv, predictions_to_json};

            let output_content = match config.output_config.format {
                vanga::config::prediction::OutputFormat::ProbabilityDistribution
                | vanga::config::prediction::OutputFormat::ConfidenceInterval
                | vanga::config::prediction::OutputFormat::All => {
                    // Save as JSON for structured formats
                    predictions_to_json(&predictions)?
                }
                vanga::config::prediction::OutputFormat::PointEstimate => {
                    // Save as CSV for simple point estimates
                    predictions_to_csv(&predictions)?
                }
            };

            std::fs::write(output_path.as_path(), output_content)?;
            log::info!("Predictions saved to: {}", output_path.display());
        }

        log::info!("Prediction configuration: {:?}", config);
        log::info!("Prediction completed successfully");
    }

    Ok(())
}

async fn handle_model_commands(action: ModelCommands) -> Result<()> {
    match action {
        ModelCommands::List => {
            log::info!("Listing available models");
            // List available models in ./models directory
            let models_dir = std::path::Path::new("./models");
            if models_dir.exists() {
                let entries = std::fs::read_dir(models_dir)?;
                log::info!("Available models:");
                for entry in entries {
                    let entry = entry?;
                    if let Some(name) = entry.file_name().to_str() {
                        if name.ends_with("_model.bin") {
                            let symbol = name.replace("_model.bin", "");
                            log::info!("  - {} ({})", symbol, name);
                        }
                    }
                }
            } else {
                log::info!("No models directory found. Train a model first.");
            }
        }

        ModelCommands::Evaluate { symbol, test_data } => {
            log::info!(
                "Evaluating model for symbol: {} with test data: {:?}",
                symbol,
                test_data
            );
            // Model evaluation implementation
            log::warn!("Model evaluation not yet implemented - feature planned for future release");
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
