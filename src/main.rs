use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Duration;
use vanga::api;
use vanga::config::{PredictionConfig, TrainingConfig};
use vanga::realtime::{start_realtime_prediction, OutputFormat, RealtimeConfig};
use vanga::utils::error::{Result, VangaError};

/// Training command parameters
struct TrainParams {
    symbol: String,
    data: PathBuf,
    fresh: bool,
    continue_training: bool,
    horizons: Option<Vec<String>>,
    features_config: Option<PathBuf>,
    config: Option<PathBuf>,
    attention: bool,
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

        /// Training configuration file (enables intelligent training)
        #[arg(long)]
        config: Option<PathBuf>,

        /// Enable attention mechanism for enhanced accuracy
        #[arg(long)]
        attention: bool,

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
            features_config,
            config,
            attention,
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
                config,
                attention,
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
    let monitor = PerformanceMonitor::new(&format!("Training {}", params.symbol));
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

                monitor.checkpoint(&format!("Config prepared for {}", sym));

                // Train the model
                match api::train_model(symbol_config.clone()).await {
                    Ok(model) => {
                        monitor.checkpoint(&format!("Model trained for {}", sym));
                        log::info!("Successfully trained model for {}", sym);

                        // Save model with consistent path
                        let model_path = vanga::utils::model_path::get_model_path(&sym);
                        let _ = vanga::utils::model_path::ensure_models_dir_exists();

                        if let Err(e) = model.save(&model_path) {
                            log::error!("Failed to save model for {}: {}", sym, e);
                        } else {
                            monitor.checkpoint(&format!("Model saved for {}", sym));
                            log::info!("💾 Model saved to: {}", model_path.display());
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
        let mut config = if let Some(config_path) = params.config {
            // Load config from file
            log::info!("🔧 Loading training config from: {:?}", config_path);
            match TrainingConfig::default()
                .symbol(params.symbol.clone())
                .data_path(params.data.clone())
                .with_training_params_from_file(&config_path)
            {
                Ok(file_config) => file_config,
                Err(e) => {
                    log::error!("Failed to load config file: {}", e);
                    log::info!("Falling back to default configuration");
                    TrainingConfig::default()
                        .symbol(params.symbol)
                        .data_path(params.data)
                }
            }
        } else {
            // Use default config
            log::info!("🔧 Using default intelligent training configuration");
            TrainingConfig::default()
                .symbol(params.symbol)
                .data_path(params.data)
        };

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

        // Configure attention if enabled
        if params.attention {
            log::info!("🎯 Attention mechanism enabled for enhanced accuracy");
            config = config.with_attention_enabled(true);
        }

        monitor.checkpoint("Configuration prepared");

        // Train the model using the API
        let model = vanga::api::train_model(config.clone()).await?;
        monitor.checkpoint("Model training completed");

        // Save model with consistent path
        let model_path = vanga::utils::model_path::get_model_path(&config.symbol);
        let _ = vanga::utils::model_path::ensure_models_dir_exists();
        model.save(&model_path)?;
        monitor.checkpoint("Model saved to disk");

        log::info!("💾 Model saved to: {}", model_path.display());
        log::info!("Training completed successfully");
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

        // Load the trained model using consistent path
        let model_path = vanga::utils::model_path::get_model_path(&config.symbol);
        let model = vanga::model::multi_target::MultiTargetLSTMModel::load(&model_path)?;

        // Validate horizon configuration against model
        config.validate_horizon_against_model(&model)?;

        // Log available horizons for user information
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

        // Make predictions using the multi-target API
        let predictions = vanga::api::predict_multi_target(config.clone(), &model).await?;

        // Save predictions if output path specified
        if let Some(ref output_path) = config.output_path {
            // Convert raw predictions to structured format using OutputFormatter
            log::info!("Converting raw predictions to structured format...");
            let structured_predictions = predictions
                .to_structured_predictions(&config, &model)
                .await?;

            // Use existing formatter method to create JSON
            let output_content =
                vanga::output::formatter::predictions_to_json(&structured_predictions)?;

            std::fs::write(output_path.as_path(), output_content)?;
            log::info!("Structured predictions saved to: {}", output_path.display());
        } else {
            // If no output path, print structured predictions to console
            log::info!("Converting raw predictions to structured format for console output...");
            let structured_predictions = predictions
                .to_structured_predictions(&config, &model)
                .await?;
            let output_content =
                vanga::output::formatter::predictions_to_json(&structured_predictions)?;
            println!("{}", output_content);
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
