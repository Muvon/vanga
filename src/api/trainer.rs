// Model trainer
use crate::config::TrainingConfig;
use crate::data::DataPipeline;
use crate::model::multi_target::MultiTargetLSTMModel;
use crate::targets::TargetGenerator;
use crate::utils::error::Result;
use ndarray::{s, Array2, Array3};

pub struct ModelTrainer {
    config: TrainingConfig,
}

impl ModelTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    pub async fn train(&self) -> Result<MultiTargetLSTMModel> {
        log::info!("Starting model training for symbol: {}", self.config.symbol);

        // Initialize data pipeline
        let data_pipeline = DataPipeline::new();

        // Load and prepare training data
        log::info!(
            "Loading training data from: {}",
            self.config.data_path.display()
        );
        let prepared_data = data_pipeline
            .prepare_training_data(&self.config.data_path, &self.config)
            .await?;

        log::info!(
            "Training data prepared: {} sequences, {} features",
            prepared_data.sequences.shape()[0],
            prepared_data.sequences.shape()[2]
        );

        // Generate targets with training config horizons
        let target_config = crate::targets::MultiTargetConfig {
            price_level_config: crate::targets::PriceLevelConfig::default(),
            direction_config: crate::targets::DirectionConfig::default(),
            volatility_config: crate::targets::VolatilityConfig::default(),
            horizons: self.config.horizons.clone(),
        };
        let target_generator = TargetGenerator::new(target_config);
        let df = crate::data::loader::DataLoader::new()
            .load_csv(&self.config.data_path)
            .await?;
        let targets = target_generator.generate_all_targets(&df).await?;

        // Get target names for multi-target model
        let target_names = target_generator.get_target_names();
        log::info!(
            "Generated {} targets: {:?}",
            target_names.len(),
            target_names
        );

        // CRITICAL FIX: Use raw targets directly for multi-target model
        // Skip TargetConverter which expands 3 targets to 14 one-hot encoded outputs
        // MultiTargetLSTMModel expects raw target values, not one-hot encoded

        // CRITICAL FIX: Only use sequences that have valid targets
        let sequence_length = prepared_data.sequences.shape()[1];
        let total_sequences = prepared_data.sequences.shape()[0];
        let num_targets = target_names.len();

        log::info!(
            "Filtering {} total sequences to match {} valid target indices",
            total_sequences,
            targets.valid_indices.len()
        );

        // Find sequences that correspond to valid target indices
        let mut valid_sequence_indices = Vec::new();

        for seq_idx in 0..total_sequences {
            // Each sequence ends at index (seq_idx + sequence_length - 1)
            let target_data_idx = seq_idx + sequence_length - 1;

            // Check if this target index is in valid_indices
            if targets.valid_indices.contains(&target_data_idx) {
                valid_sequence_indices.push(seq_idx);
            }
        }

        let num_valid_sequences = valid_sequence_indices.len();

        log::info!(
            "Found {} valid sequences out of {} total (matching valid target indices)",
            num_valid_sequences,
            total_sequences
        );

        if num_valid_sequences == 0 {
            return Err(crate::utils::error::VangaError::DataError(
                "No valid sequences found - no sequences correspond to valid target indices"
                    .to_string(),
            ));
        }

        // Create filtered sequences and aligned targets
        let mut filtered_sequences = Array3::<f64>::zeros((
            num_valid_sequences,
            sequence_length,
            prepared_data.sequences.shape()[2],
        ));
        let mut training_targets = Array2::<f64>::zeros((num_valid_sequences, num_targets));

        for (filtered_idx, &seq_idx) in valid_sequence_indices.iter().enumerate() {
            // Copy sequence
            filtered_sequences
                .slice_mut(s![filtered_idx, .., ..])
                .assign(&prepared_data.sequences.slice(s![seq_idx, .., ..]));

            // Get corresponding target data index
            let target_data_idx = seq_idx + sequence_length - 1;

            // Extract targets for this sequence
            for (target_idx, target_name) in target_names.iter().enumerate() {
                let parts: Vec<&str> = target_name.split('_').collect();
                let (target_type, horizon) =
                    if parts.len() == 3 && parts[0] == "price" && parts[1] == "level" {
                        ("price_level", parts[2])
                    } else if parts.len() == 2 {
                        (parts[0], parts[1])
                    } else {
                        continue;
                    };

                let target_data = match target_type {
                    "price_level" => targets.price_levels.get(horizon),
                    "direction" => targets.directions.get(horizon),
                    "volatility" => targets.volatility.get(horizon),
                    _ => continue,
                };

                if let Some(data) = target_data {
                    if target_data_idx < data.len() {
                        training_targets[[filtered_idx, target_idx]] = data[target_data_idx] as f64;
                    }
                }
            }
        }

        log::info!(
            "Training data prepared: {} valid sequences x {} outputs (perfect sequence-target alignment)",
            training_targets.shape()[0],
            training_targets.shape()[1]
        );

        // CRITICAL FIX: Use MultiTargetLSTMModel instead of single-target model
        let input_size = filtered_sequences.shape()[2]; // Number of features

        let mut model = self
            .get_or_create_multi_target_model(input_size, target_names)
            .await?;

        // Train the multi-target LSTM model with intelligent early stopping
        log::info!(
            "🚀 Starting perfectly aligned multi-target training - {} sequences with {} targets",
            training_targets.shape()[0],
            training_targets.shape()[1]
        );
        model
            .train_with_early_stopping(&filtered_sequences, &training_targets, &self.config)
            .await?;

        // Save the trained multi-target model
        log::info!("✅ Multi-target model training completed successfully!");
        Ok(model)
    }

    /// Get existing multi-target model or create new one based on training configuration
    async fn get_or_create_multi_target_model(
        &self,
        input_size: usize,
        target_names: Vec<String>,
    ) -> Result<MultiTargetLSTMModel> {
        // Create new model since we're not loading from file anymore
        // The caller (main.rs) will handle loading/saving based on training config
        log::info!("🆕 Creating new multi-target model for training");
        MultiTargetLSTMModel::new(&self.config.model_config, input_size, target_names)
    }
}

/// High-level training function
pub async fn train_model(config: TrainingConfig) -> Result<MultiTargetLSTMModel> {
    let trainer = ModelTrainer::new(config);
    trainer.train().await
}
