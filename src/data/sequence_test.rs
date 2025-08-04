use super::SequenceGenerator;
use crate::config::{training::DataConfig, FeatureConfig, ModelConfig};
use polars::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::SequenceLengthConfig;
    use crate::utils::error::Result;

    fn make_df(n: usize) -> DataFrame {
        let ts: Vec<_> = (0..n)
            .map(|i| format!("2024-01-01T00:{:02}:00Z", i))
            .collect();
        let open = Series::new("open", vec![42000.0; n]);
        let high = Series::new("high", vec![42500.0; n]);
        let low = Series::new("low", vec![41800.0; n]);
        let close = Series::new("close", vec![42300.0; n]);
        let volume = Series::new("volume", vec![1000.0; n]);
        DataFrame::new(vec![
            Series::new("timestamp", ts),
            open,
            high,
            low,
            close,
            volume,
        ])
        .unwrap()
    }

    #[tokio::test]
    async fn generate_training_sequences_fixed_ok() {
        let df = make_df(32);
        let horizons = vec!["1h".to_string()];
        let model_config = ModelConfig {
            sequence_length: SequenceLengthConfig::Fixed(10),
            ..Default::default()
        };
        let gen = SequenceGenerator::default(); // Use default (no overlap)
        let data_config = DataConfig::default();
        let feature_config = FeatureConfig::default();
        let res = gen
            .generate_training_sequences(
                df,
                &horizons,
                &model_config,
                &data_config,
                &feature_config,
            )
            .await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn error_on_too_short_df() {
        let df = make_df(2);
        let horizons = vec!["1h".to_string()];
        let model_config = ModelConfig {
            sequence_length: SequenceLengthConfig::Fixed(10),
            ..Default::default()
        };
        let gen = SequenceGenerator::default(); // Use default (no overlap)
        let data_config = DataConfig::default();
        let feature_config = FeatureConfig::default();
        let res = gen
            .generate_training_sequences(
                df,
                &horizons,
                &model_config,
                &data_config,
                &feature_config,
            )
            .await;
        assert!(res.is_err());
    }
}
