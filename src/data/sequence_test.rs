use super::SequenceGenerator;
use crate::config::model::SequenceLengthConfig;
use crate::config::{training::DataConfig, training::TrainingConfig, FeatureConfig};
use polars::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::error::Result;

    fn make_df(n: usize) -> DataFrame {
        let mut df = DataFrame::new(vec![]).unwrap();
        let ts: Vec<String> = (0..n)
            .map(|i| format!("2024-01-01T00:{:02}:00Z", i))
            .collect();
        df.with_column(Series::new("timestamp".into(), ts)).unwrap();
        df.with_column(Series::new("open".into(), vec![42000.0; n]))
            .unwrap();
        df.with_column(Series::new("high".into(), vec![42500.0; n]))
            .unwrap();
        df.with_column(Series::new("low".into(), vec![41800.0; n]))
            .unwrap();
        df.with_column(Series::new("close".into(), vec![42300.0; n]))
            .unwrap();
        df.with_column(Series::new("volume".into(), vec![1000.0; n]))
            .unwrap();
        df
    }

    #[tokio::test]
    async fn generate_training_sequences_fixed_ok() {
        let df = make_df(32);
        let horizons = vec!["1h".to_string()];
        let training_config = TrainingConfig {
            model: crate::config::model::ModelConfig {
                sequence_length: SequenceLengthConfig::Fixed(10),
                ..Default::default()
            },
            ..Default::default()
        };
        let gen = SequenceGenerator::default(); // Use default (no overlap)
        let data_config = DataConfig::default();
        let feature_config = FeatureConfig::default();
        let res = gen
            .generate_training_sequences(
                df,
                &horizons,
                &training_config,
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
        let training_config = TrainingConfig {
            model: crate::config::model::ModelConfig {
                sequence_length: SequenceLengthConfig::Fixed(10),
                ..Default::default()
            },
            ..Default::default()
        };
        let gen = SequenceGenerator::default(); // Use default (no overlap)
        let data_config = DataConfig::default();
        let feature_config = FeatureConfig::default();
        let res = gen
            .generate_training_sequences(
                df,
                &horizons,
                &training_config,
                &data_config,
                &feature_config,
            )
            .await;
        assert!(res.is_err());
    }
}
