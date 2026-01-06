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
        let open = Series::new("open".into(), vec![42000.0; n]).into_column();
        let high = Series::new("high".into(), vec![42500.0; n]).into_column();
        let low = Series::new("low".into(), vec![41800.0; n]).into_column();
        let close = Series::new("close".into(), vec![42300.0; n]).into_column();
        let volume = Series::new("volume".into(), vec![1000.0; n]).into_column();
        DataFrame::new(vec![Series::new("timestamp".into(), ts),
        open,
        high,
        low,
        close,
        volume,].into_iter().map(|s| s.into_column()).collect())
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
