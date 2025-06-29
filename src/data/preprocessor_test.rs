use super::DataPreprocessor;
use crate::config::training::{DataConfig, MissingDataStrategy};
use polars::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_df_with_missing() -> DataFrame {
        let ts = Series::new("timestamp", &["2024-01-01T00:00:00Z"]);
        let open = Series::new("open", &[Some(42000.0), None]);
        let high = Series::new("high", &[42500.0, 42600.0]);
        let low = Series::new("low", &[41800.0, 41900.0]);
        let close = Series::new("close", &[42300.0, 42400.0]);
        let volume = Series::new("volume", &[1000.0, 1200.0]);
        DataFrame::new(vec![ts, open, high, low, close, volume]).unwrap()
    }

    #[tokio::test]
    async fn process_for_training_forward_fill() {
        let mut df = make_df_with_missing();
        let config = DataConfig {
            missing_data_strategy: MissingDataStrategy::ForwardFill,
            ..Default::default()
        };
        let pre = DataPreprocessor::new();
        let df2 = pre.process_for_training(df, &config, None).await.unwrap();
        assert_eq!(df2.height(), 2);
        assert!(df2.column("open").unwrap().null_count() == 0);
    }

    #[tokio::test]
    async fn process_for_training_drop_missing() {
        let mut df = make_df_with_missing();
        let config = DataConfig {
            missing_data_strategy: MissingDataStrategy::Drop,
            ..Default::default()
        };
        let pre = DataPreprocessor::new();
        let df2 = pre.process_for_training(df, &config, None).await.unwrap();
        assert!(df2.height() < 2); // Should drop the row with missing open
    }
}
