use super::*;
use polars::prelude::*;

#[tokio::test]
async fn test_generate_technical_indicators_basic() {
    let mut df = DataFrame::new(vec![
        Series::new("open", &[42000.0, 42100.0, 42200.0]),
        Series::new("high", &[42500.0, 42600.0, 42700.0]),
        Series::new("low", &[41800.0, 41900.0, 42000.0]),
        Series::new("close", &[42300.0, 42400.0, 42500.0]),
        Series::new("volume", &[1000.0, 1200.0, 1300.0]),
    ])
    .unwrap();
    let df2 = generate_technical_indicators(df.clone()).await.unwrap();
    assert!(df2.get_column_names().len() > df.get_column_names().len());
    assert!(
        df2.column("sma_5").is_ok() || df2.get_column_names().iter().any(|n| n.contains("sma"))
    );
}
