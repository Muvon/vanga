use super::*;
use polars::prelude::*;

#[tokio::test]
async fn test_generate_all_targets_basic() {
    let df = DataFrame::new(
        vec![
            Series::new("close".into(), &[42300.0, 42400.0, 42500.0]),
            Series::new("high".into(), &[42500.0, 42600.0, 42700.0]),
            Series::new("low".into(), &[41800.0, 41900.0, 42000.0]),
        ]
        .into_iter()
        .map(|s| s.into_column())
        .collect(),
    )
    .unwrap();
    let generator = MultiTargetGenerator::with_defaults();
    let result = generator.generate_all_targets(&df).await;
    assert!(result.is_ok());
}
