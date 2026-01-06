use super::CryptoDataSchema;
use polars::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_df() -> DataFrame {
        DataFrame::new(vec![Series::new("timestamp".into(), &["2024-01-01T00:00:00Z"]),
        Series::new("open".into(), &[42000.0]),
        Series::new("high".into(), &[42500.0]),
        Series::new("low".into(), &[41800.0]),
        Series::new("close".into(), &[42300.0]),
        Series::new("volume".into(), &[1000.0]),].into_iter().map(|s| s.into_column()).collect())
        .unwrap()
    }

    #[test]
    fn validates_valid_schema() {
        let df = make_valid_df();
        let res = CryptoDataSchema::validate(&df);
        assert!(res.is_ok());
    }

    #[test]
    fn error_on_missing_column() {
        let df = DataFrame::new(vec![Series::new("timestamp".into(), &["2024-01-01T00:00:00Z"]),
        Series::new("open".into(), &[42000.0]),
        Series::new("close".into(), &[42300.0]),
        Series::new("volume".into(), &[1000.0]),].into_iter().map(|s| s.into_column()).collect())
        .unwrap(); // missing high, low
        let res = CryptoDataSchema::validate(&df);
        assert!(res.is_err());
        let err = format!("{}", res.unwrap_err());
        assert!(err.contains("MissingColumns"));
    }
}
