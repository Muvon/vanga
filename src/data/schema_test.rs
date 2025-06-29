use super::CryptoDataSchema;
use polars::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("timestamp", &["2024-01-01T00:00:00Z"]),
            Series::new("open", &[42000.0]),
            Series::new("high", &[42500.0]),
            Series::new("low", &[41800.0]),
            Series::new("close", &[42300.0]),
            Series::new("volume", &[1000.0]),
        ])
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
        let df = DataFrame::new(vec![
            Series::new("timestamp", &["2024-01-01T00:00:00Z"]),
            Series::new("open", &[42000.0]),
            Series::new("close", &[42300.0]),
            Series::new("volume", &[1000.0]),
        ])
        .unwrap(); // missing high, low
        let res = CryptoDataSchema::validate(&df);
        assert!(res.is_err());
        let err = format!("{}", res.unwrap_err());
        assert!(err.contains("MissingColumns"));
    }
}
