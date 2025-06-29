use super::DataLoader;
use polars::prelude::*;
use std::fs::File;
use std::io::Write;
use tempfile::NamedTempFile;

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp_csv(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");
        write!(file, "{}", contents).expect("write csv temp");
        file
    }

    #[tokio::test]
    async fn loads_valid_csv() {
        let csv = "timestamp,open,high,low,close,volume\n2024-01-01T00:00:00Z,42000,42500,41800,42300,1000";
        let file = write_temp_csv(csv);
        let loader = DataLoader::new();
        let df = loader
            .load_csv(file.path())
            .await
            .expect("should load valid csv");
        assert_eq!(df.height(), 1);
        assert!(df.get_column_names().contains(&"close".to_string()));
    }

    #[tokio::test]
    async fn error_on_missing_file() {
        let loader = DataLoader::new();
        let res = loader.load_csv("/nonexistent/path.csv").await;
        assert!(res.is_err());
        let err = format!("{}", res.unwrap_err());
        assert!(err.contains("Data file not found"));
    }

    #[tokio::test]
    async fn error_on_missing_columns() {
        let csv = "timestamp,open,high,low,close\n2024-01-01T00:00:00Z,42000,42500,41800,42300"; // missing volume
        let file = write_temp_csv(csv);
        let loader = DataLoader::new();
        let res = loader.load_csv(file.path()).await;
        assert!(res.is_err());
        let err = format!("{}", res.unwrap_err());
        assert!(err.contains("MissingColumns"));
    }
}
