// Unified model path utilities to ensure consistent naming between training and prediction
use std::path::PathBuf;

/// Generate consistent model path for multi-target models
pub fn get_multi_target_model_path(symbol: &str) -> PathBuf {
    PathBuf::from(format!("./models/{}_multi_model", symbol))
}

/// Generate consistent model directory path
pub fn get_models_dir() -> PathBuf {
    PathBuf::from("./models")
}

/// Ensure models directory exists
pub fn ensure_models_dir_exists() -> crate::utils::error::Result<()> {
    let models_dir = get_models_dir();
    std::fs::create_dir_all(&models_dir).map_err(|e| {
        crate::utils::error::VangaError::IoError(format!(
            "Failed to create models directory {}: {}",
            models_dir.display(),
            e
        ))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_path_generation() {
        let path = get_multi_target_model_path("BTCUSDT");
        assert_eq!(path, PathBuf::from("./models/BTCUSDT_multi_model"));
    }

    #[test]
    fn test_models_dir() {
        let dir = get_models_dir();
        assert_eq!(dir, PathBuf::from("./models"));
    }
}
