// Unified model path utilities to ensure consistent naming between training and prediction
use std::path::PathBuf;

/// Generate consistent model path for models (always multi-target in VANGA)
pub fn get_model_path(symbol: &str) -> PathBuf {
    PathBuf::from(format!("./models/{}", symbol))
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

/// List all available models in the models directory
pub fn list_available_models() -> crate::utils::error::Result<Vec<String>> {
    let models_dir = get_models_dir();

    if !models_dir.exists() {
        return Ok(vec![]);
    }

    let mut models = Vec::new();

    for entry in std::fs::read_dir(&models_dir)? {
        let entry = entry?;
        let path = entry.path();

        // Look for .meta files which indicate a complete model
        if path.extension().and_then(|s| s.to_str()) == Some("meta") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                models.push(stem.to_string());
            }
        }
    }

    models.sort();
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_path_generation() {
        let path = get_model_path("BTCUSDT");
        assert_eq!(path, PathBuf::from("./models/BTCUSDT"));
    }

    #[test]
    fn test_models_dir() {
        let dir = get_models_dir();
        assert_eq!(dir, PathBuf::from("./models"));
    }
}
