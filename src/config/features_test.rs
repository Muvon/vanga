use super::FeatureConfig;
use std::fs;
use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_feature_config_loads() {
        let toml_path = Path::new("configs/crypto_features.toml");
        let toml_content =
            fs::read_to_string(toml_path).expect("Failed to read crypto_features.toml");
        let config: FeatureConfig =
            toml::from_str(&toml_content).expect("Failed to deserialize valid feature config");
        assert!(config.technical_indicators.enabled);
        assert!(config.market_microstructure.enabled);
    }

    #[test]
    fn malformed_toml_returns_error() {
        let malformed = "[technical_indicators\nenabled = true"; // missing closing bracket
        let result = toml::from_str::<FeatureConfig>(malformed);
        assert!(result.is_err(), "Malformed TOML should return error");
    }

    #[test]
    fn missing_required_field_returns_error() {
        let missing = "[market_microstructure]\nenabled = true"; // missing technical_indicators
        let result = toml::from_str::<FeatureConfig>(missing);
        assert!(
            result.is_err(),
            "Missing required fields should return error"
        );
    }
}
