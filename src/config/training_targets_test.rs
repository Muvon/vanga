use crate::config::model::XGBoostConfig;
use crate::config::training::{TargetsConfig, TrainingConfig};
use std::io::Write;

fn targets_with_only_stop_level() -> TargetsConfig {
	TargetsConfig {
		price_level: false,
		price_level_definition: Default::default(),
		stop_level_definition: Default::default(),
		stop_level: true,
		direction: false,
		volatility: false,
		sentiment: false,
		volume: false,
	}
}

#[test]
fn stop_level_is_a_valid_enabled_target() {
	let targets = targets_with_only_stop_level();

	assert!(targets.validate().is_ok());
	assert_eq!(targets.get_enabled_targets(), vec!["stop_level"]);
	assert_eq!(targets.count_enabled(), 1);
}

#[test]
fn configured_active_targets_include_stop_level() {
	let targets = TargetsConfig {
		price_level: true,
		price_level_definition: Default::default(),
		stop_level_definition: Default::default(),
		stop_level: true,
		direction: true,
		volatility: false,
		sentiment: false,
		volume: false,
	};

	assert_eq!(
		targets.get_enabled_targets(),
		vec!["price_level", "stop_level", "direction"]
	);
	assert_eq!(targets.count_enabled(), 3);
}

#[test]
fn config_merge_loads_top_level_horizons() {
	let mut file = tempfile::NamedTempFile::new().unwrap();
	writeln!(file, "horizons = [\"3d\", \"7d\"]").unwrap();

	let config = TrainingConfig::default().with_config_from_file(file.path()).unwrap();

	assert_eq!(config.horizons, vec!["3d", "7d"]);
}

#[test]
fn xgboost_pool_data_defaults_to_empty_when_absent() {
	let config: toml::Value = toml::from_str(
		r#"
[model.xgboost]
enabled = true
feature_dim = 64
n_estimators = 100
max_depth = 6
objective = "RandomForest"
eval_metric = "multiclass_accuracy"
save_feature_importance = true
importance_method = "permutation"
"#,
	)
	.unwrap();
	let xgboost: XGBoostConfig = config["model"]["xgboost"].clone().try_into().unwrap();

	assert!(xgboost.pool_data.is_empty());
}
