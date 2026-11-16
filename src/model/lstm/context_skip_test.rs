use crate::config::model::{HiddenUnitsConfig, LSTMArchitecture, SequenceLengthConfig};
use crate::config::ModelConfig;
use crate::features::context::context_column_indices;
use crate::model::lstm::LSTMModel;
use crate::model::multi_target::MultiTargetLSTMModel;
use candle_core::{DType, Device, Tensor};

const INPUT_SIZE: usize = 4;
const SEQUENCE_LENGTH: usize = 3;
const BATCH_SIZE: usize = 2;
const HIDDEN_SIZE: usize = 6;
const CONTEXT_INDICES: [usize; 2] = [1, 3];

fn model_config(architecture: LSTMArchitecture) -> ModelConfig {
	ModelConfig {
		architecture,
		hidden_units: HiddenUnitsConfig::Fixed(vec![HIDDEN_SIZE as u32]),
		sequence_length: SequenceLengthConfig::Fixed(SEQUENCE_LENGTH as u32),
		..Default::default()
	}
}

fn build_model(architecture: LSTMArchitecture, context_indices: &[usize]) -> LSTMModel {
	let config = model_config(architecture);
	let mut model = LSTMModel::from_model_config_with_seed(&config, INPUT_SIZE, 5, Some(7), Some(Device::Cpu)).unwrap();
	model.set_context_input_indices(context_indices.to_vec()).unwrap();
	model.initialize_network(None).unwrap();
	model
}

/// Every value differs by batch row, timestep and column, so a wrong pick shows up.
fn distinct_input() -> Tensor {
	let values: Vec<f32> = (0..BATCH_SIZE * SEQUENCE_LENGTH * INPUT_SIZE)
		.map(|position| (position * position) as f32 * 0.01 + 1.0)
		.collect();
	Tensor::from_vec(values.clone(), (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE), &Device::Cpu).unwrap()
}

fn rows(tensor: &Tensor) -> Vec<Vec<f32>> {
	tensor.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap()
}

#[test]
fn context_column_indices_picks_only_context_names() {
	let names: Vec<String> = ["close", "ctx_hour_sin", "rsi", "ctx_volatility", "ctxlike"]
		.iter()
		.map(|name| name.to_string())
		.collect();
	assert_eq!(context_column_indices(&names), vec![1, 3]);
	assert!(context_column_indices(&[]).is_empty());
}

#[test]
fn latent_size_adds_context_width() {
	let uni = LSTMArchitecture::MultiLSTM { layers: 1 };
	let bi = LSTMArchitecture::BidirectionalLSTM { layers: 1 };
	assert_eq!(build_model(uni.clone(), &[]).latent_size(), HIDDEN_SIZE);
	assert_eq!(build_model(uni, &CONTEXT_INDICES).latent_size(), HIDDEN_SIZE + 2);
	assert_eq!(build_model(bi.clone(), &[]).latent_size(), HIDDEN_SIZE * 2);
	assert_eq!(build_model(bi, &CONTEXT_INDICES).latent_size(), HIDDEN_SIZE * 2 + 2);
}

#[test]
fn set_context_input_indices_rejects_bad_indices() {
	let config = model_config(LSTMArchitecture::MultiLSTM { layers: 1 });
	let mut model = LSTMModel::from_model_config_with_seed(&config, INPUT_SIZE, 5, Some(7), Some(Device::Cpu)).unwrap();
	assert!(model.set_context_input_indices(vec![INPUT_SIZE]).is_err());
	assert!(model.set_context_input_indices(vec![1, 1]).is_err());
	assert!(model.context_input_indices.is_empty());
	model.set_context_input_indices(vec![1, 3]).unwrap();
	model.initialize_network(None).unwrap();
	assert!(model.set_context_input_indices(vec![2]).is_err());
	model.set_context_input_indices(vec![1, 3]).unwrap();
}

#[test]
fn forward_features_ends_with_last_step_context_columns() {
	let model = build_model(LSTMArchitecture::MultiLSTM { layers: 1 }, &CONTEXT_INDICES);
	let input = distinct_input();
	let latent = rows(&model.forward_features(&input, false).unwrap());
	let raw = input.to_vec3::<f32>().unwrap();
	for (batch, row) in latent.iter().enumerate() {
		assert_eq!(row.len(), HIDDEN_SIZE + 2);
		let last_step = &raw[batch][SEQUENCE_LENGTH - 1];
		assert_eq!(row[HIDDEN_SIZE], last_step[CONTEXT_INDICES[0]]);
		assert_eq!(row[HIDDEN_SIZE + 1], last_step[CONTEXT_INDICES[1]]);
	}
}

#[test]
fn forward_features_width_is_unchanged_without_context() {
	let model = build_model(LSTMArchitecture::MultiLSTM { layers: 1 }, &[]);
	let latent = model.forward_features(&distinct_input(), false).unwrap();
	assert_eq!(latent.dims(), &[BATCH_SIZE, HIDDEN_SIZE]);
}

#[test]
fn tree_model_reads_the_last_input_step_with_context() {
	let model = build_model(LSTMArchitecture::MultiLSTM { layers: 1 }, &CONTEXT_INDICES);
	assert_eq!(model.tree_feature_size(), INPUT_SIZE);
	let input = distinct_input();
	let tree_rows = rows(&model.extract_lstm_features(&input).unwrap());
	let raw = input.to_vec3::<f32>().unwrap();
	for (batch, row) in tree_rows.iter().enumerate() {
		assert_eq!(row, &raw[batch][SEQUENCE_LENGTH - 1]);
	}
}

#[test]
fn tree_model_reads_the_recurrent_state_without_context() {
	let model = build_model(LSTMArchitecture::MultiLSTM { layers: 1 }, &[]);
	assert_eq!(model.tree_feature_size(), HIDDEN_SIZE);
	let input = distinct_input();
	assert_eq!(
		rows(&model.extract_lstm_features(&input).unwrap()),
		rows(&model.forward_features(&input, false).unwrap())
	);
}

#[test]
fn multi_target_save_load_keeps_context_path() {
	let config = model_config(LSTMArchitecture::MultiLSTM { layers: 1 });
	let model = build_model(LSTMArchitecture::MultiLSTM { layers: 1 }, &CONTEXT_INDICES);
	let input = distinct_input();
	let features_before = rows(&model.forward_features(&input, false).unwrap());
	let predictions_before = rows(&model.forward(&input, false).unwrap());

	let mut wrapper = MultiTargetLSTMModel::from_trained_models(
		vec![model],
		vec!["price_level_8h".to_string()],
		vec!["8h".to_string()],
		INPUT_SIZE,
		None,
	)
	.unwrap();
	let names: Vec<String> = ["close", "ctx_hour_sin", "rsi", "ctx_volatility"]
		.iter()
		.map(|name| name.to_string())
		.collect();
	wrapper.set_feature_names(names).unwrap();
	wrapper.set_training_config(crate::config::TrainingConfig {
		horizons: vec!["8h".to_string()],
		model: config,
		..Default::default()
	});
	let directory = tempfile::tempdir().unwrap();
	let path = directory.path().join("T");
	wrapper.save(&path).unwrap();

	let loaded = MultiTargetLSTMModel::load(&path).unwrap().extract_models().unwrap();
	let reloaded = &loaded[0];
	assert_eq!(reloaded.context_input_indices, CONTEXT_INDICES.to_vec());
	assert_eq!(
		rows(&reloaded.forward_features(&input, false).unwrap()),
		features_before
	);
	assert_eq!(rows(&reloaded.forward(&input, false).unwrap()), predictions_before);
}
