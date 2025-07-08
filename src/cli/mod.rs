// CLI module for unified symbol interface
pub mod symbol_parser;

pub use symbol_parser::{
    build_predict_command, build_train_command, generate_examples, parse_predict_args,
    parse_symbols, parse_train_args, resolve_data_paths, select_config,
    validate_symbol_compatibility, DataPaths, PredictArgs, TrainArgs,
};
