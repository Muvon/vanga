use crate::model::lstm::training::{apply_vote_power, fit_blend};
use ndarray::Array2;

const ROWS: usize = 12;
const CLASSES: usize = 5;
const PERFECT_PROBABILITY: f64 = 0.99;
const WEIGHT_TOLERANCE: f64 = 1e-9;
const TRUE_CLASS_SHARE: f64 = 0.4;
const OVERSHARP_POWER: f64 = 4.0;
const POWER_GRID_STEP: f64 = 0.05;
/// A multiple of CLASSES * CLASSES, so every favoured class meets every draw equally often
const CALIBRATED_ROWS: usize = CLASSES * CLASSES * 2;

fn class_of(row: usize) -> usize {
	row % CLASSES
}

fn labels() -> Vec<usize> {
	(0..ROWS).map(class_of).collect()
}

fn uniform() -> Array2<f64> {
	Array2::from_elem((ROWS, CLASSES), 1.0 / CLASSES as f64)
}

fn perfect() -> Array2<f64> {
	let off_class = (1.0 - PERFECT_PROBABILITY) / (CLASSES - 1) as f64;
	Array2::from_shape_fn((ROWS, CLASSES), |(row, class)| {
		if class == class_of(row) {
			PERFECT_PROBABILITY
		} else {
			off_class
		}
	})
}

/// Every row gives `TRUE_CLASS_SHARE` to one class and splits the rest evenly. The class it
/// favours is the true one in exactly `TRUE_CLASS_SHARE` of the rows, so it is calibrated.
fn calibrated() -> Array2<f64> {
	let off_class = (1.0 - TRUE_CLASS_SHARE) / (CLASSES - 1) as f64;
	Array2::from_shape_fn((CALIBRATED_ROWS, CLASSES), |(row, class)| {
		if class == favoured_class(row) {
			TRUE_CLASS_SHARE
		} else {
			off_class
		}
	})
}

fn favoured_class(row: usize) -> usize {
	row % CLASSES
}

/// True class of the rows of `calibrated()`: the favoured class in two rows of five,
/// another class in the rest.
fn calibrated_labels() -> Vec<usize> {
	(0..CALIBRATED_ROWS)
		.map(|row| {
			let draw = (row / CLASSES) % CLASSES;
			if (draw as f64) < TRUE_CLASS_SHARE * CLASSES as f64 {
				favoured_class(row)
			} else {
				(favoured_class(row) + draw) % CLASSES
			}
		})
		.collect()
}

#[test]
fn perfect_forest_and_uniform_lstm_give_weight_zero() {
	let fit = fit_blend(&uniform(), &perfect(), &labels());
	assert!(fit.lstm_weight.abs() < WEIGHT_TOLERANCE);
	assert!(fit.forest_vote_power >= 1.0);
}

#[test]
fn perfect_lstm_and_uniform_forest_give_weight_one() {
	let fit = fit_blend(&perfect(), &uniform(), &labels());
	assert!((fit.lstm_weight - 1.0).abs() < WEIGHT_TOLERANCE);
}

#[test]
fn identical_inputs_give_a_weight_inside_the_unit_interval() {
	let fit = fit_blend(&perfect(), &perfect(), &labels());
	assert!((0.0..=1.0).contains(&fit.lstm_weight));
	assert!(fit.log_loss.is_finite());
}

#[test]
fn vote_power_keeps_rows_normalised_and_zero_makes_them_uniform() {
	let raised = apply_vote_power(&perfect(), OVERSHARP_POWER);
	for row in raised.rows() {
		assert!((row.sum() - 1.0).abs() < WEIGHT_TOLERANCE);
	}
	let flat = apply_vote_power(&perfect(), 0.0);
	assert!(flat
		.iter()
		.all(|&value| (value - 1.0 / CLASSES as f64).abs() < WEIGHT_TOLERANCE));
}

fn log_loss(probabilities: &Array2<f64>, classes: &[usize]) -> f64 {
	let total: f64 = classes
		.iter()
		.enumerate()
		.map(|(row, &class)| -probabilities[[row, class]].ln())
		.sum();
	total / classes.len() as f64
}

/// The uniform partner can flatten too, so several pairs reach the calibrated loss; the test
/// checks the loss reached, not which pair.
#[test]
fn oversharp_forest_votes_are_brought_back_to_the_calibrated_loss() {
	let classes = calibrated_labels();
	let oversharp = apply_vote_power(&calibrated(), OVERSHARP_POWER);
	let knows_nothing = Array2::from_elem(oversharp.raw_dim(), 1.0 / CLASSES as f64);
	let fit = fit_blend(&knows_nothing, &oversharp, &classes);
	assert!((fit.log_loss - log_loss(&calibrated(), &classes)).abs() < WEIGHT_TOLERANCE);
	assert!(fit.log_loss < log_loss(&oversharp, &classes));
}

#[test]
fn oversharp_forest_votes_alone_get_the_exponent_that_undoes_the_sharpening() {
	let classes = calibrated_labels();
	let oversharp = apply_vote_power(&calibrated(), OVERSHARP_POWER);
	// A partner that is always wrong gets no share, so only the exponent can help
	let always_wrong = Array2::from_shape_fn(oversharp.raw_dim(), |(row, class)| {
		if class == classes[row] {
			1.0 - PERFECT_PROBABILITY
		} else {
			PERFECT_PROBABILITY / (CLASSES - 1) as f64
		}
	});
	let fit = fit_blend(&always_wrong, &oversharp, &classes);
	assert!(fit.lstm_weight.abs() < WEIGHT_TOLERANCE);
	assert!((fit.forest_vote_power - 1.0 / OVERSHARP_POWER).abs() <= POWER_GRID_STEP);
}
