use ndarray::Array2;

/// Last input step of other symbols' training sequences and their own-label targets for one head.
#[derive(Debug, Clone)]
pub struct ForestPool {
	pub inputs: Array2<f64>,
	pub targets: Array2<f64>,
}
