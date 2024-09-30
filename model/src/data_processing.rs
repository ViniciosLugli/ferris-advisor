use ndarray::{s, Array1, Array2};
use std::error::Error;
use tch::Tensor;

pub struct DataProcessor {
	pub window_size: usize,
	pub prediction_steps: usize,
}

impl DataProcessor {
	pub fn new(window_size: usize, prediction_steps: usize) -> Self {
		Self {
			window_size,
			prediction_steps,
		}
	}

	pub fn normalize_data(&self, features_array: &mut Array2<f64>) -> (Array1<f64>, Array1<f64>) {
		let (rows, cols) = features_array.dim();
		let mut means = Array1::<f64>::zeros(cols);
		let mut stds = Array1::<f64>::zeros(cols);

		for col in 0..cols {
			let column = features_array.slice(s![.., col]);
			let mean = column.mean().unwrap();
			let std = column.std(0.0);
			means[col] = mean;
			stds[col] = std;
			for row in 0..rows {
				features_array[[row, col]] = (features_array[[row, col]] - mean) / std;
			}
		}
		(means, stds)
	}

	pub fn create_sequences(&self, data: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<f64>) {
		let mut X = Vec::new();
		let mut y = Vec::new();

		let num_samples = data.shape()[0];

		for i in 0..(num_samples - self.window_size - self.prediction_steps + 1) {
			let sequence = data.slice(s![i..(i + self.window_size), ..]).to_owned();
			let target = data[[i + self.window_size + self.prediction_steps - 1, 0]];
			X.push(sequence);
			y.push(target);
		}
		(X, y)
	}

	pub fn sequences_to_tensor(&self, sequences: &[Array2<f64>]) -> Result<Tensor, Box<dyn Error>> {
		let num_samples = sequences.len();
		let window_size = sequences[0].shape()[0];
		let num_features = sequences[0].shape()[1];

		let mut data_vec = Vec::with_capacity(num_samples * window_size * num_features);

		for seq in sequences {
			data_vec.extend(seq.iter().cloned());
		}

		let tensor = Tensor::from_slice(&data_vec)
			.reshape(&[num_samples as i64, window_size as i64, num_features as i64])
			.to_kind(tch::Kind::Float);

		Ok(tensor)
	}
}
