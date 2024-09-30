use crate::data_processing::DataProcessor;
use log::info;
use ndarray::{s, Array1, Array2};
use polars::prelude::*;
use std::error::Error;
use tch::{
	nn::{self, OptimizerConfig, RNN},
	Device, Kind, Tensor,
};

pub struct TimeSeriesModel {
	gru: nn::GRU,
	fc: nn::Linear,
	vs: nn::VarStore,
	device: Device,
	pub input_size: i64,
	hidden_size: i64,
	num_layers: i64,
	dropout_rate: f64,
	pub means: Option<Array1<f64>>,
	pub stds: Option<Array1<f64>>,
	pub window_size: usize,
	pub prediction_steps: usize,
}

impl TimeSeriesModel {
	pub fn new(
		input_size: i64,
		hidden_size: i64,
		num_layers: i64,
		output_size: i64,
		dropout_rate: f64,
		window_size: usize,
		prediction_steps: usize,
	) -> Self {
		let device = Device::cuda_if_available();
		let vs = nn::VarStore::new(device);
		let gru_config = nn::RNNConfig {
			num_layers,
			dropout: dropout_rate,
			batch_first: true,
			bidirectional: true,
			..Default::default()
		};
		let gru = nn::gru(&vs.root() / "gru", input_size, hidden_size, gru_config);
		let fc = nn::linear(&vs.root() / "fc", hidden_size * 2, output_size, Default::default());
		TimeSeriesModel {
			gru,
			fc,
			vs,
			device,
			input_size,
			hidden_size,
			num_layers,
			dropout_rate,
			means: None,
			stds: None,
			window_size,
			prediction_steps,
		}
	}

	pub fn prepare_data(
		&mut self,
		df: &DataFrame,
		sample_size: Option<usize>,
	) -> Result<(Tensor, Tensor), Box<dyn Error>> {
		info!("Selecting relevant features...");
		let df_sampled = df.select(["close", "volume", "trades"])?;
		let df_sampled = match sample_size {
			Some(size) => df_sampled.head(Some(size)),
			None => df_sampled.clone(),
		};
		info!("Converting DataFrame to ndarray...");
		let features_data = df_sampled.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
		let mut features_array = Array2::<f64>::from_shape_vec(
			(features_data.shape()[0], features_data.shape()[1]),
			features_data.iter().cloned().collect(),
		)?;
		info!("Normalizing data...");
		let processor = DataProcessor::new(self.window_size, self.prediction_steps);
		let (means, stds) = processor.normalize_data(&mut features_array);
		self.means = Some(means);
		self.stds = Some(stds);
		info!("Creating sequences...");
		let (X, y) = processor.create_sequences(&features_array);
		info!("Converting sequences to tensors...");
		let x_tensor = processor.sequences_to_tensor(&X)?.to_device(self.device);
		let y_tensor = Tensor::from_slice(&y).to_kind(Kind::Float).unsqueeze(1).to_device(self.device);
		Ok((x_tensor, y_tensor))
	}

	pub fn train(
		&mut self,
		x_train: &Tensor,
		y_train: &Tensor,
		num_epochs: i64,
		learning_rate: f64,
		batch_size: i64,
	) -> Result<(), Box<dyn Error>> {
		let mut opt = nn::Adam::default().build(&self.vs, learning_rate)?;
		let train_size = x_train.size()[0];
		let num_batches = (train_size + batch_size - 1) / batch_size;
		info!("Starting training...");
		for epoch in 1..=num_epochs {
			let mut total_loss = 0.0;
			for batch_idx in 0..num_batches {
				let start = batch_idx * batch_size;
				let end = std::cmp::min(start + batch_size, train_size);
				let batch_x = x_train.narrow(0, start, end - start);
				let batch_y = y_train.narrow(0, start, end - start);
				let output = self.forward(&batch_x, true);
				let loss = output.mse_loss(&batch_y, tch::Reduction::Mean);
				opt.zero_grad();
				loss.backward();
				opt.step();
				total_loss += loss.double_value(&[]);
			}
			let avg_loss = total_loss / num_batches as f64;
			info!("Epoch {}: Training Loss: {:.6}", epoch, avg_loss);
		}
		Ok(())
	}

	pub fn train_test_split(
		&self,
		x_tensor: &Tensor,
		y_tensor: &Tensor,
		train_ratio: f64,
	) -> (Tensor, Tensor, Tensor, Tensor) {
		let train_size = (x_tensor.size()[0] as f64 * train_ratio) as i64;
		let x_train = x_tensor.narrow(0, 0, train_size);
		let y_train = y_tensor.narrow(0, 0, train_size);
		let x_test = x_tensor.narrow(0, train_size, x_tensor.size()[0] - train_size);
		let y_test = y_tensor.narrow(0, train_size, y_tensor.size()[0] - train_size);

		(x_train, y_train, x_test, y_test)
	}

	pub fn evaluate(&self, x_test: &Tensor, y_test: &Tensor) -> Result<EvaluationMetrics, Box<dyn Error>> {
		let test_output = tch::no_grad(|| self.forward(x_test, false));
		let mse = test_output.mse_loss(y_test, tch::Reduction::Mean).double_value(&[]);
		let rmse = mse.sqrt();
		let mae = test_output.f_sub(y_test)?.abs().mean(Kind::Float).double_value(&[]);
		let ss_res = test_output.f_sub(y_test)?.f_pow_tensor_scalar(2.0)?.sum(Kind::Float).double_value(&[]);
		let y_mean = y_test.mean(Kind::Float);
		let ss_tot = y_test.f_sub(&y_mean)?.f_pow_tensor_scalar(2.0)?.sum(Kind::Float).double_value(&[]);
		let r2 = 1.0 - ss_res / ss_tot;
		Ok(EvaluationMetrics {
			mse,
			rmse,
			mae,
			r2,
		})
	}

	pub fn predict_future(&self, df: &DataFrame, steps: usize) -> Result<Vec<f64>, Box<dyn Error>> {
		let df_sampled = df.select(["close", "volume", "trades"])?;
		let features_data = df_sampled.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
		let mut features_array = Array2::<f64>::from_shape_vec(
			(features_data.shape()[0], features_data.shape()[1]),
			features_data.iter().cloned().collect(),
		)?;
		if let (Some(means), Some(stds)) = (&self.means, &self.stds) {
			info!("Normalizing data with stored means and stds...");
			for col in 0..features_array.shape()[1] {
				for row in 0..features_array.shape()[0] {
					features_array[[row, col]] = (features_array[[row, col]] - means[col]) / stds[col];
				}
			}
		} else {
			return Err("Model not trained or normalization parameters missing".into());
		}
		let start = features_array.shape()[0] - self.window_size;
		let mut inputs = features_array.slice(s![start.., ..]).to_owned();
		let mut predictions = Vec::new();
		for _ in 0..steps {
			let x_input = Tensor::from_slice(inputs.as_slice().unwrap())
				.reshape(&[1, self.window_size as i64, self.input_size])
				.to_kind(Kind::Float)
				.to_device(self.device);
			let pred = self.predict(&x_input);
			let pred_value = pred.double_value(&[0, 0]);
			predictions.push(pred_value);
			let new_row =
				Array2::from_shape_vec((1, self.input_size as usize), vec![pred_value; self.input_size as usize])?;
			inputs = Array2::from_shape_vec(
				(self.window_size, self.input_size as usize),
				inputs.iter().cloned().skip(self.input_size as usize).chain(new_row.iter().cloned()).collect(),
			)?;
		}
		Ok(predictions)
	}

	pub fn predict(&self, x_input: &Tensor) -> Tensor { self.forward(x_input, false) }

	pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
		self.vs.save(path)?;
		Ok(())
	}

	pub fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
		self.vs.load(path)?;
		Ok(())
	}

	fn forward(&self, input: &Tensor, train: bool) -> Tensor {
		let h0 = nn::GRUState(Tensor::zeros(
			&[self.num_layers * 2, input.size()[0], self.hidden_size],
			(Kind::Float, self.device),
		));
		let (output, _) = self.gru.seq_init(&input, &h0);
		let last_output = output.select(1, output.size()[1] - 1);
		let output = if train { last_output.dropout(self.dropout_rate, train) } else { last_output.shallow_clone() };
		output.apply(&self.fc)
	}
}

pub struct EvaluationMetrics {
	pub mse: f64,
	pub rmse: f64,
	pub mae: f64,
	pub r2: f64,
}
