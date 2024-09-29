use std::error::Error;
use tch::{
	nn::{self, ModuleT, OptimizerConfig, RNN},
	Device, Kind, Tensor,
};

#[derive(Debug)]
pub struct LSTMModel {
	lstm: nn::LSTM,
	fc: nn::Linear,
}

impl LSTMModel {
	pub fn new(
		vs: &nn::Path,
		input_size: i64,
		hidden_size: i64,
		num_layers: i64,
		output_size: i64,
		dropout_rate: f64,
	) -> Self {
		let lstm_config = nn::RNNConfig {
			num_layers,
			dropout: dropout_rate,
			batch_first: true,
			..Default::default()
		};
		let lstm = nn::lstm(vs / "lstm", input_size, hidden_size, lstm_config);

		let fc = nn::linear(vs / "fc", hidden_size, output_size, Default::default());

		Self {
			lstm,
			fc,
		}
	}
}

impl nn::ModuleT for LSTMModel {
	fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
		let (output, _) = self.lstm.seq(&input);
		let last_output = output.select(1, output.size()[1] - 1);
		let output = if train { last_output.dropout(0.2, train) } else { last_output.shallow_clone() };
		output.apply(&self.fc)
	}
}

pub fn build_and_train_model(
	x_train: Tensor,
	y_train: Tensor,
	x_test: Tensor,
	y_test: Tensor,
	input_size: i64,
) -> Result<(), Box<dyn Error>> {
	let device = Device::cuda_if_available();
	let vs = nn::VarStore::new(device);

	let hidden_size = 64;
	let num_layers = 4;
	let output_size = 1;
	let dropout_rate = 0.2;
	let batch_size = 32;
	let num_epochs = 200;
	let learning_rate = 0.001;

	let model = LSTMModel::new(&vs.root(), input_size, hidden_size, num_layers, output_size, dropout_rate);

	let mut opt = nn::Adam::default().build(&vs, learning_rate)?;

	let train_size = x_train.size()[0];
	let num_batches = (train_size + batch_size - 1) / batch_size;

	let x_train = x_train.to_device(device);
	let y_train = y_train.to_device(device);
	let x_test = x_test.to_device(device);
	let y_test = y_test.to_device(device);

	println!("Starting training loop...");
	for epoch in 1..=num_epochs {
		let mut total_loss = 0.0;

		for batch_idx in 0..num_batches {
			let start = batch_idx * batch_size;
			let end = std::cmp::min(start + batch_size, train_size);
			let batch_x = x_train.narrow(0, start, end - start);
			let batch_y = y_train.narrow(0, start, end - start);

			let output = model.forward_t(&batch_x, true);
			let loss = output.mse_loss(&batch_y, tch::Reduction::Mean);

			opt.zero_grad();
			loss.backward();
			opt.step();

			total_loss += loss.double_value(&[]);
		}

		let avg_loss = total_loss / num_batches as f64;
		println!("Epoch {}: Training loss: {:.6}", epoch, avg_loss);

		if epoch % 10 == 0 || epoch == num_epochs {
			evaluate_model(&model, &x_test, &y_test, epoch)?;
		}
	}

	Ok(())
}

fn evaluate_model(model: &LSTMModel, x_test: &Tensor, y_test: &Tensor, epoch: i32) -> Result<(), Box<dyn Error>> {
	let test_output = tch::no_grad(|| model.forward_t(&x_test, false));
	let test_loss = test_output.mse_loss(&y_test, tch::Reduction::Mean);
	let test_loss_value = test_loss.double_value(&[]);

	let rmse = test_loss_value.sqrt();
	let mae = test_output.f_sub(&y_test)?.abs().mean(Kind::Float).double_value(&[]);

	let ss_res = test_output.f_sub(&y_test)?.f_pow_tensor_scalar(2.0)?.sum(Kind::Float).double_value(&[]);

	let y_mean = y_test.mean(Kind::Float);

	let ss_tot = y_test.f_sub(&y_mean)?.f_pow_tensor_scalar(2.0)?.sum(Kind::Float).double_value(&[]);

	let r2 = 1.0 - ss_res / ss_tot;

	println!("Evaluation at Epoch {}:", epoch);
	println!("Test loss (MSE): {:.6}", test_loss_value);
	println!("Test RMSE: {:.6}", rmse);
	println!("Test MAE: {:.6}", mae);
	println!("Test R² Score: {:.6}", r2);

	Ok(())
}
