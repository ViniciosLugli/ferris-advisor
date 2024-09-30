mod data_processing;
mod model;
mod utils;

use crate::model::TimeSeriesModel;
use env_logger::Env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
	env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

	let df = utils::df_from_csv("./ADAEUR_60.csv")?; // Test data

	let window_size = 14;
	let prediction_steps = 7;
	let sample_size = Some(1000);

	let mut model = TimeSeriesModel::new(
		3,   // input_size
		64,  // hidden_size
		4,   // num_layers
		1,   // output_size
		0.2, // dropout_rate
		window_size,
		prediction_steps,
	);

	let (x_tensor, y_tensor) = model.prepare_data(&df, sample_size)?;

	let train_ratio = 0.8;
	let (x_train, y_train, x_test, y_test) = model.train_test_split(&x_tensor, &y_tensor, train_ratio);

	model.train(&x_train, &y_train, 300, 0.001, 32)?;

	let metrics = model.evaluate(&x_test, &y_test)?;
	println!("Evaluation Metrics:");
	println!("MSE: {:.6}", metrics.mse);
	println!("RMSE: {:.6}", metrics.rmse);
	println!("MAE: {:.6}", metrics.mae);
	println!("R2 Score: {:.6}", metrics.r2);

	model.save("model.safetensors")?;

	let future_steps = 10;
	let predictions = model.predict_future(&df, future_steps)?;
	println!("Future Predictions:");
	for pred in predictions {
		println!("{:.6}", pred);
	}

	Ok(())
}
