use ndarray::Array2;
use polars::prelude::*;
use std::error::Error;
use tch::Kind;

use model::{
	data_processing::{
		create_sequences, load_data, normalize_data, select_features, sequences_to_tensor, shuffle_and_split_data,
	},
	model::build_and_train_model,
};

fn main() -> Result<(), Box<dyn Error>> {
	let df = load_data("./ADAEUR_60.csv")?;
	let df_sampled = select_features(&df)?;
	println!("Converting DataFrame to ndarray...");
	let features_data = df_sampled.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
	let mut features_array = Array2::<f64>::from_shape_vec(
		(features_data.shape()[0], features_data.shape()[1]),
		features_data.iter().cloned().collect(),
	)?;

	let (_means, _stds) = normalize_data(&mut features_array);

	let window_size = 14;
	let prediction_steps = 7;

	let (X, y) = create_sequences(&features_array, window_size, prediction_steps);

	let (X_train, y_train, X_test, y_test) = shuffle_and_split_data(X, y, 0.8);

	let x_train_tensor = sequences_to_tensor(&X_train)?;
	let y_train_tensor = tch::Tensor::from_slice(&y_train).to_kind(Kind::Float).unsqueeze(1);
	let x_test_tensor = sequences_to_tensor(&X_test)?;
	let y_test_tensor = tch::Tensor::from_slice(&y_test).to_kind(Kind::Float).unsqueeze(1);

	let num_features = features_array.shape()[1] as i64;
	build_and_train_model(x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, num_features)?;

	Ok(())
}
