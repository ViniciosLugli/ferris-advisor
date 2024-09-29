use ndarray::{s, Array1, Array2};
use polars::{io::mmap::MmapBytesReader, prelude::*};
use rand::{seq::SliceRandom, thread_rng};
use std::error::Error;

pub fn load_data(file_path: &str) -> Result<DataFrame, Box<dyn Error>> {
	println!("Loading data...");
	let file = std::fs::File::open(file_path)?;
	let file = Box::new(file) as Box<dyn MmapBytesReader>;
	let mut df = CsvReader::new(file).finish()?;

	println!("Parsing 'timestamp' column to datetime...");
	df.with_column(
		df.column("timestamp")?.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?.with_name("date".into()),
	)?;

	println!("Data preview:\n{:?}", df.head(Some(5)));
	Ok(df)
}

pub fn select_features(df: &DataFrame) -> Result<DataFrame, Box<dyn Error>> {
	println!("Selecting relevant features...");
	let df_sampled = df.select(["close", "volume", "trades"])?;
	println!("Sampling data...");
	Ok(df_sampled.head(Some(1000)))
}

pub fn normalize_data(features_array: &mut Array2<f64>) -> (Array1<f64>, Array1<f64>) {
	println!("Normalizing data...");
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

pub fn create_sequences(
	data: &Array2<f64>,
	window_size: usize,
	prediction_steps: usize,
) -> (Vec<Array2<f64>>, Vec<f64>) {
	println!("Creating sequences...");
	let mut X = Vec::new();
	let mut y = Vec::new();

	let num_samples = data.shape()[0];

	for i in 0..(num_samples - window_size - prediction_steps + 1) {
		let sequence = data.slice(s![i..(i + window_size), ..]).to_owned();
		let target = data[[i + window_size + prediction_steps - 1, 0]];
		X.push(sequence);
		y.push(target);
	}
	println!("Total sequences created: {}", X.len());
	(X, y)
}

pub fn sequences_to_tensor(sequences: &[Array2<f64>]) -> Result<tch::Tensor, Box<dyn Error>> {
	let num_samples = sequences.len();
	let window_size = sequences[0].shape()[0];
	let num_features = sequences[0].shape()[1];

	let mut data_vec = Vec::with_capacity(num_samples * window_size * num_features);

	for seq in sequences {
		data_vec.extend(seq.iter().cloned());
	}

	let tensor = tch::Tensor::from_slice(&data_vec)
		.reshape(&[num_samples as i64, window_size as i64, num_features as i64])
		.to_kind(tch::Kind::Float);

	Ok(tensor)
}

pub fn shuffle_and_split_data(
	X: Vec<Array2<f64>>,
	y: Vec<f64>,
	train_ratio: f64,
) -> (Vec<Array2<f64>>, Vec<f64>, Vec<Array2<f64>>, Vec<f64>) {
	println!("Shuffling and splitting data...");
	let num_samples = X.len();
	let mut indices: Vec<usize> = (0..num_samples).collect();
	let mut rng = thread_rng();
	indices.shuffle(&mut rng);

	let train_size = (num_samples as f64 * train_ratio) as usize;

	let train_indices = &indices[..train_size];
	let test_indices = &indices[train_size..];

	let mut X_train = Vec::with_capacity(train_size);
	let mut y_train = Vec::with_capacity(train_size);
	let mut X_test = Vec::with_capacity(num_samples - train_size);
	let mut y_test = Vec::with_capacity(num_samples - train_size);

	for &i in train_indices {
		X_train.push(X[i].clone());
		y_train.push(y[i]);
	}

	for &i in test_indices {
		X_test.push(X[i].clone());
		y_test.push(y[i]);
	}

	println!("Training samples: {}, Testing samples: {}", train_size, num_samples - train_size);

	(X_train, y_train, X_test, y_test)
}
