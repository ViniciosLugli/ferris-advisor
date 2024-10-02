use crate::services::KrakenMarketService;
use actix_web::{get, web, HttpResponse, Result};
use log::{debug, error, info};
use model::{self, model::EvaluationMetrics, TimeSeriesModel};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{self, json, Value};
use std::{error::Error, fs::create_dir_all, path::Path, time};

#[derive(Deserialize, Debug)]
struct QueryParams {
	interval: Option<String>,
	since: Option<String>, // Optional "since" parameter for incremental updates
	steps: Option<String>, // Number of future steps to predict
}

#[derive(Debug, Serialize, Deserialize)]
struct PriceHistoryEntry {
	timestamp: i64,
	open: f64,
	high: f64,
	low: f64,
	close: f64,
	volume: f64,
	trades: i64,
}

#[get("/api/get_price_history/{pair}")]
async fn get_price_history(pair: web::Path<String>, query: web::Query<QueryParams>) -> Result<HttpResponse> {
	let pair = pair.into_inner();
	let interval: u32 = query.interval.as_deref().unwrap_or("60").parse().unwrap_or(60);
	let since = query.since.as_deref().map(|s| {
		s.parse()
			.unwrap_or(time::SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap().as_secs().saturating_sub(1000))
	});

	let service = KrakenMarketService::new(pair.clone());
	info!("Fetching price history for pair {} with interval {} minutes since {:?}", pair, interval, since);

	match service.get_price_history(interval, since).await {
		Ok(df) => {
			debug!("DataFrame: {:?}", df);

			let timestamps = df.column("timestamp").unwrap().i64().unwrap().into_no_null_iter().collect::<Vec<_>>();
			let opens = df.column("open").unwrap().f64().unwrap().into_no_null_iter().collect::<Vec<_>>();
			let highs = df.column("high").unwrap().f64().unwrap().into_no_null_iter().collect::<Vec<_>>();
			let lows = df.column("low").unwrap().f64().unwrap().into_no_null_iter().collect::<Vec<_>>();
			let closes = df.column("close").unwrap().f64().unwrap().into_no_null_iter().collect::<Vec<_>>();
			let volumes = df.column("volume").unwrap().f64().unwrap().into_no_null_iter().collect::<Vec<_>>();
			let trades = df.column("trades").unwrap().i64().unwrap().into_no_null_iter().collect::<Vec<_>>();

			let price_history: Vec<PriceHistoryEntry> = timestamps
				.into_iter()
				.zip(opens)
				.zip(highs)
				.zip(lows)
				.zip(closes)
				.zip(volumes)
				.zip(trades)
				.map(|((((((timestamp, open), high), low), close), volume), trades)| PriceHistoryEntry {
					timestamp,
					open,
					high,
					low,
					close,
					volume,
					trades,
				})
				.collect();

			Ok(HttpResponse::Ok().json(price_history))
		}
		Err(e) => {
			error!("Error fetching price history: {}", e);
			Ok(HttpResponse::InternalServerError().body(e.to_string()))
		}
	}
}

fn train_time_series_model(pair: &str, df: DataFrame) -> Result<EvaluationMetrics, Box<dyn Error>> {
	let window_size = 14;
	let prediction_steps = 7;
	let sample_size = None;

	let mut model = TimeSeriesModel::new(
		3,   // input_size (close, volume, count)
		64,  // hidden_size
		4,   // num_layers
		1,   // output_size (predicting close price)
		0.2, // dropout_rate
		window_size,
		prediction_steps,
	);

	let (x_tensor, y_tensor) = model.prepare_data(&df, sample_size)?;

	let train_ratio = 0.8;
	let (x_train, y_train, x_test, y_test) = model.train_test_split(&x_tensor, &y_tensor, train_ratio);

	model.train(&x_train, &y_train, 100, 0.001, 32)?;

	let metrics = model.evaluate(&x_test, &y_test)?;
	info!("Evaluation Metrics:");
	info!("MSE: {:.6}", metrics.mse);
	info!("RMSE: {:.6}", metrics.rmse);
	info!("MAE: {:.6}", metrics.mae);
	info!("R2 Score: {:.6}", metrics.r2);

	let model_dir = "models";
	let _ = create_dir_all(model_dir);
	let model_path = format!("{}/{}_model.safetensors", model_dir, pair);
	model.save(&model_path)?;
	let metrics_path = format!("models/{}_metrics.json", pair);

	let metrics_json = serde_json::to_string(&metrics)?;
	std::fs::write(metrics_path, metrics_json)?;
	Ok(metrics)
}

#[get("/api/train_model/{pair}")]
async fn train_model(pair: web::Path<String>, query: web::Query<QueryParams>) -> Result<HttpResponse> {
	let pair = pair.into_inner();
	let interval: u32 = query.interval.as_deref().unwrap_or("60").parse().unwrap_or(60);
	let since = query.since.as_deref().map(|s| {
		s.parse().unwrap_or(
			time::SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap().as_secs().saturating_sub(86400 * 30),
		)
	});

	let service = KrakenMarketService::new(pair.clone());
	info!("Fetching price history for pair {} with interval {} minutes", pair, interval);

	match service.get_price_history(interval, since).await {
		Ok(df) => {
			debug!("DataFrame: {:?}", df);

			let column_names = df.get_column_names();
			debug!("Column names in the DataFrame: {:?}", column_names);

			info!("Starting model training...");
			let model_training_result = train_time_series_model(&pair, df);

			match model_training_result {
				Ok(metrics) => Ok(HttpResponse::Ok().json(json!({
					"mse": metrics.mse,
					"rmse": metrics.rmse,
					"mae": metrics.mae,
					"r2": metrics.r2
				}))),
				Err(e) => {
					error!("Error during model training: {}", e);
					Ok(HttpResponse::InternalServerError().body(e.to_string()))
				}
			}
		}
		Err(e) => {
			error!("Error fetching price history: {}", e);
			Ok(HttpResponse::InternalServerError().body(e.to_string()))
		}
	}
}

#[get("/api/check_model/{pair}")]
async fn check_model(pair: web::Path<String>) -> Result<HttpResponse> {
	let pair = pair.into_inner();
	let model_path = format!("models/{}_model.safetensors", pair);
	let model_metrics_path = format!("models/{}_metrics.json", pair);
	if Path::new(&model_path).exists() && Path::new(&model_metrics_path).exists() {
		let metrics_json: Value = serde_json::from_str(&std::fs::read_to_string(model_metrics_path)?)?;
		Ok(HttpResponse::Ok().json(json!({ "model_exists": true, "metrics": metrics_json })))
	} else {
		Ok(HttpResponse::Ok().json(json!({ "model_exists": false })))
	}
}

#[get("/api/predict/{pair}")]
async fn predict(pair: web::Path<String>, query: web::Query<QueryParams>) -> Result<HttpResponse> {
	let pair = pair.into_inner();
	let steps: usize = query.steps.as_deref().unwrap_or("8").parse().unwrap_or(8);

	let model_path = format!("models/{}_model.safetensors", pair);
	info!("Predicting future price for pair {} with {} steps", pair, steps);
	if !Path::new(&model_path).exists() {
		return Ok(HttpResponse::BadRequest().json(json!({
			"error": "Model does not exist. Please train the model first."
		})));
	}

	let interval: u32 = query.interval.as_deref().unwrap_or("60").parse().unwrap_or(60);
	let since = Some(time::SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap().as_secs().saturating_sub(86400));

	let service = KrakenMarketService::new(pair.clone());
	info!("Fetching latest price history for pair {} for prediction", pair);

	match service.get_price_history(interval, since).await {
		Ok(df) => {
			let window_size = 14;
			let prediction_steps = steps;

			let mut model = TimeSeriesModel::new(
				3,   // input_size (close, volume, count)
				64,  // hidden_size
				4,   // num_layers
				1,   // output_size (predicting close price)
				0.2, // dropout_rate
				window_size,
				prediction_steps,
			);

			model.load(&model_path)?;

			let predictions = model.predict_future(&df, steps)?;

			Ok(HttpResponse::Ok().json(predictions))
		}
		Err(e) => {
			error!("Error fetching price history: {}", e);
			Ok(HttpResponse::InternalServerError().body(e.to_string()))
		}
	}
}
