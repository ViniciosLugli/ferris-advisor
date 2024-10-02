use chrono::{TimeZone, Utc};
use leptos::*;
use leptos_chartistry::*;
use serde::{Deserialize, Serialize};

use crate::components::fetch_api;

#[derive(Deserialize, Serialize, Clone)]
pub struct PriceHistoryEntry {
	timestamp: i64,
	open: f64,
	high: f64,
	low: f64,
	close: f64,
	volume: f64,
	trades: i64,
}

#[server(PredictPrices, "/api")]
pub async fn predict_prices(pair: String) -> Result<Vec<f64>, ServerFnError> {
	println!("Predicting prices for pair: {}", pair);
	let url = format!("http://0.0.0.0:3333/api/predict/{}", pair);
	let response = fetch_api::<Vec<f64>>(&url).await;

	match response {
		Some(data) => Ok(data),
		None => Err(ServerFnError::new("Failed to predict prices")),
	}
}

#[server(PriceHistory, "/api")]
pub async fn history(pair: String) -> Result<Vec<PriceHistoryEntry>, ServerFnError> {
	println!("Predicting prices for pair: {}", pair);
	let url = format!("http://0.0.0.0:3333/api/get_price_history/{}", pair);
	let response = fetch_api::<Vec<PriceHistoryEntry>>(&url).await;

	match response {
		Some(data) => Ok(data),
		None => Err(ServerFnError::new("Failed to predict prices")),
	}
}

#[component]
pub fn PriceGraph(pair: ReadSignal<String>) -> impl IntoView {
	let (price_history, set_price_history) = create_signal(Vec::<PriceHistoryEntry>::new());
	let (predictions, set_predictions) = create_signal(Vec::new());

	let pair = pair.get();

	spawn_local({
		let set_price_history = set_price_history.clone();
		let pair = pair.clone();
		async move {
			let history = history(pair).await.unwrap_or_default();
			set_price_history.set(history);
		}
	});

	spawn_local({
		let set_predictions = set_predictions.clone();
		let pair = pair.clone();
		async move {
			let preds = predict_prices(pair).await.unwrap_or_default();
			set_predictions.set(preds);
		}
	});

	let series = Series::new(|data: &PriceHistoryEntry| data.timestamp as f64)
		.line(Line::new(|data: &PriceHistoryEntry| data.close).with_name("Historical Prices"))
		.line(Line::new(|data: &PriceHistoryEntry| data.close).with_name("Predicted Prices"));

	view! {
		<div class="my-10">
			<Chart
				aspect_ratio=AspectRatio::from_outer_height(300.0, 1.2)
				series=series
				data=price_history
				tooltip=Tooltip::left_cursor().show_x_ticks(false)
			/>
		</div>
	}
}
