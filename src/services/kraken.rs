use anyhow::{Error, Result};
use polars::prelude::*;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct KrakenOHLCResponse {
	error: Vec<String>,
	result: HashMap<String, Value>,
}

pub struct KrakenMarketService {
	pair: String,
	client: Client,
}

impl KrakenMarketService {
	pub fn new(pair: String) -> Self {
		Self {
			pair,
			client: Client::new(),
		}
	}

	pub async fn get_price_history(&self, interval: u32, since: Option<u64>) -> Result<DataFrame> {
		let base_url = "https://api.kraken.com/0/public/OHLC";
		let mut url = format!("{}?pair={}&interval={}", base_url, self.pair, interval);

		if let Some(since_timestamp) = since {
			url.push_str(&format!("&since={}", since_timestamp));
		}

		println!("Fetching OHLC data from Kraken API: {}", url);

		let response = self.client.get(&url).send().await?;
		let body = response.text().await?;

		let kraken_response: KrakenOHLCResponse = serde_json::from_str(&body)
			.map_err(|e| Error::msg(format!("Failed to parse JSON: {} - Raw response: {}", e, body)))?;

		if !kraken_response.error.is_empty() {
			return Err(Error::msg(format!("Kraken API Error: {:?}", kraken_response.error)));
		}

		let mut result = kraken_response.result;

		let _last = result
			.remove("last")
			.ok_or_else(|| Error::msg("Missing 'last' field in result"))?
			.as_i64()
			.ok_or_else(|| Error::msg("'last' field is not an integer"))?;

		let (_pair_key, data_value) =
			result.into_iter().next().ok_or_else(|| Error::msg("No OHLC data available in the response"))?;

		let data_array = data_value.as_array().ok_or_else(|| Error::msg("Expected data array"))?;

		let timestamps = data_array
			.iter()
			.map(|entry| {
				entry
					.as_array()
					.and_then(|arr| arr.get(0).and_then(Value::as_i64))
					.ok_or_else(|| Error::msg("Expected integer timestamp"))
			})
			.collect::<Result<Vec<_>>>()?;

		let open_prices = data_array
			.iter()
			.map(|entry| {
				entry
					.as_array()
					.and_then(|arr| arr.get(1).and_then(Value::as_str))
					.ok_or_else(|| Error::msg("Expected string open price"))?
					.parse::<f64>()
					.map_err(|e| Error::msg(format!("Failed to parse open price: {}", e)))
			})
			.collect::<Result<Vec<_>>>()?;

		let high_prices = data_array
			.iter()
			.map(|entry| {
				entry
					.as_array()
					.and_then(|arr| arr.get(2).and_then(Value::as_str))
					.ok_or_else(|| Error::msg("Expected string high price"))?
					.parse::<f64>()
					.map_err(|e| Error::msg(format!("Failed to parse high price: {}", e)))
			})
			.collect::<Result<Vec<_>>>()?;

		let low_prices = data_array
			.iter()
			.map(|entry| {
				entry
					.as_array()
					.and_then(|arr| arr.get(3).and_then(Value::as_str))
					.ok_or_else(|| Error::msg("Expected string low price"))?
					.parse::<f64>()
					.map_err(|e| Error::msg(format!("Failed to parse low price: {}", e)))
			})
			.collect::<Result<Vec<_>>>()?;

		let close_prices = data_array
			.iter()
			.map(|entry| {
				entry
					.as_array()
					.and_then(|arr| arr.get(4).and_then(Value::as_str))
					.ok_or_else(|| Error::msg("Expected string close price"))?
					.parse::<f64>()
					.map_err(|e| Error::msg(format!("Failed to parse close price: {}", e)))
			})
			.collect::<Result<Vec<_>>>()?;

		let volumes = data_array
			.iter()
			.map(|entry| {
				entry
					.as_array()
					.and_then(|arr| arr.get(6).and_then(Value::as_str))
					.ok_or_else(|| Error::msg("Expected string volume"))?
					.parse::<f64>()
					.map_err(|e| Error::msg(format!("Failed to parse volume: {}", e)))
			})
			.collect::<Result<Vec<_>>>()?;

		let counts = data_array
			.iter()
			.map(|entry| {
				entry
					.as_array()
					.and_then(|arr| arr.get(7).and_then(Value::as_i64))
					.ok_or_else(|| Error::msg("Expected integer count"))
			})
			.collect::<Result<Vec<_>>>()?;

		let df = df![
			"timestamp" => timestamps,
			"open" => open_prices,
			"high" => high_prices,
			"low" => low_prices,
			"close" => close_prices,
			"volume" => volumes,
			"trades" => counts
		]?;

		Ok(df)
	}
}
