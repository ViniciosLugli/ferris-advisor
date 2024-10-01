use anyhow::Result;
use polars::prelude::*;
use yahoo::YahooConnector;
use yahoo_finance_api as yahoo;

pub struct YahooFinanceService {
	ticker: String,
	provider: YahooConnector,
}

impl YahooFinanceService {
	pub fn new(ticker: String) -> Self {
		Self {
			ticker,
			provider: YahooConnector::new().expect("Failed to create YahooConnector"),
		}
	}

	pub async fn get_price_history(&mut self, period: &str, interval: &str) -> Result<DataFrame> {
		let response = self.provider.get_quote_range(&self.ticker, interval, period).await?;
		let quotes = response.quotes()?;
		println!("Fetched {} quotes", quotes.len());
		let date: Vec<_> = quotes.iter().map(|q| q.timestamp).collect();
		let open: Vec<_> = quotes.iter().map(|q| q.open).collect();
		let high: Vec<_> = quotes.iter().map(|q| q.high).collect();
		let low: Vec<_> = quotes.iter().map(|q| q.low).collect();
		let close: Vec<_> = quotes.iter().map(|q| q.close).collect();
		let volume: Vec<_> = quotes.iter().map(|q| q.volume as f64).collect();
		let adjclose: Vec<_> = quotes.iter().map(|q| q.adjclose).collect();

		let df = df!(
			"date" => Series::new("date", date).cast(&DataType::Date)?,
			"open" => open,
			"high" => high,
			"low" => low,
			"close" => close,
			"volume" => volume,
			"adjusted" => adjclose
		)?;

		Ok(df)
	}

	pub async fn get_latest_quote(&mut self) -> Result<DataFrame> {
		let response = self.provider.get_latest_quotes(&self.ticker, "1d").await?;
		let quote = response.last_quote()?;

		let timestamp = vec![quote.timestamp];
		let open = vec![quote.open];
		let high = vec![quote.high];
		let low = vec![quote.low];
		let close = vec![quote.close];
		let volume = vec![quote.volume as f64];
		let adjclose = vec![quote.adjclose];

		let df = df!(
			"timestamp" => timestamp,
			"open" => open,
			"high" => high,
			"low" => low,
			"close" => close,
			"volume" => volume,
			"adjusted" => adjclose
		)?;

		Ok(df)
	}
}
