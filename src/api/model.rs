use crate::services::YahooFinanceService;
use leptos::*;
use model;
use polars::prelude::*;
use time::{Duration, OffsetDateTime};

#[server(GetPriceHistoy, "/api")]
pub async fn get_price_history(ticker: String) -> Result<DataFrame, ServerFnError> {
	let mut service = YahooFinanceService::new(ticker);
	service
		.set_date_range(OffsetDateTime::now_utc().checked_sub(Duration::days(30)).unwrap(), OffsetDateTime::now_utc());
	service.fetch_price_history()?;
	Ok(service.get_price_history().expect("No price history found").clone())
}
