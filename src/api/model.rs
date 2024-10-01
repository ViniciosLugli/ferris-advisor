use crate::services::YahooFinanceService;
use actix_web::{get, web, HttpResponse, Result};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct QueryParams {
	period: Option<String>,
	interval: Option<String>,
}

#[get("/api/get_price_history/{ticker}")]
async fn get_price_history(ticker: web::Path<String>, query: web::Query<QueryParams>) -> Result<HttpResponse> {
	let ticker = ticker.into_inner();

	let period = query.period.as_deref().unwrap_or("1mo");
	let interval = query.interval.as_deref().unwrap_or("1h");

	let mut service = YahooFinanceService::new(ticker.clone());
	println!("Fetching price history for {} with period {} and interval {}", ticker, period, interval);
	match service.get_price_history(period, interval).await {
		Ok(df) => Ok(HttpResponse::Ok().json(df)),
		Err(e) => Ok(HttpResponse::InternalServerError().body(e.to_string())),
	}
}
