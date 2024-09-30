use polars::prelude::*;
use time::OffsetDateTime;
use RustQuant::{data::*, error::RustQuantError};

pub struct YahooFinanceService {
	data: YahooFinanceData,
}

impl YahooFinanceService {
	pub fn new(ticker: String) -> Self {
		Self {
			data: YahooFinanceData::new(ticker),
		}
	}

	pub fn set_date_range(&mut self, start: OffsetDateTime, end: OffsetDateTime) {
		self.data.set_date_range(start, end);
	}

	pub fn fetch_price_history(&mut self) -> Result<(), RustQuantError> {
		self.data.get_price_history()?;
		Ok(())
	}

	pub fn fetch_latest_quote(&mut self) -> Result<(), RustQuantError> {
		self.data.get_latest_quote()?;
		Ok(())
	}

	pub fn compute_returns(&mut self, returns_type: ReturnsType) -> Result<(), RustQuantError> {
		self.data.compute_returns(returns_type)?;
		Ok(())
	}

	pub fn get_price_history(&self) -> Option<&DataFrame> { self.data.price_history.as_ref() }

	pub fn get_returns(&self) -> Option<&DataFrame> { self.data.returns.as_ref() }

	pub fn get_latest_quote(&self) -> Option<&DataFrame> { self.data.latest_quote.as_ref() }

	pub fn fetch_options_chain(&mut self) -> Result<(), RustQuantError> {
		self.data.get_options_chain()?;
		Ok(())
	}

	pub fn get_options_chain(&self) -> Option<&DataFrame> { self.data.options_chain.as_ref() }
}
