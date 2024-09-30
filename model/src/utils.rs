use polars::{io::mmap::MmapBytesReader, prelude::*};
use std::error::Error;

pub fn df_from_csv(file_path: &str) -> Result<DataFrame, Box<dyn Error>> {
	let file = std::fs::File::open(file_path)?;
	let file = Box::new(file) as Box<dyn MmapBytesReader>;
	Ok(CsvReader::new(file).finish()?)
}
