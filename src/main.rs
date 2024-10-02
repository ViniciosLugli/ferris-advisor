mod api;
mod wrappers;
use wrappers::{Config, Server};

#[cfg(feature = "ssr")]
pub mod services;

#[cfg(feature = "ssr")]
#[actix_web::main]
async fn main() -> std::io::Result<()> {
	use env_logger::Env;

	env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

	let config = Config::new().await?;
	let server = Server::new(config);
	server.run().await
}

#[cfg(not(feature = "ssr"))]
pub fn main() {
	log::info!("SSR feature is disabled");
}
