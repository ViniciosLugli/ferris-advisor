use std::io;

mod service;

#[cfg(feature = "ssr")]
#[actix_web::main]
async fn main() -> io::Result<()> {
	env_logger::init();

	let config = service::Config::new().await?;
	let server = service::Server::new(config);
	server.run().await
}

#[cfg(not(any(feature = "ssr", feature = "csr")))]
pub fn main() {}

#[cfg(all(not(feature = "ssr"), feature = "csr"))]
pub fn main() {
	use ferris_advisor::app::*;

	console_error_panic_hook::set_once();
	leptos::mount_to_body(App);
}
