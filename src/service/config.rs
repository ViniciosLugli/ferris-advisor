use leptos::{get_configuration, LeptosOptions};
use std::{io, net::SocketAddr};

#[derive(Debug, Clone)]
pub struct Config {
	pub leptos_options: LeptosOptions,
	pub addr: SocketAddr,
}

impl Config {
	pub async fn new() -> io::Result<Self> {
		let conf: leptos::leptos_config::ConfFile = get_configuration(None).await.unwrap();
		Ok(Self {
			leptos_options: conf.leptos_options.clone(),
			addr: conf.leptos_options.site_addr,
		})
	}
}
