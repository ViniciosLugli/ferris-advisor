use super::Config;
use crate::api::{
	assets::{favicon, ferris},
	model::{check_model, get_price_history, predict, train_model},
};
use actix_files::Files;
use actix_web::{middleware, web, App, HttpServer};
use ferris_advisor::app::App as AppRoutes;
use leptos_actix::{generate_route_list, LeptosRoutes};
use log::info;
use std::sync::Arc;

pub struct Server {
	config: Arc<Config>,
}

impl Server {
	pub fn new(config: Config) -> Self {
		Self { config: Arc::new(config) }
	}

	pub async fn run(self) -> std::io::Result<()> {
		info!("Starting server...");

		let routes = generate_route_list(AppRoutes);
		self.print_routes(&routes);

		let config = self.config.clone();
		info!("Current configuration:");
		info!("{:?}", config);

		HttpServer::new(move || {
			let leptos_options = &config.leptos_options;
			let site_root = &leptos_options.site_root;

			App::new()
				.service(Files::new("/pkg", format!("{site_root}/pkg")))
				.service(Files::new("/public", site_root))
				.service(favicon)
				.service(ferris)
				.service(get_price_history)
				.service(train_model)
				.service(check_model)
				.service(predict)
				.leptos_routes(leptos_options.to_owned(), routes.to_owned(), AppRoutes)
				.app_data(web::Data::new(leptos_options.to_owned()))
				.wrap(middleware::Compress::default())
		})
		.bind(&self.config.addr)?
		.run()
		.await
	}

	fn print_routes(&self, routes: &[leptos_router::RouteListing]) {
		info!("Current avaliable routes:");
		for route in routes {
			info!("-> {}", route.path());
		}
		info!("listening on http://{}", &self.config.addr);
	}
}
