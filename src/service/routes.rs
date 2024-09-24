use actix_files::NamedFile;
#[cfg(feature = "ssr")]
use actix_web::{get, web, Result};
use leptos::*;

pub use ferris_advisor::app::App;

#[cfg(feature = "ssr")]
#[get("favicon.ico")]
pub async fn favicon(leptos_options: web::Data<LeptosOptions>) -> Result<NamedFile> {
	let leptos_options = leptos_options.into_inner();
	let site_root = &leptos_options.site_root;
	Ok(NamedFile::open(format!("{site_root}/favicon.ico"))?)
}
