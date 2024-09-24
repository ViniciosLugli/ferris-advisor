mod config;
mod routes;
mod server;

pub use config::Config;
pub use routes::{favicon, App as AppRoutes};
pub use server::Server;
