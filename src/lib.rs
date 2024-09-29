pub mod app;
mod components;
mod pages;

#[cfg(feature = "hydrate")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn hydrate() {
	console_error_panic_hook::set_once();
	leptos::leptos_dom::HydrationCtx::stop_hydrating();
}
