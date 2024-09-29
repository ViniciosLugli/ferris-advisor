use leptos::*;

#[component]
pub fn NotFound() -> impl IntoView {
	#[cfg(feature = "ssr")]
	{
		use actix_web::http::StatusCode;
		use leptos_actix::ResponseOptions;

		let resp = expect_context::<ResponseOptions>();
		resp.set_status(StatusCode::NOT_FOUND);
	}

	view! { <h1>"Not Found"</h1> }
}
