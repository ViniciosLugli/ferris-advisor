use charming::{component::Axis, df, series::Candlestick, Chart, WasmRenderer};

use leptos::*;
use leptos_meta::*;
use leptos_router::*;

#[component]
pub fn App() -> impl IntoView {
	// Provides context that manages stylesheets, titles, meta tags, etc.
	provide_meta_context();

	view! {
		<Stylesheet id="leptos" href="/pkg/ferris-advisor.css" />
		<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.2/dist/echarts.min.js"></script>
		<script src="https://cdn.jsdelivr.net/npm/echarts-gl@2.0.9/dist/echarts-gl.min.js"></script>
		// sets the document title
		<Title text="Welcome to Leptos" />

		// content for this welcome page
		<Router>
			<main>
				<Routes>
					<Route path="/" view=HomePage ssr=SsrMode::Async />
					<Route path="/*any" view=NotFound />
				</Routes>
			</main>
		</Router>
	}
}

#[component]
pub fn GraphExample() -> impl IntoView {
	let action = create_local_resource(
		|| (),
		|_| async move {
			let chart = Chart::new()
				.x_axis(Axis::new().data(vec!["2017-10-24", "2017-10-25", "2017-10-26", "2017-10-27"]))
				.y_axis(Axis::new())
				.series(Candlestick::new().data(df![
					[20, 34, 10, 38],
					[40, 35, 30, 50],
					[31, 38, 33, 44],
					[38, 15, 5, 42]
				]));
			let renderer = WasmRenderer::new(600, 400);
			renderer.render("chart", &chart).unwrap();
		},
	);

	view! {
		<div>
			<div id="chart"></div>
		</div>
	}
}

/// Renders the home page of your application.
#[component]
fn HomePage() -> impl IntoView {
	let (count, set_count) = create_signal(0);

	view! {
		<main class="my-0 mx-auto max-w-3xl text-center">
			<h2 class="p-6 text-4xl">"Welcome to Leptos with Tailwind"</h2>
			<p class="px-10 pb-10 text-left">
				"Tailwind will scan your Rust files for Tailwind class names and compile them into a CSS file."
			</p>
			<button
				class="py-3 px-5 text-white bg-amber-600 rounded-lg hover:bg-sky-700"
				on:click=move |_| set_count.update(|count| *count += 1)
			>
				{move || {
					if count.get() == 0 { "Click me!".to_string() } else { count.get().to_string() }
				}}
			</button>

			<div class="mt-10">
				<GraphExample />
			</div>
		</main>
	}
}

/// 404 - Not Found
#[component]
fn NotFound() -> impl IntoView {
	// set an HTTP status code 404
	// this is feature gated because it can only be done during
	// initial server-side rendering
	// if you navigate to the 404 page subsequently, the status
	// code will not be set because there is not a new HTTP request
	// to the server
	#[cfg(feature = "ssr")]
	{
		// this can be done inline because it's synchronous
		// if it were async, we'd use a server function
		let resp = expect_context::<leptos_actix::ResponseOptions>();
		resp.set_status(actix_web::http::StatusCode::NOT_FOUND);
	}

	view! { <h1>"Not Found"</h1> }
}
