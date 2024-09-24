use charming::{component::Axis, df, series::Candlestick, Chart, WasmRenderer};

use leptos::*;
use leptos_meta::*;
use leptos_router::*;
use server_fn::codec::{Cbor, Json};

#[component]
pub fn App() -> impl IntoView {
	provide_meta_context();

	view! {
		<Stylesheet id="leptos" href="/pkg/ferris-advisor.css" />
		<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.2/dist/echarts.min.js"></script>
		<script src="https://cdn.jsdelivr.net/npm/echarts-gl@2.0.9/dist/echarts-gl.min.js"></script>
		<Title text="Ferris Advisor" />

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
	let _ = create_local_resource(
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

#[server(Example, "/api")]
pub async fn example_fn(input: String) -> Result<(), ServerFnError> {
	println!("Received input: {}", input);
	Ok(())
}

#[island]
fn HomePage() -> impl IntoView {
	let (count, set_count) = create_signal(0);
	let example_fn = create_server_action::<Example>();

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
				{move || { if count.get() == 0 { "Click me!".to_string() } else { count.get().to_string() } }}
			</button>

			<ActionForm action=example_fn>
				<label>"Add a Todo" <input type="text" name="title" /></label>
				<input type="submit" value="Add" />
			</ActionForm>

			<div class="mt-10">
				<GraphExample />
			</div>
		</main>
	}
}

#[component]
fn NotFound() -> impl IntoView {
	#[cfg(feature = "ssr")]
	{
		let resp = expect_context::<leptos_actix::ResponseOptions>();
		resp.set_status(actix_web::http::StatusCode::NOT_FOUND);
	}

	view! { <h1>"Not Found"</h1> }
}
