use charming::{component::Axis, df, series::Candlestick, Chart, WasmRenderer};
use leptos::*;

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
