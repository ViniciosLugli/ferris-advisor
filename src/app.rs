use crate::pages::{Home, NotFound};
use leptos::*;
use leptos_meta::*;
use leptos_router::*;

#[component]
pub fn App() -> impl IntoView {
	provide_meta_context();

	view! {
		<!DOCTYPE html>
		<html lang="en">
			<head>
				<meta charset="utf-8" />
				<meta name="viewport" content="width=device-width, initial-scale=1" />

				<Stylesheet id="leptos" href="/pkg/ferris-advisor.css" />
				<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.2/dist/echarts.min.js"></script>
				<script src="https://cdn.jsdelivr.net/npm/echarts-gl@2.0.9/dist/echarts-gl.min.js"></script>
				<Title text="Ferris Advisor" />
			</head>
			<Router>
				<main>
					<Routes>
						<Route path="" view=Home />
						<Route path="/*" view=NotFound />
					</Routes>
				</main>
			</Router>
		</html>
	}
}
