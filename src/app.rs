use crate::pages::{Home, NotFound};
use leptos::*;
use leptos_meta::*;
use leptos_router::*;

#[component]
pub fn App() -> impl IntoView {
	provide_meta_context();

	view! {
		<Stylesheet id="leptos" href="/pkg/ferris-advisor.css" />
		<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.2/dist/echarts.min.js" async></script>
		<script src="https://cdn.jsdelivr.net/npm/echarts-gl@2.0.9/dist/echarts-gl.min.js" async></script>
		<Title text="Ferris Advisor" />

		<Router>
			<main>
				<Routes>
					<Route path="/" view=Home />
					<Route path="/*any" view=NotFound />
				</Routes>
			</main>
		</Router>
	}
}
