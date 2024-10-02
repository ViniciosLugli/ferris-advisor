use crate::components::{fetch_api, EvaluationMetrics, ModelInfo, PairSelector, PriceGraph};
use leptos::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckModelReturn {
	pub model_exists: bool,
	pub metrics: Option<EvaluationMetrics>,
}

#[server(CheckModel, "/api")]
pub async fn check_model(pair: String) -> Result<CheckModelReturn, ServerFnError> {
	println!("Checking model for pair: {}", pair);
	let url = format!("http://0.0.0.0:3333/api/check_model/{}", pair);
	let response = fetch_api::<CheckModelReturn>(&url).await;

	match response {
		Some(data) => Ok(data),
		None => Err(ServerFnError::new("Failed to check model")),
	}
}

#[server(TrainModel, "/api")]
pub async fn train_model(pair: String) -> Result<EvaluationMetrics, ServerFnError> {
	println!("Training model for pair: {}", pair);
	let url = format!("http://0.0.0.0:3333/api/train_model/{}", pair);
	let response = fetch_api::<EvaluationMetrics>(&url).await;

	match response {
		Some(data) => Ok(data),
		None => Err(ServerFnError::new("Failed to train model")),
	}
}

#[component]
pub fn Home() -> impl IntoView {
	let (selected_pair, set_selected_pair) = create_signal("XBTUSD".to_string());
	let model_check =
		Resource::new(move || selected_pair.get(), |pair| async move { check_model(pair.clone()).await.ok() });
	let training_result =
		Resource::new(move || selected_pair.get(), |pair| async move { train_model(pair.clone()).await.ok() });

	view! {
		<main class="px-4 my-0 mx-auto max-w-7xl sm:px-6 lg:px-8">
			<header class="py-10 text-center">
				<h1 class="text-5xl font-bold text-gray-900">"Ferris Advisor"</h1>
				<p class="mt-4 text-xl text-gray-600">"Decision Support System for Investing in Crypto Assets"</p>
				<img class="mx-auto mt-6" src="/public/ferris.png" alt="Ferris the crab" style="width: 200px;" />
			</header>

			<div class="mt-10">
				<PairSelector selected_pair=selected_pair.clone() set_selected_pair=set_selected_pair.clone() />

				<Suspense fallback=|| {
					"Loading..."
				}>
					{move || {
						model_check
							.with(|data| match data {
								Some(Some(CheckModelReturn { model_exists, metrics })) => {
									if *model_exists {
										if let Some(metrics) = metrics {
											view! {
												<ModelInfo metrics=Some(metrics.clone()) pair=selected_pair.clone() />
											}
										} else {
											view! { <ModelInfo metrics=None pair=selected_pair.clone() /> }
										}
									} else {
										view! { <ModelInfo metrics=None pair=selected_pair.clone() /> }
									}
								}
								_ => view! { <ModelInfo metrics=None pair=selected_pair.clone() /> },
							})
					}}
				</Suspense>

				<div class="mt-6"></div>
				<div class="flex justify-center mt-6 space-x-4">
					<button class="py-3 px-5 text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50">
						on:click=move |_| training_result.refetch() "Train Model"
					</button>

					<Suspense fallback=move || {
						view! { <p>"Training model..."</p> }
					}>
						{training_result
							.with(|result| match result {
								Some(metrics) => view! { <p>"Model trained with metrics: {metrics:?}"</p> },
								None => view! { <p>"Training failed or not started."</p> },
							})}
					</Suspense>

					<button class="py-3 px-5 text-white bg-green-600 rounded-lg hover:bg-green-700 disabled:opacity-50">
						"Predict Prices"
					</button>
				</div>

				<PriceGraph pair=selected_pair.clone() />
			</div>
		</main>
	}
}
