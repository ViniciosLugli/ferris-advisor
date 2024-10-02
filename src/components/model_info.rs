use leptos::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EvaluationMetrics {
	pub mse: f64,
	pub rmse: f64,
	pub mae: f64,
	pub r2: f64,
}

#[component]
pub fn ModelInfo(pair: ReadSignal<String>, metrics: Option<EvaluationMetrics>) -> impl IntoView {
	view! {
		<div class="my-6">
			<h2 class="text-2xl font-semibold text-gray-800">"Model Evaluation Metrics"</h2>
			{match metrics {
				Some(m) => {
					view! {
						<ul class="mt-4 text-left text-gray-700">
							<li>{format!("Mean Squared Error (MSE): {:.6}", m.mse)}</li>
							<li>{format!("Root Mean Squared Error (RMSE): {:.6}", m.rmse)}</li>
							<li>{format!("Mean Absolute Error (MAE): {:.6}", m.mae)}</li>
							<li>{format!("R² Score: {:.6}", m.r2)}</li>
						</ul>
					}
				}
				None => {
					view! {
						<ul class="mt-4 text-left text-gray-700">
							<li>{format!("Please train a model to view evaluation metrics.")}</li>
							<li>{format!("All the retraining and prediction is done on the server side.")}</li>
							<li>
								{format!("The retraining is made with the last 60 days of data, with a 1h interval.")}
							</li>
							<li>{format!("")}</li>
						</ul>
					}
				}
			}}
		</div>
	}
}
