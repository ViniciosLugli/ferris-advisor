use serde::de::DeserializeOwned;
use serde::Serialize;

#[cfg(not(feature = "ssr"))]
pub fn fetch_api<T>(path: &str) -> impl std::future::Future<Output = Option<T>> + Send + '_
where
	T: Serialize + DeserializeOwned,
{
	use leptos::on_cleanup;
	use send_wrapper::SendWrapper;

	SendWrapper::new(async move {
		let abort_controller = SendWrapper::new(web_sys::AbortController::new().ok());
		let abort_signal = abort_controller.as_ref().map(|a| a.signal());

		on_cleanup(move || {
			if let Some(abort_controller) = abort_controller.take() {
				abort_controller.abort()
			}
		});

		gloo_net::http::Request::get(path)
			.abort_signal(abort_signal.as_ref())
			.send()
			.await
			.map_err(|e| log::error!("{e}"))
			.ok()?
			.json()
			.await
			.ok()
	})
}

#[cfg(feature = "ssr")]
pub async fn fetch_api<T>(path: &str) -> Option<T>
where
	T: Serialize + DeserializeOwned,
{
	reqwest::get(path).await.map_err(|e| log::error!("{e}")).ok()?.json().await.ok()
}
