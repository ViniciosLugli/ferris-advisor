use leptos::*;

#[component]
pub fn PairSelector(selected_pair: ReadSignal<String>, set_selected_pair: WriteSignal<String>) -> impl IntoView {
	let pairs = vec!["XBTUSD", "ETHUSD", "LTCUSD"];

	view! {
		<div class="mb-6">
			<label for="pair" class="block text-lg font-medium text-gray-700">
				"Select Cryptocurrency Pair"
			</label>
			<select
				id="pair"
				name="pair"
				class="block py-2 pr-10 pl-3 mt-2 w-full text-base bg-white rounded-md border border-gray-300 sm:text-sm focus:border-blue-500 focus:ring-blue-500 focus:outline-none"
			>
				on:change=move |ev| set_selected_pair(event_target_value(&ev))
				{pairs
					.into_iter()
					.map(|pair| {
						let is_selected = move || selected_pair.get() == pair;
						view! {
							<option value=pair selected=is_selected>
								{pair}
							</option>
						}
					})
					.collect::<Vec<_>>()}
			</select>
		</div>
	}
}
