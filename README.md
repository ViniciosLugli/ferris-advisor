# Ferris Advisor

A decision support system designed for investing in cryptocurrency assets, the platform leverages data-driven models to predict future price trends, empowering investors to make more informed decisions, the system includes a price history retrieval API, a model training pipeline, and a dashboard for visualizing predictions.

![Main Page](https://github.com/user-attachments/assets/b76fe93f-e2c3-4fad-813a-eabf35b10dfd)
![Graphs Demo](https://github.com/user-attachments/assets/d2cca488-38ad-4354-b4a9-e35b3fb050aa)

## Features

-   **Price Prediction Models**: Implements a time-series prediction model for forecasting future cryptocurrency prices.
-   **API Access**: Provides RESTful API endpoints to retrieve price history and train models.
-   **Data Processing**: Uses Kraken API for fetching OHLC price data and Polars for data manipulation.
-   **Visualization Dashboard**: Interactive dashboard to explore model predictions with real-time data updates.
-   **Model Retraining**: Supports retraining the prediction model with new data.
-   **Logging**: Captures usage logs and stores historical data for future reference.

## Running the Application

Currently, the application is available as a Docker container, which simplifies the deployment process and ensures consistency across different environments, follow the steps below to run the application locally, you can find the [Dockerfile](./Dockerfile.dev) and [docker-compose](./docker-compose-dev.yml) configurations in the repository.

### Requirements

-   [Docker and Docker Compose](https://docs.docker.com/get-docker/)
-   Any unix-like system (Linux, MacOS) is recommended for running the application.
-   A modern web browser to access the dashboard (Brave, Chrome, Firefox, Safari), because the application uses WebAssembly technology.
-   _NOTE_: The application use a lot of disk space, so make sure you have at least 30GB of free space 🦀(for DEV).

1. Clone the repository:

    ```bash
    git clone https://github.com/ViniciosLugli/ferris-advisor.git
    cd ferris-advisor
    ```

2. Run the application:

    ```bash
    docker compose -f docker-compose-dev.yml up
    ```

    This command is going to build the application and start the server, it will be available at `http://localhost:3333`.

## Usage

### API Endpoints

1. **Get Price History**

    Retrieve historical price data for a specified cryptocurrency pair.

    ```bash
    GET /api/get_price_history/{pair}?interval={interval}&since={timestamp}
    ```

2. **Train Model**

    Train the prediction model using historical data.

    ```bash
    GET /api/train_model/{pair}?interval={interval}&since={timestamp}
    ```

3. **Predict Future Prices**

    Make future price predictions based on the trained model.

    ```bash
    GET /api/predict/{pair}?steps={steps}
    ```

4. **Check Model Status**

    Verify if a trained model exists for a specific cryptocurrency pair.

    ```bash
    GET /api/check_model/{pair}
    ```

### Dashboard

-   The interactive dashboard provides a graphical interface to visualize past and predicted cryptocurrency prices, it allows users to select different cryptocurrency pairs and adjust the prediction settings.

## Project Structure

```plaintext
ferris-advisor/
├── model/                      # Model logic and data processing
│   ├── src/ 				    # Model transform, training and prediction logic
│   └── Cargo.toml 			    # Rust project manifest of model workspace
├── public/                     # Public assets (favicon, images)
├── src/                        # Core application (API, components, services)
│   ├── api/  				    # API endpoints and data retrieval actions
│   ├── components/ 			# Reusable UI components
│   ├── pages/ 				    # Main dashboard pages
│   ├── services/ 			    # API service and data processing
│   ├── wrappers/ 			    # Utility wrappers for Server-Side Rendering
├── style/                      # Stylesheets (SCSS)
├── Cargo.toml                  # Rust project manifest of main workspace
├── docker-compose-dev.yml      # Development environment configuration
├── Dockerfile.dev              # Docker image configuration
├── LICENSE                     # License information (GPL-3.0)
├── rust-toolchain.toml 	    # Rust toolchain workspace configuration
├── tailwind.config.js 		    # Tailwind CSS configuration file
└── README.md                   # Project documentation
```

## Development

The project is built entirely with Rust and WebAssembly technologies, using the following libraries:

-   **Main Dependencies**:

    -   **[Leptos](https://github.com/leptos-rs/leptos)**: A framework for building fast, reactive web applications in Rust.
    -   **[Actix-Web](https://actix.rs/)**: A powerful and efficient web framework for Rust, used for serving API endpoints.
    -   **[Polars](https://pola-rs.github.io/polars/)**: A fast DataFrame library for data processing and analysis.
    -   **[Tch](https://github.com/LaurentMazare/tch-rs)**: Rust bindings for PyTorch, enabling machine learning model building and training.
    -   **WebAssembly (WASM)**: A binary instruction format for executing code on the web at near-native speed.

-   **Logging and Error Handling**:

    -   **[log](https://crates.io/crates/log)**: A lightweight logging facade for Rust applications.
    -   **[env_logger](https://crates.io/crates/env_logger)**: An environment-based logger for configurable logging output.
    -   **[console_log](https://crates.io/crates/console_log)**: Logs messages to the browser's console in WebAssembly apps.
    -   **[console_error_panic_hook](https://crates.io/crates/console_error_panic_hook)**: Captures panic messages and logs them to the browser console for debugging.

-   **Serialization and Data Formats**:

    -   **[serde](https://crates.io/crates/serde)**: A framework for serializing and deserializing Rust data structures efficiently.
    -   **[serde_json](https://crates.io/crates/serde_json)**: A JSON parsing and serialization library using Serde.
    -   **[urlencoding](https://crates.io/crates/urlencoding)**: Provides functions for URL encoding and decoding strings.

-   **Web and Networking**:

    -   **[wasm-bindgen](https://crates.io/crates/wasm-bindgen)**: Facilitates high-level interactions between Rust and JavaScript.
    -   **[reqwest](https://crates.io/crates/reqwest)**: An easy-to-use HTTP client for making network requests.
    -   **[gloo-net](https://crates.io/crates/gloo-net)**: A high-level networking library for WebAssembly applications.
    -   **[web-sys](https://crates.io/crates/web-sys)**: Raw bindings to Web APIs for low-level browser interactions.

-   **Date and Time Handling**:

    -   **[chrono](https://crates.io/crates/chrono)**: A comprehensive date and time library for Rust.
    -   **[time](https://crates.io/crates/time)**: An alternative date and time library focusing on correctness and performance.

-   **Data Processing and Machine Learning**:

    -   **[ndarray](https://crates.io/crates/ndarray)**: N-dimensional array manipulation for numerical computations.
    -   **[ndarray-stats](https://crates.io/crates/ndarray-stats)**: Statistical methods for `ndarray` data structures.
    -   **[rand](https://crates.io/crates/rand)**: A library for generating random numbers and distributions.
    -   **[anyhow](https://crates.io/crates/anyhow)**: Simplifies error handling by providing context and backtraces.

-   **Utilities**:

    -   **[leptos-chartistry](https://crates.io/crates/leptos-chartistry)**: A charting library for creating interactive graphs with Leptos.
    -   **[send_wrapper](https://crates.io/crates/send_wrapper)**: Allows non-thread-safe types to be sent between threads safely.

## Finance Data Source: [Kraken Market Service](https://docs.kraken.com/)

The **KrakenMarketService** is responsible for fetching historical cryptocurrency price data from the Kraken API, specifically using the [OHLC (Open-High-Low-Close)](https://en.wikipedia.org/wiki/Open-high-low-close_chart) data for a specified trading pair, this service is the main sourcer of market data, providing the platform with up-to-date and accurate market data for use in price prediction models.

### Components:

1. **Kraken API**:

    - The service connects to the Kraken API endpoint (`https://api.kraken.com/0/public/OHLC`) to retrieve OHLC data, which includes open, high, low, and close prices for each time interval (candlestick).
    - The API accepts parameters like the cryptocurrency pair (e.g., `BTCUSD`), time interval, and an optional timestamp to retrieve data since a specific time.

2. **Price Data**:

    - The API returns candlestick data for the requested pair, including the following fields:
        - **Timestamp**: The time at which the candlestick starts.
        - **Open Price**: The price at the start of the time interval.
        - **High Price**: The highest price during the interval.
        - **Low Price**: The lowest price during the interval.
        - **Close Price**: The price at the end of the interval.
        - **Volume**: The total volume of trades during the interval.
        - **Number of Trades**: The count of individual trades during the interval.

3. **Data Normalization**:

    - The service processes the raw API response by extracting relevant data points (timestamp, prices, volume, and trade counts) and converts them into a [`Polars DataFrame`](https://docs.pola.rs/py-polars/html/reference/dataframe/index.html). This structured format makes it easy to manipulate and analyze the data for further operations like model training and predictions.

4. **Integration with Models**:

    - The processed OHLC data from Kraken is passed to the price prediction models to train and generate predictions. The model can use this historical data to learn market patterns and forecast future prices.

5. **Pair Selection**:
    - The service allows users to specify the cryptocurrency pair they want to retrieve data for, currently supporting popular pairs like `XBTUSD`, `ETHUSD`, and `LTCUSD`, but can be extended to include other pairs available on the Kraken exchange.

## Model and Data Exploration

Is used a machine learning pipeline designed for time-series forecasting, utilizing techniques of normalization, sequence creation, and model training on cryptocurrency price data, this process ensures efficient predictions based on past trends.

The time-series prediction model has be selected based on its ability to capture temporal dependencies in the data and make accurate predictions for future prices, the model is trained on historical price data and evaluated using common metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).

### Data Processing Pipeline

The system includes a `DataProcessor` struct responsible for preparing the data. The key steps include:

1. **Normalization**:
   Data features are normalized to ensure better model performance. This step involves computing the mean and standard deviation for each feature in the dataset and scaling values accordingly.

    ```rust
    pub fn normalize_data(
    	&self,
    	features_array: &mut Array2<f64> // Input data as a 2D array (samples x features)
    ) -> (Array1<f64>, Array1<f64>) {
    	let (rows, cols) = features_array.dim(); // Get the number of rows and columns in the data
    	let mut means = Array1::<f64>::zeros(cols); // Initialize an array to store the means
    	let mut stds = Array1::<f64>::zeros(cols); // Initialize an array to store the standard deviations

    	for col in 0..cols { // Iterate over each column in the data
    		let column = features_array.slice(s![.., col]); // Get the column data
    		let mean = column.mean().unwrap(); // Compute the mean of the column
    		let std = column.std(0.0); // Compute the standard deviation of the column
    		means[col] = mean; // Store the mean in the means array
    		stds[col] = std; // Store the standard deviation in the stds array
    		for row in 0..rows { // Iterate over each row in the column
    			// Normalize the data using the mean and standard deviation
    			features_array[[row, col]] = (features_array[[row, col]] - mean) / std;
    		}
    	}
    	(means, stds)
    }
    ```

2. **Sequence Creation**:
   The data is segmented into overlapping windows, each representing a historical sequence that will be used to train the model. The target value for prediction is the price after the sequence window.

    ```rust
    pub fn create_sequences(
    	&self,
    	data: &Array2<f64> // Input data as a 2D array (samples x features)
    ) -> (Vec<Array2<f64>>, Vec<f64>) {
    	let mut X = Vec::new(); // Initialize a vector to store the input sequences
    	let mut y = Vec::new(); // Initialize a vector to store the target values

    	let num_samples = data.shape()[0]; // Get the number of samples in the data

    	// Iterate over the data to create sequences
    	for i in 0..(num_samples - self.window_size - self.prediction_steps + 1) {
    		// Get the input sequence
    		let sequence = data.slice(s![i..(i + self.window_size), ..]).to_owned();
    		// Get the target value
    		let target = data[[i + self.window_size + self.prediction_steps - 1, 0]];
    		X.push(sequence); // Add the input sequence to the X vector
    		y.push(target); // Add the target value to the y vector
    	}
    	(X, y)
    }
    ```

3. **Tensor Conversion**:
   The processed sequences are converted into `Tensor` objects, making them compatible with the machine learning model for training and predictions.

    ```rust
    pub fn sequences_to_tensor(
    	&self,
    	sequences: &[Array2<f64>] // Input sequences as a slice of 2D arrays (samples x features)
    ) -> Result<Tensor, Box<dyn Error>> {
    	let num_samples = sequences.len(); // Get the number of samples
    	let window_size = sequences[0].shape()[0]; // Get the window size
    	let num_features = sequences[0].shape()[1]; // Get the number of features

    	// Create a vector to store the data
    	let mut data_vec = Vec::with_capacity(num_samples * window_size * num_features);

    	for seq in sequences { // Iterate over the sequences
    		data_vec.extend(seq.iter().cloned()); // Add the sequence data to the vector
    	}

    	// Create a Tensor from the data vector and reshape it
    	let tensor = Tensor::from_slice(&data_vec)
    		.reshape(&[num_samples as i64, window_size as i64, num_features as i64])
    		.to_kind(tch::Kind::Float);

    	Ok(tensor)
    }
    ```

### Time-Series Prediction Model

The core of the prediction engine is a `TimeSeriesModel`, which utilizes a Gated Recurrent Unit (GRU) neural network for time-series forecasting, the model is configured with several key components:

1. **GRU Layer**:
   The GRU is a type of Recurrent Neural Network (RNN) suited for sequential data, it processes the input sequences and retains temporal dependencies between data points.

    ```rust
    // Define the GRU layer with the specified input and hidden sizes (input_size to hidden_size)
    let gru = nn::gru(&vs.root() / "gru", input_size, hidden_size, gru_config);
    ```

2. **Fully Connected Layer**:
   The final output from the GRU is passed through a fully connected (FC) layer to make predictions for future prices.

    ```rust
    // Define the fully connected layer with the specified input and output sizes (hidden_size * 2 to output_size)
    let fc = nn::linear(&vs.root() / "fc", hidden_size * 2, output_size, Default::default());
    ```

3. **Training Process**:
   The model is trained by minimizing the Mean Squared Error (MSE) between the predicted and actual prices, updating weights using the Adam optimizer, the data is processed in batches to optimize training.

    ```rust
    pub fn train(
    	&mut self,
    	x_train: &Tensor,  // Input training data as tensor
    	y_train: &Tensor,  // Target values for training
    	num_epochs: i64,   // Number of training epochs
    	learning_rate: f64, // Learning rate for optimizer
    	batch_size: i64,    // Size of each batch for training
    ) -> Result<(), Box<dyn Error>> {
    	// Initialize Adam optimizer with the specified learning rate
    	let mut opt = nn::Adam::default().build(&self.vs, learning_rate)?;

    	// Get the number of training samples
    	let train_size = x_train.size()[0];

    	// Calculate the number of batches based on the batch size
    	let num_batches = (train_size + batch_size - 1) / batch_size;

    	// Log message indicating the start of training
    	info!("Starting training...");

    	// Loop over the specified number of epochs
    	for epoch in 1..=num_epochs {
    		let mut total_loss = 0.0;  // Initialize total loss for the current epoch

    		// Loop over batches
    		for batch_idx in 0..num_batches {
    			// Calculate the start and end index for the current batch
    			let start = batch_idx * batch_size;
    			let end = std::cmp::min(start + batch_size, train_size);

    			// Narrow the training tensors to get the current batch
    			let batch_x = x_train.narrow(0, start, end - start); // Get batch input data
    			let batch_y = y_train.narrow(0, start, end - start); // Get corresponding target values

    			// Forward pass: compute the output of the model
    			let output = self.forward(&batch_x, true);

    			// Compute the mean squared error (MSE) loss
    			let loss = output.mse_loss(&batch_y, tch::Reduction::Mean);

    			// Zero gradients to prevent accumulation from previous steps
    			opt.zero_grad();

    			// Backward pass: compute gradients for the model
    			loss.backward();

    			// Perform one optimization step (update the model parameters)
    			opt.step();

    			// Accumulate the loss for this batch
    			total_loss += loss.double_value(&[]);
    		}

    		// Calculate the average loss for this epoch
    		let avg_loss = total_loss / num_batches as f64;

    		// Log the training loss for the current epoch
    		info!("Epoch {}: Training Loss: {:.6}", epoch, avg_loss);
    	}

    	// Return Ok() if the training completes without errors
    	Ok(())
    }
    ```

4. **Evaluation**:
   After training, the model's performance is assessed using common evaluation metrics such as RMSE, MAE, and R-squared, ensuring accurate predictions for unseen data.

    ```rust
    pub fn evaluate(
    	&self,
    	x_test: &Tensor, // Input test data as tensor
    	y_test: &Tensor // Target values for evaluation
    ) -> Result<EvaluationMetrics, Box<dyn Error>> {
    	// Perform the forward pass on the test data without calculating gradients (no_grad)
    	let test_output = tch::no_grad(|| self.forward(x_test, false));

    	// Compute Mean Squared Error (MSE) between the predictions and actual values
    	let mse = test_output.mse_loss(y_test, tch::Reduction::Mean).double_value(&[]);

    	// Compute Root Mean Squared Error (RMSE), which is the square root of the MSE
    	let rmse = mse.sqrt();

    	// Compute Mean Absolute Error (MAE), which is the average of the absolute differences between predictions and actual values
    	let mae = test_output.f_sub(y_test)?.abs().mean(Kind::Float).double_value(&[]);

    	// Compute Sum of Squares of Residuals (SS_res), the total squared differences between predictions and actual values
    	let ss_res = test_output.f_sub(y_test)?.f_pow_tensor_scalar(2.0)?.sum(Kind::Float).double_value(&[];

    	// Compute the mean of the actual values (y_test)
    	let y_mean = y_test.mean(Kind::Float);

    	// Compute Total Sum of Squares (SS_tot), the total squared differences between the actual values and their mean
    	let ss_tot = y_test.f_sub(&y_mean)?.f_pow_tensor_scalar(2.0)?.sum(Kind::Float).double_value(&[];

    	// Compute R-squared (R²), a measure of how well the predictions match the actual data
    	// R² = 1 - (SS_res / SS_tot)
    	let r2 = 1.0 - ss_res / ss_tot;

    	// Return the evaluation metrics (MSE, RMSE, MAE, R²) encapsulated in the EvaluationMetrics struct
    	Ok(EvaluationMetrics { mse, rmse, mae, r2 })
    }

    ```

### Data Exploration and Predictions

Once trained, the model can predict future cryptocurrency prices by processing real-time data, the process involves:

1. **Real-Time Normalization**:
   The incoming data is normalized based on the previously computed mean and standard deviation.

2. **Prediction**:
   The normalized data is fed into the GRU model, and future prices are predicted based on the historical input sequence.

    ```rust
    pub fn predict_future(
    	&self,
    	df: &DataFrame, // Input DataFrame containing historical data
    	steps: usize // Number of prediction steps to make (if historical has 1h intervals, steps=1 predicts 1h foward)
    ) -> Result<Vec<f64>, Box<dyn Error>> {
    	// Select the relevant columns ('close', 'volume', 'trades') from the DataFrame
    	let df_sampled = df.select(["close", "volume", "trades"])?;

    	// Convert the DataFrame into an ndarray for processing
    	let features_data = df_sampled.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;

    	// Convert the ndarray data into a 2D array for further manipulation
    	let mut features_array = Array2::<f64>::from_shape_vec(
    		(features_data.shape()[0], features_data.shape()[1]),
    		features_data.iter().cloned().collect(),
    	)?;

    	// Check if the model has stored means and standard deviations for normalization
    	if let (Some(means), Some(stds)) = (&self.means, &self.stds) {
    		info!("Normalizing data with stored means and stds...");

    		// Normalize each feature (column) in the dataset using the stored mean and standard deviation
    		for col in 0..features_array.shape()[1] {
    			for row in 0..features_array.shape()[0] {
    				features_array[[row, col]] = (features_array[[row, col]] - means[col]) / stds[col];
    			}
    		}
    	} else {
    		// If the model has not been trained or normalization parameters are missing, return an error
    		return Err("Model not trained or normalization parameters missing".into());
    	}

    	// Prepare the initial input sequence by selecting the last 'window_size' rows
    	let start = features_array.shape()[0] - self.window_size;
    	let mut inputs = features_array.slice(s![start.., ..]).to_owned();

    	// Initialize a vector to store predicted values
    	let mut predictions = Vec::new();

    	// Extract the mean and standard deviation for the 'close' price (first column)
    	let mean_close = self.means.as_ref().unwrap()[0];
    	let std_close = self.stds.as_ref().unwrap()[0];

    	// Iterate for the specified number of prediction steps
    	for _ in 0..steps {
    		// Check that the input tensor has the correct size
    		let num_elements = inputs.len();
    		let reshaped_size = self.window_size * self.input_size as usize;
    		if num_elements != reshaped_size as usize {
    			return Err(format!(
    				"Invalid shape for input tensor: expected {} elements but got {}",
    				reshaped_size, num_elements
    			)
    			.into());
    		}

    		// Convert the input sequence into a Tensor suitable for the model
    		let x_input = Tensor::from_slice(inputs.as_slice().unwrap())
    			.reshape(&[1, self.window_size as i64, self.input_size])
    			.to_kind(Kind::Float)
    			.to_device(self.device);

    		// Make a prediction using the model
    		let pred = self.predict(&x_input);
    		let pred_value = pred.double_value(&[0, 0]);

    		// Denormalize the predicted value (convert it back to the original scale)
    		let pred_value_denorm = pred_value * std_close + mean_close;

    		// Add the denormalized prediction to the results
    		predictions.push(pred_value_denorm);

    		// Update the input sequence with the new prediction
    		let last_row = inputs.row(inputs.nrows() - 1).to_owned();
    		let mut new_row = Array1::<f64>::zeros(self.input_size as usize);
    		new_row[0] = (pred_value_denorm - mean_close) / std_close; // Normalize the predicted value

    		// Shift the rest of the input sequence by one step
    		for i in 1..self.input_size as usize {
    			new_row[i] = last_row[i];
    		}

    		// Append the new row to the input sequence
    		inputs = Array::from_shape_vec(
    			(inputs.nrows() + 1, self.input_size as usize),
    			inputs.iter().cloned().chain(new_row.iter().cloned()).collect(),
    		)?;

    		// Ensure that the input sequence does not exceed the window size
    		if inputs.nrows() > self.window_size {
    			inputs = inputs.slice(s![-(self.window_size as isize).., ..]).to_owned();
    		}
    	}

    	// Return the vector of predicted values
    	Ok(predictions)
    }
    ```

Finally, the predictions are returned to the user, who can visualize them on the dashboard and make informed decisions based on the model's insights.

## Final Thoughts

:warning::warning:DO NOT USE RUST FOR FRONTEND WITH SSR AND LEPTOS!!!:warning::warning:

It was a painful and problematic experience, especially since I caught the wave of the 0.7 update of leptos... Many of the examples are outdated and disorganized, causing issues and confusion. It's still in a very early stage and has very little support for new technologies (which is quite understandable), so creating something professional and large-scale is very problematic. It works very well for small applications, but if you depend on a sustainable and scalable service, forget it NOW 😠😠.

This project was a great learning experience, i learned a lot about Rust with WebAssembly, and the challenges of building a full-stack application with these technologies... ❤️🦀

About the model implementation, its incredible how easy is to implement a machine learning model in Rust, the language is very powerful and has a lot of libraries to help in the process, the only downside is the lack of a large community and the difficulty of finding help when you need it. YES, the performance is BLAZING FAST⚡⚡, but the learning curve is very steep, so be prepared for a lot of headaches on beginning. 🧠🧠

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](./LICENSE) file for details.
