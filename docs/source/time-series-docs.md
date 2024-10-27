# Time Series Anomaly Detection Documentation

## Overview

This package provides robust time series anomaly detection capabilities using enhanced Isolation Forest algorithms. It offers two main implementations:

1. `TimeSeriesIsolationForest`: A basic implementation with essential features
2. `AdvancedTimeSeriesIsolationForest`: An enhanced version with additional capabilities for streaming data and visualization

Both implementations are specifically designed to handle the unique challenges of time series data, including:
- Temporal dependencies
- Seasonality
- Missing values
- Both univariate and multivariate time series

## Key Features

- **Temporal Feature Extraction**: Automatically creates relevant features from time windows
- **Seasonal Decomposition**: Handles seasonal patterns in your data
- **Missing Value Handling**: Multiple imputation strategies available
- **Multivariate Support**: Can process multiple related time series simultaneously
- **Streaming Capability**: Process data in real-time chunks
- **Confidence Scores**: Quantify the certainty of anomaly predictions
- **Visualization Tools**: Built-in plotting capabilities for anomaly visualization

## Installation

```bash
pip install anomalous-ts
```

## Basic Usage

### Simple Anomaly Detection

```python
from anomalous_ts import TimeSeriesIsolationForest

# Initialize detector
detector = TimeSeriesIsolationForest(
    window_size=5,
    contamination='auto'
)

# Detect anomalies
anomalies = detector.detect(your_time_series)
```

### With Seasonal Components

```python
detector = TimeSeriesIsolationForest(
    window_size=24,
    seasonal_period=168,  # Weekly seasonality (24*7 hours)
    contamination=0.01
)
```

### Multivariate Analysis

```python
# For multiple related time series
detector = TimeSeriesIsolationForest(
    window_size=12,
    seasonal_period=24,
    imputation_method='linear'
)

# Input as DataFrame with multiple columns
anomalies = detector.detect(multivariate_df)
```

## Advanced Usage

### Real-time Streaming Analysis

```python
from anomalous_ts import AdvancedTimeSeriesIsolationForest

detector = AdvancedTimeSeriesIsolationForest(
    window_size=10,
    n_estimators=100
)

# Process streaming data
for chunk_data, anomalies, confidence in detector.process_stream(
    data_stream,
    chunk_size=100
):
    # Handle results
    print(f"Found {anomalies.sum()} anomalies in chunk")
```

### With Visualization

```python
detector = AdvancedTimeSeriesIsolationForest()
anomalies, confidence_scores = detector.detect_and_visualize(
    data,
    title='Anomaly Detection Results'
)
```

## Class Reference

### TimeSeriesIsolationForest

#### Parameters

- `window_size` (int, default=5): Size of sliding window for temporal feature extraction
- `n_estimators` (int, default=100): Number of base estimators in ensemble
- `contamination` (float or 'auto', default='auto'): Expected proportion of outliers
- `max_features` (float, default=1.0): Features to consider per tree
- `bootstrap` (bool, default=False): Whether to use bootstrap samples
- `n_jobs` (int, default=-1): Number of parallel jobs
- `standardize` (bool, default=True): Whether to standardize features
- `seasonal_period` (int, optional): Period for seasonal feature extraction
- `imputation_method` (str, default='linear'): Method for handling missing values

#### Methods

- `fit(data)`: Train the model
- `predict(data)`: Predict anomalies
- `detect(data)`: Combined fit and predict

### AdvancedTimeSeriesIsolationForest

Inherits all parameters from TimeSeriesIsolationForest and adds:

#### Additional Parameters

- `preprocessor` (TimeSeriesPreprocessor, optional): Custom preprocessor
- `visualizer` (TimeSeriesVisualizer, optional): Custom visualizer

#### Additional Methods

- `fit_predict(data, return_confidence=False)`: Returns predictions and confidence scores
- `detect_and_visualize(data, title='')`: Detects and plots anomalies
- `process_stream(data_stream, chunk_size=100)`: Processes streaming data

## Feature Details

### Temporal Features

The algorithm automatically extracts the following features from each time window:

1. Statistical Features:
   - Mean
   - Standard deviation
   - Minimum/Maximum
   - Median
   - Range (peak-to-peak)
   - Skewness
   - Kurtosis
   - Quartiles

2. Temporal Features:
   - Current value
   - Change from previous
   - Number of increases
   - Trend indicators

3. Seasonal Features (if period specified):
   - Seasonal difference
   - Seasonal mean
   - Seasonal standard deviation

### Imputation Methods

Available methods for handling missing values:

- `'linear'`: Linear interpolation
- `'forward'`: Forward fill
- `'backward'`: Backward fill
- `'mean'`: Mean imputation
- `'median'`: Median imputation

## Best Practices

1. **Window Size Selection**:
   - Should be large enough to capture relevant patterns
   - Usually 1-2 times the length of expected anomalies
   - For seasonal data, consider using the seasonal period

2. **Contamination Parameter**:
   - Use 'auto' if unsure about anomaly proportion
   - Set to expected anomaly rate if known (e.g., 0.01 for 1%)

3. **Seasonal Period**:
   - Set to known period in your data (e.g., 24 for daily patterns in hourly data)
   - Can be omitted if no seasonality exists

4. **Performance Optimization**:
   - Use `n_jobs=-1` for parallel processing
   - Adjust `n_estimators` based on dataset size
   - Consider chunk_size in streaming for memory management

## Example Use Cases

### Quality Control
```python
detector = TimeSeriesIsolationForest(
    window_size=60,  # 1 hour of minute data
    contamination=0.001  # Expecting 0.1% anomalies
)
anomalies = detector.detect(sensor_readings)
```

### Financial Monitoring
```python
detector = AdvancedTimeSeriesIsolationForest(
    window_size=20,
    seasonal_period=5,  # Weekly patterns (5 trading days)
    contamination=0.005
)
anomalies, confidence = detector.fit_predict(
    stock_prices,
    return_confidence=True
)
```

### Server Metrics
```python
detector = AdvancedTimeSeriesIsolationForest(
    window_size=5,
    n_estimators=50
)

for chunk in detector.process_stream(metrics_stream, chunk_size=100):
    # Alert if anomalies detected
    if chunk[1].any():
        send_alert(chunk)
```

## Performance Considerations

1. **Memory Usage**:
   - Scales with window_size × n_features × n_samples
   - Use streaming for large datasets
   - Consider reducing n_estimators for faster processing

2. **Processing Time**:
   - Affected by number of features and estimators
   - Parallel processing helps with large datasets
   - Feature extraction is the main bottleneck

3. **Accuracy vs Speed**:
   - Larger window_size generally improves accuracy but increases computation
   - More estimators improve stability but increase processing time
   - Balance based on your specific needs

## Error Handling

The implementation includes robust error handling for:
- Invalid parameters
- Missing data
- Incorrect data types
- Insufficient data length
- Memory constraints

## Contributing

Contributions are welcome! Please see our contributing guidelines for details on:
- Code style
- Testing requirements
- Documentation standards
- Pull request process

## License

This project is licensed under the MIT License - see the LICENSE file for details.
