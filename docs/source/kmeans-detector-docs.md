# TimeSeriesKMeansDetector Documentation

## Overview
The TimeSeriesKMeansDetector is a Python class that implements anomaly detection for time series data using K-means clustering. It identifies anomalies by finding data points that deviate significantly from their cluster centers.

## Features
- Supports both univariate and multivariate time series data
- Handles missing values through configurable preprocessing
- Provides seasonal feature extraction
- Supports streaming data processing
- Includes visualization capabilities
- Generates cluster profiles for analysis

## Installation

### Requirements
```python
numpy
pandas
scikit-learn
scipy
```

Additional requirements for full functionality:
- `anomolous_ts.preprocessor.TimeSeriesPreprocessor`
- `anomolous_ts.visualizer.TimeSeriesVisualizer`

## Quick Start

```python
from anomolous_ts.detector import TimeSeriesKMeansDetector

# Initialize detector
detector = TimeSeriesKMeansDetector(
    window_size=24,
    n_clusters=3,
    anomaly_threshold=2.0,
    seasonal_period=24
)

# Detect anomalies
anomalies, scores = detector.fit_predict(your_time_series_data, return_scores=True)

# Visualize results
detector.detect_and_visualize(your_time_series_data, title="Anomaly Detection Results")
```

## Class Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| window_size | int | 10 | Size of sliding window for feature extraction |
| n_clusters | int | 3 | Number of clusters for K-means |
| anomaly_threshold | float | 2.0 | Number of standard deviations from cluster center to consider as anomaly |
| seasonal_period | int | None | Period for seasonal feature extraction |
| preprocessor | TimeSeriesPreprocessor | None | Custom preprocessor instance |
| visualizer | TimeSeriesVisualizer | None | Custom visualizer instance |
| random_state | int | 42 | Random state for reproducibility |

## Main Methods

### fit_predict(data, return_scores=False)
Fits the model and predicts anomalies in one step.

```python
# Get anomaly indicators only
anomalies = detector.fit_predict(data)

# Get both anomaly indicators and scores
anomalies, scores = detector.fit_predict(data, return_scores=True)
```

**Parameters:**
- `data`: pandas Series or DataFrame
- `return_scores`: bool, whether to return anomaly scores

**Returns:**
- Single output: pandas Series/DataFrame of boolean anomaly indicators
- With return_scores=True: tuple of (anomaly indicators, anomaly scores)

### detect_and_visualize(data, title="K-means Clustering Anomaly Detection")
Detects anomalies and creates a visualization in one step.

```python
anomalies, scores = detector.detect_and_visualize(data, title="My Time Series Analysis")
```

**Parameters:**
- `data`: pandas Series or DataFrame
- `title`: str, plot title

**Returns:**
- tuple of (anomalies, anomaly scores)

### process_stream(data_stream, chunk_size=100)
Processes streaming data in chunks.

```python
for chunk_data, anomalies, scores in detector.process_stream(data_stream, chunk_size=100):
    # Process results for each chunk
    pass
```

**Parameters:**
- `data_stream`: iterator yielding new data points
- `chunk_size`: int, size of data chunks to process

**Yields:**
- tuple of (chunk_data, anomalies, scores)

### get_cluster_profiles()
Returns statistical profiles of each cluster.

```python
profiles = detector.get_cluster_profiles()
```

**Returns:**
- Dictionary containing cluster statistics

## Feature Extraction
The detector creates the following features for each window:

### Basic Features:
1. Mean
2. Standard deviation
3. Minimum value
4. Maximum value
5. Median
6. Peak-to-peak (range)
7. Current value
8. Change from previous
9. Number of positive changes
10. Skewness
11. Kurtosis
12. 25th percentile
13. 75th percentile

### Seasonal Features (if seasonal_period is set):
1. Seasonal difference
2. Seasonal mean
3. Seasonal standard deviation

## Anomaly Detection Process

1. **Preprocessing:**
   - Handles missing values using the configured preprocessor
   - Scales features if a scaler is provided

2. **Feature Extraction:**
   - Creates temporal features using sliding windows
   - Adds seasonal features if specified

3. **Clustering:**
   - Applies K-means clustering to the feature space
   - Assigns data points to clusters

4. **Anomaly Scoring:**
   - Calculates distances to cluster centers
   - Converts distances to z-scores
   - Applies sigmoid function for final scoring

5. **Anomaly Determination:**
   - Points with scores above the anomaly threshold are marked as anomalies

## Examples

### Basic Usage
```python
import pandas as pd
from anomolous_ts.detector import TimeSeriesKMeansDetector

# Create detector
detector = TimeSeriesKMeansDetector(
    window_size=24,
    n_clusters=3,
    seasonal_period=24  # For daily seasonality
)

# Load data
data = pd.read_csv('timeseries.csv', parse_dates=['timestamp'], index_col='timestamp')

# Detect anomalies
anomalies, scores = detector.detect_and_visualize(data)

# Print summary
print(f"Found {anomalies.sum()} anomalies")
print(f"Average anomaly score: {scores.mean():.3f}")
```

### Multivariate Analysis
```python
# For multiple time series columns
data = pd.DataFrame({
    'metric1': [...],
    'metric2': [...]
})

# Detect anomalies across all metrics
anomalies_df, scores_df = detector.fit_predict(data, return_scores=True)

# Check anomalies per metric
for column in data.columns:
    print(f"\nMetric: {column}")
    print(f"Anomalies: {anomalies_df[column].sum()}")
    print(f"Average score: {scores_df[column].mean():.3f}")
```

### Streaming Data Processing
```python
def data_stream():
    while True:
        # Get new data
        yield new_data_point

# Process streaming data
for chunk_data, anomalies, scores in detector.process_stream(data_stream(), chunk_size=100):
    # Handle results
    print(f"Processed {len(chunk_data)} points")
    print(f"Found {anomalies.sum()} anomalies")
```

## Best Practices

1. **Window Size Selection:**
   - Should be large enough to capture relevant patterns
   - Typically set to the length of expected patterns
   - For daily patterns, use 24 for hourly data

2. **Number of Clusters:**
   - Start with a small number (3-5)
   - Increase if distinct patterns are being merged
   - Decrease if clusters are too similar

3. **Anomaly Threshold:**
   - Default of 2.0 works well for many cases
   - Lower for more sensitive detection
   - Higher to detect only extreme anomalies

4. **Seasonal Period:**
   - Set to known seasonality if present
   - Common values: 24 (daily), 168 (weekly for hourly data)
   - Leave as None if no clear seasonality

## Troubleshooting

### Common Issues and Solutions

1. **Too Many Anomalies Detected:**
   - Increase anomaly_threshold
   - Increase window_size
   - Decrease n_clusters

2. **Missing Important Anomalies:**
   - Decrease anomaly_threshold
   - Decrease window_size
   - Increase n_clusters

3. **Poor Performance on Seasonal Data:**
   - Set seasonal_period correctly
   - Increase window_size to cover seasonal pattern
   - Ensure data is properly preprocessed

4. **Memory Issues with Large Datasets:**
   - Use process_stream with appropriate chunk_size
   - Reduce window_size if possible
   - Sample data if appropriate

## Performance Considerations

1. **Time Complexity:**
   - O(n * w) for feature extraction (n = points, w = window_size)
   - O(n * k * i) for K-means (k = n_clusters, i = iterations)

2. **Memory Usage:**
   - Scales with window_size * number of features
   - Increases with n_clusters
   - Consider streaming for large datasets

3. **Optimization Tips:**
   - Use smaller window_size if possible
   - Limit n_clusters to necessary minimum
   - Process large datasets in chunks

## Contributing

To contribute to the TimeSeriesKMeansDetector:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

Please ensure all tests pass and code is well-documented.
