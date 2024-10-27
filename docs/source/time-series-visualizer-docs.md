# TimeSeriesVisualizer Documentation

## Overview

The TimeSeriesVisualizer is a Python class designed to help data scientists and analysts visualize time series data, particularly focusing on anomaly detection and data imputation results. It provides intuitive plotting functions that work with both single and multiple time series data.

## Installation

The TimeSeriesVisualizer requires the following dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy statsmodels
```

## Quick Start

```python
from time_series_visualizer import TimeSeriesVisualizer

# Initialize the visualizer
visualizer = TimeSeriesVisualizer()

# Plot anomalies
visualizer.plot_anomalies(your_data, anomalies, confidence_scores)

# Plot imputation results
visualizer.plot_imputation(original_data, imputed_data)
```

## Class Reference

### TimeSeriesVisualizer

A class for visualizing time series data with support for anomaly detection and imputation visualization.

#### Methods

##### plot_anomalies
```python
def plot_anomalies(self, data, anomalies, confidence_scores=None, title='Anomaly Detection Results')
```

Visualizes time series data with highlighted anomalies and optional confidence scores.

**Parameters:**
- `data` (pandas.Series or pandas.DataFrame): The original time series data
  - For single series: A pandas Series with datetime index
  - For multiple series: A pandas DataFrame where each column is a separate time series
- `anomalies` (pandas.Series or pandas.DataFrame): Boolean indicators for anomalies
  - Must match the structure of `data` (Series for single series, DataFrame for multiple series)
  - True values indicate anomalous points
- `confidence_scores` (pandas.Series or pandas.DataFrame, optional): Confidence scores for anomalies
  - If provided, scores will be displayed next to anomalous points
  - Values typically range from 0 to 1
- `title` (str, default='Anomaly Detection Results'): Title for the plot

**Returns:**
- None (displays the plot)

**Example:**
```python
# Single series example
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2024-01-01', periods=100, freq='D')
data = pd.Series(np.random.randn(100), index=dates)
anomalies = pd.Series(False, index=dates)
anomalies.iloc[25:30] = True
confidence_scores = pd.Series(np.random.uniform(0.8, 1.0, 100), index=dates)

visualizer = TimeSeriesVisualizer()
visualizer.plot_anomalies(data, anomalies, confidence_scores)
```

##### plot_imputation
```python
def plot_imputation(self, original_data, imputed_data, title='Imputation Results')
```

Visualizes original data with missing values alongside imputed results.

**Parameters:**
- `original_data` (pandas.Series or pandas.DataFrame): Original data containing missing values
  - For single series: A pandas Series with datetime index
  - For multiple series: A pandas DataFrame where each column is a separate time series
- `imputed_data` (pandas.Series or pandas.DataFrame): Data after imputation
  - Must match the structure of `original_data`
  - Should contain values for all missing points in `original_data`
- `title` (str, default='Imputation Results'): Title for the plot

**Returns:**
- None (displays the plot)

**Example:**
```python
# Multiple series example
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2024-01-01', periods=100, freq='D')
original_data = pd.DataFrame({
    'series1': np.random.randn(100),
    'series2': np.random.randn(100)
}, index=dates)

# Add missing values
original_data.iloc[25:30, 0] = np.nan
original_data.iloc[60:65, 1] = np.nan

# Create imputed data
imputed_data = original_data.interpolate()

visualizer = TimeSeriesVisualizer()
visualizer.plot_imputation(original_data, imputed_data)
```

## Plot Features

### Anomaly Detection Plots
- Normal data points are connected with lines
- Anomalies are highlighted with red markers
- Optional confidence scores are displayed next to anomalous points
- Multiple series are distinguished by different colors
- Includes a legend identifying normal and anomalous points
- Grid lines for better readability

### Imputation Plots
- Original data points are shown with blue circles connected by lines
- Imputed values are marked with red X markers
- Multiple series are distinguished by different colors
- Includes a legend identifying original and imputed points
- Grid lines for better readability

## Best Practices

1. **Data Preparation**
   - Ensure your time series data has a proper datetime index
   - Clean or remove any infinite values before visualization
   - For multiple series, use consistent column names across your data structures

2. **Anomaly Detection**
   - Keep confidence score values between 0 and 1 for consistency
   - Ensure anomaly boolean indicators align exactly with your data index
   - Use meaningful threshold values when determining anomalies

3. **Imputation**
   - Verify that your imputed data completely fills all missing values
   - Consider the appropriateness of your imputation method
   - Check for any remaining NaN values before visualization

## Error Handling

The visualizer handles several common error cases:
- Mismatched dimensions between data and anomalies/confidence scores
- Invalid data types
- Missing values in confidence scores
- Index misalignment between original and imputed data

## Performance Considerations

- For large datasets (>10000 points), plotting may be slower
- Consider downsampling very large datasets before visualization
- Memory usage increases with the number of series being plotted

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Add tests for any new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
