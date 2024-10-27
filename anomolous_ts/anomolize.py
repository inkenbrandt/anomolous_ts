import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TimeseriesGenerator:
    def __init__(self, start_date='2024-01-01', periods=365, frequency='D'):
        """
        Initialize the timeseries generator.

        Args:
            start_date (str): Start date for the time series
            periods (int): Number of periods to generate
            frequency (str): Frequency of data points ('D' for daily, 'H' for hourly, etc.)
        """
        self.start_date = pd.to_datetime(start_date)
        self.periods = periods
        self.frequency = frequency
        self.date_range = pd.date_range(start=start_date, periods=periods, freq=frequency)

    def _generate_seasonal_pattern(self, period, amplitude=1.0, phase=0):
        """Generate a seasonal pattern using sine waves."""
        t = np.arange(self.periods)
        return amplitude * np.sin(2 * np.pi * (t + phase) / period)

    def _generate_trend(self, slope=0.1):
        """Generate a linear trend."""
        return np.arange(self.periods) * slope

    def _add_noise(self, data, noise_level=0.1):
        """Add random noise to the data."""
        return data + np.random.normal(0, noise_level, self.periods)

    def _generate_anomalies(self, data, num_anomalies=5, amplitude_range=(2, 4)):
        """Add random anomalies to the data."""
        anomaly_indices = np.random.choice(self.periods, num_anomalies, replace=False)
        anomaly_data = data.copy()
        for idx in anomaly_indices:
            multiplier = np.random.uniform(*amplitude_range)
            sign = np.random.choice([-1, 1])
            anomaly_data[idx] += sign * multiplier * np.std(data)
        return anomaly_data, anomaly_indices

    def generate_correlated_series(self, n_series=2, correlation_matrix=None,
                                   seasonal_periods=[365, 7], trend_slopes=None,
                                   noise_levels=None, num_anomalies=5):
        """
        Generate multiple correlated time series with seasonality, trends, and anomalies.

        Args:
            n_series (int): Number of time series to generate
            correlation_matrix (np.ndarray): Correlation matrix for the series
            seasonal_periods (list): List of seasonal periods to include
            trend_slopes (list): List of trend slopes for each series
            noise_levels (list): List of noise levels for each series
            num_anomalies (int): Number of anomalies to introduce

        Returns:
            pd.DataFrame: DataFrame containing the generated time series
        """
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_series)
            correlation_matrix[correlation_matrix == 0] = 0.5

        if trend_slopes is None:
            trend_slopes = [0.1] * n_series

        if noise_levels is None:
            noise_levels = [0.1] * n_series

        # Generate base series with seasonality and trends
        base_series = []
        for i in range(n_series):
            series = np.zeros(self.periods)

            # Add multiple seasonal patterns
            for period in seasonal_periods:
                series += self._generate_seasonal_pattern(
                    period,
                    amplitude=np.random.uniform(0.5, 2.0),
                    phase=np.random.uniform(0, period)
                )

            # Add trend
            series += self._generate_trend(trend_slopes[i])
            base_series.append(series)

        # Convert to numpy array
        base_series = np.array(base_series)

        # Generate correlated noise
        L = np.linalg.cholesky(correlation_matrix)
        correlated_noise = np.dot(L, np.random.randn(n_series, self.periods))

        # Add noise to base series
        for i in range(n_series):
            base_series[i] = base_series[i] + noise_levels[i] * correlated_noise[i]

        # Create DataFrame
        df = pd.DataFrame(base_series.T, columns=[f'series_{i + 1}' for i in range(n_series)])
        df.index = self.date_range

        # Add anomalies to each series
        anomaly_locations = {}
        for column in df.columns:
            df[column], anomaly_indices = self._generate_anomalies(
                df[column].values,
                num_anomalies=num_anomalies
            )
            anomaly_locations[column] = self.date_range[anomaly_indices]

        return df, anomaly_locations


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = TimeseriesGenerator(start_date='2024-01-01', periods=365)

    # Define correlation matrix for 3 series
    correlation_matrix = np.array([
        [1.0, 0.7, -0.3],
        [0.7, 1.0, -0.5],
        [-0.3, -0.5, 1.0]
    ])

    # Generate data
    df, anomaly_locations = generator.generate_correlated_series(
        n_series=3,
        correlation_matrix=correlation_matrix,
        seasonal_periods=[365, 7],  # Annual and weekly seasonality
        trend_slopes=[0.1, 0.05, -0.08],
        noise_levels=[0.1, 0.15, 0.12],
        num_anomalies=5
    )

    print("Generated DataFrame:")
    print(df.head())
    print("\nAnomalies located at:")
    for series, dates in anomaly_locations.items():
        print(f"{series}: {dates.tolist()}")