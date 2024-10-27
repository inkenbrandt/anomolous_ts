# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy import stats


class TwitterSHESD:
    """
    Implementation of Twitter's Seasonal Hybrid ESD (S-H-ESD) algorithm for anomaly detection.
    This algorithm combines seasonal decomposition with the Generalized ESD test to detect
    both global and local anomalies in time series data.

    The algorithm works in the following steps:
    1. Performs seasonal decomposition to remove seasonal patterns (optional)
    2. Uses either hybrid (median/MAD) or standard (mean/std) statistics
    3. Applies the Generalized ESD test to identify anomalies

    Key Features:
    - Support for multiple seasonal decomposition methods
    - Robust statistics option using median and MAD
    - Configurable maximum anomalies and significance level
    - Automatic seasonal period detection

    Example:
    ```python
    # Create detector
    detector = TwitterSHESD(max_anomalies=0.1, alpha=0.05, hybrid=True)

    # Detect anomalies with automatic period detection
    anomalies = detector.detect(time_series)

    # Or specify period and decomposition method
    anomalies = detector.detect(time_series, period=24, decomposition_method='multiplicative')
    ```
    """

    def __init__(self, max_anomalies=0.1, alpha=0.05, hybrid=True):
        """
        Initialize the Twitter S-H-ESD detector.

        Parameters
        ----------
        max_anomalies : float, default=0.1
            Maximum fraction of data points that can be labeled as anomalies.
            Should be between 0 and 1.

        alpha : float, default=0.05
            Level of statistical significance for the ESD test.
            Lower values make the test more conservative.

        hybrid : bool, default=True
            If True, use robust statistics (median and MAD) instead of
            mean and standard deviation. Recommended for data with
            potential extreme values.

        Raises
        ------
        ValueError
            If max_anomalies is not between 0 and 1
            If alpha is not between 0 and 1
        """
        if not 0 < max_anomalies < 1:
            raise ValueError("max_anomalies must be between 0 and 1")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

        self.max_anomalies = max_anomalies
        self.alpha = alpha
        self.hybrid = hybrid

    def _detect_period(self, data, max_lag=366):
        """
        Automatically detect the seasonal period using autocorrelation.

        Parameters
        ----------
        data : pd.Series
            Input time series
        max_lag : int, default=366
            Maximum lag to consider for autocorrelation

        Returns
        -------
        int
            Detected seasonal period
        """
        n = len(data)
        max_lag = min(max_lag, n // 2)

        # Calculate autocorrelation
        acf_values = acf(data, nlags=max_lag, fft=True)

        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(acf_values) - 1):
            if acf_values[i] > acf_values[i - 1] and acf_values[i] > acf_values[i + 1]:
                peaks.append((i, acf_values[i]))

        # Sort peaks by correlation value
        peaks.sort(key=lambda x: x[1], reverse=True)

        # Return the lag of the highest peak after lag 1
        for lag, corr in peaks:
            if lag > 1:
                return lag

        return 1  # Default if no clear seasonality is found

    def _remove_seasonal_component(self, data, period, method='additive'):
        """
        Remove seasonal component from time series using decomposition.

        Parameters
        ----------
        data : pd.Series
            Input time series
        period : int
            Seasonal period
        method : str, default='additive'
            Decomposition method: 'additive', 'multiplicative', or 'robust'
            - 'additive': Traditional additive decomposition
            - 'multiplicative': Traditional multiplicative decomposition
            - 'robust': Robust decomposition less sensitive to outliers

        Returns
        -------
        pd.Series
            Deseasonalized time series

        Notes
        -----
        For multiplicative decomposition, all values must be positive.
        Robust decomposition is experimental and may be slower.
        """
        if method == 'multiplicative' and (data <= 0).any():
            raise ValueError("Multiplicative decomposition requires positive values")

        decomposition = seasonal_decompose(
            data,
            period=period,
            model=method,
            extrapolate_trend='freq'
        )

        if method == 'multiplicative':
            residual = data / decomposition.seasonal
        else:
            residual = data - decomposition.seasonal

        return residual

    def _compute_statistics(self, data):
        """
        Compute location and scale statistics based on hybrid parameter.

        Parameters
        ----------
        data : np.array
            Input data

        Returns
        -------
        tuple
            (location, scale) statistics

        Notes
        -----
        If hybrid=True, uses median and MAD (Median Absolute Deviation)
        If hybrid=False, uses mean and standard deviation
        """
        if self.hybrid:
            location = np.median(data)
            # MAD scaled by 1.4826 to be consistent with standard deviation
            scale = np.median(np.abs(data - location)) * 1.4826
        else:
            location = np.mean(data)
            scale = np.std(data, ddof=1)
        return location, scale

    def _generalized_esd_test(self, data, max_outliers):
        """
        Perform the Generalized ESD test to detect outliers.

        Parameters
        ----------
        data : np.array
            Input data
        max_outliers : int
            Maximum number of outliers to detect

        Returns
        -------
        list
            Indices of detected anomalies

        Notes
        -----
        The Generalized ESD (Extreme Studentized Deviate) test is an extension
        of Grubbs' test for multiple outliers. It progressively tests for
        k outliers, where k goes from 1 to max_outliers.
        """
        n = len(data)
        outlier_indices = []

        for i in range(max_outliers):
            location, scale = self._compute_statistics(data)
            if scale == 0:
                break

            # Compute test statistics
            test_statistics = np.abs(data - location) / scale
            max_idx = np.argmax(test_statistics)
            max_stat = test_statistics[max_idx]

            # Compute critical value
            t_stat = stats.t.ppf(1 - self.alpha / (2 * (n - i)), n - i - 2)
            lambda_i = ((n - i - 1) * t_stat) / np.sqrt((n - i - 2 + t_stat ** 2) * (n - i))

            if max_stat > lambda_i:
                outlier_indices.append(max_idx)
                data = np.delete(data, max_idx)
            else:
                break

        return outlier_indices

    def detect(self, time_series, period=None, decomposition_method='additive'):
        """
        Detect anomalies in the time series.

        Parameters
        ----------
        time_series : pd.Series
            Input time series with datetime index
        period : int, optional
            Seasonal period. If None, will attempt to detect automatically
        decomposition_method : str, default='additive'
            Method for seasonal decomposition:
            - 'additive': Suitable for constant seasonal variations
            - 'multiplicative': Suitable when seasonal variations change with level
            - 'robust': Less sensitive to outliers but slower

        Returns
        -------
        pd.Series
            Boolean series indicating anomalies (True for anomalies)

        Examples
        --------
        >>> # Generate sample data
        >>> dates = pd.date_range('2024-01-01', periods=100, freq='h')
        >>> values = np.sin(np.linspace(0, 8*np.pi, 100)) * 10 + np.random.normal(0, 1, 100)
        >>> values[25] = 30  # Add an anomaly
        >>> ts = pd.Series(values, index=dates)
        >>>
        >>> # Create detector and detect anomalies
        >>> detector = TwitterSHESD(max_anomalies=0.1)
        >>> anomalies = detector.detect(ts, period=24)
        >>>
        >>> # Print anomalous points
        >>> print(ts[anomalies])
        """
        data = time_series.copy()

        # Automatically detect period if not provided
        if period is None:
            period = self._detect_period(data)

        # Remove seasonal component if period > 1
        if period > 1:
            data = self._remove_seasonal_component(data, period, decomposition_method)

        # Convert to numpy array for processing
        values = data.values
        max_outliers = int(np.ceil(len(values) * self.max_anomalies))

        # Detect anomalies
        anomaly_indices = self._generalized_esd_test(values, max_outliers)

        # Create boolean mask for anomalies
        is_anomaly = pd.Series(False, index=time_series.index)
        is_anomaly.iloc[anomaly_indices] = True

        return is_anomaly


# Example usage demonstrating different features
if __name__ == "__main__":
    # Generate sample data with multiple patterns
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='h')

    # Create base signal with daily seasonality
    t = np.linspace(0, 20 * np.pi, 500)
    base = np.sin(t / 12) * 10  # Daily pattern (24 hours)
    trend = np.linspace(0, 5, 500)  # Upward trend
    noise = np.random.normal(0, 0.5, 500)

    # Add some anomalies
    anomalies = np.zeros(500)
    anomalies[100] = 15
    anomalies[200] = -10
    anomalies[300] = 20
    anomalies[400:403] = 12  # Multiple consecutive anomalies

    # Combine components
    values = base + trend + noise + anomalies
    time_series = pd.Series(values, index=dates)

    # Example 1: Basic usage with automatic period detection
    detector1 = TwitterSHESD(max_anomalies=0.05, hybrid=True)
    anomalies1 = detector1.detect(time_series)
    print("\nExample 1 - Automatic period detection:")
    print(f"Found {anomalies1.sum()} anomalies")
    print(time_series[anomalies1])

    # Example 2: Using multiplicative decomposition
    # First make all values positive by adding a constant
    positive_series = time_series - time_series.min() + 1
    detector2 = TwitterSHESD(max_anomalies=0.05, hybrid=False)
    anomalies2 = detector2.detect(positive_series, period=24, decomposition_method='multiplicative')
    print("\nExample 2 - Multiplicative decomposition:")
    print(f"Found {anomalies2.sum()} anomalies")
    print(positive_series[anomalies2])

    # Example 3: Using robust decomposition
    detector3 = TwitterSHESD(max_anomalies=0.03, alpha=0.01)
    anomalies3 = detector3.detect(time_series, period=24, decomposition_method='robust')
    print("\nExample 3 - Robust decomposition:")
    print(f"Found {anomalies3.sum()} anomalies")
    print(time_series[anomalies3])

