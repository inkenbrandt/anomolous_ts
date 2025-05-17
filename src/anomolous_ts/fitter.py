
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union, Tuple, NamedTuple
from datetime import datetime
from enum import Enum
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy.spatial.distance import cdist
from prophet import Prophet
import warnings

class DetectionMethod(Enum):
    """Available anomaly detection methods."""
    STATISTICAL = "statistical"  # Using z-scores
    ISOLATION_FOREST = "isolation_forest"  # Using Isolation Forest
    KMEANS = "kmeans"  # Using KMeans clustering
    PROPHET = "prophet"  # Using Facebook Prophet

class DecompositionResult(NamedTuple):
    """Container for seasonal decomposition results."""
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series

class ProphetResult(NamedTuple):
    """Container for Prophet model results."""
    forecast: pd.DataFrame
    model: Prophet
    changepoints: pd.Series

class TimeSeriesAnomaly:
    """Container for anomaly information with additional context."""

    def __init__(
            self,
            timestamp: datetime,
            score: float,
            value: float,
            seasonal_strength: float = 0.0,
            trend_strength: float = 0.0,
            anomaly_type: str = "point"
    ):
        self.timestamp = timestamp
        self.score = score  # How anomalous the point is (0-1)
        self.value = value  # The actual value that was anomalous
        self.seasonal_strength = seasonal_strength  # How much seasonality contributed
        self.trend_strength = trend_strength  # How much trend contributed
        self.anomaly_type = anomaly_type  # Type of anomaly: "point", "seasonal", "trend"


class TimeSeriesDetector:
    """Enhanced anomaly detector with multiple detection methods."""

    def __init__(
            self,
            method: Union[str, DetectionMethod] = DetectionMethod.STATISTICAL,
            window_size: int = 24,
            threshold: float = 3.0,
            contamination: float = 0.1,
            n_clusters: int = 5,
            seasonal_period: Optional[int] = None,
            decomposition_method: str = 'additive',
            prophet_params: Optional[Dict] = None
    ):
        """Initialize detector with proper model initialization."""
        if isinstance(method, str):
            method = DetectionMethod(method)
        self.method = method
        self.window_size = window_size
        self.threshold = threshold
        self.contamination = contamination
        self.n_clusters = n_clusters
        self.seasonal_period = seasonal_period
        self.decomposition_method = decomposition_method
        self.prophet_params = prophet_params or {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': True,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10,
            'interval_width': 0.95
        }

        # Initialize models based on method
        self._initialize_models()

        # Store results
        self.prophet_result = None
        self._cluster_distances = None

    def _initialize_models(self):
        """Initialize appropriate models based on detection method."""
        if self.method == DetectionMethod.ISOLATION_FOREST:
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
        else:
            self.isolation_forest = None

        if self.method == DetectionMethod.KMEANS:
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42
            )
        else:
            self.kmeans = None

        if self.method == DetectionMethod.PROPHET:
            self.prophet_model = Prophet(**self.prophet_params)
        else:
            self.prophet_model = None

    def detect(self, data: Union[pd.Series, pd.DataFrame]) -> List[TimeSeriesAnomaly]:
        """
        Detect anomalies with robust error handling and fallback options.
        """
        # Convert DataFrame to Series if needed
        if isinstance(data, pd.DataFrame):
            if data.shape[1] > 1:
                raise ValueError("Multiple columns not supported. Please pass a single series.")
            data = data.iloc[:, 0]

        # Validate data
        if len(data) < 2:
            raise ValueError("Data must contain at least 2 points")

        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"Index must be convertible to datetime: {str(e)}")

        # Perform seasonal decomposition
        components = self._decompose_series(data)

        try:
            # Apply selected detection method
            if self.method == DetectionMethod.PROPHET:
                try:
                    return self._detect_prophet(data, components)
                except Exception as prophet_error:
                    warnings.warn(
                        f"Prophet detection failed: {str(prophet_error)}. "
                        "Falling back to statistical method."
                    )
                    self.method = DetectionMethod.STATISTICAL
                    return self._detect_statistical(data, components)
            elif self.method == DetectionMethod.STATISTICAL:
                return self._detect_statistical(data, components)
            elif self.method == DetectionMethod.ISOLATION_FOREST:
                return self._detect_isolation_forest(data, components)
            elif self.method == DetectionMethod.KMEANS:
                return self._detect_kmeans(data, components)

        except Exception as e:
            raise RuntimeError(
                f"Error detecting anomalies with {self.method.value} method: {str(e)}"
            )

    def _extract_features(self, data: pd.Series, components: DecompositionResult) -> pd.DataFrame:
        """Extract features for model-based detection with standardization."""
        features = pd.DataFrame()

        # Rolling statistics on residuals
        rolling = components.residual.rolling(window=self.window_size, center=True)

        features['value'] = data
        features['trend'] = components.trend
        features['seasonal'] = components.seasonal
        features['residual'] = components.residual
        features['mean'] = rolling.mean()
        features['std'] = rolling.std()
        features['max'] = rolling.max()
        features['min'] = rolling.min()

        # Add lag features
        for i in range(1, min(4, self.window_size)):
            features[f'lag_{i}'] = data.shift(i)

        # First and second differences
        features['diff1'] = data.diff()
        features['diff2'] = data.diff().diff()

        # Handle missing values
        features = features.bfill().ffill()

        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )

        return features_scaled

    def _detect_period(self, data: pd.Series) -> int:
        """
        Automatically detect seasonal period using autocorrelation.
        """
        # Calculate autocorrelation
        acf_values = acf(data, nlags=min(len(data) // 2, 366), fft=True)

        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(acf_values) - 1):
            if acf_values[i] > acf_values[i - 1] and acf_values[i] > acf_values[i + 1]:
                peaks.append((i, acf_values[i]))

        # Sort peaks by correlation value
        peaks.sort(key=lambda x: x[1], reverse=True)

        # Return the first significant peak
        for period, correlation in peaks:
            if period > 1 and correlation > 0.3:  # Correlation threshold
                return period

        return 1  # Default if no clear seasonality is found

    def _decompose_series(self, data: pd.Series) -> DecompositionResult:
        """
        Perform seasonal decomposition of the time series.
        """
        # Detect period if not provided
        if self.seasonal_period is None:
            self.seasonal_period = self._detect_period(data)

        # Only decompose if we found a seasonal pattern
        if self.seasonal_period > 1:
            decomp = seasonal_decompose(
                data,
                period=self.seasonal_period,
                model=self.decomposition_method,
                extrapolate_trend='freq'
            )

            # Fill any missing values in components
            trend = pd.Series(decomp.trend, index=data.index).ffill().bfill()
            seasonal = pd.Series(decomp.seasonal, index=data.index).ffill().bfill()
            residual = pd.Series(decomp.resid, index=data.index).ffill().bfill()

            return DecompositionResult(trend, seasonal, residual)
        else:
            # Return original data as residual if no seasonality
            return DecompositionResult(
                pd.Series(0, index=data.index),
                pd.Series(0, index=data.index),
                data
            )

    def _fit_prophet(self, data: pd.Series) -> ProphetResult:
        """
        Fit Prophet model with robust error handling and simplified parameters.
        """
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })

            # Initialize Prophet with simplified, robust parameters
            model = Prophet(
                # Simplify seasonality settings
                yearly_seasonality=False,  # Unless data spans multiple years
                weekly_seasonality=True if self.seasonal_period >= 7 else False,
                daily_seasonality=True if self.seasonal_period >= 1 else False,

                # Use more stable parameters
                changepoint_prior_scale=0.01,  # Reduce flexibility
                seasonality_prior_scale=10.0,
                seasonality_mode='additive',
                mcmc_samples=0,  # Disable MCMC
                interval_width=0.95,

                # Add growth caps if needed
                growth='linear',
                cap=data.max() * 1.5 if data.min() >= 0 else None,
                floor=data.min() * 1.5 if data.max() <= 0 else None
            )

            # Add custom seasonality if needed
            if self.seasonal_period and self.seasonal_period not in [1, 7, 365]:
                model.add_seasonality(
                    name='custom',
                    period=self.seasonal_period,
                    fourier_order=5
                )

            # Suppress Prophet warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Try different optimization approaches
                try:
                    # First attempt: Default settings
                    model.fit(df)
                except Exception as e1:
                    try:
                        # Second attempt: Simplified model
                        model = Prophet(
                            yearly_seasonality=False,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                            changepoint_prior_scale=0.001,
                            seasonality_prior_scale=0.1,
                            n_changepoints=5,
                            interval_width=0.95
                        )
                        model.fit(df)
                    except Exception as e2:
                        try:
                            # Third attempt: Minimal model
                            model = Prophet(
                                growth='flat',
                                yearly_seasonality=False,
                                weekly_seasonality=False,
                                daily_seasonality=False,
                                n_changepoints=0,
                                interval_width=0.95
                            )
                            model.fit(df)
                        except Exception as e3:
                            raise RuntimeError(
                                f"All Prophet fitting attempts failed:\n"
                                f"1st attempt: {str(e1)}\n"
                                f"2nd attempt: {str(e2)}\n"
                                f"3rd attempt: {str(e3)}"
                            )

            # Generate forecast for the same period
            forecast = model.predict(df)

            # Extract changepoints
            changepoints = pd.Series(
                model.params['delta'].mean(0) if 'delta' in model.params else [],
                index=pd.to_datetime(model.changepoints) if model.changepoints is not None else []
            )

            return ProphetResult(forecast, model, changepoints)

        except Exception as e:
            raise RuntimeError(f"Prophet model fitting failed: {str(e)}")

    def _detect_prophet(self, data: pd.Series, components: DecompositionResult) -> List[TimeSeriesAnomaly]:
        """
        Detect anomalies using Prophet with robust error handling.
        """
        try:
            # Fit Prophet if not already fitted
            if self.prophet_result is None:
                self.prophet_result = self._fit_prophet(data)

            forecast = self.prophet_result.forecast

            # Calculate prediction intervals
            forecast['lower'] = forecast['yhat'] - self.threshold * forecast['yhat_std']
            forecast['upper'] = forecast['yhat'] + self.threshold * forecast['yhat_std']

            # Find anomalies
            anomalies = []
            for i in range(len(data)):
                try:
                    forecast_row = forecast.iloc[i]
                    value = data.iloc[i]

                    if value < forecast_row['lower'] or value > forecast_row['upper']:
                        # Calculate deviation score
                        std_dev = max(forecast_row['yhat_std'], 1e-10)  # Avoid division by zero
                        deviation = abs(value - forecast_row['yhat']) / std_dev
                        score = 1 / (1 + np.exp(-deviation + self.threshold))

                        # Get component values
                        values = self._get_component_values(i, data, components)

                        # Calculate total deviation
                        total_dev = sum(abs(v) for v in values.values())
                        if total_dev == 0:
                            total_dev = 1

                        # Determine anomaly type
                        if abs(values['seasonal']) > abs(values['trend']):
                            anomaly_type = "seasonal"
                        elif abs(values['trend']) > abs(values['value'] - values['trend'] - values['seasonal']):
                            anomaly_type = "trend"
                        else:
                            anomaly_type = "point"

                        anomalies.append(TimeSeriesAnomaly(
                            timestamp=data.index[i],
                            score=float(score),
                            value=float(value),
                            expected_value=float(forecast_row['yhat']),
                            prediction_interval=(
                                float(forecast_row['lower']),
                                float(forecast_row['upper'])
                            ),
                            seasonal_strength=abs(values['seasonal']) / total_dev,
                            trend_strength=abs(values['trend']) / total_dev,
                            anomaly_type=anomaly_type
                        ))
                except Exception as e:
                    warnings.warn(f"Error processing point at index {i}: {str(e)}")
                    continue

            return anomalies

        except Exception as e:
            raise RuntimeError(f"Prophet detection failed: {str(e)}")

    def _get_component_values(self, i: int, data: pd.Series, components: DecompositionResult) -> Dict[str, float]:
        """Helper function to get component values using proper indexing."""
        return {
            'value': data.iloc[i],
            'seasonal': components.seasonal.iloc[i],
            'trend': components.trend.iloc[i],
            'residual': components.residual.iloc[i]
        }

    def _calculate_strength_scores(
            self,
            i: int,
            data: pd.Series,
            components: DecompositionResult,
            score: float
    ) -> TimeSeriesAnomaly:
        """Calculate anomaly strength scores using proper indexing."""
        # Get values using proper indexing
        values = self._get_component_values(i, data, components)

        # Calculate total deviation
        total_deviation = (
                abs(values['seasonal']) +
                abs(values['trend']) +
                abs(values['value'] - values['trend'] - values['seasonal'])
        )

        if total_deviation == 0:
            total_deviation = 1

        # Determine anomaly type
        if abs(values['seasonal']) > abs(values['trend']):
            anomaly_type = "seasonal"
        elif abs(values['trend']) > abs(values['value'] - values['trend'] - values['seasonal']):
            anomaly_type = "trend"
        else:
            anomaly_type = "point"

        return TimeSeriesAnomaly(
            timestamp=data.index[i],
            score=float(score),
            value=float(values['value']),
            seasonal_strength=abs(values['seasonal']) / total_deviation,
            trend_strength=abs(values['trend']) / total_deviation,
            anomaly_type=anomaly_type
        )

    def _detect_statistical(
            self,
            data: pd.Series,
            components: DecompositionResult
    ) -> List[TimeSeriesAnomaly]:
        """Detect anomalies using statistical method."""
        # Calculate rolling statistics
        rolling_mean = data.rolling(window=self.window_size, center=True).mean()
        rolling_std = data.rolling(window=self.window_size, center=True).std()

        # Calculate z-scores
        z_scores = np.abs((data - rolling_mean) / rolling_std)
        z_scores = z_scores.fillna(0)

        # Find anomalies
        anomalies = []
        for i in range(len(data)):
            if z_scores.iloc[i] > self.threshold:
                score = 1 / (1 + np.exp(-z_scores.iloc[i] + self.threshold))
                anomalies.append(
                    self._calculate_strength_scores(i, data, components, score)
                )

        return anomalies

    def _detect_isolation_forest(
            self,
            data: pd.Series,
            components: DecompositionResult
    ) -> List[TimeSeriesAnomaly]:
        """Detect anomalies using Isolation Forest."""
        # Extract features
        features = self._extract_features(data, components)

        # Ensure model is initialized
        if self.isolation_forest is None:
            self._initialize_models()

        # Fit and predict
        self.isolation_forest.fit(features)
        scores = self.isolation_forest.score_samples(features)

        # Convert scores to probabilities
        scores = 1 / (1 + np.exp(scores))

        # Find anomalies
        anomalies = []
        for i in range(len(scores)):
            if scores[i] > 0.5:
                anomalies.append(
                    self._calculate_strength_scores(i, data, components, scores[i])
                )

        return anomalies

    def _detect_kmeans(
            self,
            data: pd.Series,
            components: DecompositionResult
    ) -> List[TimeSeriesAnomaly]:
        """Detect anomalies using KMeans clustering."""
        # Extract features
        features = self._extract_features(data, components)

        # Initialize KMeans if needed
        if self.kmeans is None:
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42
            )

        # Fit the model
        self.kmeans.fit(features)

        # Get cluster assignments and distances
        labels = self.kmeans.labels_
        distances = cdist(features, self.kmeans.cluster_centers_)

        # Calculate point distances
        point_distances = np.array([
            distances[i, label]
            for i, label in enumerate(labels)
        ])

        # Calculate z-scores of distances
        distance_scores = (point_distances - np.mean(point_distances)) / np.std(point_distances)

        # Convert to anomaly probability
        anomaly_scores = 1 / (1 + np.exp(-distance_scores + self.threshold))

        # Find anomalies
        anomalies = []
        for i in range(len(anomaly_scores)):
            if anomaly_scores[i] > 0.5:
                anomalies.append(
                    self._calculate_strength_scores(i, data, components, anomaly_scores[i])
                )

        return anomalies

    def plot(self, data: pd.Series, anomalies: List[TimeSeriesAnomaly]):
        """Plot the time series with highlighted anomalies."""
        import matplotlib.pyplot as plt

        if self.method == DetectionMethod.PROPHET and self.prophet_result is not None:
            # Create Prophet-specific visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

            # Plot original data with forecast
            forecast = self.prophet_result.forecast
            ax1.plot(data.index, data.values, 'k.', label='Actual')
            ax1.plot(data.index, forecast['yhat'], 'b-', label='Forecast')
            ax1.fill_between(
                data.index,
                forecast['yhat_lower'],
                forecast['yhat_upper'],
                color='b',
                alpha=0.2,
                label='Prediction Interval'
            )

            # Plot anomalies with corrected color map names
            if anomalies:
                colors = {
                    'point': {'color': 'red', 'cmap': 'Reds'},
                    'seasonal': {'color': 'green', 'cmap': 'Greens'},
                    'trend': {'color': 'purple', 'cmap': 'PuRd'}
                }

                # Group anomalies by type
                anomalies_by_type = {'point': [], 'seasonal': [], 'trend': []}
                for a in anomalies:
                    anomalies_by_type[a.anomaly_type].append(a)

                # Plot each type
                for atype, anom_list in anomalies_by_type.items():
                    if anom_list:
                        times = [a.timestamp for a in anom_list]
                        values = [a.value for a in anom_list]
                        scores = [a.score for a in anom_list]
                        ax1.scatter(
                            times, values,
                            c=scores,
                            cmap=colors[atype]['cmap'],
                            label=f'{atype.title()} Anomalies'
                        )

                        # Add expected value markers for Prophet
                        for anomaly in anom_list:
                            if anomaly.expected_value is not None:
                                ax1.plot(
                                    [anomaly.timestamp, anomaly.timestamp],
                                    [anomaly.value, anomaly.expected_value],
                                    '--',
                                    color=colors[atype]['color'],
                                    alpha=0.5
                                )

            ax1.set_title('Time Series with Prophet Forecast and Anomalies')
            ax1.legend()

            # Plot changepoints
            changepoints = self.prophet_result.changepoints
            ax2.plot(data.index, data.values, 'k.', alpha=0.2)
            ax2.vlines(
                changepoints.index,
                data.min(),
                data.max(),
                colors='r',
                linestyles='dashed',
                label='Changepoints'
            )
            ax2.set_title('Detected Changepoints')
            ax2.legend()

        else:
            # Decompose for plotting
            components = self._decompose_series(data)

            # Create subplots
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))

            # Plot original data with anomalies
            axes[0].plot(data.index, data.values, label='Data')

            if anomalies:
                # Group anomalies by type
                anomalies_by_type = {'point': [], 'seasonal': [], 'trend': []}
                for a in anomalies:
                    anomalies_by_type[a.anomaly_type].append(a)

                # Plot each type with correct color maps
                colors = {
                    'point': {'color': 'red', 'cmap': 'Reds'},
                    'seasonal': {'color': 'green', 'cmap': 'Greens'},
                    'trend': {'color': 'purple', 'cmap': 'PuRd'}
                }

                for atype, anom_list in anomalies_by_type.items():
                    if anom_list:
                        times = [a.timestamp for a in anom_list]
                        values = [a.value for a in anom_list]
                        scores = [a.score for a in anom_list]
                        axes[0].scatter(
                            times, values,
                            c=scores,
                            cmap=colors[atype]['cmap'],
                            label=f'{atype.title()} Anomalies'
                        )

            axes[0].set_title(f'Original Data with Anomalies ({self.method.value})')
            axes[0].legend()

            # Plot components
            axes[1].plot(components.trend.index, components.trend.values, color='blue')
            axes[1].set_title('Trend Component')

            axes[2].plot(components.seasonal.index, components.seasonal.values, color='green')
            axes[2].set_title('Seasonal Component')

            axes[3].plot(components.residual.index, components.residual.values, color='red')
            axes[3].set_title('Residual Component')

        plt.tight_layout()
        plt.show()

        # Additional visualization for KMeans method
        if self.method == DetectionMethod.KMEANS and self.kmeans is not None:
            plt.figure(figsize=(10, 5))

            # Use first two features for visualization
            features = self._extract_features(data, components)
            plt.scatter(
                features.iloc[:, 0],
                features.iloc[:, 1],
                c=self.kmeans.labels_,
                cmap='viridis'
            )

            # Plot cluster centers
            centers = self.kmeans.cluster_centers_
            plt.scatter(
                centers[:, 0],
                centers[:, 1],
                c='red',
                marker='x',
                s=200,
                linewidths=3,
                label='Cluster Centers'
            )

            plt.title('KMeans Cluster Distribution')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.show()


# Example usage:
if __name__ == "__main__":
    # Generate sample data with seasonality
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    # Daily seasonality
    seasonal = 2 * np.sin(2 * np.pi * np.arange(1000) / 24)
    # Weekly seasonality
    seasonal += np.sin(2 * np.pi * np.arange(1000) / (24 * 7))
    # Trend
    trend = 0.01 * np.arange(1000)
    # Noise
    noise = np.random.normal(0, 0.1, 1000)

    # Combine components
    values = seasonal + trend + noise

    # Add some obvious anomalies
    values[100] = 5.0  # Point anomaly
    values[500:503] += 3.0  # Seasonal anomaly
    values[800:810] += np.linspace(0, 2, 10)  # Trend anomaly

    # Create time series
    data = pd.Series(values, index=dates)

    # Test all detection methods
    for method in DetectionMethod:
        print(f"\nTesting {method.value} method:")
        detector = TimeSeriesDetector(
            method=method,
            seasonal_period=24,  # Known daily seasonality
            n_clusters=5 if method == DetectionMethod.KMEANS else None
        )
        anomalies = detector.detect(data)

        print(f"Found {len(anomalies)} anomalies")
        print("First 5 anomalies:")
        for anomaly in anomalies[:5]:
            print(f"Timestamp: {anomaly.timestamp}")
            print(f"Score: {anomaly.score:.2f}")
            print(f"Type: {anomaly.anomaly_type}")
            print(f"Seasonal strength: {anomaly.seasonal_strength:.2f}")
            print(f"Trend strength: {anomaly.trend_strength:.2f}")
            print("---")

        # Plot results
        detector.plot(data, anomalies)