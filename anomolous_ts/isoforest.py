from typing import Optional, Dict, Any, List, Iterator, Callable, TypeVar, Union
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector, StreamingDetector, DetectionResult, AnomalyScore, TimeSeriesData
from .feature_extractors import (
    FeatureExtractorPipeline,
    FeatureExtractorConfig,
    StatisticalFeatureExtractor,
    SpectralFeatureExtractor,
    WaveletFeatureExtractor
)

T = TypeVar('T', bound='TimeSeriesIsolationForest')

class IsolationForestConfig:
    """Configuration for Isolation Forest detector."""
    def __init__(
        self,
        window_size: int = 5,
        n_estimators: int = 100,
        contamination: Union[str, float] = 'auto',
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int = -1,
        seasonal_period: Optional[int] = None,
        random_state: int = 42,
        feature_settings: Dict[str, bool] = None
    ):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.seasonal_period = seasonal_period
        self.random_state = random_state
        self.feature_settings = feature_settings or {
            'statistical': True,
            'spectral': False,
            'wavelet': False
        }


class TimeSeriesIsolationForest(BaseDetector[T]):
    """Enhanced Isolation Forest for time series anomaly detection."""

    def _setup_feature_extractors(self) -> None:
        """Initialize feature extractors based on configuration."""
        extractor_config = FeatureExtractorConfig(
            window_sizes=[self.config.window_size],
            wavelet_level=3
        )

        if self.config.feature_settings.get('statistical', True):
            self._feature_extractors.append(
                StatisticalFeatureExtractor(extractor_config)
            )
        if self.config.feature_settings.get('spectral', False):
            self._feature_extractors.append(
                SpectralFeatureExtractor(extractor_config)
            )
        if self.config.feature_settings.get('wavelet', False):
            self._feature_extractors.append(
                WaveletFeatureExtractor(extractor_config)
            )

    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract all features and ensure consistent dimensions."""
        all_features = []

        for extractor in self._feature_extractors:
            features = extractor.extract(data)
            for feature_values in features.values():
                # Ensure all feature arrays have the same length as data
                if len(feature_values) < len(data):
                    padded = np.pad(
                        feature_values,
                        (0, len(data) - len(feature_values)),
                        mode='edge'
                    )
                    all_features.append(padded)
                elif len(feature_values) > len(data):
                    all_features.append(feature_values[:len(data)])
                else:
                    all_features.append(feature_values)

        # Stack features and handle any NaN values
        feature_matrix = np.column_stack(all_features)

        # Replace NaN values with feature means
        if np.any(np.isnan(feature_matrix)):
            feature_means = np.nanmean(feature_matrix, axis=0)
            for i in range(feature_matrix.shape[1]):
                mask = np.isnan(feature_matrix[:, i])
                feature_matrix[mask, i] = feature_means[i]

        return feature_matrix

    def fit(self, data: TimeSeriesData) -> T:
        """Fit the detector to the data."""
        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)

        # Convert to numpy array if needed
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data_values = data.values
        else:
            data_values = np.asarray(data)

        # Extract features and handle missing values
        features = self._extract_features(data_values)

        # Handle any remaining edge cases
        features = np.nan_to_num(features, nan=0.0)

        # Initialize and fit scaler
        self._feature_scaler = StandardScaler()
        scaled_features = self._feature_scaler.fit_transform(features)

        # Fit Isolation Forest
        self._model.fit(scaled_features)
        self._decision_scores = -self._model.score_samples(scaled_features)

        return self

    def predict(self, data: TimeSeriesData) -> DetectionResult:
        """Predict anomalies in the data."""
        if self._feature_scaler is None:
            raise ValueError("Detector must be fitted before prediction")

        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)

        # Convert to numpy array
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data_values = data.values
        else:
            data_values = np.asarray(data)

        # Extract and process features
        features = self._extract_features(data_values)


    def _get_feature_contributions(
        self,
        features: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature contributions to anomaly score."""
        # Calculate absolute deviations from mean
        mean_features = np.mean(features)
        deviations = np.abs(features - mean_features)
        total_deviation = np.sum(deviations)
        
        if total_deviation == 0:
            return {}
            
        contributions = {
            f"feature_{i}": float(d / total_deviation)
            for i, d in enumerate(deviations)
            if d / total_deviation > 0.1
        }
        
        return contributions

    def score(self, data: TimeSeriesData) -> np.ndarray:
        """Get raw anomaly scores."""
        if self._feature_scaler is None:
            raise ValueError("Detector must be fitted before scoring")
            
        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)
            
        features = self._extract_features(
            data.values if isinstance(data, (pd.Series, pd.DataFrame)) else data
        )
        scaled_features = self._feature_scaler.transform(features)
        
        return -self._model.score_samples(scaled_features)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get detector metadata."""
        return {
            "name": "TimeSeriesIsolationForest",
            "version": "2.0",
            "config": {
                "n_estimators": self.config.n_estimators,
                "window_size": self.config.window_size,
                "contamination": self.config.contamination,
                "max_features": self.config.max_features,
                "bootstrap": self.config.bootstrap,
                "n_jobs": self.config.n_jobs,
                "seasonal_period": self.config.seasonal_period,
                "feature_settings": self.config.feature_settings
            },
            "feature_extractors": [type(ex).__name__ for ex in self._feature_extractors],
            "fitted": self._feature_scaler is not None
        }

class StreamingIsolationForest(StreamingDetector[T]):
    """Streaming version of the Isolation Forest detector."""
    
    def __init__(
        self,
        config: Optional[IsolationForestConfig] = None,
        preprocessor: Optional[Any] = None,
        update_interval: int = 1000
    ):
        """Initialize the streaming detector."""
        self.base_detector = TimeSeriesIsolationForest(config, preprocessor)
        self.update_interval = update_interval
        self._buffer: List[TimeSeriesData] = []
        self._n_processed = 0

    def fit(self, data: TimeSeriesData) -> T:
        """Initial fit of the detector."""
        self.base_detector.fit(data)
        return self

    def predict(self, data: TimeSeriesData) -> DetectionResult:
        """Predict using the current model state."""
        return self.base_detector.predict(data)

    def score(self, data: TimeSeriesData) -> np.ndarray:
        """Get anomaly scores using the current model state."""
        return self.base_detector.score(data)

    def process_stream(
        self,
        data_stream: Iterator[TimeSeriesData],
        callback: Optional[Callable[[DetectionResult], None]] = None
    ) -> Iterator[DetectionResult]:
        """Process streaming data."""
        for data_point in data_stream:
            self._buffer.append(data_point)
            self._n_processed += 1
            
            if len(self._buffer) >= self.base_detector.config.window_size:
                # Create window of data
                window_data = pd.concat(
                    self._buffer[-self.base_detector.config.window_size:]
                )
                
                # Get predictions
                result = self.predict(window_data)
                
                if callback:
                    callback(result)
                    
                yield result
                
                # Update model if needed
                if self._n_processed >= self.update_interval:
                    self.update(pd.concat(self._buffer))
                    self._n_processed = 0
                    self._buffer = self._buffer[-self.base_detector.config.window_size:]

    def update(self, new_data: TimeSeriesData) -> None:
        """Update the model with new data."""
        self.base_detector.fit(new_data)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get detector metadata."""
        return {
            **self.base_detector.metadata,
            "type": "streaming",
            "update_interval": self.update_interval,
            "buffer_size": len(self._buffer),
            "n_processed": self._n_processed
        }
