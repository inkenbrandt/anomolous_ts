from typing import Dict, List, Optional, Any, Tuple, Iterator, Callable, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from dataclasses import dataclass, field

from typing import Optional, Dict, Any, List, Iterator, Callable, TypeVar
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


T = TypeVar('T', bound='TimeSeriesKMeansDetector')


from .feature_extractors import (
    StatisticalFeatureExtractor, SpectralFeatureExtractor, WaveletFeatureExtractor, PreprocessorProtocol,FeatureExtractorConfig,
)

from .base import (
    BaseDetector,
    StreamingDetector,
    DetectionResult,
    AnomalyScore,
    TimeSeriesData,
)

@dataclass
class KMeansConfig:
    """Configuration for KMeans detector."""
    n_clusters: int = 3
    window_size: int = 10
    anomaly_threshold: float = 2.0
    seasonal_period: Optional[int] = None
    random_state: int = 42
    feature_settings: Dict[str, Any] = field(default_factory=lambda: {
        'statistical': True,
        'spectral': False,
        'wavelet': False
    })
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)


class TimeSeriesKMeansDetector(BaseDetector[T]):
    """Enhanced KMeans-based anomaly detector for time series data."""

    def __init__(
            self,
            config: Optional[KMeansConfig] = None,
            preprocessor: Optional[PreprocessorProtocol] = None
    ):
        self.config = config or KMeansConfig()
        self.preprocessor = preprocessor

        # Initialize KMeans model
        self._model = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state
        )

        # Initialize feature extractors
        self._feature_extractors = []
        self._setup_feature_extractors()

        # Internal state
        self._cluster_centers: Optional[np.ndarray] = None
        self._cluster_distances: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values in feature matrix."""
        if np.any(np.isnan(data)):
            # Forward fill, then backward fill
            filled_data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill')
            # If any NaNs remain (e.g., at start/end), replace with zeros
            return filled_data.fillna(0).values
        return data

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
        """Extract all features and handle missing values."""
        all_features = []
        data = self._handle_missing_values(data)

        for extractor in self._feature_extractors:
            features = extractor.extract(data)
            for name, feature_values in features.items():
                # Handle potential NaN values in features
                feature_values = np.nan_to_num(feature_values, nan=0.0)

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

        # Stack features and handle any remaining NaNs
        feature_matrix = np.column_stack(all_features)
        return self._handle_missing_values(feature_matrix)

    def fit(self, data: TimeSeriesData) -> T:
        """Fit the detector to the data."""
        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)

        # Convert to numpy array if needed
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data_values = data.values
        else:
            data_values = np.asarray(data)

        # Extract features and ensure they're 2D
        features = self._extract_features(data_values)

        # Fit KMeans model
        self._labels = self._model.fit_predict(features)
        self._cluster_centers = self._model.cluster_centers_

        # Calculate cluster distances
        self._cluster_distances = np.min(
            cdist(features, self._cluster_centers),
            axis=1
        )

        return self

    def predict(self, data: TimeSeriesData) -> DetectionResult:
        """Predict anomalies in the data."""
        if self._cluster_centers is None:
            raise ValueError("Detector must be fitted before prediction")

        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)

        # Convert to numpy array
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data_values = data.values
        else:
            data_values = np.asarray(data)

        # Extract features
        features = self._extract_features(data_values)

        # Calculate distances and scores
        distances = np.min(cdist(features, self._cluster_centers), axis=1)
        z_scores = stats.zscore(distances)
        scores = 1 / (1 + np.exp(-z_scores))  # Sigmoid transformation

        # Create anomaly scores
        anomaly_scores = []
        for i, score in enumerate(scores):
            timestamp = data.index[i] if hasattr(data, 'index') else datetime.now()
            is_anomaly = score > self.config.anomaly_threshold

            anomaly_scores.append(AnomalyScore(
                score=float(score),
                is_anomaly=is_anomaly,
                confidence=float(abs(score - 0.5) * 2),
                timestamp=timestamp,
                contributing_features=self._get_feature_contributions(
                    features[i],
                    self._model.predict([features[i]])[0]
                )
            ))

        return DetectionResult(
            scores=anomaly_scores,
            detector_name="KMeansDetector",
            detection_time=datetime.now(),
            metadata={
                "n_clusters": self.config.n_clusters,
                "window_size": self.config.window_size,
                "threshold": self.config.anomaly_threshold,
                "cluster_sizes": np.bincount(self._labels).tolist()
            }
        )

    def _get_feature_contributions(
            self,
            features: np.ndarray,
            cluster_label: int
    ) -> Dict[str, float]:
        """Calculate feature contributions to anomaly score."""
        center = self._cluster_centers[cluster_label]
        distances = np.abs(features - center)
        total_distance = np.sum(distances)

        if total_distance == 0:
            return {}

        contributions = {
            f"feature_{i}": float(d / total_distance)
            for i, d in enumerate(distances)
            if d / total_distance > 0.1
        }

        return contributions

    def score(self, data: TimeSeriesData) -> np.ndarray:
        """Get raw anomaly scores."""
        if self._cluster_centers is None:
            raise ValueError("Detector must be fitted before scoring")

        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)

        features = self._extract_features(
            data.values if isinstance(data, (pd.Series, pd.DataFrame)) else data
        )

        distances = np.min(cdist(features, self._cluster_centers), axis=1)
        return stats.zscore(distances)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get detector metadata."""
        return {
            "name": "TimeSeriesKMeansDetector",
            "version": "2.0",
            "config": self.config.__dict__,
            "n_clusters": self.config.n_clusters,
            "feature_extractors": [type(ex).__name__ for ex in self._feature_extractors],
            "fitted": self._cluster_centers is not None
        }

class StreamingKMeansDetector(StreamingDetector[T]):
    """Streaming version of the KMeans detector."""

    def __init__(
            self,
            config: Optional[KMeansConfig] = None,
            preprocessor: Optional[PreprocessorProtocol] = None,
            update_interval: int = 1000
    ):
        self.base_detector = TimeSeriesKMeansDetector(config, preprocessor)
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
        """
        Process streaming data.

        Parameters
        ----------
        data_stream : Iterator[TimeSeriesData]
            Stream of time series data
        callback : Optional[Callable]
            Optional callback function for results

        Yields
        ------
        DetectionResult
            Detection results for each processed chunk
        """
        for data_point in data_stream:
            self._buffer.append(data_point)
            self._n_processed += 1

            if len(self._buffer) >= self.base_detector.config.window_size:
                # Create window of data
                window_data = pd.concat(self._buffer[-self.base_detector.config.window_size:])

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


class DetectionResultEnhanced(DetectionResult):
    """Enhanced DetectionResult with additional methods."""

    def get_anomalies(self) -> List[AnomalyScore]:
        """Get only the anomalous scores."""
        return [score for score in self.scores if score.is_anomaly]