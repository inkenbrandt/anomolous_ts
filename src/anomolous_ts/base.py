import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Dict, List, Iterator, Callable, Any, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# Type Definitions
TimeSeriesData = Union[pd.Series, pd.DataFrame]
TimeWindow = Tuple[datetime, datetime]
MetricValue = Union[int, float]

# Define type variables
T = TypeVar('T', bound='BaseDetector')

class AnomalyScore:
    """Container for anomaly detection results."""

    def __init__(
            self,
            score: float,
            is_anomaly: bool,
            confidence: float,
            timestamp: datetime,
            contributing_features: Optional[Dict[str, float]] = None
    ):
        self.score = score
        self.is_anomaly = is_anomaly
        self.confidence = confidence
        self.timestamp = timestamp
        self.contributing_features = contributing_features or {}


class DetectionResult:
    """Container for detection results with metadata."""

    def __init__(
            self,
            scores: List[AnomalyScore],
            detector_name: str,
            detection_time: datetime,
            metadata: Optional[Dict[str, Any]] = None
    ):
        self.scores = scores
        self.detector_name = detector_name
        self.detection_time = detection_time
        self.metadata = metadata or {}


class BaseDetector(Generic[T], ABC):
    """Abstract base class for all anomaly detectors."""

    @abstractmethod
    def fit(self, data: TimeSeriesData) -> T:
        """Fit the detector to the data."""
        pass

    @abstractmethod
    def predict(self, data: TimeSeriesData) -> DetectionResult:
        """Predict anomalies in the data."""
        pass

    def fit_predict(self, data: TimeSeriesData) -> DetectionResult:
        """Convenience method to fit and predict in one step."""
        return self.fit(data).predict(data)

    @abstractmethod
    def score(self, data: TimeSeriesData) -> np.ndarray:
        """Get raw anomaly scores."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Get detector metadata."""
        pass


class StreamingDetector(BaseDetector[T], ABC):
    """Base class for streaming anomaly detectors."""

    @abstractmethod
    def process_stream(
            self,
            data_stream: Iterator[TimeSeriesData],
            callback: Optional[Callable[[DetectionResult], None]] = None
    ) -> Iterator[DetectionResult]:
        """Process streaming data."""
        pass

    @abstractmethod
    def update(self, new_data: TimeSeriesData) -> None:
        """Update detector with new data."""
        pass



class VotingStrategy(Enum):
    """Available voting strategies for ensemble."""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    THRESHOLD = "threshold"
    UNANIMOUS = "unanimous"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble detector."""
    voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED
    score_threshold: float = 0.6
    min_detectors_agree: int = 2
    weights: Optional[Dict[str, float]] = None
    dynamic_weights: bool = True
    adaptation_rate: float = 0.1

    # Performance tracking settings
    track_performance: bool = True
    performance_window: int = 1000

    def __post_init__(self):
        if self.weights is None:
            self.weights = {}

T = TypeVar('T', bound='EnsembleDetector')
class EnsembleDetector(BaseDetector[T]):
    """
    Enhanced ensemble anomaly detector that combines multiple detection algorithms.

    Features:
    - Multiple voting strategies
    - Dynamic weight adaptation
    - Performance tracking
    - Confidence scoring
    """

    def __init__(
            self,
            config: Optional[EnsembleConfig] = None,
            detectors: Optional[Dict[str, BaseDetector]] = None
    ):
        self.config = config or EnsembleConfig()
        self._detectors: Dict[str, BaseDetector] = detectors or {}
        self._performance_history: Dict[str, List[float]] = defaultdict(list)
        self._weights: Dict[str, float] = self.config.weights.copy()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize detector weights if not provided."""
        if not self._weights:
            n_detectors = len(self._detectors)
            if n_detectors > 0:
                default_weight = 1.0 / n_detectors
                self._weights = {
                    name: default_weight for name in self._detectors
                }

    def add_detector(self, name: str, detector: BaseDetector) -> None:
        """Add a new detector to the ensemble."""
        self._detectors[name] = detector
        if name not in self._weights:
            # Initialize weight for new detector
            self._weights[name] = 1.0 / (len(self._detectors))
            # Normalize weights
            self._normalize_weights()

    def remove_detector(self, name: str) -> None:
        """Remove a detector from the ensemble."""
        if name in self._detectors:
            self._detectors.pop(name)
            self._weights.pop(name, None)
            self._performance_history.pop(name, None)
            self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        if self._weights:
            total = sum(self._weights.values())
            if total > 0:
                self._weights = {
                    k: v / total for k, v in self._weights.items()
                }

    def _update_weights(self, performances: Dict[str, float]) -> None:
        """Update detector weights based on performance."""
        if not self.config.dynamic_weights:
            return

        for name, performance in performances.items():
            current_weight = self._weights.get(name, 0.0)
            # Update weight using exponential moving average
            self._weights[name] = (
                    (1 - self.config.adaptation_rate) * current_weight +
                    self.config.adaptation_rate * performance
            )

        self._normalize_weights()

    def _combine_scores(
            self,
            detector_results: Dict[str, DetectionResult]
    ) -> List[AnomalyScore]:
        """Combine scores from multiple detectors."""
        n_points = len(next(iter(detector_results.values())).scores)
        combined_scores = []

        for i in range(n_points):
            # Collect scores and votes from all detectors
            scores = {}
            votes = {}
            confidences = {}
            features = defaultdict(float)

            for detector_name, result in detector_results.items():
                score = result.scores[i]
                weight = self._weights.get(detector_name, 1.0)

                scores[detector_name] = score.score * weight
                votes[detector_name] = score.is_anomaly
                confidences[detector_name] = score.confidence * weight

                # Combine contributing features
                for feature, importance in score.contributing_features.items():
                    features[feature] += importance * weight

            # Calculate combined score based on voting strategy
            is_anomaly = self._apply_voting_strategy(votes, scores)

            # Calculate combined confidence
            confidence = np.mean(list(confidences.values()))

            # Create combined score
            combined_scores.append(AnomalyScore(
                score=np.mean(list(scores.values())),
                is_anomaly=is_anomaly,
                confidence=confidence,
                timestamp=next(iter(detector_results.values())).scores[i].timestamp,
                contributing_features=dict(features)
            ))

        return combined_scores

    def _apply_voting_strategy(
            self,
            votes: Dict[str, bool],
            scores: Dict[str, float]
    ) -> bool:
        """Apply the selected voting strategy."""
        if self.config.voting_strategy == VotingStrategy.MAJORITY:
            return sum(votes.values()) >= len(votes) / 2

        elif self.config.voting_strategy == VotingStrategy.WEIGHTED:
            weighted_votes = sum(
                self._weights.get(name, 1.0) * vote
                for name, vote in votes.items()
            )
            return weighted_votes >= self.config.score_threshold

        elif self.config.voting_strategy == VotingStrategy.THRESHOLD:
            return sum(votes.values()) >= self.config.min_detectors_agree

        elif self.config.voting_strategy == VotingStrategy.UNANIMOUS:
            return all(votes.values())

        raise ValueError(f"Unknown voting strategy: {self.config.voting_strategy}")

    def _evaluate_detector_performance(
            self,
            detector_results: Dict[str, DetectionResult]
    ) -> Dict[str, float]:
        """Evaluate performance of individual detectors."""
        performances = {}

        for name, result in detector_results.items():
            # Calculate performance metric (e.g., confidence-weighted accuracy)
            performance = np.mean([
                score.confidence if score.is_anomaly else (1 - score.confidence)
                for score in result.scores
            ])

            # Update performance history
            self._performance_history[name].append(performance)
            if len(self._performance_history[name]) > self.config.performance_window:
                self._performance_history[name].pop(0)

            # Calculate smoothed performance
            performances[name] = np.mean(self._performance_history[name])

        return performances

    def fit(self, data: TimeSeriesData) -> T:
        """Fit all detectors in the ensemble."""
        for detector in self._detectors.values():
            detector.fit(data)
        return self

    def predict(self, data: TimeSeriesData) -> DetectionResult:
        """Get predictions from all detectors and combine them."""
        # Get predictions from all detectors
        detector_results = {
            name: detector.predict(data)
            for name, detector in self._detectors.items()
        }

        # Evaluate detector performance
        if self.config.track_performance:
            performances = self._evaluate_detector_performance(detector_results)
            self._update_weights(performances)

        # Combine scores from all detectors
        combined_scores = self._combine_scores(detector_results)

        return DetectionResult(
            scores=combined_scores,
            detector_name="EnsembleDetector",
            detection_time=datetime.now(),
            metadata={
                "weights": self._weights,
                "voting_strategy": self.config.voting_strategy.value,
                "n_detectors": len(self._detectors),
                "detector_names": list(self._detectors.keys())
            }
        )

    def score(self, data: TimeSeriesData) -> np.ndarray:
        """Get raw anomaly scores from the ensemble."""
        detector_scores = np.array([
            detector.score(data) * self._weights.get(name, 1.0)
            for name, detector in self._detectors.items()
        ])
        return np.mean(detector_scores, axis=0)

    def get_detector_weights(self) -> Dict[str, float]:
        """Get current detector weights."""
        return self._weights.copy()

    def get_performance_history(self) -> Dict[str, List[float]]:
        """Get performance history for all detectors."""
        return {
            name: history.copy()
            for name, history in self._performance_history.items()
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get detector metadata."""
        return {
            "name": "EnsembleDetector",
            "version": "1.0",
            "config": {
                "voting_strategy": self.config.voting_strategy.value,
                "score_threshold": self.config.score_threshold,
                "min_detectors_agree": self.config.min_detectors_agree,
                "dynamic_weights": self.config.dynamic_weights,
                "adaptation_rate": self.config.adaptation_rate
            },
            "detectors": {
                name: detector.metadata
                for name, detector in self._detectors.items()
            },
            "weights": self._weights,
            "n_detectors": len(self._detectors)
        }

class DetectionStrategy(Enum):
    """Enumeration of available detection strategies."""
    ISOLATION_FOREST = "isolation_forest"
    KMEANS = "kmeans"
    SHESD = "shesd"
    ENSEMBLE = "ensemble"
    DEEP = "deep_learning"


@dataclass
class DetectorMetrics:
    """Container for detector performance metrics."""
    fit_time: float = 0.0
    prediction_time: float = 0.0
    memory_usage: float = 0.0
    num_anomalies: int = 0
    average_score: float = 0.0
    additional_metrics: Dict = field(default_factory=dict)


class AnomalyAPI:
    """Main API interface for anomaly detection."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._detectors: Dict[str, BaseDetector] = {}
        self._preprocessors: Dict[str, Any] = {}
        self._metrics: Dict[str, List[DetectorMetrics]] = {}

    def register_detector(
            self,
            name: str,
            detector: BaseDetector,
            preprocessor: Optional[Any] = None
    ) -> None:
        """Register a new detector."""
        self._detectors[name] = detector
        if preprocessor:
            self._preprocessors[name] = preprocessor
        self._metrics[name] = []

    def remove_detector(self, name: str) -> None:
        """Remove a registered detector."""
        self._detectors.pop(name, None)
        self._preprocessors.pop(name, None)
        self._metrics.pop(name, None)

    def detect(
            self,
            data: TimeSeriesData,
            detector_name: str,
            **kwargs
    ) -> DetectionResult:
        """Detect anomalies using specified detector."""
        if detector_name not in self._detectors:
            raise ValueError(f"Detector {detector_name} not found")

        detector = self._detectors[detector_name]
        preprocessor = self._preprocessors.get(detector_name)

        if preprocessor:
            data = preprocessor.transform(data)

        result = detector.fit_predict(data)
        return result

    def detect_all(
            self,
            data: TimeSeriesData,
            **kwargs
    ) -> Dict[str, DetectionResult]:
        """Run all registered detectors."""
        results = {}
        for name in self._detectors:
            results[name] = self.detect(data, name, **kwargs)
        return results

    def get_detector_metrics(
            self,
            detector_name: str
    ) -> List[DetectorMetrics]:
        """Get performance metrics for a detector."""
        return self._metrics.get(detector_name, [])


class APIClient:
    """Client for interacting with the anomaly detection API."""

    def __init__(self, api_url: Optional[str] = None):
        self.api = AnomalyAPI()
        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initialize default detectors."""
        for strategy in DetectionStrategy:
            self._create_detector(strategy)

    def _create_detector(self, strategy: DetectionStrategy) -> BaseDetector:
        """Factory method to create detectors."""
        # Implementation for creating specific detectors
        pass

    def detect_anomalies(
            self,
            data: TimeSeriesData,
            strategy: Union[str, DetectionStrategy] = DetectionStrategy.ENSEMBLE,
            **kwargs
    ) -> DetectionResult:
        """High-level method to detect anomalies."""
        if isinstance(strategy, str):
            strategy = DetectionStrategy(strategy)

        return self.api.detect(data, strategy.value, **kwargs)

    def get_available_strategies(self) -> List[str]:
        """Get list of available detection strategies."""
        return [strategy.value for strategy in DetectionStrategy]

