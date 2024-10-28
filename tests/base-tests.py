
from typing import TypeVar, Generic, Optional, Dict, List, Iterator, Callable, Any, Union, Tuple

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import series, data_frames

from anomolous_ts.base import (
    AnomalyScore, DetectionResult, BaseDetector, StreamingDetector,
    VotingStrategy, EnsembleConfig, EnsembleDetector, DetectionStrategy,
    DetectorMetrics, AnomalyAPI, APIClient, TimeSeriesData
)

class TestAnomalyScore:
    def test_initialization(self):
        """Test AnomalyScore initialization with all parameters."""
        timestamp = datetime.now()
        score = AnomalyScore(
            score=0.8,
            is_anomaly=True,
            confidence=0.9,
            timestamp=timestamp,
            contributing_features={"feature1": 0.5, "feature2": 0.3}
        )
        
        assert score.score == 0.8
        assert score.is_anomaly is True
        assert score.confidence == 0.9
        assert score.timestamp == timestamp
        assert score.contributing_features == {"feature1": 0.5, "feature2": 0.3}

    def test_default_contributing_features(self):
        """Test AnomalyScore initialization without contributing_features."""
        score = AnomalyScore(
            score=0.5,
            is_anomaly=False,
            confidence=0.7,
            timestamp=datetime.now()
        )
        
        assert isinstance(score.contributing_features, dict)
        assert len(score.contributing_features) == 0

class TestDetectionResult:
    @pytest.fixture
    def sample_scores(self):
        """Create sample anomaly scores for testing."""
        timestamp = datetime.now()
        return [
            AnomalyScore(0.8, True, 0.9, timestamp),
            AnomalyScore(0.3, False, 0.8, timestamp)
        ]

    def test_initialization(self, sample_scores):
        """Test DetectionResult initialization with all parameters."""
        detection_time = datetime.now()
        metadata = {"threshold": 0.7, "window_size": 10}
        
        result = DetectionResult(
            scores=sample_scores,
            detector_name="TestDetector",
            detection_time=detection_time,
            metadata=metadata
        )
        
        assert result.scores == sample_scores
        assert result.detector_name == "TestDetector"
        assert result.detection_time == detection_time
        assert result.metadata == metadata

    def test_default_metadata(self, sample_scores):
        """Test DetectionResult initialization without metadata."""
        result = DetectionResult(
            scores=sample_scores,
            detector_name="TestDetector",
            detection_time=datetime.now()
        )
        
        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0


class MockDetector(BaseDetector):
    """Mock detector for testing."""

    def __init__(self):
        self._fitted = False

    def fit(self, data: TimeSeriesData) -> 'MockDetector':
        self._fitted = True
        return self

    def predict(self, data: TimeSeriesData) -> DetectionResult:
        """Return a score for each data point."""
        if isinstance(data, (pd.Series, pd.DataFrame)):
            n_points = len(data)
        else:
            n_points = len(np.asarray(data))

        scores = [
            AnomalyScore(
                score=0.5,
                is_anomaly=False,
                confidence=0.8,
                timestamp=datetime.now() + timedelta(seconds=i)
            )
            for i in range(n_points)
        ]

        return DetectionResult(
            scores=scores,
            detector_name="MockDetector",
            detection_time=datetime.now()
        )

    def score(self, data: TimeSeriesData) -> np.ndarray:
        """Return a score array matching input length."""
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return np.full(len(data), 0.5)
        return np.full(len(np.asarray(data)), 0.5)

    @property
    def metadata(self) -> dict:
        return {"name": "MockDetector", "fitted": self._fitted}

class TestBaseDetector:
    @pytest.fixture
    def detector(self):
        return MockDetector()

    @pytest.fixture
    def sample_data(self):
        return pd.Series([1, 2, 3, 4, 5])

    def test_fit_predict(self, detector, sample_data):
        """Test fit_predict convenience method."""
        result = detector.fit_predict(sample_data)
        
        assert isinstance(result, DetectionResult)
        assert detector._fitted
        assert len(result.scores) > 0

    def test_metadata_property(self, detector):
        """Test metadata property returns expected format."""
        metadata = detector.metadata
        
        assert isinstance(metadata, dict)
        assert "name" in metadata
        assert "fitted" in metadata

class MockStreamingDetector(StreamingDetector):
    """Mock streaming detector for testing."""
    def __init__(self):
        self.base_detector = MockDetector()
        
    def fit(self, data: TimeSeriesData) -> 'MockStreamingDetector':
        self.base_detector.fit(data)
        return self
        
    def predict(self, data: TimeSeriesData) -> DetectionResult:
        return self.base_detector.predict(data)
        
    def score(self, data: TimeSeriesData) -> np.ndarray:
        return self.base_detector.score(data)
        
    def process_stream(
        self,
        data_stream: Iterator[TimeSeriesData],
        callback: Optional[Callable] = None
    ) -> Iterator[DetectionResult]:
        for data in data_stream:
            result = self.predict(data)
            if callback:
                callback(result)
            yield result
            
    def update(self, new_data: TimeSeriesData) -> None:
        self.base_detector.fit(new_data)
        
    @property
    def metadata(self) -> dict:
        return {**self.base_detector.metadata, "type": "streaming"}

class TestStreamingDetector:
    @pytest.fixture
    def detector(self):
        return MockStreamingDetector()

    def test_streaming_processing(self, detector):
        """Test stream processing with callback."""
        callback_mock = Mock()
        data_stream = (pd.Series([i]) for i in range(5))
        
        results = list(detector.process_stream(data_stream, callback_mock))
        
        assert len(results) == 5
        assert callback_mock.call_count == 5
        assert all(isinstance(r, DetectionResult) for r in results)

    def test_update_method(self, detector):
        """Test detector update with new data."""
        new_data = pd.Series([1, 2, 3])
        detector.update(new_data)
        assert detector.base_detector._fitted

class TestEnsembleConfig:
    def test_default_initialization(self):
        """Test EnsembleConfig initialization with defaults."""
        config = EnsembleConfig()
        
        assert config.voting_strategy == VotingStrategy.WEIGHTED
        assert config.score_threshold == 0.6
        assert config.min_detectors_agree == 2
        assert config.weights == {}
        assert config.dynamic_weights is True
        assert config.adaptation_rate == 0.1

    def test_custom_initialization(self):
        """Test EnsembleConfig with custom parameters."""
        weights = {"detector1": 0.7, "detector2": 0.3}
        config = EnsembleConfig(
            voting_strategy=VotingStrategy.MAJORITY,
            score_threshold=0.8,
            weights=weights,
            dynamic_weights=False
        )
        
        assert config.voting_strategy == VotingStrategy.MAJORITY
        assert config.score_threshold == 0.8
        assert config.weights == weights
        assert config.dynamic_weights is False


class TestEnsembleDetector:
    @pytest.fixture
    def detectors(self):
        return {
            "detector1": MockDetector(),
            "detector2": MockDetector()
        }

    @pytest.fixture
    def ensemble(self, detectors):
        # Pre-fit the detectors
        data = pd.Series([1, 2, 3])
        for detector in detectors.values():
            detector.fit(data)

        return EnsembleDetector(
            config=EnsembleConfig(voting_strategy=VotingStrategy.MAJORITY),
            detectors=detectors
        )

    def test_combine_scores(self, ensemble):
        """Test score combination from multiple detectors."""
        data = pd.Series([1, 2, 3])
        result = ensemble.predict(data)

        # Verify result length matches input
        assert len(result.scores) == len(data)

        # Verify score properties
        for score in result.scores:
            assert isinstance(score, AnomalyScore)
            assert 0 <= score.score <= 1
            assert isinstance(score.is_anomaly, bool)
            assert 0 <= score.confidence <= 1
            assert isinstance(score.timestamp, datetime)
            assert isinstance(score.contributing_features, dict)

    def test_combine_scores_with_different_detectors(self, ensemble):
        """Test score combination with detectors returning different scores."""
        data = pd.Series([1, 2, 3])

        # Override one detector to return different scores
        class MockDetectorAlternate(MockDetector):
            def predict(self, data: TimeSeriesData) -> DetectionResult:
                n_points = len(data)
                scores = [
                    AnomalyScore(
                        score=0.9,
                        is_anomaly=True,
                        confidence=0.9,
                        timestamp=datetime.now() + timedelta(seconds=i),
                        contributing_features={"feature1": 0.8}
                    )
                    for i in range(n_points)
                ]
                return DetectionResult(
                    scores=scores,
                    detector_name="MockDetectorAlternate",
                    detection_time=datetime.now()
                )

        ensemble._detectors["detector2"] = MockDetectorAlternate()
        result = ensemble.predict(data)

        assert len(result.scores) == len(data)

        # Verify scores are being combined
        for score in result.scores:
            assert 0.5 <= score.score <= 0.9  # Between the two detectors' scores
            assert isinstance(score.contributing_features, dict)
            if score.contributing_features:
                assert "feature1" in score.contributing_features

    def test_combine_scores_with_weights(self, ensemble):
        """Test score combination with different detector weights."""
        data = pd.Series([1, 2, 3])

        # Set different weights for detectors
        ensemble._weights = {
            "detector1": 0.7,
            "detector2": 0.3
        }

        result = ensemble.predict(data)

        assert len(result.scores) == len(data)
        for score in result.scores:
            assert 0 <= score.score <= 1
            assert 0 <= score.confidence <= 1

class TestDetectorMetrics:
    def test_initialization(self):
        """Test DetectorMetrics initialization."""
        metrics = DetectorMetrics(
            fit_time=1.5,
            prediction_time=0.5,
            memory_usage=100.0,
            num_anomalies=10,
            average_score=0.7
        )
        
        assert metrics.fit_time == 1.5
        assert metrics.prediction_time == 0.5
        assert metrics.memory_usage == 100.0
        assert metrics.num_anomalies == 10
        assert metrics.average_score == 0.7
        assert isinstance(metrics.additional_metrics, dict)

class TestAnomalyAPI:
    @pytest.fixture
    def api(self):
        return AnomalyAPI()

    def test_register_remove_detector(self, api):
        """Test registering and removing detectors."""
        detector = MockDetector()
        
        api.register_detector("test_detector", detector)
        assert "test_detector" in api._detectors
        
        api.remove_detector("test_detector")
        assert "test_detector" not in api._detectors

    def test_detect_method(self, api):
        """Test anomaly detection with registered detector."""
        detector = MockDetector()
        api.register_detector("test_detector", detector)
        
        data = pd.Series([1, 2, 3])
        result = api.detect(data, "test_detector")
        
        assert isinstance(result, DetectionResult)

    def test_detect_all(self, api):
        """Test running all registered detectors."""
        api.register_detector("detector1", MockDetector())
        api.register_detector("detector2", MockDetector())
        
        data = pd.Series([1, 2, 3])
        results = api.detect_all(data)
        
        assert len(results) == 2
        assert all(isinstance(r, DetectionResult) for r in results.values())

class TestAPIClient:
    @pytest.fixture
    def client(self):
        return APIClient()

    def test_available_strategies(self, client):
        """Test getting available detection strategies."""
        strategies = client.get_available_strategies()
        assert len(strategies) == len(DetectionStrategy)
        assert all(isinstance(s, str) for s in strategies)

    def test_strategy_conversion(self, client):
        """Test strategy conversion from string."""
        with patch.object(client.api, 'detect') as mock_detect:
            data = pd.Series([1, 2, 3])
            client.detect_anomalies(data, strategy="isolation_forest")
            
            mock_detect.assert_called_once()
            strategy_arg = mock_detect.call_args[0][1]
            assert strategy_arg == DetectionStrategy.ISOLATION_FOREST.value

if __name__ == '__main__':
    pytest.main([__file__])
