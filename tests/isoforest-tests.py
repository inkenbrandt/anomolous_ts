import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import series, data_frames
from hypothesis.extra.numpy import arrays
from typing import Optional, Dict, Any, List, Iterator, Callable, TypeVar, Union
from anomolous_ts.base import BaseDetector
from anomolous_ts.isoforest import (
    TimeSeriesIsolationForest,
    StreamingIsolationForest,
    IsolationForestConfig,
    DetectionResult,
    AnomalyScore
)

class TestIsolationForestConfig:
    """Tests for IsolationForestConfig class."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = IsolationForestConfig()
        
        assert config.window_size == 5
        assert config.n_estimators == 100
        assert config.contamination == 'auto'
        assert config.max_features == 1.0
        assert config.bootstrap is False
        assert config.n_jobs == -1
        assert config.seasonal_period is None
        assert config.random_state == 42
        assert config.feature_settings['statistical'] is True
        assert config.feature_settings['spectral'] is False
        assert config.feature_settings['wavelet'] is False

    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = IsolationForestConfig(
            window_size=10,
            n_estimators=200,
            contamination=0.1,
            seasonal_period=24,
            feature_settings={
                'statistical': True,
                'spectral': True,
                'wavelet': True
            }
        )
        
        assert config.window_size == 10
        assert config.n_estimators == 200
        assert config.contamination == 0.1
        assert config.seasonal_period == 24
        assert all(config.feature_settings.values())

    def test_invalid_contamination(self):
        """Test invalid contamination values."""
        with pytest.raises(ValueError):
            IsolationForestConfig(contamination=1.5)
        with pytest.raises(ValueError):
            IsolationForestConfig(contamination=-0.1)


class TimeSeriesIsolationForest(BaseDetector['TimeSeriesIsolationForest']):
    """Enhanced Isolation Forest for time series anomaly detection."""

    def __init__(
            self,
            config: Optional[IsolationForestConfig] = None,
            preprocessor: Optional[Any] = None
    ) -> None:
        """
        Initialize the Isolation Forest detector.

        Parameters
        ----------
        config : Optional[IsolationForestConfig]
            Configuration object for the detector. If None, uses default config.
        preprocessor : Optional[Any]
            Optional preprocessor for the data
        """
        super().__init__()
        self.config = config if config is not None else IsolationForestConfig()
        self.preprocessor = preprocessor

        # Initialize Isolation Forest model
        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state
        )

        # Initialize feature extractors
        self._feature_extractors = []
        self._setup_feature_extractors()

        # Internal state
        self._feature_scaler: Optional[StandardScaler] = None
        self._decision_scores: Optional[np.ndarray] = None
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing."""
        config = IsolationForestConfig(
            window_size=5,
            n_estimators=100,
            contamination='auto',
            feature_settings={
                'statistical': True,
                'spectral': False,
                'wavelet': False
            }
        )
        return TimeSeriesIsolationForest(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data with potential edge cases."""
        np.random.seed(42)  # For reproducibility
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        # Generate base signal
        t = np.linspace(0, 10 * np.pi, 100)
        values = np.sin(t) + 0.1 * np.random.randn(100)

        # Add some edge cases
        values[10] = np.nan  # Missing value
        values[20] = 10.0  # Outlier
        values[30:35] = 0.0  # Constant sequence

        return pd.Series(values, index=dates)

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.config.n_estimators == 100
        assert detector.config.window_size == 5
        assert detector._feature_scaler is None
        assert detector._decision_scores is None
        assert len(detector._feature_extractors) > 0

    def test_feature_extraction(self, detector, sample_data):
        """Test feature extraction functionality."""
        features = detector._extract_features(sample_data.values)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_data)
        assert features.shape[1] > 0
        # Verify no NaN values in output
        assert not np.isnan(features).any()
        # Verify finite values
        assert np.isfinite(features).all()

    def test_feature_contributions(self, detector, sample_data):
        """Test feature contribution calculation."""
        # Fill NaN values in input data
        clean_data = sample_data.fillna(method='ffill').fillna(method='bfill')

        detector.fit(clean_data)
        result = detector.predict(clean_data)

        # Find first anomaly
        anomaly_scores = [s for s in result.scores if s.is_anomaly]
        assert len(anomaly_scores) > 0

        score = anomaly_scores[0]
        contributions = score.contributing_features

        assert isinstance(contributions, dict)
        assert len(contributions) > 0
        assert all(isinstance(k, str) for k in contributions.keys())
        assert all(isinstance(v, float) for v in contributions.values())
        assert all(0 <= v <= 1 for v in contributions.values())
        assert abs(sum(contributions.values()) - 1.0) < 1e-10

    def test_edge_cases(self, detector):
        """Test handling of edge cases."""
        # Test with all constant values
        constant_data = pd.Series(np.ones(100))
        detector.fit(constant_data)
        result = detector.predict(constant_data)
        assert len(result.scores) == len(constant_data)

        # Test with single value
        single_value = pd.Series([1.0])
        detector.fit(single_value)
        result = detector.predict(single_value)
        assert len(result.scores) == 1

        # Test with missing values
        data_with_nans = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        detector.fit(data_with_nans)
        result = detector.predict(data_with_nans)
        assert len(result.scores) == len(data_with_nans)

    def test_multivariate_data(self, detector):
        """Test with multivariate data."""
        # Create multivariate time series
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'A': np.sin(np.linspace(0, 10 * np.pi, 100)),
            'B': np.cos(np.linspace(0, 10 * np.pi, 100))
        }, index=dates)

        # Add some anomalies
        df.iloc[20, 0] = 10.0  # Anomaly in first series
        df.iloc[50, 1] = -10.0  # Anomaly in second series

        detector.fit(df)
        result = detector.predict(df)

        assert len(result.scores) == len(df)
        assert hasattr(result.scores[0], 'contributing_features')
        # Should detect at least the obvious anomalies
        anomalies = [i for i, score in enumerate(result.scores) if score.is_anomaly]
        assert len(anomalies) >= 2

    def test_prediction_without_fit(self, detector, sample_data):
        """Test error handling when predicting without fitting."""
        with pytest.raises(ValueError, match="Detector must be fitted before prediction"):
            detector.predict(sample_data)

    def test_score_without_fit(self, detector, sample_data):
        """Test error handling when scoring without fitting."""
        with pytest.raises(ValueError, match="Detector must be fitted before scoring"):
            detector.score(sample_data)

    def test_metadata(self, detector):
        """Test metadata generation."""
        metadata = detector.metadata

        assert isinstance(metadata, dict)
        assert metadata["name"] == "TimeSeriesIsolationForest"
        assert "version" in metadata
        assert "config" in metadata
        assert "feature_extractors" in metadata
        assert "fitted" in metadata
        assert metadata["fitted"] is False

    @pytest.mark.parametrize("contamination", [0.1, 0.2, 0.3])
    def test_different_contamination_levels(self, contamination):
        """Test detector with different contamination levels."""
        config = IsolationForestConfig(contamination=contamination)
        detector = TimeSeriesIsolationForest(config)

        # Generate data with known anomalies
        data = pd.Series(np.random.normal(0, 1, 1000))
        data[::10] = 10  # Add obvious anomalies

        detector.fit(data)
        result = detector.predict(data)

        anomaly_ratio = sum(1 for score in result.scores if score.is_anomaly) / len(data)
        assert abs(anomaly_ratio - contamination) < 0.1  # Allow some deviation

class TestStreamingIsolationForest:
    """Tests for StreamingIsolationForest class."""
    
    @pytest.fixture
    def streaming_detector(self):
        """Create a streaming detector instance."""
        config = IsolationForestConfig(n_estimators=100, window_size=10)
        return StreamingIsolationForest(config, update_interval=50)
        
    def test_streaming_initialization(self, streaming_detector):
        """Test streaming detector initialization."""
        assert streaming_detector.update_interval == 50
        assert len(streaming_detector._buffer) == 0
        assert streaming_detector._n_processed == 0

    def test_stream_processing(self, streaming_detector):
        """Test processing of streaming data."""
        # Generate streaming data
        data_stream = (
            pd.Series([i + np.random.normal(0, 0.1)])
            for i in range(100)
        )
        
        # Process stream
        results = list(streaming_detector.process_stream(data_stream))
        
        assert len(results) > 0
        assert all(isinstance(result, DetectionResult) for result in results)

    def test_callback_functionality(self, streaming_detector):
        """Test callback function in streaming."""
        callback_mock = Mock()
        data_stream = (
            pd.Series([i]) for i in range(100)
        )
        
        for _ in streaming_detector.process_stream(
            data_stream,
            callback=callback_mock
        ):
            pass
            
        assert callback_mock.call_count > 0

class TestIntegration:
    """Integration tests for the Isolation Forest detector."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data generation to anomaly detection."""
        # Generate data with known anomalies
        dates = pd.date_range('2024-01-01', periods=300, freq='H')
        normal_data = np.sin(np.linspace(0, 20 * np.pi, 300))
        anomalies = np.zeros(300)
        anomalies[50] = 5
        anomalies[150] = -5
        anomalies[250] = 5
        data = pd.Series(normal_data + anomalies + np.random.normal(0, 0.1, 300), index=dates)
        
        # Configure and train detector
        config = IsolationForestConfig(
            window_size=20,
            n_estimators=100,
            contamination=0.1,
            seasonal_period=24
        )
        detector = TimeSeriesIsolationForest(config)
        
        # Train and detect
        detector.fit(data)
        result = detector.predict(data)
        
        # Verify results
        assert isinstance(result, DetectionResult)
        assert len(result.scores) == len(data)
        
        # Check if known anomalies are detected
        anomaly_indices = [i for i, score in enumerate(result.scores) if score.is_anomaly]
        assert 50 in anomaly_indices
        assert 150 in anomaly_indices
        assert 250 in anomaly_indices

    def test_streaming_integration(self):
        """Test integration of streaming detector with real-time data."""
        config = IsolationForestConfig(n_estimators=100, window_size=10)
        detector = StreamingIsolationForest(config, update_interval=100)
        
        # Simulate real-time data
        def data_generator():
            for i in range(200):
                if i % 50 == 0:  # Add anomalies
                    yield pd.Series([100 + np.random.normal(0, 1)])
                else:
                    yield pd.Series([i + np.random.normal(0, 1)])
        
        results = []
        anomalies_found = []
        
        def callback(result):
            if any(score.is_anomaly for score in result.scores):
                anomalies_found.append(result)
                
        for result in detector.process_stream(data_generator(), callback=callback):
            results.append(result)
            
        assert len(results) > 0
        assert len(anomalies_found) > 0
        
        # Verify increasing update counts
        assert detector._n_processed > 0
        assert len(detector._buffer) <= detector.base_detector.config.window_size

if __name__ == '__main__':
    pytest.main([__file__])
