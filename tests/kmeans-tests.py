import pytest
import numpy as np
import pandas as pd
from typing import List, Generator
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series, data_frames

from anomolous_ts.kmeans import (
    TimeSeriesKMeansDetector,
    StreamingKMeansDetector,
    KMeansConfig,
    DetectionResult,
    AnomalyScore
)

# Helper functions for test data generation
def generate_sine_wave(
    n_points: int = 1000,
    frequency: float = 0.1,
    amplitude: float = 1.0,
    noise: float = 0.1
) -> pd.Series:
    """Generate a sine wave with optional noise."""
    t = np.linspace(0, 10 * np.pi, n_points)
    values = amplitude * np.sin(frequency * t)
    if noise > 0:
        values += np.random.normal(0, noise, n_points)
    return pd.Series(
        values,
        index=pd.date_range('2024-01-01', periods=n_points, freq='H')
    )

def add_anomalies(
    series: pd.Series,
    n_anomalies: int = 10,
    amplitude: float = 3.0
) -> pd.Series:
    """Add random anomalies to a time series."""
    result = series.copy()
    indices = np.random.choice(len(series), n_anomalies, replace=False)
    result.iloc[indices] += np.random.normal(0, amplitude, n_anomalies)
    return result

class TestKMeansConfig:
    """Tests for KMeansConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = KMeansConfig()
        assert config.n_clusters == 3
        assert config.window_size == 10
        assert config.anomaly_threshold == 2.0
        assert config.seasonal_period is None
        assert config.random_state == 42
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = KMeansConfig(
            n_clusters=5,
            window_size=20,
            anomaly_threshold=3.0,
            seasonal_period=24
        )
        assert config.n_clusters == 5
        assert config.window_size == 20
        assert config.anomaly_threshold == 3.0
        assert config.seasonal_period == 24
        
    def test_feature_settings(self):
        """Test feature settings configuration."""
        config = KMeansConfig(
            feature_settings={
                'statistical': True,
                'spectral': True,
                'wavelet': False
            }
        )
        assert config.feature_settings['statistical']
        assert config.feature_settings['spectral']
        assert not config.feature_settings['wavelet']

class TestTimeSeriesKMeansDetector:
    """Tests for TimeSeriesKMeansDetector class."""
    
    @pytest.fixture
    def detector(self) -> TimeSeriesKMeansDetector:
        """Create a detector instance for testing."""
        config = KMeansConfig(n_clusters=3, window_size=5)
        return TimeSeriesKMeansDetector(config)
    
    @pytest.fixture
    def sample_data(self) -> pd.Series:
        """Create sample time series data."""
        return generate_sine_wave(n_points=100)
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.config.n_clusters == 3
        assert detector.config.window_size == 5
        assert detector._cluster_centers is None
        assert detector._labels is None
        
    def test_fit(self, detector, sample_data):
        """Test fitting the detector."""
        detector.fit(sample_data)
        assert detector._cluster_centers is not None
        assert detector._labels is not None
        assert len(detector._labels) == len(sample_data)
        
    def test_predict(self, detector, sample_data):
        """Test prediction functionality."""
        detector.fit(sample_data)
        result = detector.predict(sample_data)
        
        assert isinstance(result, DetectionResult)
        assert len(result.scores) == len(sample_data)
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        
    def test_score(self, detector, sample_data):
        """Test scoring functionality."""
        detector.fit(sample_data)
        scores = detector.score(sample_data)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_data)
        
    @given(
        arrays(
            dtype=float,
            shape=st.integers(min_value=50, max_value=200),
            elements=st.floats(min_value=-10, max_value=10)
        )
    )
    def test_property_based(self, data):
        """Property-based tests using hypothesis."""
        series = pd.Series(data)
        detector = TimeSeriesKMeansDetector(KMeansConfig(n_clusters=3))
        
        detector.fit(series)
        result = detector.predict(series)
        
        assert len(result.scores) == len(series)
        assert all(0 <= score.confidence <= 1 for score in result.scores)
        
    def test_anomaly_detection(self, detector):
        """Test anomaly detection capabilities."""
        # Generate normal data with known anomalies
        normal_data = generate_sine_wave(n_points=200, noise=0.1)
        anomalous_data = add_anomalies(normal_data, n_anomalies=10, amplitude=5.0)
        
        detector.fit(normal_data)
        result = detector.predict(anomalous_data)
        
        # Check that anomalies were detected
        n_anomalies = sum(1 for score in result.scores if score.is_anomaly)
        assert n_anomalies > 0
        
    def test_metadata(self, detector):
        """Test metadata generation."""
        metadata = detector.metadata
        assert "name" in metadata
        assert "version" in metadata
        assert "config" in metadata
        assert "fitted" in metadata

class TestStreamingKMeansDetector:
    """Tests for StreamingKMeansDetector class."""
    
    @pytest.fixture
    def streaming_detector(self) -> StreamingKMeansDetector:
        """Create a streaming detector instance."""
        config = KMeansConfig(n_clusters=3, window_size=10)
        return StreamingKMeansDetector(config, update_interval=50)
        
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
        
    def test_streaming_update(self, streaming_detector):
        """Test model updates during streaming."""
        data = generate_sine_wave(n_points=200)
        streaming_detector.update(data)
        
        assert streaming_detector.base_detector._cluster_centers is not None
        assert streaming_detector.base_detector._labels is not None
        
    @pytest.mark.asyncio
    async def test_async_streaming(self, streaming_detector):
        """Test asynchronous streaming capabilities."""
        async def async_data_stream():
            for i in range(100):
                yield pd.Series([i + np.random.normal(0, 0.1)])
                
        results = []
        async for result in streaming_detector.process_stream(async_data_stream()):
            results.append(result)
            
        assert len(results) > 0
        
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
    """Integration tests for the KMeans detector."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data generation to anomaly detection."""
        # Generate data
        normal_data = generate_sine_wave(n_points=300)
        anomalous_data = add_anomalies(normal_data, n_anomalies=15)
        
        # Configure and train detector
        config = KMeansConfig(
            n_clusters=5,
            window_size=20,
            anomaly_threshold=2.5,
            seasonal_period=24
        )
        detector = TimeSeriesKMeansDetector(config)
        
        # Train on normal data
        detector.fit(normal_data)
        
        # Detect anomalies
        result = detector.predict(anomalous_data)
        
        # Verify results
        assert isinstance(result, DetectionResult)
        assert len(result.scores) == len(anomalous_data)
        
        # Check anomaly detection performance
        anomalies = result.get_anomalies()
        assert len(anomalies) > 0
        
    def test_streaming_integration(self):
        """Test integration of streaming detector with real-time data."""
        config = KMeansConfig(n_clusters=3)
        detector = StreamingKMeansDetector(config, update_interval=100)
        
        # Simulate real-time data
        def data_generator() -> Generator[pd.Series, None, None]:
            for i in range(200):
                if i % 50 == 0:  # Add anomalies
                    yield pd.Series([100 + np.random.normal(0, 1)])
                else:
                    yield pd.Series([i + np.random.normal(0, 1)])
                    
        results = []
        anomalies_found = []
        
        def callback(result: DetectionResult):
            anomalies = result.get_anomalies()
            if anomalies:
                anomalies_found.extend(anomalies)
                
        for result in detector.process_stream(data_generator(), callback=callback):
            results.append(result)
            
        assert len(results) > 0
        assert len(anomalies_found) > 0

if __name__ == '__main__':
    pytest.main([__file__])
