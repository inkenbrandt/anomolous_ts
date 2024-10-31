import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import series, data_frames
from hypothesis.extra.numpy import arrays

from anomolous_ts.shesd import TwitterSHESD, SHESDConfig
from anomolous_ts.base import DetectionResult, AnomalyScore

class TestSHESDConfig:
    """Tests for SHESDConfig class."""
    
    def test_valid_config(self):
        """Test valid configuration initialization."""
        config = SHESDConfig(
            max_anomalies=0.1,
            alpha=0.05,
            hybrid=True,
            seasonal_period=24,
            decomposition_method='additive'
        )
        
        assert config.max_anomalies == 0.1
        assert config.alpha == 0.05
        assert config.hybrid is True
        assert config.seasonal_period == 24
        assert config.decomposition_method == 'additive'
        
    @pytest.mark.parametrize("max_anomalies", [-0.1, 0, 1, 1.5])
    def test_invalid_max_anomalies(self, max_anomalies):
        """Test invalid max_anomalies values."""
        with pytest.raises(ValueError):
            SHESDConfig(max_anomalies=max_anomalies)
            
    @pytest.mark.parametrize("alpha", [-0.1, 0, 1, 1.5])
    def test_invalid_alpha(self, alpha):
        """Test invalid alpha values."""
        with pytest.raises(ValueError):
            SHESDConfig(alpha=alpha)
            
    def test_invalid_decomposition_method(self):
        """Test invalid decomposition method."""
        with pytest.raises(ValueError):
            SHESDConfig(decomposition_method='invalid')

class TestTwitterSHESD:
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing."""
        config = SHESDConfig(
            max_anomalies=0.1,
            alpha=0.05,
            hybrid=True,
            seasonal_period=None,
            auto_detect_period=True
        )
        return TwitterSHESD(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data with known patterns and anomalies."""
        np.random.seed(42)  # For reproducibility
        
        # Generate dates
        dates = pd.date_range('2024-01-01', periods=500, freq='H')
        
        # Create base signal with daily seasonality
        t = np.linspace(0, 20 * np.pi, 500)
        base = np.sin(t / 12) * 10  # Daily pattern (24 hours)
        trend = np.linspace(0, 5, 500)  # Upward trend
        noise = np.random.normal(0, 0.5, 500)
        
        # Add known anomalies
        anomalies = np.zeros(500)
        anomaly_indices = [100, 200, 300, 400, 401, 402]
        anomalies[anomaly_indices] = [15, -10, 20, 12, 12, 12]
        
        # Combine components
        values = base + trend + noise + anomalies
        return pd.Series(values, index=dates)
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert isinstance(detector.config, SHESDConfig)
        assert detector._seasonal_period is None
        assert detector._location_stats is None
        assert detector._scale_stats is None
    
    def test_period_detection(self, detector, sample_data):
        """Test automatic period detection."""
        period = detector._detect_period(sample_data.values)
        assert period == 24  # Should detect daily seasonality
        
        # Test with random data (should default to 1)
        random_data = pd.Series(np.random.randn(100))
        period = detector._detect_period(random_data.values)
        assert period == 1
    
    def test_seasonal_decomposition(self, detector, sample_data):
        """Test seasonal component removal."""
        # Test additive decomposition
        adjusted_data = detector._remove_seasonal_component(sample_data, period=24)
        assert len(adjusted_data) == len(sample_data)
        assert not np.allclose(adjusted_data.values, sample_data.values)
        
        # Test multiplicative decomposition
        detector.config.decomposition_method = 'multiplicative'
        positive_data = sample_data - sample_data.min() + 1  # Make all values positive
        adjusted_data = detector._remove_seasonal_component(positive_data, period=24)
        assert len(adjusted_data) == len(positive_data)
        
        # Test error with negative values in multiplicative mode
        with pytest.raises(ValueError):
            detector._remove_seasonal_component(sample_data, period=24)
    
    def test_statistics_computation(self, detector):
        """Test computation of location and scale statistics."""
        data = np.array([1, 2, 3, 10, 2, 3, 2, 1])
        
        # Test hybrid statistics (median/MAD)
        detector.config.hybrid = True
        location, scale = detector._compute_statistics(data)
        assert location == np.median(data)
        assert scale == np.median(np.abs(data - np.median(data))) * 1.4826
        
        # Test standard statistics (mean/std)
        detector.config.hybrid = False
        location, scale = detector._compute_statistics(data)
        assert location == np.mean(data)
        assert scale == np.std(data, ddof=1)
    
    def test_esd_test(self, detector):
        """Test Generalized ESD test."""
        # Create data with known outliers
        data = np.array([1, 2, 3, 10, 2, 3, 20, 1, 2, 30])
        max_outliers = 3
        
        outlier_indices = detector._generalized_esd_test(data, max_outliers)
        
        assert len(outlier_indices) <= max_outliers
        assert all(data[i] in [10, 20, 30] for i in outlier_indices)
    
    def test_anomaly_scores(self, detector):
        """Test anomaly score calculation."""
        data = np.array([1, 2, 3, 10, 2, 3, 20, 1, 2, 30])
        anomaly_indices = [3, 6, 9]  # Known anomalies
        
        # Set statistics for score calculation
        detector._location_stats = np.median(data)
        detector._scale_stats = np.median(np.abs(data - np.median(data))) * 1.4826
        
        scores = detector._calculate_anomaly_scores(data, anomaly_indices)
        
        assert len(scores) == len(data)
        assert all(isinstance(s, AnomalyScore) for s in scores)
        assert all(scores[i].is_anomaly for i in anomaly_indices)
        assert all(not scores[i].is_anomaly for i in range(len(data))
                  if i not in anomaly_indices)
    
    def test_fit_predict(self, detector, sample_data):
        """Test full fit-predict workflow."""
        detector.fit(sample_data)
        result = detector.predict(sample_data)
        
        assert isinstance(result, DetectionResult)
        assert len(result.scores) == len(sample_data)
        assert result.detector_name == "TwitterSHESD"
        
        # Check that known anomalies are detected
        anomaly_indices = [i for i, score in enumerate(result.scores)
                         if score.is_anomaly]
        assert len(anomaly_indices) > 0
        assert len(anomaly_indices) <= len(sample_data) * detector.config.max_anomalies
    
    def test_score_method(self, detector, sample_data):
        """Test raw scoring functionality."""
        scores = detector.score(sample_data)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_data)
        assert np.all(scores >= 0)  # Scores should be non-negative
    
    def test_metadata(self, detector, sample_data):
        """Test metadata generation."""
        detector.fit(sample_data)
        detector.predict(sample_data)  # Need to run prediction to get statistics
        
        metadata = detector.metadata
        
        assert isinstance(metadata, dict)
        assert metadata["name"] == "TwitterSHESD"
        assert "version" in metadata
        assert "config" in metadata
        assert "statistics" in metadata
        assert metadata["statistics"]["location"] is not None
        assert metadata["statistics"]["scale"] is not None
    
    @pytest.mark.parametrize("hybrid", [True, False])
    @pytest.mark.parametrize("decomposition_method", ['additive', 'multiplicative'])
    def test_different_configurations(self, sample_data, hybrid, decomposition_method):
        """Test detector with different configuration combinations."""
        if decomposition_method == 'multiplicative':
            data = sample_data - sample_data.min() + 1  # Make positive for multiplicative
        else:
            data = sample_data
            
        config = SHESDConfig(
            hybrid=hybrid,
            decomposition_method=decomposition_method,
            seasonal_period=24,
            auto_detect_period=False
        )
        detector = TwitterSHESD(config)
        
        result = detector.fit_predict(data)
        assert isinstance(result, DetectionResult)
        assert len(result.scores) == len(data)
    
    @given(
        arrays(
            dtype=float,
            shape=st.integers(min_value=50, max_value=200),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False)
        )
    )
    def test_property_based(self, data):
        """Property-based testing using hypothesis."""
        series = pd.Series(data)
        detector = TwitterSHESD()
        
        result = detector.fit_predict(series)
        
        assert len(result.scores) == len(series)
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        assert all(0 <= score.confidence <= 1 for score in result.scores)
        
        # Number of anomalies should not exceed max_anomalies
        n_anomalies = sum(1 for score in result.scores if score.is_anomaly)
        assert n_anomalies <= len(series) * detector.config.max_anomalies
    
    def test_edge_cases(self, detector):
        """Test handling of edge cases."""
        # Test with constant data
        constant_data = pd.Series([1.0] * 100)
        result = detector.fit_predict(constant_data)
        assert len(result.scores) == len(constant_data)
        
        # Test with single value
        single_value = pd.Series([1.0])
        result = detector.fit_predict(single_value)
        assert len(result.scores) == 1
        
        # Test with NaN values
        data_with_nans = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        with pytest.raises(ValueError):
            detector.fit_predict(data_with_nans)
        
        # Test with multivariate data
        multivariate_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        with pytest.raises(ValueError):
            detector.fit_predict(multivariate_data)

if __name__ == '__main__':
    pytest.main([__file__])
