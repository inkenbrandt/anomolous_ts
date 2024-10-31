import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from anomolous_ts.isoforest import (
    IsolationForestConfig,
    TimeSeriesIsolationForest,
    DetectionResult,
    AnomalyScore
)

class TestIsolationForestConfig:
    """Test configuration class for Isolation Forest."""
    
    def test_default_config(self):
        """Test default configuration initialization."""
        config = IsolationForestConfig()
        
        assert config.window_size == 5
        assert config.n_estimators == 100
        assert config.contamination == 'auto'
        assert config.max_features == 1.0
        assert config.bootstrap is False
        assert config.n_jobs == -1
        assert config.seasonal_period is None
        assert config.random_state == 42
        assert isinstance(config.feature_settings, dict)
        assert config.feature_settings['statistical'] is True
        assert config.feature_settings['spectral'] is False
        assert config.feature_settings['wavelet'] is False

    def test_custom_config(self):
        """Test custom configuration initialization."""
        config = IsolationForestConfig(
            window_size=10,
            n_estimators=200,
            contamination=0.1,
            max_features=0.8,
            bootstrap=True,
            n_jobs=4,
            seasonal_period=24,
            random_state=123,
            feature_settings={
                'statistical': True,
                'spectral': True,
                'wavelet': True
            }
        )
        
        assert config.window_size == 10
        assert config.n_estimators == 200
        assert config.contamination == 0.1
        assert config.max_features == 0.8
        assert config.bootstrap is True
        assert config.n_jobs == 4
        assert config.seasonal_period == 24
        assert config.random_state == 123
        assert all(config.feature_settings.values())

@pytest.fixture
def simple_timeseries():
    """Create a simple time series with known patterns."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    values = np.sin(np.linspace(0, 4*np.pi, 100))  # Two complete sine waves
    noise = np.random.normal(0, 0.1, 100)
    anomalies = np.zeros(100)
    anomalies[25] = 5  # Add known anomaly
    
    return pd.Series(values + noise + anomalies, index=dates)

@pytest.fixture
def complex_timeseries():
    """Create a more complex time series with multiple patterns."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    
    # Base signal: combination of trends and seasonality
    t = np.linspace(0, 8*np.pi, 200)
    trend = 0.01 * np.arange(200)
    daily = 2 * np.sin(2*np.pi*np.arange(200)/24)  # 24-hour seasonality
    weekly = np.sin(2*np.pi*np.arange(200)/168)    # Weekly seasonality
    noise = np.random.normal(0, 0.1, 200)
    
    # Add anomalies
    anomalies = np.zeros(200)
    anomaly_indices = [50, 100, 150]
    anomalies[anomaly_indices] = [5, -5, 5]
    
    values = trend + daily + weekly + noise + anomalies
    return pd.Series(values, index=dates)

@pytest.fixture
def multivariate_timeseries():
    """Create multivariate time series data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=150, freq='H')
    
    # Create three correlated series with different patterns
    t = np.linspace(0, 6*np.pi, 150)
    series1 = np.sin(t) + np.random.normal(0, 0.1, 150)
    series2 = np.cos(t) + np.random.normal(0, 0.1, 150)
    series3 = 0.5 * (series1 + series2) + np.random.normal(0, 0.1, 150)
    
    # Add anomalies to each series
    series1[50] = 5
    series2[100] = -5
    series3[75] = 5
    
    return pd.DataFrame({
        'series1': series1,
        'series2': series2,
        'series3': series3
    }, index=dates)

@pytest.fixture
def basic_detector():
    """Create a basic detector with default configuration."""
    config = IsolationForestConfig()
    return TimeSeriesIsolationForest(config)

@pytest.fixture
def configured_detector():
    """Create a detector with custom configuration."""
    config = IsolationForestConfig(
        window_size=10,
        n_estimators=100,
        contamination=0.1,
        seasonal_period=24,
        feature_settings={
            'statistical': True,
            'spectral': True,
            'wavelet': False
        }
    )
    return TimeSeriesIsolationForest(config)

class TestTimeSeriesIsolationForest:
    """Test the Isolation Forest implementation."""

    def test_initialization(self, basic_detector):
        """Test detector initialization."""
        assert basic_detector._feature_scaler is None
        assert basic_detector._decision_scores is None
        assert len(basic_detector._feature_extractors) > 0
        assert isinstance(basic_detector.metadata, dict)

    def test_basic_fit_predict(self, basic_detector, simple_timeseries):
        """Test basic fitting and prediction."""
        # Fit the detector
        detector = basic_detector.fit(simple_timeseries)
        assert detector._feature_scaler is not None
        assert detector._decision_scores is not None
        
        # Make predictions
        result = detector.predict(simple_timeseries)
        assert isinstance(result, DetectionResult)
        assert len(result.scores) == len(simple_timeseries)
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        
        # Check if known anomaly is detected
        anomaly_scores = [score.is_anomaly for score in result.scores]
        assert anomaly_scores[25]  # Known anomaly at index 25

    def test_feature_extraction(self, basic_detector, simple_timeseries):
        """Test feature extraction capabilities."""
        # Extract features
        features = basic_detector._preprocess_and_extract_features(simple_timeseries.values)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(simple_timeseries)
        assert features.shape[1] > 0
        assert not np.isnan(features).any()
        assert np.isfinite(features).all()

    def test_multivariate_detection(self, configured_detector, multivariate_timeseries):
        """Test detection on multivariate data."""
        # Fit and predict
        detector = configured_detector.fit(multivariate_timeseries)
        result = detector.predict(multivariate_timeseries)
        
        assert len(result.scores) == len(multivariate_timeseries)
        
        # Check if known anomalies are detected
        anomaly_scores = [score.is_anomaly for score in result.scores]
        assert any(anomaly_scores[48:52])   # Around index 50
        assert any(anomaly_scores[73:77])   # Around index 75
        assert any(anomaly_scores[98:102])  # Around index 100

    def test_seasonal_detection(self, configured_detector, complex_timeseries):
        """Test detection with seasonal patterns."""
        # Fit and predict
        detector = configured_detector.fit(complex_timeseries)
        result = detector.predict(complex_timeseries)
        
        # Verify seasonal anomalies are detected
        anomaly_scores = [score.is_anomaly for score in result.scores]
        assert any(anomaly_scores[48:52])   # Around index 50
        assert any(anomaly_scores[98:102])  # Around index 100
        assert any(anomaly_scores[148:152]) # Around index 150

    def test_edge_cases(self, basic_detector):
        """Test handling of edge cases."""
        # Test with constant values
        constant_data = pd.Series(np.ones(50))
        detector = basic_detector.fit(constant_data)
        result = detector.predict(constant_data)
        assert len(result.scores) == len(constant_data)
        
        # Test with single value
        single_value = pd.Series([1.0])
        detector = basic_detector.fit(single_value)
        result = detector.predict(single_value)
        assert len(result.scores) == 1
        
        # Test with missing values
        data_with_nans = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        detector = basic_detector.fit(data_with_nans)
        result = detector.predict(data_with_nans)
        assert len(result.scores) == len(data_with_nans)

    def test_error_handling(self, basic_detector, simple_timeseries):
        """Test error handling."""
        # Test prediction without fitting
        with pytest.raises(ValueError, match="must be fitted"):
            basic_detector.predict(simple_timeseries)
        
        # Test scoring without fitting
        with pytest.raises(ValueError, match="must be fitted"):
            basic_detector.score(simple_timeseries)

    @pytest.mark.parametrize("contamination", [0.1, 0.2, 0.3])
    def test_contamination_levels(self, contamination, simple_timeseries):
        """Test different contamination levels."""
        config = IsolationForestConfig(contamination=contamination)
        detector = TimeSeriesIsolationForest(config)
        
        detector.fit(simple_timeseries)
        result = detector.predict(simple_timeseries)
        
        anomaly_ratio = sum(1 for score in result.scores if score.is_anomaly) / len(result.scores)
        assert abs(anomaly_ratio - contamination) < 0.1  # Allow some deviation

if __name__ == '__main__':
    pytest.main([__file__])
