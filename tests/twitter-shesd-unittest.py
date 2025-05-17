import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from anomolous_ts.shesd import TwitterSHESD, SHESDConfig
from anomolous_ts.base import DetectionResult, AnomalyScore

class TestSHESDConfig(unittest.TestCase):
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
        
        self.assertEqual(config.max_anomalies, 0.1)
        self.assertEqual(config.alpha, 0.05)
        self.assertTrue(config.hybrid)
        self.assertEqual(config.seasonal_period, 24)
        self.assertEqual(config.decomposition_method, 'additive')
    
    def test_invalid_max_anomalies(self):
        """Test invalid max_anomalies values."""
        invalid_values = [-0.1, 0, 1, 1.5]
        for value in invalid_values:
            with self.assertRaises(ValueError):
                SHESDConfig(max_anomalies=value)
    
    def test_invalid_alpha(self):
        """Test invalid alpha values."""
        invalid_values = [-0.1, 0, 1, 1.5]
        for value in invalid_values:
            with self.assertRaises(ValueError):
                SHESDConfig(alpha=value)
    
    def test_invalid_decomposition_method(self):
        """Test invalid decomposition method."""
        with self.assertRaises(ValueError):
            SHESDConfig(decomposition_method='invalid')

class TestTwitterSHESD(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test."""
        self.config = SHESDConfig(
            max_anomalies=0.1,
            alpha=0.05,
            hybrid=True,
            seasonal_period=None,
            auto_detect_period=True
        )
        self.detector = TwitterSHESD(self.config)
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
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
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsInstance(self.detector.config, SHESDConfig)
        self.assertIsNone(self.detector._seasonal_period)
        self.assertIsNone(self.detector._location_stats)
        self.assertIsNone(self.detector._scale_stats)
    
    def test_period_detection(self):
        """Test automatic period detection."""
        period = self.detector._detect_period(self.sample_data.values)
        self.assertEqual(period, 24)  # Should detect daily seasonality
        
        # Test with random data (should default to 1)
        random_data = pd.Series(np.random.randn(100))
        period = self.detector._detect_period(random_data.values)
        self.assertEqual(period, 1)
    
    def test_seasonal_decomposition(self):
        """Test seasonal component removal."""
        # Test additive decomposition
        adjusted_data = self.detector._remove_seasonal_component(
            self.sample_data,
            period=24
        )
        self.assertEqual(len(adjusted_data), len(self.sample_data))
        self.assertFalse(np.allclose(adjusted_data.values, self.sample_data.values))
        
        # Test multiplicative decomposition
        self.detector.config.decomposition_method = 'multiplicative'
        positive_data = self.sample_data - self.sample_data.min() + 1
        adjusted_data = self.detector._remove_seasonal_component(positive_data, period=24)
        self.assertEqual(len(adjusted_data), len(positive_data))
        
        # Test error with negative values in multiplicative mode
        with self.assertRaises(ValueError):
            self.detector._remove_seasonal_component(self.sample_data, period=24)
    
    def test_statistics_computation(self):
        """Test computation of location and scale statistics."""
        data = np.array([1, 2, 3, 10, 2, 3, 2, 1])
        
        # Test hybrid statistics (median/MAD)
        self.detector.config.hybrid = True
        location, scale = self.detector._compute_statistics(data)
        self.assertEqual(location, np.median(data))
        expected_scale = np.median(np.abs(data - np.median(data))) * 1.4826
        self.assertAlmostEqual(scale, expected_scale)
        
        # Test standard statistics (mean/std)
        self.detector.config.hybrid = False
        location, scale = self.detector._compute_statistics(data)
        self.assertEqual(location, np.mean(data))
        self.assertEqual(scale, np.std(data, ddof=1))
    
    def test_esd_test(self):
        """Test Generalized ESD test."""
        # Create data with known outliers
        data = np.array([1, 2, 3, 10, 2, 3, 20, 1, 2, 30])
        max_outliers = 3
        
        outlier_indices = self.detector._generalized_esd_test(data, max_outliers)
        
        self.assertLessEqual(len(outlier_indices), max_outliers)
        for i in outlier_indices:
            self.assertIn(data[i], [10, 20, 30])
    
    def test_anomaly_scores(self):
        """Test anomaly score calculation."""
        data = np.array([1, 2, 3, 10, 2, 3, 20, 1, 2, 30])
        anomaly_indices = [3, 6, 9]  # Known anomalies
        
        # Set statistics for score calculation
        self.detector._location_stats = np.median(data)
        self.detector._scale_stats = np.median(np.abs(data - np.median(data))) * 1.4826
        
        scores = self.detector._calculate_anomaly_scores(data, anomaly_indices)
        
        self.assertEqual(len(scores), len(data))
        for s in scores:
            self.assertIsInstance(s, AnomalyScore)
        
        for i in anomaly_indices:
            self.assertTrue(scores[i].is_anomaly)
        
        for i in range(len(data)):
            if i not in anomaly_indices:
                self.assertFalse(scores[i].is_anomaly)
    
    def test_fit_predict(self):
        """Test full fit-predict workflow."""
        self.detector.fit(self.sample_data)
        result = self.detector.predict(self.sample_data)
        
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(len(result.scores), len(self.sample_data))
        self.assertEqual(result.detector_name, "TwitterSHESD")
        
        # Check that known anomalies are detected
        anomaly_indices = [i for i, score in enumerate(result.scores)
                         if score.is_anomaly]
        self.assertGreater(len(anomaly_indices), 0)
        self.assertLessEqual(
            len(anomaly_indices),
            len(self.sample_data) * self.detector.config.max_anomalies
        )
    
    def test_score_method(self):
        """Test raw scoring functionality."""
        scores = self.detector.score(self.sample_data)
        
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(self.sample_data))
        self.assertTrue(np.all(scores >= 0))  # Scores should be non-negative
    
    def test_metadata(self):
        """Test metadata generation."""
        self.detector.fit(self.sample_data)
        self.detector.predict(self.sample_data)  # Need to run prediction to get statistics
        
        metadata = self.detector.metadata
        
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata["name"], "TwitterSHESD")
        self.assertIn("version", metadata)
        self.assertIn("config", metadata)
        self.assertIn("statistics", metadata)
        self.assertIsNotNone(metadata["statistics"]["location"])
        self.assertIsNotNone(metadata["statistics"]["scale"])
    
    def test_different_configurations(self):
        """Test detector with different configuration combinations."""
        configs = [
            {'hybrid': True, 'decomposition_method': 'additive'},
            {'hybrid': False, 'decomposition_method': 'additive'},
            {'hybrid': True, 'decomposition_method': 'multiplicative'},
            {'hybrid': False, 'decomposition_method': 'multiplicative'}
        ]
        
        for config_params in configs:
            config = SHESDConfig(
                **config_params,
                seasonal_period=24,
                auto_detect_period=False
            )
            detector = TwitterSHESD(config)
            
            data = self.sample_data
            if config_params['decomposition_method'] == 'multiplicative':
                data = self.sample_data - self.sample_data.min() + 1
            
            result = detector.fit_predict(data)
            self.assertIsInstance(result, DetectionResult)
            self.assertEqual(len(result.scores), len(data))
    
    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Test with constant data
        constant_data = pd.Series([1.0] * 100)
        result = self.detector.fit_predict(constant_data)
        self.assertEqual(len(result.scores), len(constant_data))
        
        # Test with single value
        single_value = pd.Series([1.0])
        result = self.detector.fit_predict(single_value)
        self.assertEqual(len(result.scores), 1)
        
        # Test with NaN values
        data_with_nans = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        with self.assertRaises(ValueError):
            self.detector.fit_predict(data_with_nans)
        
        # Test with multivariate data
        multivariate_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        with self.assertRaises(ValueError):
            self.detector.fit_predict(multivariate_data)

if __name__ == '__main__':
    unittest.main()
