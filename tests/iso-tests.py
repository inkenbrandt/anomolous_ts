import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the classes to test
from anomolous_ts.preprocessor import TimeSeriesPreprocessor
from anomolous_ts.visualizer import TimeSeriesVisualizer
from your_module import TimeSeriesIsolationForest, AdvancedTimeSeriesIsolationForest

class TestTimeSeriesIsolationForest(unittest.TestCase):
    def setUp(self):
        """Set up test data that will be used across multiple tests"""
        np.random.seed(42)
        self.n_points = 500
        
        # Generate synthetic time series
        t = np.linspace(0, 4 * np.pi, self.n_points)
        base_signal = np.sin(t) * 10
        trend = np.linspace(0, 5, self.n_points)
        noise = np.random.normal(0, 0.5, self.n_points)
        
        # Add known anomalies
        self.anomaly_indices = [50, 100, 150, 200, 250]
        anomalies = np.zeros(self.n_points)
        anomalies[self.anomaly_indices] = 20  # Large obvious anomalies
        
        # Combine components
        self.time_series = base_signal + trend + noise + anomalies
        
        # Create datetime index
        self.dates = pd.date_range(
            start=datetime(2024, 1, 1),
            periods=self.n_points,
            freq='H'
        )
        self.ts = pd.Series(self.time_series, index=self.dates)
        
        # Create multivariate time series
        self.ts2 = self.ts.shift(24) + np.random.normal(0, 1, self.n_points)
        self.df = pd.DataFrame({'series1': self.ts, 'series2': self.ts2})
        
        # Add some missing values
        self.df.iloc[10:15, 1] = np.nan

    def test_initialization(self):
        """Test if the detector initializes with default parameters"""
        detector = TimeSeriesIsolationForest()
        self.assertEqual(detector.window_size, 5)
        self.assertEqual(detector.n_estimators, 100)
        self.assertEqual(detector.contamination, 'auto')
        self.assertEqual(detector.imputation_method, 'linear')

    def test_univariate_detection(self):
        """Test anomaly detection on univariate time series"""
        detector = TimeSeriesIsolationForest(contamination=0.01)
        anomalies = detector.detect(self.ts)
        
        # Basic checks
        self.assertIsInstance(anomalies, pd.Series)
        self.assertEqual(len(anomalies), len(self.ts))
        self.assertTrue(anomalies.dtype == bool)
        
        # Check if known anomalies are detected
        # We expect at least 60% of injected anomalies to be detected
        detected_known_anomalies = anomalies.iloc[self.anomaly_indices].sum()
        detection_rate = detected_known_anomalies / len(self.anomaly_indices)
        self.assertGreater(detection_rate, 0.6)

    def test_multivariate_detection(self):
        """Test anomaly detection on multivariate time series"""
        detector = TimeSeriesIsolationForest(
            window_size=10,
            contamination=0.01,
            seasonal_period=24
        )
        anomalies = detector.detect(self.df)
        
        # Basic checks
        self.assertIsInstance(anomalies, pd.DataFrame)
        self.assertEqual(anomalies.shape, self.df.shape)
        self.assertTrue(anomalies.dtypes.all() == bool)

    def test_missing_value_handling(self):
        """Test handling of missing values with different imputation methods"""
        imputation_methods = ['linear', 'forward', 'backward', 'mean', 'median']
        
        for method in imputation_methods:
            detector = TimeSeriesIsolationForest(imputation_method=method)
            anomalies = detector.detect(self.df)
            
            # Check if any NaN values in results
            self.assertFalse(anomalies.isna().any().any())

    def test_window_size_validation(self):
        """Test if appropriate window sizes are handled correctly"""
        # Test with minimum window size
        detector = TimeSeriesIsolationForest(window_size=2)
        anomalies = detector.detect(self.ts)
        self.assertEqual(len(anomalies), len(self.ts))
        
        # Test with larger window size
        detector = TimeSeriesIsolationForest(window_size=24)
        anomalies = detector.detect(self.ts)
        self.assertEqual(len(anomalies), len(self.ts))

    def test_seasonal_feature_extraction(self):
        """Test if seasonal features are correctly extracted"""
        detector = TimeSeriesIsolationForest(
            seasonal_period=24,  # Daily seasonality
            window_size=24
        )
        anomalies = detector.detect(self.ts)
        self.assertEqual(len(anomalies), len(self.ts))

class TestAdvancedTimeSeriesIsolationForest(unittest.TestCase):
    def setUp(self):
        """Set up test data for advanced detector tests"""
        # Reuse the same data generation logic as above
        np.random.seed(42)
        self.n_points = 500
        t = np.linspace(0, 4 * np.pi, self.n_points)
        base_signal = np.sin(t) * 10
        trend = np.linspace(0, 5, self.n_points)
        noise = np.random.normal(0, 0.5, self.n_points)
        
        self.anomaly_indices = [50, 100, 150, 200, 250]
        anomalies = np.zeros(self.n_points)
        anomalies[self.anomaly_indices] = 20
        
        self.time_series = base_signal + trend + noise + anomalies
        self.dates = pd.date_range(
            start=datetime(2024, 1, 1),
            periods=self.n_points,
            freq='H'
        )
        self.ts = pd.Series(self.time_series, index=self.dates)

    def test_advanced_initialization(self):
        """Test initialization with custom preprocessor and visualizer"""
        preprocessor = TimeSeriesPreprocessor()
        visualizer = TimeSeriesVisualizer()
        
        detector = AdvancedTimeSeriesIsolationForest(
            preprocessor=preprocessor,
            visualizer=visualizer
        )
        
        self.assertIsNotNone(detector.preprocessor)
        self.assertIsNotNone(detector.visualizer)

    def test_confidence_scores(self):
        """Test if confidence scores are properly calculated"""
        detector = AdvancedTimeSeriesIsolationForest()
        anomalies, confidence_scores = detector.fit_predict(
            self.ts,
            return_confidence=True
        )
        
        # Basic checks on confidence scores
        self.assertTrue(isinstance(confidence_scores, pd.Series))
        self.assertEqual(len(confidence_scores), len(self.ts))
        self.assertTrue((confidence_scores >= 0).all())
        self.assertTrue((confidence_scores <= 1).all())

    def test_streaming_detection(self):
        """Test the streaming detection functionality"""
        detector = AdvancedTimeSeriesIsolationForest()
        chunk_size = 50
        
        # Create a simple data stream
        data_stream = iter(self.ts)
        
        # Process first chunk
        chunk_data = []
        for _ in range(chunk_size):
            try:
                chunk_data.append(next(data_stream))
            except StopIteration:
                break
        
        if chunk_data:
            chunk_series = pd.Series(
                chunk_data,
                index=self.dates[:len(chunk_data)]
            )
            anomalies, confidence_scores = detector.fit_predict(
                chunk_series,
                return_confidence=True
            )
            
            # Basic checks on streaming results
            self.assertEqual(len(anomalies), len(chunk_data))
            self.assertEqual(len(confidence_scores), len(chunk_data))

    def test_detect_and_visualize(self):
        """Test the combined detection and visualization functionality"""
        detector = AdvancedTimeSeriesIsolationForest()
        anomalies, confidence_scores = detector.detect_and_visualize(
            self.ts,
            title='Test Visualization'
        )
        
        # Check results
        self.assertTrue(isinstance(anomalies, pd.Series))
        self.assertTrue(isinstance(confidence_scores, pd.Series))
        self.assertEqual(len(anomalies), len(self.ts))
        self.assertEqual(len(confidence_scores), len(self.ts))

if __name__ == '__main__':
    unittest.main()
