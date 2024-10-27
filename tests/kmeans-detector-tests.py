import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from scipy import stats

from anomolous_ts.preprocessor import TimeSeriesPreprocessor
from anomolous_ts.visualizer import TimeSeriesVisualizer
from anomolous_ts.detector import TimeSeriesKMeansDetector

class TestTimeSeriesKMeansDetector(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        
        # Create simple synthetic data with known patterns
        t = np.linspace(0, 4 * np.pi, 100)
        signal = 10 * np.sin(t) + np.random.normal(0, 0.5, 100)
        
        # Add some known anomalies
        signal[20] = 50  # Obvious anomaly
        signal[50] = -30  # Obvious anomaly
        
        self.univariate_data = pd.Series(signal, index=dates)
        self.multivariate_data = pd.DataFrame({
            'signal1': signal,
            'signal2': np.roll(signal, 5)
        }, index=dates)
        
        # Create default detector instance
        self.detector = TimeSeriesKMeansDetector(
            window_size=5,
            n_clusters=3,
            anomaly_threshold=2.0,
            seasonal_period=24
        )

    def test_initialization(self):
        """Test initialization of TimeSeriesKMeansDetector."""
        self.assertEqual(self.detector.window_size, 5)
        self.assertEqual(self.detector.n_clusters, 3)
        self.assertEqual(self.detector.anomaly_threshold, 2.0)
        self.assertEqual(self.detector.seasonal_period, 24)
        self.assertIsInstance(self.detector.preprocessor, TimeSeriesPreprocessor)
        self.assertIsInstance(self.detector.visualizer, TimeSeriesVisualizer)

    def test_create_temporal_features(self):
        """Test temporal feature creation."""
        features = self.detector._create_temporal_features(self.univariate_data.values)
        
        # Check feature matrix dimensions
        expected_n_features = 16  # 13 basic features + 3 seasonal features
        self.assertEqual(features.shape[1], expected_n_features)
        self.assertEqual(features.shape[0], len(self.univariate_data))
        
        # Check if initial padding is zero
        self.assertTrue(np.all(features[:self.detector.window_size] == 0))

    def test_calculate_anomaly_scores(self):
        """Test anomaly score calculation."""
        # Create sample features and fit the model
        features = self.detector._create_temporal_features(self.univariate_data.values)
        self.detector.model.fit(features)
        self.detector.cluster_centers_ = self.detector.model.cluster_centers_
        
        # Calculate scores
        scores = self.detector._calculate_anomaly_scores(features)
        
        # Check if scores are between 0 and 1 (sigmoid output)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))
        
        # Check if shape matches input
        self.assertEqual(len(scores), len(features))

    def test_fit_predict_univariate(self):
        """Test fit_predict method with univariate data."""
        # Test without scores
        anomalies = self.detector.fit_predict(self.univariate_data)
        self.assertIsInstance(anomalies, pd.Series)
        self.assertEqual(len(anomalies), len(self.univariate_data))
        self.assertEqual(anomalies.dtype, bool)
        
        # Test with scores
        anomalies, scores = self.detector.fit_predict(self.univariate_data, return_scores=True)
        self.assertIsInstance(scores, pd.Series)
        self.assertEqual(len(scores), len(self.univariate_data))

    def test_fit_predict_multivariate(self):
        """Test fit_predict method with multivariate data."""
        # Test without scores
        anomalies = self.detector.fit_predict(self.multivariate_data)
        self.assertIsInstance(anomalies, pd.DataFrame)
        self.assertEqual(anomalies.shape, self.multivariate_data.shape)
        
        # Test with scores
        anomalies, scores = self.detector.fit_predict(self.multivariate_data, return_scores=True)
        self.assertIsInstance(scores, pd.DataFrame)
        self.assertEqual(scores.shape, self.multivariate_data.shape)

    def test_detect_known_anomalies(self):
        """Test if detector finds known injected anomalies."""
        anomalies, scores = self.detector.fit_predict(self.univariate_data, return_scores=True)
        
        # Check if known anomalies are detected
        self.assertTrue(anomalies[20])  # First injected anomaly
        self.assertTrue(anomalies[50])  # Second injected anomaly
        
        # Check if anomaly scores are higher for known anomalies
        self.assertGreater(scores[20], scores.median())
        self.assertGreater(scores[50], scores.median())

    def test_get_cluster_profiles(self):
        """Test cluster profile generation."""
        # First fit the model
        self.detector.fit_predict(self.univariate_data)
        
        # Get profiles
        profiles = self.detector.get_cluster_profiles()
        
        # Check profile structure
        self.assertEqual(len(profiles), self.detector.n_clusters)
        for cluster in profiles.values():
            self.assertIn('size', cluster)
            self.assertIn('center', cluster)
            self.assertIn('mean_distance', cluster)
            self.assertIn('std_distance', cluster)

    def test_process_stream(self):
        """Test streaming data processing."""
        chunk_size = 20
        stream = [self.univariate_data[i:i+chunk_size] 
                 for i in range(0, len(self.univariate_data), chunk_size)]
        
        results = []
        for chunk_data, anomalies, scores in self.detector.process_stream(stream, chunk_size):
            results.append({
                'chunk_size': len(chunk_data),
                'n_anomalies': anomalies.sum(),
                'mean_score': scores.mean()
            })
            
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['chunk_size'], chunk_size)

    def test_missing_values_handling(self):
        """Test handling of missing values."""
        # Create data with missing values
        data_with_nan = self.univariate_data.copy()
        data_with_nan[10:15] = np.nan
        
        # Should not raise an error
        try:
            anomalies = self.detector.fit_predict(data_with_nan)
            self.assertEqual(len(anomalies), len(data_with_nan))
        except Exception as e:
            self.fail(f"Processing data with missing values raised an exception: {e}")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with minimum size dataset
        min_data = pd.Series(np.random.randn(self.detector.window_size + 1))
        try:
            self.detector.fit_predict(min_data)
        except Exception as e:
            self.fail(f"Processing minimum size dataset raised an exception: {e}")
            
        # Test with invalid window size
        with self.assertRaises(ValueError):
            detector = TimeSeriesKMeansDetector(window_size=0)
            
        # Test with invalid number of clusters
        with self.assertRaises(ValueError):
            detector = TimeSeriesKMeansDetector(n_clusters=0)
            
        # Test get_cluster_profiles before fitting
        detector = TimeSeriesKMeansDetector()
        with self.assertRaises(ValueError):
            detector.get_cluster_profiles()

    def test_seasonal_feature_extraction(self):
        """Test seasonal feature extraction."""
        # Create detector with and without seasonal features
        detector_seasonal = TimeSeriesKMeansDetector(seasonal_period=24)
        detector_non_seasonal = TimeSeriesKMeansDetector(seasonal_period=None)
        
        # Compare feature dimensions
        features_seasonal = detector_seasonal._create_temporal_features(self.univariate_data.values)
        features_non_seasonal = detector_non_seasonal._create_temporal_features(self.univariate_data.values)
        
        self.assertGreater(features_seasonal.shape[1], features_non_seasonal.shape[1])

    def test_visualization_integration(self):
        """Test integration with visualizer."""
        try:
            anomalies, scores = self.detector.detect_and_visualize(
                self.univariate_data,
                title="Test Visualization"
            )
            self.assertIsNotNone(anomalies)
            self.assertIsNotNone(scores)
        except Exception as e:
            self.fail(f"Visualization integration failed: {e}")

if __name__ == '__main__':
    unittest.main()
