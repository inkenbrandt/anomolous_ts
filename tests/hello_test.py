import unittest
import numpy as np
import pandas as pd

from anomolous_ts import AdvancedTimeSeriesIsolationForest


class TestAdvancedTimeSeriesIsolationForest(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample time series data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Generate normal pattern
        self.normal_data = pd.DataFrame({
            'timestamp': dates,
            'value': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100),
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })

        # Add some anomalies
        self.anomalous_data = self.normal_data.copy()
        self.anomalous_data.loc[20:25, 'value'] += 5  # Sudden spike
        self.anomalous_data.loc[50:55, 'value'] -= 5  # Sudden drop

        # Initialize the model
        self.model = AdvancedTimeSeriesIsolationForest(
            contamination=0.1,
            n_estimators=100,
            max_samples='auto',
            random_state=42
        )

    def test_initialization(self):
        """Test if the model initializes with correct parameters."""
        self.assertEqual(self.model.contamination, 0.1)
        self.assertEqual(self.model.n_estimators, 100)
        self.assertEqual(self.model.random_state, 42)
        self.assertEqual(self.model.max_samples, 'auto')

    def test_fit(self):
        """Test model fitting."""
        # Test fitting with normal data
        self.model.fit(self.normal_data)
        self.assertTrue(hasattr(self.model, 'scaler_'))
        self.assertTrue(hasattr(self.model, 'base_estimator_'))

        # Test if the model maintains the timestamp column
        self.assertIn('timestamp', self.model.feature_names_)

    def test_predict(self):
        """Test anomaly prediction."""
        self.model.fit(self.normal_data)
        predictions = self.model.predict(self.anomalous_data)

        # Check if predictions have correct shape and values
        self.assertEqual(len(predictions), len(self.anomalous_data))
        self.assertTrue(np.all(np.isin(predictions, [-1, 1])))  # Should only contain -1 (anomaly) and 1 (normal)

        # Check if known anomalies are detected
        self.assertTrue(np.any(predictions[20:25] == -1))  # Spike should be detected
        self.assertTrue(np.any(predictions[50:55] == -1))  # Drop should be detected

    def test_score_samples(self):
        """Test anomaly scoring."""
        self.model.fit(self.normal_data)
        scores = self.model.score_samples(self.anomalous_data)

        # Check if scores have correct shape
        self.assertEqual(len(scores), len(self.anomalous_data))

        # Check if anomalous periods have lower scores
        normal_period_score = np.mean(scores[0:19])  # Before spike
        anomaly_period_score = np.mean(scores[20:25])  # During spike
        self.assertLess(anomaly_period_score, normal_period_score)

    def test_feature_importance(self):
        """Test feature importance calculation."""
        self.model.fit(self.normal_data)
        importance = self.model.feature_importance()

        # Check if importance scores are returned for all features
        self.assertEqual(len(importance), len(self.normal_data.columns))
        self.assertTrue(all(importance >= 0))  # Importance scores should be non-negative
        self.assertTrue(abs(sum(importance) - 1.0) < 1e-10)  # Should sum to 1

    def test_invalid_input(self):
        """Test model behavior with invalid inputs."""
        # Test with empty DataFrame
        with self.assertRaises(ValueError):
            self.model.fit(pd.DataFrame())

        # Test with missing values
        invalid_data = self.normal_data.copy()
        invalid_data.loc[0, 'value'] = np.nan
        with self.assertRaises(ValueError):
            self.model.fit(invalid_data)

        # Test with non-datetime timestamp
        invalid_data = self.normal_data.copy()
        invalid_data['timestamp'] = range(len(invalid_data))
        with self.assertRaises(ValueError):
            self.model.fit(invalid_data)

    def test_persistence(self):
        """Test model serialization and deserialization."""
        import pickle

        # Fit the model
        self.model.fit(self.normal_data)

        # Serialize
        serialized_model = pickle.dumps(self.model)

        # Deserialize
        loaded_model = pickle.loads(serialized_model)

        # Compare predictions
        original_predictions = self.model.predict(self.anomalous_data)
        loaded_predictions = loaded_model.predict(self.anomalous_data)

        np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_partial_fit(self):
        """Test incremental learning if supported."""
        if hasattr(self.model, 'partial_fit'):
            # Split data into two parts
            first_half = self.normal_data[:50]
            second_half = self.normal_data[50:]

            # Fit incrementally
            self.model.partial_fit(first_half)
            self.model.partial_fit(second_half)

            # Compare with single fit
            model_single_fit = AdvancedTimeSeriesIsolationForest(
                contamination=0.1,
                n_estimators=100,
                random_state=42
            )
            model_single_fit.fit(self.normal_data)

            # Predictions should be similar (not exactly equal due to incremental nature)
            pred_incremental = self.model.predict(self.anomalous_data)
            pred_single = model_single_fit.predict(self.anomalous_data)

            agreement_ratio = np.mean(pred_incremental == pred_single)
            self.assertGreater(agreement_ratio, 0.8)  # At least 80% agreement


if __name__ == '__main__':
    unittest.main()