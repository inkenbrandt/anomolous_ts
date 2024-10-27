"""
Unit tests for the TimeseriesGenerator class.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from timeseries_generator import TimeseriesGenerator

class TestTimeseriesGenerator:
    @pytest.fixture
    def generator(self):
        """Create a basic generator instance for testing."""
        return TimeseriesGenerator(start_date='2024-01-01', periods=100, frequency='D')

    def test_initialization(self):
        """Test proper initialization of TimeseriesGenerator."""
        generator = TimeseriesGenerator(start_date='2024-01-01', periods=100)
        
        assert generator.periods == 100
        assert generator.frequency == 'D'
        assert generator.start_date == pd.to_datetime('2024-01-01')
        assert len(generator.date_range) == 100
        assert generator.date_range[0] == pd.to_datetime('2024-01-01')

    def test_initialization_with_invalid_date(self):
        """Test initialization with invalid date format."""
        with pytest.raises(ValueError):
            TimeseriesGenerator(start_date='invalid_date')

    def test_seasonal_pattern_generation(self, generator):
        """Test seasonal pattern generation."""
        pattern = generator._generate_seasonal_pattern(period=10, amplitude=1.0, phase=0)
        
        assert isinstance(pattern, np.ndarray)
        assert len(pattern) == generator.periods
        assert np.isclose(np.max(pattern), 1.0, atol=0.1)
        assert np.isclose(np.min(pattern), -1.0, atol=0.1)

    def test_trend_generation(self, generator):
        """Test trend generation."""
        trend = generator._generate_trend(slope=0.1)
        
        assert isinstance(trend, np.ndarray)
        assert len(trend) == generator.periods
        assert trend[0] == 0
        assert np.isclose(trend[-1], (generator.periods - 1) * 0.1)

    def test_noise_addition(self, generator):
        """Test noise addition to data."""
        data = np.zeros(generator.periods)
        noisy_data = generator._add_noise(data, noise_level=1.0)
        
        assert isinstance(noisy_data, np.ndarray)
        assert len(noisy_data) == generator.periods
        assert np.std(noisy_data) < 1.5  # Should be close to noise_level=1.0
        assert not np.array_equal(noisy_data, data)  # Should be different from input

    def test_anomaly_generation(self, generator):
        """Test anomaly generation."""
        data = np.zeros(generator.periods)
        anomaly_data, anomaly_indices = generator._generate_anomalies(
            data, 
            num_anomalies=5,
            amplitude_range=(2, 4)
        )
        
        assert isinstance(anomaly_data, np.ndarray)
        assert isinstance(anomaly_indices, np.ndarray)
        assert len(anomaly_indices) == 5
        assert len(np.unique(anomaly_indices)) == 5  # All indices should be unique
        assert not np.array_equal(anomaly_data, data)  # Should have anomalies
        
        # Check if anomalies are within expected range
        anomaly_values = anomaly_data[anomaly_indices]
        assert np.all(np.abs(anomaly_values) > 0)  # All anomalies should be non-zero

    def test_correlated_series_generation(self, generator):
        """Test generation of correlated time series."""
        n_series = 3
        correlation_matrix = np.array([
            [1.0, 0.7, -0.3],
            [0.7, 1.0, -0.5],
            [-0.3, -0.5, 1.0]
        ])
        
        df, anomaly_locations = generator.generate_correlated_series(
            n_series=n_series,
            correlation_matrix=correlation_matrix,
            seasonal_periods=[10, 20],
            trend_slopes=[0.1, 0.05, -0.08],
            noise_levels=[0.1, 0.15, 0.12],
            num_anomalies=5
        )
        
        # Test DataFrame properties
        assert isinstance(df, pd.DataFrame)
        assert len(df) == generator.periods
        assert len(df.columns) == n_series
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Test anomaly locations
        assert isinstance(anomaly_locations, dict)
        assert len(anomaly_locations) == n_series
        for series in df.columns:
            assert len(anomaly_locations[series]) == 5

    def test_correlated_series_default_parameters(self, generator):
        """Test generation of correlated series with default parameters."""
        df, anomaly_locations = generator.generate_correlated_series()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 2  # Default n_series
        assert isinstance(anomaly_locations, dict)

    def test_correlation_matrix_validation(self, generator):
        """Test validation of correlation matrix."""
        # Test with invalid correlation matrix (not symmetric)
        invalid_correlation = np.array([
            [1.0, 0.7],
            [0.7, 1.0],
            [-0.3, -0.5]
        ])
        
        with pytest.raises(ValueError):
            generator.generate_correlated_series(
                n_series=3,
                correlation_matrix=invalid_correlation
            )

    @pytest.mark.parametrize("frequency,expected_periods", [
        ('D', 100),    # Daily
        ('H', 100),    # Hourly
        ('W', 100),    # Weekly
        ('M', 100),    # Monthly
    ])
    def test_different_frequencies(self, frequency, expected_periods):
        """Test generator with different frequencies."""
        generator = TimeseriesGenerator(
            start_date='2024-01-01',
            periods=expected_periods,
            frequency=frequency
        )
        
        df, _ = generator.generate_correlated_series()
        assert len(df) == expected_periods
        assert df.index.freq == frequency

    def test_reproducibility(self):
        """Test reproducibility with fixed random seed."""
        np.random.seed(42)
        gen1 = TimeseriesGenerator(start_date='2024-01-01', periods=100)
        df1, _ = gen1.generate_correlated_series()
        
        np.random.seed(42)
        gen2 = TimeseriesGenerator(start_date='2024-01-01', periods=100)
        df2, _ = gen2.generate_correlated_series()
        
        pd.testing.assert_frame_equal(df1, df2)

    def test_extreme_values(self, generator):
        """Test generator with extreme parameter values."""
        # Test with very large amplitude seasonal pattern
        pattern = generator._generate_seasonal_pattern(period=10, amplitude=1000.0)
        assert np.max(np.abs(pattern)) > 900  # Should reflect large amplitude

        # Test with very large trend
        trend = generator._generate_trend(slope=100.0)
        assert trend[-1] > trend[0]  # Should show clear upward trend
        
        # Test with very high noise level
        data = np.zeros(generator.periods)
        noisy_data = generator._add_noise(data, noise_level=10.0)
        assert np.std(noisy_data) > 5  # Should show high variance

    def test_input_validation(self):
        """Test input validation for various parameters."""
        with pytest.raises(ValueError):
            TimeseriesGenerator(periods=-1)  # Invalid negative periods
            
        with pytest.raises(ValueError):
            TimeseriesGenerator(frequency='INVALID')  # Invalid frequency
            
        generator = TimeseriesGenerator()
        with pytest.raises(ValueError):
            generator.generate_correlated_series(n_series=0)  # Invalid number of series

if __name__ == '__main__':
    pytest.main([__file__])
