import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import the TimeSeriesVisualizer class
from anomolous_ts import TimeSeriesVisualizer  # Adjust import path as needed

@pytest.fixture
def visualizer():
    return TimeSeriesVisualizer()

@pytest.fixture
def sample_dates():
    base_date = datetime(2024, 1, 1)
    return [base_date + timedelta(days=x) for x in range(10)]

@pytest.fixture
def single_series_data(sample_dates):
    return pd.Series(
        data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        index=sample_dates
    )

@pytest.fixture
def multi_series_data(sample_dates):
    return pd.DataFrame({
        'series1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'series2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    }, index=sample_dates)

@pytest.fixture
def single_series_anomalies(sample_dates):
    return pd.Series(
        data=[False, False, True, False, True, False, False, True, False, False],
        index=sample_dates
    )

@pytest.fixture
def multi_series_anomalies(sample_dates):
    return pd.DataFrame({
        'series1': [False, False, True, False, True, False, False, True, False, False],
        'series2': [True, False, False, True, False, False, True, False, False, True]
    }, index=sample_dates)

@pytest.fixture
def confidence_scores(sample_dates):
    return pd.Series(
        data=[0.1, 0.2, 0.8, 0.3, 0.9, 0.2, 0.3, 0.85, 0.2, 0.1],
        index=sample_dates
    )

@pytest.fixture
def multi_confidence_scores(sample_dates):
    return pd.DataFrame({
        'series1': [0.1, 0.2, 0.8, 0.3, 0.9, 0.2, 0.3, 0.85, 0.2, 0.1],
        'series2': [0.95, 0.2, 0.3, 0.87, 0.2, 0.3, 0.92, 0.1, 0.2, 0.88]
    }, index=sample_dates)

class TestTimeSeriesVisualizer:
    
    def test_plot_anomalies_single_series(self, visualizer, single_series_data, single_series_anomalies):
        """Test plotting anomalies for a single time series"""
        plt.close('all')  # Close any existing plots
        visualizer.plot_anomalies(single_series_data, single_series_anomalies)
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close('all')

    def test_plot_anomalies_single_series_with_confidence(self, visualizer, single_series_data, 
                                                        single_series_anomalies, confidence_scores):
        """Test plotting anomalies with confidence scores for a single time series"""
        plt.close('all')
        visualizer.plot_anomalies(single_series_data, single_series_anomalies, confidence_scores)
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close('all')

    def test_plot_anomalies_multi_series(self, visualizer, multi_series_data, multi_series_anomalies):
        """Test plotting anomalies for multiple time series"""
        plt.close('all')
        visualizer.plot_anomalies(multi_series_data, multi_series_anomalies)
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close('all')

    def test_plot_anomalies_multi_series_with_confidence(self, visualizer, multi_series_data, 
                                                       multi_series_anomalies, multi_confidence_scores):
        """Test plotting anomalies with confidence scores for multiple time series"""
        plt.close('all')
        visualizer.plot_anomalies(multi_series_data, multi_series_anomalies, multi_confidence_scores)
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close('all')

    def test_plot_imputation_single_series(self, visualizer, single_series_data):
        """Test plotting imputation results for a single time series"""
        plt.close('all')
        # Create data with missing values
        data_with_missing = single_series_data.copy()
        data_with_missing.iloc[2:4] = np.nan
        
        # Create imputed data
        imputed_data = single_series_data.copy()
        
        visualizer.plot_imputation(data_with_missing, imputed_data)
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close('all')

    def test_plot_imputation_multi_series(self, visualizer, multi_series_data):
        """Test plotting imputation results for multiple time series"""
        plt.close('all')
        # Create data with missing values
        data_with_missing = multi_series_data.copy()
        data_with_missing.iloc[2:4, 0] = np.nan
        data_with_missing.iloc[5:7, 1] = np.nan
        
        # Create imputed data
        imputed_data = multi_series_data.copy()
        
        visualizer.plot_imputation(data_with_missing, imputed_data)
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close('all')

    def test_invalid_input_dimensions(self, visualizer, single_series_data, multi_series_anomalies):
        """Test handling of mismatched dimensions between data and anomalies"""
        plt.close('all')
        with pytest.raises(Exception):
            visualizer.plot_anomalies(single_series_data, multi_series_anomalies)
        plt.close('all')

    def test_empty_anomalies(self, visualizer, single_series_data):
        """Test plotting with no anomalies"""
        plt.close('all')
        empty_anomalies = pd.Series(False, index=single_series_data.index)
        visualizer.plot_anomalies(single_series_data, empty_anomalies)
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close('all')
