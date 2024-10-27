from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from anomolous_ts.visualizer import TimeSeriesVisualizer
from anomolous_ts.isoforest import AdvancedTimeSeriesIsolationForest

class TimeSeriesPreprocessor:
    """
    Handles preprocessing of time series data including imputation,
    scaling, and decomposition.
    """

    def __init__(
            self,
            imputation_method='linear',
            scaling_method='standard',
            decomposition_method=None,
            knn_neighbors=5
    ):
        self.imputation_method = imputation_method
        self.scaling_method = scaling_method
        self.decomposition_method = decomposition_method
        self.knn_neighbors = knn_neighbors
        self._setup_scalers()

    def _setup_scalers(self):
        """Initialize scalers based on method chosen"""
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

    def impute_missing_values(self, data):
        """
        Impute missing values using the specified method.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Time series data with missing values

        Returns
        -------
        pd.Series or pd.DataFrame
            Data with imputed values
        """
        if self.imputation_method == 'linear':
            return data.interpolate(method='linear', axis=0)
        elif self.imputation_method == 'spline':
            return data.interpolate(method='spline', order=3, axis=0)
        elif self.imputation_method == 'polynomial':
            return data.interpolate(method='polynomial', order=2, axis=0)
        elif self.imputation_method == 'forward':
            return data.fillna(method='ffill')
        elif self.imputation_method == 'backward':
            return data.fillna(method='bfill')
        elif self.imputation_method == 'mean':
            return data.fillna(data.mean())
        elif self.imputation_method == 'median':
            return data.fillna(data.median())
        elif self.imputation_method == 'knn':
            imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            if isinstance(data, pd.Series):
                return pd.Series(
                    imputer.fit_transform(data.values.reshape(-1, 1)).ravel(),
                    index=data.index
                )
            else:
                return pd.DataFrame(
                    imputer.fit_transform(data),
                    index=data.index,
                    columns=data.columns
                )
        elif self.imputation_method == 'hybrid':
            # Use KNN for gaps smaller than knn_neighbors, spline for larger gaps
            tmp = data.copy()
            small_gaps = tmp.isna() & tmp.rolling(self.knn_neighbors).count().notna()
            large_gaps = tmp.isna() & ~small_gaps

            # Fill small gaps with KNN
            if small_gaps.any().any():
                tmp[small_gaps] = self.impute_missing_values(tmp).where(small_gaps)

            # Fill large gaps with spline interpolation
            if large_gaps.any().any():
                tmp = tmp.interpolate(method='spline', order=3, axis=0)

            return tmp



if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    # Set random seed for reproducibility
    np.random.seed(42)


    # Generate sample data
    def generate_sample_data(n_points=1000):
        # Create timestamp index
        dates = pd.date_range('2024-01-01', periods=n_points, freq='H')

        # Generate base signal with daily seasonality
        t = np.linspace(0, 4 * np.pi, n_points)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / 24)  # Daily seasonality
        trend = 0.01 * np.arange(n_points)  # Upward trend
        noise = np.random.normal(0, 0.5, n_points)

        # Add some anomalies
        anomalies = np.zeros(n_points)
        anomaly_indices = np.random.choice(n_points, 50, replace=False)
        anomalies[anomaly_indices] = np.random.normal(0, 5, 50)

        # Combine components
        series1 = seasonal + trend + noise + anomalies

        # Create second series with different pattern and some missing values
        series2 = seasonal.copy() + np.random.normal(0, 1, n_points)
        series2_anomalies = np.zeros(n_points)
        series2_anomalies[np.random.choice(n_points, 30, replace=False)] = np.random.normal(0, 3, 30)
        series2 += series2_anomalies

        # Add missing values
        missing_indices = np.random.choice(n_points, 100, replace=False)
        series2[missing_indices] = np.nan

        return pd.DataFrame({
            'series1': series1,
            'series2': series2
        }, index=dates)


    # Generate data
    print("Generating sample data...")
    data = generate_sample_data()

    # Initialize preprocessor with different configurations
    print("\nTesting different preprocessing configurations...")
    imputation_methods = ['linear', 'spline', 'polynomial', 'knn', 'hybrid']
    scaling_methods = ['standard', 'robust', 'minmax']

    # Test each imputation method
    for imp_method in imputation_methods:
        print(f"\nTesting {imp_method} imputation...")
        preprocessor = TimeSeriesPreprocessor(
            imputation_method=imp_method,
            scaling_method='standard'
        )

        # Store original data for visualization
        original_data = data.copy()

        # Perform imputation
        imputed_data = preprocessor.impute_missing_values(data)

        # Create visualizer and plot results
        visualizer = TimeSeriesVisualizer()
        visualizer.plot_imputation(
            original_data,
            imputed_data,
            title=f'Imputation Results - {imp_method.capitalize()} Method'
        )

        # Print some statistics
        print(f"Missing values before: {original_data.isna().sum().sum()}")
        print(f"Missing values after: {imputed_data.isna().sum().sum()}")

        if imp_method == 'knn':
            print("Note: KNN imputation uses surrounding points for estimation")
        elif imp_method == 'hybrid':
            print("Note: Hybrid method combines KNN for small gaps and spline for large gaps")

    # Test detection with different configurations
    print("\nTesting anomaly detection with different configurations...")

    # Initialize detector with advanced configuration
    detector = AdvancedTimeSeriesIsolationForest(
        window_size=24,  # 24-hour window
        n_estimators=100,
        contamination=0.05,
        seasonal_period=24,  # Daily seasonality
        imputation_method='hybrid',
        scaling_method='robust'
    )

    # Detect anomalies with confidence scores
    print("\nDetecting anomalies...")
    anomalies, confidence_scores = detector.fit_predict(data, return_confidence=True)

    # Visualize results
    print("\nVisualizing results...")
    detector.visualizer.plot_anomalies(
        data,
        anomalies,
        confidence_scores,
        title='Anomaly Detection Results with Confidence Scores'
    )

    # Print summary statistics
    print("\nSummary Statistics:")
    for column in data.columns:
        n_anomalies = anomalies[column].sum()
        avg_confidence = confidence_scores[column][anomalies[column]].mean()
        print(f"\n{column}:")
        print(f"Number of anomalies detected: {n_anomalies}")
        print(f"Average confidence score: {avg_confidence:.3f}")

        # Print top 5 anomalies by confidence
        print("\nTop 5 anomalies by confidence:")
        top_anomalies = confidence_scores[column][anomalies[column]].nlargest(5)
        for idx, score in top_anomalies.items():
            print(f"Timestamp: {idx}, Value: {data[column][idx]:.2f}, "
                  f"Confidence: {score:.3f}")

    # Example of handling real-time data
    print("\nSimulating real-time data processing...")

    # Create a small real-time simulation
    current_time = datetime.now()
    real_time_data = []
    for i in range(100):
        # Generate a new data point
        timestamp = current_time + timedelta(minutes=i)
        value = np.sin(i / 10) + np.random.normal(0, 0.1)

        # Add occasional anomaly
        if np.random.random() < 0.05:  # 5% chance of anomaly
            value += np.random.normal(0, 2)

        real_time_data.append({
            'timestamp': timestamp,
            'value': value
        })

    real_time_df = pd.DataFrame(real_time_data).set_index('timestamp')

    # Process real-time data in chunks
    chunk_size = 20
    for i in range(0, len(real_time_df), chunk_size):
        chunk = real_time_df.iloc[i:i + chunk_size]

        # Detect anomalies in chunk
        chunk_anomalies, chunk_confidence = detector.fit_predict(
            chunk['value'],
            return_confidence=True
        )

        # Visualize chunk results
        detector.visualizer.plot_anomalies(
            chunk['value'],
            chunk_anomalies,
            chunk_confidence,
            title=f'Real-time Anomaly Detection - Chunk {i // chunk_size + 1}'
        )

        # Print chunk summary
        n_anomalies = chunk_anomalies.sum()
        if n_anomalies > 0:
            print(f"\nChunk {i // chunk_size + 1} anomalies detected: {n_anomalies}")
            print("Anomaly timestamps:")
            for idx in chunk.index[chunk_anomalies]:
                print(f"Timestamp: {idx}, Value: {chunk['value'][idx]:.2f}, "
                      f"Confidence: {chunk_confidence[idx]:.3f}")

    print("\nProcessing complete!")