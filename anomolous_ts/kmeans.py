import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
from scipy.spatial.distance import cdist

from anomolous_ts.preprocessor import TimeSeriesPreprocessor
from anomolous_ts.visualizer import TimeSeriesVisualizer

class TimeSeriesKMeansDetector:
    """
    Time series anomaly detection using K-means clustering.
    Detects anomalies by identifying points that are far from their cluster centers.
    """

    def __init__(
            self,
            window_size=10,
            n_clusters=3,
            anomaly_threshold=2.0,
            seasonal_period=None,
            preprocessor=None,
            visualizer=None,
            random_state=42
    ):
        """
        Initialize the K-means time series detector.

        Parameters
        ----------
        window_size : int, default=10
            Size of sliding window for feature extraction
        n_clusters : int, default=3
            Number of clusters for K-means
        anomaly_threshold : float, default=2.0
            Number of standard deviations from cluster center to consider as anomaly
        seasonal_period : int, optional
            Period for seasonal feature extraction
        preprocessor : TimeSeriesPreprocessor, optional
            Custom preprocessor instance
        visualizer : TimeSeriesVisualizer, optional
            Custom visualizer instance
        random_state : int, default=42
            Random state for reproducibility
        """
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.anomaly_threshold = anomaly_threshold
        self.seasonal_period = seasonal_period
        self.random_state = random_state

        # Initialize K-means model
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state
        )

        # Initialize preprocessor and visualizer
        self.preprocessor = preprocessor or TimeSeriesPreprocessor()
        self.visualizer = visualizer or TimeSeriesVisualizer()

        # Store cluster centers and distances
        self.cluster_centers_ = None
        self.cluster_distances_ = None
        self.labels_ = None

    def _create_temporal_features(self, data):
        """
        Create temporal features using sliding windows.

        Parameters
        ----------
        data : np.array
            Input time series data

        Returns
        -------
        np.array
            Feature matrix
        """
        n_samples = len(data)
        features_list = []

        for i in range(self.window_size, n_samples):
            window = data[i - self.window_size:i]

            feature_vector = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.median(window),
                np.ptp(window),
                data[i],
                data[i] - window[-1],
                np.sum(np.diff(window) > 0),
                stats.skew(window),
                stats.kurtosis(window),
                np.percentile(window, 25),
                np.percentile(window, 75)
            ]

            # Add seasonal features if specified
            if self.seasonal_period and i >= self.seasonal_period:
                seasonal_features = [
                    data[i] - data[i - self.seasonal_period],
                    np.mean(data[i - self.seasonal_period:i:self.seasonal_period]),
                    np.std(data[i - self.seasonal_period:i:self.seasonal_period])
                ]
                feature_vector.extend(seasonal_features)
            else:
                feature_vector.extend([0, 0, 0])

            features_list.append(feature_vector)

        # Pad initial window with zeros
        padding = np.zeros((self.window_size, len(features_list[0])))
        return np.vstack((padding, np.array(features_list)))

    def _calculate_anomaly_scores(self, features):
        """
        Calculate anomaly scores based on distance to cluster centers.

        Parameters
        ----------
        features : np.array
            Feature matrix

        Returns
        -------
        np.array
            Anomaly scores
        """
        # Calculate distances to assigned cluster centers
        distances = np.min(cdist(features, self.cluster_centers_), axis=1)

        # Calculate z-scores of distances
        z_scores = stats.zscore(distances)

        # Convert to probability scores using sigmoid function
        scores = 1 / (1 + np.exp(-z_scores))

        return scores

    def fit_predict(self, data, return_scores=False):
        """
        Fit the model and predict anomalies.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Input time series data
        return_scores : bool, default=False
            Whether to return anomaly scores

        Returns
        -------
        tuple or pd.Series/DataFrame
            Boolean indicators for anomalies and optionally anomaly scores
        """
        # Preprocess data
        preprocessed_data = self.preprocessor.impute_missing_values(data)

        # Create features
        if isinstance(preprocessed_data, pd.DataFrame):
            features_list = []
            for column in preprocessed_data.columns:
                features = self._create_temporal_features(preprocessed_data[column].values)
                features_list.append(features)
            features = np.hstack(features_list)
        else:
            features = self._create_temporal_features(preprocessed_data.values)

        # Scale features
        if self.preprocessor.scaler is not None:
            features = self.preprocessor.scaler.fit_transform(features)

        # Fit K-means and get cluster assignments
        self.labels_ = self.model.fit_predict(features)
        self.cluster_centers_ = self.model.cluster_centers_

        # Calculate anomaly scores
        anomaly_scores = self._calculate_anomaly_scores(features)

        # Determine anomalies based on threshold
        is_anomaly = anomaly_scores > self.anomaly_threshold

        # Prepare output
        if isinstance(data, pd.DataFrame):
            # Split results for multivariate case
            n_vars = len(data.columns)
            anomalies = np.array_split(is_anomaly, n_vars)
            scores = np.array_split(anomaly_scores, n_vars)

            anomalies_df = pd.DataFrame(
                np.column_stack(anomalies),
                index=data.index,
                columns=data.columns
            )

            if return_scores:
                scores_df = pd.DataFrame(
                    np.column_stack(scores),
                    index=data.index,
                    columns=data.columns
                )
                return anomalies_df, scores_df
            return anomalies_df

        else:
            anomalies = pd.Series(is_anomaly, index=data.index)
            if return_scores:
                scores = pd.Series(anomaly_scores, index=data.index)
                return anomalies, scores
            return anomalies

    def detect_and_visualize(self, data, title="K-means Clustering Anomaly Detection"):
        """
        Detect anomalies and visualize results in one step.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Input time series data
        title : str
            Plot title

        Returns
        -------
        tuple
            (anomalies, anomaly_scores)
        """
        # Detect anomalies with scores
        anomalies, scores = self.fit_predict(data, return_scores=True)

        # Create cluster-based visualization
        self.visualizer.plot_clusters(
            data,
            self.labels_,
            anomalies,
            scores,
            self.n_clusters,
            title
        )

        return anomalies, scores

    def get_cluster_profiles(self):
        """
        Get statistical profiles of each cluster.

        Returns
        -------
        dict
            Dictionary containing cluster statistics
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model hasn't been fitted yet!")

        profiles = {}
        for i in range(self.n_clusters):
            cluster_points = self.labels_ == i
            profiles[f'Cluster_{i}'] = {
                'size': np.sum(cluster_points),
                'center': self.cluster_centers_[i],
                'mean_distance': np.mean(self.cluster_distances_[cluster_points]),
                'std_distance': np.std(self.cluster_distances_[cluster_points])
            }

        return profiles

    def process_stream(self, data_stream, chunk_size=100):
        """
        Process streaming data in chunks.

        Parameters
        ----------
        data_stream : iterator
            Iterator yielding new data points
        chunk_size : int, default=100
            Size of data chunks to process at once

        Yields
        ------
        tuple
            (chunk_data, anomalies, scores)
        """
        chunk = []
        for data_point in data_stream:
            chunk.append(data_point)

            if len(chunk) >= chunk_size:
                # Convert chunk to appropriate format
                if isinstance(data_point, tuple):
                    chunk_data = pd.DataFrame(chunk)
                else:
                    chunk_data = pd.Series(chunk)

                # Detect anomalies
                anomalies, scores = self.fit_predict(chunk_data, return_scores=True)

                # Visualize results
                self.visualizer.plot_clusters(
                    chunk_data,
                    self.labels_,
                    anomalies,
                    scores,
                    self.n_clusters,
                    f'Streaming Anomaly Detection - Chunk Size {chunk_size}'
                )

                yield chunk_data, anomalies, scores
                chunk = []


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    # Set random seed for reproducibility
    np.random.seed(42)


    def generate_synthetic_data(n_points=1000, n_patterns=3):
        """Generate synthetic time series with multiple patterns and anomalies."""
        dates = pd.date_range('2024-01-01', periods=n_points, freq='H')

        # Generate different patterns
        t = np.linspace(0, 8 * np.pi, n_points)
        patterns = []

        # Pattern 1: Sine wave with daily seasonality
        p1 = 10 * np.sin(2 * np.pi * np.arange(n_points) / 24)
        patterns.append(p1)

        # Pattern 2: Saw tooth pattern
        p2 = 5 * stats.zscore(np.abs(np.mod(t, np.pi) - np.pi / 2))
        patterns.append(p2)

        # Pattern 3: Square wave
        p3 = 7 * np.sign(np.sin(t / 4))
        patterns.append(p3)

        # Combine patterns with transitions
        signal = np.zeros(n_points)
        pattern_length = n_points // n_patterns
        for i in range(n_patterns):
            start_idx = i * pattern_length
            end_idx = (i + 1) * pattern_length
            signal[start_idx:end_idx] = patterns[i][:pattern_length]

        # Add trend
        trend = 0.01 * np.arange(n_points)
        signal += trend

        # Add noise
        noise = np.random.normal(0, 0.5, n_points)
        signal += noise

        # Add anomalies
        n_anomalies = 50
        anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
        signal[anomaly_indices] += np.random.normal(0, 5, n_anomalies)

        # Create multivariate series
        df = pd.DataFrame({
            'pattern1': signal,
            'pattern2': np.roll(signal, 24) + np.random.normal(0, 1, n_points),
        }, index=dates)

        # Add some missing values
        missing_mask = np.random.random(n_points) < 0.05
        df.loc[missing_mask, 'pattern2'] = np.nan

        return df, anomaly_indices


    # Generate data
    print("Generating synthetic data...")
    data, true_anomalies = generate_synthetic_data()

    # Initialize detector with different configurations
    configs = [
        {
            'window_size': 24,
            'n_clusters': 3,
            'anomaly_threshold': 2.0,
            'seasonal_period': 24,
        },
        {
            'window_size': 12,
            'n_clusters': 5,
            'anomaly_threshold': 2.5,
            'seasonal_period': None,
        },
        {
            'window_size': 48,
            'n_clusters': 4,
            'anomaly_threshold': 1.5,
            'seasonal_period': 24,
        }
    ]

    for i, config in enumerate(configs, 1):
        print(f"\nTesting Configuration {i}:")
        print(f"Parameters: {config}")

        # Initialize detector
        detector = TimeSeriesKMeansDetector(**config)

        # Detect anomalies
        print("\nDetecting anomalies...")
        anomalies, scores = detector.detect_and_visualize(
            data,
            title=f"K-means Clustering Results - Configuration {i}"
        )

        # Print statistics
        print("\nDetection Statistics:")
        for column in data.columns:
            n_anomalies = anomalies[column].sum()
            avg_score = scores[column][anomalies[column]].mean()
            print(f"\n{column}:")
            print(f"Number of anomalies detected: {n_anomalies}")
            print(f"Average anomaly score: {avg_score:.3f}")

        # Get cluster profiles
        print("\nCluster Profiles:")
        profiles = detector.get_cluster_profiles()
        for cluster, profile in profiles.items():
            print(f"\n{cluster}:")
            print(f"Size: {profile['size']}")
            print(f"Mean distance: {profile['mean_distance']:.3f}")
            print(f"Std distance: {profile['std_distance']:.3f}")

    # Demonstrate streaming detection
    print("\nDemonstrating streaming detection...")


    def simulate_stream(data, chunk_size=100):
        """Simulate streaming data from DataFrame."""
        for i in range(0, len(data), chunk_size):
            yield data.iloc[i:i + chunk_size]


    # Initialize detector for streaming
    streaming_detector = TimeSeriesKMeansDetector(
        window_size=24,
        n_clusters=3,
        anomaly_threshold=2.0,
        seasonal_period=24
    )

    # Process streaming data
    stream = simulate_stream(data, chunk_size=100)
    for i, (chunk_data, chunk_anomalies, chunk_scores) in enumerate(
            streaming_detector.process_stream(stream), 1
    ):
        print(f"\nProcessed chunk {i}:")
        print(f"Chunk size: {len(chunk_data)}")
        print(f"Anomalies detected: {chunk_anomalies.sum().sum()}")
        print(f"Average anomaly score: {chunk_scores.mean().mean():.3f}")

        # Break after a few chunks for demonstration
        if i >= 5:
            break

    # Compare with different preprocessing configurations
    print("\nTesting different preprocessing configurations...")

    preprocessor_configs = [
        {
            'imputation_method': 'linear',
            'scaling_method': 'standard'
        },
        {
            'imputation_method': 'knn',
            'scaling_method': 'robust'
        },
        {
            'imputation_method': 'hybrid',
            'scaling_method': 'minmax'
        }
    ]

    for i, prep_config in enumerate(preprocessor_configs, 1):
        print(f"\nPreprocessing Configuration {i}:")
        print(f"Parameters: {prep_config}")

        # Initialize preprocessor and detector
        preprocessor = TimeSeriesPreprocessor(**prep_config)
        detector = TimeSeriesKMeansDetector(
            window_size=24,
            n_clusters=3,
            preprocessor=preprocessor
        )

        # Detect anomalies
        anomalies, scores = detector.detect_and_visualize(
            data,
            title=f"Results with Preprocessing Configuration {i}"
        )

        # Print summary
        print("\nResults Summary:")
        print(f"Total anomalies detected: {anomalies.sum().sum()}")
        print(f"Average anomaly score: {scores.mean().mean():.3f}")

    print("\nProcessing complete!")