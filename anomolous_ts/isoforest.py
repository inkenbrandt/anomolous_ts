import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats

from anomolous_ts.preprocessor import TimeSeriesPreprocessor
from anomolous_ts.visualizer import TimeSeriesVisualizer



class TimeSeriesIsolationForest:
    """
    Time Series Anomaly Detection using Isolation Forest.

    This implementation enhances the standard Isolation Forest algorithm for time series data by:
    1. Incorporating temporal features through sliding windows
    2. Adding seasonal and trend decomposition
    3. Supporting both univariate and multivariate time series
    4. Providing contextual anomaly detection
    5. Handling missing values through imputation
    """

    def __init__(
            self,
            window_size=5,
            n_estimators=100,
            contamination='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            standardize=True,
            seasonal_period=None,
            imputation_method='linear'
    ):
        """
        Initialize the Time Series Isolation Forest detector.

        Parameters
        ----------
        window_size : int, default=5
            Size of the sliding window for temporal feature extraction

        n_estimators : int, default=100
            Number of base estimators (isolation trees) in the ensemble

        contamination : float or 'auto', default='auto'
            Expected proportion of outliers in the data set

        max_features : int or float, default=1.0
            Number of features to draw from X to train each base estimator

        bootstrap : bool, default=False
            If True, individual trees are fit on random subsets of the training data

        n_jobs : int, default=-1
            Number of parallel jobs to run. -1 means using all processors

        standardize : bool, default=True
            Whether to standardize the features before fitting

        seasonal_period : int, optional
            If provided, extracts seasonal features with this period

        imputation_method : str, default='linear'
            Method to use for handling missing values:
            - 'linear': Linear interpolation
            - 'forward': Forward fill
            - 'backward': Backward fill
            - 'mean': Mean imputation
            - 'median': Median imputation
        """
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.standardize = standardize
        self.seasonal_period = seasonal_period
        self.imputation_method = imputation_method

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=42
        )

        self.scaler = StandardScaler() if standardize else None

    def _impute_missing_values(self, data):
        """
        Impute missing values in the data.

        Parameters
        ----------
        data : np.array
            Input data with possible missing values

        Returns
        -------
        np.array
            Data with imputed values
        """
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            if self.imputation_method == 'linear':
                return data.interpolate(method='linear', axis=0)
            elif self.imputation_method == 'forward':
                return data.fillna(method='ffill')
            elif self.imputation_method == 'backward':
                return data.fillna(method='bfill')
            elif self.imputation_method == 'mean':
                return data.fillna(data.mean())
            elif self.imputation_method == 'median':
                return data.fillna(data.median())
        else:
            if self.imputation_method in ['mean', 'median']:
                imputer = SimpleImputer(
                    strategy=self.imputation_method,
                    missing_values=np.nan
                )
                return imputer.fit_transform(data.reshape(-1, 1)).ravel()
            else:
                # Convert to pandas for interpolation
                series = pd.Series(data)
                imputed = self._impute_missing_values(series)
                return imputed.values

    def _create_temporal_features(self, data):
        """
        Create temporal features from the time series data using sliding windows.
        """
        # Impute missing values first
        data = self._impute_missing_values(data)

        n_samples = len(data)
        features_list = []

        # Basic temporal features
        for i in range(self.window_size, n_samples):
            window = data[i - self.window_size:i]

            feature_vector = [
                np.mean(window),  # Mean of window
                np.std(window),  # Standard deviation
                np.min(window),  # Minimum
                np.max(window),  # Maximum
                np.median(window),  # Median
                np.ptp(window),  # Peak-to-peak (range)
                data[i],  # Current value
                data[i] - window[-1],  # Change from previous
                np.sum(np.diff(window) > 0)  # Number of increases in window
            ]

            # Add seasonal features if period is specified
            if self.seasonal_period:
                if i >= self.seasonal_period:
                    seasonal_features = [
                        data[i] - data[i - self.seasonal_period],  # Seasonal difference
                        np.mean(data[i - self.seasonal_period:i:self.seasonal_period])  # Seasonal mean
                    ]
                    feature_vector.extend(seasonal_features)
                else:
                    feature_vector.extend([0, 0])  # Padding for early points

            features_list.append(feature_vector)

        # Pad the initial window_size points with zeros
        padding = np.zeros((self.window_size, len(features_list[0])))
        features = np.vstack((padding, np.array(features_list)))

        return features

    def _process_multivariate_data(self, data):
        """
        Process multivariate time series data.
        """
        if isinstance(data, pd.DataFrame):
            # Impute missing values first
            data = self._impute_missing_values(data)
            data = data.values

        n_variables = data.shape[1]
        all_features = []

        # Create temporal features for each variable
        for i in range(n_variables):
            var_features = self._create_temporal_features(data[:, i])
            all_features.append(var_features)

        # Combine features from all variables
        return np.hstack(all_features)

    def fit(self, data):
        """
        Fit the isolation forest model to the time series data.
        """
        # Convert to numpy array if necessary
        if isinstance(data, pd.Series):
            data = data.values

        # Process univariate or multivariate data
        if len(data.shape) == 1:
            features = self._create_temporal_features(data)
        else:
            features = self._process_multivariate_data(data)

        # Standardize features if requested
        if self.standardize:
            features = self.scaler.fit_transform(features)

        # Fit the isolation forest
        self.model.fit(features)
        return self

    def predict(self, data):
        """
        Predict anomalies in the time series data.
        """
        # Convert to numpy array if necessary
        if isinstance(data, pd.Series):
            data = data.values

        # Process univariate or multivariate data
        if len(data.shape) == 1:
            features = self._create_temporal_features(data)
        else:
            features = self._process_multivariate_data(data)

        # Standardize features if requested
        if self.standardize:
            features = self.scaler.transform(features)

        return self.model.predict(features)

    def detect(self, data):
        """
        Detect anomalies in the time series data.
        """
        # Fit and predict
        self.fit(data)
        predictions = self.predict(data)

        # Convert predictions to boolean mask
        is_anomaly = predictions == -1

        # Return results in the same format as input
        if isinstance(data, pd.Series):
            return pd.Series(is_anomaly, index=data.index)
        elif isinstance(data, pd.DataFrame):
            return pd.DataFrame(is_anomaly, index=data.index, columns=data.columns)
        else:
            return is_anomaly




class AdvancedTimeSeriesIsolationForest:
    """
    Enhanced Isolation Forest algorithm specifically adapted for time series anomaly detection.
    Uses separate preprocessing and visualization components for modularity.
    """

    def __init__(
            self,
            window_size=5,
            n_estimators=100,
            contamination='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            seasonal_period=None,
            preprocessor=None,
            visualizer=None
    ):
        """
        Initialize the Advanced Time Series Isolation Forest detector.

        Parameters
        ----------
        window_size : int, default=5
            Size of sliding window for temporal feature extraction
        n_estimators : int, default=100
            Number of isolation trees in the ensemble
        contamination : float or 'auto', default='auto'
            Expected proportion of outliers in the dataset
        max_features : float, default=1.0
            Number of features to draw for each tree
        bootstrap : bool, default=False
            Whether to use bootstrapping
        n_jobs : int, default=-1
            Number of parallel jobs
        seasonal_period : int, optional
            Period for seasonal decomposition
        preprocessor : TimeSeriesPreprocessor, optional
            Custom preprocessor instance for data preparation
        visualizer : TimeSeriesVisualizer, optional
            Custom visualizer instance for plotting results
        """
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.seasonal_period = seasonal_period

        # Initialize the isolation forest model
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=42
        )

        # Initialize preprocessor and visualizer if not provided
        if preprocessor is None:
            self.preprocessor = TimeSeriesPreprocessor()
        else:
            self.preprocessor = preprocessor

        if visualizer is None:
            self.visualizer = TimeSeriesVisualizer()
        else:
            self.visualizer = visualizer

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

            # Basic statistical features
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

    def _calculate_confidence_scores(self, features):
        """
        Calculate confidence scores for anomalies.

        Parameters
        ----------
        features : np.array
            Feature matrix

        Returns
        -------
        np.array
            Confidence scores between 0 and 1
        """
        scores = -self.model.score_samples(features)
        return 1 / (1 + np.exp(-scores))  # Sigmoid transformation

    def fit_predict(self, data, return_confidence=False):
        """
        Fit the model and predict anomalies with optional confidence scores.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Input time series data
        return_confidence : bool, default=False
            Whether to return confidence scores

        Returns
        -------
        tuple or pd.Series/DataFrame
            Boolean indicators for anomalies and optionally confidence scores
        """
        # Store original data for reference
        original_data = data.copy()

        # Preprocess data using the TimeSeriesPreprocessor
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

        # Scale features if scaler is available in preprocessor
        if self.preprocessor.scaler is not None:
            features = self.preprocessor.scaler.fit_transform(features)

        # Fit model and predict
        self.model.fit(features)
        predictions = self.model.predict(features)
        is_anomaly = predictions == -1

        # Calculate confidence scores if requested
        if return_confidence:
            confidence_scores = self._calculate_confidence_scores(features)

            if isinstance(data, pd.DataFrame):
                confidence_df = pd.DataFrame(
                    np.split(confidence_scores, len(data.columns)),
                    index=data.index,
                    columns=data.columns
                )
                return (
                    pd.DataFrame(is_anomaly, index=data.index, columns=data.columns),
                    confidence_df
                )
            else:
                return (
                    pd.Series(is_anomaly, index=data.index),
                    pd.Series(confidence_scores, index=data.index)
                )

        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(is_anomaly, index=data.index, columns=data.columns)
        return pd.Series(is_anomaly, index=data.index)

    def detect_and_visualize(self, data, title='Anomaly Detection Results'):
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
            (anomalies, confidence_scores)
        """
        # Detect anomalies with confidence scores
        anomalies, confidence_scores = self.fit_predict(data, return_confidence=True)

        # Visualize results using the TimeSeriesVisualizer
        self.visualizer.plot_anomalies(data, anomalies, confidence_scores, title)

        return anomalies, confidence_scores

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
            (chunk_data, anomalies, confidence_scores)
        """
        chunk = []
        for data_point in data_stream:
            chunk.append(data_point)

            if len(chunk) >= chunk_size:
                # Convert chunk to appropriate format
                if isinstance(data_point, tuple):  # Multivariate
                    chunk_data = pd.DataFrame(chunk)
                else:  # Univariate
                    chunk_data = pd.Series(chunk)

                # Detect anomalies
                anomalies, confidence_scores = self.fit_predict(chunk_data, return_confidence=True)

                # Visualize if needed
                self.visualizer.plot_anomalies(
                    chunk_data,
                    anomalies,
                    confidence_scores,
                    f'Streaming Anomaly Detection - Chunk Size {chunk_size}'
                )

                yield chunk_data, anomalies, confidence_scores
                chunk = []  # Reset chunk

