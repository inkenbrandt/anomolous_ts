from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

from typing import Optional, Union, List, Dict
import numpy as np
import pandas as pd
from scipy import stats, signal
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
#import pywt
from dataclasses import dataclass, field
from scipy.stats import entropy
import warnings
from abc import ABC, abstractmethod


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



@dataclass
class PreprocessorProtocol:
    """Configuration for time series preprocessing."""
    # Imputation settings
    imputation_method: str = 'auto'  # 'auto', 'linear', 'knn', 'ffill', etc.
    knn_neighbors: int = 5
    max_impute_gap: int = 10

    # Scaling settings
    scaling_method: str = 'robust'  # 'standard', 'robust', 'minmax'
    scaling_range: tuple = (-1, 1)

    # Decomposition settings
    decomposition_method: str = 'additive'  # 'additive', 'multiplicative'
    seasonal_period: Optional[int] = None

    # Feature engineering settings
    window_sizes: List[int] = field(default_factory=lambda: [5, 10, 20])
    enable_wavelets: bool = True
    wavelet_level: int = 3
    enable_spectral: bool = True
    enable_entropy: bool = True

    # Outlier handling
    outlier_method: str = 'clip'  # 'clip', 'remove', 'none'
    outlier_threshold: float = 3.0


from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from scipy import stats, signal
import pywt
from dataclasses import dataclass, field


@dataclass
class FeatureExtractorConfig:
    """Configuration for feature extractors."""
    window_sizes: List[int] = field(default_factory=lambda: [5, 10, 20])
    spectral_freq: float = 1.0
    wavelet_name: str = 'db4'
    wavelet_level: int = 3
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features from time series data."""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this extractor."""
        pass


class StatisticalFeatureExtractor(FeatureExtractor):
    """Extract statistical features from time series data."""

    def __init__(self, config: Optional[FeatureExtractorConfig] = None):
        self.config = config or FeatureExtractorConfig()

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        features = {}

        for window in self.config.window_sizes:
            rolled = pd.Series(data).rolling(window=window, min_periods=1)

            # Basic statistics
            features.update({
                f'mean_{window}': rolled.mean().values,
                f'std_{window}': rolled.std().values,
                f'kurt_{window}': rolled.kurt().values,
                f'skew_{window}': rolled.skew().values,
                f'max_{window}': rolled.max().values,
                f'min_{window}': rolled.min().values
            })

            # Quantile-based statistics
            for q in self.config.quantiles:
                features[f'quantile_{int(q * 100)}_{window}'] = rolled.quantile(q).values

            # Additional statistics
            features.update({
                f'mad_{window}': rolled.apply(lambda x: np.abs(x - x.mean()).mean()).values,
                f'range_{window}': rolled.apply(lambda x: x.max() - x.min()).values,
                f'iqr_{window}': rolled.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25)).values,
                f'entropy_{window}': rolled.apply(self._sample_entropy).values,
                f'moment3_{window}': rolled.apply(lambda x: stats.moment(x, moment=3)).values,
                f'moment4_{window}': rolled.apply(lambda x: stats.moment(x, moment=4)).values,
                f'var_coef_{window}': (rolled.std() / rolled.mean()).values
            })

            # Trend features
            features.update({
                f'trend_direction_{window}': rolled.apply(self._trend_direction).values,
                f'trend_strength_{window}': rolled.apply(self._trend_strength).values
            })

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        names = []
        basic_stats = ['mean', 'std', 'kurt', 'skew', 'max', 'min']
        advanced_stats = ['mad', 'range', 'iqr', 'entropy', 'moment3', 'moment4', 'var_coef']
        trend_stats = ['trend_direction', 'trend_strength']

        for window in self.config.window_sizes:
            # Add basic statistics
            names.extend([f'{stat}_{window}' for stat in basic_stats])
            # Add quantile statistics
            names.extend([f'quantile_{int(q * 100)}_{window}' for q in self.config.quantiles])
            # Add advanced statistics
            names.extend([f'{stat}_{window}' for stat in advanced_stats])
            # Add trend statistics
            names.extend([f'{stat}_{window}' for stat in trend_stats])

        return names

    def _sample_entropy(self, x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy."""
        if len(x) < 2:
            return 0.0
        r *= np.std(x)
        try:
            return stats.entropy(np.histogramdd(x, bins=m)[0])
        except:
            return 0.0

    def _trend_direction(self, x: np.ndarray) -> float:
        """Calculate trend direction using linear regression."""
        if len(x) < 2:
            return 0.0
        try:
            slope = np.polyfit(np.arange(len(x)), x, 1)[0]
            return np.sign(slope)
        except:
            return 0.0

    def _trend_strength(self, x: np.ndarray) -> float:
        """Calculate trend strength using R-squared of linear fit."""
        if len(x) < 2:
            return 0.0
        try:
            y = np.array(x)
            X = np.arange(len(x))
            slope, intercept = np.polyfit(X, y, 1)
            y_pred = X * slope + intercept
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
            return max(0, min(1, r2))  # Ensure value is between 0 and 1
        except:
            return 0.0


class SpectralFeatureExtractor(FeatureExtractor):
    """Extract spectral features from time series data."""

    def __init__(self, config: Optional[FeatureExtractorConfig] = None):
        self.config = config or FeatureExtractorConfig()

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        features = {}
        window_size = min(len(data) // 10, 100)  # Adaptive window size

        for i in range(len(data)):
            start_idx = max(0, i - window_size)
            window_data = data[start_idx:i + 1]

            if len(window_data) > 1:
                freqs, psd = signal.welch(window_data, fs=self.config.spectral_freq)
                features.setdefault('spectral_mean', []).append(np.mean(psd))
                features.setdefault('spectral_std', []).append(np.std(psd))
                features.setdefault('spectral_max', []).append(np.max(psd))
                features.setdefault('spectral_entropy', []).append(stats.entropy(psd + 1e-10))
                features.setdefault('dominant_freq', []).append(freqs[np.argmax(psd)])
                features.setdefault('power_bandwidth', []).append(self._power_bandwidth(psd, freqs))
            else:
                for key in ['spectral_mean', 'spectral_std', 'spectral_max', 'spectral_entropy',
                            'dominant_freq', 'power_bandwidth']:
                    features.setdefault(key, []).append(0)

        return {k: np.array(v) for k, v in features.items()}

    def get_feature_names(self) -> List[str]:
        return [
            'spectral_mean', 'spectral_std', 'spectral_max', 'spectral_entropy',
            'dominant_freq', 'power_bandwidth'
        ]

    def _power_bandwidth(self, psd: np.ndarray, freqs: np.ndarray) -> float:
        """Calculate power bandwidth (frequency range containing 95% of power)."""
        try:
            cumsum = np.cumsum(psd)
            total_power = cumsum[-1]
            lower_idx = np.searchsorted(cumsum, total_power * 0.025)
            upper_idx = np.searchsorted(cumsum, total_power * 0.975)
            return freqs[upper_idx] - freqs[lower_idx]
        except:
            return 0.0


class WaveletFeatureExtractor(FeatureExtractor):
    """Extract wavelet-based features from time series data."""

    def __init__(self, config: Optional[FeatureExtractorConfig] = None):
        self.config = config or FeatureExtractorConfig()

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        # Pad data to handle endpoints
        pad_size = len(data) + 2 ** self.config.wavelet_level - (len(data) % 2 ** self.config.wavelet_level)
        padded_data = np.pad(data, (0, pad_size - len(data)), mode='reflect')

        # Compute wavelet decomposition
        coeffs = pywt.wavedec(
            padded_data,
            self.config.wavelet_name,
            level=self.config.wavelet_level
        )

        features = {}
        for i, coef in enumerate(coeffs):
            prefix = 'approx' if i == 0 else f'detail_{i}'

            # Basic features
            features[f'{prefix}_mean'] = np.abs(coef[:len(data)])
            features[f'{prefix}_energy'] = np.square(coef[:len(data)])

            # Additional features
            features[f'{prefix}_std'] = self._rolling_std(coef[:len(data)])
            features[f'{prefix}_max'] = self._rolling_max(coef[:len(data)])
            features[f'{prefix}_entropy'] = self._rolling_entropy(coef[:len(data)])

        return features

    def get_feature_names(self) -> List[str]:
        names = []
        suffixes = ['mean', 'energy', 'std', 'max', 'entropy']
        # Add approximation features
        names.extend([f'approx_{suffix}' for suffix in suffixes])
        # Add detail features
        for i in range(1, self.config.wavelet_level + 1):
            names.extend([f'detail_{i}_{suffix}' for suffix in suffixes])
        return names

    def _rolling_std(self, x: np.ndarray, window: int = 10) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).std().fillna(0).values

    def _rolling_max(self, x: np.ndarray, window: int = 10) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).max().fillna(0).values

    def _rolling_entropy(self, x: np.ndarray, window: int = 10) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=1).apply(
            lambda w: stats.entropy(np.histogram(w, bins='auto')[0])
        ).fillna(0).values


class FeatureExtractorPipeline:
    """Pipeline for combining multiple feature extractors."""

    def __init__(
            self,
            config: Optional[FeatureExtractorConfig] = None,
            extractors: Optional[List[FeatureExtractor]] = None
    ):
        self.config = config or FeatureExtractorConfig()
        self.extractors = extractors or [
            StatisticalFeatureExtractor(self.config),
            SpectralFeatureExtractor(self.config),
            WaveletFeatureExtractor(self.config)
        ]

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all features using all extractors."""
        features = {}
        for extractor in self.extractors:
            features.update(extractor.extract(data))
        return features

    def get_feature_names(self) -> List[str]:
        """Get all feature names from all extractors."""
        names = []
        for extractor in self.extractors:
            names.extend(extractor.get_feature_names())
        return names

class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features from time series data."""
        pass


class StatisticalFeatures(FeatureExtractor):
    """Extract statistical features from time series data."""

    def __init__(self, window_sizes: List[int]):
        self.window_sizes = window_sizes

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        features = {}
        for window in self.window_sizes:
            rolled = pd.Series(data).rolling(window=window, min_periods=1)
            features.update({
                f'mean_{window}': rolled.mean().values,
                f'std_{window}': rolled.std().values,
                f'kurt_{window}': rolled.kurt().values,
                f'skew_{window}': rolled.skew().values,
                f'max_{window}': rolled.max().values,
                f'min_{window}': rolled.min().values,
                f'q25_{window}': rolled.quantile(0.25).values,
                f'q75_{window}': rolled.quantile(0.75).values
            })
        return features


class SpectralFeatures(FeatureExtractor):
    """Extract spectral features from time series data."""

    def __init__(self, fs: float = 1.0):
        self.fs = fs

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        # Compute spectral features using rolling window
        features = {}
        window_size = min(len(data) // 10, 100)  # Adaptive window size

        for i in range(len(data)):
            start_idx = max(0, i - window_size)
            window_data = data[start_idx:i + 1]

            if len(window_data) > 1:
                freqs, psd = signal.welch(window_data, fs=self.fs)
                features.setdefault('spectral_mean', []).append(np.mean(psd))
                features.setdefault('spectral_std', []).append(np.std(psd))
                features.setdefault('spectral_max', []).append(np.max(psd))
                features.setdefault('dominant_freq', []).append(freqs[np.argmax(psd)])
            else:
                # Padding for initial windows
                for key in ['spectral_mean', 'spectral_std', 'spectral_max', 'dominant_freq']:
                    features.setdefault(key, []).append(0)

        return {k: np.array(v) for k, v in features.items()}


class WaveletFeatures(FeatureExtractor):
    """Extract wavelet-based features from time series data."""

    def __init__(self, wavelet: str = 'db4', level: int = 3):
        self.wavelet = wavelet
        self.level = level

    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        # Pad data to handle endpoints
        pad_size = len(data) + 2 ** self.level - (len(data) % 2 ** self.level)
        padded_data = np.pad(data, (0, pad_size - len(data)), mode='reflect')

        # Compute wavelet decomposition
        coeffs = pywt.wavedec(padded_data, self.wavelet, level=self.level)

        features = {}
        for i, coef in enumerate(coeffs):
            prefix = 'approx' if i == 0 else f'detail_{i}'
            features[f'{prefix}_mean'] = np.abs(coef[:len(data)])
            features[f'{prefix}_energy'] = np.square(coef[:len(data)])

        return features


class EnhancedTimeSeriesPreprocessor:
    """
    Enhanced preprocessor for time series data with advanced feature engineering.
    """

    def __init__(self, config: Optional[PreprocessorProtocol] = None):
        self.config = config or PreprocessorProtocol()
        self._setup_components()

    def _setup_components(self):
        """Initialize preprocessing components based on configuration."""
        # Initialize scalers
        self.scaler = self._get_scaler()

        # Initialize feature extractors
        self.feature_extractors = []
        self.feature_extractors.append(
            StatisticalFeatures(self.config.window_sizes)
        )
        if self.config.enable_spectral:
            self.feature_extractors.append(SpectralFeatures())
        if self.config.enable_wavelets:
            self.feature_extractors.append(
                WaveletFeatures(level=self.config.wavelet_level)
            )

    def _get_scaler(self):
        """Get appropriate scaler based on configuration."""
        if self.config.scaling_method == 'standard':
            return StandardScaler()
        elif self.config.scaling_method == 'robust':
            return RobustScaler()
        elif self.config.scaling_method == 'minmax':
            return MinMaxScaler(feature_range=self.config.scaling_range)
        else:
            return None

    def _handle_missing_values(self, data: pd.Series) -> pd.Series:
        """Handle missing values using configured method."""
        if self.config.imputation_method == 'auto':
            # Choose method based on gap sizes
            gap_sizes = data.isna().astype(int).groupby(
                data.notna().cumsum()
            ).sum()

            if max(gap_sizes) <= self.config.max_impute_gap:
                return data.interpolate(method='linear')
            else:
                imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
                return pd.Series(
                    imputer.fit_transform(data.values.reshape(-1, 1)).ravel(),
                    index=data.index
                )
        elif self.config.imputation_method == 'linear':
            return data.interpolate(method='linear')
        elif self.config.imputation_method == 'ffill':
            return data.fillna(method='ffill')
        elif self.config.imputation_method == 'knn':
            imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
            return pd.Series(
                imputer.fit_transform(data.values.reshape(-1, 1)).ravel(),
                index=data.index
            )
        else:
            raise ValueError(f"Unknown imputation method: {self.config.imputation_method}")

    def _handle_outliers(self, data: pd.Series) -> pd.Series:
        """Handle outliers using configured method."""
        if self.config.outlier_method == 'none':
            return data

        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > self.config.outlier_threshold

        if self.config.outlier_method == 'clip':
            threshold = self.config.outlier_threshold * data.std()
            return data.clip(
                lower=data.mean() - threshold,
                upper=data.mean() + threshold
            )
        elif self.config.outlier_method == 'remove':
            return data.mask(outliers)
        else:
            raise ValueError(f"Unknown outlier method: {self.config.outlier_method}")

    def _decompose_seasonal(self, data: pd.Series) -> Dict[str, pd.Series]:
        """Perform seasonal decomposition if configured."""
        if not self.config.seasonal_period:
            return {'original': data}

        try:
            decomposition = seasonal_decompose(
                data,
                period=self.config.seasonal_period,
                model=self.config.decomposition_method
            )
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
        except Exception as e:
            warnings.warn(f"Seasonal decomposition failed: {str(e)}")
            return {'original': data}

    def fit_transform(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess time series data and extract features.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame]
            Input time series data

        Returns
        -------
        pd.DataFrame
            Preprocessed data with extracted features
        """
        # Convert to pandas Series if needed
        if isinstance(data, pd.DataFrame):
            results = []
            for column in data.columns:
                result = self.fit_transform(data[column])
                result.columns = [f'{column}_{col}' for col in result.columns]
                results.append(result)
            return pd.concat(results, axis=1)

        # Handle missing values
        clean_data = self._handle_missing_values(data)

        # Handle outliers
        clean_data = self._handle_outliers(clean_data)

        # Perform seasonal decomposition
        components = self._decompose_seasonal(clean_data)

        # Extract features for each component
        all_features = {}
        for component_name, component_data in components.items():
            # Skip if component has NaN values
            if component_data.isna().any():
                continue

            # Extract features using all configured extractors
            for extractor in self.feature_extractors:
                features = extractor.extract(component_data.values)
                # Add component prefix to feature names
                features = {
                    f'{component_name}_{name}': values
                    for name, values in features.items()
                }
                all_features.update(features)

        # Create feature DataFrame
        feature_df = pd.DataFrame(
            all_features,
            index=data.index
        )

        # Scale features if configured
        if self.scaler is not None:
            feature_df = pd.DataFrame(
                self.scaler.fit_transform(feature_df),
                index=feature_df.index,
                columns=feature_df.columns
            )

        return feature_df

    def get_feature_importance(self, target: Optional[pd.Series] = None) -> pd.Series:
        """
        Calculate feature importance scores.

        Parameters
        ----------
        target : Optional[pd.Series]
            Target variable for supervised importance calculation

        Returns
        -------
        pd.Series
            Feature importance scores
        """
        if not hasattr(self, 'feature_df_'):
            raise ValueError("Must call fit_transform before getting feature importance")

        if target is None:
            # Use variance as importance metric
            importance = self.feature_df_.var()
        else:
            # Use correlation with target as importance metric
            importance = self.feature_df_.corrwith(target).abs()

        return importance.sort_values(ascending=False)