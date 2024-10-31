from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy import stats

from .feature_extractors import PreprocessorProtocol
from .base import BaseDetector, DetectionResult, AnomalyScore, TimeSeriesData

@dataclass
class SHESDConfig:
    """Configuration for Twitter's S-H-ESD detector."""
    max_anomalies: float = 0.1
    alpha: float = 0.05
    hybrid: bool = True
    seasonal_period: Optional[int] = None
    decomposition_method: str = 'additive'
    auto_detect_period: bool = True
    max_lag: int = 366
    
    def __post_init__(self):
        if not 0 < self.max_anomalies < 1:
            raise ValueError("max_anomalies must be between 0 and 1")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.decomposition_method not in ['additive', 'multiplicative', 'robust']:
            raise ValueError("decomposition_method must be one of: additive, multiplicative, robust")

class TwitterSHESD(BaseDetector['TwitterSHESD']):
    """
    Enhanced implementation of Twitter's Seasonal Hybrid ESD algorithm.
    Adapted to conform to the common anomaly detection interface.
    """
    
    def __init__(
        self,
        config: Optional[SHESDConfig] = None,
        preprocessor: Optional[PreprocessorProtocol] = None
    ):
        self.config = config or SHESDConfig()
        self.preprocessor = preprocessor
        
        # Internal state
        self._seasonal_period: Optional[int] = self.config.seasonal_period
        self._location_stats: Optional[np.ndarray] = None
        self._scale_stats: Optional[np.ndarray] = None
        
    def _detect_period(self, data: np.ndarray) -> int:
        """Automatically detect the seasonal period using autocorrelation."""
        n = len(data)
        max_lag = min(self.config.max_lag, n // 2)
        
        # Calculate autocorrelation
        acf_values = acf(data, nlags=max_lag, fft=True)
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(acf_values) - 1):
            if acf_values[i] > acf_values[i - 1] and acf_values[i] > acf_values[i + 1]:
                peaks.append((i, acf_values[i]))
        
        # Sort peaks by correlation value
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Return the lag of the highest peak after lag 1
        for lag, corr in peaks:
            if lag > 1:
                return lag
                
        return 1  # Default if no clear seasonality is found
    
    def _remove_seasonal_component(
        self,
        data: pd.Series,
        period: int
    ) -> pd.Series:
        """Remove seasonal component from time series."""
        if (
            self.config.decomposition_method == 'multiplicative'
            and (data <= 0).any()
        ):
            raise ValueError("Multiplicative decomposition requires positive values")
            
        decomposition = seasonal_decompose(
            data,
            period=period,
            model=self.config.decomposition_method,
            extrapolate_trend='freq'
        )
        
        if self.config.decomposition_method == 'multiplicative':
            residual = data / decomposition.seasonal
        else:
            residual = data - decomposition.seasonal
            
        return residual
    
    def _compute_statistics(self, data: np.ndarray) -> tuple:
        """Compute location and scale statistics."""
        if self.config.hybrid:
            location = np.median(data)
            scale = np.median(np.abs(data - location)) * 1.4826
        else:
            location = np.mean(data)
            scale = np.std(data, ddof=1)
        return location, scale
    
    def _generalized_esd_test(
        self,
        data: np.ndarray,
        max_outliers: int
    ) -> List[int]:
        """Perform the Generalized ESD test."""
        n = len(data)
        outlier_indices = []
        working_data = data.copy()
        
        for i in range(max_outliers):
            location, scale = self._compute_statistics(working_data)
            if scale == 0:
                break
                
            # Store statistics
            self._location_stats = location
            self._scale_stats = scale
            
            # Compute test statistics
            test_statistics = np.abs(working_data - location) / scale
            max_idx = np.argmax(test_statistics)
            max_stat = test_statistics[max_idx]
            
            # Compute critical value
            t_stat = stats.t.ppf(1 - self.config.alpha / (2 * (n - i)), n - i - 2)
            lambda_i = ((n - i - 1) * t_stat) / np.sqrt((n - i - 2 + t_stat ** 2) * (n - i))
            
            if max_stat > lambda_i:
                # Map back to original index
                original_idx = np.where(data == working_data[max_idx])[0][0]
                outlier_indices.append(original_idx)
                working_data = np.delete(working_data, max_idx)
            else:
                break
                
        return outlier_indices
    
    def _calculate_anomaly_scores(
        self,
        data: np.ndarray,
        anomaly_indices: List[int]
    ) -> List[AnomalyScore]:
        """Calculate anomaly scores for all points."""
        scores = []
        normalized_deviations = np.abs(data - self._location_stats) / self._scale_stats
        
        # Convert to probability scores using sigmoid
        probability_scores = 1 / (1 + np.exp(-normalized_deviations))
        
        for i in range(len(data)):
            is_anomaly = i in anomaly_indices
            score = float(probability_scores[i])
            
            scores.append(AnomalyScore(
                score=score,
                is_anomaly=is_anomaly,
                confidence=float(abs(score - 0.5) * 2),
                timestamp=datetime.now(),
                contributing_features={
                    "deviation": float(normalized_deviations[i]),
                    "relative_score": float(score / max(probability_scores))
                }
            ))
            
        return scores
    
    def fit(self, data: TimeSeriesData) -> 'TwitterSHESD':
        """Fit the detector to the data."""
        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)
            
        # Convert to Series if DataFrame
        if isinstance(data, pd.DataFrame):
            if data.shape[1] > 1:
                raise ValueError("TwitterSHESD only supports univariate time series")
            data = data.iloc[:, 0]
            
        # Detect period if needed
        if self.config.auto_detect_period and self._seasonal_period is None:
            self._seasonal_period = self._detect_period(data.values)
            
        return self
    
    def predict(self, data: TimeSeriesData) -> DetectionResult:
        """Predict anomalies in the data."""
        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)
            
        # Convert to Series if DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
            
        # Remove seasonality if period is set
        if self._seasonal_period and self._seasonal_period > 1:
            adjusted_data = self._remove_seasonal_component(
                data,
                self._seasonal_period
            )
        else:
            adjusted_data = data.copy()
            
        # Calculate maximum number of outliers
        max_outliers = int(np.ceil(len(data) * self.config.max_anomalies))
        
        # Detect anomalies
        anomaly_indices = self._generalized_esd_test(
            adjusted_data.values,
            max_outliers
        )
        
        # Calculate scores
        scores = self._calculate_anomaly_scores(
            adjusted_data.values,
            anomaly_indices
        )
        
        return DetectionResult(
            scores=scores,
            detector_name="TwitterSHESD",
            detection_time=datetime.now(),
            metadata={
                "seasonal_period": self._seasonal_period,
                "decomposition_method": self.config.decomposition_method,
                "hybrid_statistics": self.config.hybrid,
                "max_anomalies": self.config.max_anomalies,
                "alpha": self.config.alpha,
                "num_anomalies_found": len(anomaly_indices)
            }
        )
    
    def score(self, data: TimeSeriesData) -> np.ndarray:
        """Get raw anomaly scores."""
        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)
            
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
            
        if self._seasonal_period and self._seasonal_period > 1:
            adjusted_data = self._remove_seasonal_component(
                data,
                self._seasonal_period
            )
        else:
            adjusted_data = data.copy()
            
        location, scale = self._compute_statistics(adjusted_data.values)
        return np.abs(adjusted_data.values - location) / scale
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get detector metadata."""
        return {
            "name": "TwitterSHESD",
            "version": "2.0",
            "config": {
                "max_anomalies": self.config.max_anomalies,
                "alpha": self.config.alpha,
                "hybrid": self.config.hybrid,
                "seasonal_period": self._seasonal_period,
                "decomposition_method": self.config.decomposition_method,
                "auto_detect_period": self.config.auto_detect_period
            },
            "statistics": {
                "location": float(self._location_stats) if self._location_stats is not None else None,
                "scale": float(self._scale_stats) if self._scale_stats is not None else None
            }
        }
