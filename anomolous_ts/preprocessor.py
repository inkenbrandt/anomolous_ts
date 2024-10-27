from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer


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

