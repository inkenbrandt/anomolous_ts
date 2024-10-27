
import pandas as pd

import matplotlib.pyplot as plt



class TimeSeriesVisualizer:
    """
    Handles visualization of time series data, anomalies, and imputations.
    """

    def plot_anomalies(self, data, anomalies, confidence_scores=None, title='Anomaly Detection Results'):
        """
        Plot time series data with highlighted anomalies and confidence scores.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Original time series data
        anomalies : pd.Series or pd.DataFrame
            Boolean series/dataframe indicating anomalies
        confidence_scores : pd.Series or pd.DataFrame, optional
            Confidence scores for anomalies
        title : str
            Plot title
        """
        plt.figure(figsize=(15, 8))

        if isinstance(data, pd.DataFrame):
            # Plot multiple series
            for col in data.columns:
                plt.plot(data.index, data[col], label=f'{col} (normal)')
                if anomalies[col].any():
                    plt.scatter(
                        data.index[anomalies[col]],
                        data[col][anomalies[col]],
                        color='red',
                        label=f'{col} (anomaly)'
                    )

                    if confidence_scores is not None:
                        for idx in data.index[anomalies[col]]:
                            plt.annotate(
                                f'{confidence_scores[col][idx]:.2f}',
                                (idx, data[col][idx]),
                                xytext=(5, 5),
                                textcoords='offset points'
                            )
        else:
            # Plot single series
            plt.plot(data.index, data, label='Normal')
            if anomalies.any():
                plt.scatter(
                    data.index[anomalies],
                    data[anomalies],
                    color='red',
                    label='Anomaly'
                )

                if confidence_scores is not None:
                    for idx in data.index[anomalies]:
                        plt.annotate(
                            f'{confidence_scores[idx]:.2f}',
                            (idx, data[idx]),
                            xytext=(5, 5),
                            textcoords='offset points'
                        )

        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_imputation(self, original_data, imputed_data, title='Imputation Results'):
        """
        Plot original data with missing values and imputed results.

        Parameters
        ----------
        original_data : pd.Series or pd.DataFrame
            Original data with missing values
        imputed_data : pd.Series or pd.DataFrame
            Data after imputation
        title : str
            Plot title
        """
        plt.figure(figsize=(15, 8))

        if isinstance(original_data, pd.DataFrame):
            # Plot multiple series
            for col in original_data.columns:
                # Plot original non-missing values
                plt.plot(
                    original_data.index[original_data[col].notna()],
                    original_data[col][original_data[col].notna()],
                    'o-',
                    label=f'{col} (original)'
                )
                # Plot imputed values
                plt.plot(
                    original_data.index[original_data[col].isna()],
                    imputed_data[col][original_data[col].isna()],
                    'rx',
                    label=f'{col} (imputed)'
                )
        else:
            # Plot single series
            plt.plot(
                original_data.index[original_data.notna()],
                original_data[original_data.notna()],
                'o-',
                label='Original'
            )
            plt.plot(
                original_data.index[original_data.isna()],
                imputed_data[original_data.isna()],
                'rx',
                label='Imputed'
            )

        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
