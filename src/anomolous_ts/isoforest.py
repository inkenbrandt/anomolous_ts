import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Union, Sequence, Tuple, Optional


def detect_isolation_forest(
    ts: Union[pd.Series, pd.DataFrame],
    contamination: float = 0.01,
    n_estimators: int = 200,
    max_samples: Union[int, str] = "auto",
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    scale: bool = True,
    random_state: Optional[int] = None,
    return_scores: bool = False,
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Detect point anomalies in a univariate time series using *Isolation Forest*.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time-indexed series of numeric values. If a DataFrame is passed, the first
        column is used.
    contamination : float, default ``0.01``
        Proportion of the data expected to be anomalous.  Must be in ``(0, 0.5]``.
    n_estimators : int, default ``200``
        Number of trees in the Isolation Forest.
    max_samples : int or ``"auto"``, default ``"auto"``
        Sub-sample size used to build each tree.  Follows scikit-learn’s definition.
    lags : Sequence[int], default ``(1, 2, 3, 6, 12)``
        Additional lagged features (in time steps) used to give the model context.
        Pass an empty tuple ``()`` to disable.
    scale : bool, default ``True``
        If ``True``, the feature matrix is standardised (zero mean / unit variance)
        before fitting the model.
    random_state : int or None, default ``None``
        Seed for reproducibility.
    return_scores : bool, default ``False``
        If ``True``, also return the Isolation Forest anomaly scores
        (negative outlier factor).

    Returns
    -------
    anomalies : pandas.Series
        Boolean series aligned to *ts.index* with ``True`` where an anomaly is
        flagged.
    scores : pandas.Series, optional
        Isolation Forest *decision_function* values (larger ⇒ more normal).
        Returned only if *return_scores* is ``True``.

    Notes
    -----
    Isolation Forest operates on feature vectors, not sequences.  The function
    therefore constructs a feature matrix consisting of the current value and any
    user-defined lagged values.  *NaN* rows arising from lagging are dropped
    before model training, so very short series may require ``lags=()``.

    Examples
    --------
    >>> s = pd.read_csv("power.csv", index_col=0, parse_dates=True)["load"]
    >>> anomalies, scores = detect_isolation_forest(s, contamination=0.02,
    ...                                             lags=(1, 24), return_scores=True)
    >>> s[anomalies].plot(marker="o", ls="", color="red")
    """
    # --- input sanity checks -------------------------------------------------
    if isinstance(ts, pd.DataFrame):
        ts = ts.iloc[:, 0]  # use first column

    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError("`ts` must have a DatetimeIndex")

    if not (0.0 < contamination <= 0.5):
        raise ValueError("`contamination` must be in the interval (0, 0.5]")

    # --- build feature matrix -----------------------------------------------
    df = pd.DataFrame({"y": ts})
    for lag in lags:
        df[f"lag_{lag}"] = ts.shift(lag)

    # drop rows with missing lag values
    X = df.dropna()
    X_values = X.values.astype(float)

    # optional scaling
    if scale:
        scaler = StandardScaler()
        X_values = scaler.fit_transform(X_values)

    # --- fit Isolation Forest ------------------------------------------------
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
    )
    iso.fit(X_values)

    # --- get predictions & scores -------------------------------------------
    y_pred = pd.Series(
        iso.predict(X_values) == -1,  # -1 = anomaly
        index=X.index,
        name="anomaly",
    )
    if return_scores:
        scores = pd.Series(iso.decision_function(X_values), index=X.index, name="score")
        return y_pred.reindex(ts.index, fill_value=False), scores.reindex(ts.index)

    return y_pred.reindex(ts.index, fill_value=False)
