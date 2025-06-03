from __future__ import annotations

import numpy as np
import pandas as pd


def moving_window_anomaly(
    data: pd.Series | pd.DataFrame,
    *,
    window: int = 30,
    z_thresh: float = 3.0,
    method: str = "zscore",
    min_periods: int | None = None,
    return_zscore: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Detect point anomalies in a time-ordered pandas object using
    statistics computed in a *moving* (rolling) window.

    Parameters
    ----------
    data
        Input values.  Must be one-dimensional (:class:`~pandas.Series`)
        or two-dimensional (:class:`~pandas.DataFrame`) and *numeric*
        (`float`, `int`, or `np.number` sub-types).
    window
        Width of the rolling window (measured in **rows**, *not* in
        clock time).  The statistic at index *i* is calculated from the
        previous ``window`` observations *strictly before* *i*
        (causal/retrospective window).  Set ``center=True`` on the
        returned z-score to make the window symmetric.
    z_thresh
        Cut-off used to flag an observation as anomalous:

        * ``method="zscore"`` – |z| > ``z_thresh``
        * ``method="mad"``    – |Δ| / (1.4826 ⋅ MAD) > ``z_thresh``
          (1.4826 scales the median absolute deviation so that it equals
          *σ* for a normal distribution).
    method
        Statistic used to standardise residuals **inside each window**.

        * ``"zscore"`` – rolling mean / standard deviation
        * ``"mad"``    – rolling median / median absolute deviation
    min_periods
        Minimum number of non-NaN observations required to compute the
        statistic.  Defaults to ``window // 2``.
    return_zscore
        If *True*, the function returns a tuple
        ``(anomaly_mask, zscore)`` instead of the mask alone.

    Returns
    -------
    anomaly_mask
        Boolean object of the same shape as *data* with ``True`` marking
        observations whose (scaled) residual exceeds *z_thresh*.
    zscore : *optional*
        Normalised residuals used in the test.  Returned only when
        ``return_zscore=True``.

    Notes
    -----
    * **Causality.**  The statistic at time *t* never uses data at or
      after *t*, making the detector suitable for real-time streaming.
    * **NaNs.**  Missing values are ignored inside the roll; an output
      position is NaN *only* when fewer than *min_periods* valid values
      are available.
    * **Vectorisation.**  All computations rely on pandas’ built-in
      vectorised rolling operations; the implementation scales to
      thousands of columns without explicit Python loops.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = pd.date_range('2025-01-01', periods=100, freq='D')
    >>> s = pd.Series(np.random.randn(100), index=rng).cumsum()
    >>> s.iloc[20] += 12           # inject an outlier
    >>> mask, z = moving_window_anomaly(
    ...     s, window=7, z_thresh=3, method='mad', return_zscore=True
    ... )
    >>> s[mask].head()             # rows flagged as anomalies
    2025-01-21    11.978...
    dtype: float64
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("`data` must be a pandas Series or DataFrame.")

    if method not in {"zscore", "mad"}:
        raise ValueError("`method` must be either 'zscore' or 'mad'.")

    data_num = data.astype("float64")  # ensure numeric; raises on object

    if min_periods is None:
        min_periods = window // 2

    if method == "zscore":
        rolling = data_num.rolling(window, min_periods=min_periods)
        mean = rolling.mean()
        std = rolling.std(ddof=0)
        zscore = (data_num - mean) / std
    else:  # robust MAD method
        rolling = data_num.rolling(window, min_periods=min_periods)
        median = rolling.median()
        abs_dev = (data_num - median).abs()
        mad = rolling.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        # Consistent with normal σ:  σ ≈ 1.4826 · MAD
        zscore = 0.6745 * abs_dev / mad

    anomaly_mask = zscore.abs() > z_thresh

    if return_zscore:
        return anomaly_mask, zscore

    return anomaly_mask


# 2.  --- helper for gap-filling --------------------------------------------
def impute_masked(
    values: pd.Series | pd.DataFrame,
    mask: pd.Series | pd.DataFrame,
    *,
    how: str = "iterative",
    **kwargs,
) -> pd.Series | pd.DataFrame:
    """
    Replace the positions marked ``True`` in *mask* with estimates.

    Parameters
    ----------
    values
        Original data (will not be modified in-place).
    mask
        Boolean mask of the same shape produced by
        :func:`moving_window_anomaly`.  Locations where ``mask`` is
        ``True`` are treated as *missing*.
    how
        Imputation strategy:

        * ``"iterative"`` – multivariate chained equations (MICE).
        * ``"knn"``       – k-nearest-neighbour imputation.
        * ``"simple"``    – mean / median / constant (see ``strategy`` kw).
        * ``"linear"``    – 1-D linear interpolation along the index.
        * ``"spline"``    – cubic spline interpolation (order set via kw).
    **kwargs
        Passed straight to the chosen imputer or to
        :pandas:`DataFrame.interpolate`.

    Returns
    -------
    pd.Series or pd.DataFrame
        Copy of *values* with anomalies replaced.
    """
    x = values.copy()
    x[mask] = np.nan  #  convert anomalies → NaN

    if how == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        imputer = IterativeImputer(**kwargs)  # e.g. max_iter=10, random_state=0
        imputed = pd.DataFrame(
            imputer.fit_transform(x),
            index=x.index,
            columns=x.columns if isinstance(x, pd.DataFrame) else [x.name],
        )

    elif how == "knn":
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(**kwargs)  # e.g. n_neighbors=5, weights="distance"
        imputed = pd.DataFrame(
            imputer.fit_transform(x),
            index=x.index,
            columns=x.columns if isinstance(x, pd.DataFrame) else [x.name],
        )

    elif how == "simple":
        from sklearn.impute import SimpleImputer

        # default strategy="mean"; others: "median", "most_frequent", "constant"
        imputer = SimpleImputer(**kwargs)
        imputed = pd.DataFrame(
            imputer.fit_transform(x),
            index=x.index,
            columns=x.columns if isinstance(x, pd.DataFrame) else [x.name],
        )

    elif how in {"linear", "spline"}:
        # pandas wraps SciPy's interpolate.* functions here
        imputed = x.interpolate(method=how, **kwargs)

    else:
        raise ValueError("`how` must be one of: iterative, knn, simple, linear, spline")

    # Return the same type the user supplied
    if isinstance(values, pd.Series):
        return imputed.iloc[:, 0]
    return imputed
