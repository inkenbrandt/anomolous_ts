import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend as sp_detrend
from plotly.subplots import make_subplots


def _regularize_series(
    s: pd.Series, *, target_freq: str | None = None, interp: str = "linear"
) -> pd.Series:
    """
    Ensure the Series has a regular sampling interval.

    Parameters
    ----------
    s : pandas.Series
        Time-indexed series.
    target_freq : str or None, default None
        Target pandas offset alias (e.g. 'D', 'H').  If None, infer the
        most common spacing and use that.
    interp : {'linear', 'pad', 'nearest', ...}, default 'linear'
        Interpolation method used after resampling to fill gaps.

    Returns
    -------
    rs : pandas.Series
        Regularly-spaced series, NaNs interpolated.
    """
    s = s.sort_index()

    if target_freq is None:
        # Find the mode of the index differences
        inferred = pd.infer_freq(s.index)
        if inferred is None:
            # fall back to the smallest timedelta
            inferred = pd.to_timedelta(s.index.to_series().diff().dropna().mode()[0])
        target_freq = inferred

    rs = s.resample(target_freq).mean()
    rs = rs.interpolate(method=interp)
    return rs


def fft_spectrum(
    s: pd.Series,
    *,
    detrend: str | None = "mean",  # 'mean', 'linear', or None
    window: str | None = None,
    top_n: int = 5,
    min_period: pd.Timedelta | None = pd.Timedelta("400D"),  # ignore slower cycles
):
    """
    Return the top-N spectral peaks of a time-series.

    Parameters
    ----------
    detrend : {'mean', 'linear', None}
        • 'mean'   – subtract series mean (default).
        • 'linear' – subtract best-fit line (removes drift).
        • None     – no detrending.
    min_period : Timedelta or None
        Peaks with periods *longer* than this are ignored when ranking.
        Example ``'2D'`` skips cycles ≥ 2 days (i.e. keeps ≥ 0.5 c/d).
    """
    s_reg = _regularize_series(s)
    y = s_reg.values.astype(float)

    # --- detrend ---------------------------------------------------------
    if detrend == "mean":
        y -= y.mean()
    elif detrend == "linear":
        y = sp_detrend(y)  # removes best-fit line
    elif detrend:
        raise ValueError("detrend must be 'mean', 'linear', or None")

    # --- optional window -------------------------------------------------
    if window is not None:
        y *= getattr(np, window)(len(y))

    # --- FFT -------------------------------------------------------------
    n = len(y)
    fs = 1 / pd.to_timedelta(s_reg.index.freq).total_seconds()
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    amps = np.abs(np.fft.rfft(y)) / n * 2

    # --- peak selection --------------------------------------------------
    peak_idx, _ = find_peaks(amps)
    # throw away frequency 0 and any below cut-off
    if min_period is not None:
        f_min = 1 / min_period.total_seconds()  # Hz
        peak_idx = peak_idx[freqs[peak_idx] >= f_min]

    strongest = peak_idx[np.argsort(amps[peak_idx])[::-1][:top_n]]

    out = pd.DataFrame(
        {
            "amplitude": amps[strongest],
            "frequency": freqs[strongest],
        }
    )
    out["period"] = pd.to_timedelta(1 / out["frequency"], unit="s")
    out.sort_values("amplitude", ascending=False, inplace=True, ignore_index=True)
    return out


def plot_fft(
    s: pd.Series,
    *,
    detrend: bool = True,
    window: str | None = None,
    x_unit: str = "per_day",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot the FFT amplitude spectrum.

    Parameters
    ----------
    s : pandas.Series
        Input time series.
    detrend, window : see `fft_spectrum`.
    x_unit : {'Hz', 'per_day', 'per_year'}, default 'per_day'
        Unit for the x-axis.
    ax : matplotlib.axes.Axes or None
        Axis on which to draw.  If None, a new figure/axis is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the spectrum plot.
    """
    s_reg = _regularize_series(s)
    y = s_reg.values.astype(float)
    if detrend:
        y = y - y.mean()
    if window is not None:
        y *= getattr(np, window)(len(y))

    n = len(y)
    f_samp = 1 / pd.to_timedelta(s_reg.index.freq).total_seconds()
    freqs = np.fft.rfftfreq(n, d=1.0 / f_samp)
    amps = np.abs(np.fft.rfft(y)) / n * 2

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    if x_unit == "Hz":
        x = freqs
        xlabel = "frequency [Hz]"
    elif x_unit == "per_day":
        x = freqs * 86400  # seconds → cycles per day
        xlabel = "cycles / day"
    elif x_unit == "per_year":
        x = freqs * 86400 * 365.25
        xlabel = "cycles / year"
    else:
        raise ValueError("x_unit must be 'Hz', 'per_day' or 'per_year'")

    ax.plot(x, amps, lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("amplitude")
    ax.set_title("FFT amplitude spectrum")
    ax.grid(True, which="both", ls=":")
    return ax


def plot_components(result):

    df = pd.concat([result.observed, result.trend, result.seasonal, result.resid], axis=1)
    df = df.rename(columns={0: "Original Data", "season": "seasonal", "observed": "Original Data"})
    components = df.columns
    rows = len(components)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, subplot_titles=[i for i in components]
    )

    # Plot original data
    for i, col in enumerate(components):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col), row=i + 1, col=1)

    # Update layout
    fig.update_layout(
        title="Time Series Decomposition", xaxis_title="Time", height=1200, width=1200
    )

    fig.show()

    return df, fig
