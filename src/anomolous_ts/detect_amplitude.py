import numpy as np
import pandas as pd
from typing import Optional, Tuple


def _rolling_amplitude(x: pd.Series, window: int, center: bool = True) -> pd.Series:
    """Return half the peak-to-peak range in a rolling window.

    Parameters
    ----------
    x :
        Input series.
    window :
        Window length in samples.
    center :
        If *True*, align the window center at each label; otherwise align the right edge.

    Returns
    -------
    pandas.Series
        Amplitude envelope with the same index as *x* (NaN where the window is incomplete).
    """
    roll_max = x.rolling(window, center=center).max()
    roll_min = x.rolling(window, center=center).min()
    return 0.5 * (roll_max - roll_min)


def detect_amplitude_change(
    x: pd.Series,
    window: int = 48,
    k: float = 4.0,
    min_gap: Optional[int] = None,
) -> pd.Series:
    """Detect timestamps where the signal amplitude changes significantly.

    Parameters
    ----------
    x :
        One-dimensional time-indexed data.
    window :
        Length of the sliding window (in samples) used to estimate local amplitude.
        Choose it a bit longer than the dominant period of the signal.
    k :
        Threshold multiplier: a change is flagged when
        ``|A_t - median(A)| > k * MAD(A)``, where *A* is the rolling amplitude.
        Typical values lie between 3 and 6.
    min_gap :
        Optional minimum distance (in samples) between successive change points.
        Helps prevent clusters of adjacent flags; set to ``window`` for one flag
        per sustained shift.  If *None*, no de-duplication is applied.

    Returns
    -------
    pandas.Series[bool]
        Boolean mask indexed like *x* (True = amplitude change point).
    """
    if not isinstance(x, pd.Series):
        raise TypeError("x must be a pandas Series")
    if len(x) < window * 2:
        raise ValueError("series too short for the chosen window")

    amp = _rolling_amplitude(x, window)
    med = amp.expanding(min_periods=window).median()
    mad = (amp - med).abs().expanding(
        min_periods=window
    ).median() * 1.4826  # convert MAD→σ̂ (for Gaussian noise)

    score = (amp - med).abs() / (mad.replace(0, np.nan))
    change_mask = score > k

    # Optionally discard multiple flags that are too close together
    if min_gap:
        idx = np.flatnonzero(change_mask.values)
        if idx.size:
            keep = [idx[0]]
            for i in idx[1:]:
                if i - keep[-1] >= min_gap:
                    keep.append(i)
            trimmed = pd.Series(False, index=x.index)
            trimmed.iloc[keep] = True
            change_mask = trimmed

    return change_mask


def change_intervals(mask: pd.Series) -> list[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Summarize consecutive True segments of *mask* as (start, end) tuples.

    Useful to report sustained periods of changed amplitude."""
    if mask.empty:
        return []
    segments: list[Tuple[pd.Timestamp, pd.Timestamp]] = []
    run_start: Optional[pd.Timestamp] = None
    for t, flag in mask.items():
        if flag and run_start is None:
            run_start = t
        elif not flag and run_start is not None:
            segments.append((run_start, t))
            run_start = None
    if run_start is not None:
        segments.append((run_start, mask.index[-1]))
    return segments
