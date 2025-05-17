import numpy as np
import pandas as pd
from typing import Union, Sequence, Optional


class SyntheticSeries:
    """
    Generate and manipulate synthetic time-series data.

    Parameters
    ----------
    start : str or pandas-compatible datetime-like, default ``"2020-01-01"``
        First timestamp (inclusive) of the series.
    periods : int, default ``365``
        Number of observations to create.
    freq : str, default ``"D"``
        Pandas offset alias (e.g. ``"D"``, ``"H"``, ``"30T"``).
    seed : int or None, optional
        Random-state seed for reproducibility.

    Notes
    -----
    All mutator methods (`add_trend`, `add_seasonality`, etc.) operate **in-place**
    and return ``self`` so they can be *daisy-chained*:

    >>> ts = (SyntheticSeries("2024-01-01", 730, "D", seed=42)
    ...          .add_trend(slope=0.02)
    ...          .add_seasonality(period=365, amplitude=5)
    ...          .add_white_noise(std=0.5)
    ...          .add_spikes(n_spikes=10, magnitude=12))
    >>> ts.to_series().head()
    """

    def __init__(
        self,
        start: Union[str, pd.Timestamp] = "2020-01-01",
        periods: int = 365,
        freq: str = "D",
        seed: Optional[int] = None,
    ):
        self.index = pd.date_range(start=start, periods=periods, freq=freq)
        self._values = np.zeros(periods, dtype=float)
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # Mutator helpers
    # ---------------------------------------------------------------------
    def add_trend(
        self,
        slope: float = 1.0,
        intercept: float = 0.0,
        kind: str = "linear",
        power: float = 1.0,
    ):
        """
        Add a deterministic trend.

        Parameters
        ----------
        slope : float, default ``1.0``
            Slope of the trend.
        intercept : float, default ``0.0``
            Intercept at *t = 0*.
        kind : {"linear", "power"}, default ``"linear"``
            Functional form.  ``"power"`` yields *intercept + slope · t**power*.
        power : float, default ``1.0``
            Exponent used when ``kind="power"``.
        """
        t = np.arange(len(self._values))
        if kind == "linear":
            self._values += intercept + slope * t
        elif kind == "power":
            self._values += intercept + slope * np.power(t, power)
        else:
            raise ValueError("kind must be 'linear' or 'power'")
        return self

    def add_seasonality(
        self,
        period: int,
        amplitude: float = 1.0,
        phase: float = 0.0,
        waveform: str = "sine",
    ):
        """
        Superimpose a seasonal signal.

        Parameters
        ----------
        period : int
            Number of samples per cycle (e.g. 12 for monthly seasonality in
            monthly data).
        amplitude : float, default ``1.0``
            Peak deviation from the mean.
        phase : float, default ``0.0``
            Phase shift in *radians*.
        waveform : {"sine", "cosine"}, default ``"sine"``
            Base waveform.
        """
        t = np.arange(len(self._values))
        omega = 2.0 * np.pi / period
        trig = np.sin if waveform == "sine" else np.cos
        self._values += amplitude * trig(omega * t + phase)
        return self

    def add_random_noise(self, scale: float = 1.0):
        """
        Add a **random‐walk** (cumulative) noise component.

        This yields a non-stationary series often used to emulate
        *integrated* processes.

        Parameters
        ----------
        scale : float, default ``1.0``
            Standard deviation of the *increments*.
        """
        steps = self.rng.normal(loc=0.0, scale=scale, size=len(self._values))
        self._values += np.cumsum(steps)
        return self

    def add_white_noise(self, std: float = 1.0):
        """
        Add i.i.d. Gaussian white noise.

        Parameters
        ----------
        std : float, default ``1.0``
            Standard deviation of the noise.
        """
        self._values += self.rng.normal(loc=0.0, scale=std, size=len(self._values))
        return self

    def add_offset(self, value: Union[float, Sequence[float]]):
        """
        Shift the entire series by a constant or time-varying offset.

        Parameters
        ----------
        value : float or 1-D array-like
            Amount to add.  If array-like, length must equal ``periods``.
        """
        self._values += value
        return self

    def add_block_spikes(
        self,
        width: int = 3,
        n_blocks: int = 4,
        magnitude: float = 10.0,
        direction: str = "both",
        allow_overlap: bool = False,
    ):
        """
        Insert block spikes (contiguous runs of elevated / depressed values).

        Parameters
        ----------
        width : int, default ``3``
            Number of consecutive samples per spike.
        n_blocks : int, default ``4``
            How many spikes to generate.
        magnitude : float, default ``10.0``
            Absolute height/depth of each block.
        direction : {"both", "positive", "negative"}, default ``"both"``
            Sign(s) of the spikes.
        allow_overlap : bool, default ``False``
            If *False*, blocks are guaranteed not to overlap.
        """
        if width <= 0 or n_blocks <= 0:
            return self

        L = len(self._values)
        if width > L:
            raise ValueError("width cannot exceed series length")

        # candidate start positions so the block fits entirely
        possible = np.arange(0, L - width + 1)
        rng = self.rng

        starts = []
        while len(starts) < n_blocks and len(possible) > 0:
            start = rng.choice(possible)
            starts.append(start)

            if not allow_overlap:
                # remove indices that would overlap the chosen block
                block_range = np.arange(start - width + 1, start + width)
                possible = np.setdiff1d(possible, block_range, assume_unique=True)

        signs = {
            "both": rng.choice([-1, 1], size=len(starts)),
            "positive": np.ones(len(starts)),
            "negative": -np.ones(len(starts)),
        }[direction]

        for s, sign in zip(starts, signs):
            self._values[s : s + width] += sign * magnitude

        return self

    def add_spikes(
        self,
        n_spikes: int = 5,
        magnitude: float = 10.0,
        direction: str = "both",
    ):
        """
        Inject isolated spikes/outliers.

        Parameters
        ----------
        n_spikes : int, default ``5``
            Number of spikes to insert.
        magnitude : float, default ``10.0``
            Absolute size of each spike.
        direction : {"both", "positive", "negative"}, default ``"both"``
            Sign of spikes.
        """
        if n_spikes <= 0:
            return self

        idx = self.rng.choice(len(self._values), size=n_spikes, replace=False)
        signs = {
            "both": self.rng.choice([-1, 1], size=n_spikes),
            "positive": np.ones(n_spikes),
            "negative": -np.ones(n_spikes),
        }[direction]
        self._values[idx] += signs * magnitude
        return self

    def add_step_scale(
        self,
        when,
        factor: float,
        length: Optional[int] = None,
    ):
        """
        Multiply a slice of the series by a constant factor.

        Parameters
        ----------
        when : int, str, pandas-Timestamp or datetime-like
            Starting point of the scale shift.
            • **int**  – positional index (0-based).
            • **str / Timestamp** – matching a label in the index.
        factor : float
            Multiplicative factor to apply (e.g. ``1.2`` inflates by 20 %,
            ``0.8`` deflates by 20 %).
        length : int or ``None``, default ``None``
            Number of samples to affect.
            • ``None`` -- scale from *when* **to the end**.
            • *n*      -- scale exactly *n* successive samples.

        Returns
        -------
        self : SyntheticSeries
            Enables fluent chaining.

        Examples
        --------
        >>> (SyntheticSeries("2025-01-01", 120, freq="D", seed=0)
        ...      .add_white_noise(0.3)
        ...      .add_step_scale("2025-02-15", factor=1.5)      # permanent gain
        ...      .add_step_scale(90, factor=0.7, length=7))     # one-week dip
        """
        # --- resolve `when` to a positional start index ----------------
        if isinstance(when, (str, pd.Timestamp)):
            when = pd.to_datetime(when)
            if when not in self.index:
                raise KeyError(f"{when} is not in the series index")
            start_idx = self.index.get_loc(when)
        elif isinstance(when, int):
            start_idx = when
        else:
            raise TypeError("'when' must be int or datetime-like")

        if not 0 <= start_idx < len(self._values):
            raise IndexError("start index out of bounds")

        # --- end index -------------------------------------------------
        end_idx = len(self._values) if length is None else start_idx + length
        end_idx = min(end_idx, len(self._values))

        # --- apply scaling --------------------------------------------
        self._values[start_idx:end_idx] *= factor
        return self

    def add_step_offset(
        self,
        when,
        offset: float,
        length: Optional[int] = None,
    ):
        """
        Introduce an additive offset starting at *when*.

        Parameters
        ----------
        when : int, str, pandas-Timestamp or datetime-like
            Where the step begins.
            • **int** → interpreted as positional index (0-based).
            • **str / Timestamp** → aligned to the corresponding date-index entry
              (raises ``KeyError`` if not present).
        offset : float
            Magnitude of the step change (positive or negative).
        length : int or ``None``, default ``None``
            How many samples to adjust.
            • ``None`` → apply offset from *when* **through to the end**
            • *n*     → apply only to the next *n* samples

        Returns
        -------
        self : SyntheticSeries
            Enables method chaining.

        Examples
        --------
        >>> (SyntheticSeries("2024-01-01", 100, seed=1)
        ...      .add_white_noise(0.2)
        ...      .add_step_offset("2024-02-15", offset=3.5)        # persistent
        ...      .add_step_offset(70, offset=-2.0, length=5))      # 5-point dip
        """
        # -------------------------------------------------------------
        # Resolve *when* to a slice of underlying values
        if isinstance(when, (str, pd.Timestamp, pd.DatetimeIndex)):
            when = pd.to_datetime(when)
            if when not in self.index:
                raise KeyError(f"{when} is not in the series index")
            start_idx = self.index.get_loc(when)
        elif isinstance(when, int):
            start_idx = when
        else:
            raise TypeError("'when' must be int or datetime-like")

        if not 0 <= start_idx < len(self._values):
            raise IndexError("start index out of bounds")

        # Determine end index
        end_idx = len(self._values) if length is None else start_idx + length
        end_idx = min(end_idx, len(self._values))

        # Apply the offset
        self._values[start_idx:end_idx] += offset
        return self

    # ---------------------------------------------------------------------
    # Getters / converters
    # ---------------------------------------------------------------------
    def to_series(self, name: str = "value") -> pd.Series:
        """Return the data as a :class:`pandas.Series`."""
        return pd.Series(self._values, index=self.index, name=name)

    def to_dataframe(self, col: str = "value") -> pd.DataFrame:
        """Return the data as a single-column :class:`pandas.DataFrame`."""
        return self.to_series(name=col).to_frame()

    # Convenience representation
    def __repr__(self):  # pragma: no cover
        return f"SyntheticSeries(len={len(self._values)}, freq={self.index.freqstr})"
