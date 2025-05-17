import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Sequence, Tuple, Dict


class TimeseriesGenerator:
    def __init__(
        self,
        start_date: str = "2024-01-01",
        periods: int = 365,
        frequency: str = "D",
        random_seed: int = 42,
    ):
        """
        Initialize a new :class:`TimeseriesGenerator`.

        This constructor prepares the calendar grid on which all synthetic
        values will be built.  It converts the supplied ``start_date`` to a
        :class:`pandas.Timestamp`, constructs a :class:`pandas.DatetimeIndex`
        of length ``periods`` at the requested ``frequency``, and stores a
        reproducible NumPy random number generator seeded with
        ``random_seed``.

        Parameters
        ----------
        start_date : str, default ``"2024-01-01"``
            First timestamp (inclusive) of the series.  Any string accepted
            by :func:`pandas.to_datetime` is allowed.
        periods : int, default ``365``
            Total number of time steps to generate.
        frequency : str, default ``"D"``
            Pandas offset alias defining the spacing between consecutive
            time steps (e.g. ``"H"`` for hourly, ``"30min"`` for half-hourly,
            ``"D"`` for daily).
        random_seed : int, default ``42``
            Seed for the internal random number generator.  Passing the same
            value ensures reproducible results.

        Attributes
        ----------
        start_date : pandas.Timestamp
            Parsed starting timestamp.
        periods : int
            Number of generated periods.
        frequency : str
            Frequency alias passed to :func:`pandas.date_range`.
        date_range : pandas.DatetimeIndex
            Index spanning the generated timestamps.
        rng : numpy.random.Generator
            Random number generator used by all stochastic components.

        Examples
        --------
        >>> gen = TimeseriesGenerator("2023-10-01", periods=48, frequency="H")
        >>> gen.date_range[:3]
        DatetimeIndex(['2023-10-01 00:00:00', '2023-10-01 01:00:00',
                    '2023-10-01 02:00:00'],
                    dtype='datetime64[ns]', freq='H')
        """
        self.start_date = pd.to_datetime(start_date)
        self.rng = np.random.default_rng(random_seed)
        self.periods = periods
        self.frequency = frequency
        self.date_range = pd.date_range(start=start_date, periods=periods, freq=frequency)

    def _generate_seasonal_pattern(
        self,
        period: int | float,
        amplitude: float = 1.0,
        phase: float = 0.0,
    ):
        """
        Create a sinusoidal seasonal component.

        The seasonal signal is computed as::

            s(t) = A · sin(2π (t + φ) / P)

        where *t* is the integer time step (0 → ``self.periods-1``),
        *A* is the ``amplitude``, *P* is the ``period`` (in identical
        time-step units), and *φ* is the phase offset.

        Parameters
        ----------
        period : int or float
            Length of one full seasonal cycle, expressed in number of
            generated time steps (e.g., ``365`` for an annual daily cycle or
            ``7`` for a weekly daily cycle).
        amplitude : float, default ``1.0``
            Peak-to-trough half-range of the sine wave.  The resulting series
            ranges from ``-amplitude`` to ``+amplitude``.
        phase : float, default ``0.0``
            Horizontal shift (in **time-step units**) applied to the wave.
            Positive values move the pattern earlier in time.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of length ``self.periods`` containing the
            seasonal signal.

        Notes
        -----
        The phase is expressed in the same units as the time index
        (steps).  For example, with daily data a phase of ``182.5`` shifts
        an annual cycle by half a year.

        Examples
        --------
        >>> gen = TimeseriesGenerator(periods=10, random_seed=0)
        >>> gen._generate_seasonal_pattern(period=5, amplitude=2)
        array([ 0.        ,  2.35114101,  3.80422607,  3.80422607,  2.35114101,
                0.        , -2.35114101, -3.80422607, -3.80422607, -2.35114101])
        """
        t = np.arange(self.periods)
        return amplitude * np.sin(2 * np.pi * (t + phase) / period)

    def _generate_trend(self, slope: float = 0.1):
        """
        Produce a deterministic linear trend.

        The trend is simply *t · slope* where *t* ranges from ``0`` to
        ``self.periods - 1`` (inclusive).  It can be interpreted as a
        constant growth or decay rate applied to the series.

        Parameters
        ----------
        slope : float, default ``0.1``
            Increment per time step.  Positive values create an upward trend,
            negative values a downward trend.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of length ``self.periods`` representing the
            trend component.

        Examples
        --------
        >>> gen = TimeseriesGenerator(periods=5)
        >>> gen._generate_trend(slope=2)
        array([0, 2, 4, 6, 8])
        """
        return np.arange(self.periods) * slope

    def _add_noise(
        self,
        data: np.ndarray | pd.Series,
        noise_level: float = 0.1,
    ):
        """
        Add zero-mean Gaussian noise to a series.

        A fresh sample of white noise :math:`\\varepsilon_t \\sim
        \\mathcal{N}(0, \\sigma^2)` is generated for each time step, where
        :math:`\\sigma =` ``noise_level``.  The noise is then added
        element-wise to ``data``.  The output length always equals
        ``self.periods``; if *data* is shorter, it is broadcast or repeated
        according to NumPy’s rules.

        Parameters
        ----------
        data : array_like or pandas.Series
            Base signal to which noise will be added.  It must be
            broadcast-compatible with an array of length ``self.periods``.
        noise_level : float, default ``0.1``
            Standard deviation of the Gaussian noise.

        Returns
        -------
        numpy.ndarray
            Noisy signal with the same shape as the broadcasted input.

        Notes
        -----
        Reproducibility depends on the random number generator used.
        Consider seeding NumPy’s RNG (or using the per-instance
        :pyattr:`~TimeseriesGenerator.rng` attribute introduced earlier) when
        deterministic results are required.

        Examples
        --------
        >>> gen = TimeseriesGenerator(periods=5, random_seed=0)
        >>> base = np.ones(5)
        >>> gen._add_noise(base, noise_level=0.2)      # doctest: +SKIP
        array([1.3528, 0.9380, 1.2050, 1.4849, 0.9884])
        """
        return data + np.random.normal(0, noise_level, self.periods)

    def _generate_anomalies(
        self,
        data: np.ndarray | pd.Series,
        num_anomalies: int = 5,
        amplitude_range: tuple[float, float] = (2.0, 4.0),
    ):
        """
        Inject point anomalies (outliers) into a time series.

        For each anomaly, a random index is selected without replacement.
        The value at that index is then perturbed by

        .. math::

            x_i \\leftarrow x_i \\;\\pm\\; U(a_\\text{min},\\; a_\\text{max})\\;\\sigma

        where :math:`\\sigma` is the standard deviation of the input
        ``data`` (clipped to ``1.0`` to avoid vanishing anomalies), and the
        sign is chosen with equal probability.

        Parameters
        ----------
        data : array_like or pandas.Series
            Original time-series values.  Must be at least ``self.periods``
            long; if longer, only the first ``self.periods`` elements are
            considered.
        num_anomalies : int, default ``5``
            Number of anomalous points to create.  Must satisfy
            ``0 < num_anomalies <= self.periods``.
        amplitude_range : tuple of float, default ``(2.0, 4.0)``
            Inclusive lower and upper bounds for the uniform multiplier that
            scales the standard deviation.  Larger values make anomalies
            more extreme.

        Returns
        -------
        data_with_anomalies : numpy.ndarray
            A copy of *data* with the injected anomalies.
        anomaly_indices : numpy.ndarray
            Sorted array of integer indices (shape ``(num_anomalies,)``)
            indicating where anomalies were placed.

        Notes
        -----
        * Because indices are chosen without replacement, each anomaly falls
        at a unique time step.
        * If the original standard deviation is zero, a fallback value of
        ``1.0`` is used so that anomalies remain non-trivial.

        Examples
        --------
        >>> gen = TimeseriesGenerator(periods=12, random_seed=1)
        >>> base = np.zeros(12)
        >>> noisy, idx = gen._generate_anomalies(base, num_anomalies=3, amplitude_range=(3, 3))
        >>> idx          # doctest: +SKIP
        array([ 1,  5, 11])
        >>> noisy[idx]   # doctest: +SKIP
        array([-3., +3., +3.])
        """

        anomaly_indices = self.rng.choice(
            self.periods,
            num_anomalies,
            replace=False,
        )
        anomaly_data = data.copy()

        # Calculate standard deviation, use 1.0 if it's zero
        std_dev = max(np.std(data), 1.0)

        for idx in anomaly_indices:
            multiplier = self.rng.uniform(*amplitude_range)
            sign = self.rng.choice([-1, 1])
            anomaly_data[idx] += sign * multiplier * std_dev

        return anomaly_data, anomaly_indices

    def apply_long_term_shift(
        self,
        data: np.ndarray | pd.Series,
        shift_start: str | int | pd.Timestamp | None = None,
        magnitude: float | tuple[float, float] = 1.0,
        ramp: bool = False,
        ramp_length: int | None = None,
    ) -> np.ndarray:
        """
        Apply a persistent shift (or *regime change*) to a series.

        The function constructs an offset vector *s(t)* such that

        * for *t* **before** ``shift_start``  → *s(t) = 0*
        * for *t* **after/at** ``shift_start`` → *s(t) = Δ*  (step)
        or a linear ramp that reaches Δ after ``ramp_length`` steps.

        Parameters
        ----------
        data : array_like or pandas.Series
            Original signal of length ``self.periods``.
        shift_start : {str, int, pandas.Timestamp, None}, optional
            When the shift begins.

            * **str or Timestamp** – parsed against ``self.date_range``.
            * **int** – interpreted as positional index (0-based).
            * **None** – a random index is drawn from
            ``[self.periods // 4, 3 * self.periods // 4]``.
        magnitude : float or tuple[float, float], default ``1.0``
            Desired offset Δ.  If a `(low, high)` tuple is supplied, Δ is
            drawn uniformly from that range (useful for simulation).
        ramp : bool, default ``False``
            If *True*, offset grows linearly from 0 to Δ over
            ``ramp_length`` time steps (inclusive).  Otherwise a sharp step
            change is applied.
        ramp_length : int, optional
            Number of samples over which to ramp.  Required when
            ``ramp=True``.  Ignored otherwise.

        Returns
        -------
        shifted : numpy.ndarray
            Copy of *data* with the long-term shift applied.

        Examples
        --------
        >>> gen = TimeseriesGenerator(periods=30, random_seed=0)
        >>> x = np.zeros(30)
        >>> y = gen.apply_long_term_shift(x, shift_start=15, magnitude=5)
        >>> y[14:18]
        array([0., 5., 5., 5.])

        >>> # Gradual 10-step ramp starting on 2024-02-01
        >>> y = gen.apply_long_term_shift(
        ...     x,
        ...     shift_start="2024-02-01",
        ...     magnitude=3,
        ...     ramp=True,
        ...     ramp_length=10,
        ... )
        """
        # ------------- validate & parse -----------------
        if isinstance(data, pd.Series):
            data_arr = data.values
        else:
            data_arr = np.asarray(data)

        if data_arr.shape[0] < self.periods:
            raise ValueError("Input 'data' must be at least self.periods long")

        # magnitude
        if isinstance(magnitude, (tuple, list)):
            delta = self.rng.uniform(*magnitude)
        else:
            delta = float(magnitude)

        # shift start
        if shift_start is None:  # choose random position away from edges
            low = self.periods // 4
            high = 3 * self.periods // 4
            idx0 = int(self.rng.integers(low, high))
        elif isinstance(shift_start, (str, pd.Timestamp)):
            ts = pd.to_datetime(shift_start)
            if ts not in self.date_range:
                raise ValueError("shift_start timestamp outside generator range")
            idx0 = int(np.searchsorted(self.date_range, ts))
        else:  # assume int
            idx0 = int(shift_start)

        if not (0 <= idx0 < self.periods):
            raise ValueError("shift_start index out of bounds")

        # ramp length
        if ramp:
            if ramp_length is None:
                raise ValueError("ramp_length must be provided when ramp=True")
            if ramp_length <= 0:
                raise ValueError("ramp_length must be positive")
            idx1 = min(idx0 + ramp_length, self.periods)
        else:
            idx1 = idx0 + 1  # immediate step

        # ------------- build offset vector --------------
        offset = np.zeros(self.periods)

        if ramp:
            # linear growth from 0 to Δ
            offset[idx0:idx1] = np.linspace(0, delta, idx1 - idx0, endpoint=False)
            offset[idx1:] = delta
        else:
            offset[idx0:] = delta

        # ------------- apply & return -------------------

        return data_arr + offset

    def generate_simple_series(self):
        series = self._generate_seasonal_pattern(365, amplitude=1.5) + self._generate_trend()
        return pd.Series(series, index=self.date_range, name="original")

    def generate_correlated_series(
        self,
        n_series: int = 2,
        correlation_matrix: np.ndarray | None = None,
        seasonal_periods: list[int] | None = None,
        trend_slopes: list[float] | None = None,
        noise_levels: list[float] | None = None,
        num_anomalies: int = 5,
        enforce_final_correlation: bool = False,
    ):
        """
        Synthesize one or more inter-correlated time-series, each containing
        *seasonal cycles*, a *linear trend*, *Gaussian noise*, and injected
        *point anomalies*.

        The procedure is:

        1. Build a deterministic *base signal* for every series by summing
        sine-wave seasonal components defined in ``seasonal_periods`` and
        a linear trend with slope from ``trend_slopes``.
        2. Draw a matrix of *correlated white-noise* samples using a
        Cholesky factorisation of ``correlation_matrix`` and add the
        scaled noise (``noise_levels``) to each base signal.
        3. Inject ``num_anomalies`` additive outliers into every column via
        :pymeth:`_generate_anomalies`.
        4. Return the results in a :class:`pandas.DataFrame` whose index is
        ``self.date_range``.

        Parameters
        ----------
        n_series : int, default ``2``
            Number of parallel series (columns) to generate.
        correlation_matrix : ndarray of shape (n_series, n_series), optional
            Target Pearson-correlation matrix applied to the *noise*
            component.  Must be symmetric positive-definite.  If *None*, a
            matrix with value ``1`` on the diagonal and ``0.5`` elsewhere is
            used.
        seasonal_periods : list[int] or tuple[int], optional
            Length(s) of the seasonal cycles, expressed in *time-step*
            units.  For daily data a value of ``7`` yields a weekly cycle,
            ``365`` an annual cycle, etc.  Defaults to ``[365, 7]``.  (A new
            sine wave is added for every entry.)
        trend_slopes : list[float], optional
            Per-series trend coefficients.  If *None*, every series receives
            a slope of ``0.1``.  Length must equal ``n_series``.
        noise_levels : list[float], optional
            Standard deviation multipliers for the correlated white noise.
            If *None*, each element defaults to ``0.1``.  Length must equal
            ``n_series``.
        num_anomalies : int, default ``5``
            Number of outliers (per series) to inject.  Must satisfy
            ``0 < num_anomalies <= self.periods``.
        enforce_final_correlation : bool, default ``True``
            If *True*, the routine whitens the complete signals (seasonality
            + trend + noise) and “re-colours” them so the resulting
            covariance—and therefore correlation—matrix equals
            ``correlation_matrix``.  Set to *False* to keep the faster,
            noise-only approximation.

        Returns
        -------
        df : pandas.DataFrame
            Synthetic data with columns ``"series_1"`` … ``"series_n"`` and
            a :class:`pandas.DatetimeIndex` identical to
            :pyattr:`self.date_range`.
        anomaly_locations : dict[str, pandas.DatetimeIndex]
            Mapping of column name → timestamps at which anomalies were
            placed.

        Raises
        ------
        ValueError
            If the shapes or lengths of *correlation_matrix*,
            *trend_slopes*, or *noise_levels* do not match *n_series*.

        Notes
        -----
        * Because only the noise component is forced to follow
        ``correlation_matrix``, the **final correlation** of the complete
        signals may differ—especially when strong shared seasonality or
        trends dominate.
        * All stochastic elements draw from NumPy’s global RNG.  Seed it
        (e.g. ``np.random.seed(42)``) or migrate to a dedicated
        ``numpy.random.Generator`` for reproducibility.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> np.random.seed(0)
        >>> gen = TimeseriesGenerator(periods=24, frequency="H")
        >>> df, anomalies = gen.generate_correlated_series(
        ...     n_series=3,
        ...     correlation_matrix=np.array([[1.0, 0.8, 0.2],
        ...                                  [0.8, 1.0, 0.3],
        ...                                  [0.2, 0.3, 1.0]]),
        ...     seasonal_periods=[24],        # diurnal cycle (hourly data)
        ...     trend_slopes=[0.0, 0.1, -0.05],
        ...     noise_levels=[0.05, 0.05, 0.05],
        ...     num_anomalies=2,
        ... )
        >>> df.head(3)
                                series_1  series_2  series_3
        2024-01-01 00:00:00  0.000000  0.000000  0.000000
        2024-01-01 01:00:00  0.250000  0.355398  0.292711
        2024-01-01 02:00:00  0.475528  0.641300  0.568112
        >>> anomalies["series_1"]
        DatetimeIndex(['2024-01-01 07:00:00', '2024-01-01 18:00:00'],
                    dtype='datetime64[ns]', freq=None)
        """

        # ---------- defaults & validation ----------
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_series)
            correlation_matrix[correlation_matrix == 0] = 0.5

        if correlation_matrix.shape != (n_series, n_series):
            raise ValueError("correlation_matrix must be n_series × n_series")

        if noise_levels is None:
            noise_levels = [0] * n_series
        if len(noise_levels) != n_series:
            raise ValueError("noise_levels length must equal n_series")

        # ---------- deterministic part: seasonality + trend ----------
        base_series = []
        for i in range(n_series):
            series = np.zeros(self.periods)

            # add seasonal cycles
            if seasonal_periods is None:
                series = series
            else:
                for period in seasonal_periods:
                    series += self._generate_seasonal_pattern(
                        period=period,
                        amplitude=np.random.uniform(0.5, 5.0),
                        phase=np.random.uniform(0, period),
                    )

            # add linear trend
            series += self._generate_trend(trend_slopes[i])
            base_series.append(series)

        base_series = np.array(base_series)  # shape (n_series, periods)

        # ---------- stochastic part: correlated white noise ----------
        L = np.linalg.cholesky(correlation_matrix)
        correlated_noise = L @ np.random.randn(n_series, self.periods)

        for i in range(n_series):
            base_series[i] += noise_levels[i] * correlated_noise[i]

        # ---------- *optional* exact correlation enforcement ----------
        if enforce_final_correlation:
            # 1. demean
            means = base_series.mean(axis=1, keepdims=True)
            centered = base_series - means

            # 2. whiten → covariance ≈ identity
            cov_current = centered @ centered.T / (self.periods - 1)
            # eigen-decomposition for symmetric PSD matrix
            eigvals, eigvecs = np.linalg.eigh(cov_current)
            # numerical guard against tiny/negative eigenvalues
            eigvals[eigvals < 0] = 0.0
            W = eigvecs @ np.diag(eigvals**-0.5) @ eigvecs.T
            whitened = W @ centered

            # 3. colour with target correlation
            L_target = np.linalg.cholesky(correlation_matrix)
            recoloured = L_target @ whitened

            # 4. add the original means back
            base_series = recoloured + means

        # ---------- assemble DataFrame ----------
        df = pd.DataFrame(
            base_series.T,
            index=self.date_range,
            columns=[f"series_{i + 1}" for i in range(n_series)],
        )

        # ---------- anomalies ----------
        anomaly_locations: Dict[str, pd.DatetimeIndex] = {}
        for col in df.columns:
            df[col], idx = self._generate_anomalies(df[col].values, num_anomalies=num_anomalies)
            anomaly_locations[col] = self.date_range[idx]

        return df, anomaly_locations
