"""
water_indicators.py
====================
Computation of climate / water-balance / hydrological-stress indicators for
the Climate-Water Balance Dashboard.

Works on a long-format monthly time series with one row per
(spatial unit, date) and columns:

    unit  - basin or administrative-unit name (grouping key, optional)
    date  - monthly timestamp
    PCP   - precipitation (mm)
    ET    - actual evapotranspiration (mm)          [AETI in source CSVs]
    RET   - reference / potential evapotranspiration (mm)
    TWS   - terrestrial water storage (consistent unit, e.g. mm or cm eq.)

Indicators produced:
    CWB    - Climatic Water Balance = PCP - RET (mm)
    AI     - Aridity Index (UNEP) = PCP / RET
    ESI    - Evaporative Stress Index: standardized anomaly of -(ET/RET);
             positive = more evaporative stress
    SPEI   - Standardized Precipitation-Evapotranspiration Index, computed
             with the `spei` package (Vonk, 2025): the climatic water
             balance (CWB = PCP - RET) is accumulated over `spei_scale`
             months and standardized by fitting a log-logistic (fisk)
             distribution per calendar month — the distribution recommended
             in the original SPEI literature (Vicente-Serrano et al., 2010).
             Falls back to a simplified per-calendar-month normal-fit
             standardization if the `spei` package isn't installed or the
             distribution fit fails (e.g. very short/degenerate records).
    TWSA   - standardized anomaly of TWS, per calendar month (GRACE-style
             terrestrial water storage anomaly)

    SPEI_status, Aridity_class - categorical labels for display
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from scipy import stats

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

try:
    import spei as spei_pkg

    _HAS_SPEI_PKG = True
except ImportError:  # pragma: no cover
    _HAS_SPEI_PKG = False

# DEFAULT_WEIGHTS = {"SPI": 0.3, "SPEI": 0.3, "ESI": 0.2, "TWSA": 0.2}

REQUIRED_COLUMNS = ["date", "PCP", "ET", "RET", "TWS"]

# Ordered high -> low. classify_wai walks this list and returns the first
# label whose threshold the value meets or exceeds.
WAI_THRESHOLDS = [
    (2.0, "Extremely Wet / Surplus"),
    (1.5, "Very Wet / High Availability"),
    (1.0, "Moderately Wet / Good Availability"),
    (-1.0, "Near Normal"),
    (-1.5, "Moderately Dry / Watch"),
    (-2.0, "Severely Dry / Alert"),
    (-np.inf, "Extremely Dry / Critical"),
]

SPEI_THRESHOLDS = {
    "Extremely Wet": 2.0,  # SPEI > 2
    "Very Wet": 1.5,  # 1.5 < SPEI <= 2
    "Moderately Wet": 1.0,  # 1 < SPEI <= 1.5
    "Near Normal": -1.0,  # -1 <= SPEI <= 1
    "Moderately Dry": -1.5,  # -1.5 <= SPEI < -1
    "Severely Dry": -2.0,  # -2 <= SPEI < -1.5
    # anything below -2 -> "Extremely Dry / Critical"
}

ARIDITY_THRESHOLDS = [
    (0.75, "Humid"),
    (0.5, "Dry Sub-Humid"),
    (0.2, "Semi-Arid"),
    (0.05, "Arid"),
    (-np.inf, "Hyper-Arid"),
]


@dataclass
class IndicatorConfig:
    spi_scale: int = 3
    spei_scale: int = 3
    twsa_scale: int = 1
    # weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


def classify_wai(value) -> str:
    if pd.isna(value):
        return "No Data"
    for threshold, label in WAI_THRESHOLDS:
        if value >= threshold:
            return label
    return "Extremely Dry / Critical"


def classify_spei(value: float) -> str:
    """Classify a SPEI value into a water-availability status category
    using SPEI_THRESHOLDS: |SPEI| <= 1 Near Normal; 1-1.5 / -1 to -1.5
    Moderately Wet/Dry; 1.5-2 / -1.5 to -2 Very Wet/Dry; beyond +/-2
    Extremely Wet/Dry."""
    if pd.isna(value):
        return "No Data"
    if value > SPEI_THRESHOLDS["Extremely Wet"]:
        return "Extremely Wet"
    elif value > SPEI_THRESHOLDS["Very Wet"]:
        return "Very Wet"
    elif value > SPEI_THRESHOLDS["Moderately Wet"]:
        return "Moderately Wet"
    elif value >= SPEI_THRESHOLDS["Near Normal"]:
        return "Near Normal"
    elif value >= SPEI_THRESHOLDS["Moderately Dry"]:
        return "Moderately Dry"
    elif value >= SPEI_THRESHOLDS["Severely Dry"]:
        return "Severely Dry"
    else:
        return "Extremely Dry"


def classify_aridity(ai) -> str:
    if pd.isna(ai):
        return "No Data"
    for threshold, label in ARIDITY_THRESHOLDS:
        if ai >= threshold:
            return label
    return "Hyper-Arid"


def _rolling_accum(s: pd.Series, scale: int) -> pd.Series:
    if scale <= 1:
        return s
    return s.rolling(window=scale, min_periods=scale).sum()


def _normal_standardize_by_month(values: pd.Series, months: pd.Series) -> pd.Series:
    """Per-calendar-month z-score standardization (mean/stdev of that
    calendar month's own history)."""
    df = pd.DataFrame({"v": values.values, "m": months.values}, index=values.index)
    out = pd.Series(index=values.index, dtype=float)
    for _, grp in df.groupby("m"):
        valid = grp["v"].dropna()
        if len(valid) < 3:
            out.loc[grp.index] = np.nan
            continue
        mu, sigma = valid.mean(), valid.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            out.loc[grp.index] = 0.0
        else:
            out.loc[grp.index] = (grp["v"] - mu) / sigma
    return out


def _gamma_standardize_by_month(values: pd.Series, months: pd.Series) -> pd.Series:
    """
    Zero-inflated gamma standardization per calendar month (the standard
    operational SPI method). Falls back to a per-month normal-score
    transform when scipy is unavailable or a fit fails (too few points,
    degenerate/negative data, etc.) - this keeps the dashboard usable on
    short or sparse basin records.
    """
    fallback = _normal_standardize_by_month(values, months)
    if not _HAS_SCIPY:
        return fallback

    df = pd.DataFrame({"v": values.values, "m": months.values}, index=values.index)
    out = pd.Series(index=values.index, dtype=float)
    for _, grp in df.groupby("m"):
        valid = grp["v"].dropna()
        if len(valid) < 6:
            out.loc[grp.index] = fallback.loc[grp.index]
            continue
        zeros = int((valid <= 0).sum())
        p0 = zeros / len(valid)
        positive = valid[valid > 0]
        if len(positive) < 5 or positive.std() == 0:
            out.loc[grp.index] = fallback.loc[grp.index]
            continue
        try:
            shape, _loc, scale = stats.gamma.fit(positive, floc=0)
            v = grp["v"].values
            cdf = np.where(
                v > 0,
                p0
                + (1 - p0)
                * stats.gamma.cdf(np.clip(v, 1e-9, None), shape, loc=0, scale=scale),
                p0 / 2.0,
            )
            cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
            out.loc[grp.index] = stats.norm.ppf(cdf)
        except Exception:
            out.loc[grp.index] = fallback.loc[grp.index]
    return out


def _spei_via_package(cwb: pd.Series, dates: pd.Series, timescale: int) -> pd.Series:
    """Compute SPEI from a climatic-water-balance series using the `spei`
    package (log-logistic/fisk fit per calendar month on the rolling CWB
    accumulation) — this is the literature-standard SPEI method, not the
    simplified normal-fit approximation.

    `timescale` is the accumulation window in months (0/1 = no
    accumulation). Falls back to the simplified per-calendar-month normal
    standardization (on a manually rolled CWB sum) if the `spei` package
    isn't installed or the distribution fit fails outright (e.g. too few
    points, degenerate/constant series).

    Returns a Series aligned to the original `dates`/index, with NaN for
    any leading periods the package couldn't compute (the first
    `timescale - 1` months of each unit's record).
    """
    months = dates.dt.month
    fallback = _normal_standardize_by_month(_rolling_accum(cwb, timescale), months)

    if not _HAS_SPEI_PKG:
        return fallback

    series = pd.Series(cwb.values, index=pd.DatetimeIndex(dates.values))
    try:
        result = spei_pkg.spei(series, timescale=timescale)
    except Exception:
        return fallback

    # The package drops the first (timescale - 1) periods rather than
    # returning NaN for them, so reindex back onto the full date range.
    result = result.reindex(series.index)
    out = pd.Series(result.values, index=cwb.index)
    # Any values the fit couldn't produce (non-finite) fall back to the
    # simplified normal standardization rather than being left as NaN.
    bad = ~np.isfinite(out)
    if bad.any():
        out.loc[bad] = fallback.loc[bad]
    return out


def _compute_unit(g: pd.DataFrame, config: IndicatorConfig) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    month = g["date"].dt.month

    g["CWB"] = g["PCP"] - g["RET"]
    g["AI"] = g["PCP"] / g["RET"].replace(0, np.nan)
    g["Aridity_class"] = g["AI"].apply(classify_aridity)

    evap_fraction = g["ET"] / g["RET"].replace(0, np.nan)
    ef_anom = _normal_standardize_by_month(evap_fraction, month)
    g["ESI"] = -ef_anom  # positive = more evaporative stress (ET falling short of RET)

    pcp_accum = _rolling_accum(g["PCP"], config.spi_scale)
    # g["SPI"] = _gamma_standardize_by_month(pcp_accum, month)

    g["SPEI"] = _spei_via_package(g["CWB"], g["date"], config.spei_scale)

    tws_accum = _rolling_accum(g["TWS"], config.twsa_scale)
    g["TWSA"] = _normal_standardize_by_month(tws_accum, month)

    g["SPEI_status"] = g["SPEI"].apply(classify_spei)
    return g


def compute_all_indicators(
    df: pd.DataFrame, unit_col=None, config: IndicatorConfig = None
) -> pd.DataFrame:
    """Compute all indicators, grouped by `unit_col` if given (recommended -
    calendar-month standardization is done independently per group)."""
    config = config or IndicatorConfig()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required column(s): {', '.join(missing)}"
        )

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])

    if unit_col and unit_col in work.columns:
        pieces = [_compute_unit(g, config) for _, g in work.groupby(unit_col)]
        out = pd.concat(pieces, ignore_index=True)
    else:
        out = _compute_unit(work, config)
    return out


def latest_snapshot(df: pd.DataFrame, location_col=None) -> pd.DataFrame:
    """Most recent row per unit (or overall, if no location_col)."""
    if location_col and location_col in df.columns:
        idx = df.groupby(location_col)["date"].idxmax()
        return df.loc[idx].reset_index(drop=True)
    latest_date = df["date"].max()
    return df[df["date"] == latest_date].reset_index(drop=True)
