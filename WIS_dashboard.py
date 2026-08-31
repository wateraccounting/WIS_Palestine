"""
dashboard.py
=============
Climate-Water Balance and Hydrological Stress Dashboard.

Lets the user work at either River Basin or Administrative Unit level. All
data is read automatically from a fixed folder layout on disk (see below).
Computes water availability indicators (CWB,
AI, ESI, SPEI and TWSA), and displays them via a
status-card summary, trend charts, a choropleth spatial map, a monthly
anomaly heatmap, and data export.

Expected data folder layout
----------------------------
    data/river_basins/boundary/*.shp   (or .geojson / .zip)
    data/river_basins/monthly/<unit name>.csv
    data/river_basins/annual.csv

    data/admin_units/boundary/*.shp    (or .geojson / .zip)
    data/admin_units/monthly/<unit name>.csv
    data/admin_units/annual.csv

- `boundary/` is scanned for the first usable boundary file, in this order
  of preference: .shp, .geojson/.json, .zip (a zipped shapefile/geojson).
  The unit-name column is expected to be `BASIN_NAME` for River Basins and
  `Admin_level1` for Administrative Units.
- `monthly/` holds one CSV per unit (columns: time;PCP;AETI;RET;TWS); the
  unit name is taken from the filename.
- `annual.csv` holds one row per unit/year (columns:
  year;River_basins;PCP;AETI;RET;TWS;area_sq_km).

Run:
    streamlit run dashboard.py
"""

import io
import json
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import geopandas as gpd

    _HAS_GEO = True
except ImportError:  # pragma: no cover
    _HAS_GEO = False

from water_indicators import (
    IndicatorConfig,
    classify_spei,
    compute_all_indicators,
    latest_snapshot,
)

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Climate-Water Balance & Hydrological Stress Dashboard",
    page_icon="💧",
    layout="wide",
)


STATUS_COLORS = {
    "Extremely Wet": "#08519c",
    "Very Wet": "#3182bd",
    "Moderately Wet": "#6baed6",
    "Near Normal": "#31a354",
    "Moderately Dry": "#fec44f",
    "Severely Dry": "#e6550d",
    "Extremely Dry": "#a50f15",
    "No Data": "#bdbdbd",
}

ARIDITY_COLORS = {
    "Humid": "#08519c",
    "Dry Sub-Humid": "#3182bd",
    "Semi-Arid": "#fec44f",
    "Arid": "#e6550d",
    "Hyper-Arid": "#a50f15",
    "No Data": "#bdbdbd",
}

# Standardized indices (SPEI, TWSA, ESI) are all z-score-like and
# are consistently displayed on a fixed -3..3 axis / color range so that
# magnitudes are comparable across charts, units and the map.
STD_INDEX_RANGE = (-3, 3)

# BrBG ("Brown-Blue-Green") is the standard diverging colormap for
# hydrological/drought anomalies: brown = dry / water deficit, teal-blue =
# wet / water surplus, with a neutral tan-white at zero. This reads far
# more intuitively for a water-balance audience than a generic red-blue
# temperature-style scale.
WATER_COLORSCALE = "BrBG"

# Fixed unit-name column expected in the boundary file, per spatial unit.
BOUNDARY_ID_COLUMN = {
    "River Basins": "BASIN_NAME",
    "Administrative Units": "Admin_level1",
}


def status_badge(label: str) -> str:
    color = STATUS_COLORS.get(label, "#bdbdbd")
    return (
        f"<span style='background-color:{color};color:white;padding:4px 10px;"
        f"border-radius:12px;font-size:0.85rem;font-weight:600;'>{label}</span>"
    )


# --------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------


def find_boundary_file(folder: str):
    """Scan `folder` for a usable boundary file and return its path, or
    None if the folder doesn't exist or has nothing usable.

    Preference order when several are present: .shp, then .geojson/.json,
    then .zip (a zipped shapefile/geojson).
    """
    if not folder or not os.path.isdir(folder):
        return None
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return None
    for ext in (".shp", ".geojson", ".json", ".zip"):
        matches = [f for f in entries if f.lower().endswith(ext)]
        if matches:
            return os.path.join(folder, matches[0])
    return None


def load_boundary_from_path(path: str):
    """Load a boundary file on disk — .shp, .geojson/.json, or a .zip
    containing either — with geopandas, reprojecting to EPSG:4326."""
    if path.lower().endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(".shp")]
            geo_files = [
                f
                for f in os.listdir(tmpdir)
                if f.lower().endswith((".geojson", ".json"))
            ]
            if shp_files:
                gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
            elif geo_files:
                gdf = gpd.read_file(os.path.join(tmpdir, geo_files[0]))
            else:
                raise ValueError(f"No .shp or .geojson file found inside {path}.")
    else:
        gdf = gpd.read_file(path)
    if gdf.crs is not None and str(gdf.crs).upper() not in ("EPSG:4326",):
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _read_csv_flex_path(path: str) -> pd.DataFrame:
    """Try ';'-delimited first (as specified for these products), fall back
    to auto-detected/standard comma CSV."""
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:  # wrong delimiter guessed
            raise ValueError("single column - wrong delimiter")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_monthly_csvs_from_folder(folder: str) -> pd.DataFrame:
    """One CSV per basin/admin unit in `folder`, columns:
    time;PCP;AETI;RET;TWS. The unit name is taken from the filename
    (without extension)."""
    empty = pd.DataFrame(columns=["unit", "date", "PCP", "ET", "RET", "TWS"])
    if not folder or not os.path.isdir(folder):
        return empty

    frames = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".csv"):
            continue
        unit_name = os.path.splitext(fname)[0]
        df = _read_csv_flex_path(os.path.join(folder, fname))
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "time":
                rename_map[c] = "date"
            elif cl in ("aeti", "et", "aet"):
                rename_map[c] = "ET"
            elif cl == "pcp":
                rename_map[c] = "PCP"
            elif cl == "ret":
                rename_map[c] = "RET"
            elif cl == "tws":
                rename_map[c] = "TWS"
        df = df.rename(columns=rename_map)
        df["unit"] = unit_name
        frames.append(df)
    if not frames:
        return empty
    return pd.concat(frames, ignore_index=True)


def read_annual_csv_from_path(path: str) -> pd.DataFrame:
    """Annual CSV for all units, columns:
    year;River_basins;PCP;AETI;RET;TWS;area_sq_km."""
    df = _read_csv_flex_path(path)
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in (
            "river_basins",
            "riverbasin",
            "basin",
            "basin_name",
            "adm1_name",
            "admin_unit",
            "unit",
        ):
            rename_map[c] = "unit"
        elif cl == "year":
            rename_map[c] = "year"
        elif cl in ("aeti", "et", "aet"):
            rename_map[c] = "ET"
        elif cl == "pcp":
            rename_map[c] = "PCP"
        elif cl == "ret":
            rename_map[c] = "RET"
        elif cl == "tws":
            rename_map[c] = "TWS"
        elif cl == "area_sq_km":
            rename_map[c] = "area_sq_km"
    return df.rename(columns=rename_map)


def ensure_unit_column(df: pd.DataFrame, label: str, fallback_col: str) -> pd.DataFrame:
    """If a dataframe doesn't already have a 'unit' column, use the fixed
    `fallback_col` (Admin_level1 / BASIN_NAME depending on spatial unit) —
    never prompt the user to pick one."""
    if "unit" in df.columns or df.empty:
        return df
    if fallback_col in df.columns:
        return df.rename(columns={fallback_col: "unit"})
    st.sidebar.error(
        f"{label} has no `unit` or `{fallback_col}` column (found: "
        f"{', '.join(df.columns.tolist())})."
    )
    return df


def _classify_aridity(ai: float) -> str:
    """UNEP aridity classification from the Aridity Index (PCP / RET)."""
    if pd.isna(ai):
        return "No Data"
    if ai >= 0.65:
        return "Humid"
    elif ai >= 0.5:
        return "Dry Sub-Humid"
    elif ai >= 0.2:
        return "Semi-Arid"
    elif ai >= 0.05:
        return "Arid"
    else:
        return "Hyper-Arid"


def period_average_snapshot(
    df: pd.DataFrame, location_col: str = "unit"
) -> pd.DataFrame:
    """Average every numeric indicator over the full (already date-filtered)
    period, per unit, and reclassify SPEI_status / Aridity_class from the
    averaged SPEI / AI rather than reusing any single month's label."""
    numeric_cols = [
        c
        for c in [
            "PCP",
            "ET",
            "RET",
            "TWS",
            "CWB",
            # "AI",
            # "ESI",
            "SPI",
            "SPEI",
            "TWSA",
            # "WAI",
        ]
        if c in df.columns
    ]
    agg = df.groupby(location_col, as_index=False)[numeric_cols].mean(numeric_only=True)
    agg["SPEI_status"] = (
        agg["SPEI"].apply(classify_spei) if "SPEI" in agg.columns else "No Data"
    )

    # agg["WAI_status"] = (
    #     agg["WAI"].apply(classify_wai) if "WAI" in agg.columns else "No Data"
    # )
    agg["Aridity_class"] = (
        agg["AI"].apply(_classify_aridity) if "AI" in agg.columns else "No Data"
    )

    # Keep a 'date' column (period end) purely so downstream code that reads
    # it for display/export purposes keeps working unchanged.
    agg["date"] = df["date"].max() if "date" in df.columns and not df.empty else pd.NaT
    return agg


# --------------------------------------------------------------------------
# Sidebar — spatial unit + data folder
# --------------------------------------------------------------------------

st.sidebar.title("💧 Data & Settings")

if not _HAS_GEO:
    st.sidebar.error(
        "`geopandas` / `shapely` are not installed. Run "
        "`pip install geopandas shapely` to enable boundary loading and the map view."
    )

spatial_unit = st.sidebar.radio(
    "Spatial unit", ["River Basins", "Administrative Units"]
)
unit_label = "basin" if spatial_unit == "River Basins" else "administrative unit"

if spatial_unit == "River Basins":
    data_path = os.path.join("data", "river_basins")
else:
    data_path = os.path.join("data", "admin_units")

data_root = data_path
boundary_id_col = BOUNDARY_ID_COLUMN[spatial_unit]

# with st.sidebar.container(border=True):
#     st.markdown("📂 **Expected data folder layout**")
#     st.caption(
#         f"`1-{data_root}/boundary/*.shp` (`.geojson` / `.zip`)\n"
#         f"`2-{data_root}/monthly/<unit name>.csv`\n"
#         f"`3-{data_root}/annual.csv`"
#     )

boundary_dir = os.path.join(data_path, "boundary")
monthly_dir = os.path.join(data_path, "monthly")
annual_path = os.path.join(data_path, "annual.csv")

# ---- Boundary ----
gdf = None
if _HAS_GEO:
    boundary_path = find_boundary_file(boundary_dir)
    if boundary_path is not None:
        try:
            gdf = load_boundary_from_path(boundary_path)
            if boundary_id_col in gdf.columns:
                gdf = gdf.rename(columns={boundary_id_col: "unit"})
                gdf["unit"] = gdf["unit"].astype(str).str.strip()
            else:
                st.sidebar.error(
                    f"Boundary file `{os.path.basename(boundary_path)}` has no "
                    f"`{boundary_id_col}` column (found: "
                    f"{', '.join(gdf.columns.drop('geometry').tolist())})."
                )
                gdf = None
        except Exception as e:
            st.sidebar.error(f"Could not read boundary file `{boundary_path}`: {e}")
    else:
        st.sidebar.info(f"No boundary file found in `{boundary_dir}`.")

# ---- Monthly CSVs ----
monthly_df = read_monthly_csvs_from_folder(monthly_dir)
if not monthly_df.empty:
    monthly_df["unit"] = monthly_df["unit"].astype(str).str.strip()

# ---- Annual CSV ----
annual_df = None
if os.path.isfile(annual_path):
    annual_df = read_annual_csv_from_path(annual_path)
    annual_df = ensure_unit_column(
        annual_df, "annual.csv", fallback_col=boundary_id_col
    )
    if "unit" in annual_df.columns:
        annual_df["unit"] = annual_df["unit"].astype(str).str.strip()

if monthly_df is None or monthly_df.empty:
    st.title("💧 Climate-Water Balance & Hydrological Stress Dashboard")
    st.info(
        f"No monthly data found in `{monthly_dir}`. Select **{spatial_unit}** and place "
        "data on disk following the folder layout shown in the sidebar, then reload."
    )
    with st.expander("Column reference"):
        st.markdown(
            f"""
**Monthly CSV** — one file per basin/admin unit in `{monthly_dir}/`, filename = unit name

| Column | Meaning | Units |
|---|---|---|
| `time` | Monthly date | YYYY-MM-DD |
| `PCP` | Precipitation | mm |
| `AETI` | Actual evapotranspiration and interception| mm |
| `RET` | Reference evapotranspiration | mm |
| `TWS` | Total water storage | mm equivalent |

**Annual CSV** — single file at `{annual_path}`

| Column | Meaning |
|---|---|
| `year` | Calendar year |
| `River_basins` | Basin / admin-unit name |
| `PCP`, `AETI`, `RET`, `TWS` | Annual totals / mean |
| `area_sq_km` | Unit area |

**Boundary file** — auto-detected in `{boundary_dir}/`

A `.shp` (with its `.shx`/`.dbf`/`.prj` siblings), a `.geojson`/`.json`, or a
`.zip` containing either. The first match found is used. The unit-name
column must be named **`{boundary_id_col}`** for {spatial_unit}.
            """
        )
    st.stop()

# --------------------------------------------------------------------------
# Sidebar layout — reserve slots up front so the visual order is:
#   Spatial unit (above) -> Date range -> Status basis ->
#   Indicator settings (SPEI window) -> Spatial units multiselect
# Each placeholder is filled further down, once the value it needs
# (e.g. computed indicators) is available — the fill order doesn't change
# where it renders, only the order in which we create the placeholders does.
# --------------------------------------------------------------------------

st.sidebar.markdown("---")
ph_date_range = st.sidebar.container()
ph_status_mode = st.sidebar.container()
ph_indicator_settings = st.sidebar.container()
ph_units = st.sidebar.container()

# --------------------------------------------------------------------------
# Indicator settings
# --------------------------------------------------------------------------

spei_scale = ph_indicator_settings.slider("SPEI accumulation window (months)", 1, 12, 3)

config = IndicatorConfig(spei_scale=spei_scale)

# --------------------------------------------------------------------------
# Compute indicators
# --------------------------------------------------------------------------

try:
    indicators = compute_all_indicators(monthly_df, unit_col="unit", config=config)
except ValueError as e:
    st.error(str(e))
    st.stop()

indicators["date"] = pd.to_datetime(indicators["date"])

# Status classification is based on SPEI (climatic water balance,
# standardized) rather than the composite WAI — reuses the same
# McKee-et-al. category thresholds via classify_wai, just applied to SPEI.
indicators["SPEI_status"] = (
    indicators["SPEI"].apply(classify_spei)
    if "SPEI" in indicators.columns
    else "No Data"
)

# Unit filter — with many basins/admin units, plotting all of them at once
# turns the trend charts into unreadable spaghetti, so default to a
# manageable subset. The Spatial Map always shows every unit regardless of
# this selection (see indicators_all below).
all_units = sorted(indicators["unit"].dropna().unique().tolist())
default_units = all_units[: min(5, len(all_units))]
if len(all_units) > 15:
    ph_units.caption(
        f"{len(all_units)} {unit_label}s available — defaulting to {len(default_units)} "
        "for the trend charts to keep them readable. Add more as needed."
    )
selected_units = ph_units.multiselect(
    f"{spatial_unit} (Trends & Anomaly tabs)", all_units, default=default_units
)
if not selected_units:
    ph_units.warning(f"Select at least one {unit_label} to see trend charts.")

# Date range filter (applies to every unit, selected or not — used by the map too)
min_date, max_date = indicators["date"].min(), indicators["date"].max()
date_range = ph_date_range.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
indicators_all = indicators
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    indicators_all = indicators_all[
        (indicators_all["date"] >= pd.Timestamp(start_d))
        & (indicators_all["date"] <= pd.Timestamp(end_d))
    ]

# indicators_all -> every unit, date-filtered (drives the map + alerts)
# indicators     -> only the units picked above, date-filtered (drives Trends/Anomaly)
indicators = indicators_all[indicators_all["unit"].isin(selected_units)]

if indicators_all.empty:
    st.warning("No data in the selected date range.")
    st.stop()

# Status basis — applies to the status cards, alerts, the Spatial Map, and
# the cross-unit comparison chart. "Latest month in period" keeps the old
# behavior (snapshot at the most recent month within the date range above);
# "Period average" instead averages every indicator over the whole selected
# range; "Select a month" lets you pin the status to one specific month
# within the range, independent of what's latest.
status_mode = ph_status_mode.radio(
    "Status basis (cards / map / comparisons)",
    ["Latest month in period", "Period average", "Select a month"],
    help="Controls what 'status' means for the summary cards, the Spatial "
    "Map, and the cross-unit comparison chart: the most recent month in "
    "the selected date range, an average over that whole range, or one "
    "specific month you pick below.",
)

selected_month_ts = None
if status_mode == "Select a month":
    month_choices = sorted(indicators_all["date"].dropna().unique(), reverse=True)
    month_label_map = {pd.Timestamp(d).strftime("%B %Y"): d for d in month_choices}
    selected_month_label = ph_status_mode.selectbox(
        "Month to display",
        list(month_label_map.keys()),
        key="status_month_select",
    )
    selected_month_ts = month_label_map[selected_month_label]

# --------------------------------------------------------------------------
# Header + current status cards
# --------------------------------------------------------------------------

# st.title("Jordan Climate-Water Balance & Hydrological Stress Dashboard")

st.markdown(
    """
    <p style="font-size: 2.46rem; font-weight: 700; margin-bottom: 0.2rem;">
        Palestine Climate-Water Balance & Hydrological Stress Dashboard
    </p>
    <p style="font-size: 1.6rem; font-weight: 500; color: #08519c; margin-top: 0;">
        Uisng WaPOR data
    </p>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Spatial unit: **{spatial_unit}**. Indicators: Standardized Precipitation-Evapotranspiration Index (SPEI, simplified), "
    "Climatic Water Balance (CWB), Evaporative Stress Index (ESI) and Aridity Index (AI)"
)

# Status cards reflect the Trends selection (kept compact); alerts and the
# map below scan *all* units so nothing gets missed just because it wasn't
# picked for charting.
period_start, period_end = indicators_all["date"].min(), indicators_all["date"].max()
if status_mode == "Period average":
    snapshot_all = period_average_snapshot(indicators_all, location_col="unit")
    if period_start.strftime("%b %Y") == period_end.strftime("%b %Y"):
        status_period_label = f"average, {period_start.strftime('%B %Y')}"
    else:
        status_period_label = f"average, {period_start.strftime('%b %Y')} – {period_end.strftime('%b %Y')}"
elif status_mode == "Select a month":
    snapshot_all = indicators_all[indicators_all["date"] == selected_month_ts].copy()
    status_period_label = pd.Timestamp(selected_month_ts).strftime("%B %Y")
else:
    snapshot_all = latest_snapshot(indicators_all, location_col="unit")
    status_period_label = snapshot_all["date"].max().strftime("%B %Y")
st.subheader(f"Status — {status_period_label}")

if snapshot_all.empty:
    st.warning("No data available for the chosen status basis.")
    st.stop()

snapshot = (
    snapshot_all[snapshot_all["unit"].isin(selected_units)]
    if selected_units
    else snapshot_all.iloc[0:0]
)
if snapshot.empty:
    st.caption(
        "No units selected — pick some in the sidebar to see status cards, or check the map for all units."
    )
else:
    n_cols = min(len(snapshot), 6)
    cols = st.columns(max(n_cols, 1))
    for i, (_, row) in enumerate(snapshot.sort_values("unit").iterrows()):
        with cols[i % n_cols]:
            st.markdown(f"**{row['unit']}**")
            # st.markdown(status_badge(row["WAI_status"]), unsafe_allow_html=True)
            st.markdown(status_badge(row["SPEI_status"]), unsafe_allow_html=True)
            # st.metric("WAI", f"{row['WAI']:.2f}" if pd.notna(row["WAI"]) else "n/a")
            st.metric("SPEI", f"{row['SPEI']:.2f}" if pd.notna(row["SPEI"]) else "n/a")
            st.caption(f"CWB {row['CWB']:.0f} mm · Aridity: {row['Aridity_class']}")

# Alerts panel — scans ALL units, not just the ones selected for charting
alert_rows = snapshot_all[
    snapshot_all["SPEI_status"].isin(["Severely Dry", "Extremely Dry"])
]

# alert_rows = snapshot_all[
#     snapshot_all["SPEI_class"].isin(["Very Dry", "Extremely Dry"])
# ]

if not alert_rows.empty:
    names = alert_rows["unit"].tolist()
    st.error(
        f"⚠️ Water stress alert: **{', '.join(names)}** for the selected period classified as "
        f"{'/'.join(sorted(alert_rows['SPEI_status'].unique()))}."
    )

st.markdown("---")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tab_map, tab_trends, tab_anomaly, tab_data = st.tabs(
    ["🗺️ Spatial Map", "📈 Trends", "🔥 Anomaly Analysis", "📋 Data & Exports"]
)

# ---- Spatial Map tab ----
with tab_map:
    if not _HAS_GEO:
        st.info("Install `geopandas` and `shapely` to enable the spatial map view.")
    elif gdf is None or gdf.empty:
        st.info(
            f"No {unit_label} boundary available — place a .shp/.geojson/.zip in "
            f"`{boundary_dir}` (unit-name column `{boundary_id_col}`) to enable the map view."
        )
    else:
        map_df = snapshot_all.copy()
        if "date" in map_df.columns:
            map_df["date"] = map_df["date"].astype(str)
        merged = gdf.merge(map_df, on="unit", how="left")
        merged["CWB"] = merged["CWB"].astype(float)
        merged["SPEI"] = merged["SPEI"].astype(float)

        map_metric = st.selectbox(
            "Map metric", ["CWB (continuous)", "SPEI (standardized)", "Aridity class"]
        )
        merged_reset = merged.reset_index(drop=True)
        geojson = json.loads(merged_reset.to_json())

        bounds = merged_reset.total_bounds  # minx, miny, maxx, maxy
        center = {
            "lat": (bounds[1] + bounds[3]) / 2,
            "lon": (bounds[0] + bounds[2]) / 2,
        }

        if map_metric == "Aridity class":
            fig_map = px.choropleth_map(
                merged_reset,
                geojson=geojson,
                locations=merged_reset.index,
                color="Aridity_class",
                color_discrete_map=ARIDITY_COLORS,
                hover_name="unit",
                hover_data={"AI": ":.2f"},
                center=center,
                zoom=8,
                height=600,
                opacity=0.85,
                title=f"Aridity classification by {unit_label} — {status_period_label}",
            )
        elif map_metric == "SPEI (standardized)":
            fig_map = px.choropleth_map(
                merged_reset,
                geojson=geojson,
                locations=merged_reset.index,
                color="SPEI",
                color_continuous_scale=WATER_COLORSCALE,
                range_color=STD_INDEX_RANGE,
                hover_name="unit",
                hover_data={"SPEI": ":.2f", "SPEI_status": True},
                center=center,
                zoom=8,
                height=600,
                opacity=0.85,
                title=f"Standardized Precipitation-Evapotranspiration Index (SPEI) by {unit_label} — {status_period_label}",
            )
        else:
            cwb_bound = (
                float(np.nanmax(np.abs(merged_reset["CWB"])))
                if merged_reset["CWB"].notna().any()
                else 1.0
            )
            fig_map = px.choropleth_map(
                merged_reset,
                geojson=geojson,
                locations=merged_reset.index,
                color="CWB",
                color_continuous_scale=WATER_COLORSCALE,
                range_color=(-cwb_bound, cwb_bound),
                hover_name="unit",
                hover_data={"CWB": ":.0f", "Aridity_class": True},
                center=center,
                zoom=8,
                height=600,
                opacity=0.85,
                title=f"Climatic Water Balance (CWB, mm) by {unit_label} — {status_period_label}",
            )
        fig_map.update_layout(
            map_style="carto-positron", margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_map, use_container_width=True)

        no_geom = set(map_df["unit"]) - set(gdf["unit"])
        if no_geom:
            st.warning(
                "These units from the CSV data have no matching polygon in the boundary file "
                f"(check name spelling): {', '.join(sorted(no_geom))}"
            )

# ---- Trends tab ----
with tab_trends:
    fig_spei = px.line(
        indicators,
        x="date",
        y="SPEI",
        color="unit",
        title=f"Standardized Precipitation-Evapotranspiration Index (SPEI-{spei_scale}, simplified)",
    )
    fig_spei.add_hline(y=0, line_dash="dash", line_color="gray")

    zones = [
        (-3, -2, "#B2182B", "Extremely dry"),
        (-2, -1.5, "#EF8A62", "Very dry"),
        (-1.5, -1, "#FDCC8A", "Moderately dry"),
        (-1, 1, "#E0E0E0", "Normal"),
        (1, 1.5, "#B3CDE3", "Moderately wet"),
        (1.5, 2, "#67A9CF", "Very wet"),
        (2, 3, "#2166AC", "Extremely wet"),
    ]

    alpha = 0.35

    for y0, y1, color, label in zones:
        fig_spei.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=color,
            opacity=alpha,
            line_width=0,
        )

    fig_spei.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
    st.plotly_chart(fig_spei, use_container_width=True)

    fig_esi = px.line(
        indicators,
        x="date",
        y="ESI",
        color="unit",
        title="Evaporative Stress Index (higher = more stress)",
    )
    fig_esi.add_hline(y=0, line_dash="dash", line_color="gray")

    for y0, y1, color, label in zones:
        fig_esi.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=color,
            opacity=alpha,
            line_width=0,
        )
    fig_esi.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
    st.plotly_chart(fig_esi, use_container_width=True)

    fig_cwb = px.bar(
        indicators,
        x="date",
        y="CWB",
        color="unit",
        title="Climatic Water Balance (PCP − RET, mm)",
        barmode="group",
    )
    fig_cwb.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_cwb, use_container_width=True)

    fig_twsa = px.line(
        indicators,
        x="date",
        y="TWSA",
        color="unit",
        title="Terrestrial Water Storage Anomaly (standardized)",
    )
    fig_twsa.add_hline(y=0, line_dash="dash", line_color="gray")
    for y0, y1, color, label in zones:
        fig_twsa.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=color,
            opacity=alpha,
            line_width=0,
        )

    fig_twsa.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
    st.plotly_chart(fig_twsa, use_container_width=True)

    # fig_wai = px.line(
    #     indicators,
    #     x="date",
    #     y="WAI",
    #     color="unit",
    #     title="Composite Water Availability Index (WAI) over time",
    # )
    # fig_wai.add_hrect(y0=-1, y1=1, fillcolor="green", opacity=0.07, line_width=0)
    # fig_wai.add_hrect(y0=-2, y1=-1, fillcolor="orange", opacity=0.08, line_width=0)
    # fig_wai.add_hrect(
    #     y0=STD_INDEX_RANGE[0], y1=-2, fillcolor="red", opacity=0.08, line_width=0
    # )
    # fig_wai.update_layout(
    #     yaxis_title="WAI (standardized)", yaxis=dict(range=list(STD_INDEX_RANGE))
    # )
    # st.plotly_chart(fig_wai, use_container_width=True)

    # c1, c2 = st.columns(2)
    # with c1:
    #     # fig_cwb = px.bar(
    #     #     indicators,
    #     #     x="date",
    #     #     y="CWB",
    #     #     color="unit",
    #     #     title="Climatic Water Balance (PCP − RET, mm)",
    #     #     barmode="group",
    #     # )
    #     # fig_cwb.add_hline(y=0, line_dash="dash", line_color="gray")
    #     # st.plotly_chart(fig_cwb, use_container_width=True)

    #     fig_spi = px.line(
    #         indicators,
    #         x="date",
    #         y="SPI",
    #         color="unit",
    #         title=f"Standardized Precipitation Index (SPI-{spi_scale})",
    #     )
    #     fig_spi.add_hline(y=0, line_dash="dash", line_color="gray")
    #     fig_spi.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
    #     st.plotly_chart(fig_spi, use_container_width=True)

    #     fig_tws = px.line(
    #         indicators,
    #         x="date",
    #         y="TWS",
    #         color="unit",
    #         title="Terrestrial Water Storage (raw)",
    #     )
    #     st.plotly_chart(fig_tws, use_container_width=True)

    # with c2:
    #     fig_esi = px.line(
    #         indicators,
    #         x="date",
    #         y="ESI",
    #         color="unit",
    #         title="Evaporative Stress Index (higher = more stress)",
    #     )
    #     fig_esi.add_hline(y=0, line_dash="dash", line_color="gray")
    #     fig_esi.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
    #     st.plotly_chart(fig_esi, use_container_width=True)

    # fig_spei = px.line(
    #     indicators,
    #     x="date",
    #     y="SPEI",
    #     color="unit",
    #     title=f"Standardized Precipitation-Evapotranspiration Index (SPEI-{spei_scale}, simplified)",
    # )
    # fig_spei.add_hline(y=0, line_dash="dash", line_color="gray")
    # fig_spei.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
    # st.plotly_chart(fig_spei, use_container_width=True)

    # fig_twsa = px.line(
    #     indicators,
    #     x="date",
    #     y="TWSA",
    #     color="unit",
    #     title="Terrestrial Water Storage Anomaly (standardized)",
    # )
    # fig_twsa.add_hline(y=0, line_dash="dash", line_color="gray")
    # fig_twsa.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
    # st.plotly_chart(fig_twsa, use_container_width=True)

    if annual_df is not None and not annual_df.empty and "unit" in annual_df.columns:
        st.markdown("#### Annual totals")
        adf = annual_df[annual_df["unit"].isin(selected_units)].copy()
        metric = st.selectbox(
            "Annual metric",
            [c for c in ["PCP", "ET", "RET", "TWS"] if c in adf.columns],
        )
        fig_annual = px.bar(
            adf.sort_values("year"),
            x="year",
            y=metric,
            color="unit",
            barmode="group",
            title=f"Annual {metric} by {unit_label}",
        )
        st.plotly_chart(fig_annual, use_container_width=True)

# ---- Anomaly Analysis tab ----
with tab_anomaly:
    if not selected_units:
        st.info(
            f"Select at least one {unit_label} in the sidebar to see anomaly analysis."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            heat_metric = st.selectbox("Metric", ["SPEI", "TWSA", "ESI"], index=0)
        with c2:
            unit_for_heat = st.selectbox("Unit", selected_units)

        hdf = indicators[indicators["unit"] == unit_for_heat].copy()
        hdf["Year"] = hdf["date"].dt.year
        hdf["Month"] = hdf["date"].dt.strftime("%b")
        month_order = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        pivot = hdf.pivot_table(
            index="Year", columns="Month", values=heat_metric, aggfunc="mean"
        )
        pivot = pivot.reindex(columns=month_order)

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=WATER_COLORSCALE,
                zmid=0,
                zmin=STD_INDEX_RANGE[0],
                zmax=STD_INDEX_RANGE[1],
                colorbar=dict(title=heat_metric),
            )
        )
        fig_heat.update_layout(
            title=f"{heat_metric} by month/year — {unit_for_heat}", height=450
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "Teal/blue = wetter / higher availability, brown = drier / lower availability "
            "(relative to that calendar month's own history)."
        )

        st.markdown("#### Cross-unit comparison, latest month (all units)")
        comp_metric = st.selectbox(
            "Comparison metric",
            ["SPEI", "TWSA", "ESI", "CWB"],
            key="comp_metric",
        )
        fig_comp = px.bar(
            snapshot_all.sort_values(comp_metric),
            x="unit",
            y=comp_metric,
            color="SPEI_status",
            color_discrete_map=STATUS_COLORS,
            title=f"{comp_metric} by {unit_label} — {status_period_label}",
        )
        fig_comp.update_layout(xaxis_title=None)
        if comp_metric in ("SPEI", "TWSA", "ESI"):
            fig_comp.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
        st.plotly_chart(fig_comp, use_container_width=True)

# ---- Data & Exports tab ----
with tab_data:
    export_scope = st.radio(
        "Scope",
        [
            f"All {len(all_units)} {unit_label}s",
            f"Selected {len(selected_units)} (Trends tab)",
        ],
        horizontal=True,
    )
    data_scope_df = indicators_all if export_scope.startswith("All") else indicators

    st.markdown("**Computed monthly indicators**")
    display_cols = [
        "date",
        "unit",
        "PCP",
        "ET",
        "RET",
        "TWS",
        "CWB",
        "AI",
        "ESI",
        # "SPI",
        "SPEI",
        "TWSA",
        # "WAI",
        # "WAI_status",
        "Aridity_class",
    ]
    display_cols = [c for c in display_cols if c in data_scope_df.columns]
    st.dataframe(
        data_scope_df[display_cols].sort_values(["unit", "date"]),
        use_container_width=True,
        height=400,
    )

    csv_buf = io.StringIO()
    data_scope_df[display_cols].to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download monthly indicators as CSV",
        data=csv_buf.getvalue(),
        file_name="water_availability_indicators_monthly.csv",
        mime="text/csv",
    )

    if annual_df is not None and not annual_df.empty:
        st.markdown("**Annual data**")
        scope_units = all_units if export_scope.startswith("All") else selected_units
        adf = (
            annual_df[annual_df["unit"].isin(scope_units)]
            if "unit" in annual_df.columns
            else annual_df
        )
        st.dataframe(adf, use_container_width=True, height=300)
        csv_buf2 = io.StringIO()
        adf.to_csv(csv_buf2, index=False)
        st.download_button(
            "⬇️ Download annual data as CSV",
            data=csv_buf2.getvalue(),
            file_name="water_availability_annual.csv",
            mime="text/csv",
        )

    if _HAS_GEO and gdf is not None and not gdf.empty:
        snapshot_export = snapshot_all.copy()
        if "date" in snapshot_export.columns:
            snapshot_export["date"] = snapshot_export["date"].astype(str)
        merged_export = gdf.merge(snapshot_export, on="unit", how="left")
        geojson_bytes = merged_export.to_json().encode("utf-8")
        st.download_button(
            f"⬇️ Download {unit_label} boundaries + latest status (all units) as GeoJSON",
            data=geojson_bytes,
            file_name="units_with_latest_status.geojson",
            mime="application/geo+json",
        )

    with st.expander("Method notes"):
        st.markdown(
            f"""
- **SPEI** here is a SPEI from a climatic-water-balance series using the `spei`
  package (log-logistic/fisk fit per calendar month on the rolling CWB
  accumulation) — this is the literature-standard SPEI method
- **TWSA** is the standardized anomaly of terrestrial water storage per calendar month,
  analogous to GRACE/GRACE-FO TWS anomaly products.The TWSA is downloaded fro the area of
  interest from GLDAS.
- **ESI** is the standardized anomaly of the evaporative fraction (ET / RET), sign-flipped
  so that positive values indicate more evaporative stress (ET falling short of demand).
- All standardized indices are classified using the McKee et al. (1993) SPI category
  thresholds, and displayed on a fixed -3..3 axis/color range for cross-chart comparability.
- **Aridity Index** follows the UNEP classification (PCP / RET).
- Standardization is computed independently per spatial unit and per calendar month, so
  results are comparable across basins/admin units of different climatologies.
- All data is read automatically from `{data_root}/` (boundary/, monthly/, annual.csv) —
  see the sidebar for the expected layout. The Spatial Map colors by **CWB** (mm), with
  a symmetric range around zero so surplus and deficit are equally visible; the
  boundary file's unit-name column must be `{boundary_id_col}` for {spatial_unit}.
- The **Status basis** control in the sidebar applies to the status cards, alerts, the
  Spatial Map, and the cross-unit comparison chart: "Latest month in period" snapshots
  the most recent month within the selected date range (previous default behavior);
  "Period average" averages every indicator across the whole selected range per unit
  and reclassifies SPEI status / Aridity class from those averaged values; "Select a
  month" lets you pin the status (and the map) to one specific month within the range.
            """
        )
