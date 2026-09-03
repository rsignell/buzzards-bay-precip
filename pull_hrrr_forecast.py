# Pulls HRRR forecast-48-hour precip (dynamical.org public zarr) for the 16
# Buzzards Bay CoCoRaHS stations and computes Day-1 (lead 1-24h) / Day-2
# (lead 25-48h) accumulated totals per CoCoRaHS 7am-7am US/Eastern reporting
# day, using a fixed 12Z UTC daily reference cycle (closest available HRRR
# extended cycle to the 7am ET/EDT boundary -- off by 0-1h depending on DST,
# an accepted approximation, not engineered around).
#
# Run in the protocoast-notebook conda env. Needs: xarray, zarr, rioxarray,
# pyproj, rolodex, pandas, numpy. No SkyPilot VM required -- the HRRR
# forecast-48-hour store is public/anonymous and fast to read directly.
#
# Expects buzzards_bay_cocorahs_daily.csv in cwd (for station lat/lon; run
# pull_cocorahs.py first).
import time

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  registers the .rio accessor used below
import rolodex.forecast
from pyproj import Transformer

STATIONS = [
    "MA-BR-14", "MA-BR-18", "MA-BR-52", "MA-BR-79", "MA-PL-63", "MA-PL-66",
    "MA-BA-115", "MA-BA-101", "MA-BA-109", "MA-BA-112", "MA-BA-105",
    "MA-BA-113", "MA-BA-57", "MA-BA-2", "MA-BA-87", "MA-BA-13",
]

# Reporting-day range to produce. Set to a short window first to dry-run
# against the MA-BA-13 / 2025-05-23 spot check, then widen to the full range.
START, END = "2025-01-01", "2026-08-03"
OUT_CSV = "buzzards_bay_hrrr_daily.csv"

print("Loading station coordinates from buzzards_bay_cocorahs_daily.csv ...")
cocorahs = pd.read_csv("buzzards_bay_cocorahs_daily.csv")
stations = cocorahs[["stationNumber", "latitude", "longitude"]].drop_duplicates()
stations = stations.set_index("stationNumber").loc[STATIONS].reset_index()
assert len(stations) == len(STATIONS)

print("Opening HRRR forecast-48-hour zarr ...")
ds = xr.open_zarr(
    "https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr?email=rsignell@gmail.com",
    chunks=None,
)
precip = ds["precipitation_surface"]

transformer = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True)
xs, ys = transformer.transform(stations["longitude"].values, stations["latitude"].values)
x_da = xr.DataArray(xs, dims="station")
y_da = xr.DataArray(ys, dims="station")

# 3-day lookback before START: the first reporting day's Day-2 bucket (lead
# 25-48h) is sourced from the run initialized 2 calendar days earlier, plus
# 1 day of margin for the DST edge effect at the boundary of the pull range.
init_dates = pd.date_range(pd.Timestamp(START) - pd.Timedelta(days=3), pd.Timestamp(END), freq="D")
init_times = init_dates + pd.Timedelta(hours=12)

print(f"Selecting {len(init_times)} init_times (12Z daily) x 49 lead steps x {len(stations)} stations ...")
t0 = time.time()
sel = precip.sel(init_time=init_times, x=x_da, y=y_da, method="nearest")
sel = sel.assign_coords(station=stations["stationNumber"].values)
sel = sel.load()
print(f"Loaded {sel.sizes} in {time.time() - t0:.1f}s")

valid_time = rolodex.forecast.create_lazy_valid_time_variable(
    reference_time=sel.init_time, period=sel.lead_time
)
sel = sel.assign_coords(valid_time=(("init_time", "lead_time"), valid_time.data))

# rate (kg m-2 s-1 == mm/s) * 3600 s/step -> mm per step -> inches per step
precip_in_step = sel * 3600.0 / 25.4

df = precip_in_step.to_dataframe(name="precip_in_step").reset_index()
df["lead_hours"] = df["lead_time"] / pd.Timedelta(hours=1)
df = df[df["lead_hours"] > 0]  # drop lead_time=0 (NaN, no previous step)

df["lead_day"] = np.select(
    [df["lead_hours"] <= 24, df["lead_hours"] <= 48], [1, 2], default=np.nan,
)
df = df.dropna(subset=["lead_day"]).copy()
df["lead_day"] = df["lead_day"].astype(int)

# Bucket each hourly valid_time into its CoCoRaHS 7am-7am US/Eastern
# reporting day -- identical DST-aware logic to compare_buzzards_bay_v2.py.
df["time_utc"] = df["valid_time"].dt.tz_localize("UTC")
df["time_local"] = df["time_utc"].dt.tz_convert("US/Eastern")
df["date"] = (df["time_local"] - pd.Timedelta(hours=7)).dt.floor("D").dt.date + pd.Timedelta(days=1)
df["date"] = pd.to_datetime(df["date"])

grouped = df.groupby(["station", "date", "lead_day"]).agg(
    precip_in=("precip_in_step", "sum"), n_hours=("precip_in_step", "count"),
).reset_index()
# Restrict diagnostics/output to the requested range -- dates just outside
# [START, END] are naturally partial (insufficient init lookback/lookahead
# at the edges of init_times) and would otherwise look like spurious gaps.
grouped = grouped[(grouped["date"] >= START) & (grouped["date"] <= END)]

# n_hours is 23 (not 24) for roughly half the year (EDT, UTC-4) because the
# fixed 12Z reference cycle is 1h off from the 11:00 UTC EDT boundary -- this
# is the accepted DST edge effect, not a data gap. Only flag genuine gaps.
real_gaps = grouped[grouped["n_hours"] < 23]
if len(real_gaps):
    print(f"WARNING: {len(real_gaps)} (station, date, lead_day) groups have <23 hourly values (real gap, not DST edge effect)")
print(f"n_hours distribution:\n{grouped['n_hours'].value_counts().sort_index()}")

pivoted = grouped.pivot_table(index=["station", "date"], columns="lead_day", values="precip_in").reset_index()
pivoted = pivoted.rename(columns={"station": "stationNumber", 1: "hrrr_day1_in", 2: "hrrr_day2_in"})

pivoted.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}: {len(pivoted)} rows")
print(pivoted.describe())

spot = pivoted[(pivoted["stationNumber"] == "MA-BA-13") & (pivoted["date"] == "2025-05-23")]
if len(spot):
    print("\nSpot check MA-BA-13 2025-05-23 (expect Day-1~2.10in, Day-2~2.55in):")
    print(spot)
