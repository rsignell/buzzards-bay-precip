"""
Observed vs. modeled 24-hour rainfall over the Buzzards Bay / upper Cape Cod /
MA south-coast region for a single CoCoRaHS reporting day (7am ET -> 7am ET).

- Observed  : every CoCoRaHS station inside the map extent that filed a report
              for the day -- roster from region_cocorahs_stations.csv (built by
              find_region_stations.py). Every station is drawn the same way;
              the curated 16-station Buzzards Bay watershed set is still
              flagged in the csv and in the printed summary.
- Modeled   : NOAA MRMS CONUS analysis, hourly -- the radar/gauge-corrected
              ~1 km QPE (`precipitation_surface`, MultiSensor_QPE_01H Pass-2),
              from the dynamical.org public Icechunk/Zarr store on AWS.

The MRMS hourly rate (kg m-2 s-1 == mm/s) is accumulated over the 24 hourly
steps whose (t-1h, t] window falls inside the report day's 7am-7am US/Eastern
accumulation window, then converted to inches.

Outputs:
  buzzards_bay_mrms_map.html       interactive hvplot map (MRMS raster + the two
                                   MassDEP basin outlines + CoCoRaHS points)
  buzzards_bay_mrms_map.png        static matplotlib version of the same map
  buzzards_bay_mrms_scatter.png    gauge vs. MRMS scatter, all reporting stations
  buzzards_bay_mrms_daily.csv      per-station obs / MRMS / difference table

Run in the `protocoast-notebook` conda env.
"""

import os
from datetime import date
from zoneinfo import ZoneInfo

import geopandas as gpd
import holoviews as hv
import hvplot.pandas  # noqa: F401
import hvplot.xarray  # noqa: F401
import icechunk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

hv.extension("bokeh")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
REGION_STATIONS = "region_cocorahs_stations.csv"  # from find_region_stations.py

# CoCoRaHS report date == morning the 24-h total is read out. The report dated
# D covers (D-1) 7am local -> D 7am local. This is the only knob to turn to
# re-run for another day; everything below derives from it. Defaults to today;
# override with  REPORT_DATE=2026-09-04 python compare_buzzards_bay_mrms.py
REPORT_DATE = os.environ.get("REPORT_DATE") or date.today().isoformat()

# Both ends of the window are local 7am, converted to UTC -- so the offset
# follows EDT/EST instead of being pinned to one of them. MRMS `time` is naive
# UTC, so drop the tz after converting.
ET = ZoneInfo("America/New_York")


def _local_7am_to_utc(day):
    return (pd.Timestamp(f"{day:%Y-%m-%d} 07:00", tz=ET)
            .tz_convert("UTC").tz_localize(None))


_end_day = pd.Timestamp(REPORT_DATE)
WINDOW_END_UTC = _local_7am_to_utc(_end_day)
WINDOW_START_UTC = _local_7am_to_utc(_end_day - pd.Timedelta(days=1))
OBS_COL = f"precip_{pd.Timestamp(REPORT_DATE):%m%d}"  # column in REGION_STATIONS
WINDOW_LABEL = (f"{WINDOW_START_UTC:%b %-d} 7am - {WINDOW_END_UTC:%b %-d} "
                f"7am ET {WINDOW_END_UTC:%Y}")
SCATTER_LABEL = f"{WINDOW_START_UTC:%b %-d}-{WINDOW_END_UTC:%-d}, {WINDOW_END_UTC:%Y}"

WATERSHED = "buzzards_bay_watershed.geojson"      # MassDEP BUZZARDS BAY basin (BAS_ID 24)
CAPE_WATERSHED = "cape_cod_watershed.geojson"     # MassDEP CAPE COD basin (BAS_ID 22)
MRMS_PREFIX = "noaa-mrms-conus-analysis-hourly/v0.3.0.icechunk"
PAD = 0.12       # deg of margin around the Buzzards Bay basin
EAST_EDGE = -70.34  # push the map east far enough to show the Cape Cod
                    # basin's western divide (the Buzzards-Bay-facing edge of
                    # the Falmouth peninsula where the Falmouth stations sit)

# --------------------------------------------------------------------------- #
# 1. CoCoRaHS observations for the report day (all stations in the map extent)
# --------------------------------------------------------------------------- #
roster = pd.read_csv(REGION_STATIONS)
obs = (
    roster[roster[OBS_COL].notna()]
    .rename(columns={OBS_COL: "obs_in"})
    [["stationNumber", "stationName", "latitude", "longitude",
      "obs_in", "in_curated_16"]]
    .reset_index(drop=True)
)
print(f"{len(roster)} CoCoRaHS stations in the map extent; "
      f"{len(obs)} filed a {REPORT_DATE} report "
      f"({obs['in_curated_16'].sum()} of them in the curated 16)")

# --------------------------------------------------------------------------- #
# 2. MRMS radar/gauge-corrected 1 km QPE, accumulated over the 7am-7am window
# --------------------------------------------------------------------------- #
shed = gpd.read_file(WATERSHED).to_crs(4326)
cape = gpd.read_file(CAPE_WATERSHED).to_crs(4326)
minx, miny, maxx, maxy = shed.total_bounds
lon0, lon1 = minx - PAD, max(maxx + PAD, EAST_EDGE)
lat0, lat1 = miny - PAD, maxy + PAD

print("Opening MRMS dynamical.org Icechunk store ...")
storage = icechunk.s3_storage(
    bucket="dynamical-noaa-mrms", prefix=MRMS_PREFIX,
    region="us-west-2", anonymous=True,
)
repo = icechunk.Repository.open(storage)
ds = xr.open_zarr(repo.readonly_session("main").store, chunks=None,
                  decode_timedelta=True)

# hourly stamps: t labels the (t-1h, t] accumulation -> first stamp is
# WINDOW_START+1h, last stamp is WINDOW_END.
stamps = pd.date_range(WINDOW_START_UTC + pd.Timedelta(hours=1),
                       WINDOW_END_UTC, freq="1h")
# 24 on a normal day; 23 or 25 on the two DST-transition mornings.
assert 23 <= len(stamps) <= 25, len(stamps)
n_steps = len(stamps)

# When re-run on the report morning itself, the last MRMS hours may not be
# published yet -- take whatever of the 24 steps exist.
avail = pd.DatetimeIndex(ds.time.values)
have = stamps[stamps.isin(avail)]
if len(have) < n_steps:
    print(f"  NOTE: only {len(have)}/{n_steps} hourly MRMS steps published so "
          f"far (latest {avail.max()}); accumulating those.")

sub = ds["precipitation_surface"].sel(
    time=have,
    latitude=slice(lat1, lat0),   # MRMS latitude is descending
    longitude=slice(lon0, lon1),
)
print(f"  MRMS subset {dict(sub.sizes)}; loading ...")
sub = sub.load()

# mm/s -> mm per hourly step -> inches; sum the available steps.
mrms_in = (sub * 3600.0 / 25.4).sum("time", keep_attrs=False)
mrms_in.name = "mrms_in"
mrms_in.attrs = {"long_name": "MRMS 24-h precipitation", "units": "inch"}
n_missing = int(sub.isnull().any(("latitude", "longitude")).sum())
if n_missing:
    print(f"  WARNING: {n_missing}/{len(have)} MRMS hours had missing data in the box")

# sample MRMS at each reporting station (nearest 1 km cell)
si = obs.reset_index(drop=True)
samp = mrms_in.sel(
    latitude=xr.DataArray(si["latitude"], dims="s"),
    longitude=xr.DataArray(si["longitude"], dims="s"),
    method="nearest",
)
obs["mrms_in"] = np.round(samp.values, 3)
obs["diff_in"] = np.round(obs["mrms_in"] - obs["obs_in"], 3)

obs = obs.sort_values("stationNumber").reset_index(drop=True)
obs.to_csv("buzzards_bay_mrms_daily.csv", index=False)


def stats(g):
    return g["obs_in"].corr(g["mrms_in"]), g["diff_in"].mean()


r, bias = stats(obs)
r16, bias16 = stats(obs[obs["in_curated_16"]])
print(f"\n=== 24-h rainfall, {WINDOW_LABEL} ===")
print(obs[["stationNumber", "stationName", "obs_in", "mrms_in", "diff_in",
           "in_curated_16"]].to_string(index=False))
print(f"\nAll {len(obs)} reporting stations in extent : r={r:.3f}  "
      f"MRMS-gauge bias={bias:+.3f} in  "
      f"(gauge mean {obs['obs_in'].mean():.2f}, MRMS mean {obs['mrms_in'].mean():.2f})")
print(f"Curated 16 watershed subset (n={obs['in_curated_16'].sum()})   : "
      f"r={r16:.3f}  MRMS-gauge bias={bias16:+.3f} in")
print(f"MRMS grid over box: min={float(mrms_in.min()):.2f}  max={float(mrms_in.max()):.2f} in")

# --------------------------------------------------------------------------- #
# 3. Interactive hvplot map
# --------------------------------------------------------------------------- #
cmax = float(np.ceil(max(mrms_in.max(), obs["obs_in"].max(), obs["mrms_in"].max()) * 10) / 10)
CMAP = "YlGnBu"
BB_COLOR, CC_COLOR = "#444444", "#c1440e"
GAUGE_COLOR = "#4c78a8"   # scatter marker fill
title = (f"24-h rainfall  {WINDOW_LABEL}   ({len(obs)} CoCoRaHS stations)\n"
         "MRMS radar/gauge 1 km QPE (shaded) vs CoCoRaHS gauges (circles)\n"
         "MassDEP basins: Buzzards Bay (gray solid), Cape Cod (orange dashed)")

raster = mrms_in.hvplot.quadmesh(
    x="longitude", y="latitude", geo=True, rasterize=False,
    cmap=CMAP, clim=(0, cmax), tiles="OSM",
    frame_width=680, frame_height=620, alpha=0.80, line_alpha=0,
    clabel="24-h precipitation (inch)", title=title,
    xlim=(lon0, lon1), ylim=(lat0, lat1),
)

boundary = gpd.GeoDataFrame(geometry=[shed.union_all()], crs=4326).hvplot(
    geo=True, fill_alpha=0, line_color=BB_COLOR, line_width=2,
)
cape_boundary = gpd.GeoDataFrame(geometry=[cape.union_all()], crs=4326).hvplot(
    geo=True, fill_alpha=0, line_color=CC_COLOR, line_width=2, line_dash="dashed",
)

HOVER = ["stationNumber", "stationName", "obs_in", "mrms_in", "diff_in"]
pts = obs.hvplot.points(
    x="longitude", y="latitude", geo=True, c="obs_in", cmap=CMAP,
    clim=(0, cmax), s=200, marker="o", line_color="black", line_width=0.9,
    hover_cols=HOVER, colorbar=False,
)

hvplot_map = (raster * boundary * cape_boundary * pts).opts(
    active_tools=["wheel_zoom"]
)
hv.save(hvplot_map, "buzzards_bay_mrms_map.html")
print("\nSaved buzzards_bay_mrms_map.html")

# --------------------------------------------------------------------------- #
# 4. Static PNG versions
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9, 8.4))
pm = ax.pcolormesh(mrms_in["longitude"], mrms_in["latitude"], mrms_in.values,
                   cmap=CMAP, vmin=0, vmax=cmax, shading="nearest")
shed.boundary.plot(ax=ax, color=BB_COLOR, linewidth=1.2,
                   label="Buzzards Bay basin (MassDEP)")
cape.boundary.plot(ax=ax, color=CC_COLOR, linewidth=1.2, linestyle="--",
                   label="Cape Cod basin (MassDEP)")
ax.scatter(obs["longitude"], obs["latitude"], c=obs["obs_in"], cmap=CMAP,
           vmin=0, vmax=cmax, s=100, marker="o", edgecolors="black",
           linewidths=0.9, zorder=5)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
for _, row in obs.iterrows():
    ax.annotate(f"{row['obs_in']:.2f}", (row["longitude"], row["latitude"]),
                xytext=(5, 3), textcoords="offset points", fontsize=6.5,
                zorder=7)
ax.set_xlim(lon0, lon1)
ax.set_ylim(lat0, lat1)
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_title(title, fontsize=9.5)
fig.colorbar(pm, ax=ax, label="24-h precipitation (inch)", shrink=0.85)
fig.tight_layout()
fig.savefig("buzzards_bay_mrms_map.png", dpi=140)
print("Saved buzzards_bay_mrms_map.png")

fig2, ax2 = plt.subplots(figsize=(5.8, 5.8))
lim = cmax
ax2.plot([0, lim], [0, lim], "k--", lw=1)
ax2.scatter(obs["obs_in"], obs["mrms_in"], s=45, marker="o", c=GAUGE_COLOR,
            edgecolors="white", linewidths=0.5,
            label=f"CoCoRaHS stations (n={len(obs)})")
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.set_xlabel("CoCoRaHS gauge (inch)")
ax2.set_ylabel("MRMS radar/gauge QPE (inch)")
ax2.legend(loc="lower right", fontsize=8)
ax2.set_title(f"24-h rainfall  {SCATTER_LABEL}\n"
              f"all {len(obs)} stations: r={r:.2f}, bias={bias:+.2f} in",
              fontsize=10)
fig2.tight_layout()
fig2.savefig("buzzards_bay_mrms_scatter.png", dpi=140)
print("Saved buzzards_bay_mrms_scatter.png")
