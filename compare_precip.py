# Compares NASA IMERG Analysis-Late (global, 0.1deg) vs NOAA AORC (CONUS,
# ~1km, bias-corrected radar/gauge) daily precip at a New Bedford, MA point,
# plus the CSO event gauge rainfall_in column, for 2025-01-01 to 2026-07-31.
#
# Run on a SkyPilot VM (see ../repos/ESIP-2026-virtual-agent/notebook.sky.yaml)
# in the esip2026 micromamba env, in AWS us-east-1 (both datasets live on S3
# there). Needs: dynamical-catalog>=0.7.0, xarray, s3fs (upgraded past the
# conda-pinned 0.6.0, which is incompatible with the env's aiobotocore),
# pandas, matplotlib. Expects new_bedford_cso_discharges_verified.csv in cwd.

import dynamical_catalog
import xarray as xr
import s3fs
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAT, LON = 41.6362, -70.9342  # New Bedford, MA
START, END = "2025-01-01", "2026-07-31"

# ---------------------------------------------------------------- IMERG ----
print("Opening IMERG analysis-late ...")
imerg_ds = dynamical_catalog.open("nasa-imerg-analysis-late", chunks=None)
imerg_pt = imerg_ds["precipitation_surface"].sel(
    latitude=LAT, longitude=LON, method="nearest"
).sel(time=slice(START, END))
print("IMERG units:", imerg_ds["precipitation_surface"].attrs.get("units"))
imerg_pt = imerg_pt.load()

# kg m-2 s-1 over a 30-min step -> mm per step -> inches per step
imerg_mm = imerg_pt * 1800.0
imerg_in = imerg_mm / 25.4
imerg_daily = imerg_in.resample(time="1D").sum().to_series()
imerg_daily.name = "imerg_in"

# ------------------------------------------------------------------ AORC ---
print("Opening AORC 2025 ...")
fs = s3fs.S3FileSystem(anon=True)
store = fs.get_mapper("noaa-nws-aorc-v1-1-1km/2025.zarr")
aorc_ds = xr.open_zarr(store, consolidated=True)
print("AORC APCP units:", aorc_ds["APCP_surface"].attrs.get("units"))
aorc_pt = aorc_ds["APCP_surface"].sel(latitude=LAT, longitude=LON, method="nearest")
aorc_pt = aorc_pt.load()
print("AORC nearest grid point:", aorc_pt.latitude.item(), aorc_pt.longitude.item())

aorc_units = aorc_ds["APCP_surface"].attrs.get("units", "").lower()
if "kg" in aorc_units or "mm" in aorc_units:
    aorc_in = aorc_pt / 25.4  # kg/m^2 is numerically equal to mm
else:
    aorc_in = aorc_pt  # already inches
aorc_daily = aorc_in.resample(time="1D").sum().to_series()
aorc_daily.name = "aorc_in"
# AORC 2025.zarr only covers 2025 -- no 2026 data yet, leave gap for 2026

# -------------------------------------------------------------- Gauge CSV --
print("Loading gauge CSV ...")
df = pd.read_csv("new_bedford_cso_discharges_verified.csv", parse_dates=["discharge_date"])
df = df[df["rainfall_in"] <= 20].copy()  # known bad-row filter, see project memory
df = df[(df["discharge_date"] >= START) & (df["discharge_date"] <= END)]
gauge_daily = df.groupby("discharge_date")["rainfall_in"].max()
gauge_daily.index.name = "time"
gauge_daily.name = "gauge_in"

# ------------------------------------------------------------------ merge --
combined = pd.concat([imerg_daily, aorc_daily, gauge_daily], axis=1)
combined = combined.sort_index()
combined.to_csv("precip_comparison_daily.csv")
print(combined.describe())

overlap = combined.dropna(subset=["imerg_in", "aorc_in"])
corr = overlap["imerg_in"].corr(overlap["aorc_in"])
print(f"\nIMERG vs AORC daily correlation (2025 overlap, n={len(overlap)}): r={corr:.3f}")

gauge_overlap_imerg = combined.dropna(subset=["imerg_in", "gauge_in"])
gauge_overlap_aorc = combined.dropna(subset=["aorc_in", "gauge_in"])
print(f"IMERG vs gauge on event days (n={len(gauge_overlap_imerg)}): "
      f"r={gauge_overlap_imerg['imerg_in'].corr(gauge_overlap_imerg['gauge_in']):.3f}")
print(f"AORC vs gauge on event days (n={len(gauge_overlap_aorc)}): "
      f"r={gauge_overlap_aorc['aorc_in'].corr(gauge_overlap_aorc['gauge_in']):.3f}")

# ------------------------------------------------------------------- plot --
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)

ax = axes[0]
ax.plot(combined.index, combined["imerg_in"], label="IMERG Analysis-Late (0.1deg)", lw=0.8)
ax.plot(combined.index, combined["aorc_in"], label="AORC (~1km, bias-corrected)", lw=0.8)
ax.scatter(combined.index, combined["gauge_in"], color="k", s=12, zorder=5,
           label="Gauge (CSO event rainfall_in)")
ax.set_ylabel("Daily precip (in)")
ax.set_title(f"New Bedford, MA ({LAT}, {LON}) daily precipitation, {START} to {END}")
ax.legend(fontsize=8)

ax2 = axes[1]
ax2.scatter(overlap["aorc_in"], overlap["imerg_in"], s=10, alpha=0.6)
lims = [0, max(overlap["aorc_in"].max(), overlap["imerg_in"].max()) * 1.05]
ax2.plot(lims, lims, "k--", lw=1)
ax2.set_xlabel("AORC daily (in)")
ax2.set_ylabel("IMERG daily (in)")
ax2.set_title(f"IMERG vs AORC, 2025 overlap (r={corr:.2f})")

fig.tight_layout()
fig.savefig("precip_comparison.png", dpi=130)
print("\nSaved precip_comparison.png and precip_comparison_daily.csv")
