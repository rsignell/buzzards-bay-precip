# Compares CoCoRaHS gauge daily precip against NOAA AORC (~1km, bias-
# corrected radar/gauge) at each CoCoRaHS station location across the
# Buzzards Bay watershed (MA south coast), for the 2025 overlap period
# (AORC has no 2026 data yet).
import xarray as xr
import s3fs
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cocorahs = pd.read_csv("buzzards_bay_cocorahs_daily.csv", parse_dates=["date"])
cocorahs = cocorahs[cocorahs["date"] <= "2025-12-31"].copy()  # AORC-overlap window only

stations = cocorahs[["stationNumber", "stationName", "latitude", "longitude"]].drop_duplicates()
stations = stations.reset_index(drop=True)
print(f"{len(stations)} stations, {len(cocorahs)} gauge obs in 2025 overlap window")

print("Opening AORC 2025 ...")
fs = s3fs.S3FileSystem(anon=True)
store = fs.get_mapper("noaa-nws-aorc-v1-1-1km/2025.zarr")
aorc_ds = xr.open_zarr(store, consolidated=True)

lat_da = xr.DataArray(stations["latitude"].values, dims="station")
lon_da = xr.DataArray(stations["longitude"].values, dims="station")
aorc_pts = aorc_ds["APCP_surface"].sel(latitude=lat_da, longitude=lon_da, method="nearest")
aorc_pts = aorc_pts.assign_coords(station=stations["stationNumber"].values)
print("Loading AORC point extractions (this pulls the needed zarr chunks)...")
aorc_pts = aorc_pts.load()

aorc_daily = (aorc_pts / 25.4).resample(time="1D").sum()  # kg/m^2 == mm -> inches
aorc_df = aorc_daily.to_dataframe(name="aorc_in").reset_index()[["station", "time", "aorc_in"]]
aorc_df = aorc_df.rename(columns={"station": "stationNumber", "time": "date"})

merged = cocorahs.merge(aorc_df, on=["stationNumber", "date"], how="inner")
merged.to_csv("buzzards_bay_comparison_daily.csv", index=False)
print(f"Merged {len(merged)} matched gauge/AORC day-station records")

# ---- per-station summary ----
summary = merged.groupby("stationNumber").apply(
    lambda g: pd.Series({
        "stationName": g["stationName"].iloc[0],
        "latitude": g["latitude"].iloc[0],
        "longitude": g["longitude"].iloc[0],
        "n_days": len(g),
        "corr": g["precip_in"].corr(g["aorc_in"]),
        "gauge_total_in": g["precip_in"].sum(),
        "aorc_total_in": g["aorc_in"].sum(),
        "bias_in": g["aorc_in"].sum() - g["precip_in"].sum(),
    }),
    include_groups=False,
).reset_index()
summary["pct_bias"] = 100 * summary["bias_in"] / summary["gauge_total_in"]
summary = summary.sort_values("corr", ascending=False)
summary.to_csv("buzzards_bay_comparison_summary.csv", index=False)
print(summary.to_string(index=False))

overall_r = merged["precip_in"].corr(merged["aorc_in"])
print(f"\nOverall pooled daily correlation (all stations, all days): r={overall_r:.3f}")

# ---- plots ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc = axes[0].scatter(summary["longitude"], summary["latitude"], c=summary["corr"],
                      cmap="RdYlGn", vmin=0, vmax=1, s=90, edgecolor="k")
for _, row in summary.iterrows():
    axes[0].annotate(row["stationNumber"].replace("MA-", ""),
                      (row["longitude"], row["latitude"]), fontsize=6,
                      xytext=(3, 3), textcoords="offset points")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Per-station daily corr: CoCoRaHS vs AORC (2025)")
axes[0].set_aspect("equal")
plt.colorbar(sc, ax=axes[0], label="r")

axes[1].scatter(merged["precip_in"], merged["aorc_in"], s=6, alpha=0.4)
lims = [0, max(merged["precip_in"].max(), merged["aorc_in"].max()) * 1.05]
axes[1].plot(lims, lims, "k--", lw=1)
axes[1].set_xlabel("CoCoRaHS gauge daily (in)")
axes[1].set_ylabel("AORC daily (in)")
axes[1].set_title(f"All stations pooled, 2025 (r={overall_r:.2f}, n={len(merged)})")

fig.suptitle("Buzzards Bay watershed: CoCoRaHS vs NOAA AORC daily precip, 2025")
fig.tight_layout()
fig.savefig("buzzards_bay_comparison.png", dpi=130)
print("\nSaved buzzards_bay_comparison.png, buzzards_bay_comparison_summary.csv, buzzards_bay_comparison_daily.csv")
