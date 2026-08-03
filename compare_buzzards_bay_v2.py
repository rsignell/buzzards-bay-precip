# Same as compare_buzzards_bay.py, but fixes a timing mismatch: CoCoRaHS
# observers report a 24h total each morning at ~7am *local* time (the sum
# since the previous day's ~7am reading), not a midnight-to-midnight
# calendar day. AORC here was previously bucketed into naive UTC calendar
# days, which misaligns any storm that straddles the cutoff. This version
# pulls AORC at native hourly resolution and re-buckets it into 7am-to-7am
# US/Eastern windows (DST-aware) before comparing.
import xarray as xr
import s3fs
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATIONS = [
    "MA-BR-14", "MA-BR-18", "MA-BR-52", "MA-BR-79", "MA-PL-63", "MA-PL-66",
    "MA-BA-115", "MA-BA-101", "MA-BA-109", "MA-BA-112", "MA-BA-105",
    "MA-BA-113", "MA-BA-57", "MA-BA-2", "MA-BA-87", "MA-BA-13",
]

cocorahs = pd.read_csv("buzzards_bay_cocorahs_daily.csv", parse_dates=["date"])
cocorahs = cocorahs[cocorahs["date"] <= "2025-12-31"].copy()
cocorahs = cocorahs[cocorahs["stationNumber"].isin(STATIONS)]

stations = cocorahs[["stationNumber", "stationName", "latitude", "longitude"]].drop_duplicates()
stations = stations.reset_index(drop=True)
print(f"{len(stations)} stations, {len(cocorahs)} gauge obs")

print("Opening AORC 2025 (hourly) ...")
fs = s3fs.S3FileSystem(anon=True)
store = fs.get_mapper("noaa-nws-aorc-v1-1-1km/2025.zarr")
aorc_ds = xr.open_zarr(store, consolidated=True)

lat_da = xr.DataArray(stations["latitude"].values, dims="station")
lon_da = xr.DataArray(stations["longitude"].values, dims="station")
aorc_pts = aorc_ds["APCP_surface"].sel(latitude=lat_da, longitude=lon_da, method="nearest")
aorc_pts = aorc_pts.assign_coords(station=stations["stationNumber"].values)
print("Loading hourly AORC point extractions ...")
aorc_pts = (aorc_pts / 25.4).load()  # kg/m^2 == mm -> inches, still hourly

# AORC time is UTC (tz-naive). Localize, convert to US/Eastern (DST-aware),
# then bucket each hourly value into the 7am-to-7am "CoCoRaHS day" it falls
# into: a timestamp t belongs to CoCoRaHS-date D if D-1 07:00 <= t < D 07:00
# local, i.e. floor(t_local - 7h) + 1 day == D.
df = aorc_pts.to_dataframe(name="aorc_in").reset_index()
df = df.rename(columns={"station": "stationNumber"})
df["time_utc"] = df["time"].dt.tz_localize("UTC")
df["time_local"] = df["time_utc"].dt.tz_convert("US/Eastern")
df["cocorahs_date"] = (df["time_local"] - pd.Timedelta(hours=7)).dt.floor("D").dt.date + pd.Timedelta(days=1)
df["cocorahs_date"] = pd.to_datetime(df["cocorahs_date"])

aorc_daily = df.groupby(["stationNumber", "cocorahs_date"])["aorc_in"].sum().reset_index()
aorc_daily = aorc_daily.rename(columns={"cocorahs_date": "date"})

merged = cocorahs.merge(aorc_daily, on=["stationNumber", "date"], how="inner")
merged.to_csv("buzzards_bay_comparison_daily_v2.csv", index=False)
print(f"Merged {len(merged)} matched gauge/AORC day-station records (7am-7am aligned)")

overall_r_v2 = merged["precip_in"].corr(merged["aorc_in"])
print(f"Overall pooled daily correlation (7am-aligned): r={overall_r_v2:.3f}")

nz = merged[merged["precip_in"] > 0]
print(f"Overall pooled correlation, gauge>0 only: r={nz['precip_in'].corr(nz['aorc_in']):.3f} (n={len(nz)})")

# ---- per-station scatter grid (gauge>0 only, matching prior plot) ----
ncols = 4
nrows = int(np.ceil(len(STATIONS) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = axes.flatten()

for ax, station in zip(axes, STATIONS):
    g = nz[nz["stationNumber"] == station]
    name = g["stationName"].iloc[0] if len(g) else station
    r = g["precip_in"].corr(g["aorc_in"]) if len(g) > 1 else np.nan
    ax.scatter(g["precip_in"], g["aorc_in"], s=10, alpha=0.5)
    lim = max(g["precip_in"].max(), g["aorc_in"].max(), 0.1) * 1.05 if len(g) else 1
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(f"{station}\n{name}\nr={r:.2f}, n={len(g)}", fontsize=9)
    ax.set_xlabel("Gauge (in)", fontsize=8)
    ax.set_ylabel("AORC (in)", fontsize=8)
    ax.tick_params(labelsize=7)

for ax in axes[len(STATIONS):]:
    ax.axis("off")

fig.suptitle("Buzzards Bay: CoCoRaHS vs AORC, 2025, 7am-7am ET aligned (gauge>0 days)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("buzzards_bay_per_station_scatter_v2.png", dpi=130)
print("Saved buzzards_bay_per_station_scatter_v2.png and buzzards_bay_comparison_daily_v2.csv")
