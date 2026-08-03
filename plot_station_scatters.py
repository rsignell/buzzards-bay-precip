# Per-station scatter plots (CoCoRaHS gauge vs NOAA AORC daily precip, 2025)
# for the Buzzards Bay watershed station set: the 6 stations inside MassDEP's
# official "BUZZARDS BAY" major basin, plus the Cape Cod peninsula stations
# that geographically face Buzzards Bay (North/West Falmouth, Pocasset) even
# though MassDEP's basin layer books all of Cape Cod under a separate
# "CAPE COD" basin. Excludes the Vineyard-Sound-facing East Falmouth stations.
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATIONS = [
    "MA-BR-14", "MA-BR-18", "MA-BR-52", "MA-BR-79", "MA-PL-63", "MA-PL-66",  # official basin
    "MA-BA-115", "MA-BA-101", "MA-BA-109", "MA-BA-112", "MA-BA-105",
    "MA-BA-113", "MA-BA-57", "MA-BA-2", "MA-BA-87", "MA-BA-13",             # Buzzards-facing Cape Cod
]

df = pd.read_csv("buzzards_bay_comparison_daily.csv")
df = df[df["stationNumber"].isin(STATIONS)]
df = df[df["precip_in"] > 0]  # drop CoCoRaHS zero-precip days

ncols = 4
nrows = int(np.ceil(len(STATIONS) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = axes.flatten()

for ax, station in zip(axes, STATIONS):
    g = df[df["stationNumber"] == station]
    name = g["stationName"].iloc[0] if len(g) else station
    r = g["precip_in"].corr(g["aorc_in"]) if len(g) > 1 else np.nan
    ax.scatter(g["precip_in"], g["aorc_in"], s=10, alpha=0.5)
    lim = max(g["precip_in"].max(), g["aorc_in"].max(), 0.1) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(f"{station}\n{name}\nr={r:.2f}, n={len(g)}", fontsize=9)
    ax.set_xlabel("Gauge (in)", fontsize=8)
    ax.set_ylabel("AORC (in)", fontsize=8)
    ax.tick_params(labelsize=7)

for ax in axes[len(STATIONS):]:
    ax.axis("off")

fig.suptitle("Buzzards Bay watershed: CoCoRaHS vs AORC daily precip, 2025 (per station)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("buzzards_bay_per_station_scatter.png", dpi=130)
print("Saved buzzards_bay_per_station_scatter.png")
