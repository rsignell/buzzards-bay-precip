# Compares HRRR forecast precip (Day-1 = ~1-day lead, Day-2 = ~2-day lead;
# see pull_hrrr_forecast.py) against CoCoRaHS gauge totals for the Buzzards
# Bay watershed, on non-zero gauge days. Mirrors the merge/filter/plot
# pattern of compare_buzzards_bay_v2.py, one lead at a time.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATIONS = [
    "MA-BR-14", "MA-BR-18", "MA-BR-52", "MA-BR-79", "MA-PL-63", "MA-PL-66",
    "MA-BA-115", "MA-BA-101", "MA-BA-109", "MA-BA-112", "MA-BA-105",
    "MA-BA-113", "MA-BA-57", "MA-BA-2", "MA-BA-87", "MA-BA-13",
]

cocorahs = pd.read_csv("buzzards_bay_cocorahs_daily.csv", parse_dates=["date"])
cocorahs = cocorahs[cocorahs["stationNumber"].isin(STATIONS)]

hrrr = pd.read_csv("buzzards_bay_hrrr_daily.csv", parse_dates=["date"])

merged = cocorahs.merge(hrrr, on=["stationNumber", "date"], how="inner")
merged.to_csv("buzzards_bay_hrrr_comparison_daily.csv", index=False)
print(f"Merged {len(merged)} matched gauge/HRRR day-station records")

nz = merged[merged["precip_in"] > 0].copy()
print(f"Non-zero gauge days: {len(nz)}")

LEADS = [("Day-1", "hrrr_day1_in"), ("Day-2", "hrrr_day2_in")]

summary_rows = []
for lead, col in LEADS:
    overall_r = nz["precip_in"].corr(nz[col])
    bias = (nz[col] - nz["precip_in"]).mean()
    print(f"\n{lead} ({col}): pooled r={overall_r:.3f}, mean bias={bias:+.3f} in (n={len(nz)})")
    summary_rows.append({
        "stationNumber": "ALL", "stationName": "(pooled)",
        "n": len(nz), "r": overall_r, "bias_in": bias,
    })

    for station in STATIONS:
        g = nz[nz["stationNumber"] == station]
        name = g["stationName"].iloc[0] if len(g) else station
        r = g["precip_in"].corr(g[col]) if len(g) > 1 else np.nan
        bias_s = (g[col] - g["precip_in"]).mean() if len(g) else np.nan
        summary_rows.append({
            "stationNumber": station, "stationName": name,
            "n": len(g), "r": r, "bias_in": bias_s,
        })

    ncols = 4
    nrows = int(np.ceil(len(STATIONS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, station in zip(axes, STATIONS):
        g = nz[nz["stationNumber"] == station]
        name = g["stationName"].iloc[0] if len(g) else station
        r = g["precip_in"].corr(g[col]) if len(g) > 1 else np.nan
        ax.scatter(g["precip_in"], g[col], s=10, alpha=0.5)
        lim = max(g["precip_in"].max(), g[col].max(), 0.1) * 1.05 if len(g) else 1
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{station}\n{name}\nr={r:.2f}, n={len(g)}", fontsize=9)
        ax.set_xlabel("Gauge (in)", fontsize=8)
        ax.set_ylabel(f"HRRR {lead} (in)", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(STATIONS):]:
        ax.axis("off")

    fig.suptitle(f"Buzzards Bay: CoCoRaHS vs HRRR {lead}, 7am-7am ET aligned (gauge>0 days)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = f"buzzards_bay_hrrr_scatter_{lead.lower().replace('-', '')}.png"
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")

summary = pd.DataFrame(summary_rows)
# Day-1 and Day-2 rows share stationNumber/stationName; separate columns per lead
summary_day1 = summary[summary["stationNumber"] != "ALL"].iloc[:len(STATIONS)].reset_index(drop=True)
summary_day2 = summary[summary["stationNumber"] != "ALL"].iloc[len(STATIONS):].reset_index(drop=True)
pooled = summary[summary["stationNumber"] == "ALL"].reset_index(drop=True)

out_summary = summary_day1[["stationNumber", "stationName", "n"]].copy()
out_summary["r_day1"] = summary_day1["r"]
out_summary["bias_day1_in"] = summary_day1["bias_in"]
out_summary["r_day2"] = summary_day2["r"]
out_summary["bias_day2_in"] = summary_day2["bias_in"]
out_summary.to_csv("buzzards_bay_hrrr_summary.csv", index=False)
print("\nSaved buzzards_bay_hrrr_summary.csv")
print(out_summary)
