"""
Pick the leaf-off window by measured separability, not by assumption.

The Feb 15 - Mar 31 window used in the first pass has two problems at the
confirmed oak site: a median of ONE valid observation in the 50-acre circle
(a one-scene "median"), and, at the pine control, several low outliers in
Jan-Mar that look like snow or low-sun-angle artefacts. Those drag pine's
leaf-off value down, inflate pine's deciduous index, and squeeze the very gap
the index depends on.

So: sweep candidate windows and score each by how far it separates known oak
from known pine, and how many observations it actually gets.

Run on the oak-mapping cluster.
"""

import numpy as np
import odc.stac
import pandas as pd
from odc.geo.geobox import GeoBox
from pyproj import Transformer
from pystac_client import Client

CRS = "EPSG:32619"
STAC = "https://earth-search.aws.element84.com/v1"
SCALE, MIN_DENOM = 0.0001, 0.05
HALF = 225  # ~50-acre circle

SITES = {
    "SITE oak (confirmed)": (41.72656, -70.60387),
    "control Oak/hickory": (41.72572, -70.53661),
    "control pine": (41.90640, -70.69646),
    "cranberry bog": (41.7266604, -70.5966575),
}

SUMMER = ("2026-07-01", "2026-08-15")
CANDIDATES = {
    "Feb15-Mar31 (current)": ("2026-02-15", "2026-03-31"),
    "Jan01-Mar31": ("2026-01-01", "2026-03-31"),
    "Dec01-Mar31": ("2025-12-01", "2026-03-31"),
    "Dec01-Mar15": ("2025-12-01", "2026-03-15"),
    "Nov15-Apr15": ("2025-11-15", "2026-04-15"),
}

# Snow is class 11; excluding it is not enough when a pixel is only partly
# snow-covered, so compare a strict variant that also drops class 5
# (not-vegetated), which partial snow and frost often land in.
VARIANTS = {"keep 4,5,7": [4, 5, 7], "strict 4,7": [4, 7]}

TR = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)


def median_ndvi(lat, lon, d0, d1, scl_keep):
    cx, cy = TR.transform(lon, lat)
    gb = GeoBox.from_bbox((cx - HALF, cy - HALF, cx + HALF, cy + HALF),
                          crs=CRS, resolution=10)
    items = list(Client.open(STAC).search(
        collections=["sentinel-2-l2a"],
        bbox=[lon - .03, lat - .03, lon + .03, lat + .03],
        datetime=f"{d0}T00:00:00Z/{d1}T23:59:59Z",
        query={"eo:cloud_cover": {"lt": 40}}).items())
    ds = odc.stac.load(items, bands=("red", "nir", "scl"), geobox=gb,
                       groupby="solar_day", resampling="nearest",
                       dtype="uint16", nodata=0)
    good = ds.scl.isin(scl_keep)
    red = (ds.red.where(ds.red > 0).astype("float32") * SCALE).where(good)
    nir = (ds.nir.where(ds.nir > 0).astype("float32") * SCALE).where(good)
    den = nir + red
    ndvi = ((nir - red) / den).where(den > MIN_DENOM).clip(-1, 1)
    med = float(np.nanmedian(ndvi.median("time", skipna=True).values))
    nobs = float(np.median(ndvi.notnull().sum("time").values))
    return med, nobs


def main():
    print("computing summer reference ...")
    summer = {k: median_ndvi(*v, *SUMMER, [4, 5, 7])[0] for k, v in SITES.items()}
    print("  " + "  ".join(f"{k.split()[0]}={v:.3f}" for k, v in summer.items()))

    rows = []
    for vname, keep in VARIANTS.items():
        for wname, (d0, d1) in CANDIDATES.items():
            rec = {"variant": vname, "window": wname}
            for site, (lat, lon) in SITES.items():
                med, nobs = median_ndvi(lat, lon, d0, d1, keep)
                rec[site] = summer[site] - med
                rec[f"n_{site}"] = nobs
            rows.append(rec)
            print(f"  done {vname:12s} {wname}")

    df = pd.DataFrame(rows)
    df["oak-pine gap"] = df["control Oak/hickory"] - df["control pine"]
    df["site-pine gap"] = df["SITE oak (confirmed)"] - df["control pine"]

    pd.set_option("display.width", 200)
    print("\n=== deciduous index by leaf-off window ===")
    print(f"{'variant':13s}{'window':24s}{'SITE':>7}{'oakctl':>8}{'pinectl':>8}"
          f"{'bog':>7}{'oak-pine':>10}{'site-pine':>11}{'n@site':>8}")
    for _, r in df.iterrows():
        print(f"{r['variant']:13s}{r['window']:24s}"
              f"{r['SITE oak (confirmed)']:7.3f}{r['control Oak/hickory']:8.3f}"
              f"{r['control pine']:8.3f}{r['cranberry bog']:7.3f}"
              f"{r['oak-pine gap']:10.3f}{r['site-pine gap']:11.3f}"
              f"{r['n_SITE oak (confirmed)']:8.0f}")

    best = df.loc[df["site-pine gap"].idxmax()]
    print(f"\nbest by site-pine gap: {best['variant']} / {best['window']}  "
          f"(gap {best['site-pine gap']:.3f}, {best['n_SITE oak (confirmed)']:.0f} obs at site)")
    df.to_csv("s2/leafoff_window_sweep.csv", index=False)


if __name__ == "__main__":
    main()
