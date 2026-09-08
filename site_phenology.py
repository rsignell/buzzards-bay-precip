"""
Annual NDVI phenology at the oak site, against known-oak and known-pine controls.

The deciduous index scored the site at 0.149 -- below the pine median -- because
its leaf-off NDVI came out at 0.720, which is far too green for bare oak in
March. Either the site is not behaving deciduously, or the leaf-off composite is
wrong. A per-date curve tells those apart immediately, and also shows whether
the assumed leaf-off window is even in the right place.

Controls are drawn from BIGMAP's confident pixels: a patch where it says
Oak/hickory and one where it says White/red/jack pine, both far from the site.

Run on the oak-mapping cluster.
"""

import numpy as np
import odc.stac
import pandas as pd
import rioxarray  # noqa: F401
from odc.geo.geobox import GeoBox
from pystac_client import Client
from pyproj import Transformer

CRS = "EPSG:32619"
SITE = (41.7266604, -70.5966575)
STAC = "https://earth-search.aws.element84.com/v1"
SCALE, MIN_DENOM = 0.0001, 0.05
SCL_KEEP = [4, 5, 7]
HALF = 150  # metres; ~9 ha box, comfortably inside a 50-acre stand


def controls():
    """One confident Oak/hickory patch and one confident pine patch."""
    ftg = rioxarray.open_rasterio(
        "forest/bigmap_forest_type_group.tif").squeeze(drop=True)
    out = {}
    for code, name in [(500, "control Oak/hickory"), (100, "control pine")]:
        m = ftg.values == code
        # Erode to interior pixels so the box is not a mixed edge.
        ys, xs = np.nonzero(m)
        keep = []
        for y, x in zip(ys[::37], xs[::37]):
            if y < 6 or x < 6 or y + 6 >= m.shape[0] or x + 6 >= m.shape[1]:
                continue
            if m[y - 6:y + 7, x - 6:x + 7].all():
                keep.append((float(ftg.x[x]), float(ftg.y[y])))
            if len(keep) >= 1:
                break
        out[name] = keep[0]
    return out


def ndvi_series(cx, cy, label):
    gb = GeoBox.from_bbox((cx - HALF, cy - HALF, cx + HALF, cy + HALF),
                          crs=CRS, resolution=10)
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(cx, cy)
    items = list(Client.open(STAC).search(
        collections=["sentinel-2-l2a"], bbox=[lon - .02, lat - .02, lon + .02, lat + .02],
        datetime="2025-09-01T00:00:00Z/2026-09-01T00:00:00Z",
        query={"eo:cloud_cover": {"lt": 60}}).items())
    ds = odc.stac.load(items, bands=("red", "nir", "scl"), geobox=gb,
                       groupby="solar_day", resampling="nearest",
                       dtype="uint16", nodata=0)
    good = ds.scl.isin(SCL_KEEP)
    red = ds.red.where(ds.red > 0).astype("float32") * SCALE
    nir = ds.nir.where(ds.nir > 0).astype("float32") * SCALE
    red, nir = red.where(good), nir.where(good)
    den = nir + red
    ndvi = ((nir - red) / den).where(den > MIN_DENOM).clip(-1, 1)

    n_valid = ndvi.notnull().sum(("x", "y")).values
    med = ndvi.median(("x", "y"), skipna=True).values
    dates = pd.to_datetime(ds.time.values).date
    npx = gb.shape[0] * gb.shape[1]

    rows = [(d, m, int(n)) for d, m, n in zip(dates, med, n_valid)
            if n > 0.5 * npx]  # require the box to be mostly clear
    print(f"\n=== {label}  ({lat:.5f}, {lon:.5f}) ===")
    print(f"  {len(rows)} of {len(dates)} dates with >50% of the box clear")
    for d, m, n in rows:
        bar = "#" * int(max(m, 0) * 50)
        print(f"    {d}  NDVI {m:5.3f}  ({100 * n / npx:3.0f}% clear) {bar}")
    return pd.DataFrame(rows, columns=["date", "ndvi", "nvalid"]).assign(site=label)


def main():
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    pts = {"SITE oak forest": tr.transform(SITE[1], SITE[0])}
    pts.update(controls())

    frames = [ndvi_series(cx, cy, label) for label, (cx, cy) in pts.items()]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv("s2/site_phenology.csv", index=False)

    print("\n=== seasonal summary (median NDVI by window) ===")
    df["month"] = pd.to_datetime(df.date).dt.month
    windows = {"leaf-off Feb-Mar": [2, 3], "leaf-out May": [5],
               "summer Jul-Aug": [7, 8], "senesce Oct-Nov": [10, 11],
               "winter Dec-Jan": [12, 1]}
    print(f"  {'site':22s}" + "".join(f"{w:>18s}" for w in windows))
    for label in pts:
        s = df[df.site == label]
        cells = []
        for w, months in windows.items():
            v = s[s.month.isin(months)].ndvi
            cells.append(f"{v.median():.3f} (n={len(v)})" if len(v) else "  --")
        print(f"  {label:22s}" + "".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
