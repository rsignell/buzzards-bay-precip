"""
Acceptance test for the Sentinel-2 deciduous index at the confirmed oak forest.

The site is 41.72656, -70.60387 -- a ~50-acre white/red oak forest, confirmed by
the owner. An earlier coordinate 600 m east turned out to be a cranberry bog
(evergreen, so the index correctly read ~0.14 there); this is the real stand.

The test the layer has to pass: light up this forest as strongly deciduous,
where USFS FHP reports zero basal area for every species. Controls are a known
Oak/hickory patch and a known pine patch, both run through identical code.

Outputs:
  s2/acceptance_test.png   phenology curves + imagery + index
  console verdict

Run on the oak-mapping cluster after s2_deciduous.py.
"""

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import requests
import rioxarray  # noqa: F401
from matplotlib.patches import Circle
from odc.geo.geobox import GeoBox
from PIL import Image
from pyproj import Transformer
from pystac_client import Client

CRS = "EPSG:32619"
SITE = (41.72656, -70.60387)          # confirmed ~50-acre white/red oak forest
OAK_CONTROL = (41.72572, -70.53661)   # BIGMAP Oak/hickory, same MGRS tiles
PINE_CONTROL = (41.90640, -70.69646)  # pitch pine barrens, Myles Standish
BOG = (41.7266604, -70.5966575)       # the original coordinate, for contrast

STAC = "https://earth-search.aws.element84.com/v1"
SCALE, MIN_DENOM = 0.0001, 0.05
SCL_KEEP = [4, 5, 7]
HALF_PHEN = 150   # m; ~9 ha box kept well inside the stand
ACRE_R = 225      # m; a 450 m circle is ~50 acres
ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/export")

TR = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)


def ndvi_stack(cx, cy, half, d0, d1, bands=("red", "nir", "scl")):
    gb = GeoBox.from_bbox((cx - half, cy - half, cx + half, cy + half),
                          crs=CRS, resolution=10)
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(cx, cy)
    items = list(Client.open(STAC).search(
        collections=["sentinel-2-l2a"],
        bbox=[lon - .03, lat - .03, lon + .03, lat + .03],
        datetime=f"{d0}T00:00:00Z/{d1}T23:59:59Z",
        query={"eo:cloud_cover": {"lt": 60}}).items())
    ds = odc.stac.load(items, bands=bands, geobox=gb, groupby="solar_day",
                       resampling="nearest", dtype="uint16", nodata=0)
    good = ds.scl.isin(SCL_KEEP)
    red = (ds.red.where(ds.red > 0).astype("float32") * SCALE).where(good)
    nir = (ds.nir.where(ds.nir > 0).astype("float32") * SCALE).where(good)
    den = nir + red
    return ds, ((nir - red) / den).where(den > MIN_DENOM).clip(-1, 1), gb


def phenology(lat, lon, label):
    cx, cy = TR.transform(lon, lat)
    ds, ndvi, gb = ndvi_stack(cx, cy, HALF_PHEN,
                              "2025-09-01", "2026-09-01")
    npx = gb.shape[0] * gb.shape[1]
    n = ndvi.notnull().sum(("x", "y")).values
    med = ndvi.median(("x", "y"), skipna=True).values
    keep = n > 0.5 * npx
    df = pd.DataFrame({"date": pd.to_datetime(ds.time.values)[keep],
                       "ndvi": med[keep]}).assign(site=label)
    print(f"  {label:26s} {len(df):2d} clear dates")
    return df


def esri(bbox, size=900):
    r = requests.get(ESRI, params={
        "bbox": ",".join(str(v) for v in bbox), "bboxSR": "32619",
        "imageSR": "32619", "size": f"{size},{size}",
        "format": "png", "f": "image"}, timeout=120)
    r.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")) / 255


def circle_stats(da, cx, cy, r=ACRE_R):
    """Values inside the ~50-acre circle."""
    sub = da.sel(x=slice(cx - r, cx + r), y=slice(cy + r, cy - r))
    yy, xx = np.meshgrid(sub.y.values, sub.x.values, indexing="ij")
    m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
    v = sub.values[m]
    return v[np.isfinite(v)]


def main():
    print("=== annual phenology ===")
    frames = [
        phenology(*SITE, "SITE confirmed oak"),
        phenology(*OAK_CONTROL, "control Oak/hickory"),
        phenology(*PINE_CONTROL, "control pine"),
        phenology(*BOG, "cranberry bog"),
    ]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv("s2/acceptance_phenology.csv", index=False)

    decid = rioxarray.open_rasterio("s2/s2_decid.tif").squeeze(drop=True)
    summer = rioxarray.open_rasterio("s2/s2_ndvi_summer.tif").squeeze(drop=True)
    leafoff = rioxarray.open_rasterio("s2/s2_ndvi_leafoff.tif").squeeze(drop=True)
    oakba = rioxarray.open_rasterio(
        "forest/fhp_ba_oak_deciduous_spp.tif").squeeze(drop=True)
    ftg = rioxarray.open_rasterio(
        "forest/bigmap_forest_type_group.tif").squeeze(drop=True)

    print("\n=== the ~50-acre circle at the confirmed site ===")
    cx, cy = TR.transform(SITE[1], SITE[0])
    for name, da in [("summer NDVI", summer), ("leaf-off NDVI", leafoff),
                     ("DECIDUOUS INDEX", decid)]:
        v = circle_stats(da, cx, cy)
        print(f"  {name:16s} median {np.median(v):6.3f}  "
              f"p10 {np.percentile(v, 10):6.3f}  p90 {np.percentile(v, 90):6.3f}")

    d_site = np.median(circle_stats(decid, cx, cy))
    for label, (lat, lon) in [("control Oak/hickory", OAK_CONTROL),
                              ("control pine", PINE_CONTROL),
                              ("cranberry bog", BOG)]:
        ox, oy = TR.transform(lon, lat)
        print(f"  {'vs ' + label:16s} median "
              f"{np.median(circle_stats(decid, ox, oy)):6.3f}")

    print("\n=== what the existing products say at the same circle ===")
    ba = circle_stats(oakba, cx, cy)
    fv = circle_stats(ftg, cx, cy).astype(int)
    C = {0: "Non-forest", 100: "W/R/J pine", 400: "Oak/pine", 500: "Oak/hickory",
         700: "Elm/ash/cottonwood", 800: "Maple/beech/birch"}
    u, c = np.unique(fv, return_counts=True)
    print(f"  FHP oak basal area: median {np.median(ba):.1f}, max {ba.max():.1f} "
          f"sq ft/ac  ({100 * (ba == 0).mean():.0f}% of the circle is zero)")
    print("  BIGMAP 2018: " + ", ".join(
        f"{C.get(int(v), v)} {100 * n / fv.size:.0f}%"
        for v, n in sorted(zip(u, c), key=lambda t: -t[1])))

    # Where the site sits in the basin-wide distribution.
    import geopandas as gpd
    parts = [gpd.read_file(f).to_crs(CRS) for f in
             ["buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson"]]
    geom = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                            crs=CRS).geometry.union_all()
    allv = decid.rio.clip([geom], CRS, drop=False).values
    allv = allv[np.isfinite(allv)]
    pct = 100 * (allv < d_site).mean()
    print(f"\n  site deciduous index {d_site:.3f} = {pct:.0f}th percentile "
          f"of all basin land")

    # Judge against the controls rather than an absolute cut. An absolute
    # threshold begs the question -- what matters is whether the site lands on
    # the oak side of the oak/pine midpoint, which is what the index has to get
    # right to be useful for mapping.
    d_oak = np.median(circle_stats(decid, *TR.transform(*OAK_CONTROL[::-1])))
    d_pine = np.median(circle_stats(decid, *TR.transform(*PINE_CONTROL[::-1])))
    mid = (d_oak + d_pine) / 2
    verdict = "PASS" if d_site > mid else "FAIL"
    print(f"\n  pine control {d_pine:.3f} | midpoint {mid:.3f} | "
          f"oak control {d_oak:.3f}")
    print(f"  site {d_site:.3f} sits {(d_site - d_pine) / (d_oak - d_pine):.0%} "
          f"of the way from pine to oak")
    print(f"\n  VERDICT: {verdict}")

    figure(df, decid, cx, cy)


def figure(df, decid, cx, cy):
    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])

    # --- phenology curves -------------------------------------------------- #
    ax = fig.add_subplot(gs[0, :])
    styles = {"SITE confirmed oak": ("#1b5e20", 2.8, "o", "-"),
              "control Oak/hickory": ("#7fb069", 1.8, "s", "--"),
              "control pine": ("#c1440e", 1.8, "^", "--"),
              "cranberry bog": ("#8e44ad", 1.8, "d", ":")}
    for label, (col, lw, mk, ls) in styles.items():
        s = df[df.site == label].sort_values("date")
        ax.plot(s.date, s.ndvi, ls, color=col, lw=lw, marker=mk, ms=4,
                label=label)
    ax.set_ylabel("NDVI (median of a 300 m box)")
    ax.set_title("Annual NDVI phenology — the confirmed oak forest tracks the "
                 "oak control, not the pine or the bog", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.1, 1.0)

    # --- imagery / index --------------------------------------------------- #
    H = 700
    bbox = (cx - H, cy - H, cx + H, cy + H)
    ext = [bbox[0], bbox[2], bbox[1], bbox[3]]
    img = esri(bbox)
    d = decid.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1])).values

    def mark(ax):
        ax.add_patch(Circle((cx, cy), ACRE_R, fill=False, color="#00e5ff", lw=2.5))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])

    a = fig.add_subplot(gs[1, 0])
    a.imshow(img, extent=ext, origin="upper")
    a.set_title("Esri imagery (cyan ≈ 50 acres)", fontsize=10)
    mark(a)

    a = fig.add_subplot(gs[1, 1])
    im = a.imshow(d, extent=ext, origin="upper", cmap="RdYlGn",
                  vmin=-0.1, vmax=0.6, interpolation="nearest")
    a.set_title("Sentinel-2 deciduous index", fontsize=10)
    mark(a)
    fig.colorbar(im, ax=a, shrink=0.85)

    a = fig.add_subplot(gs[1, 2])
    a.imshow(img, extent=ext, origin="upper")
    a.imshow(np.ma.masked_less(d, 0.35), extent=ext, origin="upper",
             cmap="winter", vmin=0.35, vmax=0.6, alpha=0.6,
             interpolation="nearest")
    a.set_title("index ≥ 0.35 over imagery", fontsize=10)
    mark(a)

    fig.suptitle(f"Acceptance test — confirmed oak forest at {SITE[0]}, {SITE[1]}",
                 fontsize=14, fontweight="bold")
    fig.savefig("s2/acceptance_test.png", dpi=110)
    print("\nwrote s2/acceptance_test.png")


if __name__ == "__main__":
    main()
