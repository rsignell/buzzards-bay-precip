"""
Acceptance test at two owner-confirmed oak forests.

Earlier attempts were sabotaged by coordinates that turned out not to be oak
forest: the first was a cranberry bog (evergreen, so it correctly read ~0.12)
and the second, which I picked and the owner affirmed, sat on the salt-marsh /
upland boundary. These two are given by the owner as the centres of oak stands
they walk.

The open question this settles: does the index under-read oak on the Cape --
plausibly because evergreen understory (holly, greenbrier, laurel) keeps winter
NDVI up beneath bare oaks -- or was the surrounding upland genuinely mixed
oak-pine? A pure Oak/hickory control reads 0.43 and pitch pine 0.10; where these
stands land between those says which.

Outputs:
  s2/oak_sites.png   imagery, oak index and phenology for both stands

Run on the oak-mapping cluster.
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
SITES = {
    "Oak forest A": (41.7297453, -70.6025873),
    "Oak forest B": (41.7287718, -70.6009114),
}
CONTROLS = {
    "Oak/hickory control": (41.72572, -70.53661),
    "pitch pine control": (41.90640, -70.69646),
}

STAC = "https://earth-search.aws.element84.com/v1"
SCALE, MIN_DENOM = 0.0001, 0.05
SCL_KEEP = [4, 5, 7]
ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/export")
TR = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

DRAIN = {1: "Excessively drained", 2: "Somewhat excessively", 3: "Well drained",
         4: "Moderately well", 5: "Somewhat poorly", 6: "Poorly drained",
         7: "Very poorly drained"}
FTG = {0: "Non-forest", 100: "W/R/J pine", 400: "Oak/pine", 500: "Oak/hickory",
       700: "Elm/ash/cottonwood", 800: "Maple/beech/birch"}


def box(da, cx, cy, half):
    v = da.sel(x=slice(cx - half, cx + half),
               y=slice(cy + half, cy - half)).values.astype(float)
    return v[np.isfinite(v)]


def esri(bbox, size=900):
    r = requests.get(ESRI, params={
        "bbox": ",".join(str(v) for v in bbox), "bboxSR": "32619",
        "imageSR": "32619", "size": f"{size},{size}", "format": "png",
        "f": "image"}, timeout=120)
    r.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")) / 255


def phenology(lat, lon, label, half=120):
    cx, cy = TR.transform(lon, lat)
    gb = GeoBox.from_bbox((cx - half, cy - half, cx + half, cy + half),
                          crs=CRS, resolution=10)
    items = list(Client.open(STAC).search(
        collections=["sentinel-2-l2a"],
        bbox=[lon - .03, lat - .03, lon + .03, lat + .03],
        datetime="2025-09-01T00:00:00Z/2026-09-01T00:00:00Z",
        query={"eo:cloud_cover": {"lt": 60}}).items())
    ds = odc.stac.load(items, bands=("red", "nir", "scl"), geobox=gb,
                       groupby="solar_day", resampling="nearest",
                       dtype="uint16", nodata=0)
    good = ds.scl.isin(SCL_KEEP)
    red = (ds.red.where(ds.red > 0).astype("float32") * SCALE).where(good)
    nir = (ds.nir.where(ds.nir > 0).astype("float32") * SCALE).where(good)
    den = nir + red
    ndvi = ((nir - red) / den).where(den > MIN_DENOM).clip(-1, 1)
    npx = gb.shape[0] * gb.shape[1]
    n = ndvi.notnull().sum(("x", "y")).values
    med = ndvi.median(("x", "y"), skipna=True).values
    keep = n > 0.6 * npx
    return pd.DataFrame({"date": pd.to_datetime(ds.time.values)[keep],
                         "ndvi": med[keep]}).assign(site=label)


def main():
    oak = rioxarray.open_rasterio("s2/oak_index.tif").squeeze(drop=True)
    decid = rioxarray.open_rasterio("s2/s2_decid.tif").squeeze(drop=True)
    summer = rioxarray.open_rasterio("s2/s2_ndvi_summer.tif").squeeze(drop=True)
    leafoff = rioxarray.open_rasterio("s2/s2_ndvi_leafoff.tif").squeeze(drop=True)
    drain = rioxarray.open_rasterio("soil/soil_drainage.tif").squeeze(drop=True)
    awc = rioxarray.open_rasterio("soil/soil_awc.tif").squeeze(drop=True)
    marsh = rioxarray.open_rasterio("soil/tidal_marsh_mask.tif").squeeze(drop=True)
    ftg = rioxarray.open_rasterio(
        "forest/bigmap_forest_type_group.tif").squeeze(drop=True)
    fhp = rioxarray.open_rasterio(
        "forest/fhp_ba_oak_deciduous_spp.tif").squeeze(drop=True)

    for name, (lat, lon) in SITES.items():
        cx, cy = TR.transform(lon, lat)
        print(f"\n{'=' * 66}\n{name}  ({lat}, {lon})\n{'=' * 66}")

        # Is it even canopy, and is it marsh?
        raw = decid.sel(x=slice(cx - 225, cx + 225), y=slice(cy + 225, cy - 225))
        gated = oak.sel(x=slice(cx - 225, cx + 225), y=slice(cy + 225, cy - 225))
        frac = 100 * np.isfinite(gated.values).sum() / raw.values.size
        mfrac = 100 * (box(marsh, cx, cy, 225) == 1).mean() if len(
            box(marsh, cx, cy, 225)) else 0
        print(f"  450 m circle: {frac:.0f}% passes the canopy gate, "
              f"{mfrac:.0f}% tidal marsh")

        print(f"\n  {'scale':>8}  {'oak index':>10}{'decid':>9}{'summer':>9}"
              f"{'leaf-off':>10}")
        for half, lab in [(15, "30 m"), (50, "100 m"), (150, "300 m"),
                          (225, "450 m")]:
            def med(da):
                v = box(da, cx, cy, half)
                return np.median(v) if v.size else np.nan
            print(f"  {lab:>8}  {med(oak):10.3f}{med(decid):9.3f}"
                  f"{med(summer):9.3f}{med(leafoff):10.3f}")

        dv = box(drain, cx, cy, 100)
        av = box(awc, cx, cy, 100)
        fv = box(ftg, cx, cy, 150).astype(int)
        hv = box(fhp, cx, cy, 150)
        u, c = np.unique(fv, return_counts=True)
        print(f"\n  soil: {DRAIN.get(int(np.median(dv)), '?') if dv.size else '?'}"
              f", AWC {np.median(av) if av.size else float('nan'):.1f} cm")
        print("  BIGMAP: " + ", ".join(
            f"{FTG.get(int(v), v)} {100 * n / fv.size:.0f}%"
            for v, n in sorted(zip(u, c), key=lambda t: -t[1])[:3]))
        print(f"  FHP oak basal area: median {np.median(hv) if hv.size else 0:.1f}"
              f", max {hv.max() if hv.size else 0:.1f} sq ft/ac")

    # --- where do they land between the controls? -------------------------- #
    print(f"\n{'=' * 66}\nVERDICT\n{'=' * 66}")
    ref = {}
    for name, (lat, lon) in CONTROLS.items():
        cx, cy = TR.transform(lon, lat)
        v = box(oak, cx, cy, 225)
        ref[name] = np.median(v)
        print(f"  {name:24s} {ref[name]:.3f}")
    lo = ref["pitch pine control"]
    hi = ref["Oak/hickory control"]
    for name, (lat, lon) in SITES.items():
        cx, cy = TR.transform(lon, lat)
        v = box(oak, cx, cy, 225)
        m = np.median(v) if v.size else np.nan
        print(f"  {name:24s} {m:.3f}  =  {(m - lo) / (hi - lo):.0%} "
              f"of the way from pine to oak")

    figure(oak, decid)


def figure(oak, decid):
    frames = [phenology(lat, lon, n) for n, (lat, lon) in SITES.items()]
    frames += [phenology(lat, lon, n) for n, (lat, lon) in CONTROLS.items()]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv("s2/oak_sites_phenology.csv", index=False)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])

    ax = fig.add_subplot(gs[0, :])
    style = {"Oak forest A": ("#1b5e20", 2.8, "-"),
             "Oak forest B": ("#2e8b57", 2.8, "-"),
             "Oak/hickory control": ("#7fb069", 1.8, "--"),
             "pitch pine control": ("#c1440e", 1.8, "--")}
    for label, (col, lw, ls) in style.items():
        s = df[df.site == label].sort_values("date")
        ax.plot(s.date, s.ndvi, ls, color=col, lw=lw, marker="o", ms=3.5,
                label=label)
    ax.set_ylabel("NDVI (median of a 240 m box)")
    ax.set_title("Annual NDVI phenology at the two owner-confirmed oak stands",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.1, 1.0)

    for i, (name, (lat, lon)) in enumerate(SITES.items()):
        cx, cy = TR.transform(lon, lat)
        H = 500
        bbox = (cx - H, cy - H, cx + H, cy + H)
        ext = [bbox[0], bbox[2], bbox[1], bbox[3]]
        img = esri(bbox)
        d = oak.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1])).values

        a = fig.add_subplot(gs[1, 2 * i])
        a.imshow(img, extent=ext, origin="upper")
        a.add_patch(Circle((cx, cy), 225, fill=False, color="#00e5ff", lw=2.5))
        a.set_title(f"{name} — imagery", fontsize=10)
        a.set_xticks([]); a.set_yticks([])

        a = fig.add_subplot(gs[1, 2 * i + 1])
        im = a.imshow(d, extent=ext, origin="upper", cmap="RdYlGn",
                      vmin=-0.1, vmax=0.6, interpolation="nearest")
        a.add_patch(Circle((cx, cy), 225, fill=False, color="#00e5ff", lw=2.5))
        a.set_title(f"{name} — oak index", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, shrink=0.8)

    fig.suptitle("Owner-confirmed oak stands vs. controls",
                 fontsize=14, fontweight="bold")
    fig.savefig("s2/oak_sites.png", dpi=110)
    print("\nwrote s2/oak_sites.png")


if __name__ == "__main__":
    main()
