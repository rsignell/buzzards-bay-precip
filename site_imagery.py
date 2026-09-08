"""
Look at the oak site directly, next to a same-tile oak control.

The phenology curve says the site is evergreen-flat (NDVI 0.71-0.76 from
November through May) while a known Oak/hickory patch swings 0.90 -> 0.33. The
same code produces the correct curve at the control, so this is not a bug in
the compositing -- but the site and the pine control both sit in MGRS 19TCG
while the oak control does not, so a tile-specific artefact has not been ruled
out. Hence a second oak control drawn from inside the basins, on the same
tiles as the site.

Renders true colour in both seasons plus the deciduous index, so the ground can
be identified by eye rather than argued about from numbers.

Run on the oak-mapping cluster.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import rioxarray  # noqa: F401
from matplotlib.patches import Circle
from odc.geo.geobox import GeoBox
from pyproj import Transformer
from pystac_client import Client

CRS = "EPSG:32619"
SITE = (41.7266604, -70.5966575)
STAC = "https://earth-search.aws.element84.com/v1"
SCALE, MIN_DENOM = 0.0001, 0.05
SCL_KEEP = [4, 5, 7]
HALF = 1000  # metres -> 2 km box

WINDOWS = {
    "summer": ("2026-07-01", "2026-08-15"),
    # Widened vs. the main script: at the site only one Feb-Mar date was clear,
    # and a one-scene "median" is not a median.
    "leafoff": ("2025-12-01", "2026-03-31"),
}


def load(gbox, lon, lat, d0, d1, bands=("red", "green", "blue", "nir", "scl")):
    items = list(Client.open(STAC).search(
        collections=["sentinel-2-l2a"],
        bbox=[lon - .05, lat - .05, lon + .05, lat + .05],
        datetime=f"{d0}T00:00:00Z/{d1}T23:59:59Z",
        query={"eo:cloud_cover": {"lt": 40}}).items())
    ds = odc.stac.load(items, bands=bands, geobox=gbox, groupby="solar_day",
                       resampling="nearest", dtype="uint16", nodata=0)
    good = ds.scl.isin(SCL_KEEP)
    out = {}
    for b in bands:
        if b == "scl":
            continue
        out[b] = (ds[b].where(ds[b] > 0).astype("float32") * SCALE).where(good) \
            .median("time", skipna=True)
    den = out["nir"] + out["red"]
    out["ndvi"] = ((out["nir"] - out["red"]) / den).where(den > MIN_DENOM).clip(-1, 1)
    out["n"] = ds.sizes["time"]
    return out


def truecolor(d):
    """Percentile-stretched RGB."""
    rgb = np.dstack([d["red"].values, d["green"].values, d["blue"].values])
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def same_tile_oak_control():
    """A confident BIGMAP Oak/hickory patch inside the basins (same MGRS tiles
    as the site), so tile-specific processing cannot explain a difference."""
    ftg = rioxarray.open_rasterio(
        "forest/bigmap_forest_type_group.tif").squeeze(drop=True)
    m = ftg.values == 500
    ys, xs = np.nonzero(m)
    for y, x in zip(ys[::13], xs[::13]):
        if y < 10 or x < 10 or y + 10 >= m.shape[0] or x + 10 >= m.shape[1]:
            continue
        # Require a solid 600 m block of oak/hickory and an easterly position,
        # so it sits on the Cape rather than out at the western edge.
        if m[y - 10:y + 11, x - 10:x + 11].all() and float(ftg.x[x]) > 360000:
            return float(ftg.x[x]), float(ftg.y[y])
    raise SystemExit("no same-tile oak control found")


def panel_row(axes, cx, cy, label):
    tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(cx, cy)
    gbox = GeoBox.from_bbox((cx - HALF, cy - HALF, cx + HALF, cy + HALF),
                            crs=CRS, resolution=10)
    ext = [cx - HALF, cx + HALF, cy - HALF, cy + HALF]

    data = {w: load(gbox, lon, lat, *WINDOWS[w]) for w in WINDOWS}
    decid = data["summer"]["ndvi"] - data["leafoff"]["ndvi"]

    for ax, (title, img, kw) in zip(axes, [
        (f"{label}\nsummer true colour ({data['summer']['n']} dates)",
         truecolor(data["summer"]), {}),
        (f"leaf-off true colour ({data['leafoff']['n']} dates)",
         truecolor(data["leafoff"]), {}),
        ("deciduous index", decid.values,
         dict(cmap="RdYlGn", vmin=-0.1, vmax=0.6)),
    ]):
        im = ax.imshow(img, extent=ext, origin="upper", **kw)
        # ~50 acres is a 450 m circle; mark it for scale.
        ax.add_patch(Circle((cx, cy), 225, fill=False, color="#00e5ff", lw=2))
        ax.plot(cx, cy, "+", color="#00e5ff", ms=12, mew=2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if kw:
            plt.colorbar(im, ax=ax, shrink=0.8)

    med = float(np.nanmedian(
        decid.sel(x=slice(cx - 225, cx + 225), y=slice(cy + 225, cy - 225)).values))
    print(f"  {label:28s} decid median in 450 m circle: {med:+.3f}")
    return med


def main():
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    sx, sy = tr.transform(SITE[1], SITE[0])
    ox, oy = same_tile_oak_control()
    olon, olat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(ox, oy)
    print(f"same-tile oak control at {olat:.5f}, {olon:.5f}\n")

    fig, axes = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)
    panel_row(axes[0], sx, sy, "SITE (your 50 acres)")
    panel_row(axes[1], ox, oy, "Control: BIGMAP Oak/hickory, same tiles")
    fig.suptitle("Sentinel-2 at the oak site vs. a same-tile oak control "
                 "(cyan circle ~= 50 acres)", fontsize=13, fontweight="bold")
    fig.savefig("s2/site_imagery.png", dpi=110)
    print("\nwrote s2/site_imagery.png")


if __name__ == "__main__":
    main()
