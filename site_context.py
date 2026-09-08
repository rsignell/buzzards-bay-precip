"""
The supplied site coordinate lands on a cranberry bog, not the oak forest.

41.7266604, -70.5966575 sits in the middle of a working cranberry bog -- the
flooded beds, dikes and ditches are unmistakable in sub-metre imagery, with a
golf course and condos immediately east and continuous forest to the west. That
explains the phenology exactly: cranberry is an EVERGREEN dwarf shrub, so NDVI
stays ~0.75 through the winter and the deciduous index reads 0.12. Nothing was
wrong with the compositing; it was measuring a bog.

This renders the neighbourhood so the actual oak block can be identified: Esri
imagery and the deciduous index on the same grid, side by side.

Run on the oak-mapping cluster.
"""

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import rioxarray  # noqa: F401
from matplotlib.patches import Circle
from PIL import Image
from pyproj import Transformer

CRS = "EPSG:32619"
SITE = (41.7266604, -70.5966575)
HALF = 1200  # metres
ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/export")


def esri(bbox, size=1200):
    """Basemap on the *same* CRS and bbox as the raster, so they register."""
    r = requests.get(ESRI, params={
        "bbox": ",".join(str(v) for v in bbox), "bboxSR": "32619",
        "imageSR": "32619", "size": f"{size},{size}",
        "format": "png", "f": "image"}, timeout=120)
    r.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")) / 255


def main():
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    cx, cy = tr.transform(SITE[1], SITE[0])
    bbox = (cx - HALF, cy - HALF, cx + HALF, cy + HALF)
    ext = [bbox[0], bbox[2], bbox[1], bbox[3]]

    img = esri(bbox)
    decid = (rioxarray.open_rasterio("s2/s2_decid.tif").squeeze(drop=True)
             .sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1])))
    d = decid.values

    fig, axes = plt.subplots(1, 3, figsize=(19, 7), constrained_layout=True)

    def mark(ax):
        ax.add_patch(Circle((cx, cy), 30, fill=False, color="#00e5ff", lw=2.5))
        ax.plot(cx, cy, "+", color="#00e5ff", ms=16, mew=2.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])

    axes[0].imshow(img, extent=ext, origin="upper")
    axes[0].set_title("Esri imagery — the marker is on a cranberry bog\n"
                      "golf course + condos E, forest W", fontsize=11)
    mark(axes[0])

    im = axes[1].imshow(d, extent=ext, origin="upper", cmap="RdYlGn",
                        vmin=-0.1, vmax=0.6, interpolation="nearest")
    axes[1].set_title("Sentinel-2 deciduous index\n"
                      "green = strong leaf-on/leaf-off swing = oak", fontsize=11)
    mark(axes[1])
    fig.colorbar(im, ax=axes[1], shrink=0.8)

    # Where the index is confident about deciduous canopy.
    axes[2].imshow(img, extent=ext, origin="upper")
    axes[2].imshow(np.ma.masked_less(d, 0.35), extent=ext, origin="upper",
                   cmap="winter", vmin=0.35, vmax=0.6, alpha=0.55,
                   interpolation="nearest")
    axes[2].set_title("deciduous index ≥ 0.35 over imagery\n"
                      "(the oak the index actually finds)", fontsize=11)
    mark(axes[2])

    fig.suptitle("Neighbourhood of the supplied coordinate — "
                 f"{SITE[0]}, {SITE[1]}", fontsize=13, fontweight="bold")
    fig.savefig("s2/site_context.png", dpi=110)

    # Quantify a few offsets so the forest block can be compared numerically.
    print(f"deciduous index around the supplied point:")
    for label, dx, dy in [("at the point (bog)", 0, 0),
                          ("400 m W", -400, 0), ("700 m W", -700, 0),
                          ("400 m SW", -300, -300), ("700 m SW", -500, -500),
                          ("400 m S", 0, -400), ("400 m N", 0, 400),
                          ("400 m E (golf)", 400, 0)]:
        box = decid.sel(x=slice(cx + dx - 100, cx + dx + 100),
                        y=slice(cy + dy + 100, cy + dy - 100)).values
        lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True) \
            .transform(cx + dx, cy + dy)
        print(f"  {label:18s} ({lat:.5f}, {lon:.5f})  "
              f"median {np.nanmedian(box):+.3f}")
    print("\nwrote s2/site_context.png")


if __name__ == "__main__":
    main()
