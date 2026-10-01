"""
Interactive map: where TESSERA thinks the wet ground is around a set of
reference points.

Layers (click legend entries to toggle the vectors):
  * Esri satellite imagery
  * TESSERA P(forested wetland), closed canopy only, shown from 30 %
  * "likely wet swale" -- canopy where P(forested wetland) >= 50 %, cleaned of
    specks and smoothed, as outlines with area and mean probability on hover
  * MassGIS 2016 forested wetland polygons (all sizes), for comparison
  * the reference points, from local_sites.json

local_sites.json is git-ignored on purpose: the reference points include a
private home location and this repository is public. Its shape:

  {"center": "<point name the window is centred on>",
   "title": "<map title>",
   "transect": ["<point name>", "<point name>"],   # optional: report wet
                                                   # ground on the line
   "points": {"<name>": {"lat": .., "lon": .., "label_dx": m, "label_dy": m,
                         "align": "left" | "right"}, ...}}

Output: tessera/pocasset_swale_map.html (standalone; tiles load from Esri).
"""

import json
from pathlib import Path

import geopandas as gpd
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
import numpy as np
import rasterio
import requests
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import LineString, Point, shape

SITES = json.loads(Path("local_sites.json").read_text())
HALF = 1500            # m, half-width of the window
SWALE_P = 50           # % P(forested wetland) that counts as "likely wet"
MIN_AREA = 2000        # m2; drop smaller patches
OUT = "tessera/pocasset_swale_map.html"
LCLU = ("https://arcgisserver.digital.mass.gov/arcgisserver/rest/services/"
        "AGOL/LandCoverLandUse2016/FeatureServer/0/query")

to_utm = Transformer.from_crs(4326, 32619, always_xy=True)
c = SITES["points"][SITES["center"]]
hx, hy = to_utm.transform(c["lon"], c["lat"])
bounds = (hx - HALF, hy - HALF, hx + HALF, hy + HALF)

# --- rasters -------------------------------------------------------------- #
with rasterio.open("tessera/host_proba.tif") as P, rasterio.open("s2/oak_index.tif") as O:
    win = rasterio.windows.from_bounds(*bounds, P.transform).round_offsets().round_lengths()
    tf = rasterio.windows.transform(win, P.transform)
    prob = P.read(window=win).astype("float32")
    oak = O.read(1, window=win)
prob[prob == 255] = np.nan
canopy = np.isfinite(oak)
pwet = np.where(canopy, prob[2], np.nan)
pdec = np.where(canopy, prob[0], np.nan)
h, w = pwet.shape
xs = tf.c + (np.arange(w) + 0.5) * tf.a
ys = tf.f + (np.arange(h) + 0.5) * tf.e
# Map layers are drawn in web-Mercator metres (see the CRS note at `common`).
wet_da = (xr.DataArray(np.where(pwet >= 30, pwet, np.nan), coords={"y": ys, "x": xs},
                       dims=("y", "x"), name="P_wet")
          .rio.write_crs(32619).rio.write_nodata(np.nan)
          .rio.reproject(3857, resampling=rasterio.enums.Resampling.nearest))

# --- swale polygons ------------------------------------------------------- #
m = np.nan_to_num(pwet) >= SWALE_P
m = ndimage.binary_opening(m, iterations=1)    # drop single-pixel specks
m = ndimage.binary_closing(m, iterations=2)    # join pixels split by a gap
lab, _ = ndimage.label(m)
polys = []
for geom, val in shapes(lab.astype("int32"), mask=m, transform=tf):
    g = shape(geom).buffer(10).buffer(-10).simplify(5)
    if g.area < MIN_AREA:
        continue
    pix = lab == int(val)
    polys.append({"area_ha": round(g.area / 1e4, 2),
                  "mean_P_wet": int(np.nanmean(pwet[pix])),
                  "mean_P_decid": int(np.nanmean(pdec[pix])), "geometry": g})
swales = gpd.GeoDataFrame(polys, crs=32619)

# --- MassGIS 2016 forested wetland, all sizes, this window ---------------- #
r = requests.get(LCLU, timeout=300, params=dict(
    where="COVERCODE=13", geometry=",".join(map(str, bounds)),
    geometryType="esriGeometryEnvelope", inSR=32619, outSR=32619,
    spatialRel="esriSpatialRelIntersects", outFields="COVERNAME", f="geojson")).json()
mgis = gpd.GeoDataFrame.from_features(r["features"], crs=32619) if r.get("features") else None

# --- points ---------------------------------------------------------------- #
pts = {nm: (v["lat"], v["lon"]) for nm, v in SITES["points"].items()}
rows = []
for nm, (la, lo) in pts.items():
    x, y = to_utm.transform(lo, la)
    r_, c_ = rasterio.transform.rowcol(tf, x, y)
    rr, cc = np.ogrid[-5:6, -5:6]
    disk = rr ** 2 + cc ** 2 <= 25
    win_wet = pwet[r_ - 5:r_ + 6, c_ - 5:c_ + 6][disk]
    win_dec = pdec[r_ - 5:r_ + 6, c_ - 5:c_ + 6][disk]
    d = swales.distance(Point(x, y)).min() if len(swales) else np.nan
    rows.append({"name": nm, "P_wet": int(np.nanmedian(win_wet)) if np.isfinite(win_wet).any() else -1,
                 "P_decid": int(np.nanmedian(win_dec)) if np.isfinite(win_dec).any() else -1,
                 "to_swale_m": int(d), "geometry": Point(x, y)})
points = gpd.GeoDataFrame(rows, crs=32619)

print(f"{len(swales)} likely-swale patches >= {MIN_AREA / 1e4:.1f} ha in the window")
if "transect" in SITES:
    # Is there likely-wet ground on the straight line between two points?
    a, b = SITES["transect"]
    geo = points.set_index("name").geometry
    line = LineString([geo[a], geo[b]])
    crossing = swales[swales.intersects(line)]
    print(f"{a} -> {b}: {line.length:.0f} m; crosses {len(crossing)} patch(es)"
          + (f", wet over {line.intersection(crossing.union_all()).length:.0f} m of it"
             if len(crossing) else ""))
print(points.drop(columns="geometry").to_string(index=False))

# --- map ------------------------------------------------------------------ #
# geo=True goes through geoviews, which (1.15) reads proj4_params['lon_0'] --
# a key cartopy 0.26's PlateCarree no longer has, so every geo plot raises
# KeyError. Everything is reprojected to web Mercator (the tiles' own CRS) up
# front and drawn as plain x/y over the tiles instead.
common = dict(frame_width=900, frame_height=800, data_aspect=1, xaxis=None, yaxis=None)
raster = wet_da.hvplot.image(
    x="x", y="y", cmap="Blues", clim=(30, 100), alpha=0.55,
    clabel="P(forested wetland) %", tiles="EsriImagery", rasterize=False,
    hover_tooltips=[("P(forested wetland)", "@image{0} %")], **common,
    title=SITES.get("title", "TESSERA: likely wet ground (2025 embeddings)"))
sw = swales.to_crs(3857).hvplot(
    color=None, fill_alpha=0, line_color="cyan", line_width=3, label="Likely wet swale (TESSERA ≥ 50 %)",
    hover_cols=["area_ha", "mean_P_wet", "mean_P_decid"],
    hover_tooltips=[("Likely wet swale", ""), ("Area", "@area_ha ha"),
                    ("Mean P(wetland)", "@mean_P_wet %"), ("Mean P(deciduous)", "@mean_P_decid %")],
    **common)
overlay = raster * sw
if mgis is not None and len(mgis):
    overlay = overlay * mgis.to_crs(3857).hvplot(
        color=None, fill_alpha=0, line_color="yellow", line_width=2, line_dash="dashed",
        label="MassGIS 2016 forested wetland", hover=False, **common)
pts_ll = points.to_crs(3857).assign(x=lambda d: d.geometry.x, y=lambda d: d.geometry.y)
# Hand-placed label offsets (m) from local_sites.json: the points are tens of
# metres apart and labels at one fixed offset collide.
LABEL_OFF = {nm: (v.get("label_dx", 25), v.get("label_dy", 25)) for nm, v in SITES["points"].items()}
LABEL_ALIGN = {nm: "right" for nm, v in SITES["points"].items() if v.get("align") == "right"}
labels_df = pts_ll.drop(columns="geometry").assign(
    lx=lambda d: d.x + d.name.map(lambda n: LABEL_OFF[n][0]),
    ly=lambda d: d.y + d.name.map(lambda n: LABEL_OFF[n][1]))
overlay = overlay * pts_ll.drop(columns="geometry").hvplot.points(
    x="x", y="y",
    color="red", marker="star", line_color="white", label="Reference points",
    hover_cols=["name", "P_wet", "P_decid", "to_swale_m"],
    hover_tooltips=[("", "@name"), ("P(wetland), 50 m", "@P_wet %"),
                    ("P(deciduous), 50 m", "@P_decid %"), ("Nearest swale", "@to_swale_m m")],
    **common).opts(size=22) * labels_df[labels_df.name.map(LABEL_ALIGN.get).isna()].hvplot.labels(
    x="lx", y="ly", text="name", text_color="white", text_font_size="12pt",
    text_align="left", hover=False, **common) * labels_df[labels_df.name.map(LABEL_ALIGN.get).notna()].hvplot.labels(
    x="lx", y="ly", text="name", text_color="white", text_font_size="12pt",
    text_align="right", hover=False, **common)
# Open zoomed on the reference points; the layers extend to the full window
# for panning out.
cx, cy = pts_ll.x.mean(), pts_ll.y.mean()
overlay = overlay.opts(legend_position="top_left", active_tools=["wheel_zoom"],
                       xlim=(cx - 650, cx + 650), ylim=(cy - 580, cy + 580))
hvplot.save(overlay, OUT)
print(f"wrote {OUT}")
