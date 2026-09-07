"""
Standalone interactive map of the fungal-host forest layers.

Purpose is ground-truthing: pull the map up over ground you have actually
walked, flip to the satellite basemap, and see whether the modeled oak lines up
with the canopy you recognize. Everything the rasters say is a *model* -- the
FHP layer is a per-pixel statistical fit circa 2002 and BIGMAP a 2018
classification -- so this check is the go/no-go on building a foraging map on
top of them.

Reads the GeoTIFFs written by pull_forest_hosts.py, reprojects them to Web
Mercator, and bakes them into one self-contained HTML file: no server, no tile
pyramid, nothing to host. Leaflet image overlays keep the full 30 m detail, and
a hidden single-channel copy of each continuous layer is read back through a
canvas so clicking the map reports the actual basal area under the cursor.

Outputs:
  forest/buzzards_bay_forest_map.html

Run in the `protocoast-notebook` conda env.
"""

import base64
import io
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
from matplotlib import colormaps
from PIL import Image
from pyproj import Transformer

OUTDIR = Path("forest")
WATERSHEDS = ["buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson"]
SRC_CRS = "EPSG:32619"
OUT_HTML = OUTDIR / "buzzards_bay_forest_map.html"

# Continuous basal-area layers: (key, file, label, colormap, display max).
# Display max is the basal area at which the colour ramp saturates -- roughly
# the 99th percentile of stocked pixels, so the ramp spends its range where the
# data actually lives instead of on a few outliers.
CONTINUOUS = [
    ("oak", "fhp_ba_oak_deciduous_spp.tif", "Oak (all Quercus)", "YlGn", 80),
    ("white_oak", "fhp_ba_white_oak.tif", "White oak", "YlGn", 50),
    ("black_oak", "fhp_ba_black_oak.tif", "Black oak", "YlGn", 50),
    ("scarlet_oak", "fhp_ba_scarlet_oak.tif", "Scarlet oak", "YlGn", 50),
    ("pitch_pine", "fhp_ba_pitch_pine.tif", "Pitch pine", "Oranges", 50),
    ("white_pine", "fhp_ba_eastern_white_pine.tif", "Eastern white pine", "Oranges", 80),
    ("red_maple", "fhp_ba_red_maple.tif", "Red maple", "BuPu", 80),
    ("black_cherry", "fhp_ba_black_cherry.tif", "Black cherry", "BuPu", 40),
]

# BIGMAP forest type group -> (colour, label). Codes absent here render
# transparent, which for this domain means non-forest.
FTG_STYLE = {
    500: ("#1b5e20", "Oak / hickory"),
    400: ("#7fb069", "Oak / pine"),
    100: ("#8c6d3f", "White / red / jack pine"),
    700: ("#4f9bbf", "Elm / ash / cottonwood"),
    800: ("#b07aa1", "Maple / beech / birch"),
}


def data_uri(img):
    """PIL image -> base64 PNG data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def paletted(index, palette):
    """Indexed PNG with index 0 transparent.

    These overlays are a colour ramp over 8 Mpx, so a full RGBA PNG spends most
    of its bytes re-encoding 256 distinct colours. An indexed image with the
    ramp as its palette is the same picture at roughly a third the size, which
    matters when the whole map has to travel as one self-contained file.
    """
    img = Image.fromarray(index, mode="P")
    img.putpalette(palette)
    img.info["transparency"] = bytes([0] + [255] * 255)
    return img


def to_mercator(path, geom):
    """Load a GeoTIFF, clip to the basins, reproject to Web Mercator."""
    da = rioxarray.open_rasterio(OUTDIR / path).squeeze(drop=True).astype("float32")
    da.rio.write_crs(SRC_CRS, inplace=True)
    da = da.where(da < 65535)
    da.rio.write_nodata(np.nan, inplace=True)
    da = da.rio.clip([geom], SRC_CRS, drop=False)
    return da.rio.reproject("EPSG:3857")


def bounds_latlon(da):
    """Leaflet wants the overlay's corners as lat/lon."""
    x0, y0, x1, y1 = da.rio.bounds()
    tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon0, lat0 = tr.transform(x0, y0)
    lon1, lat1 = tr.transform(x1, y1)
    return [[lat0, lon0], [lat1, lon1]]


def main():
    parts = [gpd.read_file(f).to_crs(SRC_CRS) for f in WATERSHEDS]
    basins = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=SRC_CRS)
    geom = basins.geometry.union_all()

    layers, bounds = {}, None

    # --- continuous basal-area layers ------------------------------------- #
    for key, fname, label, cmap_name, vmax in CONTINUOUS:
        da = to_mercator(fname, geom)
        v = da.values
        if bounds is None:
            bounds = bounds_latlon(da)

        # Zero basal area is "no such tree here", not a low value -- render it
        # transparent so the basemap shows through rather than painting the
        # whole basin in the bottom colour of the ramp.
        present = np.isfinite(v) & (v > 0)
        norm = np.clip(np.nan_to_num(v) / vmax, 0, 1)

        cmap = colormaps[cmap_name]
        palette = bytes(
            b for t in np.linspace(0, 1, 255)
            for b in (int(c * 255) for c in cmap(t)[:3])
        )
        index = np.where(present, 1 + np.round(norm * 254), 0).astype("uint8")
        img = paletted(index, bytes(3) + palette)

        # Greyscale copy for click-query: basal area in sq ft/acre fits in a
        # byte across this domain, so the raw value is the pixel value. Zero
        # doubles as "none mapped here" -- the same thing for our purposes.
        value = Image.fromarray(
            np.clip(np.nan_to_num(v), 0, 255).astype("uint8"), mode="L"
        )

        layers[key] = {
            "label": label,
            "kind": "continuous",
            "vmax": vmax,
            "cmap": [
                "#%02x%02x%02x" % tuple(int(c * 255) for c in cmap(t)[:3])
                for t in np.linspace(0, 1, 9)
            ],
            "img": data_uri(img),
            "data": data_uri(value),
        }
        print(f"  {label:24s} {index.shape[1]}x{index.shape[0]}  "
              f"{len(layers[key]['img']) / 1e6:4.1f} MB img  "
              f"{len(layers[key]['data']) / 1e6:4.1f} MB values")

    # --- BIGMAP forest type group (categorical) --------------------------- #
    ftg = to_mercator("bigmap_forest_type_group.tif", geom)
    fv = np.nan_to_num(ftg.values).astype(int)
    index = np.zeros(fv.shape, dtype="uint8")
    palette = bytearray(3)  # index 0 = transparent
    for i, (code, (hexcol, _)) in enumerate(FTG_STYLE.items(), start=1):
        index[fv == code] = i
        palette += bytes.fromhex(hexcol[1:])
    layers["ftg"] = {
        "label": "Forest type group (BIGMAP 2018)",
        "kind": "categorical",
        "classes": [{"color": c, "label": l} for c, l in FTG_STYLE.values()],
        "img": data_uri(paletted(index, bytes(palette))),
        "data": None,
    }
    print(f"  {'Forest type group':24s} {index.shape[1]}x{index.shape[0]}  "
          f"{len(layers['ftg']['img']) / 1e6:4.1f} MB img")

    outline = json.loads(basins.to_crs(4326).dissolve().to_json())

    html = HTML_TEMPLATE.replace("__LAYERS__", json.dumps(layers)) \
                        .replace("__BOUNDS__", json.dumps(bounds)) \
                        .replace("__OUTLINE__", json.dumps(outline))
    OUT_HTML.write_text(html)
    print(f"\nwrote {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fungal host forest layers - Buzzards Bay &amp; Cape Cod</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  html, body { margin:0; height:100%; font:13px/1.45 system-ui, -apple-system, sans-serif; }
  #map { position:absolute; inset:0; }
  .panel {
    position:absolute; top:10px; right:10px; z-index:1000; width:250px;
    background:rgba(255,255,255,.95); border-radius:8px; padding:12px 14px;
    box-shadow:0 2px 12px rgba(0,0,0,.28); max-height:calc(100% - 20px);
    overflow-y:auto;
  }
  .panel h3 { margin:0 0 8px; font-size:13px; }
  .panel h4 { margin:12px 0 5px; font-size:11px; text-transform:uppercase;
              letter-spacing:.04em; color:#666; font-weight:600; }
  .panel label { display:block; padding:2px 0; cursor:pointer; }
  .panel input[type=radio] { margin-right:6px; }
  #opacity { width:100%; }
  .ramp { height:11px; border-radius:2px; margin:5px 0 2px; }
  .ticks { display:flex; justify-content:space-between; font-size:10px; color:#666; }
  .swatch { display:inline-block; width:13px; height:13px; border-radius:2px;
            vertical-align:-2px; margin-right:6px; border:1px solid rgba(0,0,0,.2); }
  .cls { padding:1px 0; font-size:12px; }
  .readout { margin-top:10px; padding-top:9px; border-top:1px solid #ddd;
             font-size:12px; color:#333; min-height:32px; }
  .readout b { font-size:16px; }
  .note { margin-top:10px; padding-top:9px; border-top:1px solid #ddd;
          font-size:10.5px; color:#777; line-height:1.4; }
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h3>Fungal host layers</h3>
  <h4>Layer</h4>
  <div id="layers"></div>
  <h4>Opacity</h4>
  <input type="range" id="opacity" min="0" max="100" value="80">
  <div id="legend"></div>
  <div class="readout" id="readout">Click the map to read a value.</div>
  <div class="note">
    Basal area in sq ft/acre. Oak &amp; companion species: USFS FHP, 30 m,
    modeled circa 2002. Forest type group: USFS FIA BIGMAP, 30 m, 2018.
    Both predate or straddle the 2016&ndash;18 spongy moth oak mortality.
  </div>
</div>
<script>
const LAYERS  = __LAYERS__;
const BOUNDS  = __BOUNDS__;
const OUTLINE = __OUTLINE__;

const map = L.map('map');
map.fitBounds(BOUNDS);

const bases = {
  'Satellite': L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    {maxZoom:19, attribution:'Esri, Maxar, Earthstar Geographics'}),
  'Topographic': L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    {maxZoom:19, attribution:'Esri'}),
  'OpenStreetMap': L.tileLayer(
    'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    {maxZoom:19, attribution:'&copy; OpenStreetMap contributors'})
};
bases['Satellite'].addTo(map);
L.control.layers(bases, null, {position:'topleft'}).addTo(map);

L.geoJSON(OUTLINE, {style:{color:'#ffdd00', weight:2, fill:false}}).addTo(map);

// --- overlay ------------------------------------------------------------- //
let current = 'oak';
let overlay = null;

function draw() {
  if (overlay) map.removeLayer(overlay);
  overlay = L.imageOverlay(LAYERS[current].img, BOUNDS, {
    opacity: document.getElementById('opacity').value / 100,
    interactive: false
  }).addTo(map);
  legend();
}

function legend() {
  const L_ = LAYERS[current], el = document.getElementById('legend');
  if (L_.kind === 'continuous') {
    el.innerHTML =
      `<h4>${L_.label}</h4>` +
      `<div class="ramp" style="background:linear-gradient(90deg,${L_.cmap.join(',')})"></div>` +
      `<div class="ticks"><span>0</span><span>${L_.vmax/2}</span>` +
      `<span>${L_.vmax}+ sq ft/ac</span></div>`;
  } else {
    el.innerHTML = `<h4>${L_.label}</h4>` + L_.classes.map(c =>
      `<div class="cls"><span class="swatch" style="background:${c.color}"></span>${c.label}</div>`
    ).join('');
  }
}

const box = document.getElementById('layers');
box.innerHTML = Object.keys(LAYERS).map(k =>
  `<label><input type="radio" name="lyr" value="${k}"` +
  `${k === current ? ' checked' : ''}>${LAYERS[k].label}</label>`).join('');
box.addEventListener('change', e => { current = e.target.value; draw(); read(null); });
document.getElementById('opacity').addEventListener('input', e => {
  if (overlay) overlay.setOpacity(e.target.value / 100);
});

// --- click-to-read ------------------------------------------------------- //
// The value rasters ride in the red channel of a hidden PNG; decode once per
// layer into a canvas, then a click is just a pixel lookup.
const canvases = {};
function valueCanvas(key, cb) {
  if (canvases[key]) return cb(canvases[key]);
  const src = LAYERS[key].data;
  if (!src) return cb(null);
  const im = new Image();
  im.onload = () => {
    const c = document.createElement('canvas');
    c.width = im.width; c.height = im.height;
    c.getContext('2d', {willReadFrequently:true}).drawImage(im, 0, 0);
    canvases[key] = c;
    cb(c);
  };
  im.src = src;
}

function read(latlng) {
  const out = document.getElementById('readout');
  if (!latlng) { out.textContent = 'Click the map to read a value.'; return; }
  valueCanvas(current, c => {
    if (!c) { out.textContent = 'No value readout for this layer.'; return; }
    // BOUNDS is [[south, west], [north, east]]; the image is Web Mercator, so
    // interpolate in projected y, not latitude.
    const [[s, w], [n, e]] = BOUNDS;
    const merc = lat => Math.log(Math.tan(Math.PI/4 + lat*Math.PI/360));
    const px = Math.floor((latlng.lng - w) / (e - w) * c.width);
    const py = Math.floor((merc(n) - merc(latlng.lat)) / (merc(n) - merc(s)) * c.height);
    if (px < 0 || py < 0 || px >= c.width || py >= c.height) {
      out.textContent = 'Outside the mapped area.'; return;
    }
    const v = c.getContext('2d').getImageData(px, py, 1, 1).data[0];
    const where = `<br><span style="color:#888">` +
                  `${latlng.lat.toFixed(5)}, ${latlng.lng.toFixed(5)}</span>`;
    const name = LAYERS[current].label.toLowerCase();
    out.innerHTML = v === 0
      ? `<b>0</b> sq ft/ac &mdash; no ${name} mapped here` + where
      : `<b>${v}</b> sq ft/ac ${name} basal area` + where;
  });
}
map.on('click', e => read(e.latlng));

draw();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
