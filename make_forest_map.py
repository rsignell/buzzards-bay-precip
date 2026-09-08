"""
Standalone interactive map of the host layers, for ground-truthing by eye.

Pull it up over ground you have walked, flip to the satellite basemap, and see
whether the layers match the canopy you recognise. That check is what the whole
foraging model rests on.

The headline layer is now the Sentinel-2 deciduous index, which measures
leaf-on/leaf-off phenology directly instead of inheriting anyone's land-cover
classification. The USFS layers are kept alongside it deliberately, because
over this domain they are a cautionary tale rather than a reference: FHP
reports zero basal area for every species across a confirmed 50-acre oak
forest, and flipping between the two layers shows exactly where it gives up.

Reads GeoTIFFs from forest/ (30 m) and s2/ (10 m), reprojects to Web Mercator
and bakes everything into one self-contained HTML file: no server, no tile
pyramid, nothing to host. A hidden greyscale twin of each continuous layer is
read back through a canvas so clicking reports real values.

Sentinel-2 layers are written at 20 m rather than their native 10 m. That is a
display decision only -- the analysis rasters stay at 10 m -- and it keeps the
whole map near 10 MB so it still opens on a phone.

Outputs:
  forest/buzzards_bay_forest_map.html

Run on the oak-mapping cluster (it has both forest/ and s2/):
  /home/ubuntu/miniconda3/bin/python make_forest_map.py
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

WATERSHEDS = ["buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson"]
SRC_CRS = "EPSG:32619"
OUT_HTML = Path("forest") / "buzzards_bay_forest_map.html"

S2_DISPLAY_RES = 20  # metres; see module docstring

# key, path, label, colormap, vmin, vmax, units, decimals, levels, clickable
#
# `levels` is the display quantisation. The FHP layers are mostly transparent
# and compress to almost nothing at full depth, but the Sentinel-2 fields are
# smooth and dense and cost ~6 MB each at 255 levels. Dropping them to 64 costs
# nothing legible on a colour ramp and roughly halves the file. `clickable`
# adds the hidden greyscale value twin, which doubles a layer's cost -- worth it
# for the two layers whose numbers you actually want to read, not for the rest.
CONTINUOUS = [
    ("decid", "s2/s2_decid.tif",
     "Oak / deciduous index (Sentinel-2 2026)", "RdYlGn", -0.10, 0.60, "", 3,
     64, True),
    ("summer", "s2/s2_ndvi_summer.tif",
     "Summer canopy greenness (NDVI)", "YlGn", 0.20, 0.95, "", 3, 64, False),
    ("fhp_oak", "forest/fhp_ba_oak_deciduous_spp.tif",
     "Oak basal area (USFS FHP ~2002)", "YlGn", 0, 80, " sq ft/ac", 0,
     255, True),
    ("fhp_pine", "forest/fhp_ba_pitch_pine.tif",
     "Pitch pine basal area (USFS FHP ~2002)", "Oranges", 0, 50, " sq ft/ac", 0,
     255, False),
]

FTG_STYLE = {
    500: ("#1b5e20", "Oak / hickory"),
    400: ("#7fb069", "Oak / pine"),
    100: ("#8c6d3f", "White / red / jack pine"),
    700: ("#4f9bbf", "Elm / ash / cottonwood"),
    800: ("#b07aa1", "Maple / beech / birch"),
}

# Marks worth keeping on the map: the ground truth this was validated against.
PINS = [
    (41.72656, -70.60387, "Confirmed oak forest (~50 ac) — index 0.30"),
    (41.7266604, -70.5966575, "Cranberry bog — index 0.12 (evergreen)"),
    (41.72572, -70.53661, "Oak/hickory control — index 0.43"),
    (41.90640, -70.69646, "Pitch pine control — index 0.09"),
]


def data_uri(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def paletted(index, palette):
    """Indexed PNG with index 0 transparent.

    A colour ramp over tens of Mpx spends most of an RGBA PNG's bytes
    re-encoding 256 colours; an indexed image with the ramp as its palette is
    the same picture at roughly a third the size.
    """
    img = Image.fromarray(index, mode="P")
    img.putpalette(palette)
    img.info["transparency"] = bytes([0] + [255] * 255)
    return img


def prepare(path, geom, res=None):
    """Load, clip to the basins, optionally coarsen, reproject to Web Mercator."""
    da = rioxarray.open_rasterio(path).squeeze(drop=True).astype("float32")
    da = da.where(da < 65535)
    da.rio.write_crs(SRC_CRS, inplace=True)
    da.rio.write_nodata(np.nan, inplace=True)
    if res:
        native = abs(float(da.x[1] - da.x[0]))
        factor = int(round(res / native))
        if factor > 1:
            da = da.coarsen(x=factor, y=factor, boundary="trim").mean()
    da = da.rio.clip([geom], SRC_CRS, drop=False)
    return da.rio.reproject("EPSG:3857")


def bounds_latlon(da):
    x0, y0, x1, y1 = da.rio.bounds()
    tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon0, lat0 = tr.transform(x0, y0)
    lon1, lat1 = tr.transform(x1, y1)
    return [[lat0, lon0], [lat1, lon1]]


def main():
    parts = [gpd.read_file(f).to_crs(SRC_CRS) for f in WATERSHEDS]
    basins = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=SRC_CRS)
    geom = basins.geometry.union_all()

    layers = {}
    for (key, path, label, cmap_name, vmin, vmax, units, dec,
         levels, clickable) in CONTINUOUS:
        if not Path(path).exists():
            print(f"  {label:42s} SKIP (missing {path})")
            continue
        res = S2_DISPLAY_RES if path.startswith("s2/") else None
        da = prepare(path, geom, res)
        v = da.values

        present = np.isfinite(v)
        norm = np.clip((v - vmin) / (vmax - vmin), 0, 1)

        cmap = colormaps[cmap_name]
        palette = bytes(b for t in np.linspace(0, 1, levels)
                        for b in (int(c * 255) for c in cmap(t)[:3]))
        palette += bytes(3 * (255 - levels))  # pad to a full 256-entry palette
        index = np.where(present, 1 + np.round(np.nan_to_num(norm) * (levels - 1)), 0) \
            .astype("uint8")

        layers[key] = {
            "label": label, "kind": "continuous",
            "vmin": vmin, "vmax": vmax, "units": units, "decimals": dec,
            "cmap": ["#%02x%02x%02x" % tuple(int(c * 255) for c in cmap(t)[:3])
                     for t in np.linspace(0, 1, 9)],
            "bounds": bounds_latlon(da),
            "img": data_uri(paletted(index, bytes(3) + palette)),
            "data": None,
        }
        if clickable:
            # Value twin: the same normalisation quantised to a byte, decoded
            # back to physical units in the browser using vmin/vmax. Carried at
            # half the display resolution -- the JS locates a pixel by fraction
            # of canvas size, so a coarser twin just works, and a click is
            # asking "what is it around here" rather than naming one cell.
            #
            # Averaged in 2x2 blocks, not decimated. Decimating shifts the
            # sample by half a cell, which is invisible where the field is
            # smooth but put the pine control out by 0.095 -- exactly where the
            # index has a steep local gradient and you would most want to trust
            # the number.
            n = np.where(present, norm, np.nan)
            h, w = (s // 2 * 2 for s in n.shape)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(
                    n[:h, :w].reshape(h // 2, 2, w // 2, 2), axis=(1, 3))
            twin = np.where(np.isfinite(mean),
                            np.round(np.nan_to_num(mean) * 255), 0).astype("uint8")
            layers[key]["data"] = data_uri(Image.fromarray(twin, mode="L"))
        val_mb = len(layers[key]["data"]) / 1e6 if clickable else 0.0
        print(f"  {label:42s} {index.shape[1]}x{index.shape[0]}  "
              f"{len(layers[key]['img']) / 1e6:4.1f} MB img "
              f"{val_mb:4.1f} MB val")

    ftg = prepare("forest/bigmap_forest_type_group.tif", geom)
    fv = np.nan_to_num(ftg.values).astype(int)
    index = np.zeros(fv.shape, dtype="uint8")
    palette = bytearray(3)
    for i, (code, (hexcol, _)) in enumerate(FTG_STYLE.items(), start=1):
        index[fv == code] = i
        palette += bytes.fromhex(hexcol[1:])
    layers["ftg"] = {
        "label": "Forest type group (BIGMAP 2018)", "kind": "categorical",
        "classes": [{"color": c, "label": l} for c, l in FTG_STYLE.values()],
        "bounds": bounds_latlon(ftg),
        "img": data_uri(paletted(index, bytes(palette))), "data": None,
    }
    print(f"  {'Forest type group (BIGMAP 2018)':42s} {index.shape[1]}x{index.shape[0]}  "
          f"{len(layers['ftg']['img']) / 1e6:4.1f} MB img")

    outline = json.loads(basins.to_crs(4326).dissolve().to_json())
    html = (HTML_TEMPLATE
            .replace("__LAYERS__", json.dumps(layers))
            .replace("__OUTLINE__", json.dumps(outline))
            .replace("__PINS__", json.dumps(PINS)))
    OUT_HTML.parent.mkdir(exist_ok=True)
    OUT_HTML.write_text(html)
    print(f"\nwrote {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oak &amp; host layers — Buzzards Bay &amp; Cape Cod</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  html, body { margin:0; height:100%; font:13px/1.45 system-ui, -apple-system, sans-serif; }
  #map { position:absolute; inset:0; }
  .panel {
    position:absolute; top:10px; right:10px; z-index:1000; width:266px;
    background:rgba(255,255,255,.96); border-radius:8px; padding:12px 14px;
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
             font-size:12px; color:#333; min-height:34px; }
  .readout b { font-size:17px; }
  .note { margin-top:10px; padding-top:9px; border-top:1px solid #ddd;
          font-size:10.5px; color:#777; line-height:1.4; }
  .warn { color:#b34700; }
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h3>Oak &amp; host layers</h3>
  <h4>Layer</h4>
  <div id="layers"></div>
  <h4>Opacity</h4>
  <input type="range" id="opacity" min="0" max="100" value="80">
  <div id="legend"></div>
  <div class="readout" id="readout">Click the map to read a value.</div>
  <div class="note">
    <b>Deciduous index</b> = summer NDVI &minus; leaf-off NDVI, Sentinel-2
    2026, analysed at 10 m and drawn at 20 m. Calibrated on ~50-acre circles:
    pitch pine 0.09, cranberry bog 0.12, a confirmed oak forest 0.30, a pure
    Oak/hickory stand 0.43. Cranberry is evergreen, so bogs read low.
    <br><br>
    <span class="warn">USFS FHP (~2002) is shown for comparison only. Over this
    domain it reports zero basal area for every species across a confirmed
    50-acre oak forest — do not trust it here.</span>
  </div>
</div>
<script>
const LAYERS  = __LAYERS__;
const OUTLINE = __OUTLINE__;
const PINS    = __PINS__;

const map = L.map('map');
map.fitBounds(LAYERS[Object.keys(LAYERS)[0]].bounds);

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

PINS.forEach(([lat, lon, text]) => {
  L.circleMarker([lat, lon], {radius:7, color:'#00e5ff', weight:2.5,
    fillColor:'#003b46', fillOpacity:.65}).addTo(map).bindPopup(text);
});

let current = Object.keys(LAYERS)[0];
let overlay = null;

function draw() {
  if (overlay) map.removeLayer(overlay);
  overlay = L.imageOverlay(LAYERS[current].img, LAYERS[current].bounds, {
    opacity: document.getElementById('opacity').value / 100,
    interactive: false
  }).addTo(map);
  legend();
}

function fmt(L_, v) {
  return v.toFixed(L_.decimals) + L_.units;
}

function legend() {
  const L_ = LAYERS[current], el = document.getElementById('legend');
  if (L_.kind === 'continuous') {
    el.innerHTML =
      `<h4>${L_.label}</h4>` +
      `<div class="ramp" style="background:linear-gradient(90deg,${L_.cmap.join(',')})"></div>` +
      `<div class="ticks"><span>${fmt(L_, L_.vmin)}</span>` +
      `<span>${fmt(L_, (L_.vmin + L_.vmax) / 2)}</span>` +
      `<span>${fmt(L_, L_.vmax)}+</span></div>`;
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
  const L_ = LAYERS[current];
  valueCanvas(current, c => {
    if (!c) { out.textContent = 'No value readout for this layer.'; return; }
    // bounds is [[south, west], [north, east]]; the image is Web Mercator, so
    // interpolate in projected y, not latitude.
    const [[s, w], [n, e]] = L_.bounds;
    const merc = lat => Math.log(Math.tan(Math.PI/4 + lat*Math.PI/360));
    const px = Math.floor((latlng.lng - w) / (e - w) * c.width);
    const py = Math.floor((merc(n) - merc(latlng.lat)) / (merc(n) - merc(s)) * c.height);
    const where = `<br><span style="color:#888">` +
                  `${latlng.lat.toFixed(5)}, ${latlng.lng.toFixed(5)}</span>`;
    if (px < 0 || py < 0 || px >= c.width || py >= c.height) {
      out.innerHTML = 'Outside the mapped area.' + where; return;
    }
    const d = c.getContext('2d').getImageData(px, py, 1, 1).data;
    if (d[3] === 0) { out.innerHTML = 'No data here.' + where; return; }
    const v = L_.vmin + (d[0] / 255) * (L_.vmax - L_.vmin);
    out.innerHTML = `<b>${fmt(L_, v)}</b> ${L_.label.toLowerCase()}` + where;
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
