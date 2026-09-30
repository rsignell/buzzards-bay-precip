"""
The foraging app: pick a species, see where conditions are favorable today.

One self-contained HTML file. Pick one of the five species on the curated list
and the map shows favorable / marginal / unfavorable across every rateable acre
of closed canopy in the Buzzards Bay and Cape Cod watersheds, from the
Sentinel-2 oak layer, SSURGO soils and the MRMS-driven soil-moisture model.

The second view is the one that earns its keep: for anywhere not favorable, it
shows *which* factor is holding it back. "Too dry" is worth waiting out, "no
oak" never will be, "wrong month" tells you when to come back, and "rained too
recently" means the water is there but the species' own fruiting lag hasn't
passed yet. That falls straight out of scoring by Liebig minimum instead of
averaging.

Everything is baked in as indexed PNGs -- no server, no tiles. Class rasters are
three flat colours and compress to almost nothing; the continuous drivers ride
along as half-resolution greyscale twins so a click can report the actual oak
index, soil moisture, rain and days-since-soak under the cursor.

Outputs:
  app/buzzards_bay_foraging.html

Run on the oak-mapping cluster after score_species.py.
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
from rasterio.enums import Resampling

CRS = "EPSG:32619"
OUT = Path("app") / "buzzards_bay_foraging.html"
WATERSHEDS = ["buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson"]

CLASS_STYLE = {3: ("#1a8f3c", "Favorable"),
               2: ("#e8a33d", "Marginal"),
               1: ("#7d5a5a", "Unfavorable")}
LIMIT_STYLE = {1: ("#6b4c9a", "No / thin oak host"),
               2: ("#b5651d", "Wrong time of year"),
               3: ("#2b7fb8", "Too dry"),
               4: ("#3fa7a0", "Rained too recently -- give it a few days")}
# land/public_land.tif codes not-public as 1 and public as 2 (not 0/1) so that
# after reprojection, true nodata outside the basin -- which collapses to 0 in
# class_png -- stays distinguishable from "inside the basin but not public".
# Only code 1 gets a style entry, so public land (2) is left transparent.
PUBLIC_LAND_STYLE = {1: ("#4d4d4d", "Not public land")}

# key, path, label, vmin, vmax, units, decimals
CONTEXT = [
    ("oak", "s2/oak_index.tif", "oak index", 0.0, 0.60, "", 2),
    ("sm", "moisture/sm_now.tif", "soil moisture", 0.0, 1.0, "", 2),
    ("rain21", "moisture/rain_21.tif", "21-day rain", 0.0, 120.0, " mm", 0),
    # Range must cover the whole post-spin-up window (HISTORY - SPINUP = 60 d).
    # A tighter cap silently clips: a genuine 51-day dry spell displayed as 45.
    ("soak", "moisture/days_since_soak.tif", "days since 0.5\" rain",
     0.0, 60.0, " d", 0),
]


def data_uri(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def paletted(index, palette):
    img = Image.fromarray(index, mode="P")
    img.putpalette(palette + bytes(768 - len(palette)))
    img.info["transparency"] = bytes([0] + [255] * 255)
    return img


def prepare(path, geom, categorical=False, match=None):
    da = rioxarray.open_rasterio(path).squeeze(drop=True).astype("float32")
    if match is not None:
        # The oak index is native 10 m while everything else is on the 30 m
        # grid; left alone its value twin alone costs ~5 MB of the file.
        da.rio.write_crs(CRS, inplace=True)
        da = da.rio.reproject_match(match, resampling=Resampling.average)
    da.rio.write_crs(CRS, inplace=True)
    da.rio.write_nodata(np.nan, inplace=True)
    da = da.rio.clip([geom], CRS, drop=False)
    return da.rio.reproject(
        "EPSG:3857",
        resampling=Resampling.nearest if categorical else Resampling.bilinear)


def bounds_latlon(da):
    x0, y0, x1, y1 = da.rio.bounds()
    tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lo = tr.transform(x0, y0)
    hi = tr.transform(x1, y1)
    return [[lo[1], lo[0]], [hi[1], hi[0]]]


def class_png(da, style):
    v = np.rint(np.nan_to_num(da.values, nan=0)).astype(int)
    index = np.zeros(v.shape, dtype="uint8")
    palette = bytearray(3)
    for i, (code, (hexcol, _)) in enumerate(style.items(), start=1):
        index[v == code] = i
        palette += bytes.fromhex(hexcol[1:])
    return data_uri(paletted(index, bytes(palette)))


def twin(da, vmin, vmax):
    """Half-resolution greyscale value twin, block-averaged (not decimated:
    decimation shifts the sample half a cell and misreads steep gradients)."""
    v = da.values
    norm = np.clip((v - vmin) / (vmax - vmin), 0, 1)
    n = np.where(np.isfinite(v), norm, np.nan)
    h, w = (s // 2 * 2 for s in n.shape)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(n[:h, :w].reshape(h // 2, 2, w // 2, 2), axis=(1, 3))
    arr = np.where(np.isfinite(m), np.round(np.nan_to_num(m) * 254) + 1, 0)
    return data_uri(Image.fromarray(arr.astype("uint8"), mode="L"))


def main():
    summary = json.loads(Path("scores/summary.json").read_text())
    mmeta = json.loads(Path("moisture/meta.json").read_text())

    parts = [gpd.read_file(f).to_crs(CRS) for f in WATERSHEDS]
    basins = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS)
    geom = basins.geometry.union_all()

    species, bounds = {}, None
    for key, info in summary["species"].items():
        cls = prepare(f"scores/{key}_class.tif", geom, categorical=True)
        lim = prepare(f"scores/{key}_limiter.tif", geom, categorical=True)
        sc = prepare(f"scores/{key}_score.tif", geom)
        if bounds is None:
            bounds = bounds_latlon(cls)
        species[key] = {
            "label": info["label"], "note": info["note"],
            "season": info["season_score"], "phase": info["season_phase"],
            "window": info["window_days"], "lag": info["lag_days"],
            "areas": info["areas_km2"], "limiting": info["limiting_pct_of_canopy"],
            "cls": class_png(cls, CLASS_STYLE),
            "lim": class_png(lim, LIMIT_STYLE),
            "score": twin(sc, 0.0, 1.0),
        }
        print(f"  {info['label']:42s} "
              f"{len(species[key]['cls']) / 1e6:4.1f} + "
              f"{len(species[key]['lim']) / 1e6:4.1f} + "
              f"{len(species[key]['score']) / 1e6:4.1f} MB")

    pub = prepare("land/public_land.tif", geom, categorical=True)
    public_land_png = class_png(pub, PUBLIC_LAND_STYLE)
    print(f"  {'public land mask':42s} {len(public_land_png) / 1e6:4.1f} MB")

    grid30 = rioxarray.open_rasterio("moisture/sm_now.tif").squeeze(drop=True)
    context = {}
    for key, path, label, vmin, vmax, units, dec in CONTEXT:
        da = prepare(path, geom, match=grid30 if key == "oak" else None)
        context[key] = {"label": label, "vmin": vmin, "vmax": vmax,
                        "units": units, "decimals": dec, "twin": twin(da, vmin, vmax)}
        print(f"  {label:42s} {len(context[key]['twin']) / 1e6:4.1f} MB")

    payload = {
        "as_of": summary["as_of"],
        "canopy_km2": summary["rateable_canopy_km2"],
        "species": species,
        "context": context,
        "public_land": public_land_png,
        "bounds": bounds,
        "outline": json.loads(basins.to_crs(4326).dissolve().to_json()),
        "classes": [{"color": c, "label": l} for c, l in CLASS_STYLE.values()],
        "limits": [{"color": c, "label": l} for c, l in LIMIT_STYLE.values()],
        "moisture_meta": mmeta,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(TEMPLATE.replace("__DATA__", json.dumps(payload)))
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Buzzards Bay foraging conditions</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  html,body{margin:0;height:100%;font:13px/1.5 system-ui,-apple-system,sans-serif;}
  #map{position:absolute;inset:0;}
  .menu-btn{position:absolute;top:10px;right:10px;z-index:1001;width:38px;height:38px;
    border:none;border-radius:8px;background:rgba(255,255,255,.97);
    box-shadow:0 2px 10px rgba(0,0,0,.3);font-size:19px;line-height:1;cursor:pointer;
    display:flex;align-items:center;justify-content:center;padding:0;}
  .mini-bar{display:none;position:absolute;left:8px;right:8px;bottom:8px;z-index:999;
    background:rgba(255,255,255,.95);border-radius:10px;padding:8px 10px;
    box-shadow:0 2px 10px rgba(0,0,0,.3);}
  .mini-bar.show{display:block;}
  .mini-title{font-weight:700;font-size:12.5px;margin-bottom:5px;}
  .mini-legend{display:flex;flex-wrap:wrap;gap:9px;font-size:11px;}
  .mini-legend .item{display:flex;align-items:center;gap:4px;white-space:nowrap;}
  .mini-legend .sw{width:11px;height:11px;border-radius:3px;border:1px solid rgba(0,0,0,.25);flex:none;}
  .panel{display:none;position:absolute;top:56px;right:10px;z-index:1000;width:290px;
    max-width:calc(100vw - 20px);
    background:rgba(255,255,255,.97);border-radius:10px;padding:14px 16px;
    box-shadow:0 2px 16px rgba(0,0,0,.3);max-height:calc(100% - 66px);overflow-y:auto;}
  .panel.open{display:block;}
  h2{margin:0 0 2px;font-size:15px;}
  .asof{color:#666;font-size:11.5px;margin-bottom:12px;}
  h4{margin:14px 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.05em;
     color:#666;font-weight:700;}
  .sp{display:block;padding:5px 8px;margin:3px 0;border:1px solid #ddd;border-radius:6px;
      cursor:pointer;font-size:12.5px;}
  .sp.on{background:#eef6ee;border-color:#1a8f3c;font-weight:600;}
  .sp input{margin-right:7px;}
  .seg{display:flex;gap:4px;margin:4px 0 0;}
  .seg button{flex:1;padding:5px 4px;font-size:11.5px;border:1px solid #ccc;
    background:#fafafa;border-radius:5px;cursor:pointer;}
  .seg button.on{background:#333;color:#fff;border-color:#333;}
  #opacity{width:100%;}
  .row{display:flex;align-items:center;gap:7px;padding:2px 0;font-size:12px;}
  .sw{width:14px;height:14px;border-radius:3px;border:1px solid rgba(0,0,0,.25);flex:none;}
  .km{margin-left:auto;color:#666;font-variant-numeric:tabular-nums;}
  .note{font-size:11.5px;color:#555;margin-top:8px;line-height:1.45;}
  .readout{margin-top:12px;padding-top:10px;border-top:1px solid #ddd;font-size:12px;}
  .readout .big{font-size:15px;font-weight:700;}
  .readout table{width:100%;border-collapse:collapse;margin-top:6px;}
  .readout td{padding:1px 0;font-size:11.5px;}
  .readout td:last-child{text-align:right;font-variant-numeric:tabular-nums;color:#333;}
  .caveat{margin-top:12px;padding-top:10px;border-top:1px solid #ddd;
    font-size:10.5px;color:#888;line-height:1.4;}
</style></head><body>
<div id="map"></div>
<button id="menu-toggle" class="menu-btn" aria-label="Toggle menu">&#9776;</button>
<div class="mini-bar" id="miniBar">
  <div class="mini-title" id="miniTitle"></div>
  <div class="mini-legend" id="miniLegend"></div>
</div>
<div class="panel">
  <h2>Foraging conditions</h2>
  <div class="asof" id="asof"></div>
  <h4>Species</h4>
  <div id="picker"></div>
  <h4>Show</h4>
  <div class="seg">
    <button id="b-cls" class="on">Conditions</button>
    <button id="b-lim">Why not?</button>
  </div>
  <h4>Opacity</h4>
  <input type="range" id="opacity" min="0" max="100" value="78">
  <h4>Land</h4>
  <label class="row" style="cursor:pointer">
    <input type="checkbox" id="pub-toggle" style="margin-right:7px">
    <span class="sw" style="background:#4d4d4d"></span>
    Grey out non-public land
  </label>
  <div id="legend"></div>
  <div class="note" id="note"></div>
  <div class="readout" id="readout">Click the map for conditions at a point.</div>
  <div class="caveat" id="caveat"></div>
</div>
<script>
const D = __DATA__;
// Default to whichever species has the most favorable ground today, not
// just the first in the curated list -- that's the one worth opening on.
let sp = Object.keys(D.species).reduce((best, k) =>
  D.species[k].areas.favorable > D.species[best].areas.favorable ? k : best);
let view = "cls", overlay = null, pubOverlay = null;

const panelEl = document.querySelector('.panel');
const miniBar = document.getElementById('miniBar');
function syncMiniBar() { miniBar.classList.toggle('show', !panelEl.classList.contains('open')); }
if (window.innerWidth > 640) panelEl.classList.add('open');
syncMiniBar();
document.getElementById('menu-toggle').onclick = () => {
  panelEl.classList.toggle('open');
  syncMiniBar();
};

const map = L.map('map');
map.fitBounds(D.bounds);
const bases = {
 'Satellite': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
   {maxZoom:19, attribution:'Esri, Maxar'}),
 'Topographic': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
   {maxZoom:19, attribution:'Esri'}),
 'OpenStreetMap': L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
   {maxZoom:19, attribution:'&copy; OpenStreetMap'})
};
bases['Topographic'].addTo(map);
L.control.layers(bases, null, {position:'topleft'}).addTo(map);
L.geoJSON(D.outline, {style:{color:'#ffdd00',weight:2,fill:false}}).addTo(map);

document.getElementById('asof').textContent =
  `as of ${D.as_of} · ${D.canopy_km2.toLocaleString()} km² of rateable canopy`;

document.getElementById('picker').innerHTML = Object.entries(D.species).map(([k,v]) =>
  `<label class="sp${k===sp?' on':''}" data-k="${k}">
     <input type="radio" name="sp" value="${k}"${k===sp?' checked':''}>${v.label}</label>`
).join('');

document.getElementById('picker').addEventListener('change', e => {
  sp = e.target.value;
  document.querySelectorAll('.sp').forEach(el =>
    el.classList.toggle('on', el.dataset.k === sp));
  draw(); read(last);
});
for (const [id, v] of [['b-cls','cls'], ['b-lim','lim']]) {
  document.getElementById(id).onclick = () => {
    view = v;
    document.getElementById('b-cls').classList.toggle('on', v==='cls');
    document.getElementById('b-lim').classList.toggle('on', v==='lim');
    draw();
  };
}
document.getElementById('opacity').oninput = e => {
  if (overlay) overlay.setOpacity(e.target.value/100);
};
document.getElementById('pub-toggle').onchange = e => {
  if (e.target.checked) {
    pubOverlay = L.imageOverlay(D.public_land, D.bounds,
      {opacity:0.62, interactive:false}).addTo(map);
  } else if (pubOverlay) {
    map.removeLayer(pubOverlay);
    pubOverlay = null;
  }
};

function draw() {
  if (overlay) map.removeLayer(overlay);
  overlay = L.imageOverlay(D.species[sp][view], D.bounds,
    {opacity: document.getElementById('opacity').value/100, interactive:false}).addTo(map);
  if (pubOverlay) pubOverlay.bringToFront();
  legend();
}

function legend() {
  const s = D.species[sp], el = document.getElementById('legend');
  const miniItems = view === 'cls' ? D.classes : D.limits;
  document.getElementById('miniTitle').textContent =
    `${s.label} — ${view === 'cls' ? 'Conditions' : 'Why not?'}`;
  document.getElementById('miniLegend').innerHTML = miniItems.map(c =>
    `<div class="item"><span class="sw" style="background:${c.color}"></span>${c.label}</div>`
  ).join('');
  if (view === 'cls') {
    el.innerHTML = '<h4>Conditions</h4>' + D.classes.map(c =>
      `<div class="row"><span class="sw" style="background:${c.color}"></span>${c.label}
       <span class="km">${(s.areas[c.label.toLowerCase()]??0).toLocaleString()} km²</span></div>`
    ).join('');
  } else {
    el.innerHTML = '<h4>Limiting factor</h4>' + D.limits.map((c,i) => {
      const key = ['host','season','moisture','lag'][i];
      return `<div class="row"><span class="sw" style="background:${c.color}"></span>${c.label}
        <span class="km">${s.limiting[key]??0}%</span></div>`;
    }).join('') + '<div class="note">Percent of rateable canopy where this is '
      + 'the binding constraint. Favorable ground is not shaded.</div>';
  }
  const phase = s.phase.charAt(0).toUpperCase() + s.phase.slice(1);
  document.getElementById('note').innerHTML =
    `<b>${phase}</b> · wants moisture to hold for about ${s.window} days `
    + `· usually worth checking about ${s.lag} day${s.lag === 1 ? '' : 's'} `
    + `after a soaking rain<br>${s.note}`;
}

// --- click readout ------------------------------------------------------- //
const canvases = {};
function canvasFor(src, cb) {
  if (canvases[src]) return cb(canvases[src]);
  const im = new Image();
  im.onload = () => {
    const c = document.createElement('canvas');
    c.width = im.width; c.height = im.height;
    c.getContext('2d', {willReadFrequently:true}).drawImage(im, 0, 0);
    canvases[src] = c; cb(c);
  };
  im.src = src;
}
function sample(c, ll) {
  const [[s,w],[n,e]] = D.bounds;
  const merc = la => Math.log(Math.tan(Math.PI/4 + la*Math.PI/360));
  const px = Math.floor((ll.lng - w)/(e - w)*c.width);
  const py = Math.floor((merc(n) - merc(ll.lat))/(merc(n) - merc(s))*c.height);
  if (px<0||py<0||px>=c.width||py>=c.height) return null;
  const v = c.getContext('2d').getImageData(px,py,1,1).data[0];
  return v === 0 ? null : (v - 1)/254;
}

let last = null;
function read(ll) {
  last = ll;
  const out = document.getElementById('readout');
  if (!ll) { out.textContent = 'Click the map for conditions at a point.'; return; }
  const s = D.species[sp];
  canvasFor(s.score, sc => {
    const f = sample(sc, ll);
    if (f === null) {
      out.innerHTML = 'No rateable canopy here — outside forest, in tidal marsh, '
        + 'or outside the watersheds.'
        + `<br><span style="color:#888">${ll.lat.toFixed(5)}, ${ll.lng.toFixed(5)}</span>`;
      return;
    }
    const cls = f >= 0.60 ? 'Favorable' : f >= 0.35 ? 'Marginal' : 'Unfavorable';
    const col = f >= 0.60 ? '#1a8f3c' : f >= 0.35 ? '#c98214' : '#7d5a5a';
    let html = `<span class="big" style="color:${col}">${cls}</span>`
      + ` <span style="color:#888">score ${f.toFixed(2)}</span>`
      + `<table id="ctx"></table>`
      + `<div style="color:#888;margin-top:4px">${ll.lat.toFixed(5)}, ${ll.lng.toFixed(5)}</div>`;
    out.innerHTML = html;
    const rows = [];
    const keys = Object.keys(D.context);
    let done = 0;
    keys.forEach(k => {
      const c = D.context[k];
      canvasFor(c.twin, cv => {
        const t = sample(cv, ll);
        rows.push([k, c.label, t === null ? '—'
          : (c.vmin + t*(c.vmax - c.vmin)).toFixed(c.decimals) + c.units]);
        if (++done === keys.length) {
          rows.sort((a,b) => keys.indexOf(a[0]) - keys.indexOf(b[0]));
          const tbl = document.getElementById('ctx');
          if (tbl) tbl.innerHTML = rows.map(r =>
            `<tr><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('');
        }
      });
    });
  });
}
map.on('click', e => read(e.latlng));

document.getElementById('caveat').innerHTML =
  `Habitat from Sentinel-2 2026 (10 m oak index, canopy-gated, tidal marsh masked) `
  + `and SSURGO soils. Moisture from MRMS radar/gauge QPE run through a soil bucket `
  + `whose capacity is SSURGO available water in the top 25 cm. `
  + `<b>None of the species thresholds are calibrated</b> — there is no fruiting `
  + `record for this region to fit them to, so they encode ordinary mycological `
  + `expectations, not measured skill. Treat this as "conditions are favorable", `
  + `never "mushrooms are here". Check foraging rules before collecting. `
  + `<b>"Grey out non-public land" is a best-effort match, not a parcel-level `
  + `ownership record</b> — it combines MassGIS's Protected and Recreational `
  + `OpenSpace layer (federal/state/county/municipal ownership) with the `
  + `statewide property tax parcels, kept wherever the owner name matches a `
  + `government pattern ("TOWN OF ...", "COMMONWEALTH OF MASS...", "US ARMY...", `
  + `water/fire districts, housing authorities). That catches most ordinary `
  + `town, state and federal parcels the open-space layer alone misses, but an `
  + `unusual or misspelled owner name can still slip through, and it says `
  + `nothing about whether foraging is actually allowed there.`;

draw();
</script></body></html>
"""

if __name__ == "__main__":
    main()
