#!/usr/bin/env bash
# Daily refresh of the foraging app.
#
# Only the moisture state changes day to day. The habitat layers -- oak index,
# soil, marsh mask -- are static and are NOT rebuilt here.
#
# Runs either locally or on the oak-mapping cluster; it picks the interpreter
# itself. Locally the whole chain takes ~1.5 minutes and peaks near 2 GB, so
# the cluster is not needed for a daily run -- only for rebuilding the habitat
# layers, where the Sentinel-2 compositing wants more memory than a laptop has.
#
# Needs: icechunk (Python >= 3.11), rioxarray, geopandas, rasterio, PIL.
# Override the interpreter with  PY=/path/to/python ./run_daily.sh
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${PY:-}" ]]; then
  if [[ -x /home/ubuntu/miniconda3/envs/mrms/bin/python ]]; then
    PY=/home/ubuntu/miniconda3/envs/mrms/bin/python        # cluster
  elif [[ -x "$HOME/miniforge3/envs/protocoast-notebook/bin/python" ]]; then
    PY="$HOME/miniforge3/envs/protocoast-notebook/bin/python"  # laptop
  else
    echo "no interpreter with icechunk found; set PY=..." >&2; exit 1
  fi
fi
echo "interpreter: $PY"
"$PY" -c "import icechunk, rioxarray, geopandas, rasterio, PIL" || {
  echo "interpreter is missing dependencies" >&2; exit 1; }

echo "=== 1/3  MRMS -> soil moisture ==="; "$PY" -u mrms_moisture.py
echo "=== 2/3  species scores ===";        "$PY" -u score_species.py
echo "=== 3/3  app ===";                   "$PY" -u make_foraging_app.py

echo
echo "as of $("$PY" -c "import json;print(json.load(open('moisture/meta.json'))['as_of'])")"
echo "open:  explorer.exe \"\$(wslpath -w app/buzzards_bay_foraging.html)\""
