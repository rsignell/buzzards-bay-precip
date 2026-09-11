#!/usr/bin/env bash
# Daily refresh of the foraging app.
#
# Only the moisture state changes day to day. The habitat layers -- oak index,
# soil, marsh mask -- are static and are NOT rebuilt here; see the one-off
# chain in the README if they ever need regenerating.
#
# Run on the oak-mapping cluster from ~/sky_workdir. mrms_moisture.py needs the
# `mrms` env because icechunk requires Python >= 3.11; the rest run in base.
set -euo pipefail

BASE=/home/ubuntu/miniconda3/bin/python
MRMS=/home/ubuntu/miniconda3/envs/mrms/bin/python

echo "=== 1/3  MRMS -> soil moisture ==="
$MRMS mrms_moisture.py

echo "=== 2/3  species scores ==="
$BASE score_species.py

echo "=== 3/3  app ==="
$BASE make_foraging_app.py

echo
echo "as of: $(python3 -c "import json;print(json.load(open('moisture/meta.json'))['as_of'])" 2>/dev/null || true)"
echo "fetch with:  rsync -az -e ssh oak-mapping:~/sky_workdir/app/buzzards_bay_foraging.html ./app/"
