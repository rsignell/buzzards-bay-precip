"""
Lambda entry point for the daily basin-precip refresh.

Runs basin_precip.py (as a subprocess, same pattern as the foraging app's
lambda/handler.py) to recompute the most recent complete MRMS report dates
and merge them into the R2 parquet. /var/task is read-only at runtime, so
the working directory and output land in /tmp.

R2 credentials come from SSM SecureString parameters by default -- the same
/buzzards-bay-foraging/r2/* params the foraging Lambda reads, since it's the
same R2 bucket. R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY env vars override
that, for local/manual testing.

Event overrides (all optional):
  {"sel": "--last 3", "r2_key": "buzzards-bay-precip/smoke/x.parquet"}
"""

import os
import subprocess
import sys
import time

import boto3

TASK_ROOT = os.environ.get("LAMBDA_TASK_ROOT", "/var/task")
WORK = "/tmp/work"
OUT = f"{WORK}/basins_v2_precip_ts.parquet"

DEFAULT_SEL = "--last 2"          # today's and yesterday's report dates, so a
                                   # late MRMS revision to yesterday is picked up
DEFAULT_R2_KEY = "buzzards-bay-precip/basins_v2_precip_ts.parquet"
SSM_PREFIX = "/buzzards-bay-foraging/r2"


def _r2_env():
    env = dict(os.environ)
    if env.get("R2_ACCESS_KEY_ID") and env.get("R2_SECRET_ACCESS_KEY"):
        return env
    ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "us-west-2"))
    get = lambda n: ssm.get_parameter(Name=f"{SSM_PREFIX}/{n}", WithDecryption=True)["Parameter"]["Value"]
    env["R2_ACCESS_KEY_ID"] = get("access-key-id")
    env["R2_SECRET_ACCESS_KEY"] = get("secret-access-key")
    return env


def handler(event, context):
    t0 = time.time()
    os.makedirs(WORK, exist_ok=True)
    event = event or {}
    sel = event.get("sel", DEFAULT_SEL)
    r2_key = event.get("r2_key", DEFAULT_R2_KEY)

    cmd = [sys.executable, "-u", f"{TASK_ROOT}/basin_precip.py", *sel.split(),
           "--out", OUT, "--r2-key", r2_key]
    print(f"=== running {' '.join(cmd)} ===", flush=True)
    subprocess.run(cmd, cwd=WORK, env=_r2_env(), check=True)

    elapsed = time.time() - t0
    print(f"done in {elapsed:.0f}s -> https://r2-pub.openscicomp.io/{r2_key}", flush=True)
    return {"ok": True, "r2_key": r2_key, "elapsed_s": elapsed}
