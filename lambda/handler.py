"""
Lambda entry point for the daily foraging-app refresh.

Runs the same three scripts as run_daily.sh (mrms_moisture.py ->
score_species.py -> make_foraging_app.py) in /tmp, then uploads the
resulting HTML to the public Cloudflare R2 bucket instead of leaving it on
local disk. /var/task is read-only at runtime, so the static inputs baked
into the image (s2/, soil/, land/, tessera/, the watershed geojsons) are
symlinked into a writable /tmp/work rather than copied.

R2 credentials come from SSM SecureString parameters by default (the
Lambda execution role is scoped to read only that path); R2_ACCESS_KEY_ID /
R2_SECRET_ACCESS_KEY env vars override that, for local/manual testing.
"""

import os
import subprocess
import sys
import time

import boto3

WORK = "/tmp/work"
TASK_ROOT = os.environ.get("LAMBDA_TASK_ROOT", "/var/task")
STATIC = ["s2", "soil", "land", "tessera", "buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson",
          "mrms_moisture.py", "score_species.py", "make_foraging_app.py"]

R2_ENDPOINT = "https://9cbdcb4884f86a6779032ae561e474a5.r2.cloudflarestorage.com"
R2_BUCKET = "osc-pub"
R2_KEY = "foraging/buzzards_bay_foraging.html"
R2_PUBLIC_URL = f"https://r2-pub.openscicomp.io/{R2_KEY}"
SSM_PREFIX = "/buzzards-bay-foraging/r2"


def _ensure_workdir():
    os.makedirs(WORK, exist_ok=True)
    for name in STATIC:
        link = os.path.join(WORK, name)
        if not os.path.exists(link):
            os.symlink(os.path.join(TASK_ROOT, name), link)


def _run(script):
    subprocess.run([sys.executable, "-u", script], cwd=WORK, check=True)


def _r2_credentials():
    env_id, env_secret = os.environ.get("R2_ACCESS_KEY_ID"), os.environ.get("R2_SECRET_ACCESS_KEY")
    if env_id and env_secret:
        return env_id, env_secret
    ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "us-west-2"))

    def get(name):
        return ssm.get_parameter(Name=f"{SSM_PREFIX}/{name}", WithDecryption=True)["Parameter"]["Value"]

    return get("access-key-id"), get("secret-access-key")


def _upload(path):
    key_id, secret = _r2_credentials()
    s3 = boto3.client("s3", endpoint_url=R2_ENDPOINT, region_name="auto",
                       aws_access_key_id=key_id, aws_secret_access_key=secret)
    s3.upload_file(path, R2_BUCKET, R2_KEY,
                    ExtraArgs={"ContentType": "text/html", "CacheControl": "no-cache"})


def handler(event, context):
    t0 = time.time()
    _ensure_workdir()
    for script in ("mrms_moisture.py", "score_species.py", "make_foraging_app.py"):
        print(f"=== running {script} ===", flush=True)
        _run(script)
    out = os.path.join(WORK, "app", "buzzards_bay_foraging.html")
    print(f"=== uploading to {R2_PUBLIC_URL} ===", flush=True)
    _upload(out)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.0f}s -> {R2_PUBLIC_URL}")
    return {"ok": True, "url": R2_PUBLIC_URL, "elapsed_s": elapsed}
