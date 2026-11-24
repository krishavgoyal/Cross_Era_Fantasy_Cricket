# Body of the Kaggle kernel built by kaggle_runner/run_on_kaggle.sh. Runs ON Kaggle, not locally.
# run_on_kaggle.sh prepends WHICH, BRANCH, EXTRA_ARGS and CODE_B64 (base64 .tgz of local code).
# Reuses colab/remote_run.sh unchanged by giving it the /content layout it expects, then copies
# job.log and results.zip to /kaggle/working so `kaggle kernels output` downloads them.
import base64
import json
import io
import os
import shutil
import subprocess
import sys
import tarfile

REPO_URL = "https://github.com/krishavgoyal/Cross_Era_Fantasy_Cricket.git"
WORKDIR = "/kaggle/working"
LOG = os.path.join(WORKDIR, "job.log")


def finish(rc, note=""):
    with open(LOG, "a") as f:
        f.write(f"PIPELINE_DONE rc={rc}{(' (' + note + ')') if note else ''}\n")
    print(f"PIPELINE_DONE rc={rc} {note}", flush=True)
    sys.exit(0)  # exit 0 so Kaggle keeps /kaggle/working outputs; rc is in job.log


try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
except FileNotFoundError:  # Kaggle gave a CPU machine (usually: account not phone-verified)
    gpu = ""
print("GPU:", gpu or "none", flush=True)
if not gpu:
    finish(3, "no GPU: Kaggle ran this on CPU; phone-verify the account at kaggle.com/settings")
if "P100" in gpu:
    finish(3, "got a P100; torch cu128 does not support it, rerun with a T4")

os.makedirs("/content", exist_ok=True)
shutil.rmtree("/content/cricket", ignore_errors=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO_URL, "/content/cricket"], check=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode="r:gz") as t:  # noqa: F821
    try:
        t.extractall("/content/cricket", filter="data")
    except TypeError:  # Python < 3.12
        t.extractall("/content/cricket")
# Extra data (EXTRA_DATA in run_on_kaggle.sh) arrives as an attached dataset under /kaggle/input:
# either extra.tgz as uploaded, or already unpacked by Kaggle (a data/ folder).
for root, dirs, files in os.walk("/kaggle/input"):
    if "extra.tgz" in files:
        with tarfile.open(os.path.join(root, "extra.tgz"), mode="r:gz") as t:
            try:
                t.extractall("/content/cricket", filter="data")
            except TypeError:
                t.extractall("/content/cricket")
            names = t.getnames()
        print("extra data unpacked:", sorted({n.split("/")[2] for n in names if n.count("/") >= 2}), flush=True)
        break
    if os.path.basename(root) == "data" and ("processed_clips" in dirs or "historical_images" in dirs):
        shutil.copytree(root, "/content/cricket/data", dirs_exist_ok=True)
        print("extra data copied from", root, flush=True)
        break
print("code: cloned + overlaid local changes", flush=True)

env = dict(os.environ, EXTRA_ARGS=EXTRA_ARGS, CLEAN=globals().get("CLEAN", "0"))  # noqa: F821
env.update(json.loads(globals().get("REMOTE_ENV", "{}")))
with open(LOG, "w") as log:
    p = subprocess.Popen(["bash", "colab/remote_run.sh", WHICH], cwd="/content/cricket", env=env,  # noqa: F821
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    p.wait()

if os.path.exists("/content/results.zip"):
    shutil.copy("/content/results.zip", os.path.join(WORKDIR, "results.zip"))
print("runner exit", p.returncode, flush=True)
