#!/usr/bin/env bash
# COCO -> GEM-X batch, prepared for a single RunPod session (no babysitting).
# Usage on the pod: bash run_gemx_batch.sh <n_images>
# Prereqs on pod: python3.10, git, ffmpeg. Everything else installs here.
set -euo pipefail
N="${1:-8}"

git clone --recurse-submodules -q https://github.com/NVlabs/GEM-X.git /src/GEM-X
pip install -q -e /src/GEM-X -e /src/GEM-X/third_party/soma pandas pyarrow

# Pull the N first commercially-licensed person images from the ETNF manifest
python3 - "$N" <<'PYEOF'
import sys, urllib.request, os
import pandas as pd
n = int(sys.argv[1])
# TRAIN split only -- val2017 is held out for evaluation; never train on it
images = pd.read_parquet("coco_person_commercial_train2017/images.parquet")
os.makedirs("inputs", exist_ok=True)
for _, row in images.head(n).iterrows():
    urllib.request.urlretrieve(row.coco_url, f"inputs/{row.file_name}")
    print("fetched", row.file_name)
PYEOF

# Each still photo runs as a 1-frame clip (GEM-X's demo takes --video)
mkdir -p clips outputs
for img in inputs/*.jpg; do
  base=$(basename "$img" .jpg)
  ffmpeg -y -loglevel error -loop 1 -i "$img" -t 0.5 -r 4 "clips/$base.mp4"
  python3 /src/GEM-X/scripts/demo/demo_soma.py \
    --video "clips/$base.mp4" --static_cam \
    --output_root "outputs/$base"
done
echo "done -> outputs/ (SOMA-X 77-joint pose + identity per image)"
