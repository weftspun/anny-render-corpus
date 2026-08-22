"""COCO person-image license filter -> ETNF zstd-parquet relational set.

COCO's annotation files carry a per-image `license` id referencing the
`licenses` table in the same JSON (the images are Flickr photos that KEEP
their original licenses -- only COCO's annotations are CC-BY 4.0). This
filter keeps only images whose own license permits commercial use and
derivatives, which is what model training/products need:

  KEEP:  CC-BY 2.0, CC-BY-SA 2.0, "No known copyright restrictions",
         "United States Government Work"
  DROP:  every NC variant (non-commercial) and every ND variant
         (no-derivatives -- training/derived meshes are derivatives)

SA (share-alike) rows are identifiable via the licenses relation:
share-alike obligations on *derived models* are legally unsettled, so
downstream can join-and-exclude if the org wants zero SA exposure.

Output: zstd-compressed parquet in Essential Tuple Normal Form (the org's
dataset rule -- C.J. Date normalization: repeated text interned into its own
relation, observation facts split from entity facts, no NULLs, no derivable
columns):

  <out>/licenses.parquet             entity: license_id, name, url, share_alike
  <out>/images.parquet               entity: image_id, file_name, coco_url,
                                     flickr_url, width, height, license_id (FK)
  <out>/person_observations.parquet  facts:  image_id, num_persons,
                                     max_num_keypoints

Usage:
  python filter_coco_licenses.py \
      --annotations person_keypoints_val2017.json \
      --out coco_person_commercial_val2017
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd

COMMERCIAL_OK = {
    "Attribution License",                       # CC-BY 2.0
    "Attribution-ShareAlike License",            # CC-BY-SA 2.0 (see licenses relation)
    "No known copyright restrictions",
    "United States Government Work",
}
SHARE_ALIKE = {"Attribution-ShareAlike License"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True, help="output DIRECTORY for the parquet relations")
    parser.add_argument("--min-keypoints", type=int, default=8,
                        help="keep images having at least one person with this many labeled keypoints")
    args = parser.parse_args()

    with open(args.annotations) as f:
        coco = json.load(f)

    licenses = {l["id"]: l for l in coco["licenses"]}
    keep_ids = {i for i, l in licenses.items() if l["name"] in COMMERCIAL_OK}
    print("license table:")
    for i, l in sorted(licenses.items()):
        marker = "KEEP" if i in keep_ids else "drop"
        print(f"  [{marker}] {i}: {l['name']} ({l['url']})")

    person_stats = defaultdict(lambda: {"persons": 0, "max_kp": 0})
    for ann in coco["annotations"]:
        s = person_stats[ann["image_id"]]
        s["persons"] += 1
        s["max_kp"] = max(s["max_kp"], ann.get("num_keypoints", 0))

    image_rows, obs_rows = [], []
    for img in coco["images"]:
        if img["license"] not in keep_ids:
            continue
        stats = person_stats.get(img["id"])
        if not stats or stats["max_kp"] < args.min_keypoints:
            continue
        image_rows.append({
            "image_id": img["id"],
            "file_name": img["file_name"],
            "coco_url": img["coco_url"],
            "flickr_url": img["flickr_url"],
            "width": img["width"],
            "height": img["height"],
            "license_id": img["license"],
        })
        obs_rows.append({
            "image_id": img["id"],
            "num_persons": stats["persons"],
            "max_num_keypoints": stats["max_kp"],
        })

    license_rows = [
        {"license_id": i, "name": l["name"], "url": l["url"],
         "share_alike": l["name"] in SHARE_ALIKE}
        for i, l in sorted(licenses.items()) if i in keep_ids
    ]

    os.makedirs(args.out, exist_ok=True)
    for name, rows in [("licenses", license_rows), ("images", image_rows),
                       ("person_observations", obs_rows)]:
        df = pd.DataFrame(rows)
        assert not df.isnull().values.any(), f"NULLs in {name} violate ETNF"
        df.to_parquet(os.path.join(args.out, f"{name}.parquet"),
                      compression="zstd", index=False)

    total = len(coco["images"])
    sa = sum(1 for r in image_rows
             if licenses[r["license_id"]]["name"] in SHARE_ALIKE)
    print(f"\n{len(image_rows)}/{total} images kept "
          f"({sa} share-alike, resolvable via licenses.parquet) -> {args.out}/")


if __name__ == "__main__":
    main()
