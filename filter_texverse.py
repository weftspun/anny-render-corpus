"""TexVerse license filter -> ETNF zstd-parquet manifest + download list.

TexVerse (HF YiboZhang2001/TexVerse) curates 858,669 Sketchfab models. The
dataset wrapper is ODC-BY, but EACH MODEL keeps its own Sketchfab license,
recorded per-model in metadata.json. This filter keeps only models whose own
license permits commercial use AND derivatives:

  KEEP:  CC Attribution (CC-BY), CC0
  DROP:  every NonCommercial variant, every NoDerivatives variant,
         CC-BY-ShareAlike (blocklisted: share-alike obligations on derived
         models are legally unsettled), and any model with no recorded license

Output (ETNF, zstd parquet, no NULLs):
  licenses.parquet     entity: license_id, name, share_alike (always False;
                       SA models are excluded entirely)
  models.parquet       entity: model_id, name, user, license_id (FK),
                               vertex_count, face_count, max_texture,
                               is_rigged, animation_count
  glb_paths.parquet    satellite: model_id, resolution, path
  categories.parquet   satellite: model_id, category
  tags.parquet         satellite: model_id, tag
  download_list.txt    the glb paths to fetch (highest resolution per model)

Usage:
  python filter_texverse.py --metadata metadata.json --out texverse_clean \
      [--rigged-only] [--pbr-ids TexVerse_pbr_id_list.txt] [--max-models N]
"""

import argparse
import json
import os

import pandas as pd

COMMERCIAL_OK = {
    "CC Attribution",
    "CC0 Public Domain",
    "Public Domain",
}
# Share-alike is BLOCKLISTED (user directive 2026-08-14): SA obligations on
# derived models are legally unsettled, so we take zero SA exposure.
SHARE_ALIKE = {"CC Attribution-ShareAlike"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rigged-only", action="store_true")
    parser.add_argument("--pbr-ids", default="")
    parser.add_argument("--max-models", type=int, default=0)
    args = parser.parse_args()

    pbr_ids = set()
    if args.pbr_ids and os.path.exists(args.pbr_ids):
        pbr_ids = {l.strip() for l in open(args.pbr_ids) if l.strip()}
        print(f"pbr id list: {len(pbr_ids)}")

    print("loading metadata (837 MB) ...")
    with open(args.metadata, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"{len(meta)} models in metadata")

    lic_names = {}
    for m in meta.values():
        lic_names[m.get("license") or ""] = lic_names.get(m.get("license") or "", 0) + 1
    print("license distribution:")
    for k, v in sorted(lic_names.items(), key=lambda kv: -kv[1]):
        mark = "KEEP" if k in COMMERCIAL_OK else "drop"
        print(f"  [{mark}] {v:7d}  {k or '(none)'}")

    lic_ids = {name: i for i, name in enumerate(sorted(COMMERCIAL_OK))}
    models, glbs, cats, tags = [], [], [], []
    for mid, m in meta.items():
        lic = m.get("license") or ""
        if lic not in COMMERCIAL_OK:
            continue
        if args.rigged_only and not m.get("isRigged"):
            continue
        if pbr_ids and mid not in pbr_ids:
            continue
        models.append({
            "model_id": mid,
            "name": m.get("name") or "",
            "user": m.get("user") or "",
            "license_id": lic_ids[lic],
            "vertex_count": int(m.get("vertexCount") or 0),
            "face_count": int(m.get("faceCount") or 0),
            "max_texture": int(m.get("max_texture") or 0),
            "is_rigged": bool(m.get("isRigged")),
            "animation_count": int(m.get("animation") or 0),
        })
        for p in m.get("glb_paths") or []:
            res = 2048 if "_2048" in p else (1024 if "_1024" in p else 0)
            glbs.append({"model_id": mid, "resolution": res, "path": p})
        for c in m.get("categories") or []:
            cats.append({"model_id": mid, "category": c})
        for t in m.get("tags") or []:
            tags.append({"model_id": mid, "tag": t})
        if args.max_models and len(models) >= args.max_models:
            break

    os.makedirs(args.out, exist_ok=True)
    licenses = [{"license_id": i, "name": n, "share_alike": n in SHARE_ALIKE}
                for n, i in sorted(lic_ids.items(), key=lambda kv: kv[1])]
    for name, rows in [("licenses", licenses), ("models", models),
                       ("glb_paths", glbs), ("categories", cats), ("tags", tags)]:
        if not rows:
            continue
        df = pd.DataFrame(rows)
        assert not df.isnull().values.any(), f"NULLs in {name} violate ETNF"
        df.to_parquet(os.path.join(args.out, f"{name}.parquet"),
                      compression="zstd", index=False)
        print(f"  {name}.parquet: {len(df)} rows")

    # highest-resolution glb per model = the download list
    best = {}
    for g in glbs:
        cur = best.get(g["model_id"])
        if cur is None or g["resolution"] > cur["resolution"]:
            best[g["model_id"]] = g
    with open(os.path.join(args.out, "download_list.txt"), "w") as f:
        for g in best.values():
            f.write(g["path"] + "\n")
    print(f"  download_list.txt: {len(best)} glb paths")


if __name__ == "__main__":
    main()
