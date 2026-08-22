"""Convert a Roboflow-style COCO zip into the org dataset format:
ETNF zstd-parquet relations + a zstd-compressed image payload.

Input : <name>.coco.zip with split dirs (train/valid/test), each holding
        _annotations.coco.json + images.
Output: <name>/ directory containing
        images.parquet        entity: image_id, split, file_name, width, height
        categories.parquet    entity: category_id, name, supercategory
        annotations.parquet   fact:   ann_id, image_id, category_id, area, iscrowd
        bboxes.parquet        satellite (only anns that have one): ann_id, x, y, w, h
        segmentations.parquet satellite: ann_id, poly_idx, points (list<float>)
        keypoints.parquet     satellite: ann_id, kp_idx, x, y, visibility
        image_data.parquet    satellite: image_id, data (binary) - the image
                              bytes themselves, zstd-compressed parquet pages,
                              written in row-group batches

ETNF discipline: repeated text interned into entity relations, optional
attributes decomposed into satellite relations (no NULL columns anywhere),
facts separated from entities. Asserted before write.

Usage: python coco_zip_to_etnf.py <path-to.coco.zip> [--keep-zip] [--scratch DIR]

I/O note: measured 2026-08-14, the O: dataset volume runs ~171 MB/s write /
195 MB/s read vs ~1270/1745 MB/s on local C:. Conversion is write-heavy
(re-compressing every image at zstd level 10), so with --scratch the parquet
is built on a local disk and moved to the destination at the end -- roughly
7x the write bandwidth during the slow part.
"""

import argparse
import io
import json
import os
import sys
import tarfile
import zipfile

import pandas as pd
import zstandard


def convert(zip_path: str, keep_zip: bool = False, scratch: str = "") -> str:
    final_dir = zip_path[: -len(".coco.zip")] if zip_path.endswith(".coco.zip") else zip_path + ".etnf"
    if scratch:
        out_dir = os.path.join(scratch, os.path.basename(final_dir))
    else:
        out_dir = final_dir
    os.makedirs(out_dir, exist_ok=True)

    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    ann_files = [n for n in names if n.endswith("_annotations.coco.json")]
    if not ann_files:
        raise SystemExit(f"no _annotations.coco.json inside {zip_path}")

    images, categories, annotations = [], {}, []
    bboxes, segmentations, keypoints = [], [], []
    image_files = {}  # zip name -> (split, file_name)
    ann_uid = 0

    for ann_name in ann_files:
        split = ann_name.split("/")[0] if "/" in ann_name else "all"
        coco = json.loads(zf.read(ann_name))
        for c in coco.get("categories", []):
            categories[c["id"]] = {
                "category_id": c["id"],
                "name": c.get("name", ""),
                "supercategory": c.get("supercategory", ""),
            }
        id_map = {}
        for img in coco.get("images", []):
            uid = f"{split}/{img['id']}"
            id_map[img["id"]] = uid
            images.append({
                "image_id": uid,
                "split": split,
                "file_name": img.get("file_name", ""),
                "width": img.get("width", 0),
                "height": img.get("height", 0),
            })
            prefix = f"{split}/" if "/" in ann_name else ""
            image_files[f"{prefix}{img.get('file_name','')}"] = (split, img.get("file_name", ""))
        for a in coco.get("annotations", []):
            ann_uid += 1
            annotations.append({
                "ann_id": ann_uid,
                "image_id": id_map.get(a["image_id"], f"{split}/{a['image_id']}"),
                "category_id": a.get("category_id", -1),
                "area": float(a.get("area", 0.0)),
                "iscrowd": int(a.get("iscrowd", 0)),
            })
            bb = a.get("bbox")
            if bb and len(bb) == 4:
                bboxes.append({"ann_id": ann_uid, "x": float(bb[0]), "y": float(bb[1]),
                               "w": float(bb[2]), "h": float(bb[3])})
            seg = a.get("segmentation")
            if isinstance(seg, list):
                for pi, poly in enumerate(seg):
                    if isinstance(poly, list) and poly:
                        segmentations.append({"ann_id": ann_uid, "poly_idx": pi,
                                              "points": [float(v) for v in poly]})
            kps = a.get("keypoints")
            if isinstance(kps, list) and kps:
                for ki in range(0, len(kps) - 2, 3):
                    keypoints.append({"ann_id": ann_uid, "kp_idx": ki // 3,
                                      "x": float(kps[ki]), "y": float(kps[ki + 1]),
                                      "visibility": int(kps[ki + 2])})

    tables = {
        "images": pd.DataFrame(images),
        "categories": pd.DataFrame(sorted(categories.values(), key=lambda c: c["category_id"])),
        "annotations": pd.DataFrame(annotations),
        "bboxes": pd.DataFrame(bboxes),
        "segmentations": pd.DataFrame(segmentations),
        "keypoints": pd.DataFrame(keypoints),
    }
    for name, df in tables.items():
        if df.empty:
            continue
        assert not df.isnull().values.any(), f"NULLs in {name} violate ETNF"
        df.to_parquet(os.path.join(out_dir, f"{name}.parquet"), compression="zstd", index=False)
        print(f"  {name}.parquet: {len(df)} rows")

    # Image payload -> zstd parquet (binary column), row-group batches so
    # memory stays bounded. image_id joins back to images.parquet.
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("image_id", pa.string()), ("data", pa.binary())])
    pq_path = os.path.join(out_dir, "image_data.parquet")
    writer = pq.ParquetWriter(pq_path, schema, compression="zstd", compression_level=10)
    batch_ids, batch_data, batch_bytes, n_img = [], [], 0, 0
    # zip entry name -> image_id (from the annotation-driven map)
    name_to_id = {}
    for row in images:
        prefix = f"{row['split']}/" if row['split'] != "all" else ""
        name_to_id[f"{prefix}{row['file_name']}"] = row["image_id"]
    for zname in names:
        if zname.endswith((".json", "/")) or zname.startswith("README"):
            continue
        data = zf.read(zname)
        batch_ids.append(name_to_id.get(zname, zname))
        batch_data.append(data)
        batch_bytes += len(data)
        n_img += 1
        if batch_bytes >= 256 * 1024 * 1024:
            writer.write_table(pa.table({"image_id": batch_ids, "data": batch_data}, schema=schema))
            batch_ids, batch_data, batch_bytes = [], [], 0
    if batch_ids:
        writer.write_table(pa.table({"image_id": batch_ids, "data": batch_data}, schema=schema))
    writer.close()
    print(f"  image_data.parquet: {n_img} images, {os.path.getsize(pq_path)/1e6:.0f} MB "
          f"(zip was {os.path.getsize(zip_path)/1e6:.0f} MB)")

    zf.close()

    if scratch:
        import shutil
        if os.path.isdir(final_dir):
            shutil.rmtree(final_dir)
        shutil.move(out_dir, final_dir)
        print(f"  moved scratch -> {final_dir}")

    if not keep_zip:
        os.remove(zip_path)
        print(f"  removed {zip_path}")
    return final_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zips", nargs="+")
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument("--scratch", default="", help="build on this local dir, then move")
    args = parser.parse_args()
    for z in args.zips:
        print(f"== {z}")
        convert(z, keep_zip=args.keep_zip, scratch=args.scratch)
