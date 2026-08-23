"""Red/green control for build_coco_fetch_list.py.

The contamination gate is the one check in this repository whose failure is
unrecoverable -- a generated frame carrying a held-out photo cannot be found
again downstream -- so it is the last one that may be taken on trust. Same
protocol as test_preflight.py: every check is proven in both directions.

Self-contained. It builds its own train/holdout pair in a temp directory rather
than needing the real ones, because a test that skips when its fixture is absent
reads exactly like a pass.

Usage: python test_coco_fetch_list.py
"""

import os
import shutil
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import build_coco_fetch_list as B

IMAGES = pa.schema([
    ("image_id", pa.int64()), ("file_name", pa.string()),
    ("coco_url", pa.string()), ("flickr_url", pa.string()),
    ("width", pa.int64()), ("height", pa.int64()), ("license_id", pa.int64()),
])
LICENSES = pa.schema([
    ("license_id", pa.int64()), ("name", pa.string()),
    ("url", pa.string()), ("share_alike", pa.bool_()),
])


def img(i, lic):
    return {"image_id": i, "file_name": f"{i:012d}.jpg",
            "coco_url": f"http://images.cocodataset.org/train2017/{i:012d}.jpg",
            "flickr_url": "", "width": 640, "height": 480, "license_id": lic}


def write(root, name, schema, rows):
    os.makedirs(root, exist_ok=True)
    cols = {f.name: [r[f.name] for r in rows] for f in schema}
    pq.write_table(pa.table(cols, schema=schema),
                   os.path.join(root, f"{name}.parquet"), compression="zstd")


def fixture(base):
    """Four CC-BY, two ShareAlike, one no-known-restrictions in train; three
    disjoint ids in the holdout. Small enough to check by hand."""
    tr, ho = os.path.join(base, "train"), os.path.join(base, "holdout")
    lic = [{"license_id": 4, "name": "Attribution License", "url": "u", "share_alike": False},
           {"license_id": 5, "name": "Attribution-ShareAlike License", "url": "u", "share_alike": True},
           {"license_id": 7, "name": "No known copyright restrictions", "url": "u", "share_alike": False}]
    train = [img(i, 4) for i in (10, 11, 12, 13)] + [img(i, 5) for i in (20, 21)] + [img(30, 7)]
    write(tr, "images", IMAGES, train)
    write(tr, "licenses", LICENSES, lic)
    write(ho, "images", IMAGES, [img(i, 4) for i in (90, 91, 92)])
    write(ho, "licenses", LICENSES, lic)
    return tr, ho


# ---- corruptions: each targets ONE check ----------------------------------

def c_holdout_in_train(tr, ho):
    """The unrecoverable one: a held-out image_id present in the train split.
    This is exactly the shape rf-detr-keypoint-data has, where all 523 holdout
    images sit inside a split named `train`."""
    rows = pq.read_table(f"{tr}/images.parquet").to_pylist()
    rows.append(img(91, 4))
    write(tr, "images", IMAGES, rows)


def c_sa_vocabulary_gone(tr, ho):
    """The SA licence disappears from the vocabulary. The filter would then drop
    nothing and report a bigger, cleaner-looking list."""
    lic = pq.read_table(f"{tr}/licenses.parquet").to_pylist()
    for r in lic:
        if r["name"] == "Attribution-ShareAlike License":
            r["name"] = "Attribution License"
    write(tr, "licenses", LICENSES, lic)


def c_null_row(tr, ho):
    rows = pq.read_table(f"{tr}/images.parquet").to_pylist()
    rows[0]["coco_url"] = None
    write(tr, "images", IMAGES, rows)


CORRUPTIONS = [
    ("holdout id in train", c_holdout_in_train, "CONTAMINATION"),
    ("SA vocabulary gone", c_sa_vocabulary_gone, "no longer filters"),
    ("NULL in a kept column", c_null_row, "violating ETNF"),
]


def main():
    base = tempfile.mkdtemp(prefix="coco-fetch-")
    failures = 0
    try:
        tr, ho = fixture(base)
        out = os.path.join(base, "out")

        kept, problems = B.build(tr, ho, out)
        expect = 5                                   # 4 CC-BY + 1 no-restrictions
        ok = not problems and kept is not None and len(kept) == expect
        print(f"GREEN  clean fixture -> {len(kept) if kept else 0} kept "
              f"(expected {expect}), {len(problems)} problems")
        if not ok:
            failures += 1

        for label, corrupt, expect_str in CORRUPTIONS:
            work = tempfile.mkdtemp(prefix="coco-fetch-red-")
            try:
                shutil.rmtree(work)
                shutil.copytree(base, work)
                shutil.rmtree(os.path.join(work, "out"), ignore_errors=True)
                wtr, who = os.path.join(work, "train"), os.path.join(work, "holdout")
                corrupt(wtr, who)
                _, probs = B.build(wtr, who, os.path.join(work, "out"))
                hit = [p for p in probs if expect_str in p]
                print(f"{'RED  ok  ' if hit else 'RED  MISS'} {label:24s} -> "
                      f"{hit[0][:78] if hit else 'check did not fire'}")
                if not hit:
                    failures += 1
                # A failed build must write nothing. A gate that reports FAIL and
                # leaves a usable artefact behind is a gate somebody will use.
                if os.path.exists(os.path.join(work, "out", "fetch_list.parquet")):
                    print(f"         MISS {label}: wrote a fetch list despite failing")
                    failures += 1
            finally:
                shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    print(f"\n{len(CORRUPTIONS)} corruptions, {failures} unproven")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
