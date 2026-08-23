"""Build the train2017 fetch list, and refuse to build a contaminated one.

WHAT THIS EXISTS TO PREVENT. `coco_person_commercial_val2017` is the blinded
holdout, and CLAUDE.md's corollary is that if train2017 feeds a generation
pipeline, val2017 must not -- an image generated from a held-out photo carries
that photo's content into training. That mistake is unrecoverable in a way most
are not: once a generated frame carries a held-out photo's content, no licence
check, dedup or audit downstream can find it again. So the exclusion is enforced
where the URL list is built, before anything is fetched.

WHY IT ALSO DROPS SHARE-ALIKE. `filter_coco_licenses.py` keeps
`Attribution-ShareAlike License`, flags it in the licenses relation, and says in
its own docstring that downstream can "join-and-exclude if the org wants zero SA
exposure". CLAUDE.md blocklists CC-BY-SA, and `filter_texverse.py` already takes
zero SA exposure per the 2026-08-14 directive. This is the downstream that
filter was deferring to, and the deferral is measured rather than assumed: 4,400
of the 12,620 train rows are SA, which is 34.9% of a set whose directory name
says `commercial`.

ENUMERATION, NOT SAMPLING. Both populations are fixed and small -- 12,620 and
523 -- so PITFALLS 5 says enumerate. Every check below runs over every row and
reports a count, and there is no detection floor to state because nothing is
sampled.

Usage:
  python build_coco_fetch_list.py --train coco_person_commercial_train2017 \
      --holdout coco_person_commercial_val2017 --out coco_fetch_train2017
"""

import argparse
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

# CLAUDE.md blocklists CC-BY-SA. Named by licence rather than by id, because a
# COCO licence id is only meaningful inside the file that defined it.
SHARE_ALIKE_NAMES = {"Attribution-ShareAlike License"}

FETCH_LIST = pa.schema([
    ("image_id", pa.int64()),
    ("file_name", pa.string()),
    ("coco_url", pa.string()),      # images.cocodataset.org, not the flickr_url:
                                    # flickr URLs rot as users delete photos
    ("license_id", pa.int64()),
])


def read(root, name):
    """An unmet precondition is a FAIL, never a skip. A missing relation here
    would otherwise make every check below vacuously pass."""
    path = os.path.join(root, f"{name}.parquet")
    if not os.path.exists(path):
        sys.exit(f"FAIL  missing {path}: cannot verify what is not there")
    return pq.read_table(path)


def build(train_dir, holdout_dir, out_dir):
    problems, notes = [], []

    train = read(train_dir, "images")
    train_lic = read(train_dir, "licenses")
    holdout = read(holdout_dir, "images")

    lic_name = dict(zip(train_lic["license_id"].to_pylist(),
                        train_lic["name"].to_pylist()))
    sa_ids = {i for i, n in lic_name.items() if n in SHARE_ALIKE_NAMES}
    if not sa_ids:
        # The SA licence vanishing means the upstream vocabulary changed, not
        # that the corpus got cleaner. Passing silently here would certify it.
        problems.append("no share-alike licence in the licenses relation: the "
                        "vocabulary changed and this filter no longer filters")

    rows = train.to_pylist()
    holdout_ids = set(holdout["image_id"].to_pylist())
    notes.append(f"holdout population enumerated: {len(holdout_ids)} image_ids")

    kept, dropped_sa, dropped_holdout, dropped_null = [], 0, 0, 0
    for r in rows:
        if any(r[c] is None for c in FETCH_LIST.names):
            dropped_null += 1                       # ETNF: no NULLs, so a NULL
            continue                                # row is a defect, not a gap
        if r["image_id"] in holdout_ids:
            dropped_holdout += 1
            continue
        if r["license_id"] in sa_ids:
            dropped_sa += 1
            continue
        kept.append({c: r[c] for c in FETCH_LIST.names})

    # THE CONTAMINATION CHECK IS AN ASSERTION, NOT A FILTER RESULT. Dropping the
    # overlap silently would leave a clean-looking list built from a dirty
    # source, and the fact that an overlap existed at all is the thing worth
    # knowing -- it means the two sets were not disjoint upstream.
    if dropped_holdout:
        problems.append(
            f"CONTAMINATION: {dropped_holdout} train rows carry a val2017 "
            "image_id. The two sets are not disjoint upstream, so the holdout "
            "is not blinded and this list must not be fetched")
    if dropped_null:
        problems.append(f"{dropped_null} rows carry NULLs, violating ETNF")

    # Post-condition, re-derived rather than trusted: the arithmetic has to close.
    if len(kept) + dropped_sa + dropped_holdout + dropped_null != len(rows):
        problems.append("row arithmetic does not close; the filter lost rows")

    # And the emitted list is re-checked against the holdout from scratch. The
    # loop above already excluded them; this asserts the loop did what it says.
    leaked = [k["image_id"] for k in kept if k["image_id"] in holdout_ids]
    if leaked:
        problems.append(f"POST-CHECK: {len(leaked)} holdout ids survived the filter")
    if any(k["license_id"] in sa_ids for k in kept):
        problems.append("POST-CHECK: share-alike rows survived the filter")

    print(f"train rows            {len(rows)}")
    print(f"  dropped share-alike {dropped_sa}")
    print(f"  dropped holdout     {dropped_holdout}")
    print(f"  dropped NULL        {dropped_null}")
    print(f"FETCH LIST            {len(kept)}")
    for n in notes:
        print(f"  ..  {n}")

    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        return None, problems

    os.makedirs(out_dir, exist_ok=True)
    cols = {c: [k[c] for k in kept] for c in FETCH_LIST.names}
    pq.write_table(pa.table(cols, schema=FETCH_LIST),
                   os.path.join(out_dir, "fetch_list.parquet"), compression="zstd")
    print(f"  ok    wrote {out_dir}/fetch_list.parquet")
    return kept, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="coco_person_commercial_train2017")
    ap.add_argument("--holdout", default="coco_person_commercial_val2017")
    ap.add_argument("--out", default="coco_fetch_train2017")
    a = ap.parse_args()
    kept, problems = build(a.train, a.holdout, a.out)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
