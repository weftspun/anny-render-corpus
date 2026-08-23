"""Fetch the train2017 images named by the fetch list, into zstd parquet shards.

RE-ASSERTS THE HOLDOUT EXCLUSION BEFORE THE FIRST NETWORK CALL. The list this
reads was built clean by `build_coco_fetch_list.py`, and a fetcher that trusts
its input is a way around that gate: a hand-edited or stale parquet would be
fetched without complaint. The check costs one set intersection over a fixed
523-row population, so it is enumerated here again rather than assumed.

PAYLOAD HASHES, BECAUSE THE ARCHIVE RULE ASKS FOR THEM. Every image carries its
sha256 in the same row as its bytes, so a shard can be verified later without a
second store to keep in step -- the same argument RENDER_DATA already makes for
keeping image bytes inside parquet.

FAILURES ARE NAMED, NOT COUNTED. `download_texverse_glbs.py` increments n_err
and discards the reason, which makes a 500 from one host indistinguishable from
4,000 dead URLs. PITFALLS 3 says unchecked things are named and counted, so
every failure is written with its URL and its error, and a run with failures
exits non-zero.

A NON-IMAGE PAYLOAD IS A FAILURE, NOT A ROW. An error page served with HTTP 200
would otherwise be stored as an image and found months later, so each payload is
checked for the JPEG magic bytes before it is kept.

Usage:
  python download_coco_images.py [--limit N] [--workers 8]
"""

import argparse
import hashlib
import os
import queue
import sys
import threading
import time
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

SHARD_BYTES = 256 * 1024 * 1024
JPEG_MAGIC = b"\xff\xd8\xff"

SCHEMA = pa.schema([
    ("image_id", pa.int64()),
    ("file_name", pa.string()),
    ("license_id", pa.int64()),
    ("sha256", pa.string()),
    ("image", pa.binary()),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-list", default="coco_fetch_train2017/fetch_list.parquet")
    ap.add_argument("--holdout", default="coco_person_commercial_val2017/images.parquet")
    ap.add_argument("--out", default="coco_images_train2017")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: fetch only N")
    a = ap.parse_args()

    for p in (a.fetch_list, a.holdout):
        if not os.path.exists(p):                    # unmet precondition is a FAIL
            sys.exit(f"FAIL  missing {p}")

    rows = pq.read_table(a.fetch_list).to_pylist()
    holdout = set(pq.read_table(a.holdout)["image_id"].to_pylist())
    leaked = [r["image_id"] for r in rows if r["image_id"] in holdout]
    if leaked:
        sys.exit(f"FAIL  CONTAMINATION: {len(leaked)} rows in the fetch list are "
                 f"val2017 holdout images. Refusing to fetch. First: {leaked[:5]}")
    print(f"  ok    holdout re-check: 0 of {len(rows)} rows are among the "
          f"{len(holdout)} held-out ids")

    os.makedirs(a.out, exist_ok=True)
    done_file = os.path.join(a.out, "_downloaded.txt")
    done = set()
    if os.path.exists(done_file):
        done = {int(l) for l in open(done_file) if l.strip()}
    todo = [r for r in rows if r["image_id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"  ..    {len(todo)} to fetch ({len(done)} already done)")
    if not todo:
        print("  ok    nothing to do")
        return 0

    q, results = queue.Queue(maxsize=200), queue.Queue()

    def worker():
        while True:
            r = q.get()
            if r is None:
                break
            try:
                req = urllib.request.Request(
                    r["coco_url"], headers={"User-Agent": "weftspun-corpus/0.1"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                if not data.startswith(JPEG_MAGIC):
                    raise ValueError(f"not a JPEG ({len(data)} bytes, "
                                     f"starts {data[:8]!r})")
                results.put((r, data, None))
            except Exception as exc:                             # noqa: BLE001
                results.put((r, None, f"{type(exc).__name__}: {exc}"[:200]))
            q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    for t in threads:
        t.start()

    def feeder():
        for r in todo:
            q.put(r)
        for _ in threads:
            q.put(None)
    threading.Thread(target=feeder, daemon=True).start()

    shard_idx = len([f for f in os.listdir(a.out) if f.endswith(".parquet")])
    buf, buf_bytes, n_ok, t0 = [], 0, 0, time.time()
    failures = []
    df = open(done_file, "a")

    def flush(idx):
        cols = {f.name: [b[f.name] for b in buf] for f in SCHEMA}
        pq.write_table(pa.table(cols, schema=SCHEMA),
                       os.path.join(a.out, f"images_{idx:04d}.parquet"),
                       compression="zstd", compression_level=10)

    for _ in range(len(todo)):
        r, data, err = results.get()
        if err:
            failures.append({"image_id": r["image_id"], "coco_url": r["coco_url"],
                             "error": err})
            continue
        buf.append({"image_id": r["image_id"], "file_name": r["file_name"],
                    "license_id": r["license_id"],
                    "sha256": hashlib.sha256(data).hexdigest(), "image": data})
        buf_bytes += len(data)
        n_ok += 1
        df.write(f"{r['image_id']}\n")
        if buf_bytes >= SHARD_BYTES:
            flush(shard_idx)
            el = time.time() - t0
            print(f"  ..    shard {shard_idx}: {n_ok} ok / {len(failures)} err | "
                  f"{buf_bytes/1e6:.0f} MB | {n_ok/max(el,1e-9):.1f} img/s", flush=True)
            shard_idx += 1
            buf, buf_bytes = [], 0
            df.flush()
    if buf:
        flush(shard_idx)
    df.close()

    if failures:
        # Named and counted, in the same normal form as everything else.
        pq.write_table(pa.table({k: [f[k] for f in failures]
                                 for k in ("image_id", "coco_url", "error")}),
                       os.path.join(a.out, "failures.parquet"), compression="zstd")
        print(f"  FAIL  {len(failures)} of {len(todo)} failed; "
              f"named in {a.out}/failures.parquet")
        for f in failures[:5]:
            print(f"          {f['image_id']}  {f['error']}")

    print(f"  {'ok' if not failures else '!!'}    {n_ok} fetched, "
          f"{len(failures)} failed, {(time.time()-t0)/60:.1f} min")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
