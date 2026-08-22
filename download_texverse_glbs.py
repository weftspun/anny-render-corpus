"""Download TexVerse GLBs from a manifest download_list.txt, packing them
directly into zstd parquet shards (no loose files, no zip)."""
import os, sys, time, urllib.request, threading, queue
import pyarrow as pa, pyarrow.parquet as pq

MANIFEST = sys.argv[1]
OUT = sys.argv[2]
SHARD_BYTES = 512 * 1024 * 1024
BASE = "https://huggingface.co/datasets/YiboZhang2001/TexVerse/resolve/main/"
WORKERS = 16

os.makedirs(OUT, exist_ok=True)
paths = [l.strip() for l in open(os.path.join(MANIFEST, "download_list.txt")) if l.strip()]
done_file = os.path.join(OUT, "_downloaded.txt")
done = set()
if os.path.exists(done_file):
    done = {l.strip() for l in open(done_file)}
todo = [p for p in paths if p not in done]
print(f"{len(todo)} to download ({len(done)} already done)", flush=True)

schema = pa.schema([("model_id", pa.string()), ("path", pa.string()), ("glb", pa.binary())])
q = queue.Queue(maxsize=200)
results = queue.Queue()

def worker():
    while True:
        p = q.get()
        if p is None: break
        try:
            with urllib.request.urlopen(BASE + urllib.parse.quote(p), timeout=120) as r:
                data = r.read()
            mid = os.path.basename(p).split("_")[0]
            results.put((mid, p, data))
        except Exception as e:
            results.put(("ERR", p, str(e).encode()[:200]))
        q.task_done()

import urllib.parse
threads = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
for t in threads: t.start()

def feeder():
    for p in todo: q.put(p)
    for _ in threads: q.put(None)
threading.Thread(target=feeder, daemon=True).start()

shard_idx = len([f for f in os.listdir(OUT) if f.endswith(".parquet")])
buf_ids, buf_paths, buf_data, buf_bytes = [], [], [], 0
n_ok = n_err = 0
t0 = time.time()
df = open(done_file, "a")
for _ in range(len(todo)):
    mid, p, data = results.get()
    if mid == "ERR":
        n_err += 1; continue
    buf_ids.append(mid); buf_paths.append(p); buf_data.append(data); buf_bytes += len(data)
    n_ok += 1
    df.write(p + "\n")
    if buf_bytes >= SHARD_BYTES:
        pq.write_table(pa.table({"model_id": buf_ids, "path": buf_paths, "glb": buf_data}, schema=schema),
                       os.path.join(OUT, f"glbs_{shard_idx:04d}.parquet"), compression="zstd", compression_level=10)
        el = time.time()-t0
        print(f"shard {shard_idx}: {n_ok} ok / {n_err} err | {buf_bytes/1e6:.0f} MB | {n_ok/el:.1f} files/s", flush=True)
        shard_idx += 1; buf_ids, buf_paths, buf_data, buf_bytes = [], [], [], 0
        df.flush()
if buf_ids:
    pq.write_table(pa.table({"model_id": buf_ids, "path": buf_paths, "glb": buf_data}, schema=schema),
                   os.path.join(OUT, f"glbs_{shard_idx:04d}.parquet"), compression="zstd", compression_level=10)
df.close()
print(f"DONE: {n_ok} ok, {n_err} err in {(time.time()-t0)/60:.1f} min", flush=True)
