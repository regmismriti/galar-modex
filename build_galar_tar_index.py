"""Build the GALAR tar index used by main_galar_c.py --tar_index.

Scans the headers of every .tar shard (no image data is read) and writes
{"shards": [...], "shards_dir": ..., "index": {"<video>/frame_NNNNNN.PNG": (shard_idx, offset, size)}}.
One process per shard. Skips the build if --out already exists.
"""
import argparse
import os
import pickle
import time
from multiprocessing import Pool
import tarfile


def scan_shard(path):
    t0 = time.time()
    frames = {}
    with tarfile.open(path, "r:") as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith((".png", ".jpg", ".jpeg")):
                # the repaired 41_to_50 shard stores names with a leading './'
                frames[m.name.lstrip("./")] = (m.offset_data, m.size)
    print(f"[index] {os.path.basename(path)}: {len(frames)} frames ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    if os.path.isfile(a.out):
        print(f"[index] {a.out} already exists, not rebuilding", flush=True)
        return

    shards = sorted(f for f in os.listdir(a.shards_dir) if f.endswith(".tar"))
    print(f"[index] scanning {len(shards)} shards in {a.shards_dir}: {shards}", flush=True)
    with Pool(min(a.workers, len(shards))) as pool:
        per_shard = pool.map(scan_shard, [os.path.join(a.shards_dir, s) for s in shards], chunksize=1)

    index = {}
    for si, frames in enumerate(per_shard):
        for name, (off, size) in frames.items():
            index[name] = (si, off, size)
    videos = sorted({k.split("/", 1)[0] for k in index}, key=lambda v: int(v) if v.isdigit() else v)
    print(f"[index] {len(index)} frames from {len(videos)} videos: {videos}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"shards": shards, "shards_dir": a.shards_dir, "index": index}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, a.out)
    print(f"[index] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
