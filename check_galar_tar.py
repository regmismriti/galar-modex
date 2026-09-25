"""Pre-flight check for main_galar_c.py --tar_index, with no GPU and no training.

Builds train/val/test through load_galar exactly as main_galar_c.py does (its
per-video tar / image_dir / dropped counts are printed while loading), decodes
a few frames from each split and times the train DataLoader.
"""
import argparse
import os
import time

import numpy as np

from datasets_galar_c import load_galar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tar_index", required=True)
    p.add_argument("--shard_dir", required=True)
    p.add_argument("--image_dir", default="")
    p.add_argument("--split_path", required=True)
    p.add_argument("--training_features", default="section")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    a = p.parse_args()

    task_dir = os.path.join(a.split_path, a.training_features)
    print(f"[check] {task_dir}: {sorted(os.listdir(task_dir))}")

    t0 = time.time()
    loaders = load_galar(
        image_dirs=a.image_dir, split_path=a.split_path, training_features=a.training_features,
        fold=a.fold, batch_size=a.batch_size, image_size=a.image_size, num_workers=a.num_workers,
        tar_index=a.tar_index, shard_dir=a.shard_dir,
    )
    train_loader, val_loader, test_loader, label_cols = loaders
    print(f"[check] datasets built in {time.time() - t0:.0f}s; labels={label_cols}")

    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        ds = loader.dataset
        counts = np.bincount(ds.targets, minlength=len(label_cols))
        print(f"[check] {name}: {len(ds)} frames, per class {dict(zip(label_cols, counts.tolist()))}")
        for j in np.random.default_rng(0).choice(len(ds), min(8, len(ds)), replace=False):
            x, _ = ds[int(j)]
            assert tuple(x.shape) == (3, a.image_size, a.image_size), x.shape
        print(f"[check] {name}: decoded 8 random frames OK")

    t0, n = time.time(), 0
    for x, _ in train_loader:
        n += x.size(0)
        if n >= 30 * a.batch_size:
            break
    print(f"[check] train loader: {n / (time.time() - t0):.0f} img/s with {a.num_workers} workers "
          f"(includes worker start-up)")


if __name__ == "__main__":
    main()
