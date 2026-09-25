import io
import os
import pickle
import time
from collections import Counter

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


NON_LABEL_COLUMNS = {"path", "unknown"}

# Official GALAR multiclass tasks and their class columns, in the order used by
# github.com/EKFZ-AI-Endoscopy/GalarCapsuleML (train.py / generate_folds.py).
TASK_LABELS = {
    "section": ["mouth", "esophagus", "stomach", "small intestine", "colon"],
    "technical_multiclass": ["good view", "reduced view", "no view"],
}


def _label_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_LABEL_COLUMNS]


def _norm_key(name):
    """Canonical frame key '<video>/frame_NNNNNN.PNG'. The repaired 41_to_50 shard
    has a leading './'; same normalisation the index builder (vce-triage
    datasets/galar.py) applied, so CSV paths and index keys match."""
    return os.path.normpath(str(name)).lstrip("./")


def _lookup_key(path, entries):
    """Index key for a CSV path. Official split CSVs zero-pad the video id
    ('003/frame_...'); tolerate either padding on the index side."""
    if path in entries:
        return path
    video, _, fname = path.partition("/")
    if video.isdigit():
        for alt in (str(int(video)), video.zfill(3)):
            key = f"{alt}/{fname}"
            if key in entries:
                return key
    return None


def _read_split_csv(csv_path, label_cols, filter_unknown=False):
    """Returns (paths, targets, stats). Label = the one active class column, as in
    the official loader. Rows with zero or several active classes are dropped
    (the official loader's argmax would silently call them class 0) and counted.
    unknown != 0 rows are only dropped with filter_unknown (official: kept)."""
    header = pd.read_csv(csv_path, nrows=0).columns
    missing = [c for c in label_cols if c not in header]
    if missing:
        raise ValueError(f"{csv_path} lacks label columns {missing}; columns are {list(header)}")
    df = pd.read_csv(csv_path, dtype={"unknown": "str"} if "unknown" in header else None)
    stats = {"columns": list(header), "csv_rows": len(df)}

    if "unknown" in df.columns:
        is_unknown = df["unknown"].fillna("0") != "0"
        stats["unknown_rows"] = int(is_unknown.sum())
        if filter_unknown and is_unknown.any():
            df = df[~is_unknown].reset_index(drop=True)

    onehot = df[label_cols].values.astype(np.int64)
    active = onehot.sum(axis=1)
    stats["rows_no_label"] = int((active == 0).sum())
    stats["rows_multi_label"] = int((active > 1).sum())
    keep = active == 1
    if not keep.all():
        print(f"[datasets_galar] {csv_path}: dropped {int((~keep).sum())}/{len(df)} rows without exactly one "
              f"active label ({stats['rows_no_label']} none, {stats['rows_multi_label']} several)")
        df = df.loc[keep].reset_index(drop=True)
        onehot = onehot[keep]
    return df["path"].map(_norm_key).tolist(), onehot.argmax(axis=1).astype(np.int64), stats


def load_tar_index(tar_index_path):
    """Load the prebuilt {'shards', 'shards_dir', 'index': name -> (shard_idx, offset, size)}."""
    t0 = time.time()
    with open(tar_index_path, "rb") as f:
        idx = pickle.load(f)
    print(f"[datasets_galar] loaded tar index {tar_index_path}: {len(idx['index'])} frames, "
          f"{len(idx['shards'])} shards ({time.time() - t0:.0f}s)")
    return idx


class GalarTarDataset(Dataset):
    """Reads GALAR frames straight out of the tar shards via the offset index.

    One large sequential file per ~10 videos instead of ~1.9M loose PNGs spread
    over several CephFS PVCs. Frames missing from the index are dropped and
    reported per video (never silently), optionally read from `fallback_dirs`.

    Samples are held as flat numpy arrays (not the 3.5M-entry dict) so DataLoader
    workers share them copy-on-write instead of each duplicating the index."""

    def __init__(self, csv_path, index, shards_dir, transform, label_cols,
                 fallback_dirs=(), max_per_class=None, seed=0, split="train", filter_unknown=False):
        self.transform = transform
        self.shard_paths = [os.path.join(shards_dir, s) for s in index["shards"]]
        self._fh = None  # per-process file descriptors, opened lazily
        self._fh_pid = None

        paths, targets, csv_stats = _read_split_csv(csv_path, label_cols, filter_unknown)
        entries = index["index"]
        keys = [_lookup_key(p, entries) for p in paths]
        in_index = np.array([k is not None for k in keys], dtype=bool)

        # Frames not in the tar shards: optionally recover from a loose-frame dir.
        # One listdir per affected video, never an os.path.exists per row.
        fallback = np.zeros(len(paths), dtype=bool)
        fallback_path = [None] * len(paths)
        missing = np.where(~in_index)[0]
        if len(missing) and fallback_dirs:
            listing = {}
            for i in missing:
                video, fname = paths[i].split("/", 1)
                videos = [video] + ([str(int(video)), video.zfill(3)] if video.isdigit() else [])
                for base in fallback_dirs:
                    for v in dict.fromkeys(videos):
                        key = (base, v)
                        if key not in listing:
                            try:
                                listing[key] = set(os.listdir(os.path.join(base, v)))
                            except OSError:
                                listing[key] = set()
                        if fname in listing[key]:
                            fallback[i] = True
                            fallback_path[i] = os.path.join(base, v, fname)
                            break
                    if fallback[i]:
                        break

        dropped = ~in_index & ~fallback
        self.report = {
            "csv": csv_path,
            **csv_stats,
            "rows": len(paths),
            "from_tar": int(in_index.sum()),
            "from_fallback_dir": int(fallback.sum()),
            "dropped": int(dropped.sum()),
            "fallback_per_video": dict(Counter(paths[i].split("/", 1)[0] for i in np.where(fallback)[0])),
            "dropped_per_video": dict(Counter(paths[i].split("/", 1)[0] for i in np.where(dropped)[0])),
        }
        print(f"[datasets_galar] {split} {csv_path}: {self.report['rows']} rows -> "
              f"{self.report['from_tar']} from tar, {self.report['from_fallback_dir']} from fallback dir, "
              f"{self.report['dropped']} DROPPED")
        if self.report["fallback_per_video"]:
            print(f"[datasets_galar]   fallback per video: {self.report['fallback_per_video']}")
        if self.report["dropped_per_video"]:
            print(f"[datasets_galar]   DROPPED per video: {self.report['dropped_per_video']}")

        keep = np.where(~dropped)[0]
        if max_per_class:
            rng = np.random.default_rng(seed)
            capped = []
            for c in np.unique(targets[keep]):
                idx_c = keep[targets[keep] == c]
                if len(idx_c) > max_per_class:
                    idx_c = rng.choice(idx_c, max_per_class, replace=False)
                capped.append(idx_c)
            keep = np.sort(np.concatenate(capped))
            print(f"[datasets_galar] {split}: capped to <= {max_per_class}/class -> {len(keep)} frames")

        # shard == -1 marks a fallback (loose file) sample.
        self.shard = np.full(len(keep), -1, dtype=np.int16)
        self.offset = np.zeros(len(keep), dtype=np.int64)
        self.size = np.zeros(len(keep), dtype=np.int64)
        self.fallback_paths = {}
        for j, i in enumerate(keep):
            if in_index[i]:
                self.shard[j], self.offset[j], self.size[j] = entries[keys[i]]
            else:
                self.fallback_paths[j] = fallback_path[i]
        self.targets = targets[keep]
        self.report["used"] = int(len(keep))
        self.report["class_counts"] = {label_cols[c]: int(n) for c, n in
                                       zip(*np.unique(self.targets, return_counts=True))}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None  # fds are per-process; spawned workers reopen lazily
        return state

    def __len__(self):
        return len(self.targets)

    def _read(self, j):
        si = int(self.shard[j])
        if si < 0:
            with open(self.fallback_paths[j], "rb") as f:
                return f.read()
        if self._fh is None or self._fh_pid != os.getpid():
            self._fh, self._fh_pid = {}, os.getpid()  # never reuse a parent's fds
        fd = self._fh.get(si)
        if fd is None:
            fd = os.open(self.shard_paths[si], os.O_RDONLY)
            self._fh[si] = fd
        return os.pread(fd, int(self.size[j]), int(self.offset[j]))

    def __getitem__(self, j):
        with Image.open(io.BytesIO(self._read(j))) as im:
            img = self.transform(im.convert("RGB"))
        return img, int(self.targets[j])


class GalarCSVDataset(Dataset):
    """Reads per-frame PNGs from one or more image_dirs (priority order).
    Handles phantom CSV rows via a per-path existence filter for videos
    whose disk file count is less than their CSV row count."""

    def __init__(self, csv_path, image_dirs, transform, label_cols,
                 available_videos=None, video_disk_counts=None):
        self.image_dirs = list(image_dirs)
        self.transform = transform
        self.label_cols = label_cols

        df = pd.read_csv(
            csv_path,
            dtype={"unknown": "str"} if "unknown" in pd.read_csv(csv_path, nrows=0).columns else None,
        )
        if "unknown" in df.columns:
            df = df[df["unknown"].fillna("0") == "0"].reset_index(drop=True)

        onehot = df[label_cols].values.astype(np.int64)
        row_sums = onehot.sum(axis=1)
        keep = row_sums == 1
        if not keep.all():
            dropped = int((~keep).sum())
            print(f"[datasets_galar] {csv_path}: dropped {dropped}/{len(df)} rows without exactly one active label")
            df = df.loc[keep].reset_index(drop=True)
            onehot = onehot[keep]

        if available_videos is not None:
            video_ids = df["path"].astype(str).str.split("/").str[0]
            keep = video_ids.isin(available_videos).to_numpy()
            if not keep.all():
                dropped = int((~keep).sum())
                print(f"[datasets_galar] {csv_path}: dropped {dropped}/{len(df)} rows for videos not present on disk")
                df = df.loc[keep].reset_index(drop=True)
                onehot = onehot[keep]

        if video_disk_counts is not None:
            per_video = df.groupby(df["path"].astype(str).str.split("/").str[0]).size()
            suspect_videos = [
                v for v in per_video.index
                if video_disk_counts.get(v, 0) < per_video[v]
            ]
            if suspect_videos:
                print(f"[datasets_galar] {csv_path}: scanning {len(suspect_videos)} video(s) with disk-vs-CSV gap: {suspect_videos}")
                video_series = df["path"].astype(str).str.split("/").str[0]
                mask_suspect = video_series.isin(suspect_videos).to_numpy()
                keep = np.ones(len(df), dtype=bool)
                for i in np.where(mask_suspect)[0]:
                    rel = os.path.normpath(str(df["path"].iloc[i]))
                    if not any(os.path.exists(os.path.join(b, rel)) for b in self.image_dirs):
                        keep[i] = False
                if not keep.all():
                    dropped = int((~keep).sum())
                    print(f"[datasets_galar] {csv_path}: dropped {dropped} phantom-frame rows")
                    df = df.loc[keep].reset_index(drop=True)
                    onehot = onehot[keep]

        self.paths = df["path"].tolist()
        self.targets = onehot.argmax(axis=1).astype(np.int64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel = os.path.normpath(str(self.paths[idx]))
        for base in self.image_dirs:
            full = os.path.join(base, rel)
            if os.path.exists(full):
                with Image.open(full) as im:
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    img = self.transform(im)
                return img, int(self.targets[idx])
        raise FileNotFoundError(f"[datasets_galar] {rel} not found under any of {self.image_dirs}")


def build_transforms(image_size: int):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return train_tf, eval_tf


def _scan_available_videos(image_dirs):
    videos = set()
    for base in image_dirs:
        try:
            for d in os.listdir(base):
                if os.path.isdir(os.path.join(base, d)):
                    videos.add(d)
        except FileNotFoundError:
            print(f"[datasets_galar] image_dir does not exist, skipping: {base}")
    return videos


def _scan_disk_counts_per_video(image_dirs):
    """Max PNG count per video across all image_dirs. Used to detect phantom CSV rows."""
    counts = {}
    for base in image_dirs:
        try:
            for d in os.listdir(base):
                p = os.path.join(base, d)
                if not os.path.isdir(p):
                    continue
                try:
                    n = sum(1 for f in os.listdir(p) if f.endswith(".PNG"))
                except OSError:
                    n = 0
                counts[d] = max(counts.get(d, 0), n)
        except FileNotFoundError:
            continue
    return counts


def load_galar(
    image_dirs,
    split_path: str,
    training_features: str,
    fold: int,
    batch_size: int = 64,
    image_size: int = 256,
    num_workers: int = 8,
    tar_index: str = None,
    shard_dir: str = None,
    max_per_class: int = None,
    filter_unknown: bool = False,
    **_ignored,
):
    """With `tar_index`, frames are read from the tar shards and `image_dirs` is only
    a fallback for frames absent from the shards. Without it, the legacy loose-PNG
    path below is used. Returns (train_loader, val_loader, test_loader, label_cols, report)."""
    if isinstance(image_dirs, str):
        image_dirs = [d.strip() for d in image_dirs.split(",") if d.strip()]
    image_dirs = list(image_dirs or [])

    train_csv = os.path.join(split_path, training_features, f"split_{fold}", "train.csv")
    val_csv = os.path.join(split_path, training_features, f"split_{fold}", "val.csv")
    test_csv = os.path.join(split_path, training_features, "test.csv")
    for p in (train_csv, val_csv, test_csv):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing split CSV: {p}")

    if tar_index:
        if training_features not in TASK_LABELS:
            raise ValueError(f"Unknown multiclass task {training_features!r}; expected one of {list(TASK_LABELS)}")
        return _load_galar_tar(train_csv, val_csv, test_csv, TASK_LABELS[training_features], image_dirs,
                               tar_index, shard_dir, batch_size, image_size, num_workers,
                               max_per_class, filter_unknown)

    peek = pd.read_csv(train_csv, nrows=0)
    label_cols = _label_columns(peek)
    if not label_cols:
        raise ValueError(f"No label columns in {train_csv}. Columns: {list(peek.columns)}")

    available_videos = _scan_available_videos(image_dirs)
    disk_counts = _scan_disk_counts_per_video(image_dirs)
    print(f"[datasets_galar] image_dirs (priority order): {image_dirs}")
    print(f"[datasets_galar] found {len(available_videos)} unique video folders across image_dirs: "
          f"{sorted(available_videos, key=lambda s: int(s) if s.isdigit() else s)}")

    train_tf, eval_tf = build_transforms(image_size)

    train_ds = GalarCSVDataset(train_csv, image_dirs, train_tf, label_cols, available_videos, disk_counts)
    val_ds = GalarCSVDataset(val_csv, image_dirs, eval_tf, label_cols, available_videos, disk_counts)
    test_ds = GalarCSVDataset(test_csv, image_dirs, eval_tf, label_cols, available_videos, disk_counts)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, label_cols, None


def _load_galar_tar(train_csv, val_csv, test_csv, label_cols, fallback_dirs, tar_index,
                    shard_dir, batch_size, image_size, num_workers, max_per_class, filter_unknown):
    index = load_tar_index(tar_index)
    shards_dir = shard_dir or index["shards_dir"]
    missing_shards = [s for s in index["shards"] if not os.path.isfile(os.path.join(shards_dir, s))]
    if missing_shards:
        raise FileNotFoundError(f"Shards listed in the index are missing under {shards_dir}: {missing_shards}")
    print(f"[datasets_galar] shards_dir={shards_dir} fallback_dirs={fallback_dirs or 'none'}")

    train_tf, eval_tf = build_transforms(image_size)
    common_ds = dict(index=index, shards_dir=shards_dir, label_cols=label_cols, fallback_dirs=fallback_dirs,
                     filter_unknown=filter_unknown)
    # The class cap only thins the training set; val/test keep the true distribution.
    train_ds = GalarTarDataset(train_csv, transform=train_tf, max_per_class=max_per_class, split="train", **common_ds)
    val_ds = GalarTarDataset(val_csv, transform=eval_tf, split="val", **common_ds)
    test_ds = GalarTarDataset(test_csv, transform=eval_tf, split="test", **common_ds)
    del index, common_ds  # the full dict is ~GBs in RAM; datasets keep compact arrays

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True,
                              persistent_workers=num_workers > 0, **common)
    val_loader = DataLoader(val_ds, shuffle=False, persistent_workers=num_workers > 0, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    report = {"train": train_ds.report, "val": val_ds.report, "test": test_ds.report}
    return train_loader, val_loader, test_loader, label_cols, report
