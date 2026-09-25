import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


NON_LABEL_COLUMNS = {"path", "unknown"}


def _label_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_LABEL_COLUMNS]


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
    **_ignored,
):
    if isinstance(image_dirs, str):
        image_dirs = [d.strip() for d in image_dirs.split(",") if d.strip()]

    train_csv = os.path.join(split_path, training_features, f"split_{fold}", "train.csv")
    val_csv = os.path.join(split_path, training_features, f"split_{fold}", "val.csv")
    test_csv = os.path.join(split_path, training_features, "test.csv")
    for p in (train_csv, val_csv, test_csv):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing split CSV: {p}")

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

    return train_loader, val_loader, test_loader, label_cols
