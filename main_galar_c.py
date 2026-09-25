import os
import json
import random
import argparse
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import vgg16_bn, MLP
from train import train, test
from uq import conf_calibration, to_serializable
from datasets_galar_c import load_galar


class MODEXForGALAR(nn.Module):
    def __init__(self, num_classes, dropout_rate, spect_norm, device,
                 hidden_dim, num_layers, activation):
        super().__init__()
        self.device = device
        self.activation = activation

        backbone = vgg16_bn(num_classes, dropout_rate, spect_norm)
        self.f = backbone.features
        self.g_alpha = backbone.classifier
        embedding_dim = 512

        self.g_tau = MLP(embedding_dim, num_classes, hidden_dim, num_layers, dropout_rate, spect_norm)
        self.g_w = MLP(embedding_dim, num_classes, hidden_dim, num_layers, dropout_rate, spect_norm)

    def _embed(self, x):
        features = self.f(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        return features.view(features.size(0), -1)

    def forward(self, x):
        embedded = self._embed(x)

        if self.activation == "softplus":
            alpha = F.softplus(self.g_alpha(embedded))
            tau = F.softplus(self.g_tau(embedded))
        elif self.activation == "exp":
            alpha = torch.exp(self.g_alpha(embedded))
            tau = torch.exp(self.g_tau(embedded))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        alpha = alpha + 1e-6
        tau = tau + 1e-6
        w = F.softmax(self.g_w(embedded), dim=1)
        return alpha, w, tau

    def get_logits(self, x):
        return self.g_w(self._embed(x))


def get_args():
    parser = argparse.ArgumentParser(description="MODEX on GALAR")

    # Data
    parser.add_argument("--tar_index", type=str, default=None,
                        help="Prebuilt tar-offset index (.pkl). When set, frames are read from the tar shards "
                             "and --image_dir is only a fallback for frames not in any shard.")
    parser.add_argument("--image_dir", type=str, default="",
                        help="Comma-separated loose-frame dirs. Required without --tar_index; with it, fallback only.")
    parser.add_argument("--shard_dir", type=str, default=None,
                        help="Directory of .tar shards; defaults to the shards_dir stored in the index.")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Cap training frames per class (val/test untouched). Default: no cap.")
    parser.add_argument("--check_data_only", action="store_true",
                        help="Build the datasets, decode a few frames, write data_report.json, then exit (no GPU needed).")
    parser.add_argument("--split_path", type=str, required=True)
    parser.add_argument("--training_features", type=str, default="section")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=12)

    # Training
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--tau_guidance", type=float, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")

    # MODEX head
    parser.add_argument("--activation", type=str, default="exp")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Writable directory for results + checkpoints (RW PVC)")

    return parser.parse_args()


def main():
    args = get_args()
    if not args.tar_index and not args.image_dir:
        raise SystemExit("Pass --tar_index (recommended) or --image_dir.")
    device = args.device if torch.cuda.is_available() else "cpu"

    results_dir = os.path.join(args.output_dir, "results_modex_galar")
    models_dir = os.path.join(args.output_dir, "models_modex_galar")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"[main_galar] loading GALAR: task={args.training_features} fold={args.fold} "
          f"image_size={args.image_size} batch_size={args.batch_size} num_workers={args.num_workers}")

    trainloader, validloader, testloader, label_cols, data_report = load_galar(
        image_dirs=args.image_dir,
        split_path=args.split_path,
        training_features=args.training_features,
        fold=args.fold,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        tar_index=args.tar_index,
        shard_dir=args.shard_dir,
        max_per_class=args.max_per_class,
    )
    num_classes = len(label_cols)
    print(f"[main_galar] num_classes={num_classes} labels={label_cols}")
    print(f"[main_galar] train={len(trainloader.dataset)} val={len(validloader.dataset)} test={len(testloader.dataset)}")

    run_name = f"galar_{args.training_features}_fold{args.fold}"
    if data_report is not None:
        with open(os.path.join(results_dir, run_name + "_data_report.json"), "w") as f:
            json.dump(to_serializable(data_report), f, indent=2)

    if args.check_data_only:
        for name, loader in (("train", trainloader), ("val", validloader), ("test", testloader)):
            ds = loader.dataset
            for j in np.random.default_rng(0).choice(len(ds), min(8, len(ds)), replace=False):
                x, y = ds[int(j)]
                assert x.shape == (3, args.image_size, args.image_size), x.shape
            print(f"[main_galar] decoded 8 random {name} frames OK")
        t0, n = time.time(), 0
        for x, _ in trainloader:
            n += x.size(0)
            if n >= 20 * args.batch_size:
                break
        print(f"[main_galar] loader throughput: {n / (time.time() - t0):.0f} img/s "
              f"with {args.num_workers} workers (includes worker start-up)")
        print(f"[main_galar] check_data_only: done, report under {results_dir}")
        return

    model = MODEXForGALAR(
        num_classes=num_classes,
        dropout_rate=args.dropout_rate,
        spect_norm=True,
        device=device,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
    )

    train(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        num_epochs=args.num_epochs,
        tau_guidance=args.tau_guidance,
        trainloader=trainloader,
        validloader=validloader,
        num_classes=num_classes,
        device=device,
        ckpt_path=os.path.join(models_dir, run_name + "_last_ckpt.pt"),
    )

    test_acc = test(model, testloader, device)
    conf_auroc, conf_aupr, brier = conf_calibration(model, testloader, device)

    result = {
        "args": vars(args),
        "training_features": args.training_features,
        "fold": args.fold,
        "num_classes": num_classes,
        "label_cols": label_cols,
        "Test Accuracy": test_acc,
        "CONF AUROC": conf_auroc,
        "CONF AUPR": conf_aupr,
        "BRIER": brier,
    }
    print(result)

    rand_id = random.randint(10000, 99999)
    suffix = f"_galar_{args.training_features}_fold{args.fold}_lr{args.learning_rate}_bs{args.batch_size}_id{rand_id}"
    with open(os.path.join(results_dir, suffix + ".json"), "w") as f:
        json.dump(to_serializable(result), f, indent=2)
    torch.save(model.state_dict(), os.path.join(models_dir, suffix + ".pt"))
    print(f"[main_galar] saved results + checkpoint under {args.output_dir}")


if __name__ == "__main__":
    main()
