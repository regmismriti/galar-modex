import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import uq
from uq import *


def smooth_one_hot(y, classes, smoothing=0.1):
    confidence = 1.0 - smoothing
    label = torch.full((y.size(0), classes), smoothing / (classes - 1), device=y.device)
    label.scatter_(1, y.unsqueeze(1), confidence)
    return label


def MoD_loss(alpha, w, tau, y, tau_guidance):
    B, K = alpha.shape

    mu, var = compute_moments(alpha, w, tau)
    y_onehot = F.one_hot(y, num_classes=K).float()

    loss_main = ((mu - y_onehot) ** 2).sum(dim=1)
    loss_main += ((y_onehot - w) ** 2).sum(dim=1)

    target_tau = smooth_one_hot(y, K, smoothing=0.1)
    tau_log_probs = F.log_softmax(tau, dim=1)
    reg_tau_guide = tau_guidance * F.kl_div(tau_log_probs, target_tau, reduction="batchmean")

    total_loss = loss_main.mean() + reg_tau_guide
    return total_loss


def train(model, learning_rate, weight_decay, step_size, num_epochs, tau_guidance, trainloader, validloader, num_classes, device, ckpt_path=None):
    """ckpt_path: if given, state is saved there after every epoch and training
    resumes from it when it already exists (survives pod eviction/restart)."""

    use_amp = device.startswith("cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    patience, epochs_no_improve = 30, 0
    best_acc = -1.0
    start_epoch = 0
    model.to(device)

    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_acc = state["best_acc"]
        epochs_no_improve = state["epochs_no_improve"]
        print(f"Resumed from {ckpt_path} at epoch {start_epoch + 1} (best val acc {best_acc:.2f}%)")
        if epochs_no_improve >= patience:
            print("Checkpoint had already triggered early stopping.")
            return model

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss_sum, running_seen = 0.0, 0
        epoch_start = time.time()
        if use_amp:
            torch.cuda.reset_peak_memory_stats()

        for x, y in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast("cuda", enabled=True):
                    alpha, w, tau = model(x)
                    loss = MoD_loss(alpha, w, tau, y, tau_guidance)
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=5.0)

                scaler.step(optimizer)
                scaler.update()

            else:
                alpha, w, tau = model(x)
                loss = MoD_loss(alpha, w, tau, y, tau_guidance)
                loss.backward()

                clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()

            bs = x.size(0)
            running_loss_sum += float(loss.detach().item()) * bs
            running_seen += bs

        scheduler.step()

        avg_train_loss = running_loss_sum / max(1, running_seen)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} training finished. Avg Loss: {avg_train_loss:.4f}, Current LR: {current_lr:.6f}, "
              f"{epoch_time/60:.1f} min, {running_seen/max(epoch_time, 1e-6):.0f} img/s")
        if use_amp:
            print(f"Epoch {epoch+1} peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

        # -------------------
        # Validation
        # -------------------
        model.eval()
        val_loss_sum, correct_v, total_v = 0.0, 0, 0

        with torch.inference_mode():
            for x_v, y_v in validloader:
                x_v = x_v.to(device, non_blocking=True)
                y_v = y_v.to(device, non_blocking=True)

                with autocast("cuda", enabled=use_amp):
                    alpha_v, w_v, tau_v = model(x_v)
                    mu_v, _ = compute_moments(alpha_v, w_v, tau_v)

                    loss_v = MoD_loss(alpha_v, w_v, tau_v, y_v, tau_guidance)

                bs = x_v.size(0)
                val_loss_sum += float(loss_v.item()) * bs
                pred_v = mu_v.argmax(1)
                correct_v += (pred_v == y_v).sum().item()
                total_v += bs

        val_acc = 100.0 * correct_v / max(1, total_v)
        avg_val_loss = val_loss_sum / max(1, total_v)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if ckpt_path:
            tmp = ckpt_path + ".tmp"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "epochs_no_improve": epochs_no_improve,
            }, tmp)
            os.replace(tmp, ckpt_path)

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

    return model


def test(model, testloader, device):
    model.eval()
    total_t, correct_t1, correct_t2 = 0, 0, 0

    with torch.no_grad():
        for i, (x_t, y_t) in enumerate(tqdm(testloader)):
            x_t, y_t = x_t.to(device), y_t.to(device)

            alpha_t, w_t, tau_t = model(x_t)
            mu_t, var_t = compute_moments(alpha_t, w_t, tau_t)

            y_pred_t1 = mu_t.argmax(1)
            correct_t1 += (y_pred_t1 == y_t).sum().item()

            top2 = torch.topk(mu_t, k=2, dim=1).indices
            correct_t2 += (top2 == y_t.unsqueeze(1)).any(dim=1).sum().item()

            total_t += y_t.size(0)

    top1_acc = 100 * correct_t1 / total_t
    top2_acc = 100 * correct_t2 / total_t

    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-2 Accuracy: {top2_acc:.2f}%")

    return top1_acc, top2_acc
