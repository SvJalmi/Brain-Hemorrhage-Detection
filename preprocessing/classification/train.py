import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from preprocessing.classification.dataset import (
    LABELS, TBIDataset, compute_pos_weight, get_transforms,
)

SEED = 42


def make_loader(split, args, train):
    ds = TBIDataset(split, transform=get_transforms(train, args.img_size))
    if args.subset and len(ds) > args.subset:
        rng = np.random.default_rng(SEED)
        ds = Subset(ds, rng.choice(len(ds), args.subset, replace=False).tolist())
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )


def per_class_auroc(targets, probs):
    out = {}
    for i, name in enumerate(LABELS):
        if len(np.unique(targets[:, i])) < 2:
            out[name] = float("nan")   # only happens on tiny debug subsets
        else:
            out[name] = float(roc_auc_score(targets[:, i], probs[:, i]))
    return out


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    probs, targets = [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total += criterion(logits, labels).item() * len(images)
        n += len(images)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.cpu().numpy())
    probs, targets = np.concatenate(probs), np.concatenate(targets)
    aucs = per_class_auroc(targets, probs)
    return total / n, float(np.nanmean(list(aucs.values()))), aucs


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total, n, start = 0.0, 0, time.time()
    for step, (images, labels, _) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(images)
        n += len(images)
        if step % 50 == 0 or step == len(loader):
            per = (time.time() - start) / step
            print(f"  step {step}/{len(loader)}  loss {total / n:.4f}  "
                  f"{per:.2f}s/batch  ~{per * len(loader) / 60:.1f} min/epoch", flush=True)
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="efficientnet_b0")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--subset", type=int, default=0, help="cap images per split (debug)")
    p.add_argument("--out", default="runs/effb0")
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("Device:", device, "| AMP:", use_amp, "| Args:", vars(args))

    train_loader = make_loader("train", args, train=True)
    val_loader = make_loader("validation", args, train=False)

    model = timm.create_model(args.model, pretrained=True, num_classes=len(LABELS)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight().to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    writer = SummaryWriter(str(out / "tb"))

    best, bad, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_auc, aucs = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}: train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val macro-AUROC {val_auc:.4f}")
        print("  per-class AUROC:", {k: round(v, 3) for k, v in aucs.items()})
        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("auroc/val_macro", val_auc, epoch)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
                        "val_macro_auroc": val_auc, "val_auroc": aucs})
        (out / "history.json").write_text(json.dumps(history, indent=2))

        if val_auc > best:
            best, bad = val_auc, 0
            torch.save({"model": args.model, "img_size": args.img_size,
                        "state_dict": model.state_dict(), "epoch": epoch,
                        "val_macro_auroc": val_auc}, out / "best.pt")
            print("  saved best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    writer.close()
    print(f"\nDone. Best validation macro-AUROC: {best:.4f}")
    print("The test set has NOT been touched. It is reserved for the final evaluation.")


if __name__ == "__main__":
    main()