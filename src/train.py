import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import sys
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.append(CURR_DIR)

from models import create_model
from dataset import build_dataloaders


def parse_args():
    ap = argparse.ArgumentParser("3-class branded image classifier training")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "vgg16"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--feature-extract", action="store_true")
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Early stopping
    ap.add_argument("--es-metric", type=str, default="val_acc", choices=["val_acc", "val_loss"], help="early stopping metric")
    ap.add_argument("--es-patience", type=int, default=5, help="early stopping patience epochs")
    ap.add_argument("--es-min-delta", type=float, default=1e-4, help="minimum improvement for early stopping")
    # Class imbalance handling
    ap.add_argument("--class-weighted-loss", action="store_true", help="use class weights in CE loss")
    ap.add_argument("--weighted-sampler", action="store_true", help="use WeightedRandomSampler for training data")
    return ap.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, mixed_precision=False):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in tqdm(dataloader, desc="train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        if mixed_precision:
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="val", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    device = args.device
    torch.backends.cudnn.benchmark = device.startswith("cuda")

    dataloaders, dataset_sizes, class_names = build_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    num_classes = len(class_names)
    model = create_model(args.model, num_classes=num_classes, pretrained=True, feature_extract=args.feature_extract)
    model = model.to(device)

    if args.feature_extract:
        params_to_update = [p for p in model.parameters() if p.requires_grad]
    else:
        params_to_update = model.parameters()

    optimizer = AdamW(params_to_update, lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # ----- Class imbalance: class weights and/or weighted sampler -----
    def get_labels_from_dataset(ds):
        # ds can be ImageFolder or Subset(ImageFolder)
        if hasattr(ds, "dataset") and hasattr(ds, "indices"):
            base = ds.dataset
            return [base.targets[i] for i in ds.indices]
        elif hasattr(ds, "samples") and hasattr(ds, "targets"):
            return list(ds.targets)
        elif hasattr(ds, "targets"):
            return list(ds.targets)
        else:
            # fallback (slow): iterate samples attribute
            if hasattr(ds, "samples"):
                return [lbl for _, lbl in ds.samples]
            raise RuntimeError("Unsupported dataset type for class counting")

    train_ds = dataloaders["train"].dataset
    labels = get_labels_from_dataset(train_ds)
    num_classes = len(class_names)
    counts = [0] * num_classes
    for lbl in labels:
        counts[lbl] += 1
    if args.class_weighted_loss:
        total = sum(counts)
        class_weights = [total / (c if c > 0 else 1) for c in counts]
        w = torch.tensor(class_weights, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=w.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    if args.weighted_sampler:
        import math
        inv_freq = [1 / (c if c > 0 else 1) for c in counts]
        sample_weights = [inv_freq[lbl] for lbl in labels]
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        dataloaders["train"] = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=device.startswith("cuda"))

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.startswith("cuda"))

    best_acc = 0.0
    run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join("artifacts", f"{args.model}_best")
    os.makedirs(save_dir, exist_ok=True)

    # ----- Early stopping -----
    best_metric = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device, scaler, args.mixed_precision)
        val_loss, val_acc = eval_one_epoch(model, dataloaders["val"], criterion, device)

        scheduler.step()
        print(f"epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model,
                "class_names": class_names,
                "img_size": args.img_size,
                "feature_extract": args.feature_extract,
            }, os.path.join(save_dir, "best.pt"))

            model_cpu = model.cpu().eval()
            example = torch.randn(1, 3, args.img_size, args.img_size)
            traced = torch.jit.trace(model_cpu, example)
            traced.save(os.path.join(save_dir, "best_ts.pt"))

            metadata = {
                "model_name": args.model,
                "class_names": class_names,
                "img_size": args.img_size,
                "feature_extract": args.feature_extract,
                "run_name": run_name,
            }
            with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Early stopping check
        current_metric = val_acc if args.es_metric == "val_acc" else -val_loss  # maximize
        if best_metric is None or (current_metric - best_metric) > args.es_min_delta:
            best_metric = current_metric
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.es_patience:
                print(f"Early stopping triggered at epoch {epoch} (patience {args.es_patience}).")
                break

    print(f"best val acc: {best_acc:.4f}. artifacts saved to {save_dir}")


if __name__ == "__main__":
    main()