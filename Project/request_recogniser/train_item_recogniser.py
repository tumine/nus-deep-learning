"""Fine-tune an ImageNet-pretrained ConvNeXt classifier on YOLO-labelled items.

The dataset is expected to contain YOLO detection splits such as
``train/images`` and ``train/labels``.  Each labelled bounding box becomes one
classification sample, cropped with a small surrounding context margin.

Example:
    python train_item_recogniser.py --data request_dataset --epochs 60
"""

import argparse
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class YoloCropDataset(Dataset):
    """Loads object crops defined by YOLO labels for image classification."""

    def __init__(self, samples: list[tuple[Path, tuple[float, float, float, float], int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, box, label = self.samples[index]
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        width, height = image.size
        x_center, y_center, box_width, box_height = box
        left = max(0, int((x_center - box_width / 2) * width))
        top = max(0, int((y_center - box_height / 2) * height))
        right = min(width, int((x_center + box_width / 2) * width))
        bottom = min(height, int((y_center + box_height / 2) * height))
        if right <= left or bottom <= top:
            raise ValueError(f"Invalid bounding box in {image_path}")
        return self.transform(image.crop((left, top, right, bottom))), label


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="ConvNeXt small-item classifier trainer")
    parser.add_argument("--data", type=Path, default=project_root / "request_dataset",
                        help="YOLO dataset root containing train/valid/test image and label folders")
    parser.add_argument("--output", type=Path, default=project_root / "training_output",
                        help="Directory for checkpoints and training_metrics.png")
    parser.add_argument("--class-names", nargs="+", default=["pencil", "eraser", "building_block"],
                        help="Class names in YOLO numeric-id order")
    parser.add_argument("--backbone", choices=("convnext_small", "convnext_base"), default="convnext_small")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Reduce to 8 if ConvNeXt-Small at 384px exhausts 12GB VRAM")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_for_label(label_path: Path, image_dir: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{label_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def read_samples(data_root: Path, num_classes: int) -> list[tuple[Path, tuple[float, float, float, float], int]]:
    samples = []
    for split in ("train", "valid", "val", "test"):
        label_dir, image_dir = data_root / split / "labels", data_root / split / "images"
        if not label_dir.is_dir() or not image_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_for_label(label_path, image_dir)
            if image_path is None:
                print(f"Warning: image missing for {label_path}")
                continue
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    print(f"Warning: skipping malformed label {label_path}:{line_number}")
                    continue
                class_id = int(parts[0])
                coordinates = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                if not 0 <= class_id < num_classes or not all(0 < value <= 1 for value in coordinates[2:]):
                    print(f"Warning: skipping invalid label {label_path}:{line_number}")
                    continue
                samples.append((image_path, coordinates, class_id))
    if not samples:
        raise RuntimeError(f"No labelled image crops found under {data_root}")
    return samples


def stratified_split(samples, val_ratio: float, seed: int):
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample[2]].append(sample)
    rng = random.Random(seed)
    train_samples, val_samples = [], []
    for class_id, class_samples in by_class.items():
        if len(class_samples) < 2:
            raise ValueError(f"Class ID {class_id} has fewer than 2 labelled objects; cannot split it.")
        rng.shuffle(class_samples)
        val_count = max(1, round(len(class_samples) * val_ratio))
        val_count = min(val_count, len(class_samples) - 1)
        val_samples.extend(class_samples[:val_count])
        train_samples.extend(class_samples[val_count:])
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def build_transforms(image_size: int):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, validation_transform


def build_model(backbone: str, num_classes: int) -> nn.Module:
    if backbone == "convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    else:
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    return model


def macro_f1(predictions: Iterable[int], targets: Iterable[int], num_classes: int) -> float:
    scores = []
    for class_id in range(num_classes):
        true_positive = sum(pred == class_id and target == class_id for pred, target in zip(predictions, targets))
        false_positive = sum(pred == class_id and target != class_id for pred, target in zip(predictions, targets))
        false_negative = sum(pred != class_id and target == class_id for pred, target in zip(predictions, targets))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(sum(scores) / len(scores))


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp_enabled=False):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, correct, total = 0.0, 0, 0
    predictions, targets = [], []
    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with amp_context:
                logits = model(images)
                loss = criterion(logits, labels)
            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            batch_predictions = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (batch_predictions == labels).sum().item()
            total += labels.size(0)
            predictions.extend(batch_predictions.cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, macro_f1(predictions, targets, model.classifier[-1][-1].out_features)


def plot_metrics(history: dict[str, list[float]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Matplotlib is required to write training_metrics.png. Install it with: pip install matplotlib") from error
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[1].plot(epochs, history["train_acc"], label="train accuracy")
    axes[1].plot(epochs, history["val_acc"], label="validation accuracy")
    axes[1].plot(epochs, history["val_f1"], label="validation macro F1")
    axes[1].set(title="Validation quality", xlabel="Epoch", ylabel="Score")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not args.data.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    if amp_enabled:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    all_samples = read_samples(args.data, len(args.class_names))
    train_samples, val_samples = stratified_split(all_samples, args.val_ratio, args.seed)
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"All labelled crops: {len(all_samples)}; train: {len(train_samples)}; validation: {len(val_samples)}")
    print("Class counts:", {args.class_names[key]: value for key, value in sorted(Counter(sample[2] for sample in all_samples).items())})

    train_transform, val_transform = build_transforms(args.image_size)
    loader_options = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    if args.workers > 0:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(YoloCropDataset(train_samples, train_transform), batch_size=args.batch_size,
                              shuffle=True, **loader_options)
    val_loader = DataLoader(YoloCropDataset(val_samples, val_transform), batch_size=args.batch_size,
                            shuffle=False, **loader_options)

    model = build_model(args.backbone, len(args.class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = defaultdict(list)
    best_f1, best_loss, best_epoch, stale_epochs = -1.0, float("inf"), 0, 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp_enabled)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device, amp_enabled=amp_enabled)
        scheduler.step()
        for key, value in {"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc,
                           "val_acc": val_acc, "val_f1": val_f1}.items():
            history[key].append(value)
        print(f"Epoch {epoch:03d}/{args.epochs} | train loss/acc {train_loss:.4f}/{train_acc:.3f} | "
              f"val loss/acc/F1 {val_loss:.4f}/{val_acc:.3f}/{val_f1:.3f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | {time.time() - started:.1f}s")

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "class_to_idx": {name: index for index, name in enumerate(args.class_names)},
                        "idx_to_class": args.class_names,
                        "best_epoch": epoch, "val_f1": val_f1, "val_loss": val_loss,
                        "backbone": args.backbone, "image_size": args.image_size}, args.output / "best_model.pth")
        if val_loss < best_loss:
            best_loss, stale_epochs = val_loss, 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping: validation loss did not improve for {args.patience} epochs.")
                break
    plot_metrics(history, args.output / "training_metrics.png")
    print(f"Finished. Best checkpoint: {args.output / 'best_model.pth'} (epoch {best_epoch}, macro F1 {best_f1:.3f})")


if __name__ == "__main__":
    main()