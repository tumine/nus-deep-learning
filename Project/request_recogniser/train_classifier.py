"""Train a ConvNeXt classifier for pencil, eraser, and building-block images.

Expected annotation output is either class directories below ``collected_images``::

    collected_images/pencil/*.jpg
    collected_images/eraser/*.jpg
    collected_images/building_block/*.jpg

Or a ``labels.csv`` / ``annotations.csv`` / ``labels.json`` / ``annotations.json``
file in that directory. Manifest records need an image path field (``path``,
``file``, or ``image_path``) and a label field (``label``, ``class``, or
``category``).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = ("pencil", "eraser", "building_block")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent


class LabeledImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Image samples that can originate from folders or an annotation manifest."""

    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            return self.transform(image.convert("RGB")), label


def get_transforms(input_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Create ImageNet-normalized transforms suitable for vehicle camera captures."""
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, validation_transform


def discover_folder_samples(data_dir: Path) -> list[tuple[Path, int]]:
    """Read images from annotation-tool-created directories named after classes."""
    samples: list[tuple[Path, int]] = []
    seen: set[Path] = set()
    for label, class_name in enumerate(CLASS_NAMES):
        for class_dir in data_dir.rglob(class_name):
            if not class_dir.is_dir():
                continue
            for image_path in class_dir.rglob("*"):
                resolved = image_path.resolve()
                if image_path.suffix.lower() in IMAGE_EXTENSIONS and resolved not in seen:
                    samples.append((resolved, label))
                    seen.add(resolved)
    return samples


def read_manifest_records(manifest_path: Path) -> Iterable[dict[str, Any]]:
    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            yield from csv.DictReader(file)
        return

    with manifest_path.open("r", encoding="utf-8") as file:
        content = json.load(file)
    if isinstance(content, list):
        yield from content
    elif isinstance(content, dict):
        for key in ("annotations", "images", "data", "items"):
            if isinstance(content.get(key), list):
                yield from content[key]
                return
        yield from content.values()


def discover_manifest_samples(data_dir: Path) -> list[tuple[Path, int]]:
    """Support simple CSV/JSON manifests emitted by labelling utilities."""
    manifest_names = (
        "labels.csv", "annotations.csv", "labels.json", "annotations.json",
    )
    manifest_path = next((data_dir / name for name in manifest_names if (data_dir / name).is_file()), None)
    if manifest_path is None:
        return []

    samples: list[tuple[Path, int]] = []
    for record in read_manifest_records(manifest_path):
        if not isinstance(record, dict):
            continue
        image_name = next((record.get(key) for key in ("path", "file", "image_path", "image")), None)
        class_name = next((record.get(key) for key in ("label", "class", "category")), None)
        if not isinstance(image_name, str) or not isinstance(class_name, str):
            continue
        if class_name not in CLASS_NAMES:
            raise ValueError(f"{manifest_path} contains unsupported label: {class_name!r}")
        image_path = (data_dir / image_name).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image named by {manifest_path} does not exist: {image_path}")
        samples.append((image_path, CLASS_NAMES.index(class_name)))
    print(f"Loaded {len(samples)} labelled images from {manifest_path}")
    return samples


def load_samples(data_dir: Path) -> list[tuple[Path, int]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    samples = discover_folder_samples(data_dir)
    if not samples:
        samples = discover_manifest_samples(data_dir)
    if not samples:
        raise FileNotFoundError(
            "No labelled images found. Create pencil/, eraser/, and building_block/ "
            f"directories below {data_dir}, or add a supported labels/annotations CSV or JSON file."
        )
    counts = Counter(CLASS_NAMES[label] for _, label in samples)
    missing = [class_name for class_name in CLASS_NAMES if counts[class_name] == 0]
    if missing:
        raise ValueError(f"Missing labelled images for: {', '.join(missing)}")
    if any(count < 2 for count in counts.values()):
        raise ValueError("Each class needs at least two images for an 85%/15% split.")
    print("Dataset:", ", ".join(f"{name}={counts[name]}" for name in CLASS_NAMES))
    return samples


def split_samples(samples: list[tuple[Path, int]], validation_ratio: float, seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Create a reproducible, class-stratified train/validation split."""
    random_generator = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    validation_samples: list[tuple[Path, int]] = []
    for label in range(len(CLASS_NAMES)):
        class_samples = [sample for sample in samples if sample[1] == label]
        random_generator.shuffle(class_samples)
        validation_count = max(1, round(len(class_samples) * validation_ratio))
        validation_count = min(validation_count, len(class_samples) - 1)
        validation_samples.extend(class_samples[:validation_count])
        train_samples.extend(class_samples[validation_count:])
    random_generator.shuffle(train_samples)
    random_generator.shuffle(validation_samples)
    return train_samples, validation_samples


def build_model(variant: str, pretrained: bool) -> nn.Module:
    model_builders = {
        "tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "small": (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1),
        "base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
    }
    builder, weights = model_builders[variant]
    model = builder(weights=weights if pretrained else None)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, len(CLASS_NAMES))
    return model


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: AdamW | None = None, scaler: torch.amp.GradScaler | None = None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    amp_context = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with amp_context:
                logits = model(images)
                loss = criterion(logits, labels)
            if optimizer is not None:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    model.eval()
    for images, labels in loader:
        predictions = model(images.to(device, non_blocking=True)).argmax(dim=1).cpu().numpy()
        for actual, predicted in zip(labels.numpy(), predictions):
            matrix[actual, predicted] += 1

    print("\nClassification report")
    print(f"{'class':<16}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    for index, class_name in enumerate(CLASS_NAMES):
        true_positive = matrix[index, index]
        precision = true_positive / max(matrix[:, index].sum(), 1)
        recall = true_positive / max(matrix[index, :].sum(), 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-12)
        print(f"{class_name:<16}{precision:>10.3f}{recall:>10.3f}{f1_score:>10.3f}{matrix[index, :].sum():>10}")
    print("\nConfusion matrix (rows=true, columns=predicted)")
    print(" " * 16 + " ".join(f"{name[:8]:>8}" for name in CLASS_NAMES))
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:<16}" + " ".join(f"{value:>8}" for value in matrix[index]))


def export_onnx(model: nn.Module, output_path: Path, input_size: int, device: torch.device) -> None:
    model.eval()
    example = torch.randn(1, 3, input_size, input_size, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, example, output_path, input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
    )
    print(f"Exported ONNX model: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained ConvNeXt request recogniser.")
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "collected_images")
    parser.add_argument("--variant", choices=("tiny", "small", "base"), default="tiny")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download or load ImageNet weights.")
    parser.add_argument("--no-export-onnx", action="store_true")
    args = parser.parse_args()

    if not 0 < args.patience or args.epochs < 1:
        raise ValueError("--epochs and --patience must both be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Using CUDA: {torch.cuda.get_device_name(0)} (AMP enabled)")
    else:
        print("CUDA is unavailable; training will run on CPU without AMP.")

    input_size = 224
    batch_size = args.batch_size or (32 if args.variant == "base" else 64)
    samples = load_samples(args.data.resolve())
    train_samples, validation_samples = split_samples(samples, validation_ratio=0.15, seed=args.seed)
    print(f"Split: train={len(train_samples)}, validation={len(validation_samples)}")
    train_transform, validation_transform = get_transforms(input_size)
    loader_settings = {"batch_size": batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(LabeledImageDataset(train_samples, train_transform), shuffle=True, **loader_settings)
    validation_loader = DataLoader(LabeledImageDataset(validation_samples, validation_transform), shuffle=False, **loader_settings)

    model = build_model(args.variant, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights_dir = SCRIPT_DIR / "weights"
    best_path = weights_dir / "best_convnext.pth"
    best_accuracy = -1.0
    best_validation_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion, device)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs} | Train Loss {train_loss:.4f} | Train Acc {train_accuracy:.2%} | "
            f"Val Loss {validation_loss:.4f} | Val Acc {validation_accuracy:.2%} | LR {learning_rate:.2e}"
        )
        if validation_accuracy > best_accuracy:
            weights_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "classes": CLASS_NAMES, "variant": args.variant, "input_size": input_size}, best_path)
            best_accuracy = validation_accuracy
            print(f"Saved best validation-accuracy model: {best_path}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping: validation loss did not improve for {args.patience} epochs.")
                break
        scheduler.step()

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    print(f"\nBest validation accuracy: {best_accuracy:.2%}")
    evaluate(model, validation_loader, device)
    if not args.no_export_onnx:
        export_onnx(model, SCRIPT_DIR / "model.onnx", input_size, device)


if __name__ == "__main__":
    main()