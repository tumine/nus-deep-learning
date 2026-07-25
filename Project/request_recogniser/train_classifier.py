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
import importlib.util
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
    # ConvNeXt ImageNet pretrained weights expect these normalization statistics.
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
    # Class directories are the preferred output of the annotation workflow.
    samples = discover_folder_samples(data_dir)
    if not samples:
        # Fall back to a manifest only when directory labels are unavailable.
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
        # Split each class separately so all three labels occur in both datasets.
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
    # Preserve the pretrained feature extractor and replace only its final head.
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, len(CLASS_NAMES))
    return model


def has_working_triton() -> bool:
    """Return whether PyTorch can use Triton for the Inductor backend."""
    try:
        from torch.utils._triton import has_triton

        return has_triton()
    except (ImportError, AttributeError):
        return importlib.util.find_spec("triton") is not None


def is_triton_error(error: Exception) -> bool:
    """Identify lazy Inductor failures caused by a missing or unusable Triton."""
    return "triton" in str(error).lower() or "triton" in type(error).__name__.lower()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_channels_last: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    # AMP is enabled only on CUDA; CPU follows the same loop in full precision.
    amp_context = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
    for images, labels in loader:
        if use_channels_last:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with amp_context:
                logits = model(images)
                loss = criterion(logits, labels)
            if optimizer is not None:
                if scaler is not None:
                    # GradScaler protects fp16 gradients from underflow on CUDA.
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
        # Rows are ground truth labels; columns are model predictions.
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
        # A dynamic first dimension allows edge deployment with different batch sizes.
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
    )
    print(f"Exported ONNX model: {output_path}")


def export_torchscript(model: nn.Module, output_path: Path, input_size: int) -> None:
    """Export a self-contained TorchScript model for PyTorch C++/Python inference."""
    model = model.to("cpu").eval()
    example = torch.randn(1, 3, input_size, input_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        scripted_model = torch.jit.trace(model, example)
    scripted_model.save(str(output_path))
    print(f"Exported TorchScript model: {output_path}")


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
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader worker count (default: automatically choose up to 12)")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="Batches each DataLoader worker prepares ahead of time")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download or load ImageNet weights.")
    parser.add_argument("--no-export-onnx", action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile CUDA kernel fusion")
    parser.add_argument("--no-channels-last", action="store_true",
                        help="Disable the CUDA-optimized NHWC memory layout")
    args = parser.parse_args()

    if not 0 < args.patience or args.epochs < 1:
        raise ValueError("--epochs and --patience must both be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Benchmark is effective because every batch uses the same 224x224 shape.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)} (AMP enabled)")
    else:
        print("CUDA is unavailable; training will run on CPU without AMP.")

    input_size = 224
    batch_size = args.batch_size or (32 if args.variant == "base" else 64)
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(12, max(1, (torch.get_num_threads() or 1)))
    samples = load_samples(args.data.resolve())
    train_samples, validation_samples = split_samples(samples, validation_ratio=0.15, seed=args.seed)
    print(f"Split: train={len(train_samples)}, validation={len(validation_samples)}")
    train_transform, validation_transform = get_transforms(input_size)
    loader_settings = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_settings["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        LabeledImageDataset(train_samples, train_transform), shuffle=True,
        drop_last=len(train_samples) >= batch_size, **loader_settings,
    )
    validation_loader = DataLoader(
        LabeledImageDataset(validation_samples, validation_transform), shuffle=False,
        **loader_settings,
    )
    print(
        f"DataLoader: batch_size={batch_size}, workers={num_workers}, "
        f"persistent_workers={num_workers > 0}, prefetch_factor="
        f"{args.prefetch_factor if num_workers > 0 else 'n/a'}"
    )

    model = build_model(args.variant, pretrained=not args.no_pretrained).to(device)
    use_channels_last = device.type == "cuda" and not args.no_channels_last
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("Enabled channels_last (NHWC) memory format for ConvNeXt.")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    training_model = model
    compile_enabled = device.type == "cuda" and not args.no_compile and hasattr(torch, "compile")
    if compile_enabled and not has_working_triton():
        compile_enabled = False
        print("Triton is unavailable or unsupported; using eager mode. Pass --no-compile to silence this message.")
    if compile_enabled:
        try:
            training_model = torch.compile(model, mode="default")
            print("Enabled torch.compile (Inductor kernel fusion; the first epoch may compile slowly).")
        except Exception as error:
            compile_enabled = False
            print(f"torch.compile unavailable; continuing in eager mode: {error}")
    weights_dir = SCRIPT_DIR / "weights"
    best_path = weights_dir / "best_convnext.pth"
    best_accuracy = -1.0
    best_validation_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        try:
            train_loss, train_accuracy = run_epoch(
                training_model, train_loader, criterion, device, optimizer, scaler, use_channels_last,
            )
        except Exception as error:
            if not compile_enabled or not is_triton_error(error):
                raise
            compile_enabled = False
            training_model = model
            print(f"torch.compile could not initialize Triton; retrying epoch {epoch} in eager mode: {error}")
            train_loss, train_accuracy = run_epoch(
                training_model, train_loader, criterion, device, optimizer, scaler, use_channels_last,
            )
        validation_loss, validation_accuracy = run_epoch(
            training_model, validation_loader, criterion, device,
            use_channels_last=use_channels_last,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs} | Train Loss {train_loss:.4f} | Train Acc {train_accuracy:.2%} | "
            f"Val Loss {validation_loss:.4f} | Val Acc {validation_accuracy:.2%} | LR {learning_rate:.2e}"
        )
        if validation_accuracy > best_accuracy:
            # Save by accuracy for deployment, independent of early-stopping loss.
            weights_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "classes": CLASS_NAMES, "variant": args.variant, "input_size": input_size}, best_path)
            best_accuracy = validation_accuracy
            print(f"Saved best validation-accuracy model: {best_path}")
        if validation_loss < best_validation_loss:
            # Validation loss drives early stopping because it is more sensitive to overfitting.
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
    export_torchscript(model, SCRIPT_DIR / "model.pt", input_size)


if __name__ == "__main__":
    main()